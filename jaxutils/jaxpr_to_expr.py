import types
import sys
import re
import numpy as np
import itertools

from beartype.typing import Callable, Optional, Any, Dict, List
from beartype import beartype

import pytest

import jax
import jaxlib

if jaxlib.version.__version__ <= "0.4":
    from jax.experimental import pjit
    from jax.interpreters import pxla
    import jax.core as jaxcore

    xla_call_p = jax.interpreters.xla.xla_call_p
else:
    import jax._src
    import jax._src.core as jaxcore
    from jax._src import pjit
    import jaxlib.xla_extension as xla_ext

    xla_call_p = None

from pprint import pprint

from jax import lax
import jax.numpy as jnp

import jaxutils
from jaxutils.expr import Expr, Var, Const, Lambda, Let, Call, mkTuple

from jaxutils.expr import (
    dictassign,
    optimize,
    to_ast,
    astunparse,
)

import ast

if sys.version_info >= (3, 9):
    unparse = ast.unparse
else:
    import astunparse

    unparse = astunparse.unparse


def tree_all(t):
    return all(jax.tree_util.tree_leaves(t))


def unspecified(x):
    return tree_all(jax.tree_map(jax._src.sharding_impls.is_unspecified, x))


def kwargs_to_dict_call(dict):
    if not dict:
        return []

    dict_pairs = [[Const(key), val] for key, val in dict.items()]
    return [Call(Var("**jaxutils_p2d"), list(itertools.chain(*dict_pairs)))]


def new_call(fn: str, *args, **kwargs):
    """
    Convenience function to make a Call node from a string, args, and kwargs
    """
    return Call(Var(fn), list(args) + kwargs_to_dict_call(kwargs))


translators: Dict[jax.core.Primitive, Callable[..., Optional[Expr]]] = {}


@dictassign(translators, lax.convert_element_type_p)
def _(val, new_dtype=None, weak_type=False):
    if weak_type.val == False:
        return new_call("convert_element_type", val, new_dtype)


@dictassign(translators, lax.broadcast_in_dim_p)
def _(val, shape=None, broadcast_dimensions=None):
    return new_call("broadcast_in_dim", val, shape, broadcast_dimensions)


@dictassign(translators, lax.squeeze_p)
def _(val, dimensions=None):
    return new_call("squeeze", val, dimensions)


@dictassign(translators, lax.slice_p)
def _(val, start_indices=None, limit_indices=None, strides=None):
    if all(v == 1 for v in strides.val):
        return new_call("slice", val, start_indices, limit_indices)
    else:
        return new_call("slice", val, start_indices, limit_indices, strides=strides)


@dictassign(translators, lax.dot_general_p)
def _(a, b, dimension_numbers=None, precision=None, preferred_element_type=None):
    if precision.val is None and preferred_element_type.val is None:
        # TODO Reverse-engineer to matmul where that's true, and if kwargs are default
        # mm_pattern = (((1,), (0,)), ((), ()))
        # if dimension_numbers == mm_pattern:
        #     ...

        return new_call("dot_general", a, b, dimension_numbers)

    # Or just call the general
    return new_call(
        "dot_general",
        a,
        b,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


translate_pjit_passthru = True


@dictassign(translators, pjit.pjit_p)
def _(
    *args,
    jaxpr=None,
    in_shardings=None,
    out_shardings=None,
    resource_env=None,
    donated_invars=None,
    name="",
    keep_unused=False,
    inline=False,
):
    # Elide the call if the shardings are unspecified
    if (
        translate_pjit_passthru
        and unspecified(in_shardings.val)
        and unspecified(out_shardings.val)
    ):
        return Call(jaxpr, list(args))


@dictassign(translators, jax.lax.scatter_add_p)
def _(*args, **kwargs):
    #    v_35 = scatter_add(
    #         v_34,
    #         v_21,
    #         v_32,
    #         ScatterDimensionNumbers(
    #             update_window_dims=(0, 1),
    #             inserted_window_dims=(1,),
    #             scatter_dims_to_operand_dims=(1,),
    #         ),
    #     **jaxutils_p2d(
    #         "indices_are_sorted",
    #         True,
    #         "unique_indices",
    #         True,
    #         "mode",
    #         GatherScatterMode.PROMISE_IN_BOUNDS,
    #     ),
    # )

    # Check that update_jaxpr is just an add, else how do we convert?
    # The call to optimize is because update_jaxpr is sometimes of the form
    #    lambda a,b: let tmp = add(a,b) in tmp
    # which simplifies to
    #    add
    # Note that if this assert fails because update_jaxpr is some large complicated thing
    # be aware of possible quadratic complexity in calling optimize
    update_jaxpr = kwargs.pop("update_jaxpr")
    assert optimize(update_jaxpr) == Var("add_p.bind")

    # Hope update_consts is Const(())
    assert not kwargs.pop("update_consts").val

    # Move dimension_numbers from kwargs to args
    dimension_numbers = kwargs.pop("dimension_numbers")

    # Override default name, 'scatter-add'
    return new_call("scatter_add", *args, dimension_numbers, **kwargs)


if xla_call_p:

    @dictassign(translators, xla_call_p)
    def _(*args, **kwargs):
        call_jaxpr = kwargs.pop("call_jaxpr")
        # TODO:
        #   1. check all kwargs are benign before erasing
        #   2. make this work if needed
        #      xla_call_p wants the jaxpr pulled out of kwargs
        #      return Call(Var("xla_call_p.bind"), [call_jaxpr, *args] + kwargs_to_dict_call(kwargs))
        return Call(call_jaxpr, list(args))


def jaxpr_to_expr(jaxpr) -> Lambda:
    """
    Convert Jaxpr to jaxutils.Expr
    """

    # Will return Lambda(args, body)
    body = mkTuple([jaxpr_to_expr_aux(v) for v in jaxpr.outvars])

    for cv in reversed(jaxpr.constvars):
        body = Let([jaxpr_to_expr_aux(cv)], Const(cv.aval), body)

    for eqn in reversed(jaxpr.eqns):
        new_params = {key: jaxpr_to_expr_aux(val) for key, val in eqn.params.items()}
        args = [jaxpr_to_expr_aux(v) for v in eqn.invars]

        val = None
        if eqn.primitive in translators:
            val = translators[eqn.primitive](*args, **new_params)

        if val is None:
            params = kwargs_to_dict_call(new_params)
            val = Call(Var(eqn.primitive.name + "_p.bind"), args + params)

        vars = map(jaxpr_to_expr_aux, eqn.outvars)
        body = Let([*vars], val, body)

    args = [jaxpr_to_expr_aux(v) for v in jaxpr.invars]
    return Lambda(args, body)


def jaxpr_to_expr_aux(x) -> Expr:
    if x is None or isinstance(
        x,
        (
            tuple,
            jax.lax.GatherDimensionNumbers,
            jax.lax.GatherScatterMode,
            str,
            bool,
            int,
            np.float32,
            np.int32,
            float,
            np.dtype,
        ),
    ):
        return Const(x)

    if isinstance(x, jaxcore.Literal):
        return Const(x.val)

    if isinstance(x, jaxcore.Jaxpr):
        return jaxpr_to_expr(x)

    if isinstance(x, jaxcore.ClosedJaxpr):
        return jaxpr_to_expr(x.jaxpr)

    if isinstance(x, jaxcore.Var):
        # return Var(f"v_{x.count+11}{x.suffix}")
        return Var(f"v_{x}")

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    # Explicitly add verified cases to
    assert False, f"Check {type(x)} shouldn't be transformed [{repr(x)}]"


import inspect


def jaxutils_p2d(*pairs):
    it = iter(pairs)
    return {a: b for (a, b) in zip(it, it)}


def test_jaxutils_p2d():
    assert jaxutils_p2d("a", 2, "b", 3) == {"a": 2, "b": 3}
    assert jaxutils_p2d() == {}


def show_jaxpr(
    f, args, name=None, file=sys.stdout, add_decls=None, optimize=True, static_argnums=(), **kwargs
):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python

    If `optimize`` is True, everal optimization passes are run, to simplify the resulting code.
    If global arg PJIT_PASSTHRU is True, then pjit(f)(args) with unspecified sharding
    is just rewritten to f(args), which can result in significantly simpler code.
    """

    if name is None:
        name = f.__name__

    doc = f.__doc__

    # Add decls if saving to file, but not to screen
    our_file = None
    if isinstance(file, str):
        our_file = open(file, 'w')
        print(f'show_jaxpr: Saving to [{file}]')
        file = our_file

    if add_decls == None:
        add_decls = file != sys.stdout

    jaxpr = jax.make_jaxpr(f, static_argnums=static_argnums, **kwargs)
    closed_jaxpr = jaxpr(*args)
    e = jaxpr_to_expr(closed_jaxpr.jaxpr)

    if optimize:
        e = jaxutils.expr.optimize(e)

    as_ast = to_ast(e, name)
    body_code = astunparse.unparse(as_ast)

    if add_decls:
        print(
            f"""#
# show_jaxpr {f}
from numpy import float32,int32
from jax.lax import *
from jax.lax import transpose_p
import jax.numpy as jnp
import jax
import jaxlib
if jaxlib.version.__version__ <= "0.4":
    from jax.experimental import pjit
    from jax.interpreters import pxla
    from jax.interpreters.xla import xla_call_p
    import jax.core as jaxcore
    DeviceArray = jnp.array
else:
    import jax._src
    import jax._src.core as jaxcore
    from jax._src import pjit
    import jaxlib.xla_extension as xla_ext
    Array = jnp.array

nan = jnp.nan
array = jnp.array

from jax._src.ad_util import add_any_p

{inspect.getsource(jaxutils_p2d)}
""",
            file=file,
        )

    print(body_code, file=file)

    import numpy

    numpy.set_printoptions(threshold=1e6)  # more than that, let's think about pickling
    if add_decls:
      non_static_args = [*args]
      if static_argnums:
        del non_static_args[static_argnums]
      flat_args = jax.tree_util.tree_leaves(non_static_args)
      test_args = ",".join(repr(arg) for arg in flat_args)

      print(
          f"""
if __name__ == '__main__':
    {name}({test_args})
""",
          file=file,
      )

    if add_decls:
        # Dump jaxpr too
        print("""
###############################################################################
#                                                                             #
#                       #   ##   #    # #####  #####                          #
#                       #  #  #   #  #  #    # #    #                         #
#                       # #    #   ##   #    # #    #                         #
#                       # ######   ##   #####  #####                          #
#                   #   # #    #  #  #  #      #   #                          #
#                   ####  #    # #    # #      #    #                         #
#                                                                             #
###############################################################################
""", file=file)

        print('jaxpr  = """', closed_jaxpr, '"""', file=file)

    if add_decls:
        # Dump XLA too
        print("""
###############################################################################
#                                                                             #
#                          #    # #       ####                                #
#                          #    # #      #    #                               #
#                          ###### #      #    #                               #
#                          #    # #      #    #                               #
#                          #    # #      #    #                               #
#                          #    # ######  ####                                #
#                                                                             #
###############################################################################
""", file=file)

        import jaxlib.xla_extension as xla_ext

        xc = jax.xla_computation(f, static_argnums=static_argnums)(*args)

        backend = jax.lib.xla_bridge.get_backend()
        e = backend.compile(xc)
        module = e.hlo_modules()[0]

        # You could use the presets
        option = xla_ext.HloPrintOptions.short_parsable()
        print('hlo_module  = """', module.to_string(option), '"""', file=file)

        # option = xla_ext.HloPrintOptions.canonical()
        # print(module.to_string(option))

        # option = xla_ext.HloPrintOptions.fingerprint()
        # print(module.to_string(option))

        # # Or set each option manually
        # option = xla_ext.HloPrintOptions()
        # option.print_metadata = False
        # option.include_layout_in_shapes = False
        # print(module.to_string(option))



    if our_file:
        our_file.close()


def test_basic():
    def foo(p, x):
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))

    print("f(args)=")
    print(gradf(*args))

    show_jaxpr(gradf, args, name="gradf")

    # pjit not supported on JaxDecompiler
    # from JaxDecompiler import decompiler

    # decompiled_f, python_code = decompiler.python_jaxpr_python(
    #     gradf, args, is_python_returned=True
    # )
    # print(python_code)


def test_vmap():
    def foo(p, x):
        x = x @ (p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))
    vargs = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("foo(args)=")
    print(foo(*args))

    show_jaxpr(foo, args, name="f")

    show_jaxpr(vmapgradf, vargs, name="vmapgradf")


import importlib.util


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n", [0, 1])
def test_roundtrip(n):
    import os

    def foo(p, x):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    prng = jax.random.PRNGKey(42)
    if n == 0:
        f = foo
        args = (2.2, jax.random.normal(prng, (2, 5)))
    else:
        f = vmapgradf
        args = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("f(args)=")
    print(f(*args))

    def save(f, args, name, fn):
        jaxutils.expr._global_name_id = 0

        with open(fn, "w") as file:
            show_jaxpr(f, args, name=name, file=file, add_decls=True)

        os.system(f"black -q -l 120 {fn}")

    # Save to file
    fn = "tmp/show_jaxpr_jaxpr.py"
    save(f, args, "f", fn)

    # Load from file
    module = load_module(fn, "show_jaxpr_roundtrip")

    # Check roundtrip: does module.f give the same result?
    assert jnp.allclose(module.f(*args), f(*args))

    # Save again
    fn2 = "tmp/show_jaxpr_roundtrip.py"
    save(module.f, args, "f", fn2)

    # Does it render the same?  Probably not, as nested calls have been optimized,
    # changing the results of uniquify_names, even with uniquify_names
    os.system(f"diff {fn} {fn2}")

    print(f"code --diff {fn} {fn2} # Do view diffs in vs code")


if __name__ == "__main__":
    test_roundtrip(0)
    test_roundtrip(1)
