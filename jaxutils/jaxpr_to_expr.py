import types
import sys
import re
import numpy as np
import functools

from beartype.typing import Callable, Optional, Any, Dict, List
from beartype import beartype

import pytest

import jax
from jax import lax
import jax.numpy as jnp
import jax._src as jaxsrc
from jax.extend import core as jaxcore
from jax.extend import linear_util as jaxlu
from jax.extend.core import primitives as jaxprim

from pprint import pprint

import jaxutils
import jaxutils.expr as jex
from jaxutils.expr import Expr, Var, Const, Lambda, Eqn, Let, Call, kwargs_to_dict_call

from jaxutils.expr import dictassign

import ast

if sys.version_info >= (3, 9):
    unparse = ast.unparse
else:
    import astunparse

    unparse = astunparse.unparse


def new_call(fn: str, *args, **kwargs):
    """
    Convenience function to make a Call node from a string, args, and kwargs
    """
    return Call(Var(fn), list(args) + kwargs_to_dict_call(kwargs))


_prim_to_expr: Dict[jaxcore.Primitive, Callable[..., Optional[Expr]]] = {}


def declare_prim_to_expr(prim):
    """
    Decorator to register a function as a primitive-to-expression translator.

    Behaviour: the prim_translator can return a new Expr, or None
    If None, we will simply call prim.bind(*args, **kwargs)

    It is called as
      prim_translator(*args, **kwargs)
    where the args are Exprs.  An example might be (if matmul were a prim)

      @declare_prim_to_expr(matmul_p)
      def _(a, b):
        return new_call("matmul", a, b)

    This is trivial, changing the rendering from
        matmul_p.bind(a, b)
    to
        matmul(a, b)
    but the design allows for more complex transformations.
    """
    return dictassign(_prim_to_expr, prim)


_prim_to_expr_simple = {
    lax.eq_p: "operator.__eq__",
    lax.ne_p: "operator.__ne__",
    lax.lt_p: "operator.__lt__",
    lax.le_p: "operator.__le__",
    lax.gt_p: "operator.__gt__",
    lax.ge_p: "operator.__ge__",
    lax.add_p: "operator.__add__",
    jaxprim.add_jaxvals_p: "operator.__add__",
    lax.sub_p: "operator.__sub__",
    lax.mul_p: "operator.__mul__",
    lax.div_p: "operator.__truediv__",
    lax.pow_p: "operator.__pow__",
    lax.neg_p: "operator.__neg__",
    lax.squeeze_p: "lax.squeeze",
    lax.max_p: "lax.max",
    lax.square_p: "lax.square",
    lax.exp_p: "lax.exp",
    lax.sqrt_p: "lax.sqrt",
    lax.select_n_p: "lax.select_n",
    lax.stop_gradient_p: "lax.stop_gradient",
    # lax.rem_p: "operator.__mod__",
    # lax.lshift_p: "operator.__lshift__",
    # lax.rshift_p: "operator.__rshift__",
    # lax.bitor_p: "operator.__bitor__",
    # lax.bitxor_p: "operator.__bitxor__",
    # lax.bitand_p: "operator.__bitand__",
    # lax.matmul_p: "operator.__matmul__",
}

for prim, fn in _prim_to_expr_simple.items():
    _prim_to_expr[prim] = lambda *args, _prim_to_expr_fn=fn, **kwargs: new_call(
        _prim_to_expr_fn, *args, **kwargs
    )


@declare_prim_to_expr(lax.integer_pow_p)
def _(x, y=0):
    return new_call("operator.__pow__", x, y)


@declare_prim_to_expr(lax.broadcast_in_dim_p)
def _(x, shape=None, broadcast_dimensions=None, sharding=None):
    return new_call(
        "lax.broadcast_in_dim",
        x,
        shape,
        broadcast_dimensions,
        out_sharding=sharding,
    )


@declare_prim_to_expr(lax.transpose_p)
def _(x, permutation=None):
    if permutation.isConst and permutation.val == (1, 0):
        return new_call("getattr", x, Const("T"))
    else:
        return new_call("jnp.transpose", x, axes=permutation)


@declare_prim_to_expr(lax.log_p)
def _(x, accuracy=None):
    if jex.isNone(accuracy):
        return new_call("jnp.log", x)
    else:
        return new_call("jnp.log", x, accuracy=accuracy)


@declare_prim_to_expr(lax.convert_element_type_p)
def _(val, new_dtype=None, weak_type=False, sharding=None):
    if weak_type.val == False:
        return new_call("lax.convert_element_type", val, new_dtype)


@declare_prim_to_expr(lax.slice_p)
def _(val, start_indices=None, limit_indices=None, strides=None):
    if not strides or not strides.val or all(v == 1 for v in strides.val):
        return new_call("lax.slice", val, start_indices, limit_indices)
    else:
        return new_call("lax.slice", val, start_indices, limit_indices, strides=strides)


@declare_prim_to_expr(lax.dot_general_p)
def _(
    a,
    b,
    dimension_numbers=None,
    precision=None,
    preferred_element_type=None,
    out_sharding=None,
):
    if (
        jex.isNone(precision)
        and (
            jex.isNone(preferred_element_type)
            or preferred_element_type.val == np.float32
        )
        and jex.isNone(out_sharding)
    ):
        # TODO Reverse-engineer to matmul where that's true, and if kwargs are default
        # mm_pattern = (((1,), (0,)), ((), ()))
        # if dimension_numbers == mm_pattern:
        #     ...
        (
            (lhs_contracting_dims, rhs_contracting_dims),
            (lhs_batch_dims, rhs_batch_dims),
        ) = dimension_numbers.val
        if lhs_batch_dims == () and rhs_batch_dims == ():
            # This is a matmul
            if lhs_contracting_dims == (1,) and rhs_contracting_dims == (0,):
                # This is a matmul, but
                # TODO: only if we know shape is 2D and dtype matches preferred...
                return new_call("operator.__matmul__", a, b)

            # # This is a transpose
            # if lhs_contracting_dims == (0,) and rhs_contracting_dims == (1,):
            #     # This is a transpose
            #     return new_call("operator.__matmul__", a, b)

    # Or just call the general
    return new_call(
        "lax.dot_general",
        a,
        b,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )


translate_pjit_passthru = True
global pjit_name_count
pjit_name_count = 0


def unspecified(xs):
    is_unspec = lambda x: isinstance(x, jax._src.named_sharding.UnspecifiedValue)
    return jax.tree.all(jax.tree.map(is_unspec, xs))


@declare_prim_to_expr(jaxsrc.pjit.pjit_p)
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
    in_layouts=None,
    out_layouts=None,
    compiler_options_kvs=None,
    ctx_mesh=None,
):
    if name:
        global pjit_name_count
        f = Var(f"pjit_{name.val}_{pjit_name_count}")
        pjit_name_count += 1
        return Let([Eqn([f], jaxpr)], Call(f, list(args)))
    else:
        # Elide the call if the shardings are unspecified
        if (
            translate_pjit_passthru
            and unspecified(in_shardings.val)
            and unspecified(out_shardings.val)
        ):
            return Call(jaxpr, list(args))


@declare_prim_to_expr(jaxprim.custom_jvp_call_p)
def _(*args, call_jaxpr=None, jvp_jaxpr_fun=None, num_consts=None, symbolic_zeros=None):
    # From https://github.com/jax-ml/jax/blob/89807ba73ebd96ed8262f11be9ff02b009a30a19/jax/experimental/jax2tf/jax2tf.py#L1444
    # Just erase the custom_jvp_call; although it does lose the custom_jvp rules
    # A common example is jnn.relu, which has a custom_jvp rule which would need to be
    # reinstated in any AD for this Expr library.
    print(f"custom_jvp_call_p: losing connection to {jvp_jaxpr_fun.val.f_transformed}")
    return Call(call_jaxpr, list(args))


@declare_prim_to_expr(jax.lax.scatter_add_p)
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
    #         indices_are_sorted=True,
    #         unique_indices=True,
    #         mode=GatherScatterMode.PROMISE_IN_BOUNDS,
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
    if not update_jaxpr.isVar:
        assert jex.optimize(update_jaxpr) == Var("operator.__add__")

    # Hope update_consts is Const(())
    assert not kwargs.pop("update_consts").val

    # Move dimension_numbers from kwargs to args
    dimension_numbers = kwargs.pop("dimension_numbers")

    # Override default name, 'scatter-add'
    return new_call("scatter_add", *args, dimension_numbers, **kwargs)


@functools.cache  # TODO: could be LRU, and we would then miss some CSE
def jaxpr_to_expr(jaxpr) -> Lambda:
    """
    Convert Jaxpr to Lambda
    """

    # Will return Lambda(args, body)
    args = [jaxpr_to_expr_aux(v) for v in jaxpr.invars]

    # Eqns for constvars
    new_eqns = [Eqn([jaxpr_to_expr_aux(cv)], Const(cv.aval)) for cv in jaxpr.constvars]

    # And eqns for eqns
    for eqn in jaxpr.eqns:
        new_params = {key: jaxpr_to_expr_aux(val) for key, val in eqn.params.items()}
        eqn_args = [jaxpr_to_expr_aux(v) for v in eqn.invars]

        val = None
        if eqn.primitive in _prim_to_expr:
            val = _prim_to_expr[eqn.primitive](*eqn_args, **new_params)

        if val is None:
            params = jex.kwargs_to_dict_call(new_params)
            val = Call(Var(eqn.primitive.name + "_p.bind"), eqn_args + params)

        vars = map(jaxpr_to_expr_aux, eqn.outvars)
        new_eqns += [Eqn([*vars], val)]

    body = Let(new_eqns, jex.mkTuple([jaxpr_to_expr_aux(v) for v in jaxpr.outvars]))

    return Lambda(args, body, "j2e_" + jex.get_new_name())


varname_n = 0


@functools.cache
def varname(x):
    global varname_n
    varname_n += 1
    return f"v{varname_n:02x}"


def jaxpr_to_expr_aux(x) -> Expr:
    if isinstance(
        x,
        (
            types.NoneType,
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
        return Var(varname(x))

    if isinstance(x, (types.FunctionType, jaxlu.WrappedFun)):
        return Const(x)

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    # Explicitly add verified cases to
    assert False, f"Check {type(x)} shouldn't be transformed [{repr(x)}]"


def show_jaxpr(
    f,
    args,
    name=None,
    file=sys.stdout,
    optimize=True,
    add_decls=None,
    add_jaxpr=False,
    add_hlo=False,
    reset_ids=False,
    static_argnums=(),
    **kwargs,
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
        our_file = open(file, "w")
        print(f"show_jaxpr: Saving to [{file}]")
        file = our_file

    if add_decls == None:
        add_decls = file != sys.stdout

    jaxpr = jax.make_jaxpr(f, static_argnums=static_argnums, **kwargs)
    closed_jaxpr = jaxpr(*args)

    if reset_ids:
        jex.reset_new_name_ids()
        global varname_n
        varname_n = 0
        global pjit_name_count
        pjit_name_count = 0

    e = jaxpr_to_expr(closed_jaxpr.jaxpr)

    if optimize:
        e = jex.optimize(e)

    as_ast = jex.to_ast(e, name)
    body_code = unparse(as_ast)

    fvs = list(v.name for v in jex.freevars(e))
    fvs_prims = list(v[:-5] for v in fvs if v.endswith(".bind"))

    if add_decls:
        print(
            f"""#
# show_jaxpr {f}
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import *

from numpy import (inf, nan)

from jax.extend.core.primitives import (
   {",\n   ".join(fvs_prims)},
   add_jaxvals_p
)

from jaxutils.expr_lib import *
""",
            file=file,
        )

    print(body_code, file=file)

    import numpy

    numpy.set_printoptions(threshold=1e6)  # more than that, let's think about pickling
    if add_decls:
        non_static_args = [*args]
        if static_argnums != ():
            del non_static_args[static_argnums]
        flat_args = jax.tree_util.tree_leaves(non_static_args)

        def test_repr(x):
            if hasattr(x, "dtype"):
                return f"rand({x.shape}, np.{x.dtype})"
            else:
                return repr(x)

        test_args = ",".join(test_repr(arg) for arg in flat_args)

        print(
            f"""
def rand(shape, dtype):
    if np.issubdtype(dtype, np.floating):
        return jnp.array(np.random.rand(*shape).astype(dtype))
    else:
        return jnp.array(np.random.randint(0, 256, shape, dtype))

if __name__ == '__main__':
    {name}({test_args})
""",
            file=file,
        )

    if add_jaxpr:
        # Dump jaxpr too
        print(
            """
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
""",
            file=file,
        )

        print('jaxpr  = """', closed_jaxpr, '"""', file=file)

    if add_hlo:
        # Dump XLA too
        print(
            """
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

# is broken - see https://github.com/jax-ml/jax/discussions/7068
""",
            file=file,
        )

        # xc = closed_jaxpr.lower(*args)
        # e = xc.compile()
        # module = e.hlo_modules()[0]

        # You could use the presets
        # option = xla_ext.HloPrintOptions.short_parsable()
        # print('hlo_module  = """', e.as_text(), '"""', file=file)

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
