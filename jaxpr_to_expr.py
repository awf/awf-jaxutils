import types
import sys
import re
import numpy as np

import jax
import jax._src
import jax._src.core
import jaxlib.xla_extension as xla_ext

def pythonize(name):
    new_name = re.sub("-", "_", name)
    assert re.match("[a-zA-Z0-9_]+", new_name)
    if new_name != name and name != "scatter-add":
        print(f"Note: pythonize converted {name} to {new_name}")
    return new_name

def pytype(x):
    if isinstance(x, jax.core.ShapedArray):
        return (
            f"ShapedArray({x.shape}, {x.dtype}, {x.weak_type}, {repr(x.named_shape)})"
        )

    return "Any"


# TODO:
#             bdy_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1013)(bdx_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d2af0>, num_consts=0)
#             bpl_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1019)(bpk_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d29d0>, num_consts=0)
#             brc_ = iota(, dtype=int32, shape=(31, 1), dimension=0)
#             ok_ = scatter-add(oj_,oc_,oi_, update_jaxpr={ [34m[22m[1mlambda [39m[22m[22m; a[35m:f32[39m b[35m:f32[39m. [34m[22m[1mlet[39m[22m[22m c[35m:f32[39m = add a b [34m[22m[1min [39m[22m[22m(c,) }, update_consts=(), dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1, 2), scatter_dims_to_operand_dims=(0, 1, 2)), indices_are_sorted=True, unique_indices=True, mode=GatherScatterMode.PROMISE_IN_BOUNDS)
#             def xla_call1003(a_: ShapedArray(float32[128,32,512]),b_: ShapedArray(int32, weak_type=True)):

from jax import lax
import jax.numpy as jnp
from icecream import ic
from jaxutils.expr import Var,Const,Lambda,Let,Call

def toExpr(x):
    if isinstance(
        x,
        (
            tuple,
            type(None),
            jax.numpy.dtype,
            jax.lax.GatherScatterMode,
            jax.lax.GatherDimensionNumbers,
            str, bool, int, np.float32, np.int32, float
         ),
    ):
        return Const(x)

    if isinstance(x, jax._src.core.Literal):
        return Const(x.val)

    if isinstance(x, jax.core.Jaxpr):
        return jaxpr_to_expr(x)

    if isinstance(x, jax.core.ClosedJaxpr):
        new_val = jaxpr_to_expr(x.jaxpr)

    if isinstance(x, jax.core.Var):
        return Var(f'v_{x.count:02d}{x.suffix}')

    # if isinstance(x, (types.FunctionType)):
    #     return Var(x.__module__ + "." + x.__name__)

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    # Explicitly add verified cases to 
    assert False, f"Check {type(x)} shouldn't be transformed [{repr(x)}]"

def toExprs(vars):
    return [toExpr(v) for v in vars]

def jaxpr_to_expr(jaxpr) -> Lambda:
    """
    Convert Jaxpr to jaxutils.Expr
    """

    # Will return Lambda(args, body)
    args = toExprs(jaxpr.invars)

    body = Call(Var('tuple'), toExprs(jaxpr.outvars))

    for cv in reversed(jaxpr.constvars):
        body = Let(toExpr(cv), Const(cv.aval), body)

    for eqn in reversed(jaxpr.eqns):

        ## Recursively dump sub-jaxprs, and add references to params
        new_params = {}
        for key, val in eqn.params.items():
            if isinstance(val, tuple):
                assert not any(
                    isinstance(x, (jax.core.Jaxpr, jax.core.ClosedJaxpr)) for x in val
                ), "Don't expect sub_jaxprs in tuples"

            # # Special cases: scatter-add
            # if eqn.primitive is lax.scatter_add_p and key in (
            #     "update_jaxpr",
            #     "update_consts",
            # ):
            #     print('jaxpr_to_expr:ignoring update_* in scatter_add')
            #     continue

            if isinstance(val, jax.core.Jaxpr):
                new_val = jaxpr_to_expr(val)
            elif isinstance(val, jax.core.ClosedJaxpr):
                new_val = jaxpr_to_expr(val.jaxpr)
            else:
                new_val = Const(val)

            new_params[key] = new_val

        primname = eqn.primitive.name + "_p.bind"
        if False and eqn.primitive is jax.interpreters.xla.xla_call_p:
            # TODO Handle xla_call specially - essentially erase it.  TODO: do we ever need to preserve the xla_call annotations?
            callee = new_params["call_jaxpr"]
            # translation = f"{callee}({intercommavars(*eqn.invars)}) # {new_params}"

        elif eqn.primitive is jax._src.pjit.pjit_p:
            # TODO Handle pjit_p specially - essentially erase it.  TODO: do we ever need to preserve the pjit annotations?
            callee = new_params["jaxpr"]
            # translation = f"{callee}({intercommavars(*eqn.invars)}) # {new_params}"

        else:
            if eqn.primitive is lax.scatter_add_p:
                primname = "scatter_add"

            # bind_args = intercomma(
            #     *(f"{varstr(v)}" for v in eqn.invars),
            #     *(f"{n}={varstr(v)}" for (n, v) in new_params.items()),
            # )

            # translation = f"{primname}({bind_args})"

        vars = map(toExpr, eqn.outvars)
        params = [Call(Const(key + '='), [val]) for key,val in new_params.items()]
        val = Call(Var(primname), [toExpr(v) for v in eqn.invars] + params)
        body = Let([*vars], val, body)

    return Lambda(args, body)


def show_jaxpr(f, args, name=None, file=sys.stdout, add_decls=False, **kwargs):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python
    """

    if name is None:
        name = f.__name__

    doc = f.__doc__

    jaxpr = jax.make_jaxpr(f, **kwargs)
    closed_jaxpr = jaxpr(*args)
    e = jaxpr_to_expr(closed_jaxpr.jaxpr)
    
    from jaxutils.expr import to_ast, freevars
    import ast
    print(ast.unparse(to_ast(e, 'e')))



def test_basic():
    def foo(p, x):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    f = vmapgradf

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("f(args)=")
    print(f(*args))

    show_jaxpr(f, args, name="f")

test_basic()

# def test_roundtrip():
#     import os

#     def foo(p, x):
#         x = jax.numpy.matmul(x, p * x.T)
#         return (x + x[3]).std()

#     gradf = jax.grad(foo, argnums=1)
#     vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

#     f = vmapgradf

#     prng = jax.random.PRNGKey(42)
#     args = (2.2, jax.random.normal(prng, (3, 2, 5)))

#     print("f(args)=")
#     print(f(*args))

#     # Save to file
#     fn = "tmp/show_jaxpr_jaxpr.py"
#     with open(fn, "w") as file:
#         show_jaxpr(f, args, name="f", file=file, add_decls=True)

#     os.system(f"black {fn}")

#     # Load from file
#     import importlib.util

#     module_name = "show_jaxpr_roundtrip"
#     spec = importlib.util.spec_from_file_location(module_name, fn)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)

#     # Check rountrip: does module.f give the same result?
#     assert jnp.allclose(module.f(*args), f(*args))

#     # Save again
#     fn2 = "tmp/show_jaxpr_roundtrip.py"
#     with open(fn2, "w") as file2:
#         show_jaxpr(module.f, args, file=file2, add_decls=True)

#     os.system(f"black {fn2}")

#     # Reload for 2nd roundtrip to test string equality
#     module_name = "show_jaxpr_roundtrip2"
#     spec = importlib.util.spec_from_file_location(module_name, fn2)
#     module2 = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module2
#     spec.loader.exec_module(module2)

#     assert jnp.allclose(module.f(*args), f(*args))

#     # Sand save 2nd roundtrip
#     fn3 = "tmp/show_jaxpr_roundtrip2.py"
#     with open(fn3, "w") as file3:
#         show_jaxpr(module2.f, args, file=file3, add_decls=True)

#     os.system(f"black {fn3}")

#     print(f"code --diff {fn2} {fn3} # Do view diffs in vs code")

# test_roundtrip()
