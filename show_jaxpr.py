import types
import numpy as np
import jax
import sys
import re


def cat(xs):
    return "".join(xs)


def intercomma(*xs):
    return ", ".join(xs)


def intercommastr(*xs):
    return ", ".join((str(x) for x in xs))


def intercommavars(*xs):
    return ", ".join((varstr(x) for x in xs))


def justone(iter):
    l = list(iter)
    assert len(l) == 1
    return l[0]


tab = "    "


def doc_from_source_line(source_info):
    def tostr(f):
        return f"{f.file_name}:{f.line_num}:{f.function_name}"

    fnames = list(
        tostr(f)
        for f in source_info.traceback.frames
        if ("site-packages" not in f.file_name and "show_jaxpr" != f.function_name)
    )
    return fnames[0]


foo_num = 1000


def pythonize(name):
    new_name = re.sub("-", "_", name)
    assert re.match("[a-zA-Z0-9_]+", new_name)
    if new_name != name and name != "scatter-add":
        print(f"Note: pythonize converted {name} to {new_name}")
    return new_name


def new_name(base):
    global foo_num
    n = f"{base}{foo_num}"
    foo_num += 1
    return n


def varstr(x):
    if isinstance(
        x,
        (
            jax.core.DropVar,
            jax.core.Literal,
            tuple,
            type(None),
            jax.numpy.dtype,
            jax.lax.GatherScatterMode,
        ),
    ):
        return str(x)

    if isinstance(x, jax.core.Var):
        return str(x) + "_"

    if isinstance(x, (jax.lax.GatherDimensionNumbers,)):
        return "GatherDimensionNumbers" + repr(x)

    if isinstance(x, (types.FunctionType)):
        return x.__module__ + "." + x.__name__

    if x is np.float32:
        return "float32"
    if x is np.int32:
        return "int32"

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    assert isinstance(
        x, (str, bool, int, jax.lax.GatherDimensionNumbers)
    ), f"Check this shouldn't be transformed [{repr(x)}]"

    return repr(x)


def pytype(x):
    if isinstance(x, jax.ShapedArray):
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


def examine_jaxpr(f, jaxpr, *, indent="", doc="", file=sys.stdout):
    args = intercomma(*(varstr(v) + f": '{pytype(v.aval)}'" for v in jaxpr.invars))
    print(f"\n{indent}def {f}({args}):", file=file)
    indent += tab
    if doc:
        print(f'{indent}"""{doc}"""', file=file)

    for cv in jaxpr.constvars:
        assert cv not in ["if", "in", "is"]  # if it is, use varstr(cv) on next line
        print(f"{indent}{cv} = ?", file=file)

    for eqn in jaxpr.eqns:

        ## Recursively dump sub-jaxprs, and add references to params
        new_params = {}
        for key, val in eqn.params.items():
            if isinstance(val, tuple):
                assert not any(
                    isinstance(x, (jax.core.Jaxpr, jax.core.ClosedJaxpr)) for x in val
                ), "Don't expect sub_jaxprs in tuples"

            # Special cases: scatter-add
            if eqn.primitive is lax.scatter_add_p and key in (
                "update_jaxpr",
                "update_consts",
            ):
                continue

            if isinstance(val, jax.core.Jaxpr):
                sub_jaxpr = val
            elif isinstance(val, jax.core.ClosedJaxpr):
                sub_jaxpr = val.jaxpr
            else:
                # No sub_jaxpr - just update new_params and continue
                new_params[key] = val
                continue

            # Sub-jaxpr, make a name, and recurse
            n = new_name(pythonize(eqn.primitive.name))

            doc = doc_from_source_line(eqn.source_info)
            examine_jaxpr(n, sub_jaxpr, indent=indent, doc=doc, file=file)

            new_params[key] = jax.core.Literal(n, None)

        primname = eqn.primitive.name + "_p.bind"
        if eqn.primitive is jax.interpreters.xla.xla_call_p:
            # Handle xla_call specially - essentially erase it.  TODO: do we ever need to preserve the xla_call annotations?
            callee = new_params["call_jaxpr"]
            translation = f"{callee}({intercommavars(*eqn.invars)}) # {new_params}"

        else:
            if eqn.primitive is lax.scatter_add_p:
                primname = "scatter_add"

            bind_args = intercomma(
                *(f"{varstr(v)}" for v in eqn.invars),
                *(f"{n}={varstr(v)}" for (n, v) in new_params.items()),
            )

            translation = f"{primname}({bind_args})"

        print(f"{indent}{intercommavars(*eqn.outvars)} = {translation}", file=file)

    print(f"{indent}return ({intercommavars(*jaxpr.outvars)})", file=file)


def show_jaxpr(f, args, name=None, file=sys.stdout, **kwargs):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python
    """

    print(
        f"""
# show_jaxpr {f}
from jax.lax import *
import jax.numpy as jnp
from numpy import float32,int32

add_any_p = add_p

DeviceArray = jnp.array
""",
        file=file,
    )

    if name is None:
        name = f.__name__

    closed_jaxpr = jax.make_jaxpr(f, **kwargs)(*args)
    examine_jaxpr(name, closed_jaxpr.jaxpr, doc=f.__doc__, file=file)

    print(
        f"""
if __name__ == '__main__':
    {name}{args}
""",
        file=file,
    )


def show_xla(f, args, file=sys.stdout, optimized=False, **kwargs):
    """
    Show XLA for f, using template args
    """
    xla = jax.xla_computation(f, **kwargs)(*args)
    print("XLA=", xla.as_hlo_text(), file=file)

    if optimized:
        e = jax.lib.xla_bridge.get_backend().compile(xla)
        module = e.hlo_modules()[0]
    else:
        module = xla.get_hlo_module()
    option = xla_ext.HloPrintOptions.short_parsable()
    print(module.to_string(option), file=file)


def show_jaxpr_and_xla(f, args, file=sys.stdout, **kwargs):
    show_jaxpr(f, args, file=file, **kwargs)
    show_xla(f, args, file=file, **kwargs)


def test_basic():
    def foo(p, x, q):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2, None))

    f = vmapgradf

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (3, 2, 5)), "q")

    print("f(args)=")
    print(f(*args))

    show_jaxpr(f, args, name="f")


def test_roundtrip():
    import os

    def foo(p, x, q):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2, None))

    f = vmapgradf

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (3, 2, 5)), "q")

    print("f(args)=")
    print(f(*args))

    # Save to file
    fn = "tmp/show_jaxpr_jaxpr.py"
    with open(fn, "w") as file:
        show_jaxpr(f, args, name="f", file=file)

    os.system(f"black {fn}")

    # Load from file
    import importlib.util

    module_name = "show_jaxpr_roundtrip"
    spec = importlib.util.spec_from_file_location(module_name, fn)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Check rountrip: does module.f give the same result?
    assert jnp.allclose(module.f(*args), f(*args))

    # Save again
    fn2 = "tmp/show_jaxpr_roundtrip.py"
    with open(fn2, "w") as file2:
        show_jaxpr(module.f, args, file=file2)

    os.system(f"black {fn2}")

    # Reload for 2nd roundtrip to test string equality
    module_name = "show_jaxpr_roundtrip2"
    spec = importlib.util.spec_from_file_location(module_name, fn2)
    module2 = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module2
    spec.loader.exec_module(module2)

    assert jnp.allclose(module.f(*args), f(*args))

    # Sand save 2nd roundtrip
    fn3 = "tmp/show_jaxpr_roundtrip2.py"
    with open(fn3, "w") as file3:
        show_jaxpr(module2.f, args, file=file3)

    os.system(f"black {fn3}")

    print(f"code --diff {fn2} {fn3} # Do view diffs in vs code")
