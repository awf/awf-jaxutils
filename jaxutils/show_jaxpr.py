import types
import sys
import re
import numpy as np
from functools import lru_cache

import jax
import jaxlib.xla_extension as xla_ext
import jax._src as jaxsrc
from jax.extend import core as jaxcore
from jax._src import source_info_util as jaxsi


def isJaxpr(x):
    return isinstance(x, (jaxcore.Jaxpr, jaxcore.ClosedJaxpr))


def subJaxpr(x):
    if isinstance(x, jaxcore.Jaxpr):
        return x
    if isinstance(x, jaxcore.ClosedJaxpr):
        return x.jaxpr
    return None


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


foo_num = 100


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


@lru_cache
def getname(x):
    return new_name("v")


def varstr(x):
    if isinstance(
        x,
        (
            jax.core.DropVar,
            jaxcore.Literal,
            tuple,
            type(None),
            jax.numpy.dtype,
            jax.lax.GatherScatterMode,
        ),
    ):
        return str(x)

    if isinstance(x, jaxcore.Var):
        return getname(x)

    if isinstance(x, (jax.lax.GatherDimensionNumbers,)):
        return "GatherDimensionNumbers" + repr(x)

    if isinstance(x, (types.FunctionType, jaxsrc.linear_util.WrappedFun)):
        return x.__module__ + "." + x.__name__

    if x is np.float32:
        return "float32"
    if x is np.int32:
        return "int32"

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    if not isinstance(x, (str, bool, int, dict, jax.lax.GatherDimensionNumbers)):
        assert False, f"Check this shouldn't be transformed [{repr(x)}]"

    return repr(x)


def pytype(x):
    if isinstance(x, jax.core.ShapedArray):
        return f"ShapedArray({x.shape}, {x.dtype}, {x.weak_type})"

    return "Any"


# TODO:
# bdy_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1013)(bdx_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d2af0>, num_consts=0)
# bpl_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1019)(bpk_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d29d0>, num_consts=0)
# brc_ = iota(, dtype=int32, shape=(31, 1), dimension=0)
# ok_ = scatter-add(oj_,oc_,oi_, update_jaxpr={ [34m[22m[1mlambda [39m[22m[22m; a[35m:f32[39m b[35m:f32[39m. [34m[22m[1mlet[39m[22m[22m c[35m:f32[39m = add a b [34m[22m[1min [39m[22m[22m(c,) }, update_consts=(), dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1, 2), scatter_dims_to_operand_dims=(0, 1, 2)), indices_are_sorted=True, unique_indices=True, mode=GatherScatterMode.PROMISE_IN_BOUNDS)
# def xla_call1003(a_: ShapedArray(float32[128,32,512]),b_: ShapedArray(int32, weak_type=True)):

from jax import lax
import jax.numpy as jnp
from icecream import ic

prim_as_python_map = {
    lax.log_p: lambda x, accuracy=None: (
        f"jnp.log({x})" if accuracy is None else f"jnp.log({x}, {accuracy=})"
    ),
    lax.lt_p: lambda x, y: f"{x} < {y}",
    lax.add_p: lambda x, y: f"{x} + {y}",
    lax.mul_p: lambda x, y: f"{x} * {y}",
    lax.sub_p: lambda x, y: f"{x} - {y}",
    lax.div_p: lambda x, y: f"{x} / {y}",
}


def print_jaxpr_as_python(f, jaxpr, *, indent="", doc="", file=sys.stdout):
    args = intercomma(*(varstr(v) + f": '{pytype(v.aval)}'" for v in jaxpr.invars))
    print(f"\n{indent}def {f}({args}):", file=file)
    indent += tab
    if doc:
        print(f'{indent}"""{doc}"""', file=file)

    for cv in jaxpr.constvars:
        assert cv not in ["if", "in", "is"]  # if it is, use varstr(cv) on next line
        print(f"{indent}{varstr(cv)} = {cv.aval} # constant", file=file)

    for eqn in jaxpr.eqns:
        ## Recursively dump sub-jaxprs, and add references to params
        new_params = {}
        for key, val in eqn.params.items():
            if isinstance(val, tuple):
                # We don't expect sub_jaxprs in tuples
                assert not any(isJaxpr(x) for x in val)

            # Special cases: the jaxprs in scatter-add are not handled at the moment
            # print them and allow the user to see where they occur.
            # scatter-adds which don't use the jaxprs are handled as normal
            if eqn.primitive is lax.scatter_add_p and key in (
                "update_jaxpr",
                "update_consts",
            ):
                continue

            if sub_jaxpr := subJaxpr(val):
                # Sub-jaxpr, make a name, and recurse
                if "name" in eqn.params:
                    nm = eqn.primitive.name + "_" + eqn.params["name"]
                else:
                    nm = eqn.primitive.name

                nm = new_name(pythonize(nm))
                doc = doc_from_source_line(eqn.source_info)
                # recurse, generating
                #    def myfunc0012(...):
                #       ...
                print_jaxpr_as_python(nm, sub_jaxpr, indent=indent, doc=doc, file=file)
                # and replace jaxpr val with literal name, so call will become
                # y = custom_jvp_call(..., call_jaxpr=myfunc0012, ...)
                val = jaxcore.Literal(nm, None)

            # Add val to new_params
            new_params[key] = val

        if eqn.primitive is jaxsrc.pjit.pjit_p:

            # pjits are all of the form pjit(func_var, args).
            # Emit as func_var(args)
            # TODO: do we ever need to preserve the pjit annotations?
            callee = new_params["jaxpr"]
            translation = f"{callee}({intercommavars(*eqn.invars)}) # {new_params}"

        elif eqn.primitive in prim_as_python_map:
            mkpy = prim_as_python_map[eqn.primitive]
            translation = mkpy(
                *map(varstr, eqn.invars),
                **{k: varstr(v) for (k, v) in new_params.items()},
            )

        else:

            bind_args = intercomma(
                *(f"{varstr(v)}" for v in eqn.invars),
                *(f"{n}={varstr(v)}" for (n, v) in new_params.items()),
            )

            translation = f"{get_primitive_name(eqn)}({bind_args})"

        if len(eqn.outvars):
            print(f"{indent}{intercommavars(*eqn.outvars)} = {translation}", file=file)
        else:
            print(f"{indent}{translation}", file=file)

    print(f"{indent}return ({intercommavars(*jaxpr.outvars)})", file=file)


def get_primitive_name(eqn):
    if eqn.primitive is lax.scatter_add_p:
        return "scatter_add"

    if False and (eqn.primitive in (
        lax.select_n_p,
        lax.broadcast_in_dim_p,
        lax.gather_p,
        lax.reduce_sum_p,
    )):
        return eqn.primitive.name

    return eqn.primitive.name + "_p.bind"


def inline_jaxpr(eqn, new_eqn_invars, new_eqn_outvars, var_mapping):
    if eqn.primitive not in (
        jaxsrc.pjit.pjit_p,
        # jaxsrc.custom_derivatives.custom_jvp_call_p,
    ):
        return None

    # Inlining call
    #   new_os = foo(new_is)
    # where foo is
    #   inner_os[1] = bar(inner_is[0], inner_is[1])
    #   inner_os[0] = bar(inner_os[1], inner_is[1])
    # So we rewrite the body to
    #   new_os[1] = bar(new_is[0], new_is[1])
    #   new_os[0] = bar(new_os[1], new_is[1])

    if eqn.primitive is jaxsrc.pjit.pjit_p:
        callee = eqn.params["jaxpr"].jaxpr
    elif eqn.primitive is jaxsrc.custom_derivatives.custom_jvp_call_p:
        callee = eqn.params["call_jaxpr"].jaxpr

    if len(callee.eqns) > 0:
        return None

    print(f"Inlining jaxpr {callee} {eqn.params['name']}")

    # rename variables in the callee
    invars_mapping = {
        inner: value for inner, value in zip(callee.invars, new_eqn_invars)
    }
    outvars_mapping = {
        inner: here
        for inner, here in zip(callee.outvars, new_eqn_outvars)
        if inner not in invars_mapping
    }

    for aliased_outvar, new_eqn_outvar in zip(callee.outvars, new_eqn_outvars):
        if aliased_outvar in callee.invars:
            # map new_outvar[i] to new_invar[i] for the rest of the translation
            invar = invars_mapping[aliased_outvar]
            for k, v in var_mapping.items():
                if v == new_eqn_outvar:
                    var_mapping[k] = invar
            var_mapping[new_eqn_outvar] = invar

    new_callee = simplify_jaxpr(callee, outvars_mapping | invars_mapping)

    return new_callee.eqns


def simplify_jaxpr(jaxpr, var_mapping=None, deep=True):
    if var_mapping is None:
        var_mapping = {}

    recurse = lambda x: simplify_jaxpr(x, var_mapping)

    if isinstance(jaxpr, jaxcore.ClosedJaxpr):
        return jaxcore.ClosedJaxpr(recurse(jaxpr.jaxpr), jaxpr.consts)

    def new_var(v):
        if not deep:
            return v

        if isinstance(v, jaxcore.Var):
            # Not cloning avals - correct?
            if v in var_mapping:
                return var_mapping[v]
            else:
                vnew = jaxcore.Var(v.suffix, v.aval)
                var_mapping[v] = vnew
                return vnew

        assert isinstance(v, jaxcore.Literal)
        return v

    if isinstance(jaxpr, jaxcore.Jaxpr):
        new_constvars = jaxpr.constvars

        new_invars = [new_var(v) for v in jaxpr.invars]
        new_outvars = [new_var(v) for v in jaxpr.outvars]

        def new_eqn(eqn):
            new_eqn_invars = [new_var(v) for v in eqn.invars]
            new_eqn_outvars = [new_var(v) for v in eqn.outvars]

            # inline the "direct call" primitives pjit and custom_jvp_call
            inlined_eqns = inline_jaxpr(
                eqn, new_eqn_invars, new_eqn_outvars, var_mapping
            )
            if inlined_eqns is not None:
                return inlined_eqns

            def new_param(p):
                if isJaxpr(p):
                    return recurse(p)
                else:
                    return p

            new_params = {k: new_param(param) for k, param in eqn.params.items()}

            new_source_info = jaxsi.SourceInfo(
                eqn.source_info.traceback, eqn.source_info.name_stack.extend("simplify")
            )

            new_eqn = jaxcore.JaxprEqn(
                new_eqn_invars,
                new_eqn_outvars,
                eqn.primitive,
                new_params,
                eqn.effects,
                new_source_info,
                eqn.ctx,
            )

            return [new_eqn]

        new_eqns = list(eqn for eqn0 in jaxpr.eqns for eqn in new_eqn(eqn0))

        new_effects = jaxpr.effects
        new_debug_info = jaxpr.debug_info
        return jaxcore.Jaxpr(
            new_constvars,
            new_invars,
            new_outvars,
            new_eqns,
            new_effects,
            new_debug_info,
        )

    assert False, f"Don't know how to simplify {jaxpr} of type {type(jaxpr)}"


def show_jaxpr(
    f, args, name=None, file=sys.stdout, add_decls=False, add_main=False, **kwargs
):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python
    """

    print("#fmt: off", file=file)

    if add_decls:
        print(
            f"""
#fmt: off
# show_jaxpr {f}
from jax.lax import *
from jax.lax import transpose_p
import jax.numpy as jnp
from numpy import float32,int32,nan
import numpy as np

add_any_p = add_p

""",
            file=file,
        )

    if name is None:
        name = f.__name__

    doc = f.__doc__

    jaxpr = jax.make_jaxpr(f, **kwargs)
    closed_jaxpr = jaxpr(*args)
    print("simplify ...")
    closed_jaxpr = simplify_jaxpr(closed_jaxpr)
    print("simplify ... done")

    # run it...
    try:
        nonstatic_args = args
        if "static_argnums" in kwargs:
            static_argnums = kwargs["static_argnums"]
            if not isinstance(static_argnums, (list, tuple)):
                static_argnums = [static_argnums]
            nonstatic_arg_inds = set(range(len(args))) - set(static_argnums)
            nonstatic_args = [args[i] for i in nonstatic_arg_inds]

        flatargs = jax.tree.flatten(nonstatic_args)[0]  # TODO, hack static argum =1
        print("eval1 ...")
        ans1 = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *flatargs)
        print("eval2 ...")
        ans2 = jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *flatargs)
        ok = jax.tree.map(jnp.allclose, ans1, ans2)
        assert all(ok)
    except Exception as e:
        print("Error evaluating jaxpr", e)

    print_jaxpr_as_python(name, closed_jaxpr.jaxpr, doc=doc, file=file)

    if add_main:
        print(
            f"""
if __name__ == '__main__':
    Array = jnp.array
    {name}{args}
""",
            file=file,
        )


def show_xla(f, args, file=sys.stdout, optimized=False, **kwargs):
    """
    Show XLA for f, using template args
    """
    xla = jax.xla_computation(f, **kwargs)(*args)

    if optimized:
        e = jax.lib.xla_bridge.get_backend().compile(xla.get_hlo_module())
        module = e.hlo_modules()[0]
    else:
        module = xla.get_hlo_module()
    option = xla_ext.HloPrintOptions.short_parsable()
    print(module.to_string(option), file=file)


def show_jaxpr_and_xla(f, args, file=sys.stdout, optimized=False, **kwargs):
    show_jaxpr(f, args, file=file, **kwargs)
    show_xla(f, args, file=file, optimized=optimized, **kwargs)


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
    # show_xla(f, args)
    # show_xla(f, args, optimized=True)


def test_roundtrip():
    import os

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

    # Save to file
    fn = "tmp/show_jaxpr_jaxpr.py"
    with open(fn, "w") as file:
        show_jaxpr(f, args, name="f", file=file, add_decls=True)

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
        show_jaxpr(module.f, args, file=file2, add_decls=True)

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
        show_jaxpr(module2.f, args, file=file3, add_decls=True)

    os.system(f"black {fn3}")

    print(f"code --diff {fn2} {fn3} # Do view diffs in vs code")
