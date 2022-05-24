import types
import numpy as np
import jax
import sys
import re

def cat(xs):
    return "".join(xs)


def intercomma(xs):
    return ", ".join(xs)


def intercommastr(xs):
    return ", ".join((str(x) for x in xs))

def intercommavars(xs):
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
    name = re.sub('-','_',name)
    assert re.match('[a-zA-Z0-9_]+', name)
    return name

def new_name(base):
    global foo_num
    n = f"{base}{foo_num}"
    foo_num += 1
    return n

def varstr(x):
    if isinstance(x, (jax.core.DropVar, jax.core.Literal, tuple, type(None), 
                        jax.numpy.dtype,jax.lax.GatherScatterMode)):
        return str(x)

    if isinstance(x, jax.core.Var):
        return str(x) + '_'

    if isinstance(x, (str, bool, int)):
        return repr(x)

    if isinstance(x, (types.FunctionType)):
        return x.__module__ + '.' + x.__name__

    assert False, f"Check this shouldn't be transformed [{repr(x)}]"
    return str(x)

def pytype(x):
    if isinstance(x, jax.ShapedArray):
        return f'ShapedArray({x.shape}, {x.dtype}, {x.weak_type}, {repr(x.named_shape)})'

    return 'Any'

# TODO:
#             bdy_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1013)(bdx_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d2af0>, num_consts=0)
#             bpl_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1019)(bpk_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d29d0>, num_consts=0)
#             brc_ = iota(, dtype=int32, shape=(31, 1), dimension=0)
#             ok_ = scatter-add(oj_,oc_,oi_, update_jaxpr={ [34m[22m[1mlambda [39m[22m[22m; a[35m:f32[39m b[35m:f32[39m. [34m[22m[1mlet[39m[22m[22m c[35m:f32[39m = add a b [34m[22m[1min [39m[22m[22m(c,) }, update_consts=(), dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1, 2), scatter_dims_to_operand_dims=(0, 1, 2)), indices_are_sorted=True, unique_indices=True, mode=GatherScatterMode.PROMISE_IN_BOUNDS)
#             def xla_call1003(a_: ShapedArray(float32[128,32,512]),b_: ShapedArray(int32, weak_type=True)):


def examine_jaxpr(f, jaxpr, *, indent="", doc="", file=sys.stdout):
    args = intercomma((varstr(v) + f": {pytype(v.aval)}" for v in jaxpr.invars))
    print(f"\n{indent}def {f}({args}):", file=file)
    indent += tab
    if doc:
        print(f'{indent}"""{doc}"""', file=file)
    for cv in jaxpr.constvars:
        assert cv not in ['if', 'in', 'is'] # if it is, use varstr(cv) on next line
        print(f"{indent}{cv} = ?", file=file)

    for eqn in jaxpr.eqns:
        sub_jaxpr = None
        if eqn.primitive.name == "custom_jvp_call_jaxpr":
            sub_jaxpr = "fun_jaxpr"
            cj = eqn.params[sub_jaxpr].jaxpr
            pass
        if eqn.primitive.name == "scan":
            sub_jaxpr = "jaxpr"
            cj = eqn.params[sub_jaxpr].jaxpr
            pass
        if eqn.primitive.name == "xla_pmap":
            sub_jaxpr = "call_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass
        if eqn.primitive.name == "scatter-add":
            sub_jaxpr = "update_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass
        if eqn.primitive.call_primitive:
            sub_jaxpr = "call_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass

        if sub_jaxpr:
            pyname = pythonize(eqn.primitive.name)
            n = new_name(pyname)
            primname = f'{pyname}({n})'

            doc = doc_from_source_line(eqn.source_info)
            examine_jaxpr(n, cj, indent=indent, doc=doc, file=file)
            params = intercomma(
                f"{n}={varstr(v)}" for (n, v) in eqn.params.items() if n != sub_jaxpr
            )

        else:
            params = intercomma(f"{n}={varstr(v)}" for (n, v) in eqn.params.items())
            primname = pythonize(str(eqn.primitive))

        if eqn.invars:
            params = intercommavars(eqn.invars) + (', ' + params if params else '')

        print(
            f"{indent}{intercommavars(eqn.outvars)} = {primname}({params})", 
            file=file
        )

    print(f"{indent}return ({intercommavars(jaxpr.outvars)})", file=file)


def show_jaxpr(f, args, file=sys.stdout, **kwargs):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python
    """
    print(f"\n# show_jaxpr", file=file)
    closed_jaxpr = jax.make_jaxpr(f, **kwargs)(*args)
    examine_jaxpr(f.__name__, closed_jaxpr.jaxpr, doc=f.__doc__, file=file)


def show_xla(f, args, file=sys.stdout, **kwargs):
    """
    Show XLA for f, using template args 
    """
    xla = jax.xla_computation(f, **kwargs)(*args)
    print("XLA=", xla.as_hlo_text(), file=file)


def show_jaxpr_and_xla(f, args, file=sys.stdout, **kwargs):
    show_jaxpr(f, args, file=file, **kwargs)
    show_xla(f, args, file=file, **kwargs)


if __name__ == "__main__":
    def f(p,x,q):
        for i in range(5):
            x = jax.numpy.matmul(x,i*p*x)
        return (x + x[3]).std()

    f = jax.grad(f, argnums=1)
    f = jax.vmap(f, in_axes=(None,2,None))
    
    args = (2.2, jax.numpy.ones((3,3,5)), 'q')
    show_jaxpr_and_xla(f, args)

    fjit = jax.jit(f, static_argnums=(0))
    show_jaxpr_and_xla(fjit, args, static_argnums=(0))

    with open("/tmp/show_jaxpr.jaxpr.py", "w") as file:
        show_jaxpr(f, args, file=file)

    import os
    os.system("ls -lrt /tmp/show_jaxpr.jaxpr.py")
