import types
import sys
import re
import numpy as np
from typing import Callable, Optional, Any
from beartype import beartype

import jax
import jax._src
import jax._src.core
from jax._src.interpreters import pxla
import jaxlib.xla_extension as xla_ext
from pprint import pprint


from jax import lax
import jax.numpy as jnp

from jaxutils.expr import Expr, Var, Const, Lambda, Let, Call

from jaxutils.expr import to_ast, freevars, uniquify_names, mkvars, transform

import ast


def tree_all(t):
    return all(jax.tree_util.tree_leaves(t))


def unspecified(x):
    return tree_all(jax.tree_map(jax._src.sharding_impls.is_unspecified, x))


def toExpr(x) -> Expr:
    if isinstance(
        x,
        (
            tuple,
            jax.lax.GatherScatterMode,
            jax.lax.GatherDimensionNumbers,
            str,
            bool,
            int,
            np.float32,
            np.int32,
            float,
            types.NoneType,
            np.dtype,
        ),
    ):
        return Const(x)

    if isinstance(x, jax._src.core.Literal):
        return Const(x.val)

    if isinstance(x, jax.core.Jaxpr):
        return jaxpr_to_expr(x)

    if isinstance(x, jax.core.ClosedJaxpr):
        return jaxpr_to_expr(x.jaxpr)

    if isinstance(x, jax.core.Var):
        return Var(f"v_{x.count:02d}{x.suffix}")

    # This check just to ensure we have eyeballed all cases that need to be 'repr'ed
    # Explicitly add verified cases to
    assert False, f"Check {type(x)} shouldn't be transformed [{repr(x)}]"


translators: dict[jax.core.Primitive, Callable[..., Optional[Expr]]] = {}

PJIT_PASSTHRU = True


def translate_pjit(
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
    if not PJIT_PASSTHRU:
        return Call(Var("pjit"), [jaxpr] + list(*args))

    if unspecified(in_shardings.val) and unspecified(out_shardings.val):
        return Call(jaxpr, list(*args))


translators[jax._src.pjit.pjit_p] = translate_pjit


def jaxpr_to_expr(jaxpr) -> Lambda:
    """
    Convert Jaxpr to jaxutils.Expr
    """

    # Will return Lambda(args, body)
    body = mkTuple([toExpr(v) for v in jaxpr.outvars])

    for cv in reversed(jaxpr.constvars):
        body = Let(toExpr(cv), Const(cv.aval), body)

    for eqn in reversed(jaxpr.eqns):
        new_params = {key: toExpr(val) for key, val in eqn.params.items()}
        args = [toExpr(v) for v in eqn.invars]

        val = None
        if eqn.primitive in translators:
            val = translators[eqn.primitive](args, **new_params)

        if val is None:
            params = [Call(Var(key + "="), [val]) for key, val in new_params.items()]
            val = Call(Var(eqn.primitive.name), args + params)

        vars = map(toExpr, eqn.outvars)
        body = Let([*vars], val, body)

    args = [toExpr(v) for v in jaxpr.invars]
    return Lambda(args, body)


def mkTuple(es: list[Expr]) -> Expr:
    if len(es) == 1:
        return es[0]
    else:
        return Call(Var("tuple"), es)


def all_equal(xs, ys):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x != y:
            return False
    return True


def test_all_equal():
    xs = [1, 2, 3]
    ys = [1, 2]
    assert not all_equal(xs, ys)
    assert not all_equal([], ys)
    assert not all_equal(xs, [])
    assert all_equal(ys, ys)
    assert all_equal(xs, xs)


test_all_equal()


def inline_call_of_lambda(e: Expr) -> Expr:
    # call(lambda l_args: body, args)
    #  ->  let l_args = args in body
    # Name clashes will happen unless all bound var names were uniquified
    if e.isCall and e.f.isLambda:
        return Let(e.f.args, mkTuple(e.args), e.f.body)


def inline_trivial_letbody(e: Expr) -> Expr:
    # let var = val in var -> val
    if e.isLet and len(e.vars) == 1:
        if e.vars == [e.body]:
            return e.val


def inline_lambda_of_call(e: Expr) -> Expr:
    # Lambda(args, Call(f, args)) -> f
    if e.isLambda:
        if e.body.isCall:
            if all_equal(e.body.args, e.args):
                return e.body.f


def detuple_tuple_assignments(e: Expr) -> Expr:
    # Let([a, b, c], Call(tuple, [aprime, bprime, cprime]),
    #       body) ->
    #  Let([a], aprime,
    #    Let([b], bprime,
    #      Let([c], cprime,
    #           body)))
    if e.isLet and len(e.vars) > 1:
        if e.val.isCall and e.val.f == Var("tuple"):
            new_body = e.body
            for lhs, rhs in reversed(tuple(zip(e.vars, e.val.args))):
                new_body = Let([lhs], rhs, new_body)
            return new_body


def to_ssa(e):
    e = uniquify_names(e)
    assignments = []
    expr = to_ssa_aux(e, assignments)
    # now rip through assignments, making Lets
    for vars, val in reversed(assignments):
        expr = Let(vars, val, expr)
    return expr


def to_ssa_aux(e, assignments):
    if e.isConst or e.isVar:
        return e

    if e.isLet:
        new_val = to_ssa_aux(e.val, assignments)
        inner_assignments = []
        abody = to_ssa_aux(e.body, inner_assignments)
        assign = (e.vars, new_val)
        assignments += [assign]
        assignments += inner_assignments
        return abody

    if e.isLambda:
        return Lambda(e.args, to_ssa(e.body))

    if e.isCall:
        new_f = to_ssa_aux(e.f, assignments)
        new_args = [to_ssa_aux(arg, assignments) for arg in e.args]
        return Call(new_f, new_args)

    assert False


def is_trivial(e: Expr):
    if e.isVar:
        return True

    if e.isConst and isinstance(e.val, (float, str, int)):
        return True

    return False


def inline_trivial_assignments(e):
    return inline_trivial_assignments_aux(e, {})


@beartype
def inline_trivial_assignments_aux(e: Expr, translations: dict[str, Expr]):
    # let a = b in body -> body[a->b]

    recurse = inline_trivial_assignments_aux

    if e.isConst:
        return e

    if e.isVar:
        return translations.get(e.name, e)

    if e.isLet:
        # let vars = val in body
        if len(e.vars) == 1 and is_trivial(e.val):
            argset = {v.name for v in e.vars}
            new_translations = {
                name: val for (name, val) in translations.items() if name not in argset
            }

            assert e.vars[0].name not in new_translations
            new_translations[e.vars[0].name] = e.val
            return recurse(e.body, new_translations)
        else:
            new_val = recurse(e.val, translations)
            new_body = recurse(e.body, translations)
            return Let(e.vars, new_val, new_body)

    if e.isLambda:
        argset = {v.name for v in e.args}
        new_translations = {
            name: val for (name, val) in translations.items() if name not in argset
        }
        new_body = recurse(e.body, new_translations)
        return Lambda(e.args, new_body)

    if e.isCall:
        new_f = recurse(e.f, translations)
        new_args = [recurse(arg, translations) for arg in e.args]
        return Call(new_f, new_args)

    assert False


def test_inline_trivial_assignments():
    a, b, c, d = mkvars("a,b,c,d")
    e = Let([a], b, Call(a, [Let([a], c, Call(a, [a, b, c]))]))
    out = inline_trivial_assignments(e)
    pprint(out)
    expect = Call(b, [Call(c, [c, b, c])])
    assert out == expect


@beartype
def run_eval(e: Expr, bindings: dict[Var, Any]) -> Any:
    recurse = run_eval

    if e.isConst:
        return e.val

    if e.isVar:
        return bindings[e]

    if e.isCall:
        new_f = recurse(e.f, bindings)
        new_args = [recurse(arg, bindings) for arg in e.args]
        assert isinstance(new_f, Callable)
        return new_f(*new_args)

    if e.isLet:
        # let vars = val in body
        argset = {v.name for v in e.vars}
        new_bindings = {
            name: val for (name, val) in bindings.items() if name not in argset
        }

        tupval = recurse(e.val, bindings)
        assert isinstance(tupval, tuple)

        for var, val in zip(e.vars, tupval):
            new_bindings[var] = val

        return recurse(e.body, new_bindings)

    if e.isLambda:

        def runLambda(e, vals, bindings):
            argset = {v.name for v in e.args}
            new_bindings = {
                name: val for (name, val) in bindings.items() if name not in argset
            }
            for var, val in zip(e.args, vals):
                new_bindings[var] = val

            return recurse(e.body, new_bindings)

        return lambda vals: runLambda(e, vals, bindings)

    assert False


def test_eval():
    import operator

    a, b, c = mkvars("a,b,c")
    f_tuple, f_add = mkvars("tuple, add")
    f_defs = {
        f_add: operator.add,
        f_tuple: lambda *args: tuple(args),
    }

    v = run_eval(Call(f_add, [Const(2), Const(3)]), f_defs)
    assert v == 5

    v = run_eval(
        Let([a, b], Call(f_tuple, [Const(2), Const(3)]), Call(f_add, [a, b])), f_defs
    )
    assert v == 5



def show_jaxpr(f, args, name=None, file=sys.stdout, add_decls=False, **kwargs):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python

    Several optimization passes are run, to simplify the resulting code.
    If global arg PJIT_PASSTHRU is True, then pjit(f)(args) with unspecified sharding
    is just rewritten to f(args), which can result in significantly simpler code.
    """

    if name is None:
        name = f.__name__

    doc = f.__doc__

    jaxpr = jax.make_jaxpr(f, **kwargs)
    closed_jaxpr = jaxpr(*args)
    print(closed_jaxpr)

    e = jaxpr_to_expr(closed_jaxpr.jaxpr)

    def signature(e):
        return set.union(freevars(e), {Var("tuple")})

    global sig
    sig = signature(e)

    def check(ex):
        global sig
        new_sig = signature(ex)
        assert sig == new_sig
        # sig = new_sig - if we wanted to keep going

    e = uniquify_names(e)
    check(e)

    e = transform(inline_call_of_lambda, e)
    check(e)
    e = transform(inline_trivial_letbody, e)
    check(e)
    e = transform(inline_lambda_of_call, e)
    check(e)
    e = to_ssa(e)
    check(e)
    e = transform(detuple_tuple_assignments, e)
    check(e)
    e = inline_trivial_assignments(e)
    check(e)
    print(ast.unparse(to_ast(e, "e")))


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
