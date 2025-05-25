from more_itertools import one
from prettyprinter import pretty_call
from prettyprinter import cpprint
from prettyprinter import register_pretty, pretty_call

from prettyprinter import register_pretty

# Wadler constructors
from prettyprinter.doc import (
    # flat_choice,
    annotate,  # annotations affect syntax coloring
    concat,
    group,  # make what's in here a single line if enough space
    nest,  # newlines within here are followed by indent + arg
    align,
    hang,
    # NIL,
    LINE,  # Space or Newline
    # SOFTLINE,  # nothing or newline
    HARDLINE,  # newline
)


from jaxutils.expr import (
    Expr,
    Let,
    Eqn,
    Call,
    Const,
    Var,
    Lambda,
    freevars,
    preorder_visit,
    _run_eval,
    to_ast,
    ast_to_expr,
    mkvars,
    expr_to_python_code,
    expr_for,
    eval_expr,
    transform_postorder,
    let_to_lambda,
    is_global_function_name,
)


@register_pretty(Const)
def _(e, ctx):
    return repr(e.val)


@register_pretty(Var)
def _(e, ctx):
    return str(e.name)


@register_pretty(Call)
def _(e, ctx):
    # args = [pretty_call(ctx, arg) for arg in e.args]
    if e.f.isVar:
        return pretty_call(ctx, e.f.name, *e.args)
    else:
        return pretty_call(ctx, "call", e.f, *e.args)


@register_pretty(Let)
def _(e, ctx):
    if any(len(eqn.vars) > 1 for eqn in e.eqns):
        # If any equation has multiple variables, just pass through
        return pretty_call(ctx, "let", e.eqns, body=e.body)
    else:
        kwargs = {one(eqn.vars).name: eqn.val for eqn in e.eqns}
        c = pretty_call(ctx, "let", **kwargs, body=e.body)
        return c


@register_pretty(Lambda)
def _(e, ctx):
    args = [arg.name for arg in e.args]
    return pretty_call(ctx, "lambda", args=e.args, body=e.body)


import wadler_lindig as wl


def wl_pdoc(e):
    if not isinstance(e, Expr):
        return None

    recurse = lambda x: wl.pdoc(x, custom=wl_pdoc)
    spc = wl.BreakDoc(" ")
    semi = wl.BreakDoc("; ")

    if e.isVar:
        return wl.TextDoc(e.name)

    if e.isConst:
        return wl.TextDoc(repr(e.val))

    if e.isCall:
        fn = recurse(e.f)
        brk = wl.BreakDoc("")
        args = [recurse(arg) for arg in e.args]
        args = wl.join(wl.comma, args)
        args = wl.NestDoc(wl.ConcatDoc(brk, wl.GroupDoc(args)), indent=2)
        return fn + wl.TextDoc("(") + args + brk + wl.TextDoc(")")

    if e.isLambda:
        args = [wl.TextDoc(arg.name) for arg in e.args]
        args = wl.join(wl.comma, args)
        head = (wl.TextDoc("lambda") + wl.TextDoc(" ") + args + wl.TextDoc(":")).group()
        body = recurse(e.body)
        return (wl.TextDoc("{") + head + spc + body + wl.TextDoc("}")).group().nest(2)

    if e.isLet:

        def doeqn(vars, val):
            vars = wl.join(wl.comma, list(map(recurse, vars)))
            return (
                ((vars + spc + wl.TextDoc("=")).group() + spc + recurse(val))
                .group()
                .nest(2)
            )

        eqns = [doeqn(eqn.vars, eqn.val) for eqn in e.eqns]
        eqns = wl.join(semi, eqns)
        body = recurse(e.body)
        let_doc = (wl.TextDoc("let") + spc + eqns).group().nest(2)
        in_doc = (wl.TextDoc("in") + spc + body).group().nest(2)
        return (let_doc + spc + in_doc).group()

    return None


def _make_e():
    # Make an expr for testing
    import math

    foo, w, x, y, z = mkvars("foo, w, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    a_lam = Lambda([z], Call(v_sin, [Call(v_sin, [z])]))
    call_lam = Call(a_lam, [x])
    foo_lam = Lambda(
        [x, y],
        Let(
            [
                Eqn([w, x], Call(v_add, [x, Const(3.3)])),
                Eqn([z], Call(v_add, [x, w])),
                Eqn([y], Call(v_mul, [z, w])),
            ],
            Call(v_mul, [call_lam, Let([Eqn([y], Const(2.2))], y)]),
        ),
    )
    e = Let(
        [
            Eqn([foo], foo_lam),
            Eqn([z], Call(v_sin, [Const(1.1)])),
        ],
        Call(foo, [Const(1.1), Const("some string")]),
    )
    return e


def test_pretty():
    e = _make_e()
    for width in (80, 40, 30):
        print(f"\n --- Width = {width} ---\n")
        wl.pprint(e, custom=wl_pdoc, width=width)
