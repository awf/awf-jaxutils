from more_itertools import one

import wadler_lindig as wl

from jaxutils.expr import (
    Expr,
    Let,
    Eqn,
    Call,
    Const,
    Var,
    Lambda,
)


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


from jaxutils.expr import mkvars


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
