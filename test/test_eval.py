import operator


from jaxutils.expr import (
    Let,
    Eqn,
    Call,
    Const,
    Var,
    mkvars,
)
from jaxutils.expr_eval import _run_eval


def test_eval():

    a, b, c = mkvars("a,b,c")

    f_defs = {
        "add": operator.add,
        "g_tuple": lambda *args: tuple(args),
        "getattr": getattr,
    }

    e = Call(Var("add"), [Const(2), Const(3)])
    v = _run_eval(e, f_defs)
    assert v == 5

    v = _run_eval(
        Let(
            [
                Eqn([a, b], Call(Var("g_tuple"), [Const(2), Const(3)])),
            ],
            Call(Var("add"), [a, b]),
        ),
        f_defs,
    )
    assert v == 5
