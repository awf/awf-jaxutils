from jaxutils.expr import (
    Let,
    Eqn,
    Call,
    Const,
    Lambda,
    mkvars,
)


def _make_e():
    # Make an expr for testing
    import math

    foo, w, x, y, z = mkvars("foo, w, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    a_lam = Lambda([z], Call(v_sin, [Call(v_sin, [z])]), "a")
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
        "foo",
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
        print(e)
