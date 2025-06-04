from jaxutils.expr import (
    Let,
    Eqn,
    Call,
    Const,
    Lambda,
    freevars,
    preorder_visit,
    mkvars,
    transform_postorder,
    let_to_lambda,
)


def _make_e():
    # Make an expr for testing
    import math

    foo, w, x, y, z = mkvars("foo, w, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    a_lam = Lambda([z], Call(v_sin, [z]), "a")
    call_lam = Call(a_lam, [x])
    foo_lam = Lambda(
        [x, y],
        Let(
            [
                Eqn([w], Call(v_add, [x, Const(3.3)])),
                Eqn([z], Call(v_add, [x, w])),
            ],
            Call(v_mul, [call_lam, Let([Eqn([y], v_sin)], y)]),
        ),
        "foo",
    )
    e = Let(
        [
            Eqn([foo], foo_lam),
        ],
        Call(foo, [Const(1.1), Const(2.2)]),
    )
    return e


def test_basic():
    foo, x, y, z = mkvars("foo, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    e = _make_e()

    print(e)
    assert freevars(e) == {v_sin, v_add, v_mul}
    assert freevars(e.eqns[0].val) == {v_sin, v_add, v_mul}
    assert freevars(e.eqns[0].val.body) == {x, v_sin, v_add, v_mul}


def test_visit():
    e = _make_e()
    l = list(preorder_visit(e, lambda ex, bindings: type(ex).__name__, {}))
    print(l)


def test_let_to_lambda():
    e = _make_e()
    l = transform_postorder(let_to_lambda, e, {})

    def check(e):
        assert not e.isLet

    preorder_visit(l, check, {})
