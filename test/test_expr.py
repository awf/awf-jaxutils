import pytest
import sys
import enum
from dataclasses import dataclass
from beartype import beartype
from beartype.typing import List, Set, Any, Tuple, Dict, List, Callable, Optional
from pprint import pprint
from itertools import chain
import operator

import ast

if sys.version_info >= (3, 9):
    astunparse = ast
else:
    import astunparse

from jaxutils.expr import (
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


def test_eval():

    a, b, c = mkvars("a,b,c")

    f_defs = {
        "add": operator.add,
        "tuple": lambda *args: tuple(args),
        "getattr": getattr,
    }

    e = Call(Var("add"), [Const(2), Const(3)])
    v = _run_eval(e, f_defs)
    assert v == 5

    v = _run_eval(
        Let(
            [
                Eqn([a, b], Call(Var("tuple"), [Const(2), Const(3)])),
            ],
            Call(Var("add"), [a, b]),
        ),
        f_defs,
    )
    assert v == 5


def test_let_to_lambda():
    e = _make_e()
    l = transform_postorder("let_to_lambda", let_to_lambda, e, {})

    def check(e):
        assert not e.isLet

    preorder_visit(l, check, {})


def test_ast():
    a, b, c = mkvars("a,b, c")
    e = Let(
        [
            Eqn([a, b], Const(123)),
            Eqn([c], Call(Var("add"), [a, b])),
        ],
        Const(234),
    )
    pprint(e)
    a = to_ast(e, "e")
    print(astunparse.unparse(a))

    e = _make_e()
    a = to_ast(e, "e")
    print(astunparse.unparse(a))

    # code = compile(a, 'bar', 'exec')
    # exec(code)


def test_ast_to_expr():
    import jax.numpy as jnp

    def foo(f):
        a, b = 123, 2
        c = f * b
        d = jnp.sin(c)
        return (lambda x: d + x)(c)

    expected = foo(5)

    e = expr_for(foo)
    print(expr_to_python_code(e, "foo"))

    got = eval_expr(e, [5], {"jnp": jnp})
    assert expected == got


def test_ast_to_expr2():
    import jax

    def foo(p, x):
        x = x @ (p * x.T)
        return (x + x[3]).std()

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))

    expected = foo(*args)
    print(expected)

    e_foo = expr_for(foo)
    print(expr_to_python_code(e_foo, "foo"))

    got = eval_expr(e_foo, args)

    assert expected == got


def test_ast_raises_default_arg():
    def bar(a):
        def foo(x, y=1):
            return x + y

        return foo(4)

    with pytest.raises(ValueError, match="Default arguments not implemented"):
        expr_for(bar)


def test_freevars():
    v_x = Var("x")
    v_y = Var("y")
    v_z = Var("z")
    v_foo = Var("foo")

    def fv(s):
        e = ast_to_expr(ast.parse(s), [])
        fvs = freevars(e)
        return {v for v in fvs if not is_global_function_name(v.name)}

    assert fv("x + y") == {v_x, v_y}
    assert fv("def foo(): x(1); return (lambda x: x + y)(x(2))") == {v_x, v_y}
