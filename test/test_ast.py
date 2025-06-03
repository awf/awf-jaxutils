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

from jaxutils.expr import (
    Let,
    Eqn,
    Call,
    Const,
    Var,
    Lambda,
    mkvars,
)
from jaxutils.expr_ast import (
    expr_to_python_code,
    expr_for,
    to_ast,
)
from jaxutils.expr_eval import eval_expr


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
    print(ast.unparse(a))

    e = _make_e()
    a = to_ast(e, "e")
    print(ast.unparse(a))

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
