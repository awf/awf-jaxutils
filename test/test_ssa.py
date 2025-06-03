import jax
import jax.numpy as jnp
import numpy as np

from jaxutils.vjp import softmax, relu


import jaxutils.expr as jex
from jaxutils.expr import (
    to_ssa,
    assert_is_ssa,
)
from jaxutils.expr_ast import expr_for, expr_to_python_code
from jaxutils.expr_parser import parse_expr
from jaxutils.expr_eval import eval_expr
from more_itertools import one
import math

import re


def _tostr(e):
    s = jex.expr_format(e, width=40)
    return re.sub(r"# [^\n]*\n", r"# ..\n", s)


def test_uniquify():
    #
    # let
    #   a = let a = 1 in a
    #   a = let a = 2 in a
    #   b = a
    # in
    #   b

    # ->
    # let
    #   a = let a_1 = 1 in a_1
    #   a_2 = let a_3 = 2 in a_3
    #   b = a_2
    # in
    #   b
    jex.reset_new_name_ids()

    example = """
    let
      a = let 
            a = 1;
            b = 3
          in
            f(a, b);
      a = let
            a = 2;
            b = lambda x, b: h(x, a, b) 
          in
            b(a, 42);
      b = a
    in
      tuple(a, b)
    """

    example_out = """
    let
      a = let 
            a_2 = 1;
            b_1 = 3
          in
            f(a_2, b_1);
      a_1 = let
            a_3 = 2;
            b_2 = lambda x, b_3: h(x, a_3, b_3) 
          in
            b_2(a_3, 42);
      b = a_1
    in
      tuple(a_1, b)
    """

    e = parse_expr(example)
    print(_tostr(e))

    u = jex.uniquify_names(e)
    ustr = _tostr(u)
    print(ustr)

    expected = parse_expr(example_out)
    exstr = _tostr(expected)
    print(exstr)

    assert ustr == exstr

    # Check that doing it again doesn't change any more names
    u2 = jex.uniquify_names(u)
    u2str = _tostr(u2)
    print(ustr)
    assert u2str == ustr


def test_to_ssa():
    #
    # let
    #   a = let a = 1 in a
    #   a = let a = 2 in a
    #   b = a
    # in
    #   b

    # ->
    # let
    #   a = let a_1 = 1 in a_1
    #   a_2 = let a_3 = 2 in a_3
    #   b = a_2
    # in
    #   b
    jex.reset_new_name_ids()

    example = """
    let
      a = let 
            a = 1;
            b = 3
          in
            f(a, b);
      a = let
            a = 2;
            b = lambda x, y: h(x, a, y) 
          in
            b(a, 42);
      b = a
    in
      tuple(a, b)
    """

    expected = """
    let
      a_2 = 1;
      b_1 = 3;
      a = f(a_2, b_1);
      a_3 = 2;
      b_2 = lambda x, y: let _ssa_03 = h(x, a_3, y) in _ssa_03;
      a_1 = b_2(a_3, 42);
      b = a_1;
      _ssa_06 = tuple(a_1, b)
    in
      _ssa_06
    """

    e_in = parse_expr(example)
    e = to_ssa(e_in)
    print(e)

    e_expected = parse_expr(expected)
    assert _tostr(e) == _tostr(e_expected)


def test_to_ssa_foo():
    def foo(x):
        return math.sin(x)

    def bar(x):
        y = x + (1 + foo(foo(x)))
        z = (lambda q: q * (q + foo(q)))(y + 2 + 4)
        return z

    e = expr_for(bar, foo)
    print(expr_to_python_code(e, "bar"))
    e = to_ssa(e)
    print(expr_to_python_code(e, "bar"))
    assert_is_ssa(e)
    x = 3.1
    bindings = {"math": math}
    assert eval_expr(e, (x,), bindings) == bar(x)


def ffn_tupled(W, x):
    W1, b1, W2, b2 = W
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2 @ y1 + b2
    return softmax(y2)


def test_to_ssa_ffn():
    e = expr_for(ffn_tupled)
    print(expr_to_python_code(e, "ffn_tupled"))
    e = to_ssa(e)
    print(expr_to_python_code(e, "ffn_tupled"))
    assert_is_ssa(e)

    e = jex.detuple_lets(e)

    e = jex.uniquify_names(e)
    ucs = jex.compute_variable_use_counts(e)
    for k, v in ucs.items():
        print(k, v)

    e = jex.inline_single_usages(e)
    e = jex.dce(e)
    e = to_ssa(e)
    print(expr_to_python_code(e, "ssa"))
    bindings = {"jax": jax, "softmax": softmax, "relu": relu}

    np.random.seed(1)
    rand = lambda *args: jnp.array(np.random.rand(*args))

    B = 1
    W = (rand(11, 7), rand(11, B), rand(10, 11), rand(10, B))
    x = rand(7, B)

    np.testing.assert_allclose(eval_expr(e, (W, x), bindings), ffn_tupled(W, x))


def foo_for_loop_accum(x):
    t1 = 3 * x + 4
    y1 = relu(t1)
    y2 = 0.7 * 0.6 * (y1 * t1)
    zs = 0
    for k in range(4):
        zs += pow(y2, k)
    return zs


from awfutils import import_from_file
import pytest
import operator
from jaxutils.expr_lib import *
from jaxutils.expr import (
    to_ssa_tidy,
    rename_let_v_in_v,
    freevars,
)

from jaxutils.array_expr import (
    annotate_with_shadow_types,
    strip_annotations,
    global_getattrs_to_names,
)


def test_ssa_for_loop_accum():
    x = 3.3
    ret = foo_for_loop_accum(x)
    print(ret)

    e = expr_for(foo_for_loop_accum)

    def check(e):
        bindings = {x.name: eval(x.name) for x in freevars(e)}
        val = eval_expr(e, (x,), bindings, add_operators=False)
        np.testing.assert_allclose(ret, val, atol=1e-4)

    shadow_bindings = {
        "range": range,
        "pow": pow,
        "sum": sum,
        "relu": lambda x: x,
        "softmax": lambda x: x,
    }

    e = annotate_with_shadow_types(e, (x,), shadow_bindings)
    check(e)

    e = global_getattrs_to_names(e)
    check(e)

    print(expr_to_python_code(e, "ssa"))

    e_ssa = to_ssa_tidy(e)
    check(e_ssa)
    e_ssa = jex.optimize(e_ssa)
    check(e_ssa)
    e_ssa = rename_let_v_in_v(e_ssa, "ssa")
    check(e_ssa)
    e = e_ssa
