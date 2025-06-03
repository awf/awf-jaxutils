import jax
import jax.numpy as jnp
import numpy as np

from jaxutils.expr_eval import eval_expr
from jaxutils.vjp import softmax, relu


def ffn_tupled(W, x):
    W1, b1, W2, b2 = W
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2 @ y1 + b2
    return softmax(y2)


np.random.seed(1)
rand = lambda *args: jnp.array(np.random.rand(*args))

B = 1
W = (rand(11, 7), rand(11, B), rand(10, 11), rand(10, B))
x = rand(7, B)


def loss(W, x):
    return -jnp.log(ffn_tupled(W, x)[5, 0])


import jaxutils.expr as jex
from jaxutils.expr import (
    freevars,
    Expr,
    Let,
    Eqn,
    Call,
    Var,
    Const,
    Lambda,
    to_ssa,
    to_ssa_tidy,
    assert_is_ssa,
    transform_postorder,
    rename_let_v_in_v,
)
from jaxutils.expr_ast import expr_for, expr_to_python_code
from jaxutils.expr_parser import parse_expr
from more_itertools import one
import math


def test_to_ssa_foo():
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
    example = """
    let
      a = let 
            a = 1;
            b = 3
          in f(a, b);
      a = let a = 2; b = lambda x, y: h(x, a, y) in b(a, 42)
      ;
      b = a
    in
      b
    """

    e = parse_expr(example)
    print(to_ssa(e))


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
    np.testing.assert_allclose(eval_expr(e, (W, x), bindings), ffn_tupled(W, x))


def make_vjp_ssa(e: Expr, d_ins: list[Var] = None) -> Expr:
    """
    # The python transform:
    def foo(ins1, ins2):
      outs1 = foo(ins1, 8)
      outs2 = bar(outs1, ins2)
      return outs2

    def foo_vjp(ins1, ins2, d_outs2):
      outs1 = foo(ins1, 8)
      outs2 = bar(outs1, ins2)

      d_outs1, d_ins2 += bar_vjp(outs1, ins2, d_outs2)
      d_ins1, d_8 += foo_jvp(ins1, 8, d_outs)
      return d_ins1, d_ins2

    # The Expr transform:
    let foo = lambda ins1, ins2:
      let
        outs1 = foo(ins1, 8)
        outs2 = bar(outs1, ins2)
      in
        outs2
    let foo_vjp = lambda ins1, ins2, d_outs2:
      let
        outs1 = foo(ins1)
        outs2 = bar(outs1, ins2)
        d_outs1, d_ins2 += bar_vjp(outs1, ins2, d_outs2)
        d_ins1, d_8 += foo_jvp(ins1, 8, d_outs1)
      in
        d_ins1, d_ins2

    """

    def d(v):
        return Var("d_" + v.name)

    if e.isVar:
        return d(e)

    if e.isConst:
        return Var("_")

    if e.isLet:

        def make_for_call(eqn):
            assert eqn.val.isCall
            assert all(v.isVar or v.isConst for v in eqn.val.args)
            assert eqn.val.f.isVar

            ins = eqn.val.args
            d_ins = [make_vjp_ssa(v) for v in ins]
            outs = eqn.vars
            d_outs = [make_vjp_ssa(v) for v in outs]

            f_vjp = Var(f"vjp[{eqn.val.f.name}]")
            val = Call(f_vjp, ins + d_outs)
            return Eqn(d_ins, val)

        def make_for_lambda(eqn):
            """
            lambda ins: let eqns in outs
            ->
            lambda ins, d_outs: let D[eqns] in d_ins
            """

            lam = eqn.val
            assert lam.isLambda
            assert lam.body.isLet

            out = lam.body.body
            assert out.isVar
            d_out = make_vjp_ssa(out)

            d_ins = [make_vjp_ssa(v) for v in lam.args]
            body_vjp = make_vjp_ssa(lam.body, d_ins)
            new_args = lam.args + [d_out]
            new_lam = Lambda(new_args, body_vjp, lam.id + "/vjp_ssa")

            var = one(eqn.vars)
            return Eqn([d(var)], new_lam)

        def make_for_eqn(eqn):
            if eqn.val.isCall:
                return make_for_call(eqn)
            if eqn.val.isLambda:
                return make_for_lambda(eqn)
            if eqn.val.isVar:
                # out = in
                # ->
                # d_in = d_out
                vin = eqn.val
                vout = one(eqn.vars)
                d_in = d(vin)
                d_out = d(vout)
                return Eqn([d_in], d_out)
            assert False, f"Unknown eqn type {eqn.val}"

        eqn_vjps = [make_for_eqn(eqn) for eqn in reversed(e.eqns)]
        eqns = e.eqns + eqn_vjps
        assert e.body.isVar
        body = make_vjp_ssa(e.body)
        return Let(eqns, jex.mkTuple(d_ins))

    # if e.isLambda:
    #     assert e.body.isLet
    #     body_vjp = make_vjp(e.body)
    #     args = e.args + body_vjp.args
    #     return Lambda(args, body_vjp)

    # if e.isCall:
    #     assert e.f.isVar
    #     f_vjp = Var(e.f.name + "_vjp")
    #     assert all(v.isVar for v in e.args)
    #     d_ins = [make_vjp(v) for v in e.args]
    #     return Call(f_vjp, e.args + d_outs)

    assert False


from jaxutils.array_expr import (
    annotate_with_shadow_types,
    strip_annotations,
    ShadowArray,
    global_getattrs_to_names,
)


def inline_var_eq_var(e):
    def transform(e, bindings):
        if e.isLet:
            new_eqns = []
            for eqn in e.eqns:
                new_val = eqn.val
                if eqn.val.isVar and eqn.val.name in bindings:
                    # Inline the variable
                    new_val = bindings[eqn.val.name]

                new_eqns += [Eqn(eqn.vars, new_val)]
            return Let(new_eqns, e.body)

    return transform_postorder("v=v", transform, e, {})


from jaxutils.vjp import softmax, relu, transpose


def foo_ffn(W1, b1, W2, b2, x):
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2 @ y1 + b2
    return softmax(y2)


def foo_funny(W1, b1, W2, b2, x):
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2.T @ W2 @ (y1 * t1)  # + b2
    return softmax(y2)


def foo_list_append(W1, b1, W2, b2, x):
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2.T @ W2 @ (y1 * t1)  # + b2
    zs = []
    for k in range(4):
        zs += [pow(y2, k)]
    sums = sum(zs)
    return softmax(sums * 1.0e-6)


def for_loop_accum(W1, b1, W2, b2, x):
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = W2.T @ W2 @ (y1 * t1)  # + b2
    zs = 0
    for k in range(4):
        zs += pow(y2, k)
    return softmax(zs * 1.0e-6)


from awfutils import import_from_file
import pytest
import operator
from jaxutils.expr_lib import *
from jaxutils.expr_eval import annotate_eval

from enum import Enum


def ExprArg(*args):
    pass


def ExprRet(*args):
    pass


def ExprCall(*args):
    return ExprRet


def annotate_expr_base(e: Expr) -> Expr:
    """
    Annotate the expression with extremely minimal "types".
    All function calls return type "ExprRet", and all arguments are of type "ExprArg".
    The main value is that it means that any expr whose value is a Lambda is identifiable,
    using `e.annot.__name__ == "runLambda"`.
    """
    bindings = {v.name: ExprCall for v in freevars(e)}
    bindings["g_scan"] = lambda f, init, xs, *args: f(init, xs, *args)
    return annotate_eval(e, [ExprArg] * 5, bindings, add_operators=False)


@pytest.mark.parametrize("opt", ["raw", "opt"])
@pytest.mark.parametrize("ssa", ["ssa", "orig"])
@pytest.mark.parametrize(
    "funcname",
    [
        "foo_ffn",
        "foo_funny",
        "for_loop_accum",
        "foo_list_append",
    ],
)
def test_vjp(funcname, opt, ssa):
    func = globals()[funcname]
    W1, b1, W2, b2 = W
    ret = func(W1, b1, W2, b2, x)
    print(ret)

    global_names = {"pow"}
    e = expr_for(func, global_names=global_names)

    def check(e):
        bindings = {x.name: eval(x.name) for x in freevars(e)}
        val = eval_expr(e, (W1, b1, W2, b2, x), bindings, add_operators=False)
        np.testing.assert_allclose(ret, val, atol=1e-4)

    print(expr_to_python_code(e, funcname))

    if ssa == "ssa":
        e = to_ssa_tidy(e)
        check(e)

    print(expr_to_python_code(e, funcname))

    e = annotate_expr_base(e)  # needed before vjp
    e = global_getattrs_to_names(e)

    dfuncname = "d" + e.body.name
    fvs, vjp_raw = jex.make_vjp(e, [Var(dfuncname)])

    print("\n\n*** VJP (RAW) ***")
    code = expr_to_python_code(vjp_raw, dfuncname)
    print(code)

    if opt == "raw":
        vjp = vjp_raw
    elif opt == "opt":
        vjp = jex.optimize(vjp_raw)
        vjp = jex.dce(vjp)
    else:
        raise ValueError(f"Unknown optimization option: {opt}")

    vjp = strip_annotations(vjp)

    print("\n\n*** VJP ***")
    code = expr_to_python_code(vjp, dfuncname)
    print(code)

    filename = f"tmp/test_vjp_tmp_{funcname}_{ssa}_{opt}.py"
    with open(filename, "w") as f:
        print(
            """
import jax
from jaxutils.expr_lib import *

import operator as ssa_operator
import operator
from jaxutils.vjp import (
    softmax, softmax_vjp,
    relu, relu_vjp,
    transpose, transpose_vjp,
    add_vjp, mm_vjp, mul_vjp,
    sum_vjp, pow_vjp, range_vjp
)

g_vjp_table |= {
    ssa_operator.__add__: add_vjp,
    ssa_operator.__matmul__: mm_vjp,
    ssa_operator.__mul__: mul_vjp,
    operator.__add__: add_vjp,
    operator.__matmul__: mm_vjp,
    operator.__mul__: mul_vjp,
    softmax: softmax_vjp,
    relu: relu_vjp,
    transpose: transpose_vjp,
    sum: sum_vjp,
    pow: pow_vjp,
    range: range_vjp,
}
""",
            file=f,
        )
        print(code, file=f)

    mod = import_from_file(filename, "test_vjp_tmp")

    dret = rand(*ret.shape)
    g0 = jax.vjp(func, W1, b1, W2, b2, x)[1](dret)

    dfunc = getattr(mod, dfuncname)
    g = dfunc(W1, b1, W2, b2, x, dret)

    for g, g0 in zip(g, g0):
        np.testing.assert_allclose(g, g0, atol=1e-4)

    print("VJPs match")
