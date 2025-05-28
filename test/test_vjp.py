import jax
import jax.numpy as jnp
import numpy as np


def ffn(W, x):
    W1, b1, W2, b2 = W
    t1 = W1 @ x + b1
    y1 = jax.nn.relu(t1)
    y2 = W2 @ y1 + b2
    return jax.nn.softmax(y2)


np.random.seed(1)
rand = lambda *args: jnp.array(np.random.rand(*args))

B = 1
W = (rand(11, 7), rand(11, B), rand(10, 11), rand(10, B))
x = rand(7, B)


def loss(W, x):
    return -jnp.log(ffn(W, x)[5, 0])


import jaxutils.expr as jex
from jaxutils.expr import (
    expr_for,
    expr_to_python_code,
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
from more_itertools import one


def test_to_ssa():
    def foo(x):
        y = x + (1 + foo(foo(x)))
        z = (lambda q: q * (q + foo(q)))(y + 2 + 4)
        return z

    e = expr_for(foo)
    print(expr_to_python_code(e, "foo"))
    e = to_ssa(e)
    print(expr_to_python_code(e, "foo"))
    assert_is_ssa(e)

    e = expr_for(ffn)
    print(expr_to_python_code(e, "foo"))
    e = to_ssa(e)
    print(expr_to_python_code(e, "foo"))
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
            new_lam = Lambda(new_args, body_vjp)

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

    return transform_postorder(transform, e, {})


def global_getattrs_to_names(e):
    def transform(e, bindings):
        if e.isCall and e.f.isVar and e.f.name == "getattr":
            obj = e.args[0]
            attr = e.args[1]
            if obj.isVar and obj.name not in bindings:
                # It's a reference to a global variable, assume it's a module
                return Var(f"{obj.name}.{attr.val}")

    return transform_postorder(transform, e, {})


from jaxutils.vjp import softmax, relu, transpose


def ffn_flat(W1, b1, W2, b2, x):
    t1 = W1 @ x + b1
    y1 = relu(t1)
    y2 = transpose(W2) @ W2 @ (y1 * t1)  # + b2
    return y2
    # return softmax(y2)


from awfutils import import_from_file

import pytest


@pytest.mark.parametrize("opt", ["opt", "raw"])
@pytest.mark.parametrize("ssa", ["ssa", "orig"])
def test_vjp(opt, ssa):
    W1, b1, W2, b2 = W
    print(ffn_flat(W1, b1, W2, b2, x))

    e = expr_for(ffn_flat)
    print(expr_to_python_code(e, "ffn_flat"))

    e = global_getattrs_to_names(e)
    e = inline_var_eq_var(e)

    if ssa == "ssa":
        e_ssa = to_ssa_tidy(e)
        e_ssa = rename_let_v_in_v(e_ssa, "ffn_flat")
        e = e_ssa

    print(expr_to_python_code(e, "ffn_flat"))

    # fvs, vjp_raw = jex.make_vjp(e_ssa, [Var("dffn_flat_ssa")])
    fvs, vjp_raw = jex.make_vjp(e, [Var("dffn_flat")])
    # assert not fvs, f"Free vars in VJP: {fvs}"

    print("\n\n*** VJP (RAW) ***")
    code = expr_to_python_code(vjp_raw, "d_ffn_flat")
    print(code)

    if opt == "raw":
        vjp = vjp_raw
    elif opt == "opt":
        vjp = jex.optimize(vjp_raw)
        vjp = jex.dce(vjp)
    else:
        raise ValueError(f"Unknown optimization option: {opt}")

    print("\n\n*** VJP ***")
    code = expr_to_python_code(vjp, "d_ffn_flat")
    print(code)

    with open("tmp/ffn_flat_vjp.py", "w") as f:
        print(
            """
from jaxutils.expr_lib import g_zeros_like
import operator as ssa_operator
import operator
from jaxutils.vjp import (
    softmax, softmax_vjp,
    relu, relu_vjp,
    transpose, transpose_vjp,
    add_vjp, mm_vjp, mul_vjp,
)

vjp = {
    ssa_operator.__add__: add_vjp,
    ssa_operator.__matmul__: mm_vjp,
    ssa_operator.__mul__: mul_vjp,
    operator.__add__: add_vjp,
    operator.__matmul__: mm_vjp,
    operator.__mul__: mul_vjp,
    softmax: softmax_vjp,
    relu: relu_vjp,
    transpose: transpose_vjp,
}
""",
            file=f,
        )
        print(code, file=f)

    mod = import_from_file("tmp/ffn_flat_vjp.py", "ffn_flat_vjp")

    W1, b1, W2, b2 = W
    d_out10 = rand(11, B)
    g0 = jax.vjp(ffn_flat, W1, b1, W2, b2, x)[1](d_out10)

    g = mod.d_ffn_flat(W1, b1, W2, b2, x, d_out10)

    for g, g0 in zip(g, g0):
        np.testing.assert_allclose(g, g0, atol=1e-5)

    print("VJPs match")
