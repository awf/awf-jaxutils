import jax
import jax.numpy as jnp
import numpy as np


def ffn(W, x):
    W1, b1, W2, b2 = W
    t1 = W1 @ x + b1
    y1 = jax.nn.relu(t1)
    y2 = W2 @ y1 + b2
    return jax.nn.softmax(y2)


rand = np.random.rand
W = (rand(11, 7), rand(11), rand(10, 11), rand(10))
x = rand(7)


def loss(W, x):
    return -jnp.log(ffn(W, x)[5])


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
)


def make_vjp(e: Expr, d_outs: list[Var] = None) -> Expr:
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
    if e.isVar:
        return Var("d_" + e.name)

    if e.isConst:
        return Var("drop")

    if e.isLet:

        def make_for_call(eqn):
            assert eqn.val.isCall
            assert all(v.isVar or v.isConst for v in eqn.val.args)
            assert eqn.val.f.isVar

            ins = eqn.val.args
            d_ins = [make_vjp(v) for v in ins]
            outs = eqn.vars
            d_outs = [make_vjp(v) for v in outs]

            f_vjp = Var(eqn.val.f.name + "_vjp")
            val = Call(f_vjp, ins + d_outs)
            return Eqn(d_ins, val)

        def make_for_lambda(eqn):
            lam = eqn.val
            assert lam.isLambda
            assert lam.body.isLet
            assert lam.body.body.isVar
            out = lam.body.body
            d_out = make_vjp(out)

            body_vjp = make_vjp(lam.body)
            # d_ins = [make_vjp(v) for v in lam.args]
            args = lam.args + [d_out]
            return Eqn(args, Lambda(args, body_vjp))

        def make_for_eqn(eqn):
            if eqn.val.isCall:
                return make_for_call(eqn)
            if eqn.val.isLambda:
                return make_for_lambda(eqn)
            assert False, f"Unknown eqn type {eqn.val}"

        eqn_vjps = [make_for_eqn(eqn) for eqn in reversed(e.eqns)]
        eqns = e.eqns + eqn_vjps
        assert e.body.isVar
        body = make_vjp(e.body)
        return Let(eqns, body)

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


from itertools import chain


def to_ssa(e: Expr) -> Let | Var | Const:
    """
    Convert an expression to SSA form.
    """
    if e.isVar or e.isConst:
        return e

    if e.isLet:
        # let
        #   vs1 = e1
        #   vs2 = e2
        #   vs3 = var3
        # in
        #   e4
        # becomes
        # let
        #   vs1 = let eqns1 in var1
        #   vs2 = let eqns2 in var2
        #   vs3 = var3
        # in
        #   let eqns4 in var4
        # becomes
        # let
        #   eqns1
        #   vs1 = var1
        #   eqns2
        #   vs2 = var2
        #   vs3 = var3
        #   eqns4
        # in var4

        eqns = []
        for eqn in e.eqns:
            val = to_ssa(eqn.val)
            if val.isLet:
                eqns += val.eqns
                val = val.body
            eqns += [Eqn(eqn.vars, val)]

        body = to_ssa(e.body)
        if body.isLet:
            eqns += body.eqns
            body = body.body

        return Let(eqns, body)

    if e.isLambda:
        # lambda args: body
        # becomes
        # let f = lambda args: let eqns in body
        # in f
        body = to_ssa(e.body)
        lam = Lambda(e.args, body)
        var = Var("f" + jex.get_new_name())
        return Let([Eqn([var], lam)], var)

    if e.isCall:
        #   Call(f, e1, e2, ...)
        # becomes
        #   Call(let eqns0 in var0, let eqns1 in var1, let eqns2 in var2, ...)
        # becomes
        #   let
        #     eqns0
        #     eqns1
        #     eqns2
        #     out = Call(vs0, vs1, vs2)
        #   in
        #     out

        eqns = []
        f = to_ssa(e.f)
        if f.isLet:
            eqns += f.eqns
            f = f.body

        args = []
        for v in e.args:
            v = to_ssa(v)
            if v.isLet:
                eqns += v.eqns
                args += [v.body]
            else:
                args += [v]

        call_val = Call(f, args)
        call_var = Var("out" + jex.get_new_name())
        return Let(eqns + [Eqn([call_var], call_val)], call_var)

    assert False


def assert_is_ssa(e: Expr) -> bool:
    """
    Check if an expression is in SSA form.
    """

    def is_val(e):
        return e.isVar or e.isConst

    if e.isLet:
        # let
        #   vs1 = rhs1
        #   vs2 = rhs2
        #   vs3 = rhs3
        # in
        #   body
        # is SSA if all rhss are vals and body is val
        # and also recurse into lambdas
        for eqn in e.eqns:
            if eqn.val.isLambda or eqn.val.isCall:
                assert_is_ssa(eqn.val)
            else:
                assert is_val(eqn.val)

    elif e.isLambda:
        assert_is_ssa(e.body)

    elif e.isCall:
        assert is_val(e.f)
        assert all(map(is_val, e.args))
    else:
        assert is_val(e)


def to_ssa_tidy(e):
    """
    Convert an expression to SSA form and tidy it up.
    """
    e = to_ssa(e)
    e = jex.detuple_lets(e)
    e = jex.inline_trivial_rhs(e)
    e = jex.dce(e)
    e = to_ssa(e)
    return e


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

    # ucs = jex.compute_variable_use_counts(e)
    # for k, v in ucs.items():
    #     print(k, v)

    e = jex.inline_trivial_rhs(e)
    e = jex.dce(e)
    e = to_ssa(e)
    print(expr_to_python_code(e, "ssa"))


def test_vjp():
    e = expr_for(ffn)
    print(expr_to_python_code(e, "ffn"))

    e = to_ssa_tidy(e)
    print(expr_to_python_code(e, "ffn"))

    vjp = make_vjp(e)
    print(expr_to_python_code(vjp, "ffn_vjp"))

    def ffn_vjp(W_1_0, x_1_0, d_out11):
        out08_0 = W_1_0[2]
        out09_0 = W_1_0[0]
        out0a_0 = out09_0 @ x_1_0
        out0b_0 = W_1_0[1]
        out0c_0 = out0a_0 + out0b_0
        out0d_0 = jax.nn.relu(out0c_0)
        out0e_0 = out08_0 @ out0d_0
        out0f_0 = W_1_0[3]
        out10_0 = out0e_0 + out0f_0
        out11_0 = jax.nn.softmax(out10_0)
        d_out10 = jax.nn.softmax_vjp(out10_0, d_out11)
        d_out0e, d_out0f = operator.__add___vjp(out0e_0, out0f_0, d_out10)
        d_W_1, drop = g_subscript_vjp(W_1_0, 3, d_out0f)
        d_out08, d_out0d = operator.__matmul___vjp(out08_0, out0d_0, d_out0e)
        d_out0c = jax.nn.relu_vjp(out0c_0, d_out0d)
        d_out0a, d_out0b = operator.__add___vjp(out0a_0, out0b_0, d_out0c)
        d_W_1_0, drop_0 = g_subscript_vjp(W_1_0, 1, d_out0b)
        d_out09, d_x_1 = operator.__matmul___vjp(out09_0, x_1_0, d_out0a)
        d_W_1_1, drop_1 = g_subscript_vjp(W_1_0, 0, d_out09)
        d_W_1_2, drop_2 = g_subscript_vjp(W_1_0, 2, d_out08)
        return d_out11
