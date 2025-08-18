import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from icecream import ic
from numpy.random import randn


def allclose(ref, test, rtol, atol, verbose=False):
    if jnp.allclose(test, ref, rtol, atol):
        return True

    if verbose:
        ic(ref)
        ic(test)

    return False


def tree_close(tref, ttest, rtol, atol, verbose=True):
    return jax.tree.map(lambda a, b: allclose(a, b, rtol, atol, verbose), ttest, tref)


def tree_all(t):
    return all(jax.tree.leaves(t))


def tree_allclose(tref, ttest, rtol, atol, verbose=True):
    return tree_all(tree_close(tref, ttest, rtol, atol, verbose))


def randlike(x):
    return randn(*x.shape)


def ensure_tuple(val):
    return val if isinstance(val, tuple) else (val,)


# vjp checker.
# Tols set for 32 bit float
def check(f, f_grad, *args, rtol=1e-4, atol=1e-6, verbose=False):
    print("Checking", f, end="...")
    val = f(*args)

    valj, jax_vjp = jax.vjp(f, *args)

    np.testing.assert_allclose(val, valj, rtol, atol)

    probe = jax.tree.map(randlike, val)
    gj = jax_vjp(probe)
    gf = f_grad(*args, *ensure_tuple(probe))
    gf = ensure_tuple(gf)
    if verbose:
        print(jax.tree_util.tree_structure(gj), "=?=", jax.tree_util.tree_structure(gf))
    assert jax.tree_util.tree_structure(gj) == jax.tree_util.tree_structure(gf)
    isclose = tree_close(gj, gf, rtol, atol)
    print(isclose)
    if verbose:
        ic(gj, gf)
        print("diff=", jax.tree.map(lambda a, b: a - b, gj, gf))

    assert tree_all(isclose)


#########################################
##
## Vector-Jacobian products for various simple ops
##

# add


def add(x, y):
    return x + y


def add_vjp(x, y, dret):
    return dup(dret)


def test_add():
    check(add, add_vjp, randn(13, 5), randn(13, 5))


# dup


def dup(x):
    return (x, x)


def dup_vjp(x, *dret):
    return add(*dret)


def test_dup():
    check(dup, dup_vjp, randn(13, 7))


# pair


def pair(x, y):
    return (x, y)


def pair_vjp(x, y, *dret):
    return dret


def test_pair():
    check(pair, pair_vjp, randn(13, 7), randn(13, 7))


# sum

# No need to define sum - it's built in


def sum_vjp(args, dret):
    return list(dret for _ in args)


def test_sum():
    check(sum, sum_vjp, [randn(13, 5), randn(13, 5), randn(13, 5)])


# scale scalar * Tensor


def scale(s, x):
    assert jnp.shape(s) == ()
    return s * x


def scale_vjp(s, x, dret):
    ds = dotall(x, dret)
    dx = scale(s, dret)

    return (ds, dx)


def test_scale():
    check(scale, scale_vjp, 3.3, randn(13, 7))


# axpy


def axpy(A, x, y):
    return A @ x + y


def axpy_vjp(A, x, y, dret):
    dt2, dy = dret, dret
    dA, dx = jnp.outer(dt2, x), A.T @ dt2
    return (dA, dx, dy)


def test_axpy():
    check(axpy, axpy_vjp, randn(13, 7), randn(7), randn(13))


# mm: (A, A) -> dA
# mm_vjp: (A, A, dA) -> (dA, dA)
def mm(A, B):
    return A @ B


def mm_vjp(A, B, dret):
    # nxk kxm, dret: nxm
    dA = mm(dret, B.T)
    dB = mm(A.T, dret)
    return (dA, dB)


def test_mm():
    A = randn(13, 7).astype(np.float64)
    B = randn(7, 3).astype(np.float64)
    with jax.default_matmul_precision("highest"):
        check(mm, mm_vjp, A, B)


# mul: (A, A) -> A
# mul_vjp: (A, A, dA) -> (dA, dA)
def mul(A, B):
    return A * B


def mul_vjp(A, B, dret):
    # nxk kxm, dret: nxm
    dA = dret * B
    dB = A * dret
    return (dA, dB)


def test_mul():
    A = randn(13, 7).astype(np.float64)
    B = randn(13, 7).astype(np.float64)
    with jax.default_matmul_precision("highest"):
        check(mul, mul_vjp, A, B)


# dotall: (A, A) -> R
# dotall_vjp: (A, A, dR) -> (dA, dA)
def dotall(A, B):
    """dot(vec(A), vec(B))"""
    return (A * B).sum()


def dotall_vjp(A, B, dret):
    dA = B * dret
    dB = A * dret
    return (dA, dB)


def test_dotall():
    check(dotall, dotall_vjp, randn(13, 7), randn(13, 7))


# matmul scaled
# mm_scaled: (A, A, R, R) -> A
# mm_scaled_vjp: (A, A, R, R, dA) -> (dA, dA, dR, dR)
def mm_scaled(A, B, sA, sB):
    """
    Take matrices A and B, with associated scale factors sA, sB,
    and compute (sA * A) @ (sB * B) = (sA * sB) * (A @ B)

      C = mm(scale(sA, A), scale(sB, B))
      C = scale(sA*sB, mm(A, B))

    """

    AB = mm(A, B)
    C = scale(sA * sB, AB)
    return C


def mm_scaled_vjp_simple(A, B, sA, sB, dC):
    AB = mm(A, B)
    sC = sA * sB
    C = scale(sC, AB)

    dsC, dAB = scale_vjp(sC, AB, dC)
    dsA, dsB = sB * dsC, sA * dsC
    dA, dB = mm_vjp(A, B, dAB)

    return (dA, dB, dsA, dsB)


def test_mm_scaled_simple():
    check(
        mm_scaled,
        mm_scaled_vjp_simple,
        randn(7, 5),
        randn(5, 3),
        1.23,
        2.34,
    )


def mm_scaled_vjp(A, B, sA, sB, dC):
    dA = sA * mm_scaled(dC, B.T, 1.0, sB)
    dB = sB * mm_scaled(A.T, dC, sA, 1.0)

    if True:  # Want dsA, dsB
        AB = mm(A, B)
        dsC = dotall(AB, dC)
        dsA = sB * dsC
        dsB = sA * dsC

    return (dA, dB, dsA, dsB)


def test_mm_scaled():
    check(
        mm_scaled,
        mm_scaled_vjp,
        randn(7, 5),
        randn(5, 3),
        1.23,
        2.34,
    )


# recip: A -> A
# recip_vjp : (A, dA) -> dA
def recip(x):
    r = 1 / x
    return r


def recip_vjp(x, dr):
    return -dr / x**2


def test_recip():
    check(recip, recip_vjp, 0.001 + np.random.rand(13, 7))


# relu: A -> A
# relu_vjp : (A, dA) -> dA
def relu(x):
    return jnn.relu(x)


def relu_vjp(x, dret):
    return (x > 0) * dret


def test_relu():
    check(relu, relu_vjp, randn(7))


# softmax: A -> A
# softmax_vjp : (A, dA) -> dA
def softmax(x):
    """
    Column softmax
    """
    return jnn.softmax(x, axis=0)


def softmax_vjp(x, dret):
    assert x.shape == dret.shape
    ret = jnn.softmax(x, axis=0)
    return ret * dret - ret * jnp.sum(ret * dret, axis=0)


def test_softmax():
    check(softmax, softmax_vjp, randn(13, 3))


# index: A, i -> R
# index_vjp : (A, i, dR) -> dA
def index(x, i):
    return x[i]


def index_vjp(x, i, dret):
    assert len(x.shape) == 1 or all(v == 1 for v in x.shape[1:])
    return jnn.one_hot(i, len(x)).reshape(x.shape) * dret


def test_index():
    check(
        lambda x: index(x, 3),
        lambda x, dret: index_vjp(x, 3, dret),
        np.random.rand(13),
    )


# transpose: A -> A'
# transpose_vjp : (A, dA') -> dA
def transpose(x):
    return jnp.transpose(x)


def transpose_vjp(x, dret):
    return jnp.transpose(dret)


def test_transpose():
    check(transpose, transpose_vjp, np.random.rand(13, 7))


# log: A -> A
# log_vjp : (A, dA) -> dA
log = jnp.log


def log_vjp(x, dret):
    return dret / x


def test_log():
    check(log, log_vjp, np.random.rand(13, 7))


# exp: A -> A
# exp_vjp : (A, dA) -> dA
exp = jnp.exp


def exp_vjp(x, dret):
    return dret * exp(x)


def test_exp():
    check(exp, exp_vjp, np.random.rand(13, 7))


# pow: (A, Z) -> A
# pow_vjp : (A, Z, dA) -> (dA, None)
def pow_vjp(x, p, dret):
    assert np.issubdtype(type(p), np.integer)
    dx = dret * p * x ** (p - 1)
    dp = dret * log(x) * x**p

    return dx, dp


def test_negate():
    check(negate, negate_vjp, np.random.rand(13, 7))


# negate: A -> A
# negate_vjp : (A, dA) -> dA
def negate(x):
    return -x


def negate_vjp(x, dret):
    return -dret


def test_negate():
    check(negate, negate_vjp, np.random.rand(13, 7))


# range: (Z,Z,Z) -> seq(Z)
# range_vjp : (Z,Z,Z,dseq) -> (dZ,dZ,dZ)
def range_vjp(*args):
    return (None,) * (len(args) - 1)
