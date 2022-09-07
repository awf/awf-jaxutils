import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from icecream import ic


def allclose(ref, test, rtol, atol, verbose=False):
    if jnp.allclose(test, ref, rtol, atol):
        return True

    if verbose:
        ic(ref)
        ic(test)

    return False


def tree_close(tref, ttest, rtol, atol, verbose=True):
    return jax.tree_map(lambda a, b: allclose(a, b, rtol, atol, verbose), ttest, tref)


def tree_all(t):
    return all(jax.tree_util.tree_leaves(t))


def tree_allclose(tref, ttest, rtol, atol, verbose=True):
    return tree_all(tree_close(tref, ttest, rtol, atol, verbose))


def randlike(x):
    return np.random.randn(*x.shape)


def ensure_tuple(val):
    return val if isinstance(val, tuple) else (val,)


# vjp checker.
# Tols set for 32 bit float
def check(f, f_grad, *args, rtol=1e-5, atol=1e-7, verbose=False):
    print("Checking", f, end="...")
    val = f(*args)

    valj, jax_vjp = jax.vjp(f, *args)
    if verbose:
        ic(val, valj)

    assert tree_all(tree_close(val, valj, rtol, atol))

    probe = jax.tree_map(randlike, val)
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
        print("diff=", jax.tree_map(lambda a, b: a - b, gj, gf))

    assert tree_all(isclose)


def add(x, y):
    return x + y


def add_vjp(x, y, dret):
    return dup(dret)


def test_add():
    check(add, add_vjp, np.random.randn(13, 5), np.random.randn(13, 5))


# dup


def dup(x):
    return (x, x)


def dup_vjp(x, *dret):
    return add(*dret)


def test_dup():
    check(dup, dup_vjp, np.random.randn(13, 7))


# pair


def pair(x, y):
    return (x, y)


def pair_vjp(x, y, *dret):
    return dret


def test_pair():
    check(pair, pair_vjp, np.random.randn(13, 7), np.random.randn(13, 7))


# axpy


def axpy(A, x, y):
    return A @ x + y


def axpy_vjp(A, x, y, dret):
    dt2, dy = dret, dret
    dA, dx = jnp.outer(dt2, x), A.T @ dt2
    return (dA, dx, dy)


def test_axpy():
    check(
        axpy, axpy_vjp, np.random.randn(13, 7), np.random.randn(7), np.random.randn(13)
    )


# mm


def mm(A, B):
    return A @ B


def mm_vjp(A, B, dret):
    # nxk kxm, dret: nxm
    dA = mm(dret, B.T)
    dB = mm(A.T, dret)
    return (dA, dB)


def test_mm():
    check(mm, mm_vjp, np.random.randn(13, 7), np.random.randn(7, 3))


# dotall


def dotall(A, B):
    """dot(vec(A), vec(B))"""
    return (A * B).sum()


def dotall_vjp(A, B, dret):
    dA = B * dret
    dB = A * dret
    return (dA, dB)


def test_dotall():
    check(dotall, dotall_vjp, np.random.randn(13, 7), np.random.randn(13, 7))


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


def mm_scaled_vjp(A, B, sA, sB, dC) -> Tuple[F16, F16, F16, F16]:
    AB = mm(A, B)
    sC = sA * sB
    C = scale(sC, AB)

    dsC, dAB = scale_vjp(sC, AB, dC)
    dsA, dsB = sB * dsC, sA * dsC
    dA, dB = mm_vjp(A, B, dAB)

    return (dA, dB, dsA, dsB)


def test_mm_scaled():
    check(
        mm_scaled,
        mm_scaled_vjp,
        np.random.randn(7, 5),
        np.random.randn(5, 3),
        1.23,
        2.34,
    )


def mm_scaledopt_vjp(A, B, sA, sB, dC) -> Tuple[F16, F16, F16, F16]:
    s = sA * sB

    dA = s * mm(dC, B.T)
    dB = s * mm(A.T, dC)

    if True:  # Want dsA, dsB
        AB = mm(A, B)
        dsC = dotall(AB, dC)
        dsA = sB * dsC
        dsB = sA * dsC

    return (dA, dB, dsA, dsB)


def test_mm_scaledopt():
    check(
        mm_scaled,
        mm_scaledopt_vjp,
        np.random.randn(7, 5),
        np.random.randn(5, 3),
        1.23,
        2.34,
    )


# scale


def scale(s, x):
    assert jnp.shape(s) == ()
    return s * x


def scale_vjp(s, x, dret):
    ds = dotall(x, dret)
    dx = scale(s, dret)

    return (ds, dx)


def test_scale():
    check(scale, scale_vjp, 3.3, np.random.randn(13, 7))


# recip
def recip(x):
    r = 1 / x
    return r


def recip_vjp(x, dr):
    return -dr / x**2


def test_recip():
    check(recip, recip_vjp, 0.001 + np.random.rand(13, 7))


# relu


def relu(x):
    return jnn.relu(x)


def relu_vjp(x, dret):
    return (x > 0) * dret


def test_relu():
    check(relu, relu_vjp, np.random.randn(7))


# softmax


def softmax(x):
    return jnn.softmax(x)


def softmax_vjp(x, dret):
    ret = jnn.softmax(x)
    return ret * dret - ret * jnp.dot(dret, ret)


def test_softmax():
    check(softmax, softmax_vjp, np.random.randn(13))


# index


def index(x, i):
    return x[i]


def index_vjp(x, i, dret):
    ret = jnn.softmax(x)
    return jnn.one_hot(i, len(x)) * dret


def test_index():
    check(
        lambda x: index(x, 3), lambda x, dret: index_vjp(x, 3, dret), np.random.rand(13)
    )


# index
log = jnp.log


def log_vjp(x, dret):
    return dret / x


def test_log():
    check(log, log_vjp, np.random.rand(13, 7))
