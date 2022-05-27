import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn


# vjp checker
def check(f, f_grad, *args, verbose=False):
    rng = jax.random.PRNGKey(id(f))

    print("Checking", f, end="...")
    val = f(*args)
    leaves, treedef = jax.tree_flatten(val)
    probe_leaves = []
    for x in leaves:
        rng, rng1 = jax.random.split(rng)
        probe_leaves.append(jax.random.normal(rng1, shape=jnp.shape(x)))
    probe = treedef.unflatten(probe_leaves)

    valj, jax_vjp = jax.vjp(f, *args)
    if verbose:
        print(val)
        print(valj)
    # print(val-valj, end="...")
    assert np.allclose(val, valj)
    gj = jax_vjp(probe)
    gf = f_grad(*args, probe)
    if not isinstance(gf, tuple):
        gf = (gf,)
    if verbose:
        print(jax.tree_structure(gj), "=?=", jax.tree_structure(gf))
    assert jax.tree_structure(gj) == jax.tree_structure(gf)
    tree_close = jax.tree_multimap(np.allclose, gj, gf)
    print(tree_close)
    if verbose:
        print("gj=", gj)
        print("gf=", gf)
    assert all(jax.tree_leaves(tree_close))


def add(x, y):
    return x + y


def add_vjp(x, y, dret):
    return dup(dret)


def test_add():
    check(add, add_vjp, np.random.randn(13, 7), np.random.randn(13, 7))


# dup


def dup(x):
    return (x, x)


def dup_vjp(x, dret):
    return add(*dret)


def test_dup():
    check(dup, dup_vjp, np.random.randn(13, 7))


# pair


def pair(x, y):
    return (x, y)


def pair_vjp(x, y, dret):
    return dret


def test_pair():
    check(pair, pair_vjp, np.random.randn(13, 7), np.random.randn(13, 7))


# linear (axpy)


def linear(A, x, y):
    return A @ x + y


def linear_vjp(A, x, y, dret):
    dA, dx = jnp.outer(dret, x), A.T @ dret
    return (dA, dx, dret)


def test_linear():
    check(
        linear,
        linear_vjp,
        np.random.randn(13, 7),
        np.random.randn(7),
        np.random.randn(13),
    )


# mm


def mm(A, x):
    return A @ x


def mm_vjp(A, x, dret):
    dA, dx = jnp.outer(dret, x), A.T @ dret
    return (dA, dx)


def test_mm():
    check(mm, mm_vjp, np.random.randn(13, 7), np.random.randn(7))


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


def test_vjp():
    check(log, log_vjp, np.random.rand(13, 7))
