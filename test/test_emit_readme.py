import jax
import jax.numpy as jnp
import numpy as np

from jaxutils.jaxpr_to_expr import show_jaxpr


def ffn(W, x):
    ((W1, b1), (W2, b2)) = W
    t1 = W1 @ x + b1
    y1 = jax.nn.relu(t1)
    y2 = W2 @ y1 + b2
    return jax.nn.softmax(y2)


randn = np.random.randn
W = (
    (randn(17, 11), randn(17)),  # W1, b1
    (randn(10, 17), randn(10)),  # W2, b2
)
x = randn(11)


def loss(W, x):
    return -jnp.log(ffn(W, x)[5])


def test_emit_readme():
    show_jaxpr(ffn, (W, x))

    show_jaxpr(jax.grad(loss), (W, x))
