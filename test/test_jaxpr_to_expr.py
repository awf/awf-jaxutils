import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np

import jaxutils
from jaxutils.jaxpr_to_expr import show_jaxpr

from awfutils import import_from_file


def test_basic():
    def foo(p, x):
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))

    print("f(args)=")
    print(gradf(*args))

    show_jaxpr(gradf, args, name="gradf")

    # pjit not supported on JaxDecompiler
    # from JaxDecompiler import decompiler

    # decompiled_f, python_code = decompiler.python_jaxpr_python(
    #     gradf, args, is_python_returned=True
    # )
    # print(python_code)


def test_emit_readme():
    def ffn(W, x):
        ((W1, b1), (W2, b2)) = W
        t1 = W1 @ x + b1
        y1 = jax.nn.relu(t1)
        y2 = W2 @ y1 + b2
        return jax.nn.softmax(y2)

    W = (
        (np.random.rand(11, 7), np.random.rand(11)),
        (np.random.rand(10, 11), np.random.rand(10)),
    )
    x = np.random.rand(7)

    show_jaxpr(ffn, (W, x))

    show_jaxpr(jax.grad(lambda W, x: -jnp.log(ffn(W, x)[5])), (W, x))


def test_vmap():
    def foo(p, x):
        x = x @ (p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))
    vargs = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("foo(args)=")
    print(foo(*args))

    show_jaxpr(foo, args, name="f")

    show_jaxpr(vmapgradf, vargs, name="vmapgradf")


@pytest.mark.parametrize("n", ("f", "vmapgradf"))
def test_roundtrip(n):
    import os

    def foo(p, x):
        x = jax.numpy.matmul(x, p * x.T)
        x = jax.nn.relu(x)
        x = jax.nn.softmax(x)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    prng = jax.random.PRNGKey(42)
    if n == "f":
        f = foo
        args = (2.2, jax.random.normal(prng, (2, 5)))
    elif n == "vmapgradf":
        f = vmapgradf
        args = (2.2, jax.random.normal(prng, (3, 2, 5)))
    else:
        assert False, f"Unknown test {n}"

    print("f(args)=")
    print(f(*args))

    def save(f, args, name, fn):
        jaxutils.expr.reset_new_name_ids()

        with open(fn, "w") as file:
            show_jaxpr(f, args, name=name, file=file, add_decls=True)

        # os.system(f"black -q -l 120 {fn}")
        os.system(f"sed -E 's/\\bv\\w+/v__/g' {fn} > {fn}.v__")

    # Save to file
    fn = "tmp/show_jaxpr_jaxpr.py"
    save(f, args, "f", fn)

    # Load from file
    module = import_from_file(fn, "show_jaxpr_roundtrip")

    # Check roundtrip: does module.f give the same result?
    assert jnp.allclose(module.f(*args), f(*args))

    # Save again
    fn2 = "tmp/show_jaxpr_roundtrip.py"
    save(module.f, args, "f", fn2)

    # Does it render the same?  Probably not, as nested calls have been optimized,
    # changing the results of uniquify_names, even with uniquify_names
    os.system(f"diff -yb -W120 {fn}.v__ {fn2}.v__")

    print(f"code --diff {fn} {fn2} # Do view diffs in vs code")
