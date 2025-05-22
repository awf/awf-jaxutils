import pytest
import sys
import jax
from jaxutils.old_show_jaxpr import old_show_jaxpr


def test_basic():
    def foo(p, x):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    f = vmapgradf

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("f(args)=")
    print(f(*args))

    old_show_jaxpr(f, args, name="f")
    # show_xla(f, args)
    # show_xla(f, args, optimized=True)


@pytest.mark.skip(reason="deprecating old_show_jaxpr")
def test_roundtrip():
    import os
    from awfutils import import_from_file

    def foo(p, x):
        x = jax.numpy.matmul(x, p * x.T)
        return (x + x[3]).std()

    gradf = jax.grad(foo, argnums=1)
    vmapgradf = jax.vmap(gradf, in_axes=(None, 2))

    f = vmapgradf

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (3, 2, 5)))

    print("f(args)=")
    print(f(*args))

    # Save to file
    fn = "tmp/show_jaxpr_jaxpr.py"
    with open(fn, "w") as file:
        old_show_jaxpr(f, args, name="f", file=file, add_decls=True)

    os.system(f"black {fn}")

    # Load from file

    module = import_from_file(fn, "show_jaxpr_roundtrip")

    # Check rountrip: does module.f give the same result?
    assert jnp.allclose(module.f(*args), f(*args))

    # Save again
    fn2 = "tmp/show_jaxpr_roundtrip.py"
    with open(fn2, "w") as file2:
        old_show_jaxpr(module.f, args, file=file2, add_decls=True)

    os.system(f"black {fn2}")

    # Reload for 2nd roundtrip to test string equality
    module2 = import_from_file(fn2, "show_jaxpr_roundtrip2")

    assert jnp.allclose(module2.f(*args), f(*args))

    # Sand save 2nd roundtrip
    fn3 = "tmp/show_jaxpr_roundtrip2.py"
    with open(fn3, "w") as file3:
        old_show_jaxpr(module2.f, args, file=file3, add_decls=True)

    os.system(f"black {fn3}")

    print(f"code --diff {fn2} {fn3} # Do view diffs in vs code")
