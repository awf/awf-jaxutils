import jax
import jax.numpy as jnp
import numpy as np

from jaxutils.ParamsDict import ParamsDict
import jaxutils.expr as jex

# Frozen copy of https://github.com/awf/functional-transformer


def matrix_init_uniform(in_features: int, out_features: int):
    rnd_range = 1 / in_features**0.5
    return np.random.rand(in_features, out_features) * (2 * rnd_range) - rnd_range


def transformer_init(
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    d_ff: int,
    max_len=4096,
):
    assert d_k * n_heads == d_model

    # Build config struct for call
    config = ParamsDict()
    config.d_model = d_model
    config.d_ff = d_ff
    config.d_k = d_k
    config.heads = n_heads
    config.lambda_e = d_model**-0.5
    config.lambda_pe = 1.0
    config.tau = 1 / d_k**0.5

    # Build initializers for params
    params = ParamsDict()

    # Create embedding layer
    params.embeddings = np.random.randn(n_vocab, d_model)

    # Positional encodings initialized to zeros
    params.positional_encodings = jnp.zeros((max_len, d_model))

    # For transformer layers
    params.layers = []
    for _ in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = jnp.ones(d_model)

        layer.heads = []
        for _ in range(n_heads):
            head = ParamsDict()
            head.query = matrix_init_uniform(d_model, d_k)
            head.key = matrix_init_uniform(d_model, d_k)
            head.value = matrix_init_uniform(d_model, d_k)

            layer.heads.append(head)

        layer.norm_ff = jnp.ones(d_model)

        layer.ffn1 = matrix_init_uniform(d_model, d_ff)
        layer.ffn2 = matrix_init_uniform(d_ff, d_model)

        params.layers.append(layer)

    # Final normalization and output layer
    params.pre_output_norm = jnp.ones(d_model)
    params.output = matrix_init_uniform(d_model, n_vocab)

    return config, params


def standardize(x, eps):
    return (x - x.mean()) / (x.std() + eps)


def standardize_rows(x):
    return jax.vmap(standardize, in_axes=(0, None))(x, 1e-5)


def softmax(x):
    return jax.nn.softmax(x, axis=1)


# Format off for the size annotations
# fmt: off
def transformer(cfg, params, x):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits

    Obviously, this is just one example of a transformer. There
    are many variations, depending where normalizations go, 
    whether or not there is bias, what kinds of position 
    encodings, etc.
    """
    print("Compiling for L=", x.shape)

    L, = x.shape # x is just 1D. Vmap/pmap will handle batching

    # Create mask: 0 to attend, -Inf to ignore
    mask = jnp.log(jnp.tril(jnp.ones((L, L))))

    # Start with token embeddings
    embeddings = cfg.lambda_e * params.embeddings[x, :]

    # Add (learned) positional encodings
    embeddings += cfg.lambda_pe * params.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in params.layers:

        # Layer-normalize embeddings
        t1 = standardize_rows(embeddings)
        t1 = t1 @ jnp.diag(layer.norm_self_attn)

        # Multi-head self-attention
        self_attns = []
        for head in layer.heads:

            # Project into this head's query/key space
            query = t1 @ head.query
            key = t1 @ head.key

            # Compute L x L attention matrix
            score = query @ key.T + mask
            attn = softmax(cfg.tau * score)

            value = t1 @ head.value
            self_attn = attn @ value

            # Add this head's contribution to the list
            self_attns += [self_attn]  # [LxDk for #heads]

        embeddings += jnp.hstack(self_attns)

        # Layer-normalize embeddings
        t2 = standardize_rows(embeddings)
        t2 = t2 @ jnp.diag(layer.norm_ff)

        # Feedforward fully connected
        t2 = t2 @ layer.ffn1
        t2 = jax.nn.relu(t2)
        t2 = t2 @ layer.ffn2

        # Add this layer's contribution into embeddings
        embeddings += t2

    # Layer-normalize embeddings
    embeddings = standardize_rows(embeddings)
    embeddings = embeddings @ jnp.diag(params.pre_output_norm)

    # And linearly project to output dimension
    return embeddings @ params.output # L x n_vocab 
# fmt: on


def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]


def seq_crossentropy(output: jnp.ndarray, targets: jnp.ndarray):
    return jax.vmap(crossentropy)(output, targets).mean()


def transformer_loss(cfg, params, x, transformer_callable):
    """
    # Transformer loss for one example

    cfg: Config, from init
    params: Current transformer parameters, initialized in init
    x: 1D array of integers, representing the input sequence
    """
    output = transformer_callable(cfg, params, x)

    return seq_crossentropy(output[:-1], x[1:])


def _make_test_data():
    np.random.seed(42)

    # Initialize the transformer
    n_vocab = 17
    cfg, params = transformer_init(
        n_vocab=n_vocab, d_model=22, n_layers=3, n_heads=2, d_k=11, d_ff=43
    )

    # Create a random input sequence
    L = 133
    x = np.random.randint(0, n_vocab, L)

    return cfg, params, x


def test_transformer():
    cfg, params, x = _make_test_data()
    loss = transformer_loss(cfg, params, x, transformer)
    np.testing.assert_almost_equal(loss, 3.107, decimal=3)


from jaxutils.expr import expr_for, expr_to_python_code, freevars
import types
import jaxutils


def test_transformer_to_expr():
    expr_to_python_code(expr_for(transformer), "transformer")

    fn = "tmp/transformer_to_expr.py"

    with open(fn, "w") as f:
        print(
            """
# This file was generated by test_transformer_to_expr.py
from jaxutils.expr_lib import *
import jax
import jax.numpy as jnp
""",
            file=f,
        )

        fvs = {"transformer"}
        while fvs:
            vname = fvs.pop()
            if vname not in globals():
                print(f"Free var {vname} not in globals")
                continue

            v = globals()[vname]
            if isinstance(v, types.ModuleType):
                # Skip modules
                if v.__name__ == vname:
                    print(f"import {v.__name__}")
                else:
                    print(f"import {v.__name__} as {vname}")
                continue

            if vname in dir(jaxutils.expr_lib):
                # Skip names in jaxutils.expr_lib
                continue

            print(f"Free var {vname} in globals, generating code")
            e = expr_for(v)
            print(expr_to_python_code(e, vname), "\n\n", file=f)
            fvs |= {v.name for v in freevars(e)}

    from awfutils import import_from_file

    module = import_from_file(fn, "transformer_via_expr")

    cfg, params, x = _make_test_data()
    loss = transformer_loss(cfg, params, x, module.transformer)
    np.testing.assert_almost_equal(loss, 3.107, decimal=3)


from jaxutils.expr import Var, Call, Const


def resolve_getattr(e, bindings=None):
    def doit(e, bindings):
        if e.isCall and e.f == Var("getattr"):
            obj, attr = e.args
            if obj.isVar and attr.isConst:
                attr = attr.val
                if attr == "T":
                    return Call(Var("g_transpose"), [obj])
                elif obj.name in bindings:
                    val = bindings[obj.name]
                    if isinstance(val, types.ModuleType):
                        return Var(f"{val.__name__}.{attr}")
            pass

    return jex.transform_postorder(doit, e, bindings or {})


def mkcall(name: str):
    return lambda *args: (name, *args)


def my_getattr(obj, attr):
    if isinstance(obj, types.ModuleType):
        return mkcall(f"{obj.__name__}.{attr}")
    else:
        return mkcall("getattr")(obj, attr)

    #  g_pairs_to_dict,jax,print,standardize_rows


def test_transformer_vjp():
    e = expr_for(transformer)
    cfg, params, x = _make_test_data()

    # bindings = dict(
    #     jax=jax,
    #     jnp=jnp,
    #     standardize_rows=mkcall("standardize_rows"),
    #     softmax=mkcall("softmax"),
    #     print=print,
    # )
    # bindings |= {k: mkcall(k) for k in jex._bindings_for_operators()}
    # bindings |= {k: mkcall(k) for k in jex._bindings_for_expr_lib()}
    # bindings["getattr"] = my_getattr
    # bindings["**g_pairs_to_dict"] = mkcall("g_pairs_to_dict")
    # e2 = jex.eval_expr(e, (cfg, Var("params"), Var("x")), bindings, add_operators=False)
    # print(e2)

    # bindings = {"jax": jax, "jnp": jnp, "jax.nn": jax.nn}
    # e = resolve_getattr(e, bindings)

    e = jex.uniquify_names(e)
    vars, vjp = jex.make_vjp(e, [jex.Var("f")])

    with open("tmp/transformer_vjp.py", "w") as f:
        print(expr_to_python_code(vjp, "transformer_vjp"), file=f)

    vjpo = jex.optimize(vjp)

    with open("tmp/transformer_vjpo.py", "w") as f:
        print(expr_to_python_code(vjpo, "transformer_vjpo"), file=f)
