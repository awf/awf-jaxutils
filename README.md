# awf-jaxutils

A few utils for JAX (and general python) programming

## Usage

This is very much code-in-progress.  When I use it, I typically just put it as a submodule under whatever else I'm building, so I can easily bugfix jaxutils as I do other work:

```sh
$ git submodule add https://github.com/awf/awf-jaxutils jaxutils
$ python -c 'from jaxutils.ParamsDict import ParamsDict ; p = ParamsDict(); p.a = 4; p.print()'
/a: 4
```

# ParamsDict

A helper, like the [chex dataclass](https://github.com/deepmind/chex#dataclass-dataclasspy), for storing parameters of learning models.

Compared to chex, this involves less typing, but hence less type-safety.
You might like to start with `ParamsDict` and then move to dataclass.

```python
# The chex way.  Grown up and safe.
@chex.dataclass
class LinearParameters:
  W: chex.ArrayDevice
  b: chex.ArrayDevice

def linear_init() -> LinearParameters:
  return LinearParameters(
      W=rand(5,7),
      b=rand(7)
  )

@chex.dataclass
class Parameters:
  x: chex.ArrayDevice
  y: chex.ArrayDevice
  layer: LinearParameters

def my_init() -> Parameters:
  return Parameters(
    x=jnp.ones((2, 2)),
    y=jnp.ones((1, 2)),
    layer=linear_init()
  )
```

```python
# The jaxutils way.  Keeps us [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) in the early development stage.
def linear_init() -> ParamsDict:
    params = ParamsDict()
    params.W = rand(5,7)
    params.b = rand(7)
    return params

def my_init() -> ParamsDict:
    params = ParamsDict()
    params.x = jnp.ones((2, 2))
    params.y = jnp.ones((1, 2))
    params.layer = linear_init()
    return params
```

# Show JAXPR

A 'decompiler' for jaxprs back into python
```python
def ffn(W, x):
    ((W1,b1),(W2,b2)) = W
    t1 = W1 @ x + b1
    y1 = jnn.relu(t1)
    y2 = W2 @ y1 + b2
    return jnn.softmax(y2)
```
And the JAXPR, not super pretty, but perhaps more readable than the XLA, at least for python programmers
```python
# show_jaxpr | black
def ffn(
    a_: ShapedArray((11, 7), float32, False, {}),
    b_: ShapedArray((11,), float32, False, {}),
    c_: ShapedArray((10, 11), float32, False, {}),
    d_: ShapedArray((10,), float32, False, {}),
    e_: ShapedArray((7,), float32, False, {}),
):
    f_ = dot_general(
        a_,
        e_,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=None,
        preferred_element_type=None,
    )
    g_ = add(f_, b_)

    def custom_jvp_call_jaxpr1000(a_: ShapedArray((11,), float32, False, {})):
        """/tmp/ipykernel_10736/3248565795.py:7:ffn"""

        def xla_call1001(a_: ShapedArray((11,), float32, False, {})):
            """/tmp/ipykernel_10736/3248565795.py:7:ffn"""
            b_ = max(a_, 0.0)
            return b_

        b_ = xla_call(xla_call1001)(
            a_,
            device=None,
            backend=None,
            name="relu",
            donated_invars=(False,),
            inline=False,
            keep_unused=False,
        )
        return b_

    h_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1000)(
        g_, jvp_jaxpr_thunk=jax.interpreters.partial_eval.memoized, num_consts=0
    )
    i_ = dot_general(
        c_,
        h_,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=None,
        preferred_element_type=None,
    )
    j_ = add(i_, d_)
    k_ = reduce_max(j_, axes=(0,))
    l_ = broadcast_in_dim(k_, shape=(1,), broadcast_dimensions=())
    m_ = stop_gradient(l_)
    n_ = sub(j_, m_)
    o_ = exp(n_)
    p_ = reduce_sum(o_, axes=(0,))
    q_ = broadcast_in_dim(p_, shape=(1,), broadcast_dimensions=())
    r_ = div(o_, q_)
    return r_
```

# VJP: Vector-Jacobian Products

A collection of small and composable vjps, to simplify hand-derived chain rule programming.

E.g. given this network and loss definition
```python
def ffn(W, x):
    ((W1,b1),(W2,b2)) = W
    t1 = mm(W1, x) + b1
    y1 = relu(t1)
    y2 = mm(W2, y1) + b2
    return softmax(y2)

def loss(W, x, l):
    z = ffn(W,x)
    return -jnp.log(z[l])
```
we might want to explore custom loss gradients.
So first we write it in jaxpr-like form
```python
def loss0(W, x, l):
    ((W1,b1),(W2,b2)) = W
    s1 = mm(W1, x)
    t1 = add(s1, b1)
    y1 = relu(t1)
    p2 = mm(W2, y1)
    y2 = add(p2, b2)
    z = softmax(y2)
    zl = index(z,l)
    return -jnp.log(zl)
```
and manually or otherwise get the very simple grad:
```python
def grad_loss0_manual(W, x, l):
    ((W1,b1),(W2,b2)) = W
    s1 = mm(W1, x)
    t1 = add(s1, b1) 
    y1 = relu(t1_long)
    p2 = mm(W2, y1)
    y2 = add(p2, b2)
    z = softmax(y2)
    zl = index(z,l)
    val0 = -log(zl)

    # Reverse..
    dzl = log_vjp(zl, -1)
    dz = index_vjp(z,l,dzl)
    dy2 = softmax_vjp(y2,dz)
    dp2,db2 = add_vjp(dy2)
    dW2,dy1 = mm_vjp(W2, y1, dp2)
    dt1 = relu_vjp(t1, dy1) 
    ds1,db1 = add_vjp(dt1)
    dW1,dx = mm_vjp(W1, x, ds1)
    return ((dW1,db1),(dW2,db2))
```
And then we would manually work on that to get faster code:
```python
def grad_loss2(W, x, l):
    ((W1,b1),(W2,b2)) = W
    q1 = W1 @ x + b1        
    long_y1 = jnn.relu(q1)  # long lived
    t2 = W2 @ long_y1 
    y2 = t2 + b2
    z = jnn.softmax(y2)
    zl = z[l]
    # Reverse..
    dy2 = z.at[l].add(-1.0)
    dW2 = jnp.outer(dy2,long_y1)      # dW2 large, rank1, but must be returned
    dy1 = W2.T @ dy2
    dq1 = (long_y1 > 0) * dy1 
    dW1 = jnp.outer(dq1,x)        

    return ((dW1,dq1),(dW2,dy2))
```
And now a gradient-accumulating version, allowing more optimization
```python
W_Type = Pair[Pair[Tensor,Tensor],Pair[Tensor,Tensor]]

def grad_loss2(W : W_Type, x, l, dW_in : W_Type) -> W_Type:
    ((W1,b1),(W2,b2)) = W
    q1 = W1 @ x + b1
    long_y1 = jnn.relu(q1)
    t2 = W2 @ long_y1
    y2 = t2 + b2
    z = jnn.softmax(y2)
    zl = z[l]
    # Reverse..
    ((dW1_in,db1_in),(dW2_in,db2_in)) = dW_in
    dy2 = z.at[l].add(-1.0)
    dW2 = xpyzt(dW2_in,dy2,long_y1) # Efficient x + yz^T    
    dy1 = W2.T @ dy2
    dq1 = jnp.sign(long_y1) * dy1 
    dW1 = xpyzt(dW1_in, dq1,x)  # Efficient x + yz^T
    db1 = db1_in + dq1
    db2 = db2_in + dy2

    return ((dW1,db1),(dW2,db2))
```
A lot of these optimizations _can_ be done by XLA, but if they're not, it's useful to be able to do them by hand.