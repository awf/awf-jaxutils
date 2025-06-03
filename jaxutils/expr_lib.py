import jax.numpy as jnp

# Definitions of global functions used in expr_to_python_code

g_vjp_table = {}


def g_vjp(f):
    if f in g_vjp_table:
        return g_vjp_table[f]
    raise KeyError(f"VJP for {f} not found in g_vjp_table. ")


def g_identity(x):
    return x


def g_tuple(*a):
    return tuple(a)


def g_fst(a):
    return a[0]


def g_pairs_to_dict(*pairs):
    it = iter(pairs)
    return {a: b for (a, b) in zip(it, it)}


g_slice = slice


def g_scan(f, init, xs, *args):
    carry = init
    for x in xs:
        carry = f(carry, x, *args)
    return carry


def g_scan_vjp(f_and_vjp, init, xs, *args):
    _, f, df = f_and_vjp

    dret = args[-1]
    args = args[:-1]
    carries = [init]
    # TODO: Quadratic if carry is appending lists of lists
    # (but GPU memory not quadratic, the lists will point to reused values)
    for x in xs:
        carries += [f(carries[-1], x, *args)]

    # assuming ratio of list length to compute is small, so using indexing for readability
    n = len(xs)
    dcarry = dret
    dxs = [0] * n
    dargs = [0] * len(args)
    for i in reversed(range(n)):
        dci, dxi, dargsi = df(carries[i], xs[i], *args, dcarry)
        dxs[i] = dxi
        for k in range(len(args)):
            dargs[k] += dargsi
        dcarry = dci

    return None, dcarry, dxs, *dargs


g_vjp_table[g_scan] = g_scan_vjp


def g_list(*args):
    return list(args)


def g_zeros_like(x):
    return jnp.zeros_like(x)


def g_subscript(x, inds):
    if isinstance(inds, tuple):
        return x[*inds]
    else:
        return x[inds]


# g_for = Var("g_for")
# g_fst = Var("g_fst")
# g_scan = Var("g_scan")
# g_tuple = Var("g_tuple")
# g_slice = Var("g_slice")
# g_list = Var("g_list")
# g_identity = Var("g_identity")
