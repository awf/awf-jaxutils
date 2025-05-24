# Definitions of global functions used in expr_to_python_code


def g_tuple(*a):
    return tuple(a)


def g_fst(a):
    return a[0]


def g_pairs_to_dict(*pairs):
    it = iter(pairs)
    return {a: b for (a, b) in zip(it, it)}


g_slice = slice


def g_scan(f, init, xs):
    carry = init
    for x in xs:
        carry = f(carry, x)
    return carry


def g_list(*args):
    return list(args)


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
