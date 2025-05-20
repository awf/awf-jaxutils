import sys
import enum
from dataclasses import dataclass
from beartype import beartype
from beartype.typing import List, Set, Any, Tuple, Dict, List, Callable, Optional
from pprint import pprint
from itertools import chain
import operator

import ast

if sys.version_info >= (3, 9):
    astunparse = ast
else:
    import astunparse


### TODO: General utils - should move elsewhere
def dictassign(d: Dict[Any, Callable], key: Any):
    """
    A decorator to add functions to dicts.
       ```
       @dictassign(fundict, 'times')
       def my_times_impl():
          ...
       ```
    is the same as
       ```
       def my_times_impl():
          ...
       fundict['times'] = my_times_impl
       ```
    The advantages are readability: we can see at the time of `def` that
    this function will go in `fundict`.  In fact, the name of the function
    is irrelevant: recommended usage is
    ```
       @dictassign(fundict, 'times')
       def _():
          ...
    ```
    In this case (i.e. if the wrapped function's name is '_'), dictassign
    will replace the name of the wrapped function with `_@dictassign_{str(key)}`,
    which gives useful documentation in backtraces.
    """

    def wrapper(func):
        d[key] = func

        if func.__name__ == "_":
            newname = "_@dictassign_" + str(key)
            func.__name__ = newname
            func.__code__ = func.__code__.replace(co_name=newname)

            if func.__qualname__ != "_":
                print(f"NOTE: {func.__qualname__} not overridden")

            if func.__qualname__ == "_":
                func.__qualname__ = newname

        return func

    return wrapper


def all_equal(xs, ys):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x != y:
            return False
    return True


def test_all_equal():
    xs = [1, 2, 3]
    ys = [1, 2]
    assert not all_equal(xs, ys)
    assert not all_equal([], ys)
    assert not all_equal(xs, [])
    assert all_equal(ys, ys)
    assert all_equal(xs, xs)


test_all_equal()

### New name factory

_expr_global_name_id = 0


def get_new_name():
    global _expr_global_name_id
    _expr_global_name_id += 1
    return f"{_expr_global_name_id:02d}"


def reset_new_name_ids():
    global _expr_global_name_id
    _expr_global_name_id = 0


## Declare Expr classes


########################################################################################
#
#
#   88888888888
#   88
#   88
#   88aaaaa     8b,     ,d8 8b,dPPYba,  8b,dPPYba,
#   88"""""      `Y8, ,8P'  88P'    "8a 88P'   "Y8
#   88             )888(    88       d8 88
#   88           ,d8" "8b,  88b,   ,a8" 88
#   88888888888 8P'     `Y8 88`YbbdP"'  88
#                           88
#                           88
########################################################################################

@beartype
class Expr:
    """
    A classic tiny-Expr library.

    These are the subclasses

        Const:   Constant
        Var:     Variable name
        Let:     Let (vars1 = val1, ..., varsN = valsN) in Expr
        Lambda:  Lambda vars . body
        Call:    Call (func: Expr) (args : List[Expr])

    All operators (add, mul, etc) are just calls

    Methods:

      freevars(e: Expr) -> List[Var]
      let_to_lambda(e: Expr) -> Expr   # Remove all "Let" nodes, replacing with Lambdas

    """

    @property
    def isConst(self):
        return isinstance(self, Const)

    @property
    def isVar(self):
        return isinstance(self, Var)

    @property
    def isCall(self):
        return isinstance(self, Call)

    @property
    def isLambda(self):
        return isinstance(self, Lambda)

    @property
    def isLet(self):
        return isinstance(self, Let)


def exprclass(klass, **kwargs):
    """
    Decorator to simplify declaring expr subclasses.
    """
    return beartype(dataclass(klass, frozen=True, **kwargs))


@exprclass
class Const(Expr):
    val: Any


@exprclass
class Var(Expr):
    name: str


@exprclass
class Eqn:
    vars: List[Var]
    val: Expr


@exprclass
class Let(Expr):
    eqns: List[Eqn]
    body: Expr


@exprclass
class Lambda(Expr):
    args: List[Var]
    body: Expr


@exprclass
class Call(Expr):
    f: Expr
    args: List[Expr]


def mkvars(s: str) -> Tuple[Var]:
    """
    Little convenience method for making Vars, like sympy.symbols

    x,y,z = mkvars('x,y,z')
    """
    s = s.replace(" ", "")
    return [Var(s) for s in s.split(",")]


def isNone(x: Optional[Expr]) -> bool:
    return x is None or x.isConst and x.val is None


def transform_postorder(transformer: Callable[[Expr], Expr], e: Expr = None):
    if e is None:
        # Make transform(transformer)(e) work like transform(transformer, e)
        return lambda e: transform_postorder(transformer, e)

    recurse = lambda e: transform_postorder(transformer, e)

    # Recurse into children
    if e.isLet:
        new_eqns = [Eqn(eqn.vars, recurse(eqn.val)) for eqn in e.eqns]
        new_body = recurse(e.body)
        e = Let(new_eqns, new_body)

    if e.isLambda:
        e = Lambda(e.args, recurse(e.body))

    if e.isCall:
        new_f = recurse(e.f)
        new_args = [recurse(arg) for arg in e.args]
        e = Call(new_f, new_args)

    # And pass self to the transformer, with updated children
    return transformer(e) or e


def freevars(e: Expr) -> Set[Var]:
    if e.isConst:
        return set()

    if e.isVar:
        return {e}

    if e.isLet:
        bound_vars = set()
        fv_vals = set()
        for eqn in e.eqns:
            fv_vals |= set.difference(freevars(eqn.val), bound_vars)
            bound_vars = set.union(bound_vars, set(eqn.vars))
        fv_body = set.difference(freevars(e.body), bound_vars)
        return set.union(fv_vals, fv_body)

    if e.isLambda:
        return set.difference(freevars(e.body), e.args)

    if e.isCall:
        return set.union(freevars(e.f), *(freevars(arg) for arg in e.args))

    assert False


def _make_e():
    # Make an expr for testing
    import math

    foo, w, x, y, z = mkvars("foo, w, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    a_lam = Lambda([z], Call(v_sin, [z]))
    call_lam = Call(a_lam, [x])
    foo_lam = Lambda(
        [x, y],
        Let(
            [
                Eqn([w], Call(v_add, [x, Const(3.3)])),
                Eqn([z], Call(v_add, [x, w])),
            ],
            Call(v_mul, [call_lam, y]),
        ),
    )
    e = Let(
        [
            Eqn([foo], foo_lam),
        ],
        Call(foo, [Const(1.1), Const(2.2)]),
    )
    return e


def test_basic():
    foo, x, y, z = mkvars("foo, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    e = _make_e()

    pprint(e)
    assert freevars(e) == {v_sin, v_add, v_mul}
    assert freevars(e.eqns[0].val) == {v_sin, v_add, v_mul}
    assert freevars(e.eqns[0].val.body) == {x, y, v_sin, v_add, v_mul}


def preorder_visit(e: Expr, f: Callable[[Expr], Any]):
    # Call f on e
    yield f(e)

    # And recurse into Expr children
    if e.isLet:
        for eqn in e.eqns:
            yield from preorder_visit(eqn.val, f)
        yield from preorder_visit(e.body, f)

    if e.isLambda:
        yield from preorder_visit(e.body, f)

    if e.isCall:
        yield from preorder_visit(e.f, f)
        for arg in e.args:
            yield from preorder_visit(arg, f)

def test_visit():
    e = _make_e()
    preorder_visit(e, lambda x: print(type(x)))


### Global functions
g_tuple = Var("g_tuple")
g_identity = Var("g_identity")


def mkTuple(es: List[Expr]) -> Expr:
    if len(es) == 1:
        return es[0]
    else:
        return Call(g_tuple, es)


########################################################################################
#
#
#     ,ad8888ba,                      88                    88
#    d8"'    `"8b               ,d    ""                    ""
#   d8'        `8b              88
#   88          88 8b,dPPYba, MM88MMM 88 88,dPYba,,adPYba,  88 888888888  ,adPPYba,
#   88          88 88P'    "8a  88    88 88P'   "88"    "8a 88      a8P" a8P_____88
#   Y8,        ,8P 88       d8  88    88 88      88      88 88   ,d8P'   8PP"""""""
#    Y8a.    .a8P  88b,   ,a8"  88,   88 88      88      88 88 ,d8"      "8b,   ,aa
#     `"Y8888Y"'   88`YbbdP"'   "Y888 88 88      88      88 88 888888888  `"Ybbd8"'
#                  88
#                  88
########################################################################################


def signature(e):
    """
    A "signature" is a loose hash.
    Optimization might change the expression a lot, so the signature
    should really be the same for two experessions which compute the
    same quantities, which we know is uncomputable.

    This just computes the freevars of the expression, which will generally
    be the list of external functions called. For trivial optimizations this
    may be fine, but e.g. DCE or user-level rewrites might result in fewer
    functions being called...
    """
    fvs = {v.name for v in freevars(e)}
    return set.union(fvs, {g_tuple.name, g_identity.name})


def optimize(e: Expr) -> Expr:
    def run(transformation_name, ex, transformation=None):
        print(f"Running {transformation_name}")
        if not transformation:
            if transformation_name.startswith("t-"):
                transformation = globals()[transformation_name[2:]]
                transformation = transform_postorder(transformation)
            else:
                transformation = globals()[transformation_name]

        new_ex = transformation(ex)

        ex2py(f"{run.count:02d}-{transformation_name}", new_ex)
        run.count += 1

        osig = signature(ex)
        sig = signature(new_ex)
        if sig != osig:
            assert False
        return new_ex

    run.count = 1

    print(f"Starting optimization, {str(signature(e))[:80]}")
    ex2py(f"00-before-optimization", e)
    e = run("uniquify_names", e)
    for t in (
        elide_empty_lhs,
        inline_call_of_lambda,
        inline_trivial_letbody,
        inline_lambda_of_call,
        inline_lambda_of_let_of_call,
        identify_identities,
        eliminate_identities,
    ):
        e = run("t-" + t.__name__, e, transform_postorder(t))

    e = run("to_anf", e)
    e = run("t-detuple_tuple_assignments", e)
    e = run("inline_trivial_assignments", e)

    for t in (eliminate_identities,):
        e = run("t-" + t.__name__, e, transform_postorder(t))

    e = run("inline_trivial_assignments", e)

    ex2py(f"99-after-optimization", e)

    return e


def inline_call_of_lambda(e: Expr) -> Expr:
    # call(lambda l_args: body, args)
    #  ->  let l_args = args in body
    # Name clashes will happen unless all bound var names were uniquified
    if e.isCall and e.f.isLambda:
        return Let([Eqn(e.f.args, mkTuple(e.args))], e.f.body)


def inline_lambda_of_let_of_call(e: Expr) -> Expr:
    # lambda args: let eqns in call(f, args)
    #  ->  let eqns in f
    # Name clashes will happen unless all bound var names were uniquified
    if e.isLambda and e.body.isLet and e.body.body.isCall:
        if e.body.body.args == e.args:
            return Let(e.body.eqns, e.body.body.f)


def inline_trivial_letbody(e: Expr) -> Expr:
    # let var = val in var -> val
    if e.isLet and len(e.eqns) == 1 and len(e.eqns[0].vars) == 1:
        if e.eqns[0].vars == [e.body]:
            return e.eqns[0].val
    if e.isLet and len(e.eqns) == 0:
        return e.body


def inline_lambda_of_call(e: Expr) -> Expr:
    # Lambda(args, Call(f, args)) -> f
    if e.isLambda:
        if e.body.isCall:
            if all_equal(e.body.args, e.args):
                return e.body.f


def detuple_tuple_assignments(e: Expr) -> Expr:
    # Let([Eqn([a, b, c], Call(tuple, [aprime, bprime, cprime]))],
    #       body) ->
    #  Let([Eqn(a, aprime),
    #       Eqn(b, bprime),
    #       Eqn(c, cprime)],
    #           body)))
    if not e.isLet:
        return e

    def detuple_eqn(eqn):
        vars = eqn.vars
        val = eqn.val
        if len(vars) > 1 and val.isCall and val.f == g_tuple:
            for var, arg in zip(vars, val.args):
                yield Eqn([var], arg)
        else:
            yield Eqn(vars, val)

    new_eqns = list(chain(*map(detuple_eqn, e.eqns)))
    return Let(new_eqns, e.body)

# from collections import Counter

# def hash_expr(e):
#     if e.isConst:
#         return hash(e.val)

#     if e.isVar:
#         return hash(e.name)

#     if e.isCall:
#         myhash = hash_expr(e.f,) + tuple(hash(arg) for arg in e.args)

#     if e.isLet:
#         return (e.eqns,) + tuple(hash(e.body))

#     if e.isLambda:
#         return (e.args,) + hash(e.body)

# def cse(e):
#     """
#     Common Subexpression Elimination
#     """
#     ## Count occurrences


# def test_cse():
#     def foo(x):
#         a = str(x)
#         b = str(x)
#         return a + b

#     e = expr_for(foo)
#     cse(e)


def to_anf(e):
    e = uniquify_names(e)
    assignments = []
    expr = to_anf_aux(e, assignments)
    return Let(assignments, expr)


def to_anf_aux(e, assignments):
    if e.isConst or e.isVar:
        return e

    if e.isLet:
        for eqn in e.eqns:
            new_val = to_anf_aux(eqn.val, assignments)
            assignments += [Eqn(eqn.vars, new_val)]
        abody = to_anf_aux(e.body, assignments)
        return abody

    if e.isLambda:
        return Lambda(e.args, to_anf(e.body))

    if e.isCall:
        new_f = to_anf_aux(e.f, assignments)
        new_args = [to_anf_aux(arg, assignments) for arg in e.args]
        return Call(new_f, new_args)

    assert False


def inline_trivial_assignments(e):
    return inline_trivial_assignments_aux(e, {})


@beartype
def inline_trivial_assignments_aux(e: Expr, translations: Dict[Var, Expr]):
    # let a = b in body -> body[a->b] if b is trivial

    recurse = inline_trivial_assignments_aux

    def is_trivial(e: Expr):
        if e.isVar:
            return True

        if e.isConst and isinstance(e.val, (float, str, int)):
            return True

        return False

    if e.isConst:
        return e

    if e.isVar:
        return translations.get(e, e)

    if e.isLet:

        # let vars = val in body
        new_eqns = []

        inner_translations = {**translations}

        for eqn in e.eqns:
            vars = eqn.vars
            new_val = recurse(eqn.val, inner_translations)
            if len(vars) == 1 and is_trivial(new_val):
                # Add val to translations, and don't add it to our new eqns
                inner_translations[vars[0]] = new_val
            else:
                # keep val, and delete newly bound vars from translations
                new_eqns += [Eqn(vars, new_val)]
                for var in vars:
                    if var in inner_translations:
                        del inner_translations[var]

        new_body = recurse(e.body, inner_translations)

        if len(new_eqns):
            return Let(new_eqns, new_body)
        else:
            return new_body

    if e.isLambda:
        argset = set(e.args)
        new_translations = {
            var: val for (var, val) in translations.items() if var not in argset
        }
        assert all(v not in new_translations for v in e.args)
        new_body = recurse(e.body, new_translations)
        return Lambda(e.args, new_body)

    if e.isCall:
        new_f = recurse(e.f, translations)
        new_args = [recurse(arg, translations) for arg in e.args]
        return Call(new_f, new_args)

    assert False


def test_inline_trivial_assignments():
    a, b, c, d = mkvars("a,b,c,d")
    e = Let(
        [
            Eqn([a], b),
        ],
        Call(
            a,
            [
                Let(
                    [
                        Eqn([a], c),
                    ],
                    Call(a, [a, b, c]),
                )
            ],
        ),
    )
    out = inline_trivial_assignments(e)
    pprint(out)
    expect = Call(b, [Call(c, [c, b, c])])
    assert out == expect


def let_to_lambda(e: Expr) -> Expr:
    """
    let x1 = val1, x2 = val2 in body
    ->
    call(lambda x1,x2: body, (val1, val2))
    """
    if e.isLet:
        args = []
        vals = []
        for eqn in e.eqns:
            assert len(eqn.vars) == 1, "Use detuple_lets before let_to_lambda"
            args += eqn.vars

            val = transform_postorder(let_to_lambda, eqn.val)
            vals += [val]

        body = transform_postorder(let_to_lambda, e.body)
        return Call(Lambda(args, body), vals)


def test_let_to_lambda():
    e = from_ast(ast.parse("let x = 2, y = 3 in x + y"))
    l = transform_postorder(let_to_lambda, e)
    assert l.isCall and l.f.isLambda and len(l.args) == 2


def elide_empty_lhs(e: Expr) -> Expr:
    """
    let [] = val1, x2 = val2 in body
    ->
    let x2 = val2 in body
    """
    if e.isLet:
        return Let(
            [
                Eqn(eqn.vars, transform_postorder(elide_empty_lhs, eqn.val))
                for eqn in e.eqns
                if len(eqn.vars) > 0
            ],
            transform_postorder(elide_empty_lhs, e.body),
        )


def ex2py(name, ex):
    fvs = list(v.name for v in freevars(ex))

    filename = "tmp/ex-" + name + ".txt"
    with open(filename, "w") as f:
        print("Freevars:", *fvs, file=f)
        pprint(ex, stream=f)

    filename = "tmp/py-" + name + ".py"
    with open(filename, "w") as f:
        print("#Freevars:", *fvs, file=f)
        print(astunparse.unparse(to_ast(ex, "ret")), file=f)


def identify_identities(e: Expr) -> Expr:
    """
    lambda x: x -> g_identity
    """
    if e.isLambda and [e.body] == e.args:
        return g_identity


def eliminate_identities(e: Expr) -> Expr:
    """
    g_identity(args) -> args
    """
    if e.isCall and e.f == g_identity:
        return mkTuple(e.args)


def test_let_to_lambda():
    e = _make_e()
    l = transform_postorder(let_to_lambda, e)

    def check(e):
        assert not e.isLet

    preorder_visit(l, check)


def uniquify_names(e: Expr, reset_ids: bool = False) -> Expr:
    if reset_ids:
        reset_new_name_ids()
    # To start with, add freevars to scope, with their own names
    translations = {v.name: v.name for v in freevars(e)}
    return uniquify_names_aux(e, translations)


def uniquify_names_aux(e: Expr, translations) -> Expr:
    # foo(let x = 2 in f(x), x)
    #  -> foo(let x_new = 2 in f(x_new), x)
    # i.e. a let which binds a var which is already in scope will
    # need to rename that var.
    # Thus there will be a translation table of oldnames to newnames
    # When entering a let or lambda, if a var is already in the table
    # it will need further renaming, so make a new entry for the body
    # Vars that come newly in scope should be translated to themselves
    if e.isConst:
        return e

    if e.isVar:
        assert e.name in translations
        return Var(translations[e.name])

    if e.isCall:
        return Call(
            uniquify_names_aux(e.f, translations),
            [uniquify_names_aux(arg, translations) for arg in e.args],
        )

    if e.isLambda:
        vars = e.args

        new_translations = {**translations}  # TODO-Q
        new_vars = []
        for varname in [var.name for var in vars]:
            # This var has just come in scope.
            # If its name is already in translations, it is clashing,
            # so in the body of this lambda, it will need a new new name
            if varname in new_translations:
                newname = "t_" + get_new_name()
            else:
                newname = varname
            new_translations[varname] = newname
            new_vars.append(Var(newname))

        new_body = uniquify_names_aux(e.body, new_translations)
        return Lambda(new_vars, new_body)

    if e.isLet:
        new_translations = {**translations}
        new_eqns = []
        for eqn in e.eqns:
            # First recurse into val using new_translations so far
            newval = uniquify_names_aux(eqn.val, new_translations)
            # Now add the new vars to the translation table
            new_vars = []
            for var in eqn.vars:
                # This var has just come in scope.
                # If its name is already in translations, it is clashing,
                # so in the body of this let, it will need a new name
                if var.name in new_translations:
                    newname = "t_" + get_new_name()
                else:
                    newname = var.name
                new_translations[var.name] = newname
                new_vars += [Var(newname)]
            new_eqns += [Eqn(new_vars, newval)]

        new_body = uniquify_names_aux(e.body, new_translations)
        return Let(new_eqns, new_body)

    assert False  # unreachable


def test_uniquify_names():
    a, b, c, d = mkvars("a,b,c,d")

    e = Let(
        [
            Eqn([a], a),
        ],
        Let(
            [
                Eqn([a], b),
            ],
            Call(b, [a, b]),
        ),
    )
    pprint(e)
    out = uniquify_names(e)
    pprint(out)
    assert out.eqns[0].vars[0] != a
    assert out.eqns[0].val == a
    assert out.body.eqns[0].vars[0] != a
    assert out.body.eqns[0].vars[0] != out.eqns[0].vars[0]


########################################################################################
#
#
#   88888888888                    88
#   88                             88
#   88                             88
#   88aaaaa 8b       d8 ,adPPYYba, 88
#   88""""" `8b     d8' ""     `Y8 88
#   88       `8b   d8'  ,adPPPPP88 88
#   88        `8b,d8'   88,    ,88 88
#   88888888888 "8"     `"8bbdP"Y8 88
#
#
########################################################################################

def run_eval(e: Expr, bindings: Dict[str, Any]) -> Any:
    new_bindings = {Var(key): val for key, val in bindings.items()}
    # Check all the freevars have been bound
    unbound_vars = [v.name for v in set(freevars(e)) - set(new_bindings)]
    if len(unbound_vars) > 0:
        raise ValueError(f"Unbound variable(s): {','.join(unbound_vars)}")

    return run_eval_aux(e, new_bindings)


@beartype
def run_eval_aux(e: Expr, bindings: Dict[Var, Any]) -> Any:
    recurse = run_eval_aux

    if e.isConst:
        return e.val

    if e.isVar:
        if e not in bindings:
            raise ValueError(f"Unbound variable {e.name}")
        return bindings[e]

    if e.isCall:
        new_f = recurse(e.f, bindings)
        new_args = [recurse(arg, bindings) for arg in e.args]
        assert isinstance(new_f, Callable)
        return new_f(*new_args)

    if e.isLet:
        # let
        #   vars1 = val1 [fvs x]
        #   vars2 = val2 [fvs x vars1]
        # in
        #   body
        argset = set()
        new_bindings = {**bindings}
        for eqn in e.eqns:
            tupval = recurse(eqn.val, new_bindings)

            if len(eqn.vars) > 1:
                assert isinstance(tupval, tuple)
            else:
                tupval = (tupval,)

            for var, val in zip(eqn.vars, tupval):
                new_bindings[var] = val

        return recurse(e.body, new_bindings)

    if e.isLambda:

        def runLambda(e, vals, bindings):
            argset = {v.name for v in e.args}
            new_bindings = {
                name: val for (name, val) in bindings.items() if name not in argset
            }
            for var, val in zip(e.args, vals):
                new_bindings[var] = val

            return recurse(e.body, new_bindings)

        return lambda *vals: runLambda(e, vals, bindings)

    assert False


def test_eval():

    a, b, c = mkvars("a,b,c")
    f_tuple, f_add = mkvars("tuple, add")
    f_defs = {
        "add": operator.add,
        "tuple": lambda *args: tuple(args),
        "getattr": getattr,
    }

    v = run_eval(Call(f_add, [Const(2), Const(3)]), f_defs)
    assert v == 5

    v = run_eval(
        Let(
            [
                Eqn([a, b], Call(f_tuple, [Const(2), Const(3)])),
            ],
            Call(f_add, [a, b]),
        ),
        f_defs,
    )
    assert v == 5


########################################################################################
#
#
#   888888888888                  db        ad88888ba 888888888888
#        88                      d88b      d8"     "8b     88
#        88                     d8'`8b     Y8,             88
#        88  ,adPPYba,         d8'  `8b    `Y8aaaaa,       88
#        88 a8"     "8a       d8YaaaaY8b     `"""""8b,     88
#        88 8b       d8      d8""""""""8b          `8b     88
#        88 "8a,   ,a8"     d8'        `8b Y8a     a8P     88
#        88  `"YbbdP"'     d8'          `8b "Y88888P"      88
#
#
########################################################################################

import re


def prettify_repr(s):
    """
    Perform various transformations to make the repr of a value more readable
       builtins.() -> ()
       numpy.dtypes.dtype('int32') -> np.int32
    """
    s = re.sub(r"^builtins\.", "", s)
    s = re.sub(r"^numpy\.dtypes\.dtype\('(\w+)'\)", r"np.\1", s)
    s = re.sub(r"^numpy\.(array.*), dtype=(\w+)", r"np.\1, dtype=np.\2", s)
    s = re.sub(r"^numpy\.", r"np.", s)
    s = re.sub(r"^jax._src.lax.slicing", "jax.lax", s)
    return s


_ast_cmpops = {
    "__eq__": ast.Eq,
    "__ne__": ast.NotEq,
    "__lt__": ast.Lt,
    "__le__": ast.LtE,
    "__gt__": ast.Gt,
    "__ge__": ast.GtE,
}

_ast_binops = {
    "__add__": ast.Add,
    "__sub__": ast.Sub,
    "__mul__": ast.Mult,
    "__floordiv__": ast.Div,
    "__truediv__": ast.Div,
    "__floordiv__": ast.FloorDiv,
    "__mod__": ast.Mod,
    "__pow__": ast.Pow,
    "__lshift__": ast.LShift,
    "__rshift__": ast.RShift,
    "__or__": ast.BitOr,
    "__xor__": ast.BitXor,
    "__and__": ast.BitAnd,
    "__matmul__": ast.MatMult,
}

_ast_unaryops = {
    "__neg__": ast.USub,
    "__pos__": ast.UAdd,
    "__not__": ast.Not,
    "__inv__": ast.Invert,
}

_ast_ops = _ast_cmpops | _ast_binops | _ast_unaryops

_ast_op_to_operator = {v: k for k, v in _ast_ops.items()}


lambda_to_name = {}

def to_ast(e, name):
    e = uniquify_names(e)
    global lambda_to_name
    lambda_to_name = {}

    assignments = []
    expr = to_ast_aux(e, assignments)
    assignments += [ast.Assign([ast.Name(name, ast.Store())], expr)]
    a = ast.Module(body=assignments, type_ignores=[])
    # Do one pass to inline '**dict'
    a = _RewriteDictCallToSplat().visit(a)
    ast.fix_missing_locations(a)
    return a


class _RewriteDictCallToSplat(ast.NodeTransformer):

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = []
        keywords = []
        for arg in node.args:
            if (
                ## Is it a splat?
                (isinstance(arg, ast.keyword) and arg.arg is None)
                and
                ## It's a splat, is it a call of dict?
                (
                    isinstance(arg.value, ast.Call)
                    and isinstance(arg.value.func, ast.Name)
                    and arg.value.func.id == "dict"
                )
            ):
                # Add the keywords to the keywords list, nothing to args
                keywords += [self.visit(kw) for kw in arg.value.keywords]
            else:
                # Just add to args
                args += [self.visit(arg)]

        for k in node.keywords:
            arg = k.arg
            if arg is None:
                # Splatting
                pass
            value = self.visit(k.value)
            keywords += [ast.keyword(arg=arg, value=value)]

        return ast.Call(func, args, keywords)


@beartype
def to_ast_args(vars: List[Var]) -> ast.arguments:
    aargs = [ast.arg(v.name, annotation=None) for v in vars]
    return ast.arguments(
        args=aargs,
        vararg=None,
        kwarg=None,
        defaults=[],
        posonlyargs=[],
        kwonlyargs=[],
        kw_defaults=None,
    )


@beartype
def to_ast_constant(val):
    if isinstance(val, enum.Enum):
        # repr doesn't work for enum...
        rep = str(val)
    else:
        rep = repr(val)
    rep = type(val).__module__ + "." + rep
    rep = prettify_repr(rep)

    module = ast.parse(rep)
    ast_expr: ast.Expr = module.body[0]
    assert isinstance(ast_expr, ast.Expr)
    return ast_expr.value


def to_ast_aux(e, assignments, binders=None):
    if e.isConst:
        return to_ast_constant(e.val)

    if e.isVar:
        return ast.Name(e.name, ast.Load())

    if e.isLet:
        for eqn in e.eqns:
            avars = [ast.Name(var.name, ast.Store()) for var in eqn.vars]
            if len(avars) > 1:
                avars = [ast.Tuple(avars, ast.Store())]

            if aval := to_ast_aux(eqn.val, assignments, binders=eqn.vars):
                assignments += [ast.Assign(targets=avars, value=aval)]

        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        assignments += inner_assignments
        return abody

    if e.isLambda:
        # If I know whom I am bound to, then use that name for the FunctionDef
        if binders is None:
            v = Var("f" + get_new_name())
        else:
            assert len(binders) == 1
            v = binders[0]

        # Recurse, generating inner assignments
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        inner_assignments += [ast.Return(abody)]

        # Make a FunctionDef
        assignments += [
            ast.FunctionDef(
                name=v.name,
                args=to_ast_args(e.args),
                body=inner_assignments,
                decorator_list=[],
                lineno=0,
            )
        ]

        # And return a reference to the function's name
        if binders is None:
            return ast.Name(v.name, ast.Load())
        else:
            return None

    if e.isCall:
        # Special case: **_pairs_to_dict
        if e.f.isVar and e.f.name == "**_pairs_to_dict":
            # convert to dict call
            keywords = [
                ast.keyword(arg=k.val, value=to_ast_aux(value, assignments))
                for k, value in zip(e.args[::2], e.args[1::2])
            ]
            dictval = ast.Call(
                func=ast.Name("dict", ast.Load()), args=[], keywords=keywords
            )
            # Splatting is a keyword with arg=None
            return ast.keyword(value=dictval)

        args = [to_ast_aux(arg, assignments) for arg in e.args]
        f = to_ast_aux(e.f, assignments)

        if isinstance(f, ast.Name) and f.id.startswith("operator."):
            # Special case: operator.*
            op = f.id[9:]
            if op in _ast_cmpops:
                ops = [_ast_cmpops[op]()]
                return ast.Compare(left=args[0], ops=ops, comparators=[args[1]])

            if op in _ast_binops:
                return ast.BinOp(left=args[0], op=_ast_binops[op](), right=args[1])

            if op in _ast_unaryops:
                return ast.UnaryOp(op=_ast_unaryops[op](), operand=args[0])

            assert False, f"Unknown operator {op}"
        elif isinstance(f, ast.Name) and f.id == "getattr":
            # Special case: getattr
            assert len(args) == 2
            return ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load())
        else:
            # Normal case
            return ast.Call(func=f, args=args, keywords=[])

    assert False


def test_ast():
    a, b, c = mkvars("a,b, c")
    e = Let(
        [
            Eqn([a, b], Const(123)),
            Eqn([c], Call(Var("add"), [a, b])),
        ],
        Const(234),
    )
    pprint(e)
    a = to_ast(e, "e")
    print(astunparse.unparse(a))

    e = _make_e()
    a = to_ast(e, "e")
    print(astunparse.unparse(a))

    # code = compile(a, 'bar', 'exec')
    # exec(code)


########################################################################################
#
#
#   88888888888                                                 db        ad88888ba 888888888888
#   88                                                         d88b      d8"     "8b     88
#   88                                                        d8'`8b     Y8,             88
#   88aaaaa 8b,dPPYba,  ,adPPYba,  88,dPYba,,adPYba,         d8'  `8b    `Y8aaaaa,       88
#   88""""" 88P'   "Y8 a8"     "8a 88P'   "88"    "8a       d8YaaaaY8b     `"""""8b,     88
#   88      88         8b       d8 88      88      88      d8""""""""8b          `8b     88
#   88      88         "8a,   ,a8" 88      88      88     d8'        `8b Y8a     a8P     88
#   88      88          `"YbbdP"'  88      88      88    d8'          `8b "Y88888P"      88
#
#
########################################################################################


def from_ast(a: ast.AST):
    recurse = from_ast

    if isinstance(a, ast.Module):
        assert len(a.body) == 1
        return recurse(a.body[0])

    if isinstance(a, ast.arguments):
        assert not a.vararg
        assert not a.kwonlyargs
        assert not a.kw_defaults
        assert not a.kwarg
        assert not a.defaults
        if sys.version_info >= (3, 8):
            assert not a.posonlyargs

        return [Var(arg.arg) for arg in a.args]

    if isinstance(a, ast.FunctionDef):
        name = Var(a.name)
        args = recurse(a.args)

        eqns = []
        for stmt in a.body[:-1]:
            assert isinstance(stmt, ast.Assign)
            assert len(stmt.targets) == 1
            s_vars = stmt.targets[0]
            if isinstance(s_vars, ast.Tuple):
                vars = [Var(var.id) for var in s_vars.elts]
            else:
                vars = [Var(s_vars.id)]
            val = recurse(stmt.value)
            eqns += [Eqn(vars, val)]

        assert isinstance(a.body[-1], ast.Return)
        retval = recurse(a.body[-1].value)

        body = Let(eqns, retval)
        lam = Lambda(args, body)

        return Let([Eqn([name], lam)], name)

    if isinstance(a, ast.Lambda):
        return Lambda(recurse(a.args), recurse(a.body))

    if isinstance(a, ast.Constant):
        return Const(a.value)

    # Nodes which encode to Var
    if isinstance(a, ast.Name):
        return Var(a.id)

    # if isinstance(a, ast.operator):
    #     return Var("ast." + type(a).__name__)

    # Nodes which encode to (Var or Call)
    if isinstance(a, ast.Attribute):
        # if isinstance(a.value, ast.Name):
        #     # just make it a name
        #     return Var(val.name + '.' + a.attr)
        # else:
        #     # make it a getattr call.
        val = recurse(a.value)
        return Call(Var("getattr"), [val, Const(a.attr)])

    # Nodes which encode to Call
    if isinstance(a, ast.Call):
        func = recurse(a.func)
        args = [recurse(arg) for arg in a.args]
        assert len(a.keywords) == 0
        return Call(func, args)

    if isinstance(a, ast.Tuple):
        return Call(g_tuple, [recurse(e) for e in a.elts])

    if isinstance(a, ast.BinOp):
        operator_op = _ast_op_to_operator[type(a.op)]  # check that op is in the dict
        op = Var("operator." + operator_op)
        return Call(op, [recurse(a.left), recurse(a.right)])

    if isinstance(a, ast.Subscript):
        return Call(Var("ast.Subscript"), [recurse(a.value), recurse(a.slice)])

    if isinstance(a, ast.Index):
        return recurse(a.value)

    if isinstance(a, (ast.ExtSlice, ast.Slice)):
        # Plan is to overload "ast.Subscript" implementation to handle them all.
        assert False

    # Fallthru
    assert False, f"TODO:{type(a)}"


def test_ast_to_expr():
    import jax.numpy as jnp

    def foo(f):
        a, b = 123, 2
        c = f * b
        d = jnp.sin(c)
        return (lambda x: d + x)(c)

    expected = foo(5)

    e = expr_for(foo)
    print(expr_to_python_code(e, "foo"))

    got = eval_expr(
        e,
        [5],
        {
            "g_tuple": lambda *args: tuple(args),
            "getattr": getattr,
            "jnp": jnp,
        },
    )
    assert expected == got


def test_ast_to_expr2():
    import jax

    def foo(p, x):
        x = x @ (p * x.T)
        return (x + x[3]).std()

    prng = jax.random.PRNGKey(42)
    args = (2.2, jax.random.normal(prng, (2, 5)))

    expected = foo(*args)
    print(expected)

    e_foo = expr_for(foo)
    print(expr_to_python_code(e_foo, "foo"))

    got = eval_expr(
        e_foo,
        args,
        {
            "ast.Subscript": lambda a, slice: a[slice],
            "ast.MatMult": operator.matmul,
            "ast.Mult": operator.mul,
            "ast.Add": operator.add,
            "tuple": lambda *args: tuple(args),
            "getattr": getattr,
        },
    )


#### Misc


def expr_for(f: Callable) -> Expr:
    import inspect
    import textwrap

    a = ast.parse(textwrap.dedent(inspect.getsource(f)))
    return from_ast(a)


def expr_to_python_code(e: Expr, name: str) -> str:
    as_ast = to_ast(e, name)
    return astunparse.unparse(as_ast)


def bindings_for_operators():
    return {("operator." + op): operator.__dict__[op] for op in _ast_ops.keys()}


def eval_expr(e: Expr, args, bindings):
    bindings |= bindings_for_operators()
    return run_eval(Call(e, [Const(a) for a in args]), bindings)
