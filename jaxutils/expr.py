import sys
import enum
from dataclasses import dataclass
from beartype import beartype
from beartype.typing import List, Set, Any, Tuple, Dict, List, Callable, Optional
from pprint import pprint
from itertools import chain

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


######################################################################################
#
#
#
#    oooooooooooo
#    `888'     `8
#     888         oooo    ooo oo.ooooo.  oooo d8b
#     888oooo8     `88b..8P'   888' `88b `888""8P
#     888    "       Y888'     888   888  888
#     888       o  .o8"'88b    888   888  888
#    o888ooooood8 o88'   888o  888bod8P' d888b
#                              888
#                             o888o
#
#
######################################################################################


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
    import operator

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
    f(e)

    # And recurse into Expr children
    if e.isLet:
        for eqn in e.eqns:
            preorder_visit(eqn.val, f)
        preorder_visit(e.body, f)

    if e.isLambda:
        preorder_visit(e.body, f)

    if e.isCall:
        preorder_visit(e.f, f)
        for arg in e.args:
            preorder_visit(arg, f)


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


#####################################################################################
#
#   .oooooo.                  .    o8o                     o8o
#  d8P'  `Y8b               .o8    `"'                     `"'
# 888      888 oo.ooooo.  .o888oo oooo  ooo. .oo.  .oo.   oooo    oooooooo  .ooooo.
# 888      888  888' `88b   888   `888  `888P"Y88bP"Y88b  `888   d'""7d8P  d88' `88b
# 888      888  888   888   888    888   888   888   888   888     .d8P'   888ooo888
# `88b    d88'  888   888   888 .  888   888   888   888   888   .d8P'  .P 888    .o
#  `Y8bood8P'   888bod8P'   "888" o888o o888o o888o o888o o888o d8888888P  `Y8bod8P'
#               888
#              o888o
#
#####################################################################################


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
    filename = "tmp/ex-" + name + ".txt"
    with open(filename, "w") as f:
        pprint(ex, stream=f)

    filename = "tmp/py-" + name + ".py"
    with open(filename, "w") as f:
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


def optimize(e: Expr) -> Expr:
    def signature(e):
        """
        A "signature" is a loose hash.
        Optimization might change the expression a lot, so the signature
        should really be the same for two experessions which compute the
        same quantities, which we know is uncomputable.

        This just computes the freevars of the expression, which will essentially
        be the list of external functions called. For trivial optimizations this
        may be fine, but e.g. DCE or user-level rewrites might result in fewer
        functions being called...
        """
        fvs = {v.name for v in freevars(e)}
        return set.union(fvs, {g_tuple.name, g_identity.name})

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


######################################################################################
#
#    oooooooooooo                       oooo
#    `888'     `8                       `888
#     888         oooo    ooo  .oooo.    888
#     888oooo8     `88.  .8'  `P  )88b   888
#     888    "      `88..8'    .oP"888   888
#     888       o    `888'    d8(  888   888
#    o888ooooood8     `8'     `Y888""8o o888o
#
######################################################################################


def run_eval(e: Expr, bindings: Dict[str, Any]) -> Any:
    new_bindings = {Var(key): val for key, val in bindings.items()}
    # Check all the freevars have been bound
    for v in freevars(e):
        assert v in new_bindings

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
    import operator

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


######### To AST

######################################################################################
#
#          .o.        .oooooo..o ooooooooooooo
#        .888.      d8P'    `Y8 8'   888   `8
#        .8"888.     Y88bo.           888
#      .8' `888.     `"Y8888o.       888
#      .88ooo8888.        `"Y88b      888
#    .8'     `888.  oo     .d8P      888
#    o88o     o8888o 8""88888P'      o888o
#
#
######################################################################################


def to_ast_FunctionDef(name, args, body):
    return ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], lineno=0)


def to_ast(e, name):
    e = uniquify_names(e)
    assignments = []
    expr = to_ast_aux(e, assignments)
    assignments += [ast.Assign([ast.Name(name, ast.Store())], expr)]
    a = ast.Module(body=assignments, type_ignores=[])
    ast.fix_missing_locations(a)
    return a


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

            aval = to_ast_aux(eqn.val, assignments, binders=eqn.vars)
            # If it was a function definition, it will have emitted
            # def foo(...):
            # and returned Name(foo)
            # so we would be adding an assignment foo = foo
            elide = (
                isinstance(aval, ast.Name)
                and len(eqn.vars) == 1
                and aval.id == eqn.vars[0].name
            )
            if not elide:
                assign = ast.Assign(targets=avars, value=aval)
                assignments += [assign]

        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        assignments += inner_assignments
        return abody

    if e.isLambda:
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        inner_assignments += [ast.Return(abody)]
        if binders is None:
            v = Var("f" + get_new_name())
        else:
            assert len(binders) == 1
            v = binders[0]
        aargs = to_ast_args(e.args)
        fdef = to_ast_FunctionDef(v.name, aargs, inner_assignments)
        assignments += [fdef]
        return to_ast_aux(v, None)

    if e.isCall:
        f = to_ast_aux(e.f, assignments)
        args = [to_ast_aux(arg, assignments) for arg in e.args]
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


#### From AST


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

    if isinstance(a, ast.operator):
        return Var("ast." + type(a).__name__)

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
        return Call(recurse(a.op), [recurse(a.left), recurse(a.right)])

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

    import operator

    got = eval_expr(
        e,
        [5],
        {
            "ast.Mult": operator.mul,
            "ast.Add": operator.add,
            "jnp": jnp,
            "tuple": lambda *args: tuple(args),
            "getattr": getattr,
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
    import operator

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


def eval_expr(e: Expr, args, bindings):
    return run_eval(Call(e, [Const(a) for a in args]), bindings)


######################################################################################
#
#
#
#
#
#
#
#
#
######################################################################################
