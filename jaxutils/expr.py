from dataclasses import dataclass
from beartype import beartype
from beartype.typing import (
    List,
    Set,
    Any,
    Tuple,
    Dict,
    Sequence,
    List,
    Callable,
    Optional,
)
from itertools import chain
from more_itertools import one
from jaxutils.ast_utils import better_repr

warn = print


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


### New name factory

_expr_global_name_id = 0


def get_new_name():
    global _expr_global_name_id
    _expr_global_name_id += 1
    return f"{_expr_global_name_id:02x}"


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

    def __str__(self):
        return expr_format(self)


def exprclass(klass, **kwargs):
    """
    Decorator to simplify declaring expr subclasses.
    """

    dclass = dataclass(klass, frozen=True, **kwargs)

    # TODO: [Note 1]
    # Annotations are on all dataclasses:
    #  - an optional, argument to init
    #  - not on Eqns
    #  - not participate in hashing
    # We could unify the addition of "annot" fields to these dataclasses.
    # and automatically add the __hash__ method.
    #
    # Pros: DRY
    # Cons: possibly fragile, need to duplicate a lot of dataclasses.py

    # Commented out for now, but can be used later if needed.
    # # Expr hashes don't include annotations.
    # if "annot" in dclass.__dataclass_fields__:
    #     hash_fields = [f for f in dclass.__dataclass_fields__ if f != "annot"]
    #     dclass.__hash__ = lambda self: hash(
    #         tuple(getattr(self, f) for f in hash_fields)
    #     )

    return beartype(dclass)


@exprclass
class Const(Expr):
    val: Any

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash(Const, self.val)


@exprclass
class Var(Expr):
    name: str

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash((Var, self.name))

    def __eq__(self, b):
        # Var equality ignores annotations
        return isinstance(b, Var) and self.name == b.name


@dataclass(frozen=True)
class Eqn:
    vars: List[Var]
    val: Expr


@exprclass
class Let(Expr):
    eqns: List[Eqn]
    body: Expr

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash(Let, self.eqns, self.body)


# Lambdas are kinda singletons, this id is useful to keep track of them
LambdaId = str


@exprclass
class Lambda(Expr):
    args: List[Var]
    body: Expr
    id: LambdaId

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash(Lambda, self.args, self.body, self.id)


@exprclass
class Call(Expr):
    f: Expr
    args: List[Expr]

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash(Call, self.f, self.args)


########################################################################################
def mkvars(s: str) -> Tuple[Var]:
    """
    Little convenience method for making Vars, like sympy.symbols

    x,y,z = mkvars('x,y,z')
    """
    s = s.replace(" ", "")
    return [Var(s) for s in s.split(",")]


def isNone(x: Optional[Expr]) -> bool:
    return x is None or x.isConst and x.val is None


def _subscript(e: Expr, i: int) -> Call:
    return Call(Var("g_subscript"), [e, Const(i)])


def _zeros_like(e: Expr) -> Call:
    return Call(Var("g_zeros_like"), [e])


def _add(e1: Expr, e2: Expr) -> Call:
    return Call(Var("operator.__add__"), [e1, e2])


def transform_postorder(name: str, transformer: Callable[[Expr, Dict], Expr], *args):
    if len(args) == 0:
        # Make transform(name, transformer)(e) work like transform(name, transformer, e)
        return lambda *args: transform_postorder(name, transformer, *args)

    recurse = lambda e, bindings: transform_postorder(name, transformer, e, bindings)

    e, bindings = args

    # Recurse into children
    if e.isLet:
        new_eqns = []
        new_bindings = bindings.copy()
        for eqn in e.eqns:
            # Recurse into the value of the equation
            new_val = recurse(eqn.val, new_bindings)
            new_vars = list(recurse(var, new_bindings) for var in eqn.vars)
            # And add the equation with the new value
            new_eqns += [Eqn(new_vars, new_val)]
            # Add the new value to the bindings
            if len(new_vars) == 1:
                new_bindings[one(new_vars).name] = new_val
            else:
                for i, var in enumerate(new_vars):
                    new_bindings[var.name] = _subscript(new_val, i)

        new_body = recurse(e.body, new_bindings)
        e = Let(new_eqns, new_body, annot=new_body.annot)

    if e.isLambda:
        new_bindings = bindings.copy()
        for arg in e.args:
            new_bindings[arg.name] = None
        new_body = recurse(e.body, new_bindings)
        new_args = list(recurse(v, new_bindings) for v in e.args)
        e = Lambda(new_args, new_body, e.id + "/" + name, annot=new_body.annot)

    if e.isCall:
        new_f = recurse(e.f, bindings)
        new_args = [recurse(arg, bindings) for arg in e.args]
        e = Call(new_f, new_args, annot=e.annot)

    # And pass self to the transformer, with updated children
    return transformer(e, bindings) or e


def freevars(e: Expr) -> Set[Var]:
    return freevars_aux(e, set())


@beartype
def freevars_aux(e: Expr, bound_vars: Set[Var]) -> Set[Var]:
    if e.isConst:
        return set()

    if e.isVar:
        if e in bound_vars:
            return set()
        else:
            return {e}

    if e.isLet:
        let_bound_vars = set() | bound_vars
        fvs = set()
        for eqn in e.eqns:
            fv_val = freevars_aux(eqn.val, let_bound_vars)
            fvs |= fv_val
            let_bound_vars |= set(eqn.vars)
        fv_body = freevars_aux(e.body, let_bound_vars)
        return fvs | fv_body

    if e.isLambda:
        inner_bound_vars = set(e.args) | bound_vars
        return freevars_aux(e.body, inner_bound_vars)

    if e.isCall:
        return set.union(
            freevars_aux(e.f, bound_vars),
            *(freevars_aux(arg, bound_vars) for arg in e.args),
        )

    assert False


def preorder_visit(e: Expr, f: Callable[[Expr, Dict], Any], bindings: Dict[str, Any]):
    # Call f on e
    yield f(e, bindings)

    # And recurse into Expr children
    if e.isLet:
        inner_bindings = bindings.copy()
        for eqn in e.eqns:
            yield from preorder_visit(eqn.val, f, inner_bindings)
            for var in eqn.vars:
                inner_bindings[var.name] = eqn.val
        yield from preorder_visit(e.body, f, inner_bindings)

    if e.isLambda:
        inner_bindings = bindings.copy()
        for arg in e.args:
            inner_bindings[arg.name] = None
        yield from preorder_visit(e.body, f, inner_bindings)

    if e.isCall:
        yield from preorder_visit(e.f, f, bindings)
        for arg in e.args:
            yield from preorder_visit(arg, f, bindings)


### Global functions


def is_global_function_name(name):
    """
    Check if a variable is a global function name
    """
    if name.startswith("g_"):
        return True
    if name.startswith("operator."):
        return True
    if name in ("getattr", "**g_pairs_to_dict"):
        return True

    return False


def mkTuple(es: List[Expr]) -> Expr:
    if len(es) == 1:
        return es[0]
    else:
        return Call(Var("g_tuple"), es)


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
    return set.union(fvs, {"g_tuple", "g_identity"})


def optimize(e: Expr) -> Expr:
    from jaxutils.expr_ast import ex2py

    def run(transformation_name, ex, transformation=None):
        print(f"Running {transformation_name}")
        if not transformation:
            if transformation_name.startswith("t-"):
                transformation = globals()[transformation_name[2:]]
                transformation = transform_postorder(
                    transformation_name, transformation
                )
            else:
                transformation = globals()[transformation_name]

        new_ex = (
            transformation(ex, {})
            if transformation != uniquify_names
            else uniquify_names(ex)
        )

        ex2py(f"{run.count:02d}-{transformation_name}", new_ex)
        run.count += 1

        sig = signature(ex)
        new_sig = signature(new_ex)
        # Set of freevars might shrink, but not grow
        if new_sig > sig:
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
        e = run("t-" + t.__name__, e, transform_postorder(t.__name__, t))

    e = run("to_anf", e)
    e = run("t-detuple_tuple_assignments", e)
    e = run("inline_trivial_assignments", e)

    for t in (eliminate_identities,):
        e = run("t-" + t.__name__, e, transform_postorder(t.__name__, t))

    e = run("inline_trivial_assignments", e)

    ex2py(f"99-after-optimization", e)

    return e


def inline_call_of_lambda(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # call(lambda l_args: body, args)
    #  ->  let l_args = args in body
    # Name clashes will happen unless all bound var names were uniquified
    if e.isCall and e.f.isLambda:
        return Let([Eqn(e.f.args, mkTuple(e.args))], e.f.body)


def inline_lambda_of_let_of_call(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # lambda args: let eqns in call(f, args)
    #  ->  let eqns in f
    # Name clashes will happen unless all bound var names were uniquified
    if e.isLambda and e.body.isLet and e.body.body.isCall:
        if e.body.body.args == e.args:
            return Let(e.body.eqns, e.body.body.f)


def inline_trivial_letbody(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # let var = val in var -> val
    if e.isLet and len(e.eqns) == 1 and len(e.eqns[0].vars) == 1:
        if e.eqns[0].vars == [e.body]:
            return e.eqns[0].val
    if e.isLet and len(e.eqns) == 0:
        return e.body


def inline_lambda_of_call(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # Lambda(args, Call(f, args)) -> f
    if e.isLambda:
        if e.body.isCall:
            if e.body.args == e.args:
                return e.body.f


from collections import defaultdict


def compute_variable_use_counts(e: Expr) -> Expr:

    var_to_count: Dict[str, int] = defaultdict(lambda: 0)

    def doit(e: Expr) -> Expr:
        if e.isConst:
            pass

        elif e.isVar:
            var_to_count[e.name] += 1

        elif e.isLet:
            for eqn in e.eqns:
                assert all(var.name not in var_to_count for var in eqn.vars)
            for eqn in e.eqns:
                doit(eqn.val)
            doit(e.body)

        elif e.isLambda:
            for arg in e.args:
                assert arg.name not in var_to_count
            doit(e.body)

        elif e.isCall:
            doit(e.f)
            for arg in e.args:
                doit(arg)
        else:
            assert False

    doit(e)

    return var_to_count


def inline_single_usages(e: Expr) -> Expr:
    # let
    #   v1 = e1
    #   v2 = e2
    # in
    #   body
    var_to_count = compute_variable_use_counts(e)

    def recurse(e, var_to_val):
        if e.isConst:
            return e

        if e.isVar:
            if var_to_count[e.name] == 1 and e.name in var_to_val:
                return var_to_val[e.name]
            else:
                return e

        if e.isLet:
            inner_var_to_val = {**var_to_val}
            eqns = []
            for eqn in e.eqns:
                val = recurse(eqn.val, inner_var_to_val)
                inner_var_to_val[one(eqn.vars).name] = val
                eqns += [Eqn(eqn.vars, val)]

            body = recurse(e.body, inner_var_to_val)
            return Let(eqns, body)

        if e.isCall:
            f = recurse(e.f, var_to_val)
            args = [recurse(a, var_to_val) for a in e.args]
            return Call(f, args)

        if e.isLambda:
            inner_var_to_val = {**var_to_val}
            for arg in e.args:
                inner_var_to_val[arg.name] = arg
            body = recurse(e.body, inner_var_to_val)
            return Lambda(e.args, body, e.id + "/isu")

        assert False

    return recurse(e, {})


def dce(e: Expr) -> Expr:
    """
    Dead Code Elimination
    Remove bindings to unused variables.
    """

    def recurse(e, var_to_count):
        if e.isConst:
            return e

        if e.isVar:
            var_to_count[e.name] += 1
            return e

        if e.isLet:
            local_var_to_count = var_to_count.copy()
            new_vals = []
            for eqn in e.eqns:
                val = recurse(eqn.val, local_var_to_count)
                for var in eqn.vars:
                    local_var_to_count[var.name] = 0
                new_vals += [val]
            body = recurse(e.body, local_var_to_count)

            new_eqns = []
            for eqn, val in zip(e.eqns, new_vals):
                any_used = any(local_var_to_count[var.name] > 0 for var in eqn.vars)
                if any_used:
                    # Replace any unused with "_"
                    new_vars = [
                        var if local_var_to_count[var.name] > 0 else Var("_")
                        for var in eqn.vars
                    ]
                    new_eqns += [Eqn(new_vars, val)]

            if not new_eqns:
                # If no equations left, just return the body
                return body
            else:
                return Let(new_eqns, body)

        if e.isCall:
            f = recurse(e.f, var_to_count)
            args = [recurse(a, var_to_count) for a in e.args]
            return Call(f, args)

        if e.isLambda:
            local_var_to_count = var_to_count.copy()
            for arg in e.args:
                local_var_to_count[arg.name] = 0
            body = recurse(e.body, local_var_to_count)
            return Lambda(e.args, body, e.id + "/dce")

        assert False

    var_to_count = defaultdict(lambda: 0)
    return recurse(e, var_to_count)


def detuple_lets(e: Expr) -> Expr:
    # Let([Eqn([a, b, c], v),
    #       body) ->
    #  Let([Eqn(t, v)
    #     Eqn(a, t[0]),
    #       Eqn(b, t[1]),
    #       Eqn(c, t[2])],
    #           body)))

    def doit(e, bindings):
        if not e.isLet:
            return e

        def detuple_eqn(eqn):
            vars = eqn.vars
            val = eqn.val
            if len(vars) > 1:
                if val.isVar:
                    tupvar = val
                else:
                    tupvar = Var("tup_" + get_new_name())
                    yield Eqn([[tupvar, val]])

                for i, var in enumerate(vars):
                    yield Eqn([var], Call(Var("g_subscript"), [tupvar, Const(i)]))
            else:
                yield Eqn(vars, val)

        new_eqns = list(chain(*map(detuple_eqn, e.eqns)))
        return Let(new_eqns, e.body)

    return transform_postorder("dtl", doit, e, {})


def detuple_tuple_assignments(e: Expr, bindings: Dict[str, Any]) -> Expr:
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
        if len(vars) > 1 and val.isCall and val.f == Var("g_tuple"):
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


def to_anf(e, bindings):
    e = uniquify_names(e)
    assignments = []
    expr = to_anf_aux(e, assignments, bindings)
    return Let(assignments, expr)


def to_anf_aux(e, assignments, bindings):
    if e.isConst or e.isVar:
        return e

    if e.isLet:
        for eqn in e.eqns:
            new_val = to_anf_aux(eqn.val, assignments, bindings)
            assignments += [Eqn(eqn.vars, new_val)]
        abody = to_anf_aux(e.body, assignments, bindings)
        return abody

    if e.isLambda:
        return Lambda(e.args, to_anf(e.body, bindings), e.id + "/anf")

    if e.isCall:
        new_f = to_anf_aux(e.f, assignments, bindings)
        new_args = [to_anf_aux(arg, assignments, bindings) for arg in e.args]
        return Call(new_f, new_args)

    assert False


def inline_trivial_assignments(e, bindings):
    return inline_trivial_assignments_aux(e, bindings)


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
        return Lambda(e.args, new_body, e.id + "/ita")

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
    out = inline_trivial_assignments(e, {})
    print(out)
    expect = Call(b, [Call(c, [c, b, c])])
    assert out == expect


def let_to_lambda(e: Expr, bindings: Dict[str, Any]) -> Expr:
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
            vals += [eqn.val]

        return Call(Lambda(args, e.body, "l2l_" + get_new_name()), vals)


def elide_empty_lhs(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    let [] = val1, x2 = val2 in body
    ->
    let x2 = val2 in body
    """
    if e.isLet:
        eqns = [Eqn(eqn.vars, eqn.val) for eqn in e.eqns if len(eqn.vars) > 0]
        return Let(eqns, e.body)


def identify_identities(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    lambda x: x -> g_identity
    """
    if e.isLambda and [e.body] == e.args:
        return Var("g_identity")


def eliminate_identities(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    g_identity(args) -> args
    """
    if e.isCall and e.f == Var("g_identity"):
        return mkTuple(e.args)


import re


def make_new_name(name_container: Dict[str, Any] | Set[str], name: str) -> str:
    if m := re.match(r"(.*)_(\d+)$", name):
        basename = m[1]
        i = int(m[2]) + 1
    else:
        basename = name
        i = 1
    newname = name
    while newname in name_container:
        newname = basename + f"_{i}"
        i += 1
    return newname


def uniquify_names(e: Expr) -> Expr:
    # To start with, add freevars to scope, with their own names
    translations = {v.name: v.name for v in freevars(e)}
    all_names = set(translations.keys())

    def make_new(name):
        newname = make_new_name(all_names, name)
        all_names.add(newname)
        return newname

    def doit(e: Expr, translations) -> Expr:
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
            return Var(translations[e.name], annot=e.annot)

        if e.isCall:
            new_f = doit(e.f, translations)
            new_args = [doit(arg, translations) for arg in e.args]

            return Call(new_f, new_args, annot=e.annot)

        if e.isLambda:
            vars = e.args

            new_translations = translations.copy()  # TODO-Q
            new_vars = []
            for var in vars:
                newname = make_new(var.name)
                new_translations[var.name] = newname
                new_vars += [Var(newname, annot=var.annot)]

            new_body = doit(e.body, new_translations)
            return Lambda(new_vars, new_body, e.id + "/unq", annot=e.annot)

        if e.isLet:
            new_translations = {**translations}
            new_eqns = []
            for eqn in e.eqns:
                newval = doit(eqn.val, new_translations)

                new_vars = []
                for var in eqn.vars:
                    # This var has just come in scope.
                    # If its name is already in translations, it is clashing,
                    # so in the body of this let, it will need a new name
                    newname = make_new(var.name)
                    new_translations[var.name] = newname
                    new_vars += [Var(newname, annot=var.annot)]

                # Now recurse into val using new_translations so far
                new_eqns += [Eqn(new_vars, newval)]

            new_body = doit(e.body, new_translations)
            return Let(new_eqns, new_body, annot=e.annot)

        assert False  # unreachable

    return doit(e, translations)


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
    print(e)
    out = uniquify_names(e)
    print(out)
    assert out.eqns[0].vars[0] != a
    assert out.eqns[0].val == a
    assert out.body.eqns[0].vars[0] != a
    assert out.body.eqns[0].vars[0] != out.eqns[0].vars[0]


########################################################################################
#
#
#   888888888888            ad88888ba   ad88888ba        db
#        88                d8"     "8b d8"     "8b      d88b
#        88                Y8,         Y8,             d8'`8b
#        88  ,adPPYba,     `Y8aaaaa,   `Y8aaaaa,      d8'  `8b
#        88 a8"     "8a      `"""""8b,   `"""""8b,   d8YaaaaY8b
#        88 8b       d8            `8b         `8b  d8""""""""8b
#        88 "8a,   ,a8"    Y8a     a8P Y8a     a8P d8'        `8b
#        88  `"YbbdP"'      "Y88888P"   "Y88888P" d8'          `8b
#
#
########################################################################################


def to_ssa(e: Expr) -> Let | Var | Const | Lambda:
    """
    Convert an expression to SSA form.
    """
    e = uniquify_names(e)

    if e.isConst or e.isVar:
        return e

    if e.isLambda:
        # lambda xs: e
        # Becomes
        # lambda xs: let eqns in var
        lam = Lambda(e.args, to_ssa(e.body), e.id + "/ssa")
        lam_id = Var("_ssa_" + get_new_name())
        return Let([Eqn([lam_id], lam)], lam_id)

    if e.isLet:
        # let
        #   vs1 = e1
        #   vs2 = e2
        #   vs3 = var3
        # in
        #   e4
        # Becomes (first pass):
        # let
        #   vs1 = let eqns1 in var1 # names in eqns do not shadow bound names in e
        #   vs2 = let eqns2 in var2
        #   vs3 = var3
        # in
        #   let eqns4 in var4
        # Becomes (second pass):
        # let
        #   eqns1
        #   vs1 = var1
        #   eqns2
        #   vs2 = var2
        #   vs3 = var3
        #   eqns4
        # in var4
        eqns = []
        for eqn in e.eqns:
            val = to_ssa(eqn.val)
            if val.isLet:
                # Micro opt,
                # a = let b=e1; c=e2; d=e3 in d
                # ->
                # b=e1; c=e2; d=e3; a=d
                # -> elide that last var=var
                # b=e1; c=e2; a=e3
                assert val.body.isVar
                lhs = eqn.vars
                rhs_body = val.body
                done = 0
                for rhs_eqn in val.eqns:
                    if rhs_eqn.vars == [rhs_body]:
                        eqns += [Eqn(lhs, rhs_eqn.val)]
                        done += 1
                    else:
                        eqns += [rhs_eqn]
                assert done == 1

            else:
                eqns += [Eqn(eqn.vars, val)]

        body = to_ssa(e.body)
        if body.isLet:
            eqns += body.eqns
            body = body.body

        return Let(eqns, body)

    if e.isCall:
        #   Call(f, e1, e2, ...)
        # becomes
        #   Call(let eqns0 in var0, let eqns1 in var1, let eqns2 in var2, ...)
        # becomes
        #   let
        #     eqns0
        #     eqns1
        #     eqns2
        #     out = Call(vs0, vs1, vs2)
        #   in
        #     out

        eqns = []
        f = to_ssa(e.f)
        if f.isLet:
            eqns += f.eqns
            f = f.body

        args = []
        for v in e.args:
            v = to_ssa(v)
            if v.isLet:
                eqns += v.eqns
                args += [v.body]
            else:
                args += [v]

        # if f.isVar and f.name.startswith("operator."):
        #     f = Var(f"ssa_{f.name}")

        call_val = Call(f, args)
        call_var = Var("_ssa_" + get_new_name())
        return Let(eqns + [Eqn([call_var], call_val)], call_var)

    assert False


def assert_is_ssa(e: Expr) -> bool:
    """
    Check if an expression is in SSA form.
    """

    def is_val(e):
        return e.isVar or e.isConst

    if e.isLet:
        # let
        #   vs1 = rhs1
        #   vs2 = rhs2
        #   vs3 = rhs3
        # in
        #   body
        # is SSA if all rhss are vals and body is val
        # and also recurse into lambdas
        for eqn in e.eqns:
            if eqn.val.isLambda or eqn.val.isCall:
                assert_is_ssa(eqn.val)
            else:
                assert is_val(eqn.val)

    elif e.isLambda:
        assert_is_ssa(e.body)

    elif e.isCall:
        assert is_val(e.f) or e.f.isLambda
        assert all(map(is_val, e.args))
    else:
        assert is_val(e)


def to_ssa_tidy(e):
    """
    Convert an expression to SSA form and tidy it up.
    """
    e = to_ssa(e)
    e = detuple_lets(e)
    e = inline_single_usages(e)
    e = dce(e)
    e = to_ssa(e)
    return e


########################################################################################
#
#
#   88888888ba
#   88      "8b                         ,d      ,d
#   88      ,8P                         88      88
#   88aaaaaa8P' 8b,dPPYba,  ,adPPYba, MM88MMM MM88MMM 8b       d8
#   88""""""'   88P'   "Y8 a8P_____88   88      88    `8b     d8'
#   88          88         8PP"""""""   88      88     `8b   d8'
#   88          88         "8b,   ,aa   88,     88,     `8b,d8'
#   88          88          `"Ybbd8"'   "Y888   "Y888     Y88'
#                                                         d8'
#                                                        d8'
########################################################################################

# Pretty printing of exprs

import wadler_lindig as wl


def wl_pdoc(e):
    if not isinstance(e, Expr):
        return None

    recurse = lambda x: wl.pdoc(x, custom=wl_pdoc)
    spc = wl.BreakDoc(" ")
    semi = wl.BreakDoc("; ")

    if e.isConst:
        return wl.TextDoc(better_repr(e.val))

    if e.isVar:
        name = wl.TextDoc(e.name)
        if e.annot is None:
            return name

        annot = wl.TextDoc(better_repr(e.annot))
        return (name + wl.TextDoc("{") + annot + wl.TextDoc("}")).group()

    if e.isCall:
        fn = recurse(e.f)
        brk = wl.BreakDoc("")
        if len(e.args) > 0:
            args = [recurse(arg) for arg in e.args]
            args = wl.join(wl.comma, args)
            args = (brk + args.group()).nest(2)
            return (fn + wl.TextDoc("(")).group() + args + brk + wl.TextDoc(")")
        else:
            return (fn + wl.TextDoc("()")).group()

    if e.isLambda:

        args = wl.join(wl.comma, list(map(recurse, e.args)))
        head = (
            wl.TextDoc("lambda") + wl.TextDoc(" (") + args + wl.TextDoc("): ")
        ).group()
        body = recurse(e.body)
        return (
            (wl.TextDoc("{" + f"# {e.id}\n") + head + spc + body + wl.TextDoc("}"))
            .group()
            .nest(2)
        )

    if e.isLet:

        def doeqn(vars, val):
            if vars:
                vars = wl.join(wl.comma, list(map(recurse, vars)))
            else:
                vars = wl.TextDoc("__warning__null_vars__")
            return (
                ((vars + spc + wl.TextDoc("=")).group() + spc + recurse(val))
                .group()
                .nest(2)
            )

        eqns = [doeqn(eqn.vars, eqn.val) for eqn in e.eqns]
        eqns = (
            wl.join(semi, eqns) if eqns else wl.TextDoc("__warning__no_eqns__ = None")
        )
        body = recurse(e.body)
        let_doc = (wl.TextDoc("let") + spc + eqns).group().nest(2)
        in_doc = (wl.TextDoc("in") + spc + body).group().nest(2)
        return (let_doc + spc + in_doc).group()

    return None


def expr_format(e, **kwargs):
    if "width" not in kwargs:
        kwargs["width"] = 120
    return wl.pformat(e, custom=wl_pdoc, **kwargs)


########################################################################################
#
#
#   88b           d88 88
#   888b         d888 ""
#   88`8b       d8'88
#   88 `8b     d8' 88 88 ,adPPYba,  ,adPPYba,
#   88  `8b   d8'  88 88 I8[    "" a8"     ""
#   88   `8b d8'   88 88  `"Y8ba,  8b
#   88    `888'    88 88 aa    ]8I "8a,   ,aa
#   88     `8'     88 88 `"YbbdP"'  `"Ybbd8"'
#
#
########################################################################################


def rename_let_v_in_v(e, name):
    # Force name of first bound item to be 'name'
    if e.isLet:
        if len(e.eqns) > 0 and len(e.eqns[0].vars) == 1:
            ename = e.eqns[0].vars[0].name
            if e.body.isVar and ename == e.body.name:
                # If the first bound item is the same as the body, then
                # we can just use the name of the first bound item
                return Let(
                    [Eqn([Var(name)], e.eqns[0].val)] + e.eqns[1:],
                    Var(name),
                )
        # It's a let, but the first bound item is not a single var
        print(f"rename_let_v_in_v: {e}\nrename_let_v_in_v: cannot rename to {name}")
        return e
    else:
        # Wrap it in a Let
        return Let([Eqn([Var(name)], e)], Var(name))


from typing import Sequence


def concatlists(lists: Sequence[List[Any]]) -> List[Any]:
    """
    Concatenate a list of lists into a single list.
    """
    return [item for sublist in lists for item in sublist]


def make_vjp(e, drets) -> tuple[List[Var], Expr]:
    # vjp(e, drets) returns a pair of (d(freevars), expr)
    def oplus_(d, d2):
        for k, v in d2.items():
            if k in d:
                d[k] = _add(d[k], v)
            else:
                d[k] = v

    def d(v):
        return Var(f"d{v.name}")

    def vjp_name(e):
        # assert e.isVar, f"vjp_name: e must be a Var, not {e}"
        return Call(Var("g_vjp"), [e])

    assert isinstance(drets, list), "drets must be a list"

    if e.isConst:
        return [], None

    if e.isVar:
        return [d(e)], mkTuple(drets)

    if e.isCall:
        # D[e, drets] = pair of (d[freevars] : list of Var, val: Expr<fvs, drets> returning tuple of vals)
        # D[f(arg1, ..., argn), drets] =
        #  [d(fvs)], let
        #              _d1, ..., _dn = vjp[f](arg1, ..., argn, drets)
        #              dfv1_1, dfv4_1, dfvk_1 = D[arg1, _d1] # This eqn contribs to fv1, fv4, fvk
        #              ...
        #              dfv2_2, dfv4_2, dfv7_2 = D[argn, _dn]
        #            in
        #              g_tuple(dfv1_1, ..., dfv4_1 + dfv4_2, ... dfv_k_n)

        f_vjp = vjp_name(e.f)
        dis = [Var(f"_d{i}_" + get_new_name()) for i, arg in enumerate(e.args)]

        # Replace any lambdas l in args with a pair (l, vjp_l)
        new_args = []
        for arg in e.args:
            assert arg.isVar or arg.isConst
            if (
                arg.isVar
                and isinstance(arg.annot, Callable)
                and arg.annot.__name__ == "runLambda"
            ):
                # If arg is a lambda, replace it with a tuple of (arg, d(arg))
                new_args += [mkTuple([Const("f&v"), arg, d(arg)])]
            else:
                # Otherwise, just use the arg
                new_args += [arg]

        eqns = [Eqn(dis, Call(f_vjp, new_args + drets))]
        fv_contribs = {}
        for i, (arg, di) in enumerate(zip(e.args, dis)):
            fvs, expr = make_vjp(arg, [di])
            if not expr:
                assert not fvs
                continue
            fvs_i = [Var(v.name + "_" + str(i) + "_" + get_new_name()) for v in fvs]
            eqns += [Eqn(fvs_i, expr)]
            oplus_(fv_contribs, {fv: fv_i for fv, fv_i in zip(fvs, fvs_i)})

        fvs = list(fv_contribs.keys())
        rets = mkTuple(list(fv_contribs.values()))
        ret = Let(eqns, rets)

        return fvs, ret

    if e.isLet:
        # D[let
        #     vs1 = e1
        #     f = lambda x: e
        #     vs2 = e2
        #   in
        #     body,
        #   dret] =
        #
        # [fvs(e)], let
        #             vs1 = e1
        #             f = lambda x: e
        #             df = vjp[lambda x: e]
        #             vs2 = e2
        #             dfvs_body = D[body, dret]  # map fvs of body (say vs2, vs1, g) to contribs
        #             dfvs_e2 = D[e2, dvs2]   # mapping from freevars of e2 (say vs1, g) to diffs
        #             dfvs_e1 = D[e1, dvs1]   # mapping from freevars of e1 to diffs
        #           in
        #             osum(dfvs_body, dfvs_e2, dfvs_e1)
        fvs_body, dbody = make_vjp(e.body, drets)
        if not dbody:
            # Whole let is constant, return
            return [], None

        fvs_body_0 = [Var(v.name + "_body") for v in fvs_body]

        # Check there's no name shadowing in this let - code should
        # work, but hasn't been tested
        bound_vars = concatlists(eqn.vars for eqn in e.eqns)
        assert len(set(bound_vars)) == len(bound_vars), "untested"

        eqns = []
        # Emit vjps for forward pass, and lambdas
        for eqn in e.eqns:
            eqns += [Eqn(eqn.vars, eqn.val)]
            if eqn.val.isLambda:
                douts = [d(one(eqn.vars))]
                fvs, dval = make_vjp(eqn.val, douts)
                eqns += [Eqn(fvs, dval)]
            else:
                # hope it's not a nested expression that evals to a lambda
                assert not (
                    isinstance(eqn.val.annot, Callable)
                    and eqn.val.annot.__name__ == "runLambda"
                )

        # Emit vjps for the body
        eqns += [Eqn(fvs_body_0, dbody)]

        fv_contribs = {fv: fv_body_0 for fv, fv_body_0 in zip(fvs_body, fvs_body_0)}

        for i, eqn in enumerate(reversed(e.eqns)):
            if not eqn.val.isLambda:
                # Make douts, and remove from contribs if present - they can't be needed
                # again, as they just came into scope in the primal, so are going out of
                # scope in the adjoint.
                douts = []
                for out in eqn.vars:
                    dout = d(out)
                    if dout in fv_contribs:
                        douts += [fv_contribs[dout]]
                        del fv_contribs[dout]
                    else:
                        douts += [dout]

                # Make the vjp for the rhs
                fvs, dval = make_vjp(eqn.val, douts)
                if not dval:
                    assert not fvs
                    continue

                # Rename the fvs so we can access them in the contribs
                fvs_i = [Var(v.name + "_" + str(i)) for v in fvs]
                eqns += [Eqn(fvs_i, dval)]
                oplus_(fv_contribs, {fv: fv_i for fv, fv_i in zip(fvs, fvs_i)})

        fvs = list(fv_contribs.keys())
        rets = mkTuple(list(fv_contribs.values()))
        ret = Let(eqns, rets)

        return fvs, ret

    if e.isLambda:
        # D[lambda args: body, _drets] =
        #   [fvs(body)-args], lambda args, dret: D[body, dret]
        fvs_in = drets
        drets = [Var(f"dret{get_new_name()}")]
        fvs_body, dbody = make_vjp(e.body, drets)
        dargs_assigned = [
            d(arg) if d(arg) in fvs_body else _zeros_like(arg) for arg in e.args
        ]
        dbody = Let([Eqn(fvs_body, dbody)], mkTuple(dargs_assigned))
        dargs = set(d(arg) for arg in e.args)
        fvs_remaining = [fv for fv in fvs_body if fv not in dargs]
        if fvs_remaining:
            print(
                f"make_vjp: {len(fvs_remaining)} free vars in lambda not in args: {fvs_remaining}"
            )

        return fvs_in, Lambda(e.args + drets, dbody, e.id + "/vjp")
