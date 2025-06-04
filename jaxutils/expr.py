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
def get_function_shortname(f: Callable) -> str:
    """
    Get the short name of a function, which is used for logging.
    """
    if hasattr(f, "shortname"):
        return f.shortname
    if hasattr(f, "__name__"):
        return f.__name__
    return str(f)


_all_shortnames = {}


def shortname(s: str):
    """
    A decorator to add a shortname to a function.
    This is used for logging purposes.
    """

    def wrapper(func):
        if not hasattr(func, "shortname"):
            if _all_shortnames.get(s, func) != func:
                print(f"Shortname {s} already used, for func {func}")
            _all_shortnames[s] = func

            func.shortname = s
        return func

    return wrapper


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

    @property
    def isEqn(self):
        return isinstance(self, Eqn)

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
        return hash((Const, self.val))


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


@exprclass
class Eqn(Expr):
    vars: List[Var]
    val: Expr

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash((Eqn, self.vars, self.val))


@exprclass
class Let(Expr):
    eqns: List[Eqn]
    body: Expr

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash((Let, self.eqns, self.body))


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
        return hash((Lambda, self.args, self.body, self.id))


@exprclass
class Call(Expr):
    f: Expr
    args: List[Expr]

    # [Note 1]:
    annot: Any = None

    def __hash__(self):
        return hash((Call, self.f, self.args))


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


def extend_id(id: LambdaId, transform: Callable) -> LambdaId:
    return id + "/" + get_function_shortname(transform)


def transform_postorder(transformer: Callable[[Expr, Dict], Expr], *args):
    if len(args) == 0:
        # decorator mode
        # Make transform(transformer)(e) work like transform(transformer, e)
        l = lambda *args: transform_postorder(transformer, *args)
        l.__name__ = transformer.__name__
        l.shortname = transformer.shortname
        return l

    recurse = lambda e, bindings: transform_postorder(transformer, e, bindings)

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
            new_eqn = Eqn(new_vars, new_val)
            new_eqn = recurse(new_eqn, new_bindings)
            new_eqns += [new_eqn]
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
        new_id = extend_id(e.id, transformer)
        e = Lambda(new_args, new_body, new_id, annot=new_body.annot)

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

    def run(transformation, ex, wrapper=None):
        print(f"Running {transformation.shortname}:{transformation.__name__}")

        new_ex = transformation(ex, {})

        ex2py(f"{run.count:02d}-{transformation.__name__}", new_ex)
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
    for t in (
        uniquify_names,
        elide_empty_lhs,
        inline_call_of_lambda,
        inline_trivial_letbody,
        inline_lambda_of_call,
        inline_lambda_of_let_of_call,
        identify_identities,
        eliminate_identities,
        to_anf,
        detuple_tuple_assignments,
        inline_trivial_assignments,
        eliminate_identities,
        inline_trivial_assignments,
    ):
        e = run(t, e)
    ex2py(f"99-after-optimization", e)

    return e


@transform_postorder
@shortname("icl")
def inline_call_of_lambda(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # call(lambda l_args: body, args)
    #  ->  let l_args = args in body
    # Name clashes will happen unless all bound var names were uniquified
    if e.isCall and e.f.isLambda:
        return Let([Eqn(e.f.args, mkTuple(e.args))], e.f.body)


@transform_postorder
@shortname("ill")
def inline_lambda_of_let_of_call(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # lambda args: let eqns in call(f, args)
    #  ->  let eqns in f
    # Name clashes will happen unless all bound var names were uniquified
    if e.isLambda and e.body.isLet and e.body.body.isCall:
        if e.body.body.args == e.args:
            return Let(e.body.eqns, e.body.body.f)


@transform_postorder
@shortname("itb")
def inline_trivial_letbody(e: Expr, bindings: Dict[str, Any]) -> Expr:
    # let var = val in var -> val
    if e.isLet and len(e.eqns) == 1 and len(e.eqns[0].vars) == 1:
        if e.eqns[0].vars == [e.body]:
            return e.eqns[0].val
    if e.isLet and len(e.eqns) == 0:
        return e.body


@transform_postorder
@shortname("ilc")
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


@shortname("isu")
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
            return Lambda(e.args, body, extend_id(e.id, inline_single_usages))

        assert False

    return recurse(e, {})


def splitlet(e):
    assert e.isLet and len(e.eqns) > 0
    vars = e.eqns[0].vars
    val = e.eqns[0].val
    new_let = Let(e.eqns[1:], e.body)
    return vars, val, new_let


def prependlet(vars, val, e):
    if e.isLet:
        return Let([Eqn(vars, val)] + e.eqns, e.body)
    else:
        return Let([Eqn(vars, val)], e)


@shortname("dce")
def dce(e: Expr) -> Expr:
    """
    Dead Code Elimination
    Remove bindings to unused variables.
    """

    # dce(k) = {}, k
    # dce(v) = {v}, v  # As only called on rhs vars
    # dce(f (e1..n)) = fvsf, f' = dce(f, bindings):
    #                  fvs1, e1' = dce(e1, bindings)
    #                  fvs2, e2' = dce(e2, bindings)
    #                  ...
    #                  fvsf + fvs1 + fvs2 + ... , Call(f', [e1', e2', ...])
    # dce(e) -> (fvs, e')
    # where fvs is the set of free variables in e'
    # dce(let v = e1 in e2) = {
    #                          fvs1, e1' = dce(e1, bindings)
    #                          fvse2, e2' = dce(e2, bindings + {v})
    #                         }:
    #                         fv1 | fvs2, let v = e1' in e2' {if v in fvs}
    #                         fvs2, e2'                      {if v not in fvs}
    # dce(lambda xs. e) = fvs, e' = dce(e, bindings + {xs})

    def recurse(e, bindings):
        if e.isConst:
            return set(), e

        if e.isVar:
            return {e} - bindings, e

        if e.isLet:
            if len(e.eqns) == 0:
                return recurse(e.body, bindings)

            # peel one equation at a time
            # dce(let var = e1 in body) =
            #      {
            #       fvs1, e1' = dce(e1, bindings)
            #       fvsbody, body' = dce(body, bindings + {var})
            #      }:
            #      fv1 | fvs2, let v = e1' in e2' {if v in fvsbody}
            #      fvs2, e2'                      {if v not in fvs}
            vars, e1, body = splitlet(e)
            fvs1, new_e1 = recurse(e1, bindings)
            fvsbody, new_body = recurse(body, bindings | set(vars))
            if not (set(vars) < fvsbody):
                return fvs1 | fvsbody, prependlet(vars, new_e1, new_body)
            else:
                return fvsbody, new_body

        if e.isCall:
            fvs, new_f = recurse(e.f, bindings)
            new_args = []
            for a in e.args:
                fvs_a, new_a = recurse(a, bindings)
                new_args += [new_a]
                fvs |= fvs_a

            return fvs, Call(new_f, new_args)

        if e.isLambda:
            fvs, new_body = recurse(e.body, bindings | set(e.args))
            return fvs, Lambda(e.args, new_body, extend_id(e.id, dce))

        assert False

    fvs, e = recurse(e, set())
    return e


@shortname("dtl")
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

    return transform_postorder(doit, e, {})


@transform_postorder
@shortname("dta")
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


@shortname("anf")
def to_anf(e, bindings):
    e = uniquify_names(e, bindings)
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
        new_id = extend_id(e.id, to_anf)
        return Lambda(e.args, to_anf(e.body, bindings), new_id)

    if e.isCall:
        new_f = to_anf_aux(e.f, assignments, bindings)
        new_args = [to_anf_aux(arg, assignments, bindings) for arg in e.args]
        return Call(new_f, new_args)

    assert False


@shortname("ita")
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
        new_id = extend_id(e.id, inline_trivial_assignments)
        return Lambda(e.args, new_body, new_id)

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


@shortname("l2l")
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


@transform_postorder
@shortname("eel")
def elide_empty_lhs(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    let [] = val1, x2 = val2 in body
    ->
    let x2 = val2 in body
    """
    if e.isLet:
        eqns = [Eqn(eqn.vars, eqn.val) for eqn in e.eqns if len(eqn.vars) > 0]
        return Let(eqns, e.body)


@transform_postorder
@shortname("iid")
def identify_identities(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    lambda x: x -> g_identity
    """
    if e.isLambda and [e.body] == e.args:
        return Var("g_identity")


@transform_postorder
@shortname("eid")
def eliminate_identities(e: Expr, bindings: Dict[str, Any]) -> Expr:
    """
    g_identity(args) -> args
    """
    if e.isCall and e.f == Var("g_identity"):
        return mkTuple(e.args)


import re


@beartype
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


@beartype
def make_new_var(var_container: Set[Var], v: Var) -> Var:
    # TODO, slow... instead duplicate logic above, but let's wait until solidified
    name_container = {v.name for v in var_container}
    newname = make_new_name(name_container, v.name)
    return Var(newname, annot=v.annot)


@shortname("unq")
def uniquify_names(e: Expr, _bindings) -> Expr:
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
            new_id = e.id + "/" + get_function_shortname(uniquify_names)
            return Lambda(new_vars, new_body, new_id, annot=e.annot)

        if e.isLet:
            # Commandeer new names for everyone in this let
            local_newnames = {
                (id(eqn), id(var)): make_new(var.name)
                for eqn in e.eqns
                for var in eqn.vars
            }

            new_translations = {**translations}
            new_eqns = []
            for eqn in e.eqns:
                newval = doit(eqn.val, new_translations)

                new_vars = []
                for var in eqn.vars:
                    newname = local_newnames[(id(eqn), id(var))]
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
    out = uniquify_names(e, {})
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


@shortname("ssa")
def to_ssa(e: Expr) -> Let | Var | Const | Lambda:
    """
    Convert an expression to SSA form.
    """
    e = uniquify_names(e, {})

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


def d(v):
    return Var(f"d{v.name}")


def make_vjp(e, drets) -> tuple[List[Var], Expr]:
    # vjp(e, drets) returns a pair of (d(freevars), expr)
    def oplus_(d, d2):
        for k, v in d2.items():
            if k in d:
                d[k] = _add(d[k], v)
            else:
                d[k] = v

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
        #              dfv4_2, dfv7_2 = D[argn, _dn]         # This eqn contribs to fv4, fv7
        #            in
        #              g_tuple(dfv1_1, ..., dfv4_1 + dfv4_2, ... dfv_k_n)

        # Replace any lambdas l in args with a pair (l, vjp_l)
        new_args = []
        for arg in e.args:
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

        f_vjp = vjp_name(e.f)

        # Need to make temporaries for all args
        dis = [Var(f"_d{i}_" + get_new_name()) for i, arg in enumerate(e.args)]
        eqns = [Eqn(dis, Call(f_vjp, new_args + drets))]

        # Now add in contributions to freevars
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

        # Check there's no name shadowing in this let - code should
        # work, but hasn't been tested
        bound_vars = concatlists(eqn.vars for eqn in e.eqns)
        assert len(set(bound_vars)) == len(bound_vars), "untested"

        eqns = []
        # Emit vjps for forward pass, and lambdas
        for eqn in e.eqns:
            # emit the eqn
            eqns += [Eqn(eqn.vars, eqn.val)]

            # and, for a lambda, emit the vjp
            if eqn.val.isLambda:
                dlamname = d(one(eqn.vars))
                fvs, dlam = make_vjp_for_lambda(eqn.val)
                eqns += [Eqn([dlamname], dlam)]

        dfvs_body, dbody = make_vjp(e.body, drets)

        # Emit vjps for the body
        eqns += [Eqn(dfvs_body, dbody)]

        dfvs_in_scope = set(dfvs_body)

        for i, eqn in enumerate(reversed(e.eqns)):
            if eqn.val.isLambda:
                continue

            douts = [d(out) for out in eqn.vars]

            # Make the vjp for the rhs
            dfvs_val, dval = make_vjp(eqn.val, douts)

            if not dfvs_val:
                continue

            # For values that are in scope, we'll need to add the contributions,
            # so use a dummy in the lhs.
            # i.e., we had
            #   foo(a,b,c,d)
            #      r = a+b
            #      s = r+d
            #      t = a+s
            #      return sin(t)
            # and so far, we have
            #   dfoo(a,b,c,d, dret)
            #      r = a+b
            #      s = r+d
            #      t = a+s
            #      dt = cos(t) * dret
            #      da, ds = +'(a,s,dt)
            #      dr, dd = +'(r,d,ds)
            #      # da, db = +'(a,b,dr) # oops about to overwrite da
            #      da_1, db = +'(a,b,dr) # so put it in a temp
            #      da = da + da_1  # and add it
            #      return (da, db, dc, dd)

            # Rename the fvs so we can access them in the contribs
            dv_contribs = [make_new_var(dfvs_in_scope, dv) for dv in dfvs_val]
            eqns += [Eqn(dv_contribs, dval)]

            # And add any that got renamed
            for dv, dv_contrib in zip(dfvs_val, dv_contribs):
                if dv != dv_contrib:
                    eqns += [Eqn([dv], _add(dv, dv_contrib))]

            # Now these are all in scope
            dfvs_in_scope |= set(dfvs_val)

            # And my douts are going out of scope - this was the eqn than introduced them
            dfvs_in_scope -= set(douts)

        dfvs = list(dfvs_in_scope)

        # dfvs = [
        #     d(v) for v in freevars(e) if v in dfvs_in_scope
        # ]  # TODO: derive from above pass

        ret = Let(eqns, mkTuple(dfvs))

        return dfvs, ret

    if e.isLambda:
        assert False, "all lambdas should be let-bound"


def is_global(v: Expr) -> bool:
    if isinstance(v.annot, Callable) and v.annot.__name__ in ("ExprCall",):
        # If the annot is an ExprCall, then this is a global variable
        return True

    return False


def make_vjp_for_lambda(e: Lambda) -> tuple[List[Var], Lambda]:
    # D[lambda args: body, _drets] =
    #   [dfvs],
    #   lambda args, dret: let dfvs_body = D[body, dret] in dargs {*+ dfvs*}
    # Where the returned set is:
    #  - the derivatives wrt each arg, whether or not it was free in the body.
    # and TODO: {*+ dfvs*}
    #  - the derivatives wrt each free var in lambda, in the order returned in [dfvs]
    #    hence dfvs is a list, rather than a set, to emphasize the order dependence.
    print(f"make_vjp: lambda {e.id}")
    fvs_body_0 = freevars(e.body)

    dret = make_new_var(fvs_body_0, Var("dret"))
    drets = [dret]

    dfvs_body, dbody = make_vjp(e.body, drets)
    # assert {d(v) for v in fvs_body_0} == set(dfvs_body)

    # Any args not in fvs_body will have zero gradient contributions, but must be
    # returned in args order
    dargs = set(d(arg) for arg in e.args)
    dfvs = [dfv for dfv in dfvs_body if dfv not in dargs and not is_global(dfv)]

    dargs_assigned = [
        d(arg) if d(arg) in dfvs_body else _zeros_like(arg) for arg in e.args
    ]
    dbody_with_args = Let([Eqn(dfvs_body, dbody)], mkTuple(dargs_assigned))

    dlam = Lambda(e.args + drets, dbody_with_args, e.id + "/vjp")

    if dfvs:
        print(f"make_vjp: {len(dfvs)} free vars in lambda not in args: {dfvs}")

    return dfvs, dlam
