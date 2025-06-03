from beartype import beartype
from beartype.typing import Any, Dict, Sequence, Callable
from more_itertools import one

from jaxutils.expr import (
    Expr,
    Var,
    Const,
    Let,
    Lambda,
    LambdaId,
    Call,
    Eqn,
    freevars,
)
from jaxutils.expr_ast import _bindings_for_operators
from jaxutils import expr_lib

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


def eval_expr(e: Expr, args, bindings=None, add_operators=True):

    bindings = bindings or {}
    if add_operators:
        bindings = _bindings_for_operators() | bindings
        bindings = _bindings_for_expr_lib() | bindings
    return _run_eval(Call(e, [Const(a) for a in args]), bindings)


def _bindings_for_expr_lib():
    return {f: getattr(expr_lib, f) for f in dir(expr_lib) if f.startswith("g_")}


def _run_eval(e: Expr, bindings: Dict[str, Any]) -> Any:
    new_bindings = {Var(key): val for key, val in bindings.items()}
    # Check all the freevars have been bound
    unbound_vars = [v.name for v in set(freevars(e)) - set(new_bindings)]
    if len(unbound_vars) > 0:
        raise ValueError(f"Unbound variable(s): {', '.join(unbound_vars)}")

    return _run_eval_aux(e, new_bindings)


@beartype
def _run_eval_aux(e: Expr, bindings: Dict[Var, Any]) -> Any:
    recurse = _run_eval_aux

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
        new_bindings = bindings.copy()
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

        # TODO: Lexical scope?

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


########################################################################################
#
#
#          db                                 88888888888                    88
#         d88b                                88                             88
#        d8'`8b                               88                             88
#       d8'  `8b     8b,dPPYba,  8b,dPPYba,   88aaaaa 8b       d8 ,adPPYYba, 88
#      d8YaaaaY8b    88P'   `"8a 88P'   `"8a  88""""" `8b     d8' ""     `Y8 88
#     d8""""""""8b   88       88 88       88  88       `8b   d8'  ,adPPPPP88 88
#    d8'        `8b  88       88 88       88  88        `8b,d8'   88,    ,88 88
#   d8'          `8b 88       88 88       88  88888888888 "8"     `"8bbdP"Y8 88
#
########################################################################################


@beartype
def annotate_eval(
    e: Expr, args: Sequence[Any], bindings: Dict[str, Any], add_operators=True
) -> Expr:
    """
    Evalute the expression `e` with the given `args`, and annotate all expr nodes
    with their evaluated values.

    This can be used to annotate an expression with types or other
    information.
    """
    if add_operators:
        bindings |= _bindings_for_operators()
        bindings |= _bindings_for_expr_lib()

    annotated_lambdas: Dict[int, Lambda] = {}

    call = Call(e, [Const(a, annot=a) for a in args])

    e1 = _annotate_eval_aux(call, bindings, annotated_lambdas, cache_calls=True)

    e2 = _annotate_eval_lookup_lambdas(e1, annotated_lambdas)

    return e2.f


@beartype
def _annotate_eval_aux(
    e: Expr,
    bindings: Dict[str, Any],
    annotated_lambdas: Dict[LambdaId, Lambda],
    cache_calls: bool,
) -> Expr:

    recurse = lambda e, b: _annotate_eval_aux(e, b, annotated_lambdas, cache_calls)

    if e.isConst:
        return Const(e.val, annot=e.val)

    if e.isVar:
        if e.name not in bindings:
            raise ValueError(f"Unbound variable {e.name}")
        return Var(e.name, annot=bindings[e.name])

    if e.isLet:
        # let
        #   vars1 = val1 [fvs x]
        #   vars2 = val2 [fvs x vars1]
        # in
        #   body
        new_bindings = bindings.copy()
        new_eqns = []
        for eqn in e.eqns:
            new_val = recurse(eqn.val, new_bindings)

            var = one(eqn.vars)  # No Expr tuple so need to detuple_lets first
            new_bindings[var.name] = new_val.annot
            new_vars = [Var(var.name, annot=new_val.annot)]

            new_eqns += [Eqn(new_vars, new_val)]

        new_body = recurse(e.body, new_bindings)
        return Let(new_eqns, new_body, annot=new_body.annot)

    if e.isCall:
        new_f = recurse(e.f, bindings)
        new_args = [recurse(arg, bindings) for arg in e.args]

        new_f_val = new_f.annot
        assert isinstance(new_f_val, Callable)
        arg_vals = tuple(arg.annot for arg in new_args)
        val = new_f_val(*arg_vals)
        return Call(new_f, new_args, annot=val)

    if e.isLambda:

        lam = e
        lam_id = e.id
        lam_argset = {v.name for v in lam.args}

        run_lambda = None
        if lam_id in annotated_lambdas and cache_calls:
            # We've already run this one...
            # Just return a trivial lambda
            print(f"Using cached call to {lam_id}")
            val = annotated_lambdas[lam_id].annot
            run_lambda = lambda *args: val
        else:

            def runLambda(*arg_vals) -> Any:
                new_bindings = {
                    name: val
                    for (name, val) in bindings.items()
                    if name not in lam_argset
                }
                assert len(arg_vals) == len(lam.args)
                new_args = []
                for var, val in zip(lam.args, arg_vals):
                    new_bindings[var.name] = val
                    new_args += [Var(var.name, annot=val)]

                new_body = recurse(lam.body, new_bindings)
                new_lam = Lambda(
                    new_args, new_body, lam_id + "/ata", annot=new_body.annot
                )
                annotated_lambdas[lam_id] = new_lam

                # Now we know that running this lambda with these args produced "ret"
                msg = f"Ran lam {lam_id} with args {list(str(a) for a in arg_vals)}"
                if len(msg) > 73:
                    msg = msg[:70] + "..."
                print(f"{msg} -> {str(new_body.annot)}")

                return new_body.annot

            run_lambda = runLambda

        # The value of a lambda is the expr
        return Lambda(e.args, e.body, e.id, annot=run_lambda)

    assert False


def _annotate_eval_lookup_lambdas(e: Expr, annotated_lambdas) -> Expr:
    recurse = lambda e: _annotate_eval_lookup_lambdas(e, annotated_lambdas)

    if e.isConst or e.isVar:
        return e

    if e.isLet:
        new_eqns = [Eqn(eqn.vars, recurse(eqn.val)) for eqn in e.eqns]
        new_body = recurse(e.body)
        return Let(new_eqns, new_body, annot=new_body.annot)

    if e.isCall:
        new_f = recurse(e.f)
        new_args = list(map(recurse, e.args))
        return Call(new_f, new_args, annot=e.annot)

    if e.isLambda:
        lam_id = e.id
        if lam_id.endswith("/ll"):
            lam_id = lam_id[: -len("/ll")]
        assert e.annot is not None or lam_id in annotated_lambdas
        if lam_id not in annotated_lambdas:
            raise ValueError(
                f"Lambda {lam_id} not found in annotated lambdas - are you sure it got called?"
            )
        new_e = annotated_lambdas[lam_id]
        new_body = recurse(new_e.body)
        new_e = Lambda(new_e.args, new_body, id=e.id + "/ll", annot=new_e.annot)
        return new_e
