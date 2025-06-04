import sys
import ast
import operator
from itertools import chain
from more_itertools import one
from beartype import beartype
from beartype.typing import List, List, Callable, Optional, Set

from jaxutils.expr import (
    Expr,
    Var,
    Const,
    Let,
    Lambda,
    Call,
    Eqn,
    mkTuple,
    isNone,
    get_new_name,
    freevars,
)

from jaxutils.ast_utils import FreeVarAnalyzer, better_repr

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


def kwargs_to_dict_call(dict):
    if not dict:
        return []

    dict_pairs = [[Const(key), val] for key, val in dict.items()]
    return [Call(Var("**g_pairs_to_dict"), list(chain(*dict_pairs)))]


def _bindings_for_operators():
    ops = {("operator." + op): operator.__dict__[op] for op in _ast_ops.keys()}
    ops["getattr"] = getattr
    return ops


def _ast_op_to_Var(op):
    if op in _ast_op_to_operator:
        return Var("operator." + _ast_op_to_operator[op])
    else:
        raise ValueError(f"Unknown operator {op}")


def to_ast(e, name, flat_lets=False) -> ast.Module:
    var = Var(name)
    assignments = []
    if expr := to_ast_aux(e, assignments, [var], flat_lets):
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
    aargs = [ast.arg(v.name, annotation=mkannot(v)) for v in vars]
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
    rep = better_repr(val)

    module = ast.parse(rep)
    ast_expr: ast.Expr = module.body[0]
    assert isinstance(ast_expr, ast.Expr)
    return ast_expr.value


def mkannot(e):
    if e.annot is not None:
        return to_ast_constant(e.annot)
    else:
        return None


def to_ast_aux(e, assignments, binders, flat_lets):
    recurse = lambda e, assignments, binders=None: to_ast_aux(
        e, assignments, binders, flat_lets
    )

    if e.isConst:
        return to_ast_constant(e.val)

    if e.isVar:
        return ast.Name(e.name, ast.Load())

    if e.isLet:
        local_assignments = []
        for eqn in e.eqns:
            avars = [ast.Name(var.name, ast.Store()) for var in eqn.vars]
            if len(avars) > 1:
                avars = [ast.Tuple(avars, ast.Store())]

            if aval := recurse(eqn.val, local_assignments, binders=eqn.vars):
                if eqn.val.annot is not None:
                    local_assignments += [
                        ast.AnnAssign(
                            target=one(avars),
                            value=aval,
                            annotation=mkannot(eqn.val),
                            simple=1,
                        )
                    ]
                else:
                    local_assignments += [ast.Assign(targets=avars, value=aval)]

        abody = recurse(e.body, local_assignments)
        if flat_lets:
            assignments += local_assignments
        else:
            assignments += [
                ast.With(
                    items=[ast.withitem(context_expr=ast.Name("g_let", ast.Load()))],
                    body=local_assignments,
                )
            ]
        return abody

    if e.isLambda:
        # If I know whom I am bound to, then use that name for the FunctionDef
        if binders is None:
            v = Var("f" + get_new_name())
        else:
            v = one(binders)

        # first assignment is "_id = e.id"
        inner_assignments = [
            ast.Assign(
                targets=[ast.Name("_id", ast.Store())],
                value=to_ast_constant(e.id),
            )
        ]

        # Recurse, generating inner assignments
        abody = recurse(e.body, inner_assignments)
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
        # Special case: **g_pairs_to_dict
        if e.f.isVar and e.f.name == "**g_pairs_to_dict":
            # convert to dict call
            keywords = [
                ast.keyword(arg=k.val, value=recurse(value, assignments))
                for k, value in zip(e.args[::2], e.args[1::2])
            ]
            dictval = ast.Call(
                func=ast.Name("dict", ast.Load()), args=[], keywords=keywords
            )
            # Splatting is a keyword with arg=None
            return ast.keyword(value=dictval)

        # Special case: g_tuple
        if e.f.isVar and e.f.name == "g_tuple":
            # convert to tuple call
            elts = [recurse(a, assignments) for a in e.args]
            return ast.Tuple(elts=elts, ctx=ast.Load())

        # Special case: g_subscript
        # g_subscript(x, g_tuple(i, g_slice(None, None, None)))
        if e.f.isVar and e.f.name == "g_subscript":
            # convert to subscript call
            assert len(e.args) == 2
            value = recurse(e.args[0], assignments)
            slices = recurse(e.args[1], assignments)
            return ast.Subscript(value=value, slice=slices, ctx=ast.Load())

        # Special case: g_slice
        if e.f.isVar and e.f.name == "g_slice":
            # convert to slice call
            assert len(e.args) == 3
            args = (None if isNone(a) else recurse(a, assignments) for a in e.args)
            lower, upper, step = args
            return ast.Slice(lower=lower, upper=upper, step=step)

        args = [recurse(arg, assignments) for arg in e.args]
        f = recurse(e.f, assignments)

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

            print(f"to:ast: Unknown operator {op}")
            return ast.Call(func=f, args=args, keywords=[])

        elif isinstance(f, ast.Name) and f.id == "getattr":
            # Special case: getattr
            assert len(args) == 2
            return ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load())
        else:
            # Normal case
            return ast.Call(func=f, args=args, keywords=[])

    assert False


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


def _ast_shortstr(a):
    if a is None:
        return ""

    if isinstance(a, ast.FunctionDef):
        return f"def[{a.name}]"

    return type(a).__name__


def _ast_error(msg, path):
    loc = "/".join(map(_ast_shortstr, path))
    raise ValueError(f"{loc}: {msg}")


@beartype
def ast_to_expr(
    a: Optional[ast.AST],
    path_to_a: List[Optional[ast.AST]],
    global_names: Optional[Set[str]] = None,
) -> Expr | List[Expr]:
    recurse = lambda sub_expr: ast_to_expr(sub_expr, path_to_a + [a], global_names)
    err = lambda msg: _ast_error(msg, path_to_a)
    ast_to_eqn = lambda stmt: _ast_to_eqn(stmt, path_to_a + [a], global_names)

    if isinstance(a, ast.Module):
        return recurse(one(a.body))

    if isinstance(a, ast.arguments):
        assert not a.vararg
        assert not a.kwonlyargs
        assert not a.kw_defaults
        assert not a.kwarg
        assert not a.defaults, err("Default arguments not implemented")
        if sys.version_info >= (3, 8):
            assert not a.posonlyargs

        return [Var(arg.arg) for arg in a.args]

    if isinstance(a, ast.FunctionDef):
        eqn = ast_to_eqn(a)
        return Let([eqn], mkTuple(eqn.vars))

    if isinstance(a, ast.Lambda):
        return Lambda(recurse(a.args), recurse(a.body), "ast:" + get_new_name())

    if isinstance(a, ast.Constant):
        return Const(a.value)

    if a is None:
        return Const(None)

    # Nodes which encode to Var
    if isinstance(a, ast.Name):
        return Var(a.id)

    # Nodes which encode to (Var or Call)
    if isinstance(a, ast.Attribute):
        val = recurse(a.value)
        return Call(Var("getattr"), [val, Const(a.attr)])

    # Nodes which encode to Call
    if isinstance(a, ast.Call):
        func = recurse(a.func)
        args = [recurse(arg) for arg in a.args]
        kwargs = {kw.arg: recurse(kw.value) for kw in a.keywords}
        return Call(func, args + kwargs_to_dict_call(kwargs))

    if isinstance(a, ast.Tuple):
        return Call(Var("g_tuple"), [recurse(e) for e in a.elts])

    if isinstance(a, ast.BinOp):
        op = _ast_op_to_Var(type(a.op))
        return Call(op, [recurse(a.left), recurse(a.right)])

    if isinstance(a, ast.UnaryOp):
        op = _ast_op_to_Var(type(a.op))
        return Call(op, [recurse(a.operand)])

    if isinstance(a, ast.Subscript):
        return Call(Var("g_subscript"), [recurse(a.value), recurse(a.slice)])

    if isinstance(a, ast.Slice):
        return Call(
            Var("g_slice"), [recurse(a.lower), recurse(a.upper), recurse(a.step)]
        )

    if isinstance(a, ast.Index):
        return recurse(a.value)

    if isinstance(a, ast.List):
        return Call(Var("g_list"), [recurse(elt) for elt in a.elts])

    if isinstance(a, ast.Expr):
        return recurse(a.value)

    # Fallthru
    assert False, f"TODO:{type(a)}"


def _ast_to_eqn(stmt, path_to_a, global_names):
    recurse = lambda sub_expr: ast_to_expr(sub_expr, path_to_a + [stmt], global_names)
    err = lambda msg: _ast_error(msg, path_to_a)
    ast_to_eqn = lambda stmt: _ast_to_eqn(stmt, path_to_a + [stmt], global_names)

    if isinstance(stmt, ast.FunctionDef):
        args = recurse(stmt.args)

        eqns = [ast_to_eqn(a) for a in stmt.body[:-1]]

        assert isinstance(stmt.body[-1], ast.Return)
        retval = recurse(stmt.body[-1].value)

        body = Let(eqns, retval)
        lam = Lambda(args, body, "ast:" + stmt.name + get_new_name())
        return Eqn([Var(stmt.name)], lam)

    if isinstance(stmt, ast.Assign):
        val = recurse(stmt.value)
        s_vars = one(stmt.targets)  # TODO: tuple of tuple
        if isinstance(s_vars, ast.Tuple):
            vars = [Var(var.id) for var in s_vars.elts]
            if len(vars) == 1:
                # This is a 1-var detuple - we don't represent it specially,
                # so pull the value off the tuple
                val = Call(Var("g_fst"), [val])
        else:
            vars = [Var(s_vars.id)]
        return Eqn(vars, val)

    if isinstance(stmt, ast.AugAssign):
        var = Var(stmt.target.id)
        val = recurse(stmt.value)
        val = Call(_ast_op_to_Var(type(stmt.op)), [var, val])
        return Eqn([var], val)

    if isinstance(stmt, ast.Expr):
        var = Var("_" + get_new_name())
        return Eqn([var], recurse(stmt.value))

    if isinstance(stmt, ast.For):
        assert not stmt.orelse
        assert not stmt.type_comment
        target = Var(stmt.target.id)
        iter = recurse(stmt.iter)
        body_eqns = [ast_to_eqn(stmt) for stmt in stmt.body]

        # TODO: this whole thing should be a visitor
        analyzer = FreeVarAnalyzer(global_names)
        for body_stmt in stmt.body:
            analyzer.visit(body_stmt)

        iteration_var_names = analyzer.bound_stack[-1] & analyzer.free
        iteration_vars = [Var(name) for name in iteration_var_names]

        if not iteration_vars:
            # We missed a write or side-effect
            raise ValueError(
                "For loop has no iteration variables.  Missed side-effect?"
            )

        lambda_args = iteration_vars + [target]
        extra_args = [
            Var(name) for name in analyzer.free - set(a.name for a in lambda_args)
        ]

        # Scan a function over leading array axes while carrying along state.
        # g_scan(f, init, xs)
        init = one(iteration_vars)
        xs = iter
        lambda_body = Let(body_eqns, one(iteration_vars))
        lambda_id = "for_body_" + get_new_name()
        scan_lambda = Lambda(lambda_args + extra_args, lambda_body, lambda_id)
        scan_lambda_var = Var(lambda_id)

        val = Let(
            [Eqn([scan_lambda_var], scan_lambda)],
            Call(Var("g_scan"), [scan_lambda_var, init, xs, *extra_args]),
        )
        return Eqn(iteration_vars, val)

    print(ast.dump(stmt))
    assert False, f"Unknown statement {stmt}"


def ex2py(name, ex):
    fvs = list(v.name for v in freevars(ex))

    filename = "tmp/ex-" + name + ".txt"
    with open(filename, "w") as f:
        print("Freevars:", *fvs, file=f)
        print(ex, file=f)

    filename = "tmp/py-" + name + ".py"
    with open(filename, "w") as f:
        print("#Freevars:", *fvs, file=f)
        print(ast.unparse(to_ast(ex, "ret")), file=f)


def expr_for(
    func: Callable,
    *callees: Callable,
    global_names: Optional[Set[str]] = None,
) -> Expr:
    """
    Create an expression that represents Callables `func` and `callees`.

    It will parse the source code of the function and its callees,
    and create a Let expression that binds func and callees to their names.
    The returned expression refers to `func`, if `callees` are not called from `func`,
    they may be later removed by dead code elimination.

    The callees should be topologically sorted, i.e. if `func` calls `callee1` and
    `callee1` calls `callee2`, then `callee1` should be before `callee2` in
    the list of callees.

    TODO: mutual recursion with "let*" or top-level-functions
    """
    import inspect
    import textwrap

    eqns = []
    for f in reversed((func,) + callees):
        a = ast.parse(textwrap.dedent(inspect.getsource(f)))
        e = ast_to_expr(a, [], global_names)
        assert e.isLet and len(e.eqns) == 1 and one(e.eqns[0].vars) == e.body
        if f == func:
            func_var = e.body
        eqns += e.eqns

    return Let(eqns, func_var)


def expr_to_python_code(e: Expr, name: str, flat_lets=True) -> str:
    as_ast = to_ast(e, name, flat_lets)
    return ast.unparse(as_ast)
