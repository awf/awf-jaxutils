from dataclasses import dataclass
from beartype import beartype
from beartype.typing import List, Set, Any, Tuple, Type, List, Callable


@beartype
def setminus(s: set, v: Any):
    """
    return s - {v}
    """
    return set.difference(s, set((v,)))


assert setminus(set((1, 3, 6)), 3) == set((1, 6))

## Declare Expr classes


@beartype
class Expr:
    """
    A classic tiny-Expr library.

    These are the subclasses

        Const:   Constant
        Var:     Variable name
        Let:     Let var = val in Expr
        Lambda:  Lambda vars . body
        Call:    Call (func: Expr) (args : List[Expr])

    All operators (add, mul, etc) are just calls

    Methods:

      freevars(e: Expr) -> List[Var]
      let_to_lambda(e: Expr) -> Expr   # Remove all "Let" nodes, replacing with Lambdas

    """

    @property
    def ty(self):
        """
        Return type of this object, assumed a sublcass of Expr,
        from the fixed list Const, Var, Let, Lambda, Call.

        Primary use is in writing `e.ty is Var` rather than `isinstance(e, Var)`
        """
        for ty in (Const, Var, Let, Lambda, Call):
            if isinstance(self, ty):
                return ty
        assert f"Bad type {self}"


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
class Let(Expr):
    var: Var
    val: Expr
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


def freevars(e: Expr) -> set[Var]:
    if e.ty is Const:
        return set()

    if e.ty is Var:
        return {e}

    if e.ty is Let:
        fv_body = setminus(freevars(e.body), e.var)
        fv_val = freevars(e.val)
        return set.union(fv_val, fv_body)

    if e.ty is Lambda:
        return set.difference(freevars(e.body), e.args)

    if e.ty is Call:
        return set.union(freevars(e.f), *(freevars(arg) for arg in e.args))

    assert False


def _make_e():
    # Make an expr for testing
    import math
    import operator

    foo, x, y, z = mkvars("foo, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    a_lam = Lambda([z], Call(v_sin, [z]))
    call_lam = Call(a_lam, [x])
    foo_lam = Lambda(
        [x, y], Let(z, Call(v_add, [x, Const(3.3)]), Call(v_mul, [call_lam, y]))
    )
    e = Let(foo, foo_lam, Call(foo, [Const(1.1), Const(2.2)]))
    return e


def test_basic():
    foo, x, y, z = mkvars("foo, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    e = _make_e()

    from prettyprinter import cpprint

    cpprint(e)
    assert freevars(e) == {v_sin, v_add, v_mul}
    assert freevars(e.val) == {v_sin, v_add, v_mul}
    assert freevars(e.val.body) == {x, y, v_sin, v_add, v_mul}


def visit(e: Expr, f: Callable[[Expr], Any]):
    # Call f on e
    f(e)

    # And recurse into children
    if e.ty is Let:
        visit(e.val, f)
        visit(e.body, f)

    if e.ty is Lambda:
        visit(e.body, f)

    if e.ty is Call:
        visit(e.f, f)
        for arg in e.args:
            visit(arg, f)


def test_visit():
    e = _make_e()
    visit(e, lambda x: print(x.ty))


def let_to_lambda(e: Expr) -> Expr:
    """
    let x = val in body
    ->
    call(lambda x: body, val)
    """
    if e.ty in (Const, Var):
        return e

    if e.ty is Let:
        val = let_to_lambda(e.val)
        body = let_to_lambda(e.body)
        return Call(Lambda([e.var], body), [val])

    if e.ty is Lambda:
        body = let_to_lambda(e.body)
        return Lambda(e.args, body)

    if e.ty is Call:
        f = let_to_lambda(e.f)
        args = [let_to_lambda(arg) for arg in e.args]
        return Call(f, args)

    assert False, str(e)


def test_let_to_lambda():
    e = _make_e()
    l = let_to_lambda(e)

    def check(e):
        assert not e.ty is Let

    visit(l, check)


######### AST

import ast


def to_ast(e, name):
    assignments = []
    expr = to_ast_aux(e, assignments)
    assignments += [ast.Return(expr)]
    args = to_ast_args(list(freevars(e)))
    a = to_ast_fndef(name, args, assignments)
    a = ast.Module(body=[a], type_ignores=[])
    ast.fix_missing_locations(a)
    return a


@beartype
def to_ast_args(vars: List[Var]) -> ast.arguments:
    aargs = [ast.arg(v.name) for v in vars]
    return ast.arguments(
        args=aargs, defaults=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[]
    )


def to_ast_fndef(name, args, body):
    return ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], lineno=0)


name_id = 0


def get_new_name():
    global name_id
    name_id += 1
    return f"f{name_id:02d}"


def to_ast_aux(e, assignments):
    if e.ty is Const:
        return ast.Constant(value=e.val)

    if e.ty is Var:
        return ast.Name(e.name, ast.Load())

    if e.ty is Let:
        print('Nested assignments may be flaky - use "let_to_lambda"')
        avar = ast.Name(e.var.name, ast.Store())
        aval = to_ast_aux(e.val, assignments)
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        assign = ast.Assign(targets=[avar], value=aval)
        assignments += [assign]
        assignments += inner_assignments
        return abody

    if e.ty is Lambda:
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        inner_assignments += [ast.Return(abody)]
        name = get_new_name()
        aargs = to_ast_args(e.args)
        fdef = to_ast_fndef(name, aargs, inner_assignments)
        assignments += [fdef]
        return to_ast_aux(Var(name), None)

    if e.ty is Call:
        f = to_ast_aux(e.f, assignments)
        args = [to_ast_aux(arg, assignments) for arg in e.args]
        return ast.Call(func=f, args=args, keywords=[])

    assert False


def test_ast():
    import astunparse

    e = _make_e()
    a = to_ast(e, "e")
    print(ast.unparse(a))

    # code = compile(a, 'bar', 'exec')
    # exec(code)
