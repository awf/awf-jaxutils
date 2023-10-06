from dataclasses import dataclass
from beartype import beartype
from beartype.typing import List, Set, Any, Tuple, Type, List, Callable
from pprint import pprint

name_id = 0


def get_new_name():
    global name_id
    name_id += 1
    return f"{name_id:02d}"


## Declare Expr classes


@beartype
class Expr:
    """
    A classic tiny-Expr library.

    These are the subclasses

        Const:   Constant
        Var:     Variable name
        Let:     Let vars = val in Expr
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
class Let(Expr):
    vars: List[Var]
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

def transform(transformer: Callable[[Expr], Expr], e: Expr):
    recurse = lambda e: transform(transformer, e)

    # Recurse into children
    if e.isLet:
        new_val = recurse(e.val)
        new_body = recurse(e.body)
        e = Let(e.vars, new_val, new_body)

    if e.isLambda:
        e = Lambda(e.args, recurse(e.body))

    if e.isCall:
        new_f = recurse(e.f)
        new_args = [recurse(arg) for arg in e.args]
        e = Call(new_f, new_args)

    # And pass self to the transformer, with updated children
    return transformer(e) or e



def freevars(e: Expr) -> set[Var]:
    if e.isConst:
        return set()

    if e.isVar:
        return {e}

    if e.isLet:
        fv_body = set.difference(freevars(e.body), set(e.vars))
        fv_val = freevars(e.val)
        return set.union(fv_val, fv_body)

    if e.isLambda:
        return set.difference(freevars(e.body), e.args)

    if e.isCall:
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
        [x, y], Let([z], Call(v_add, [x, Const(3.3)]), Call(v_mul, [call_lam, y]))
    )
    e = Let([foo], foo_lam, Call(foo, [Const(1.1), Const(2.2)]))
    return e


def test_basic():
    foo, x, y, z = mkvars("foo, x, y, z")
    v_sin, v_add, v_mul = mkvars("sin, add, mul")

    e = _make_e()

    pprint(e)
    assert freevars(e) == {v_sin, v_add, v_mul}
    assert freevars(e.val) == {v_sin, v_add, v_mul}
    assert freevars(e.val.body) == {x, y, v_sin, v_add, v_mul}


def visit(e: Expr, f: Callable[[Expr], Any]):
    # Call f on e
    f(e)

    # And recurse into children
    if e.isLet:
        visit(e.val, f)
        visit(e.body, f)

    if e.isLambda:
        visit(e.body, f)

    if e.isCall:
        visit(e.f, f)
        for arg in e.args:
            visit(arg, f)


def test_visit():
    e = _make_e()
    visit(e, lambda x: print(type(x)))


def let_to_lambda(e: Expr) -> Expr:
    """
    let x = val in body
    ->
    call(lambda x: body, val)
    """
    if e.isLet:
        val = transform(let_to_lambda, e.val)
        body = transform(let_to_lambda, e.body)
        return Call(Lambda(e.vars, body), [val])


def test_let_to_lambda():
    e = _make_e()
    l = transform(let_to_lambda, e)

    def check(e):
        assert not e.isLet

    visit(l, check)

def uniquify_names(e : Expr):
    translations = {v.name:v.name for v in freevars(e)}
    return uniquify_names_aux(e, translations)

def uniquify_names_aux(e : Expr, translations) -> Expr:
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
        return Var(translations.get(e.name, e.name))
        
    if e.isCall:
        return Call(
            uniquify_names_aux(e.f, translations),
             [uniquify_names_aux(arg, translations) for arg in e.args]
        )
    
    assert e.isLet or e.isLambda

    if e.isLet:
        vars = e.vars
    if e.isLambda:
        vars = e.args

    new_translations = {**translations}
    new_vars = []
    for var in [var.name for var in vars]:
        # This var has just come in scope.
        # If its name is already in translations, it is clashing,
        # so in the body of this let, it will need a new name
        if var in new_translations:
            newname = 't_' + get_new_name()
        else:
            newname = var
        new_translations[var] = newname
        new_vars.append(Var(newname))

    if e.isLet:
        new_val = uniquify_names_aux(e.val, translations)
        new_body = uniquify_names_aux(e.body, new_translations)
        return Let(new_vars, new_val, new_body)

    if e.isLambda:
        new_body = uniquify_names_aux(e.body, new_translations)
        return Lambda(new_vars, new_body)

def test_uniquify_names():
    a,b,c,d = mkvars('a,b,c,d')
    

    e = Let([a], a, Let([a], b, Call(b, [a, b])))
    pprint(e)
    out = uniquify_names(e)
    pprint(out)
    assert out.vars[0] != a
    assert out.val == a
    assert out.body.vars[0] != a
    assert out.body.vars[0] != out.vars[0]

######### AST

import ast

def to_ast_FunctionDef(name, args, body):
    return ast.FunctionDef(name=name, args=args, body=body, decorator_list=[], lineno=0)


def to_ast(e, name):
    e = uniquify_names(e)
    assignments = []
    expr = to_ast_aux(e, assignments)
    assignments += [ast.Return(expr)]
    args = to_ast_args(list(freevars(e)))
    a = to_ast_FunctionDef(name, args, assignments)
    a = ast.Module(body=[a], type_ignores=[])
    ast.fix_missing_locations(a)
    return a


@beartype
def to_ast_args(vars: List[Var]) -> ast.arguments:
    aargs = [ast.arg(v.name) for v in vars]
    return ast.arguments(
        args=aargs, defaults=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[]
    )



def to_ast_aux(e, assignments):
    if e.isConst:
        return ast.Constant(value=e.val)

    if e.isVar:
        return ast.Name(e.name, ast.Load())

    if e.isLet:
        avars = [ast.Name(var.name, ast.Store()) for var in e.vars]
        if len(avars) > 1:
            avars = [ast.Tuple(avars, ast.Store())]

        aval = to_ast_aux(e.val, assignments)
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        assign = ast.Assign(targets=avars, value=aval)
        assignments += [assign]
        assignments += inner_assignments
        return abody

    if e.isLambda:
        inner_assignments = []
        abody = to_ast_aux(e.body, inner_assignments)
        inner_assignments += [ast.Return(abody)]
        name = 'f' + get_new_name()
        aargs = to_ast_args(e.args)
        fdef = to_ast_FunctionDef(name, aargs, inner_assignments)
        assignments += [fdef]
        return to_ast_aux(Var(name), None)

    if e.isCall:
        f = to_ast_aux(e.f, assignments)
        args = [to_ast_aux(arg, assignments) for arg in e.args]
        return ast.Call(func=f, args=args, keywords=[])

    assert False

def foo():
    a,b = 123


def test_ast():
    # To see what bits of AST should be
    # print(ast.dump(ast.parse(inspect.getsource(foo)), indent=2))

    a,b = mkvars("a,b")
    e = Let([a,b], Const(123), Const(234))
    pprint(e)
    a = to_ast(e, "e")
    print(ast.unparse(a))

    e = _make_e()
    a = to_ast(e, "e")
    print(ast.unparse(a))

    # code = compile(a, 'bar', 'exec')
    # exec(code)
