from jaxutils.expr_parser import parse_expr
from jaxutils.expr import Expr, Let, Lambda, Call, Var, Const


def test_parser():
    example = """
    let
      a = let 
            a = 1;
            b = 3
          in f(a, b);
      a = let a = 2; b = lambda x, y: h(x, a, y) in b(a, 42)
      ;
      b = a
    in
      b
    """

    e = parse_expr(example)
    print("Parsed Expression:")
    print(e)

    assert e.isLet
    assert e.eqns[0].vars == [Var("a")]
    assert e.eqns[0].val.isLet
    assert e.eqns[0].val.eqns[0].vars == [Var("a")]
    assert e.eqns[0].val.eqns[0].val == Const(1)
    assert e.eqns[0].val.eqns[1].vars == [Var("b")]
    assert e.eqns[0].val.eqns[1].val == Const(3)
