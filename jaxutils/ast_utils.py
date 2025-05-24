import ast

# ChatGPT generated - looks ok....

import ast
from typing import Set, List


class FreeVarAnalyzer(ast.NodeVisitor):
    """
    AST visitor that collects names used but not bound (free variables).
    """

    def __init__(self):
        # Stack of bound variable sets for each scope
        self.bound_stack: List[Set[str]] = [set()]
        # Collected free variable names
        self.free: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Add function name to current (outer) scope as bound
        self.bound_stack[-1].add(node.name)
        # Compute names bound in this function: its args and later assignments
        func_scope = set(arg.arg for arg in node.args.args)
        if node.args.vararg:
            func_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            func_scope.add(node.args.kwarg.arg)
        # Push new scope
        self.bound_stack.append(func_scope)
        # Process body
        self.generic_visit(node)
        # Pop scope
        self.bound_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        # Class name bound in outer scope
        self.bound_stack[-1].add(node.name)
        # New class scope (methods are definitions, do not inherit outer variables automatically)
        self.bound_stack.append(set())
        self.generic_visit(node)
        self.bound_stack.pop()

    def visit_AugAssign(self, node: ast.AugAssign):
        # use the name
        self.visit_Name(ast.Name(id=node.target.id, ctx=ast.Load()))
        # and then bind the name
        self.visit_Name(node.target)
        # Visit value (right side)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            # Binding occurrence: add to current scope
            self.bound_stack[-1].add(node.id)
        elif isinstance(node.ctx, ast.Load):
            # Usage occurrence: if not bound in any enclosing scope, it's free
            if not any(node.id in scope for scope in reversed(self.bound_stack)):
                self.free.add(node.id)

    def visit_comprehension(self, node: ast.comprehension):
        # Comprehension target binds names
        self.visit(node.target)
        # But comprehension has its own inner scope for generators
        self.visit(node.iter)
        for if_clause in node.ifs:
            self.visit(if_clause)

    def visit_Attribute(self, node: ast.Attribute):
        # Only visit the value, attribute names are not separate variables
        self.visit(node.value)

    def generic_visit(self, node):
        super().generic_visit(node)


def ast_freevars(ast_node):
    analyzer = FreeVarAnalyzer()
    analyzer.visit(ast_node)
    return analyzer.free


def test_ast_freevars():
    # Example
    code = """
x = 10
def outer(a):
    b = 1
    def inner(c):
        c = a + b + c + d
        return c
    return inner
"""

    ast_node = ast.parse(code)
    fvs = ast_freevars(ast_node)
    assert fvs == {"d"}
