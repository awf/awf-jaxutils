from typing import List, Tuple
from jaxutils.expr import Expr, Const, Var, Call, Lambda, Let, Eqn, get_new_name

# ChatGPT generated - looks ok....


#
# === Lexer ===
#

# Token types
TOK_INT = "INT"
TOK_IDENT = "IDENT"
TOK_KEYWORD = "KEYWORD"
TOK_SYMBOL = "SYMBOL"
TOK_EOF = "EOF"

KEYWORDS = {"let", "in", "lambda"}


class Token:
    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def peek(self) -> str:
        if self.pos < self.length:
            return self.text[self.pos]
        else:
            return "\0"

    def advance(self):
        self.pos += 1

    def skip_whitespace_and_comments(self):
        while True:
            # skip spaces/tabs/newlines
            while self.peek().isspace():
                self.advance()

            # skip comments that start with '#'
            if self.peek() == "#":
                while self.peek() not in {"\n", "\0"}:
                    self.advance()
                continue
            break

    def next_token(self) -> Token:
        self.skip_whitespace_and_comments()
        ch = self.peek()
        start = self.pos

        if ch == "\0":
            return Token(TOK_EOF, "", self.pos)

        # Integer literal
        if ch.isdigit():
            num_str = ""
            while self.peek().isdigit():
                num_str += self.peek()
                self.advance()
            return Token(TOK_INT, num_str, start)

        # Identifier or keyword
        if ch.isalpha() or ch == "_":
            id_str = ""
            while self.peek().isalnum() or self.peek() == "_":
                id_str += self.peek()
                self.advance()
            if id_str in KEYWORDS:
                return Token(TOK_KEYWORD, id_str, start)
            else:
                return Token(TOK_IDENT, id_str, start)

        # Symbols: one‐char tokens = '=;(),:'
        if ch in "=;(),:":
            self.advance()
            return Token(TOK_SYMBOL, ch, start)

        # Unexpected character
        raise SyntaxError(f"Unexpected character {ch!r} at position {self.pos}")


#
# === Parser ===
#


class Parser:
    def __init__(self, text: str):
        self.lexer = Lexer(text)
        self.cur_token = self.lexer.next_token()

    def eat(self, kind: str, value: str = None):
        if self.cur_token.kind != kind:
            raise SyntaxError(
                f"Expected token kind {kind} but got {self.cur_token.kind} at pos {self.cur_token.pos}"
            )
        if value is not None and self.cur_token.value != value:
            raise SyntaxError(
                f"Expected token value {value!r} but got {self.cur_token.value!r} at pos {self.cur_token.pos}"
            )
        self.cur_token = self.lexer.next_token()

    def parse(self) -> Expr:
        expr = self.parse_expr()
        if self.cur_token.kind != TOK_EOF:
            raise SyntaxError(
                f"Unexpected token {self.cur_token} after end of expression"
            )
        return expr

    #
    # Grammar:
    #   expr       := let_expr
    #                | lambda_expr
    #                | app_expr
    #
    #   let_expr   := "let" binding_list "in" expr
    #   binding_list := binding (";" binding)*
    #   binding    := IDENT "=" expr
    #
    #   lambda_expr := "lambda" param_list ":" expr
    #   param_list := IDENT ("," IDENT)*    # zero or more parameters
    #
    #   app_expr   := primary_expr [ "(" arg_list? ")" ]*
    #   primary_expr := INT | IDENT | "(" expr ")"
    #   arg_list   := expr ("," expr)*
    #

    def parse_expr(self) -> Expr:
        if self.cur_token.kind == TOK_KEYWORD and self.cur_token.value == "let":
            return self.parse_let()
        elif self.cur_token.kind == TOK_KEYWORD and self.cur_token.value == "lambda":
            return self.parse_lambda()
        else:
            return self.parse_app()

    def parse_let(self) -> Let:
        self.eat(TOK_KEYWORD, "let")

        bindings: List[Tuple[str, Expr]] = []
        while True:
            # TODO: Handle multiple lhs
            if self.cur_token.kind != TOK_IDENT:
                raise SyntaxError(
                    f"Expected identifier in let-binding at pos {self.cur_token.pos}"
                )
            var_name = self.cur_token.value
            self.eat(TOK_IDENT)

            self.eat(TOK_SYMBOL, "=")
            rhs_expr = self.parse_expr()
            bindings.append(Eqn([Var(var_name)], rhs_expr))

            if self.cur_token.kind == TOK_SYMBOL and self.cur_token.value == ";":
                self.eat(TOK_SYMBOL, ";")
                continue
            else:
                break

        if not (self.cur_token.kind == TOK_KEYWORD and self.cur_token.value == "in"):
            raise SyntaxError(
                f"Expected 'in' or ';' after let-bindings at pos {self.cur_token.pos}"
            )
        self.eat(TOK_KEYWORD, "in")

        body_expr = self.parse_expr()
        return Let(bindings, body_expr)

    def parse_lambda(self) -> Lambda:
        # current token is 'lambda'
        self.eat(TOK_KEYWORD, "lambda")

        params: List[Var] = []
        # At least zero parameters allowed? If you want to enforce one or more,
        # check for IDENT first and error if none. Here we'll allow zero or more.
        if self.cur_token.kind == TOK_IDENT:
            # First parameter
            params.append(Var(self.cur_token.value))
            self.eat(TOK_IDENT)
            # Additional comma-separated parameters
            while self.cur_token.kind == TOK_SYMBOL and self.cur_token.value == ",":
                self.eat(TOK_SYMBOL, ",")
                if self.cur_token.kind != TOK_IDENT:
                    raise SyntaxError(
                        f"Expected identifier after ',' in lambda params at pos {self.cur_token.pos}"
                    )
                params.append(Var(self.cur_token.value))
                self.eat(TOK_IDENT)

        # Expect colon
        self.eat(TOK_SYMBOL, ":")
        body_expr = self.parse_expr()
        return Lambda(params, body_expr, id="exprparser" + get_new_name())

    def parse_app(self) -> Expr:
        expr = self.parse_primary()
        while self.cur_token.kind == TOK_SYMBOL and self.cur_token.value == "(":
            expr = self.parse_call(expr)
        return expr

    def parse_call(self, func_expr: Expr) -> Call:
        self.eat(TOK_SYMBOL, "(")
        args: List[Expr] = []

        if not (self.cur_token.kind == TOK_SYMBOL and self.cur_token.value == ")"):
            args.append(self.parse_expr())
            while self.cur_token.kind == TOK_SYMBOL and self.cur_token.value == ",":
                self.eat(TOK_SYMBOL, ",")
                args.append(self.parse_expr())

        self.eat(TOK_SYMBOL, ")")
        return Call(func_expr, args)

    def parse_primary(self) -> Expr:
        tok = self.cur_token

        if tok.kind == TOK_INT:
            val = int(tok.value)
            self.eat(TOK_INT)
            return Const(val)

        if tok.kind == TOK_IDENT:
            name = tok.value
            self.eat(TOK_IDENT)
            return Var(name)

        if tok.kind == TOK_SYMBOL and tok.value == "(":
            self.eat(TOK_SYMBOL, "(")
            inner = self.parse_expr()
            self.eat(TOK_SYMBOL, ")")
            return inner

        raise SyntaxError(
            f"Unexpected token {tok} in primary expression at pos {tok.pos}"
        )


def parse_expr(text: str) -> Expr:
    """
    Parse a string into an Expr object.
    """
    parser = Parser(text)
    return parser.parse()
