class PType:
    """
    PType simply provides *syntax* for playing with type annotations.

    This allows one to write code like this

    ```
    def foo(s : F32, x: F32[...], y: F32[N]) -> F16[N]:
      ...
    ```

    but this is purely for documentation -- there's no actual checking,
    and it's not even mypy compatible.

    """

    def __init__(self, path, type=None):
        self.path = path
        self.type = type

    def __getitem__(self, i):
        return PType([self.path, i])

    def __repr__(self):
        if self.type:
            type_str = f", {self.type}"
        else:
            type_str = ""
        return f"PType({self.path.__repr__()}{type_str})"

    def __and__(self, that):
        if self.type and that.type:
            assert self.type == that.type

        type = self.type if self.type else that.type

        return PType([self.path, that.path], type)


def Tree(*args):
    return None


F8 = PType("F8")
F16 = PType("F16")
F32 = PType("F32")
Tuple = PType("Tuple")
