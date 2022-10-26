from jaxutils.PType import PType

from typing import TypeAlias


def foo(x: int) -> float:
    return x + 2.2


foo(1.1)  # Should error


Tensor = PType("Tensor")
W = PType("Width", int)
H = PType("Height", int)
C = PType("Channels", int)
N = PType("N", int)
DTypeInt = PType("DType", int)
Pair = PType("Pair")

DType = PType("DType")

F8 = PType("F8")
F16 = PType("F16")
F32 = PType("F32")


def foo2(s: F32, x: F32[...], y: F32[N]) -> F16[N]:
    return None
