import argparse
from typing import Any

from Arg import Arg

def test_Arg():
    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)

    Arg.get_parsed_args('-lr 0.125 -beta2 4'.split())

    print(Arg.str())

    assert lr() == 0.125
    assert beta1() == 0.9
    assert beta2() == 4.0

    assert Arg.str() == 'lr=0.125 beta2=4.0'

