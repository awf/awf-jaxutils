import argparse
from typing import Any, Dict
import sys

class Arg:
    """
    Convenient 'distributed' interface to argparse.

    Declare parameters anywhere in the program, and refer to them when needed.

    ```
    lr = Arg('lr', 'Learning rate', default=0.001)
    ```
    
    And some time later, use `lr()` in code
    ```
    optimizer = SGD(stepsize = lr())
    ```

    If `sys.argv` has not been parsed at that poinr, or if its last parse was before `lr` was 
    declared, it will be re-parsed.

    You can also summarize the changes from default using
        Arg.str()
    which can be useful for experiment names
    """
    parser = argparse.ArgumentParser(add_help=False)
    parsed_args = None
    parsed_args_at : int = -1
    all_args : Dict[str, 'Arg']= {}

    _default_sentinel = object()

    def __init__(self, flag : str, default : Any, doc: str=''):
        if flag in Arg.all_args:
            raise Exception(f'Flag {flag} used multiple times.')

        self.flag = flag
        self.default = default
        self.override = None

        Arg.all_args[flag] = self

        if isinstance(default, bool) and default == False:
            Arg.parser.add_argument('-'+flag, help=doc, 
                            default=Arg._default_sentinel, action='store_true', dest=flag)
        else:
            Arg.parser.add_argument('-'+flag, help=doc, type=type(default), 
                                        default=Arg._default_sentinel, dest=flag)

    def __call__(self):
        """
        Parse args if not done already, and return this arg's value
        """
        ns = Arg.get_parsed_args()
        return self.get_from_argparse_ns(ns)

    def peek(self):
        """
        Check in the arg list if this arg has been set, but don't complain about unrecognized arguments, and don't cache the parsed args.

        Useful when we want to act on an arg before they have all been declared (e.g. before __main__)
        """
        ns,_unused = Arg.parser.parse_known_args()
        return self.get_from_argparse_ns(ns)

    def get_from_argparse_ns(self, ns):
        arg_dict = ns.__dict__
        if self.override:
            return self.override
        if self.flag not in arg_dict or arg_dict[self.flag] is Arg._default_sentinel:
            return self.default
        return arg_dict[self.flag]


    @classmethod
    def get_parsed_args(cls, argv=None):
        if not argv:
            argv = sys.argv[1:]

        newhash = hash(tuple(sorted(cls.all_args.keys())))
        if not cls.parsed_args or cls.parsed_args_at != newhash:
            if not cls.parsed_args:
                cls.parser.add_argument('-help', action='help', help='Print this help')
            cls.parsed_args = cls.parser.parse_args(argv)
            cls.parsed_args_at = newhash
        return cls.parsed_args

    @classmethod
    def str(cls):
        """
        Return a short representation of the args that have been changed from their defaults
        """
        pas = cls.get_parsed_args().__dict__.items()
        return ' '.join(f'{k}={Arg.all_args[k]()}' for (k,v) in pas if v != Arg._default_sentinel)

    @classmethod
    def config(cls):
        """
        Return a simple dict of all known args and values
        """
        return {k: Arg.all_args[k]() for k in cls.get_parsed_args().__dict__}
