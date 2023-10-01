import types
import json
import jax

import numbers


def is_simple_type(x):
    return isinstance(x, (numbers.Number, bool, str))


@jax.tree_util.register_pytree_node_class
class ParamsDict(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tree_flatten(self):
        return jax.tree_flatten(
            self.__dict__, lambda a: a is not self.__dict__
        )  # only flatten one step

    @classmethod
    def tree_unflatten(cls, aux, values):
        return ParamsDict(**jax.tree_unflatten(aux, values))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __hash__(self):
        # Should overload setattr to warn if setattr is called after hash has been computed
        return hash(tuple(hash(x) for (_, x) in self.__dict__.items()))

    def print(self, path=""):
        for k, v in self.items(path):
            print(k + ":", v)

    @classmethod
    def labels_aux(cls, path, obj):
        if isinstance(obj, (list, tuple)) and any(not is_simple_type(x) for x in obj):
            for i, vi in enumerate(obj):
                yield from cls.labels_aux(f"{path}[{i}]", vi)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.labels_aux(path + "/" + k, v)
        elif isinstance(obj, ParamsDict):
            yield from cls.labels_aux(path, obj.__dict__)
        else:
            yield (path, obj)

    def items(self, path=""):
        yield from self.labels_aux(path, self)


if __name__ == "__main__":
    p = ParamsDict()
    p.a = ParamsDict()
    p.a.a1 = "a1"
    p.a.a2 = 222
    a32 = ParamsDict()
    a32.a321 = "a321"
    p.a.a3 = [(2, 3, "l"), a32, "a33_str"]
    p.b = {"fred": 3, "una": 7}

    p.print("test")

    for k, v in p.items():
        print(k, v)
