from types import NoneType, ModuleType
import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence, Any

from jaxutils.expr import Expr, Var, Call, transform_postorder, shortname
from jaxutils.expr_lib import g_identity
from jaxutils.expr_eval import annotate_eval

from dataclasses import dataclass
import numpy as np

from dataclasses import replace


def annotate_with_shadow_types(
    e: Expr,
    args: Sequence[Any],
    bindings: dict[str, Any],
) -> Expr:
    """
    Evaluate `e` at `args`, recording shape and type of arrays, and type of other values

    This assumes all functions idempotent at the shape/type level (i.e. each function
    is evaluated only once).
    """
    shadow_args = [jax.tree.map(make_shadow, arg) for arg in args]

    shadow_bindings = get_shadow_bindings_for_jax()
    shadow_bindings |= bindings

    # Annotate e with the values encountered during an evaluation
    e = annotate_eval(e, shadow_args, shadow_bindings)

    # Now strip any annots that are not ShadowArrays
    @shortname("snsa")
    def strip_non_shadow_annot(e, _bindings):
        if not isinstance(e.annot, (ShadowArray, type)):
            return replace(e, annot=type(e.annot))

    e = transform_postorder(strip_non_shadow_annot, e, {})

    return e


@dataclass
class ShadowArray:
    """
    An "empty" array that behaves like a numpy array, but occupies no memory.
    It is used to represent shapes and dtypes in a way that can be used for type
    annotations and broadcasting without allocating memory, or doing the flops.
    """

    shape: tuple[int, ...]
    dtype: np.dtype | type

    def __str__(self):
        name = self.dtype.__name__ if isinstance(self.dtype, type) else self.dtype.name
        name = name.replace("float", "f").replace("int", "i").replace("complex", "z")
        return f"{name}[{'x'.join(map(str, self.shape))}]"

    def __repr__(self):
        name = self.dtype.__name__ if isinstance(self.dtype, type) else self.dtype.name
        name = name.replace("float", "f").replace("int", "i").replace("complex", "z")
        return f"{name}[{','.join(map(str, self.shape))}]"

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def T(self):
        return ShadowArray(self.shape[::-1], self.dtype)

    def __add__(self, b):
        return broadcast_shape_and_type(self, b)

    def __radd__(self, b):
        return broadcast_shape_and_type(self, b)

    def __mul__(self, b):
        return broadcast_shape_and_type(self, b)

    def __rmul__(self, b):
        return broadcast_shape_and_type(self, b)

    def __matmul__(self, b):
        assert len(self.shape) == 2
        assert len(b.shape) == 2
        assert self.shape[1] == b.shape[0]
        shape = (self.shape[0], b.shape[1])
        dtype = np.promote_types(self.dtype, b.dtype)
        return ShadowArray(shape, dtype)

    def __pow__(self, b):
        return broadcast_shape_and_type(self, b)

    def __getitem__(self, indices):
        # https://numpy.org/devdocs//user/basics.indexing.html
        # indices shorter than self.shape will return a subarray

        def dim_shape(i):
            if isinstance(indices[i], slice):
                start = indices[i].start or 0
                stop = indices[i].stop or self.shape[i]
                step = indices[i].step or 1
                return (stop - start) // step
            if isinstance(indices[i], int):
                return 1
            if isinstance(indices[i], (jnp.ndarray, ShadowArray)):
                return indices[i].shape[0]
            raise TypeError(f"Unsupported index type: {type(indices[i])}")

        if np.issubdtype(type(indices), np.integer):
            indices = (indices,)

        out_shape = tuple(dim_shape(i) for i in range(len(indices)))

        # Compare to numpy to check, but only if we are a small array
        if self.size < 10000:
            new_indexes = tuple(
                i.make_zeros() if isinstance(i, ShadowArray) else i for i in indices
            )
            assert self.make_ones()[new_indexes].shape == out_shape

        return ShadowArray(out_shape, self.dtype)

    def make_ones(self):
        return np.ones(self.shape, self.dtype)

    def make_zeros(self):
        return np.zeros(self.shape, self.dtype)


def broadcast_shape_and_type(a, b):
    a_shape = a.shape if hasattr(a, "shape") else 1
    b_shape = b.shape if hasattr(b, "shape") else 1
    out_shape = np.broadcast_shapes(a_shape, b_shape)
    a_dtype = a.dtype if hasattr(a, "dtype") else type(a)
    b_dtype = b.dtype if hasattr(b, "dtype") else type(b)
    out_dtype = np.promote_types(a_dtype, b_dtype)
    return ShadowArray(out_shape, out_dtype)


def make_shadow(a: Any) -> Any:
    """
    Make an "empty" version of object `obj`, which occupies less memory, but otherwise
    behaves as much like `obj` as it can.
    """
    if hasattr(a, "shape") and hasattr(a, "dtype"):
        return ShadowArray(a.shape, a.dtype)
    else:
        return a


sp_jax = ModuleType("sp_jax")
sp_jax.nn = ModuleType("sp_jax_nn")
sp_jax.nn.softmax = g_identity
sp_jax.nn.relu = g_identity

sp_jax.numpy = ModuleType("sp_jax_numpy")
sp_jax.numpy.log = g_identity
sp_jax.numpy.tril = g_identity
sp_jax.numpy.ones = lambda shape, dtype=jnp.float32: ShadowArray(shape, dtype)
sp_jax.numpy.diag = lambda a: ShadowArray((a.shape[0], a.shape[0]), a.dtype)


def _hstack_shadows(xs):
    if len(xs) == 0:
        return ShadowArray((0,), jnp.float32)

    dtype = xs[0].dtype
    assert all(x.dtype == dtype for x in xs[1:])

    shape = xs[0].shape
    assert all(len(x.shape) == len(shape) for x in xs[1:])

    if len(shape) == 1:
        return ShadowArray((sum(xs),), xs[0].dtype)
    if len(shape) == 2:
        assert all(x.shape[0] == shape[0] for x in xs[1:])
        return ShadowArray((shape[0], sum(x.shape[1] for x in xs)), dtype)

    raise ValueError(f"Unsupported shape for hstack: {shape}")


sp_jax.numpy.hstack = _hstack_shadows


def get_shadow_bindings_for_jax():
    """
    Get a dictionary of shadow bindings for JAX functions.
    """
    return {
        "jax": sp_jax,
        "jnp": sp_jax.numpy,
        "print": lambda *args: NoneType,
        "getattr": getattr,
    }


@transform_postorder
@shortname("san")
def strip_annotations(e: Expr, _bindings) -> Expr:
    return replace(e, annot=None)


@transform_postorder
@shortname("ga2n")
def global_getattrs_to_names(e, bindings):
    if e.isCall and e.f.isVar and e.f.name == "getattr":
        obj = e.args[0]
        attr = e.args[1]
        if obj.isVar and obj.name not in bindings:
            # It's a reference to a global variable, assume it's a module
            return Var(f"{obj.name}.{attr.val}")
        if attr.val == "T":
            if not isinstance(obj.annot, (jnp.ndarray, ShadowArray)):
                print(f"Warning: Assuming {obj.name}.T is a transpose")
            return Call(Var("transpose"), [obj])
