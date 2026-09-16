"""Test environment setup for ComfyUI_Junnan_Tools.

The project root conftest already stubs comfy_api.latest. This conftest
adds stubs for comfy.model_management and torch so tests run without a
full ComfyUI / torch install. Both stubs are pure-Python (no numpy / torch
dependency) and cover only the surface used by _build_bbox_masks.

NOTE: stubs verify the algorithm logic, not torch-specific behavior. For
production-grade verification, run the suite in a Python with real torch
+ ComfyUI installed (per project CLAUDE.md testing prerequisites).
"""
import sys
import types


_FLOAT32 = "fake_float32"


def _infer_shape(lst):
    shape = []
    cur = lst
    while isinstance(cur, list):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(shape)


def _all_recursive(data):
    if isinstance(data, list):
        return all(_all_recursive(d) for d in data)
    return bool(data)


def _eq_recursive(a, b):
    if isinstance(a, list):
        return [_eq_recursive(x, y) for x, y in zip(a, b)]
    return 1 if a == b else 0


def _eq_scalar_recursive(a, scalar):
    if isinstance(a, list):
        return [_eq_scalar_recursive(x, scalar) for x in a]
    return 1 if a == scalar else 0


class _FakeTensor:
    """Minimal stand-in for torch.Tensor used by tests."""

    def __init__(self, data, dtype=None, shape=None):
        self._data = data
        self.dtype = dtype if dtype is not None else _FLOAT32
        self.shape = tuple(shape) if shape is not None else _infer_shape(data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            cur = self._data
            first_slice_seen = False
            for k in key:
                if isinstance(k, int):
                    if (
                        isinstance(cur, list)
                        and len(cur) > 0
                        and isinstance(cur[0], list)
                        and first_slice_seen
                    ):
                        cur = [row[k] for row in cur]
                    else:
                        cur = cur[k]
                elif isinstance(k, slice):
                    if not first_slice_seen:
                        cur = cur[k]
                        first_slice_seen = True
                    else:
                        cur = [row[k] for row in cur]
            if isinstance(cur, list):
                return _FakeTensor(cur)
            return cur
        item = self._data[key]
        if isinstance(item, list):
            return _FakeTensor(item)
        return item

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row_slice, col_slice = key
            if isinstance(row_slice, slice) and isinstance(col_slice, slice):
                for row in self._data[row_slice]:
                    for j in range(*col_slice.indices(len(row))):
                        row[j] = value
                return
        raise NotImplementedError(
            f"fake stub does not support __setitem__ key={key!r}"
        )

    def __eq__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(_eq_recursive(self._data, other._data))
        return _FakeTensor(_eq_scalar_recursive(self._data, other))

    def all(self):
        return _all_recursive(self._data)

    def __repr__(self):
        return f"_FakeTensor(shape={self.shape})"


def _fake_zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)

    def build(s):
        if len(s) == 0:
            return 0.0
        if len(s) == 1:
            return [0.0] * s[0]
        return [build(s[1:]) for _ in range(s[0])]

    return _FakeTensor(build(shape), dtype=dtype)


def _fake_stack(tensors, dim=0):
    if dim != 0:
        raise NotImplementedError("stub stack only supports dim=0")
    stacked_data = [t._data for t in tensors]
    return _FakeTensor(stacked_data)


def _fake_equal(a, b):
    return _all_recursive(_eq_recursive(a._data, b._data))


class _TorchStub:
    float32 = _FLOAT32
    float64 = "fake_float64"
    zeros = staticmethod(_fake_zeros)
    stack = staticmethod(_fake_stack)
    equal = staticmethod(_fake_equal)
    tensor = staticmethod(lambda data, dtype=None: _FakeTensor(data, dtype=dtype))


def _ensure_comfy_model_management():
    try:
        import comfy.model_management  # noqa: F401
        return
    except ImportError:
        pass

    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is None:
        comfy_pkg = types.ModuleType("comfy")
        sys.modules["comfy"] = comfy_pkg

    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.intermediate_device = lambda: "cpu"
    comfy_pkg.model_management = mm_mod
    sys.modules["comfy.model_management"] = mm_mod


def _ensure_torch():
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        sys.modules["torch"] = _TorchStub()


_ensure_comfy_model_management()
_ensure_torch()
