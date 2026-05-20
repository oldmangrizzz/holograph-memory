"""Loader for the optional Rust accelerator extension.

If `holograph._native` (built by the rust-kernel/ crate via maturin) is
importable, this module exposes the Rust-backed RealKernel and TernaryKernel
that match the pure-Python interface exactly.  Otherwise the loader is a
no-op and `make_kernel` falls back to the pure-Python implementation.

Build the accelerator with:

    cd rust-kernel
    maturin develop --release

Then in Python the runtime picks it up automatically:

    >>> from holograph.hdc.kernel import make_kernel
    >>> k = make_kernel("ternary", dim=16000)
    >>> type(k).__name__
    'TernaryKernelRustAdapter'   # if compiled
    # else 'TernaryKernel'       # pure-Python fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - environmental
    from holograph import _native  # type: ignore[attr-defined]
    HAS_RUST = True
except Exception:  # pragma: no cover
    _native = None  # type: ignore[assignment]
    HAS_RUST = False


# ---------------------------------------------------------------------------
# Adapters that wrap the Rust classes to expose the same Python-side API
# ---------------------------------------------------------------------------


@dataclass
class RealKernelRustAdapter:
    dim: int = 10000
    name: str = field(default="real", init=False)
    dtype: np.dtype = field(default=np.dtype(np.float32), init=False)
    _backend: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not HAS_RUST:  # pragma: no cover - defensive
            raise RuntimeError("Rust accelerator not available")
        self._backend = _native.RealKernelRs(self.dim)  # type: ignore[union-attr]

    def random_basis(self, n_rows: int, seed: Optional[int] = None) -> np.ndarray:
        return np.asarray(self._backend.random_basis(n_rows, int(seed or 0)))

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    def encode_scalar(self, scaled: float, embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.encode_scalar(float(scaled),
                                                       embedding.astype(np.float32),
                                                       basis.astype(np.float32)))

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.bind(a.astype(np.float32, copy=False),
                                              b.astype(np.float32, copy=False)))

    def bundle(self, vs: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(self._backend.bundle([v.astype(np.float32, copy=False) for v in vs]))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self._backend.similarity(a.astype(np.float32, copy=False),
                                                b.astype(np.float32, copy=False)))

    def pack(self, hv: np.ndarray) -> bytes:
        return bytes(self._backend.pack(hv.astype(np.float32, copy=False)))

    def unpack(self, blob: bytes) -> np.ndarray:
        return np.asarray(self._backend.unpack(blob))

    def quantize(self, real_hv: np.ndarray) -> np.ndarray:
        return real_hv.astype(self.dtype, copy=False)


@dataclass
class TernaryKernelRustAdapter:
    dim: int = 16000
    deadband: float = 0.0
    name: str = field(default="ternary", init=False)
    dtype: np.dtype = field(default=np.dtype(np.int8), init=False)
    _backend: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not HAS_RUST:  # pragma: no cover - defensive
            raise RuntimeError("Rust accelerator not available")
        self._backend = _native.TernaryKernelRs(self.dim, float(self.deadband))  # type: ignore[union-attr]

    def random_basis(self, n_rows: int, seed: Optional[int] = None) -> np.ndarray:
        return np.asarray(self._backend.random_basis(n_rows, int(seed or 0)))

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.int8)

    def encode_scalar(self, scaled: float, embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.encode_scalar(float(scaled),
                                                       embedding.astype(np.float32),
                                                       basis.astype(np.float32)))

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.bind(a.astype(np.int8, copy=False),
                                              b.astype(np.int8, copy=False)))

    def bundle(self, vs: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(self._backend.bundle([v.astype(np.int8, copy=False) for v in vs]))

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self._backend.similarity(a.astype(np.int8, copy=False),
                                                b.astype(np.int8, copy=False)))

    def quantize(self, real_hv: np.ndarray) -> np.ndarray:
        return np.asarray(self._backend.quantize(real_hv.astype(np.float32, copy=False)))

    def pack(self, hv: np.ndarray) -> bytes:
        return bytes(self._backend.pack(hv.astype(np.int8, copy=False)))

    def unpack(self, blob: bytes) -> np.ndarray:
        return np.asarray(self._backend.unpack(blob))


__all__ = ["HAS_RUST", "RealKernelRustAdapter", "TernaryKernelRustAdapter"]
