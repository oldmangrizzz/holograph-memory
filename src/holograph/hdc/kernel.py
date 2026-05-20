"""HDC kernels: Real (float32 + tanh) and Ternary (trits with STE).

Both kernels share an interface so the rest of the system is backend-agnostic.
The kernel owns:
    * dimensionality D
    * a fixed random projection basis B
    * the three core operators: encode, bind, bundle, similarity
    * a pack/unpack pair for storage

The Ternary kernel additionally exposes a quantize() entry point used at the
seam between the GNN reader (which lives in continuous space) and the HDC
composition layer (which operates on trits).

Design notes
------------
1. Hypervectors are stored as plain numpy arrays in their canonical dtype
   (float32 for Real, int8 for Ternary). External APIs are NumPy-first.

2. The Ternary kernel uses an additive bundling rule:
       bundle = sign_deadband(sum(vs))
   with deadband proportional to sqrt(len(vs)) — half the expected random-walk
   magnitude. The deadband zeros emerge as an abstention channel.

3. Bind in ternary is the trit Hadamard product (sign-preserving multiplication
   of trits), which collapses to element-wise int multiplication.

4. Storage: ternary HVs pack into 2 bits per dimension (4 trits per byte).
   Encoding: 00 -> 0, 01 -> +1, 10 -> -1, 11 reserved (treated as 0 on unpack).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class HDCKernel(ABC):
    """Backend-agnostic HDC kernel.

    All hypervector arrays returned by a kernel are 1-D numpy arrays of the
    kernel's canonical dtype.  Callers MUST treat them as immutable.
    """

    name: str = "abstract"
    dtype: np.dtype = np.float32
    dim: int = 0

    # ---- construction --------------------------------------------------

    @abstractmethod
    def random_basis(self, n_rows: int, seed: Optional[int] = None) -> np.ndarray:
        """Return a (n_rows, dim) random basis matrix in the kernel's dtype."""

    @abstractmethod
    def zeros(self) -> np.ndarray:
        """Return a zero hypervector."""

    @abstractmethod
    def encode_scalar(self, scaled: float, embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Encode a scalar in [-1, 1] using a (low-dim) trainable embedding and the basis.

        Implements the PSP-HDC step:
                h = phi( (scaled * e)^T B )
        where phi is the kernel-specific nonlinearity (tanh for Real, sign-with-
        deadband for Ternary, applied after a real-valued accumulation).
        """

    # ---- core operators ------------------------------------------------

    @abstractmethod
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise binding."""

    @abstractmethod
    def bundle(self, vs: Sequence[np.ndarray]) -> np.ndarray:
        """Aggregate evidence across a set of hypervectors."""

    @abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similarity in [-1, 1].

        For Real: cosine similarity on unit-normalized hypervectors.
        For Ternary: (matches - mismatches) / max(active_dims, 1)
        """

    # ---- storage helpers ----------------------------------------------

    @abstractmethod
    def pack(self, hv: np.ndarray) -> bytes:
        """Serialize a hypervector to a compact byte string for storage."""

    @abstractmethod
    def unpack(self, blob: bytes) -> np.ndarray:
        """Reverse of pack()."""

    # ---- ternary-only helpers (default no-op so callers can branch) ---

    def quantize(self, real_hv: np.ndarray) -> np.ndarray:
        """Quantize a real-valued hypervector to the kernel's representation.

        Real kernel: identity. Ternary kernel: sign-with-deadband.
        """
        return real_hv.astype(self.dtype, copy=False)


# ---------------------------------------------------------------------------
# Real-valued kernel (float32 + tanh + cosine)
# ---------------------------------------------------------------------------


@dataclass
class RealKernel(HDCKernel):
    """Float32 hypervectors with tanh nonlinearity and cosine similarity.

    Faithful to the encoder formulation in Ge et al.'s PSP-HDC paper:
        h_{i,j} = tanh( (x_tilde_{i,j} * e_j)^T B )
    """

    dim: int = 10000
    name: str = field(default="real", init=False)
    dtype: np.dtype = field(default=np.dtype(np.float32), init=False)

    def random_basis(self, n_rows: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Row-normalised Gaussian rows; preserves near-orthogonality.
        b = rng.standard_normal((n_rows, self.dim)).astype(np.float32)
        norms = np.linalg.norm(b, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return b / norms

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    def encode_scalar(self, scaled: float, embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
        # Per PSP-HDC: project (scaled * e) through B then apply tanh elementwise.
        proj = (float(scaled) * embedding.astype(np.float32)) @ basis
        return np.tanh(proj).astype(np.float32)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.float32) * b.astype(np.float32)).astype(np.float32)

    def bundle(self, vs: Sequence[np.ndarray]) -> np.ndarray:
        if len(vs) == 0:
            return self.zeros()
        acc = np.zeros(self.dim, dtype=np.float32)
        for v in vs:
            acc = acc + v.astype(np.float32, copy=False)
        n = float(np.linalg.norm(acc))
        if n > 0.0:
            acc = acc / n
        return acc.astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # Defensive: sanitize non-finite inputs to zero so a malformed or
        # NaN/Inf-poisoned hypervector cannot propagate non-finite similarities
        # downstream into prototype retrieval and attribution.
        af = a.astype(np.float32, copy=False)
        bf = b.astype(np.float32, copy=False)
        if not np.isfinite(af).all() or not np.isfinite(bf).all():
            af = np.where(np.isfinite(af), af, 0.0)
            bf = np.where(np.isfinite(bf), bf, 0.0)
        na = float(np.linalg.norm(af))
        nb = float(np.linalg.norm(bf))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(af, bf) / (na * nb))

    def pack(self, hv: np.ndarray) -> bytes:
        return hv.astype(np.float32, copy=False).tobytes()

    def unpack(self, blob: bytes) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != self.dim:
            raise ValueError(f"RealKernel.unpack: expected {self.dim} floats, got {arr.size}")
        return arr.copy()


# ---------------------------------------------------------------------------
# Ternary kernel (trits + STE + deadband bundling + bit-packing)
# ---------------------------------------------------------------------------


@dataclass
class TernaryKernel(HDCKernel):
    """Trits {-1, 0, +1} with sign-with-deadband nonlinearity.

    Storage is bit-packed at 2 bits per dim (4 trits / byte).  Operations work
    on int8 unpacked form for clarity; the Rust accelerator (if compiled in)
    can use the packed form directly.

    The encoder uses a straight-through estimator: the forward pass quantizes
    via sign-with-deadband; the gradient (computed externally, e.g. by the
    ScalarEncoder training loop) flows through the pre-quantization values.
    """

    dim: int = 16000
    deadband: float = 0.0      # |x| <= deadband -> 0 at encode time
    name: str = field(default="ternary", init=False)
    dtype: np.dtype = field(default=np.dtype(np.int8), init=False)

    # ---- helpers -------------------------------------------------------

    def _sign_dead(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Sign with deadband: |x| <= threshold -> 0, else sign(x)."""
        t = max(threshold, self.deadband)
        out = np.where(np.abs(x) <= t, 0, np.sign(x))
        return out.astype(np.int8)

    # ---- construction --------------------------------------------------

    def random_basis(self, n_rows: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Real-valued Gaussian basis used during the encode projection;
        # the deadband-sign is applied AFTER projection during encode_scalar.
        b = rng.standard_normal((n_rows, self.dim)).astype(np.float32)
        norms = np.linalg.norm(b, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return b / norms

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.int8)

    def encode_scalar(self, scaled: float, embedding: np.ndarray, basis: np.ndarray) -> np.ndarray:
        proj = (float(scaled) * embedding.astype(np.float32)) @ basis  # float32
        return self._sign_dead(proj, threshold=0.0)

    # ---- core operators ------------------------------------------------

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Trit Hadamard via integer multiplication.
        return (a.astype(np.int8) * b.astype(np.int8)).astype(np.int8)

    def bundle(self, vs: Sequence[np.ndarray]) -> np.ndarray:
        """Ternary bundle with deadband proportional to sqrt(len(vs)).

        Without a deadband, summing many trits always saturates ±1 in nearly
        every position and discards the abstention channel.  We threshold at
        half the expected random-walk magnitude (~ 0.5 * sqrt(n)), giving an
        adaptive cutoff that preserves zeros where evidence is genuinely mixed.
        """
        if len(vs) == 0:
            return self.zeros()
        acc = np.zeros(self.dim, dtype=np.int32)
        for v in vs:
            acc = acc + v.astype(np.int32, copy=False)
        n = len(vs)
        threshold = 0.5 * float(np.sqrt(max(n, 1)))
        return self._sign_dead(acc.astype(np.float32), threshold=threshold)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ai = a.astype(np.int8, copy=False)
        bi = b.astype(np.int8, copy=False)
        active = (ai != 0) & (bi != 0)
        n_active = int(active.sum())
        if n_active == 0:
            return 0.0
        matches = int(((ai == bi) & active).sum())
        mismatches = int(((ai != bi) & active).sum())
        return (matches - mismatches) / float(n_active)

    def quantize(self, real_hv: np.ndarray) -> np.ndarray:
        return self._sign_dead(real_hv.astype(np.float32, copy=False), threshold=0.0)

    # ---- storage: 2 bits per trit -------------------------------------
    #   trit  0  ->  00
    #   trit +1 ->  01
    #   trit -1 ->  10
    # Reserved 11 is decoded as 0 (defensive).

    def pack(self, hv: np.ndarray) -> bytes:
        if hv.shape[0] != self.dim:
            raise ValueError(f"pack: expected dim={self.dim}, got {hv.shape[0]}")
        codes = np.zeros(self.dim, dtype=np.uint8)
        codes[hv == 1] = 1
        codes[hv == -1] = 2
        # Pad to a multiple of 4 for 4-trits-per-byte packing.
        pad = (-self.dim) % 4
        if pad:
            codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
        codes = codes.reshape(-1, 4)
        packed = (codes[:, 0]
                  | (codes[:, 1] << 2)
                  | (codes[:, 2] << 4)
                  | (codes[:, 3] << 6)).astype(np.uint8)
        return packed.tobytes()

    def unpack(self, blob: bytes) -> np.ndarray:
        packed = np.frombuffer(blob, dtype=np.uint8)
        c0 = packed & 0b11
        c1 = (packed >> 2) & 0b11
        c2 = (packed >> 4) & 0b11
        c3 = (packed >> 6) & 0b11
        codes = np.stack([c0, c1, c2, c3], axis=1).reshape(-1)[: self.dim]
        out = np.zeros(self.dim, dtype=np.int8)
        out[codes == 1] = 1
        out[codes == 2] = -1
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_kernel(kind: Literal["real", "ternary"] = "real",
                dim: Optional[int] = None,
                deadband: float = 0.0,
                prefer_native: bool = True) -> HDCKernel:
    """Construct a kernel by name.

    Args:
        kind: "real" or "ternary".
        dim: override the default hypervector dimensionality.
        deadband: ternary-only; |x| <= deadband -> 0 in encode-time sign.
        prefer_native: if True (default) and the Rust accelerator extension
            (built via `cd rust-kernel && maturin develop --release`) is
            importable, the returned kernel will be the Rust-backed adapter.
            Set to False to force the pure-Python implementation.

    Returns:
        A configured kernel instance.  Real kernels default to D=10000,
        ternary kernels default to D=16000.
    """
    # Try Rust backend first.
    if prefer_native:
        try:
            from . import _native_loader as _nl  # local import to avoid cycle
            if _nl.HAS_RUST:
                if kind == "real":
                    return _nl.RealKernelRustAdapter(dim=dim or 10000)  # type: ignore[return-value]
                if kind == "ternary":
                    return _nl.TernaryKernelRustAdapter(dim=dim or 16000, deadband=deadband)  # type: ignore[return-value]
        except Exception:
            pass  # fall through to pure-Python

    if kind == "real":
        return RealKernel(dim=dim or 10000)
    if kind == "ternary":
        return TernaryKernel(dim=dim or 16000, deadband=deadband)
    raise ValueError(f"unknown kernel kind: {kind!r}")
