"""Trainable scalar-to-hypervector encoder.

Per PSP-HDC (Ge et al.), each scalar parameter j has a trainable embedding
vector e_j in R^d (where d << D), and a fixed random basis B in R^{d x D}.

For a scaled scalar x_tilde_{i,j} in [-1, 1], the parameter hypervector is

        h_{i,j} = phi( (x_tilde_{i,j} * e_j)^T B )

where phi is kernel-specific (tanh for Real, sign-with-deadband for Ternary).

Training
--------
The encoder is trained with a proxy classification loss against the prototype
memories: we use a margin-based loss between the true class prototype's
similarity and the strongest competitor's similarity.  Backprop is via PyTorch.

For the Ternary kernel we use a straight-through estimator (STE): the forward
pass quantizes via sign-with-deadband, but the gradient is computed as if the
identity function had been applied — a standard BitNet/QAT trick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .kernel import HDCKernel, RealKernel, TernaryKernel


# ---------------------------------------------------------------------------
# Straight-through quantizer
# ---------------------------------------------------------------------------


class _SignDeadbandSTE(torch.autograd.Function):
    """Forward: sign with deadband. Backward: identity (clipped to ±1).

    The clip mirrors BitNet's standard practice — gradients only flow when the
    pre-quantization activation is inside the saturating regime.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, deadband: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        sign = torch.sign(x)
        sign = torch.where(torch.abs(x) <= deadband, torch.zeros_like(x), sign)
        return sign

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        # Pass gradient through; clip where the pre-activation is saturated.
        grad = grad_output.clone()
        grad = torch.where(torch.abs(x) > 1.0, torch.zeros_like(grad), grad)
        return grad, None


def ste_sign_deadband(x: torch.Tensor, deadband: float = 0.0) -> torch.Tensor:
    return _SignDeadbandSTE.apply(x, deadband)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class ScalarEncoder:
    """Parameter-specific scalar-to-hypervector encoder.

    Holds:
        param_names         the names of the P scalar parameters
        embed_dim           the dimensionality d of each e_j  (d << D)
        kernel              the HDC kernel that owns D and the nonlinearity
        basis               (d, D) random projection basis
        embeddings          torch.nn.Parameter of shape (P, d)
        x_min / x_max       per-parameter normalization extrema (filled on fit)

    The encoder mirrors PSP-HDC Eq. (16) and supports both Real and Ternary
    kernels — the kernel decides the nonlinearity.  Training is delegated to
    `fit_to_prototypes` which uses a margin-based proxy loss.
    """

    param_names: List[str]
    kernel: HDCKernel
    embed_dim: int = 64
    seed: int = 0
    basis: np.ndarray = field(init=False)
    _torch_basis: torch.Tensor = field(init=False, repr=False)
    embeddings: nn.Parameter = field(init=False, repr=False)
    x_min: np.ndarray = field(init=False)
    x_max: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.param_names:
            raise ValueError("ScalarEncoder needs at least one parameter name")
        P = len(self.param_names)
        rng = np.random.default_rng(self.seed)
        self.basis = self.kernel.random_basis(self.embed_dim, seed=self.seed)
        self._torch_basis = torch.tensor(self.basis, dtype=torch.float32)
        # Embeddings init: small Gaussian; nn.Parameter so PyTorch tracks grad.
        e0 = rng.standard_normal((P, self.embed_dim)).astype(np.float32) * 0.1
        self.embeddings = nn.Parameter(torch.tensor(e0, dtype=torch.float32))
        self.x_min = np.zeros(P, dtype=np.float32)
        self.x_max = np.ones(P, dtype=np.float32)

    # ---- normalization -------------------------------------------------

    def fit_normalization(self, x: np.ndarray) -> None:
        """Set per-parameter min/max from training matrix x of shape (N, P)."""
        if x.ndim != 2 or x.shape[1] != len(self.param_names):
            raise ValueError(f"fit_normalization: expected (N, {len(self.param_names)}), got {x.shape}")
        self.x_min = x.min(axis=0).astype(np.float32)
        self.x_max = x.max(axis=0).astype(np.float32)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        # Defensive: replace non-finite inputs with the midpoint of the
        # normalised range (0.5 in [0, 1] domain) so encoding cannot produce
        # NaN/Inf hypervectors regardless of upstream data contamination.
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            mid = (self.x_min + self.x_max) * 0.5
            x = np.where(np.isfinite(x), x, mid)
        eps = 1e-8
        x_hat = (x - self.x_min) / (self.x_max - self.x_min + eps)
        x_hat = np.clip(x_hat, 0.0, 1.0)
        return (2.0 * x_hat - 1.0).astype(np.float32)

    # ---- forward (numpy, inference) -----------------------------------

    def encode_row(self, x_row: np.ndarray) -> List[np.ndarray]:
        """Encode a single row of P scalars into a list of P hypervectors."""
        x_tilde = self._scale(x_row[None, :])[0]
        e_np = self.embeddings.detach().cpu().numpy()
        hvs: List[np.ndarray] = []
        for j, name in enumerate(self.param_names):
            hv = self.kernel.encode_scalar(float(x_tilde[j]), e_np[j], self.basis)
            hvs.append(hv)
        return hvs

    def encode_matrix(self, x: np.ndarray) -> List[List[np.ndarray]]:
        """Encode a matrix (N, P) of scalars into per-sample per-parameter HVs."""
        return [self.encode_row(row) for row in x]

    # ---- forward (torch, training) ------------------------------------

    def torch_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward in torch, returning a (N, P, D) tensor.

        The returned tensor lives in the kernel's "numeric envelope":
            * Real kernel: float in [-1, 1] from tanh.
            * Ternary kernel: trits in {-1, 0, +1} produced by an STE
              sign-with-deadband; gradients flow back through identity.
        """
        if x.dim() != 2 or x.shape[1] != len(self.param_names):
            raise ValueError(f"torch_encode: expected (N, {len(self.param_names)}), got {tuple(x.shape)}")
        # Scale to [-1, 1] using the fitted extrema.
        x_min = torch.tensor(self.x_min, dtype=torch.float32, device=x.device)
        x_max = torch.tensor(self.x_max, dtype=torch.float32, device=x.device)
        x_hat = (x - x_min) / (x_max - x_min + 1e-8)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
        x_tilde = 2.0 * x_hat - 1.0  # (N, P)

        # Project per-parameter: for each j, h_j = phi( (x_tilde_j * e_j) @ B )
        # We compute this as (N, P, d) * (1, P, d) -> (N, P, d), then @ B.
        # x_tilde: (N, P) -> (N, P, 1)
        # e:       (P, d) -> (1, P, d)
        e = self.embeddings.unsqueeze(0)
        pre = x_tilde.unsqueeze(-1) * e            # (N, P, d)
        basis = self._torch_basis.to(pre.device)   # (d, D)
        proj = torch.einsum("npd,dD->npD", pre, basis)  # (N, P, D)

        if isinstance(self.kernel, RealKernel):
            return torch.tanh(proj)
        if isinstance(self.kernel, TernaryKernel):
            return ste_sign_deadband(proj, self.kernel.deadband)
        raise NotImplementedError(f"torch_encode: unsupported kernel {self.kernel.name}")

    def parameters(self) -> List[nn.Parameter]:
        return [self.embeddings]


__all__ = ["ScalarEncoder", "ste_sign_deadband"]
