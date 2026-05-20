"""Tests for the HDC kernels."""

from __future__ import annotations

import numpy as np
import pytest

from holograph.hdc.kernel import RealKernel, TernaryKernel, make_kernel


# ---------------------------------------------------------------------------
# Real kernel
# ---------------------------------------------------------------------------


class TestRealKernel:
    def test_random_basis_shape_and_norm(self):
        k = RealKernel(dim=1024)
        b = k.random_basis(8, seed=0)
        assert b.shape == (8, 1024)
        # Rows are unit-norm.
        norms = np.linalg.norm(b, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_bind_is_elementwise_product(self):
        k = RealKernel(dim=64)
        a = np.array([1.0, -1.0, 0.5, 0.0] * 16, dtype=np.float32)
        b = np.array([1.0, 1.0, 2.0, 3.0] * 16, dtype=np.float32)
        assert np.allclose(k.bind(a, b), a * b)

    def test_bundle_is_normalized_sum(self):
        k = RealKernel(dim=64)
        a = np.ones(64, dtype=np.float32)
        b = -np.ones(64, dtype=np.float32)
        # a + b = 0 -> normalized zero.
        assert np.allclose(k.bundle([a, b]), 0.0)
        # Two identical vectors sum to 2a, normalized -> unit.
        out = k.bundle([a, a])
        assert np.allclose(np.linalg.norm(out), 1.0, atol=1e-5)

    def test_similarity_cosine(self):
        k = RealKernel(dim=32)
        a = np.array([1.0] * 32, dtype=np.float32)
        assert k.similarity(a, a) == pytest.approx(1.0, abs=1e-6)
        assert k.similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)
        z = np.zeros(32, dtype=np.float32)
        assert k.similarity(a, z) == 0.0

    def test_pack_unpack_lossless(self):
        k = RealKernel(dim=128)
        rng = np.random.default_rng(0)
        a = rng.standard_normal(128).astype(np.float32)
        roundtrip = k.unpack(k.pack(a))
        assert np.array_equal(a, roundtrip)

    def test_encode_scalar_bounded(self):
        k = RealKernel(dim=128)
        basis = k.random_basis(8, seed=1)
        e = np.ones(8, dtype=np.float32) * 0.1
        hv = k.encode_scalar(0.5, e, basis)
        assert hv.shape == (128,)
        assert hv.min() >= -1.0 and hv.max() <= 1.0


# ---------------------------------------------------------------------------
# Ternary kernel
# ---------------------------------------------------------------------------


class TestTernaryKernel:
    def test_random_basis_shape(self):
        k = TernaryKernel(dim=1024)
        b = k.random_basis(8, seed=0)
        assert b.shape == (8, 1024)

    def test_bind_trit_table(self):
        k = TernaryKernel(dim=12)
        a = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int8)
        b = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 1, -1], dtype=np.int8)
        # Bind is integer multiplication.
        assert np.array_equal(k.bind(a, b), (a.astype(np.int32) * b.astype(np.int32)).astype(np.int8))

    def test_bundle_deadband_creates_zeros(self):
        k = TernaryKernel(dim=64)
        # 4 +1 votes and 4 -1 votes: each position is 50/50 in expectation.
        rng = np.random.default_rng(0)
        vs = [rng.choice([-1, 1], size=64).astype(np.int8) for _ in range(8)]
        bundled = k.bundle(vs)
        # With deadband ~ 0.5 * sqrt(8) ~ 1.41, positions with |sum|<=1 -> 0.
        # We should see some zeros (abstention channel).
        assert (bundled == 0).any()

    def test_similarity_matches_minus_mismatches(self):
        k = TernaryKernel(dim=8)
        a = np.array([1, -1, 0, 1, 1, 0, -1, 1], dtype=np.int8)
        b = np.array([1, 1, 0, 1, -1, 0, -1, 1], dtype=np.int8)
        # Active positions (both nonzero): indices 0,1,3,4,6,7 -> 6 active
        # Matches: 0,3,6,7 -> 4
        # Mismatches: 1,4 -> 2
        # Score: (4-2)/6 = 1/3
        assert k.similarity(a, b) == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_pack_unpack_roundtrip(self):
        k = TernaryKernel(dim=257)  # odd-length to exercise padding
        rng = np.random.default_rng(0)
        a = rng.choice([-1, 0, 1], size=257).astype(np.int8)
        roundtrip = k.unpack(k.pack(a))
        assert np.array_equal(a, roundtrip)

    def test_pack_storage_compression(self):
        k = TernaryKernel(dim=10000)
        rng = np.random.default_rng(0)
        a = rng.choice([-1, 0, 1], size=10000).astype(np.int8)
        packed = k.pack(a)
        assert len(packed) == 2500  # 10000 trits / 4 trits per byte

    def test_quantize_sign_with_deadband(self):
        k = TernaryKernel(dim=10, deadband=0.0)
        x = np.array([2.0, -3.0, 0.0, 0.1, -0.1, 5.0, -5.0, 1e-9, -1e-9, 0.5], dtype=np.float32)
        q = k.quantize(x)
        # deadband 0.0 -> only exact zero maps to 0.
        assert q[2] == 0
        # All nonzero entries take the sign.
        for i in (0, 1, 3, 4, 5, 6, 9):
            assert q[i] == np.sign(x[i])

    def test_make_kernel(self):
        assert isinstance(make_kernel("real"), RealKernel)
        assert isinstance(make_kernel("ternary"), TernaryKernel)
        with pytest.raises(ValueError):
            make_kernel("nope")  # type: ignore[arg-type]
