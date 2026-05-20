"""Tests for prototype memory, scalar encoder, and MAS diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from holograph.hdc.kernel import make_kernel
from holograph.hdc.memory import PrototypeMemory, mas
from holograph.hdc.encoder import ScalarEncoder, ste_sign_deadband


@pytest.mark.parametrize("kernel_kind", ["real", "ternary"])
class TestPrototypeMemory:
    def test_fit_and_predict_separates_classes(self, kernel_kind):
        k = make_kernel(kernel_kind, dim=2048)
        rng = np.random.default_rng(0)
        # Two well-separated classes: random ±1 hypervectors with shared seed.
        a1 = rng.choice([-1, 1], size=2048).astype(np.float32 if kernel_kind == "real" else np.int8)
        a2 = a1.copy()
        b1 = rng.choice([-1, 1], size=2048).astype(np.float32 if kernel_kind == "real" else np.int8)
        b2 = b1.copy()
        # Add noise so prototypes aren't degenerate.
        if kernel_kind == "real":
            a1 = a1 + 0.05 * rng.standard_normal(2048).astype(np.float32)
            b1 = b1 + 0.05 * rng.standard_normal(2048).astype(np.float32)
        proto = PrototypeMemory(k)
        proto.fit({"A": [a1, a2], "B": [b1, b2]})
        # Each training vector classifies into its class.
        for v, true_cls in ((a1, "A"), (a2, "A"), (b1, "B"), (b2, "B")):
            cls, margin, _ = proto.predict(v)
            assert cls == true_cls
            assert margin > 0.0

    def test_update_grows_prototype(self, kernel_kind):
        k = make_kernel(kernel_kind, dim=1024)
        rng = np.random.default_rng(1)
        proto = PrototypeMemory(k)
        cls = "X"
        # Should start empty.
        assert cls not in proto.prototypes
        for _ in range(5):
            v = rng.choice([-1, 1], size=1024).astype(np.float32 if kernel_kind == "real" else np.int8)
            proto.update(cls, v)
        assert cls in proto.prototypes
        assert proto.prototypes[cls].shape == (1024,)

    def test_persistence_roundtrip(self, kernel_kind):
        k = make_kernel(kernel_kind, dim=512)
        rng = np.random.default_rng(2)
        proto = PrototypeMemory(k)
        a = rng.choice([-1, 1], size=512).astype(np.float32 if kernel_kind == "real" else np.int8)
        b = rng.choice([-1, 1], size=512).astype(np.float32 if kernel_kind == "real" else np.int8)
        proto.fit({"A": [a, a.copy()], "B": [b]})
        blob = proto.to_bytes()
        new_proto = PrototypeMemory(k)
        new_proto.load_bytes(blob)
        assert set(new_proto.classes()) == {"A", "B"}
        for c in new_proto.classes():
            assert np.array_equal(new_proto.prototypes[c], proto.prototypes[c])


class TestMAS:
    def test_alignment_separation(self):
        k = make_kernel("real", dim=512)
        # Build two prototypes and per-class component memories that match each.
        rng = np.random.default_rng(3)
        a_proto = rng.standard_normal(512).astype(np.float32)
        a_proto = a_proto / np.linalg.norm(a_proto)
        b_proto = rng.standard_normal(512).astype(np.float32)
        b_proto = b_proto / np.linalg.norm(b_proto)
        class_protos = {"A": a_proto, "B": b_proto}
        # Component memory for parameter "p" that mirrors the protos.
        comp = {"p": {"A": a_proto, "B": b_proto}}
        report = mas(k, comp, class_protos)
        assert report["p"]["alignment"] == pytest.approx(1.0, abs=1e-5)
        # B and A are random, so similarity is low; separation should be positive.
        assert report["p"]["separation"] > 0.5


class TestScalarEncoder:
    @pytest.mark.parametrize("kernel_kind", ["real", "ternary"])
    def test_encode_row_shape(self, kernel_kind):
        k = make_kernel(kernel_kind, dim=512)
        enc = ScalarEncoder(["p1", "p2"], kernel=k, embed_dim=16, seed=0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        enc.fit_normalization(X)
        hvs = enc.encode_row(np.array([2.0, 3.0], dtype=np.float32))
        assert len(hvs) == 2
        assert all(h.shape == (512,) for h in hvs)
        for h in hvs:
            assert h.dtype == k.dtype

    def test_torch_encode_gradients_flow_real(self):
        import torch
        k = make_kernel("real", dim=256)
        enc = ScalarEncoder(["p1", "p2"], kernel=k, embed_dim=8, seed=0)
        # Fit normalization on a wider range than the inputs we backprop on,
        # so the clamp doesn't saturate (and zero) the gradient.
        X_fit = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        enc.fit_normalization(X_fit)
        X_train = np.array([[3.0, 5.0], [6.0, 4.0]], dtype=np.float32)
        x = torch.tensor(X_train, dtype=torch.float32)
        out = enc.torch_encode(x)  # (N, P, D)
        loss = (out ** 2).sum()  # nonzero gradient even if some entries are tiny
        loss.backward()
        assert enc.embeddings.grad is not None
        assert not torch.allclose(enc.embeddings.grad, torch.zeros_like(enc.embeddings.grad))

    def test_ste_sign_deadband_gradient_passes_through(self):
        import torch
        x = torch.tensor([0.5, -0.5, 0.0001, -2.0], requires_grad=True)
        y = ste_sign_deadband(x, deadband=0.0)
        # Forward is sign with deadband.
        assert torch.allclose(y, torch.tensor([1.0, -1.0, 1.0, -1.0]))
        loss = y.sum()
        loss.backward()
        # Gradient is identity except clipped where |x| > 1.
        expected = torch.tensor([1.0, 1.0, 1.0, 0.0])
        assert torch.allclose(x.grad, expected)
