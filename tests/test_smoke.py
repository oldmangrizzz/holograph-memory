"""Smoke tests: happy-path coverage across every subsystem.

These tests confirm the system runs end-to-end on representative workloads
without crashing or producing degenerate output.  They are intentionally broad
and shallow; regression and adversarial tests live in separate modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest


def test_imports_top_level():
    import holograph
    assert holograph.__version__
    assert holograph.make_kernel is not None
    assert holograph.HoloGraph is not None


def test_make_kernel_both_backends():
    from holograph.hdc.kernel import make_kernel, RealKernel, TernaryKernel
    kr = make_kernel("real", dim=1024, prefer_native=False)
    kt = make_kernel("ternary", dim=2048, prefer_native=False)
    assert isinstance(kr, RealKernel)
    assert isinstance(kt, TernaryKernel)
    assert kr.dim == 1024
    assert kt.dim == 2048


@pytest.mark.parametrize("kind", ["real", "ternary"])
def test_kernel_round_trip(kind):
    from holograph.hdc.kernel import make_kernel
    k = make_kernel(kind, dim=512, prefer_native=False)
    basis = k.random_basis(8, seed=0)
    e = np.full(8, 0.2, dtype=np.float32)
    hv = k.encode_scalar(0.5, e, basis)
    blob = k.pack(hv)
    rt = k.unpack(blob)
    assert rt.shape == hv.shape
    # Real should be lossless; ternary should be exact.
    assert np.array_equal(rt, hv)


@pytest.mark.parametrize("kind", ["real", "ternary"])
def test_kernel_compose_bundle_similarity(kind):
    from holograph.hdc.kernel import make_kernel
    k = make_kernel(kind, dim=2048, prefer_native=False)
    rng = np.random.default_rng(0)
    if kind == "real":
        a = rng.standard_normal(2048).astype(np.float32)
        b = rng.standard_normal(2048).astype(np.float32)
    else:
        a = rng.choice([-1, 0, 1], size=2048).astype(np.int8)
        b = rng.choice([-1, 0, 1], size=2048).astype(np.int8)
    bound = k.bind(a, b)
    bundled = k.bundle([a, b])
    assert bound.shape == a.shape
    assert bundled.shape == a.shape
    # Self-similarity is positive.
    assert k.similarity(a, a) > 0


def test_runtime_full_pipeline_real(spacy_nlp):
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=2048, top_k=6, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob in Paris yesterday.", anchor="doc1")
    hg.ingest_text("Bob is a friend of Carol.", anchor="doc2")
    out = hg.read("Who did Alice meet?")
    assert out.activated_ids, "reader returned no activations"
    assert out.composed_hv is not None
    summary = hg.summary()
    assert summary["entities"] >= 3
    assert summary["edges"] >= 1
    hg.close()


def test_runtime_full_pipeline_ternary(spacy_nlp):
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="ternary", dim=4096, top_k=6, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob in Paris yesterday.", anchor="doc1")
    out = hg.read("Where was Alice?")
    assert out.activated_ids
    hg.close()


def test_runtime_psp_pipeline(spacy_nlp):
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=2048, spacy_nlp=spacy_nlp)
    params = ["p1", "p2", "p3"]
    hg.configure_scalar_encoder(params, embed_dim=16, seed=0)
    X = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
    ], dtype=np.float32)
    y = ["low", "low", "high", "high"]
    hg.fit_psp_prototypes(X, y, groups={"g1": [0], "g2": [1, 2]})
    for row, true in zip(X, y):
        hv, _ = hg.encode_psp_sample(row, groups={"g1": [0], "g2": [1, 2]})
        pred, _, _ = hg.memory.predict(hv)
        assert pred == true
    hg.close()


def test_runtime_feedback_step(spacy_nlp):
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=2048, top_k=6, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob in Paris.", anchor="d1")
    alice = hg.substrate.lookup_by_surface("Alice")
    bob = hg.substrate.lookup_by_surface("Bob")
    ev = hg.feedback("Where did Alice meet Bob?",
                      gold_doc_anchors=["d1"],
                      gold_entity_ids=[i for i in (alice, bob) if i is not None])
    assert ev.reward.recall >= 0.0
    assert -10.0 < ev.total_reward < 10.0
    hg.close()


def test_persistence_roundtrip(tmp_path, spacy_nlp):
    from holograph.runtime import HoloGraph
    db = tmp_path / "smoke.db"
    hg = HoloGraph(kernel_kind="real", dim=1024, db_path=str(db), spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob.", anchor="d1")
    n_ents = hg.substrate.n_entities()
    n_edges = hg.substrate.n_edges()
    hg.close()
    # Reopen
    hg2 = HoloGraph(kernel_kind="real", dim=1024, db_path=str(db), spacy_nlp=spacy_nlp)
    assert hg2.substrate.n_entities() == n_ents
    assert hg2.substrate.n_edges() == n_edges
    hg2.close()


def test_native_loader_falls_back_cleanly():
    """The Python loader must succeed regardless of Rust availability."""
    from holograph.hdc._native_loader import HAS_RUST
    from holograph.hdc.kernel import make_kernel
    k = make_kernel("real", dim=512, prefer_native=False)
    assert k.dim == 512
    if HAS_RUST:
        k2 = make_kernel("real", dim=512, prefer_native=True)
        # Whether Rust is loaded or not, the call must work.
        assert k2.dim == 512


def test_mas_diagnostic_runs():
    from holograph.hdc.kernel import make_kernel
    from holograph.hdc.memory import PrototypeMemory, mas
    k = make_kernel("real", dim=1024, prefer_native=False)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1024).astype(np.float32); a /= np.linalg.norm(a)
    b = rng.standard_normal(1024).astype(np.float32); b /= np.linalg.norm(b)
    proto = PrototypeMemory(k)
    proto.fit({"A": [a, a.copy()], "B": [b, b.copy()]})
    comp = {"x": {"A": a, "B": b}}
    report = mas(k, comp, proto.prototypes)
    assert "alignment" in report["x"]
    assert "separation" in report["x"]
    assert "ratio" in report["x"]


def test_attribution_keys_consistent(spacy_nlp):
    """Per-entity attribution keys must reference real activated entities."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, top_k=4, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob. Bob knows Carol.", anchor="d1")
    out = hg.read("Who does Bob know?")
    if out.attribution and out.attribution.per_entity:
        for eid in out.attribution.per_entity:
            ent = hg.substrate.get_entity(eid)
            assert ent is not None, f"attribution references missing entity {eid}"
    hg.close()
