"""Regression tests: boundary conditions, error paths, and recovery semantics.

These cover edge cases that smoke testing skips and protect against silent
regressions in operations that are easy to get subtly wrong.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# HDC kernel boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["real", "ternary"])
def test_bundle_empty_returns_zeros(kind):
    from holograph.hdc.kernel import make_kernel
    k = make_kernel(kind, dim=128, prefer_native=False)
    out = k.bundle([])
    assert out.shape == (128,)
    assert (out == 0).all()


@pytest.mark.parametrize("kind", ["real", "ternary"])
def test_bundle_single_element(kind):
    from holograph.hdc.kernel import make_kernel
    k = make_kernel(kind, dim=128, prefer_native=False)
    rng = np.random.default_rng(0)
    if kind == "real":
        v = rng.standard_normal(128).astype(np.float32)
    else:
        v = rng.choice([-1, 0, 1], size=128).astype(np.int8)
    out = k.bundle([v])
    # Real returns normalized v; ternary returns sign-of-v.
    assert out.shape == (128,)


def test_similarity_zero_vector_returns_zero():
    from holograph.hdc.kernel import make_kernel
    kr = make_kernel("real", dim=64, prefer_native=False)
    kt = make_kernel("ternary", dim=64, prefer_native=False)
    z = kr.zeros()
    rng = np.random.default_rng(0)
    v_r = rng.standard_normal(64).astype(np.float32)
    v_t = rng.choice([-1, 0, 1], size=64).astype(np.int8)
    assert kr.similarity(z, v_r) == 0.0
    assert kr.similarity(v_r, z) == 0.0
    assert kt.similarity(np.zeros(64, dtype=np.int8), v_t) == 0.0


def test_ternary_pack_unpack_idempotent_with_padding():
    """Dim not divisible by 4 should still round-trip."""
    from holograph.hdc.kernel import TernaryKernel
    for dim in (1, 2, 3, 5, 7, 13, 33, 1023):
        k = TernaryKernel(dim=dim)
        rng = np.random.default_rng(0)
        v = rng.choice([-1, 0, 1], size=dim).astype(np.int8)
        rt = k.unpack(k.pack(v))
        assert np.array_equal(rt, v), f"pack/unpack failed at dim={dim}"


def test_real_unpack_size_mismatch_raises():
    from holograph.hdc.kernel import RealKernel
    k = RealKernel(dim=128)
    # Pass too few bytes.
    with pytest.raises(ValueError):
        k.unpack(b"\x00" * 32)


def test_ternary_pack_size_mismatch_raises():
    from holograph.hdc.kernel import TernaryKernel
    k = TernaryKernel(dim=128)
    with pytest.raises(ValueError):
        k.pack(np.zeros(64, dtype=np.int8))


@pytest.mark.parametrize("kind", ["real", "ternary"])
def test_bundle_large_n(kind):
    """Bundling many vectors should remain numerically stable."""
    from holograph.hdc.kernel import make_kernel
    k = make_kernel(kind, dim=512, prefer_native=False)
    rng = np.random.default_rng(0)
    if kind == "real":
        vs = [rng.standard_normal(512).astype(np.float32) for _ in range(1024)]
    else:
        vs = [rng.choice([-1, 0, 1], size=512).astype(np.int8) for _ in range(1024)]
    out = k.bundle(vs)
    assert out.shape == (512,)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Substrate boundary conditions
# ---------------------------------------------------------------------------


def test_substrate_lookup_nonexistent_returns_none():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    assert g.lookup_by_surface("nobody") is None
    assert g.get_entity(99999) is None
    assert g.get_hv_blob(99999) is None
    g.close()


def test_substrate_self_loops_handled():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    a = g.upsert_entity("A")
    g.upsert_edge(a, a, "self_ref")
    roles = g.structural_roles()
    # Self-loops shouldn't crash centrality computation.
    assert a in roles
    comm = g.communities()
    assert a in comm
    g.close()


def test_substrate_parallel_edges_distinct_relations():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    a = g.upsert_entity("A"); b = g.upsert_entity("B")
    e1 = g.upsert_edge(a, b, "r1")
    e2 = g.upsert_edge(a, b, "r2")  # different relation
    assert e1 != e2
    assert g.n_edges() == 2


def test_substrate_edge_weight_clamping():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    a = g.upsert_entity("A"); b = g.upsert_entity("B")
    eid = g.upsert_edge(a, b, "r", weight=1.0)
    # Push high; should clamp at 10.
    for _ in range(50):
        g.update_edge_weight(eid, 1.0)
    w_high = next(e.weight for e in g.all_edges() if e.id == eid)
    assert w_high == 10.0
    # Push low; should clamp at 0.
    for _ in range(100):
        g.update_edge_weight(eid, -1.0)
    w_low = next(e.weight for e in g.all_edges() if e.id == eid)
    assert w_low == 0.0
    g.close()


def test_substrate_empty_graph_roles():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    assert g.structural_roles() == {}
    assert g.communities() == {}
    g.close()


def test_substrate_singleton_graph():
    from holograph.graph.substrate import GraphSubstrate
    g = GraphSubstrate(":memory:")
    a = g.upsert_entity("only")
    roles = g.structural_roles()
    assert roles[a]["degree"] == 0.0
    assert roles[a]["betweenness"] == 0.0
    g.close()


# ---------------------------------------------------------------------------
# Composer boundary conditions
# ---------------------------------------------------------------------------


def test_composer_empty_activated_set():
    from holograph.graph.substrate import GraphSubstrate
    from holograph.hdc.kernel import make_kernel
    from holograph.reader.composer import HDCComposer
    g = GraphSubstrate(":memory:")
    k = make_kernel("real", dim=128, prefer_native=False)
    composer = HDCComposer(k)
    composed, paths = composer.compose(g, [], {})
    assert composed.shape == (128,)
    assert (composed == 0).all()
    assert paths == []


def test_composer_isolated_activated_entities():
    """Entities activated but with no edges between them in G_q."""
    from holograph.graph.substrate import GraphSubstrate
    from holograph.hdc.kernel import make_kernel
    from holograph.reader.composer import HDCComposer
    g = GraphSubstrate(":memory:")
    k = make_kernel("real", dim=256, prefer_native=False)
    rng = np.random.default_rng(0)
    a = g.upsert_entity("A"); b = g.upsert_entity("B")
    for eid in (a, b):
        hv = rng.standard_normal(256).astype(np.float32)
        hv = hv / np.linalg.norm(hv)
        g.set_hv(eid, k.pack(hv), "real")
    composer = HDCComposer(k)
    composed, paths = composer.compose(g, [a, b], {a: 1.0, b: 1.0})
    # No edges in G_q -> fallback to bundling entity hypervectors directly.
    assert composed.shape == (256,)
    assert paths == []


# ---------------------------------------------------------------------------
# Reader boundary conditions
# ---------------------------------------------------------------------------


def test_reader_empty_graph_returns_empty(spacy_nlp):
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, top_k=4, spacy_nlp=spacy_nlp)
    out = hg.read("Anything?")
    assert out.activated_ids == []
    assert out.activated_subgraph_edges == []
    hg.close()


def test_reader_query_with_no_matches(spacy_nlp):
    """Query mentions entities not in the graph."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, top_k=4, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob.", anchor="d1")
    out = hg.read("What about Zaphod Beeblebrox?")
    # No crash; reader should return some activations (the graph is non-empty).
    assert isinstance(out.activated_ids, list)
    hg.close()


# ---------------------------------------------------------------------------
# Feedback boundary conditions
# ---------------------------------------------------------------------------


def test_feedback_no_gold_supervision(spacy_nlp):
    """Feedback with empty gold sets should not crash and should produce zero reward."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob.", anchor="d1")
    ev = hg.feedback("Where did Alice meet Bob?",
                      gold_doc_anchors=[],
                      gold_entity_ids=[])
    assert ev.reward.recall == 0.0
    assert ev.reward.precision == 0.0
    hg.close()


def test_feedback_invalid_entity_ids_dont_crash(spacy_nlp):
    """Gold entity ids that don't exist in the substrate should be handled gracefully."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob.", anchor="d1")
    ev = hg.feedback("Test query.",
                      gold_doc_anchors=["d1"],
                      gold_entity_ids=[99999, 99998])
    # Should produce a valid (probably zero) reward, not crash.
    assert ev is not None
    hg.close()


# ---------------------------------------------------------------------------
# PSP encoder boundary conditions
# ---------------------------------------------------------------------------


def test_psp_encoder_single_param():
    from holograph.hdc.encoder import ScalarEncoder
    from holograph.hdc.kernel import make_kernel
    k = make_kernel("real", dim=512, prefer_native=False)
    enc = ScalarEncoder(["only"], kernel=k, embed_dim=8, seed=0)
    X = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    enc.fit_normalization(X)
    hvs = enc.encode_row(np.array([2.0], dtype=np.float32))
    assert len(hvs) == 1
    assert hvs[0].shape == (512,)


def test_psp_encoder_outside_fit_range():
    """Values outside the fit range should clip, not crash."""
    from holograph.hdc.encoder import ScalarEncoder
    from holograph.hdc.kernel import make_kernel
    k = make_kernel("real", dim=256, prefer_native=False)
    enc = ScalarEncoder(["a", "b"], kernel=k, embed_dim=8, seed=0)
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    enc.fit_normalization(X)
    # Out-of-range query
    hvs = enc.encode_row(np.array([-1000.0, 1000.0], dtype=np.float32))
    assert all(np.isfinite(h).all() for h in hvs)


def test_psp_encoder_rejects_wrong_shape():
    from holograph.hdc.encoder import ScalarEncoder
    from holograph.hdc.kernel import make_kernel
    k = make_kernel("real", dim=256, prefer_native=False)
    enc = ScalarEncoder(["a", "b"], kernel=k, embed_dim=8, seed=0)
    X_bad = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 1-D
    with pytest.raises(ValueError):
        enc.fit_normalization(X_bad)
    X_bad2 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # wrong width
    with pytest.raises(ValueError):
        enc.fit_normalization(X_bad2)


# ---------------------------------------------------------------------------
# Prototype memory boundary conditions
# ---------------------------------------------------------------------------


def test_prototype_memory_empty_predict_raises():
    from holograph.hdc.kernel import make_kernel
    from holograph.hdc.memory import PrototypeMemory
    k = make_kernel("real", dim=128, prefer_native=False)
    proto = PrototypeMemory(k)
    v = np.zeros(128, dtype=np.float32)
    with pytest.raises(RuntimeError):
        proto.predict(v)


def test_prototype_memory_single_class():
    from holograph.hdc.kernel import make_kernel
    from holograph.hdc.memory import PrototypeMemory
    k = make_kernel("real", dim=128, prefer_native=False)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(128).astype(np.float32)
    proto = PrototypeMemory(k)
    proto.fit({"only": [a]})
    cls, margin, scores = proto.predict(a)
    assert cls == "only"
    # Margin is undefined with one class; should be the top score minus 0.0.
    assert margin == pytest.approx(scores["only"], abs=1e-6)
