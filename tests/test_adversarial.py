"""Adversarial / threat-model tests.

Threat model: an attacker with the ability to feed arbitrary text, structured
triples, scalar data, queries, gold supervision, and packed hypervector blobs
into the runtime. The attacker MAY have read access to disk artifacts (the
SQLite database, hypervector blobs) and write access to the runtime's text
input channels.  The attacker DOES NOT have shell access or the ability to
inject Python code.

Goals of the attacker (defender's nightmare list):
    1. Crash the process (denial of service).
    2. Corrupt the graph or prototype memories (data integrity).
    3. Exhaust memory or CPU (resource exhaustion).
    4. Read memory that should be private to other queries (information leak).
    5. Achieve persistent state poisoning that survives restart.
    6. Bypass alias resolution to impersonate entities.
    7. Cause numerical instability (NaN / Inf propagation).
    8. Hijack the Rust loader path (supply-chain).

These tests probe the surface for each goal.  Tests that uncover defects
should be marked with ``pytest.xfail`` and a follow-up patch should land in
test_regression.py.  At end of file we run a smoke pipeline to confirm the
system as a whole still works after the adversarial probing.
"""

from __future__ import annotations

import gc
import io
import os
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Goal 1: crash via malformed input
# ---------------------------------------------------------------------------


class TestCrashFromInput:
    """Probe for crashes from hostile text, scalar, and hypervector input."""

    def test_unicode_homoglyph_entity_collision(self, spacy_nlp):
        """A latin 'A' and a Cyrillic 'А' (U+0410) must produce distinct entities
        OR must be deliberately unified.  Either way the system must not crash."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=512, spacy_nlp=spacy_nlp)
        hg.ingest_text("A is friends with B.", anchor="d1")
        hg.ingest_text("А (Cyrillic) is friends with B.", anchor="d2")
        # The system must remain queryable.
        out = hg.read("Who is friends with B?")
        assert isinstance(out.activated_ids, list)
        hg.close()

    def test_rtl_override_in_entity_name(self, spacy_nlp):
        """U+202E (right-to-left override) characters in names must not break
        anything downstream (storage, lookup, attribution)."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=256, spacy_nlp=spacy_nlp)
        evil = "Alice‮evil"
        hg.ingest_text(f"{evil} met Bob.", anchor="d1")
        out = hg.read("Who met Bob?")
        assert isinstance(out.activated_ids, list)
        hg.close()

    def test_zero_width_space_in_alias(self):
        """Zero-width spaces in alias surface forms must not cause silent merges."""
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        a = g.upsert_entity("Alice", aliases=["Ali"])
        b = g.upsert_entity("A​lice")  # zero-width space inside
        # The two entities must be distinct -- no silent merge.
        assert a != b
        g.close()

    def test_oversized_text_input(self, spacy_nlp):
        """A very long text input must not crash the writer."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=256, spacy_nlp=spacy_nlp)
        # ~30k characters; bounded but realistic for one document.
        text = ("Alice met Bob. " * 2000)
        hg.ingest_text(text, anchor="d-big")
        assert hg.substrate.n_entities() >= 2
        hg.close()

    def test_pathological_nesting(self, spacy_nlp):
        """Deeply parenthesized text must not blow up spaCy or the writer."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=256, spacy_nlp=spacy_nlp)
        text = "Alice " + "(" * 200 + " met Bob " + ")" * 200 + "."
        hg.ingest_text(text, anchor="d-nest")
        # Should not raise; may or may not extract triples.
        hg.close()

    def test_malformed_real_hv_blob_too_short(self):
        from holograph.hdc.kernel import RealKernel
        k = RealKernel(dim=128)
        with pytest.raises(ValueError):
            k.unpack(b"\x00\x01\x02")

    def test_malformed_real_hv_blob_nan_inf(self):
        """Float blobs containing NaN/Inf must round-trip but not propagate to
        prototype similarity as Inf."""
        from holograph.hdc.kernel import RealKernel
        k = RealKernel(dim=64)
        bad = np.array([np.nan, np.inf, -np.inf] + [0.0] * 61, dtype=np.float32)
        blob = k.pack(bad)
        roundtrip = k.unpack(blob)
        # The values round-trip (this is by design: pack is bit-exact).  But
        # similarity must produce a finite or zero result, not propagate Inf.
        v = np.ones(64, dtype=np.float32)
        sim = k.similarity(bad, v)
        assert np.isfinite(sim) or sim == 0.0, f"similarity propagated {sim} with NaN/Inf input"

    def test_malformed_ternary_hv_reserved_codes(self):
        """The reserved trit code 0b11 must decode to 0 (defensive)."""
        from holograph.hdc.kernel import TernaryKernel
        k = TernaryKernel(dim=16)
        # Every byte 0xFF -> all four trits are 0b11 -> all map to 0
        blob = b"\xff" * 4  # 4 bytes * 4 trits = 16 dims
        v = k.unpack(blob)
        assert v.shape == (16,)
        assert (v == 0).all()

    def test_scalar_encoder_nan_input(self):
        """NaN in scalar input must clamp or error, not silently produce NaN HVs."""
        from holograph.hdc.encoder import ScalarEncoder
        from holograph.hdc.kernel import make_kernel
        k = make_kernel("real", dim=128, prefer_native=False)
        enc = ScalarEncoder(["a"], kernel=k, embed_dim=8, seed=0)
        enc.fit_normalization(np.array([[0.0], [10.0]], dtype=np.float32))
        out = enc.encode_row(np.array([np.nan], dtype=np.float32))
        # NaN in -> any NaN out is a defect.  Clip to range or hard-clamp to 0.
        assert all(np.isfinite(h).all() for h in out), "NaN propagated through encoder"


# ---------------------------------------------------------------------------
# Goal 2: corrupt graph / prototype memories via injection
# ---------------------------------------------------------------------------


class TestInjection:
    def test_sql_injection_in_surface_form(self):
        """SQL metacharacters in entity surfaces must not affect the DB schema."""
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        evil = "Bob'; DROP TABLE entities; --"
        eid = g.upsert_entity(evil, type="Person")
        # The table must still exist and the entity must be queryable.
        assert g.lookup_by_surface(evil) == eid
        assert g.n_entities() == 1
        g.close()

    def test_sql_injection_in_alias(self):
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        eid = g.upsert_entity("Real", aliases=["normal", "x' OR 1=1 --"])
        # Alias must resolve back to the same entity.
        assert g.lookup_by_surface("x' OR 1=1 --") == eid
        # Tables intact.
        assert g.n_entities() == 1
        g.close()

    def test_sql_injection_in_relation(self):
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        a = g.upsert_entity("A"); b = g.upsert_entity("B")
        evil_rel = "knows'; DROP TABLE edges; --"
        eid = g.upsert_edge(a, b, evil_rel)
        assert g.n_edges() == 1
        # Relation is preserved verbatim.
        assert g.all_edges()[0].relation == evil_rel
        g.close()

    def test_sql_injection_in_document_anchor(self):
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        a = g.upsert_entity("A")
        g.upsert_edge(a, a, "self", source="d1' OR '1'='1")
        g.upsert_document("d1' OR '1'='1", "evil text")
        docs = g.documents_for_entities([a])
        assert len(docs) == 1
        g.close()


# ---------------------------------------------------------------------------
# Goal 3: resource exhaustion
# ---------------------------------------------------------------------------


class TestResourceExhaustion:
    @pytest.mark.timeout(30)
    def test_graph_bombing_co_occurs(self, spacy_nlp):
        """Crafted input with many named entities in one sentence triggers the
        co_occurs fallback for every adjacent pair.  Verify the writer caps
        runtime and doesn't enter quadratic behaviour."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=512, spacy_nlp=spacy_nlp)
        # Build a sentence with many quoted proper nouns.
        names = " and ".join(f"Person{i}" for i in range(50))
        text = f"{names} all worked together."
        t0 = time.time()
        hg.ingest_text(text, anchor="d-bomb")
        dt = time.time() - t0
        assert dt < 5.0, f"co_occurs fallback took {dt:.2f}s for 50 entities"
        hg.close()

    @pytest.mark.timeout(30)
    def test_bundle_many_hypervectors(self):
        """Bundling 10k hypervectors must complete in reasonable time."""
        from holograph.hdc.kernel import make_kernel
        k = make_kernel("real", dim=1024, prefer_native=False)
        rng = np.random.default_rng(0)
        vs = [rng.standard_normal(1024).astype(np.float32) for _ in range(10000)]
        t0 = time.time()
        out = k.bundle(vs)
        dt = time.time() - t0
        assert out.shape == (1024,)
        assert dt < 15.0, f"bundle of 10k vecs took {dt:.2f}s"

    @pytest.mark.timeout(30)
    def test_betweenness_capped_on_large_graph(self):
        """Substrate's betweenness computation is k-sampled to keep cost bounded
        on large graphs."""
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        # 300 nodes, ring topology + a few cross-edges
        ids = [g.upsert_entity(f"e{i}") for i in range(300)]
        for i in range(300):
            g.upsert_edge(ids[i], ids[(i + 1) % 300], "next")
        for i in range(0, 300, 30):
            g.upsert_edge(ids[i], ids[(i + 150) % 300], "cross")
        t0 = time.time()
        roles = g.structural_roles()
        dt = time.time() - t0
        assert dt < 10.0, f"structural_roles on 300-node graph took {dt:.2f}s"
        assert len(roles) == 300
        g.close()


# ---------------------------------------------------------------------------
# Goal 4: information leak via attribution
# ---------------------------------------------------------------------------


class TestInformationLeak:
    def test_attribution_only_for_activated_entities(self, spacy_nlp):
        """Per-entity attribution keys must be a subset of the activated set.
        Leaking attribution for entities outside G_q would expose
        cross-query information."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=512, top_k=3, spacy_nlp=spacy_nlp)
        hg.ingest_text("Alice met Bob.", anchor="d1")
        hg.ingest_text("Carol met Dave.", anchor="d2")
        hg.ingest_text("Eve knows nothing.", anchor="d3")
        out = hg.read("Who met Bob?")
        active = set(out.activated_ids)
        if out.attribution and out.attribution.per_entity:
            leaked = set(out.attribution.per_entity.keys()) - active
            assert not leaked, f"attribution leaked entities outside activated set: {leaked}"
        hg.close()

    def test_path_attribution_only_from_activated_subgraph(self, spacy_nlp):
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=512, top_k=3, spacy_nlp=spacy_nlp)
        hg.ingest_text("Alice met Bob.", anchor="d1")
        hg.ingest_text("Carol met Dave.", anchor="d2")
        out = hg.read("Who met Bob?")
        if out.attribution and out.attribution.per_path_top:
            valid = {(u, r, v) for (u, r, v) in out.activated_subgraph_edges}
            for (u, r, v), _ in out.attribution.per_path_top:
                assert (u, r, v) in valid


# ---------------------------------------------------------------------------
# Goal 5: persistent state poisoning
# ---------------------------------------------------------------------------


class TestStatePoisoning:
    def test_feedback_oscillation_does_not_explode_weights(self, spacy_nlp):
        """Alternating positive/negative feedback on the same edges must keep
        weights bounded."""
        from holograph.runtime import HoloGraph
        hg = HoloGraph(kernel_kind="real", dim=512, spacy_nlp=spacy_nlp)
        hg.ingest_text("Alice met Bob.", anchor="d1")
        alice = hg.substrate.lookup_by_surface("Alice")
        bob = hg.substrate.lookup_by_surface("Bob")
        # Alternate gold supervision.
        for r in range(20):
            golds = [alice, bob] if r % 2 == 0 else []
            hg.feedback("Who did Alice meet?",
                         gold_doc_anchors=["d1"] if golds else [],
                         gold_entity_ids=golds)
        for e in hg.substrate.all_edges():
            assert 0.0 <= e.weight <= 10.0, f"edge weight {e.weight} out of clamp range"
        hg.close()

    def test_prototype_repeated_update_does_not_explode(self):
        """Repeated incremental prototype updates must keep norms bounded."""
        from holograph.hdc.kernel import make_kernel
        from holograph.hdc.memory import PrototypeMemory
        k = make_kernel("real", dim=512, prefer_native=False)
        proto = PrototypeMemory(k)
        rng = np.random.default_rng(0)
        for _ in range(200):
            v = rng.standard_normal(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            proto.update("c", v)
        norm = float(np.linalg.norm(proto.prototypes["c"]))
        assert 0.0 < norm <= 1.0 + 1e-3, f"prototype norm {norm} drifted"


# ---------------------------------------------------------------------------
# Goal 6: alias bypass
# ---------------------------------------------------------------------------


class TestAliasBypass:
    def test_case_variation_still_resolves(self):
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        eid = g.upsert_entity("Alice", aliases=["Ali"])
        for s in ("alice", "ALICE", "Ali", "ali", "ALI"):
            assert g.lookup_by_surface(s) == eid
        g.close()

    def test_unicode_normalisation_distinguished(self):
        """Composed vs decomposed unicode (é as 1 char vs e + combining accent)
        produce distinct entities -- alias normalisation is OUT OF SCOPE for
        the PoC and the test documents that fact."""
        from holograph.graph.substrate import GraphSubstrate
        g = GraphSubstrate(":memory:")
        composed = "Café"
        decomposed = "Café"
        e1 = g.upsert_entity(composed)
        e2 = g.upsert_entity(decomposed)
        # PoC behaviour: not unified.  A production hardening pass should
        # apply NFC normalisation.
        assert e1 != e2
        g.close()


# ---------------------------------------------------------------------------
# Goal 7: numerical instability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_similarity_with_extreme_magnitude(self):
        from holograph.hdc.kernel import RealKernel
        k = RealKernel(dim=128)
        v_big = np.full(128, 1e30, dtype=np.float32)
        v_one = np.ones(128, dtype=np.float32)
        sim = k.similarity(v_big, v_one)
        assert np.isfinite(sim)

    def test_bundle_with_zero_input(self):
        from holograph.hdc.kernel import RealKernel
        k = RealKernel(dim=128)
        zeros = np.zeros(128, dtype=np.float32)
        out = k.bundle([zeros, zeros, zeros])
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Goal 8: native extension supply-chain
# ---------------------------------------------------------------------------


class TestNativeLoaderSafety:
    def test_loader_fails_safely_when_extension_corrupt(self):
        """If `holograph._native` exists but raises on use, the loader must
        fall back to pure-Python rather than crash the runtime."""
        import importlib

        class _Stub:
            def __getattr__(self, name):
                raise RuntimeError("simulated corrupt native extension")

        # Insert a fake module that raises on attribute access.
        original_native = sys.modules.get("holograph._native")
        sys.modules["holograph._native"] = _Stub()
        try:
            # Force a fresh loader import to pick up the new module shape.
            sys.modules.pop("holograph.hdc._native_loader", None)
            from holograph.hdc import _native_loader as nl  # noqa: F401
            # Even when the native module imports successfully but raises on
            # access, make_kernel(prefer_native=True) must produce a usable
            # kernel (falling back to pure-Python under the hood).
            from holograph.hdc.kernel import make_kernel
            k = make_kernel("real", dim=128, prefer_native=True)
            rng = np.random.default_rng(0)
            v = rng.standard_normal(128).astype(np.float32)
            assert k.bind(v, v).shape == (128,)
        finally:
            # Restore module table.
            if original_native is None:
                sys.modules.pop("holograph._native", None)
            else:
                sys.modules["holograph._native"] = original_native
            sys.modules.pop("holograph.hdc._native_loader", None)


# ---------------------------------------------------------------------------
# End-of-run pipeline still healthy
# ---------------------------------------------------------------------------


def test_pipeline_healthy_after_adversarial_run(spacy_nlp):
    """After all the adversarial probing, the runtime must still work."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=1024, spacy_nlp=spacy_nlp)
    hg.ingest_text("Sanity check: Alice met Bob.", anchor="d-final")
    out = hg.read("Who met Bob?")
    assert out.activated_ids
    hg.close()
