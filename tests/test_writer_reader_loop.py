"""Integration tests for the writer-reader-feedback loop."""

from __future__ import annotations

import numpy as np
import pytest

from holograph.runtime import HoloGraph


@pytest.fixture(scope="module")
def populated_runtime(spacy_nlp):
    hg = HoloGraph(kernel_kind="real", dim=2048, top_k=6, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob in Paris.", anchor="doc-a")
    hg.ingest_text("Bob is a friend of Carol.", anchor="doc-b")
    hg.ingest_text("Carol lives in Berlin.", anchor="doc-c")
    yield hg
    hg.close()


def test_ingest_creates_entities_and_edges(populated_runtime):
    s = populated_runtime.summary()
    assert s["entities"] >= 4  # Alice, Bob, Paris, Carol, Berlin
    assert s["edges"] >= 2


def test_read_returns_activations(populated_runtime):
    out = populated_runtime.read("Where did Alice meet Bob?")
    assert out.plan.mentions
    assert out.activated_ids
    # Alice should be top-1 or top-2.
    top2 = out.activated_ids[:2]
    alice = populated_runtime.substrate.lookup_by_surface("Alice")
    assert alice in top2


def test_feedback_updates_edge_weights(populated_runtime):
    # Take a snapshot of edge weights, run feedback, check at least one edge weight changed.
    before = {e.id: e.weight for e in populated_runtime.substrate.all_edges()}
    alice = populated_runtime.substrate.lookup_by_surface("Alice")
    paris = populated_runtime.substrate.lookup_by_surface("Paris")
    ev = populated_runtime.feedback(
        query="Where did Alice meet Bob?",
        gold_doc_anchors=["doc-a"],
        gold_entity_ids=[i for i in (alice, paris) if i is not None],
    )
    after = {e.id: e.weight for e in populated_runtime.substrate.all_edges()}
    assert any(before[i] != after[i] for i in before if i in after)
    assert ev.reward.recall >= 0.0


def test_reader_attribution_keys_match_paths(populated_runtime):
    out = populated_runtime.read("Where did Alice meet Bob?")
    assert out.attribution is not None
    # Every per-path entry should reference an entity pair that appears in
    # the activated subgraph edges OR be empty if no edges connected.
    if out.attribution.per_path_top:
        valid_edges = {(u, r, v) for (u, r, v) in out.activated_subgraph_edges}
        for (u, r, v), _ in out.attribution.per_path_top:
            assert (u, r, v) in valid_edges


def test_ternary_runtime_runs(spacy_nlp):
    hg = HoloGraph(kernel_kind="ternary", dim=4096, top_k=6, spacy_nlp=spacy_nlp)
    hg.ingest_text("Alice met Bob.", anchor="d1")
    out = hg.read("Who did Alice meet?")
    assert out.activated_ids
    hg.close()
