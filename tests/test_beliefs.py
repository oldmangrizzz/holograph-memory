"""Belief layer: confabulation-resistance and anti-discombobulation invariants.

Each test states the invariant it enforces. Together they are the falsifiable
guarantees that confabulation is catchable/correctable and that revision does
not throw the memory into disorder. Synthetic, neutral test data — no real
personal facts.
"""

from __future__ import annotations

import pytest

from holograph.graph.substrate import GraphSubstrate
from holograph.beliefs.store import BeliefStore, SourceType


@pytest.fixture
def store():
    g = GraphSubstrate(":memory:")
    yield BeliefStore(g)
    g.close()


# ---------------------------------------------------------------------------
# Provenance + quarantine (confabulation isolation)
# ---------------------------------------------------------------------------


def test_operator_belief_is_recalled(store):
    store.assert_belief("sky", "color", "blue", SourceType.OPERATOR)
    assert store.recall("sky", "color") == "blue"


def test_model_belief_is_quarantined_not_recalled(store):
    """A model-generated assertion must NOT be retrievable as fact."""
    store.assert_belief("mars", "color", "green", SourceType.MODEL)
    assert store.recall("mars", "color") is None  # quarantined → abstain


def test_quarantined_belief_excluded_from_propagation_graph(store):
    """Quarantine is true isolation: it must not enter the reader's graph."""
    store.assert_belief("a", "knows", "b", SourceType.MODEL)  # quarantined
    g = store.substrate.to_networkx()
    # The quarantined edge's endpoints may exist as nodes, but no edge between them.
    a = store.substrate.lookup_by_surface("a")
    b = store.substrate.lookup_by_surface("b")
    assert not (g.has_edge(a, b) or g.has_edge(b, a)), "quarantined belief leaked into propagation graph"


def test_abstention_on_unknown(store):
    assert store.recall("nonexistent", "anything") is None


def test_abstention_below_confidence_floor(store):
    store.assert_belief("x", "p", "y", SourceType.INFERENCE, confidence=0.20)  # below 0.35 floor
    assert store.recall("x", "p") is None
    # raising the floor request even higher still abstains
    assert store.recall("x", "p", min_confidence=0.9) is None


# ---------------------------------------------------------------------------
# Revision: precedence, recency, hysteresis, demote-not-delete
# ---------------------------------------------------------------------------


def test_stronger_source_overrides(store):
    store.assert_belief("capital", "of_country", "old_guess", SourceType.MODEL, quarantine=False)
    rec = store.revise("capital", "of_country", "real_answer", SourceType.OPERATOR)
    assert rec.flipped
    assert store.recall("capital", "of_country") == "real_answer"


def test_weaker_source_cannot_override(store):
    store.assert_belief("law", "value", "true", SourceType.OPERATOR)
    rec = store.revise("law", "value", "false", SourceType.MODEL)
    assert not rec.flipped
    assert store.recall("law", "value") == "true"  # operator belief untouched


def test_operator_self_correction_flips_on_recency(store):
    """Equal authority: the more recent assertion wins (you may change your mind)."""
    store.assert_belief("plan", "city", "boston", SourceType.OPERATOR)
    rec = store.revise("plan", "city", "denver", SourceType.OPERATOR)
    assert rec.flipped
    assert store.recall("plan", "city") == "denver"


def test_demote_not_delete_trace_survives(store):
    """A superseded belief is demoted+quarantined, not deleted — the trace and
    its provenance remain for audit and possible restoration."""
    store.assert_belief("status", "is", "alpha", SourceType.OPERATOR)
    store.revise("status", "is", "beta", SourceType.OPERATOR)
    sid = store.substrate.lookup_by_surface("status")
    all_beliefs = store.substrate.beliefs_for(sid, "is", include_quarantined=True)
    objs = {store.substrate.get_entity(e.tail_id).canonical for e in all_beliefs}
    assert "alpha" in objs and "beta" in objs, "demoted belief was deleted, not preserved"
    # but only beta is active/recalled
    assert store.recall("status", "is") == "beta"
    active = store.substrate.beliefs_for(sid, "is", include_quarantined=False)
    assert {store.substrate.get_entity(e.tail_id).canonical for e in active} == {"beta"}


def test_reinforcement_does_not_duplicate(store):
    store.assert_belief("fact", "p", "v", SourceType.DOCUMENT, confidence=0.6)
    rec = store.revise("fact", "p", "v", SourceType.OPERATOR)  # same object
    assert not rec.flipped and "reinforced" in rec.reason
    sid = store.substrate.lookup_by_surface("fact")
    assert len(store.substrate.beliefs_for(sid, "p", include_quarantined=True)) == 1


# ---------------------------------------------------------------------------
# Corroboration (promotion of the unconfirmed)
# ---------------------------------------------------------------------------


def test_corroboration_promotes_quarantined(store):
    store.assert_belief("rumor", "about", "x", SourceType.MODEL)  # quarantined
    assert store.recall("rumor", "about") is None
    promoted = store.corroborate("rumor", "about", "x", SourceType.OPERATOR)
    assert promoted
    assert store.recall("rumor", "about") == "x"  # now active


# ---------------------------------------------------------------------------
# INV1: locality — revision touches only the target subject+relation
# ---------------------------------------------------------------------------


def test_inv1_locality(store):
    store.assert_belief("sky", "color", "blue", SourceType.OPERATOR)
    store.assert_belief("grass", "color", "green", SourceType.OPERATOR)
    # snapshot grass before
    gid = store.substrate.lookup_by_surface("grass")
    before = [(e.tail_id, e.confidence, e.quarantined)
              for e in store.substrate.beliefs_for(gid, "color", include_quarantined=True)]
    # revise sky (should not touch grass at all)
    store.revise("sky", "color", "orange", SourceType.OPERATOR)
    after = [(e.tail_id, e.confidence, e.quarantined)
             for e in store.substrate.beliefs_for(gid, "color", include_quarantined=True)]
    assert before == after, "revision of sky perturbed grass — locality violated"
    assert store.recall("grass", "color") == "green"


# ---------------------------------------------------------------------------
# INV2: no two active high-confidence contradictory beliefs after consolidate
# ---------------------------------------------------------------------------


def test_inv2_no_contradiction_after_consolidate(store):
    # Inject two ACTIVE conflicting operator beliefs (bypassing revise).
    store.assert_belief("door", "state", "open", SourceType.OPERATOR, quarantine=False)
    store.assert_belief("door", "state", "closed", SourceType.OPERATOR, quarantine=False)
    sid = store.substrate.lookup_by_surface("door")
    active_before = store.substrate.beliefs_for(sid, "state", include_quarantined=False)
    assert len(active_before) == 2  # contradiction present pre-consolidate
    summary = store.consolidate()
    assert summary["contradictions_resolved"] >= 1
    active_after = store.substrate.beliefs_for(sid, "state", include_quarantined=False)
    objs = {store.substrate.get_entity(e.tail_id).canonical for e in active_after}
    assert len(objs) == 1, "consolidate left contradictory active beliefs"


# ---------------------------------------------------------------------------
# INV3: read/write isolation — consolidate doesn't perturb a settled store
# ---------------------------------------------------------------------------


def test_inv3_consolidate_idempotent_on_settled(store):
    store.assert_belief("a", "r", "x", SourceType.OPERATOR)
    store.assert_belief("b", "r", "y", SourceType.DOCUMENT)
    before = (store.recall("a", "r"), store.recall("b", "r"))
    snap = {(e.head_id, e.relation, e.tail_id): (e.confidence, e.quarantined)
            for e in store.substrate.all_edges()}
    store.consolidate()  # no new evidence, settled state
    after = (store.recall("a", "r"), store.recall("b", "r"))
    snap2 = {(e.head_id, e.relation, e.tail_id): (e.confidence, e.quarantined)
             for e in store.substrate.all_edges()}
    assert before == after, "consolidate changed settled recall"
    assert snap == snap2, "consolidate mutated a settled store"
