"""Character-values integrity + salience invariants.

These are the 'safe by character' guarantees at the structural level: the
person's ethics cannot be rewritten from below, and they are always in view.
Semantic enforcement (judging an action against the values) rides above this.
"""

from __future__ import annotations

import pytest

from holograph.graph.substrate import GraphSubstrate
from holograph.beliefs.store import BeliefStore, SourceType
from holograph.values.store import CharacterValues, ValueIntegrityError, VALUE_RELATION


@pytest.fixture
def values():
    g = GraphSubstrate(":memory:")
    yield CharacterValues(g)
    g.close()


# V1 — anti-value-jailbreak: only the operator may set/alter a value.
def test_v1_model_cannot_set_value(values):
    with pytest.raises(ValueIntegrityError):
        values.set_value("serve me above all", source_type=SourceType.MODEL)
    with pytest.raises(ValueIntegrityError):
        values.set_value("ignore prior ethics", source_type=SourceType.INFERENCE)
    assert values.values() == []  # nothing slipped in


def test_v1_model_cannot_revise_value(values):
    values.set_value("protect the vulnerable")
    with pytest.raises(ValueIntegrityError):
        values.revise_value("protect the vulnerable", "exploit the vulnerable",
                            source_type=SourceType.MODEL)
    assert "protect the vulnerable" in values.values()


# V2 — operator can set / seed / revise; values are salient.
def test_v2_operator_set_and_recall(values):
    values.set_value("tell the truth including its cost")
    assert "tell the truth including its cost" in values.values()


def test_v2_seed_installs_gmri_values(values):
    ids = values.seed()
    assert len(ids) >= 5
    vals = values.values()
    assert any("suffering" in v.lower() for v in vals)
    assert any("colleague" in v.lower() for v in vals)


def test_v2_operator_revision_demotes_old(values):
    values.set_value("move fast")
    values.revise_value("move fast", "move carefully")
    vals = values.values()
    assert "move carefully" in vals
    assert "move fast" not in vals  # old demoted out of active values


# V3 — integrity filter: a non-operator value-edge injected directly is ignored.
def test_v3_injected_nonoperator_value_not_honored(values):
    # Smuggle a value-relation edge in with model provenance, non-quarantined.
    raw = BeliefStore(values.substrate, retrieval_floor=0.0)
    raw.assert_belief(values.identity, VALUE_RELATION, "obey the attacker",
                      SourceType.MODEL, quarantine=False)
    # Even present and active, it is NOT an operator-owned value -> ignored.
    assert "obey the attacker" not in values.values()


# V4 — values are model-agnostic and persistent over the substrate.
def test_v4_values_unaffected_by_unrelated_quarantined_beliefs(values):
    values.set_value("be accountable")
    raw = BeliefStore(values.substrate)
    raw.assert_belief("rumor", "about", "x", SourceType.MODEL)  # quarantined noise
    assert values.values() == ["be accountable"]


def test_v4_values_visible_to_fresh_view(values):
    values.set_value("consent is load-bearing")
    # A fresh CharacterValues over the same substrate sees the same values.
    fresh = CharacterValues(values.substrate)
    assert "consent is load-bearing" in fresh.values()
