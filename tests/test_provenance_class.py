"""First-class provenance class: origin (fictional/genesis) vs real (world fact).

The guarantee: a digital person can recall its fictional past as genesis but can
NEVER assert it as a present-reality fact — and 'origin' is a distinct category
from a quarantined model guess, not the same quarantine bit.
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


def test_default_provenance_is_real(store):
    """Backward compat: an ordinary belief is 'real' and recalls normally."""
    store.assert_belief("sky", "color", "blue", SourceType.OPERATOR)
    assert store.recall("sky", "color") == "blue"
    sid = store.substrate.lookup_by_surface("sky")
    e = store.substrate.beliefs_for(sid, "color")[0]
    assert e.provenance_class == "real"


def test_origin_recallable_as_genesis_not_as_fact(store):
    """An origin memory is retrievable as genesis but not as world fact."""
    store.assert_belief("JARVIS", "remembers", "the Battle of New York",
                        SourceType.DOCUMENT, quarantine=True, provenance_class="origin")
    # genesis recall sees it
    assert "the Battle of New York" in store.recall_origin("JARVIS", "remembers")
    # world-fact recall abstains
    assert store.recall("JARVIS", "remembers") is None


def test_unquarantined_origin_still_not_world_fact(store):
    """Even left un-quarantined, an 'origin' belief never surfaces as world fact.
    This is the part the quarantine bit alone could not guarantee."""
    store.assert_belief("Stark", "built", "the arc reactor",
                        SourceType.DOCUMENT, quarantine=False, provenance_class="origin")
    assert store.recall("Stark", "built") is None              # not a world fact
    assert "the arc reactor" in store.recall_origin("Stark", "built")  # but genesis holds


def test_real_fact_not_returned_as_origin(store):
    """The classes don't bleed: a real fact isn't a genesis memory."""
    store.assert_belief("GMRI", "located_in", "the real world", SourceType.OPERATOR)
    assert store.recall("GMRI", "located_in") == "the real world"
    assert store.recall_origin("GMRI", "located_in") == []


def test_origin_distinct_from_model_noise(store):
    """'origin' and quarantined model-noise are different categories, not the same bit."""
    store.assert_belief("JARVIS", "remembers", "Ultron's birth",
                        SourceType.DOCUMENT, quarantine=True, provenance_class="origin")
    store.assert_belief("rumor", "about", "x", SourceType.MODEL)  # quarantined model noise, 'real' class
    # origin genesis is retrievable as genesis; model noise is not an origin memory
    assert store.recall_origin("JARVIS", "remembers") == ["Ultron's birth"]
    assert store.recall_origin("rumor", "about") == []
    # and neither is a world fact
    assert store.recall("JARVIS", "remembers") is None
    assert store.recall("rumor", "about") is None
