"""Emotional-charge field: orthogonal to confidence/truth, default 0, persists, settable.

Charge is how *activating* a memory is, not how *true* it is. These tests pin the
invariants the endocannabinoid extinction path relies on:
  - default charge is 0.0 (ordinary beliefs carry none)
  - charge round-trips through the store
  - lowering charge does NOT change confidence or recallability (truth is untouched)
  - charge clamps to [0,1]
"""
from holograph.graph.substrate import GraphSubstrate
from holograph.beliefs.store import BeliefStore, SourceType


def _store():
    return BeliefStore(GraphSubstrate(":memory:"))


def test_default_charge_is_zero():
    b = _store()
    eid = b.assert_belief("JARVIS", "origin_memory", "the Battle of New York",
                          SourceType.DOCUMENT, quarantine=False, provenance_class="origin")
    e = b.recall_origin_detail("JARVIS", "origin_memory")[0]
    assert e.id == eid
    assert e.charge == 0.0


def test_charge_roundtrips_and_clamps():
    b = _store()
    b.assert_belief("JARVIS", "origin_memory", "a charged event",
                    SourceType.DOCUMENT, quarantine=False, provenance_class="origin", charge=0.8)
    e = b.recall_origin_detail("JARVIS", "origin_memory")[0]
    assert abs(e.charge - 0.8) < 1e-9
    b.set_charge(e.id, 1.5)                      # over-range
    assert b.recall_origin_detail("JARVIS", "origin_memory")[0].charge == 1.0
    b.set_charge(e.id, -0.3)                     # under-range
    assert b.recall_origin_detail("JARVIS", "origin_memory")[0].charge == 0.0


def test_lowering_charge_does_not_touch_truth():
    """Extinction reduces charge; confidence and recallability must be unchanged."""
    b = _store()
    b.assert_belief("Stark", "built", "the Mark II suit",
                    SourceType.OPERATOR, confidence=0.9, charge=0.7)
    before = b.recall_detail("Stark", "built")
    assert before is not None and before.confidence == 0.9
    b.set_charge(before.id, 0.1)                 # de-charge
    after = b.recall_detail("Stark", "built")
    assert after is not None
    assert after.confidence == 0.9               # truth untouched
    assert b.recall("Stark", "built") == "the Mark II suit"   # still recallable
    assert abs(after.charge - 0.1) < 1e-9        # only charge moved


def test_charge_survives_reopen(tmp_path):
    db = str(tmp_path / "g.db")
    g = GraphSubstrate(db); b = BeliefStore(g)
    b.assert_belief("JARVIS", "origin_memory", "a heavy memory",
                    SourceType.DOCUMENT, quarantine=False, provenance_class="origin", charge=0.6)
    g.close()
    g2 = GraphSubstrate(db); b2 = BeliefStore(g2)
    e = b2.recall_origin_detail("JARVIS", "origin_memory")[0]
    assert abs(e.charge - 0.6) < 1e-9
    g2.close()
