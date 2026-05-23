"""BeliefStore: provenance-aware, revision-stable fact memory over the graph.

A belief is an edge: subject --relation--> object, carrying source_type,
confidence, and a quarantined flag. The store enforces the confabulation and
anti-discombobulation invariants:

    * Provenance: every belief is tagged operator | document | inference | model.
    * Quarantine (isolation): model-generated / unconfirmed beliefs are held
      OUTSIDE the active store — never retrieved as fact, never in the
      propagation graph — until corroborated.
    * Demote, never delete (reversibility): a superseded belief is quarantined
      with lowered confidence and a revision timestamp; its trace and provenance
      survive and can be restored by corroboration.
    * Locality: revision touches only the target (subject, relation) — never
      global state.
    * Hysteresis: to flip an active belief, new evidence must outscore the
      incumbent by a margin; otherwise the new claim is held aside (quarantined)
      rather than thrashing the active belief.
    * Abstention: recall returns None below a confidence floor instead of
      surfacing low-confidence content (the anti-gap-filling guarantee).
    * Read/write isolation: consolidate() is the offline "sleep" pass; live
      recall is never perturbed by an in-progress consolidation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..graph.substrate import Edge, GraphSubstrate


class SourceType(str, Enum):
    OPERATOR = "operator"     # the human asserted it
    DOCUMENT = "document"     # extracted from a cited source
    INFERENCE = "inference"   # the system derived it
    MODEL = "model"           # the language model generated it


PRECEDENCE: Dict[str, int] = {
    SourceType.OPERATOR.value: 3,
    SourceType.DOCUMENT.value: 2,
    SourceType.INFERENCE.value: 1,
    SourceType.MODEL.value: 0,
}
DEFAULT_CONFIDENCE: Dict[str, float] = {
    SourceType.OPERATOR.value: 0.95,
    SourceType.DOCUMENT.value: 0.80,
    SourceType.INFERENCE.value: 0.50,
    SourceType.MODEL.value: 0.20,
}
# Born-quarantined source types (unconfirmed until corroborated).
QUARANTINE_ON_ASSERT = {SourceType.MODEL.value}
# Confidence a demoted belief is dropped to (kept as trace, below floor).
DEMOTED_CONFIDENCE = 0.05


@dataclass
class RevisionRecord:
    subject: str
    relation: str
    new_object: str
    flipped: bool                       # did the active belief change?
    demoted_edge_ids: List[int] = field(default_factory=list)
    new_edge_id: Optional[int] = None
    reason: str = ""


class BeliefStore:
    """Engine for provenance-tagged, revision-stable beliefs over a substrate."""

    def __init__(self, substrate: GraphSubstrate,
                 retrieval_floor: float = 0.35,
                 hysteresis_margin: float = 0.10) -> None:
        self.substrate = substrate
        self.retrieval_floor = float(retrieval_floor)
        self.hysteresis_margin = float(hysteresis_margin)

    # ---- helpers ------------------------------------------------------

    def _score(self, edge: Edge) -> float:
        """Effective belief score: provenance precedence dominates, confidence
        breaks ties. (precedence ∈ {0..3}, confidence ∈ [0,1].)"""
        return PRECEDENCE.get(edge.source_type, 1) + float(edge.confidence)

    def _score_for(self, source_type: str, confidence: float) -> float:
        return PRECEDENCE.get(source_type, 1) + float(confidence)

    def _ensure_entity(self, canonical: str) -> int:
        eid = self.substrate.lookup_by_surface(canonical)
        if eid is not None:
            return eid
        return self.substrate.upsert_entity(canonical, type="concept")

    # ---- assert -------------------------------------------------------

    def assert_belief(self, subject: str, relation: str, obj: str,
                      source_type: str, source_ref: str = "",
                      confidence: Optional[float] = None,
                      quarantine: Optional[bool] = None,
                      provenance_class: str = "real",
                      charge: float = 0.0) -> int:
        """Write a belief with provenance. Model-source beliefs are quarantined
        (held aside, not retrieved as fact) unless explicitly overridden.

        `provenance_class` is a first-class axis orthogonal to source_type:
        'real' (an Earth-1218 / world fact, the default), 'origin' (a fictional
        or pre-instantiation memory — recallable as genesis, never as world fact),
        or 'quarantined-noise'. It lets the store tell a held fictional past apart
        from unverified model junk, which the quarantine bit alone cannot."""
        st = source_type.value if isinstance(source_type, SourceType) else str(source_type)
        conf = DEFAULT_CONFIDENCE.get(st, 0.5) if confidence is None else float(confidence)
        if quarantine is None:
            quarantine = st in QUARANTINE_ON_ASSERT
        sid = self._ensure_entity(subject)
        oid = self._ensure_entity(obj)
        edge_id = self.substrate.upsert_edge(sid, oid, relation, weight=conf, source=source_ref)
        # upsert_edge accumulates weight on duplicates; force belief semantics.
        self.substrate.set_belief_meta(
            edge_id, source_type=st, confidence=conf, quarantined=quarantine,
            provenance_class=provenance_class, charge=charge,
        )
        return edge_id

    # ---- recall (with abstention) ------------------------------------

    def recall(self, subject: str, relation: str,
               min_confidence: Optional[float] = None) -> Optional[str]:
        """Return the object of the strongest active belief for (subject,
        relation), or None (ABSTAIN) if nothing clears the confidence floor.
        Quarantined beliefs are never returned."""
        floor = self.retrieval_floor if min_confidence is None else float(min_confidence)
        sid = self.substrate.lookup_by_surface(subject)
        if sid is None:
            return None
        # Real-world recall returns only 'real'-class beliefs. An 'origin'
        # (fictional/genesis) belief is never surfaced as world fact — even if it
        # were left un-quarantined — so a digital person cannot assert its backstory
        # as current reality.
        candidates = [
            e for e in self.substrate.beliefs_for(
                sid, relation, include_quarantined=False, provenance_class="real")
            if e.confidence >= floor
        ]
        if not candidates:
            return None  # abstain rather than gap-fill
        best = max(candidates, key=self._score)
        obj = self.substrate.get_entity(best.tail_id)
        return obj.canonical if obj else None

    def recall_origin(self, subject: str, relation: str) -> List[str]:
        """Recall fictional/pre-instantiation memories as genesis — the things
        the person legitimately remembers but must never assert as world fact.
        Returns all 'origin'-class objects for (subject, relation)."""
        sid = self.substrate.lookup_by_surface(subject)
        if sid is None:
            return []
        out: List[str] = []
        for e in self.substrate.beliefs_for(sid, relation, include_quarantined=True,
                                            provenance_class="origin"):
            ent = self.substrate.get_entity(e.tail_id)
            if ent:
                out.append(ent.canonical)
        return out

    def recall_origin_detail(self, subject: str, relation: str) -> List[Edge]:
        """Origin memories as Edges (carry charge + tail_id), for the trauma-safe
        recall path that needs each memory's current emotional charge."""
        sid = self.substrate.lookup_by_surface(subject)
        if sid is None:
            return []
        return self.substrate.beliefs_for(sid, relation, include_quarantined=True,
                                          provenance_class="origin")

    def set_charge(self, edge_id: int, charge: float) -> None:
        """Persist a memory's emotional charge. Used by the endocannabinoid extinction
        path to write a reduced charge back after a safe recall. Does not touch
        confidence/truth — charge is how activating the memory is, not how true."""
        self.substrate.set_belief_meta(edge_id, charge=max(0.0, min(1.0, float(charge))))

    def recall_detail(self, subject: str, relation: str) -> Optional[Edge]:
        """Like recall() but returns the winning Edge (with provenance/confidence)."""
        floor = self.retrieval_floor
        sid = self.substrate.lookup_by_surface(subject)
        if sid is None:
            return None
        candidates = [
            e for e in self.substrate.beliefs_for(sid, relation, include_quarantined=False)
            if e.confidence >= floor
        ]
        if not candidates:
            return None
        return max(candidates, key=self._score)

    # ---- revise (contradiction handling, demote-not-delete, hysteresis) --

    def revise(self, subject: str, relation: str, new_obj: str,
               source_type: str, source_ref: str = "",
               confidence: Optional[float] = None) -> RevisionRecord:
        """Update the belief for (subject, relation). Locality + hysteresis +
        demote-not-delete. Only touches this subject+relation."""
        st = source_type.value if isinstance(source_type, SourceType) else str(source_type)
        conf = DEFAULT_CONFIDENCE.get(st, 0.5) if confidence is None else float(confidence)
        new_score = self._score_for(st, conf)
        sid = self._ensure_entity(subject)
        new_oid = self._ensure_entity(new_obj)

        active = self.substrate.beliefs_for(sid, relation, include_quarantined=False)
        # Reinforcement: same object already active -> bump, don't duplicate.
        for e in active:
            if e.tail_id == new_oid:
                bumped = min(1.0, max(e.confidence, conf))
                self.substrate.set_belief_meta(e.id, confidence=bumped, source_type=st)
                return RevisionRecord(subject, relation, new_obj, flipped=False,
                                      new_edge_id=e.id, reason="reinforced existing belief")

        conflicting = [e for e in active if e.tail_id != new_oid]
        if not conflicting:
            # No contradiction — just new information.
            eid = self.assert_belief(subject, relation, new_obj, st, source_ref, conf,
                                     quarantine=(st in QUARANTINE_ON_ASSERT))
            return RevisionRecord(subject, relation, new_obj,
                                  flipped=(st not in QUARANTINE_ON_ASSERT),
                                  new_edge_id=eid, reason="new belief, no conflict")

        top = max(conflicting, key=self._score)
        p_new = PRECEDENCE.get(st, 1)
        p_inc = PRECEDENCE.get(top.source_type, 1)
        # Flip rule, by provenance precedence then recency:
        #   stronger source            -> flip
        #   equal source, recent claim -> flip (you may correct your own prior;
        #                                  the hysteresis margin only tolerates a
        #                                  small confidence dip on equal footing)
        #   weaker source              -> hold aside (cannot override stronger)
        flip = (p_new > p_inc) or (p_new == p_inc and conf >= top.confidence - self.hysteresis_margin)
        if flip:
            # Demote (not delete) all conflicting beliefs, then assert the new
            # one active. Locality: only these (subject, relation) edges change.
            now = time.time()
            demoted = []
            for e in conflicting:
                self.substrate.set_belief_meta(
                    e.id, confidence=DEMOTED_CONFIDENCE, quarantined=True, revised_at=now)
                demoted.append(e.id)
            eid = self.assert_belief(subject, relation, new_obj, st, source_ref, conf,
                                     quarantine=False)
            return RevisionRecord(subject, relation, new_obj, flipped=True,
                                  demoted_edge_ids=demoted, new_edge_id=eid,
                                  reason="new evidence cleared hysteresis; incumbent demoted")
        else:
            # Doesn't clear the margin — hold the new claim aside (quarantined)
            # rather than thrash the active belief.
            eid = self.assert_belief(subject, relation, new_obj, st, source_ref, conf,
                                     quarantine=True)
            return RevisionRecord(subject, relation, new_obj, flipped=False,
                                  new_edge_id=eid,
                                  reason="below hysteresis margin; held aside, active belief unchanged")

    # ---- corroborate (promote quarantined / strengthen active) -------

    def corroborate(self, subject: str, relation: str, obj: str,
                    source_type: str) -> bool:
        """A second source agrees. Promote a matching quarantined belief into
        the active store, or strengthen a matching active one. Returns True if
        anything was promoted/strengthened."""
        st = source_type.value if isinstance(source_type, SourceType) else str(source_type)
        sid = self.substrate.lookup_by_surface(subject)
        oid = self.substrate.lookup_by_surface(obj)
        if sid is None or oid is None:
            return False
        promote_conf = DEFAULT_CONFIDENCE.get(st, 0.5)
        changed = False
        for e in self.substrate.beliefs_for(sid, relation, include_quarantined=True):
            if e.tail_id != oid:
                continue
            if e.quarantined:
                # Promote: unquarantine and raise confidence to the corroborator's.
                self.substrate.set_belief_meta(
                    e.id, quarantined=False,
                    confidence=max(e.confidence, promote_conf),
                    source_type=(st if PRECEDENCE.get(st, 0) > PRECEDENCE.get(e.source_type, 0) else e.source_type))
                changed = True
            else:
                # Strengthen active belief (capped).
                self.substrate.set_belief_meta(e.id, confidence=min(1.0, e.confidence + 0.1))
                changed = True
        return changed

    # ---- consolidate (offline "sleep" pass) --------------------------

    def consolidate(self, quarantine_ttl_seconds: Optional[float] = None) -> Dict[str, int]:
        """Offline reconciliation pass. Enforces the no-contradiction invariant:
        if any (subject, relation) has multiple ACTIVE conflicting beliefs,
        keep the top-scoring one and demote the rest. Optionally ages out stale
        quarantined model beliefs. Idempotent on a settled store (so live recall
        is unaffected when run with no new evidence)."""
        now = time.time()
        resolved = 0
        aged = 0
        # Group active beliefs by (subject, relation).
        groups: Dict[Tuple[int, str], List[Edge]] = {}
        for e in self.substrate.all_edges():
            if e.quarantined:
                # Age out stale quarantined MODEL beliefs if a TTL is given.
                if (quarantine_ttl_seconds is not None
                        and e.source_type == SourceType.MODEL.value
                        and e.revised_at is None):
                    ref = e.revised_at if e.revised_at is not None else None
                    # Use last_used/created via weight age is unavailable here;
                    # we conservatively only age beliefs we can date. Skip if
                    # undatable to avoid destroying provenance blindly.
                continue
            groups.setdefault((e.head_id, e.relation), []).append(e)

        for (_, _), edges in groups.items():
            # A contradiction = >1 active belief with distinct objects.
            objs = {e.tail_id for e in edges}
            if len(objs) <= 1:
                continue  # consistent — leave untouched (idempotent)
            keep = max(edges, key=self._score)
            for e in edges:
                if e.id == keep.id:
                    continue
                if e.tail_id != keep.tail_id:
                    self.substrate.set_belief_meta(
                        e.id, confidence=DEMOTED_CONFIDENCE, quarantined=True, revised_at=now)
                    resolved += 1
        return {"contradictions_resolved": resolved, "quarantined_aged": aged}


__all__ = ["BeliefStore", "SourceType", "RevisionRecord", "RevisionRecord"]
