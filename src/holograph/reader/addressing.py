"""Soft addressing: compute the multi-cue stimulus se(q) per entity.

This implements SAGE's Eq. (1) faithfully (modulo our simpler entity-linking
fallback):

    se(q) = lambda1 * Exact(e, E_exp)
          + lambda2 * Alias(e, A)
          + lambda3 * max_m cos(Emb(desc(e)), Emb(q_m))
          + lambda4 * Type(e, tau)
          + lambda5 * Cons(e, C_hard)
          + lambda6 * sum_{ξ in NER(q)} EL(e | ξ)

Then a softmax with temperature T0 yields the initial activation distribution
p0(e | q) over entities -- this is the input to the GNN propagation step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph.substrate import GraphSubstrate, Entity
from .planner import QueryPlan


# Default scoring weights, tuned to give exact/alias matches dominant influence
# while keeping semantic similarity, type, and constraint channels meaningful.
DEFAULT_WEIGHTS = {
    "exact": 3.0,
    "alias": 2.0,
    "semantic": 1.5,
    "type": 1.0,
    "constraint": 0.75,
    "entity_link": 1.25,
}


@dataclass
class AddressingResult:
    scores: Dict[int, float] = field(default_factory=dict)        # raw stimulus se(q)
    activation: Dict[int, float] = field(default_factory=dict)    # softmaxed p0(e|q)
    per_channel: Dict[int, Dict[str, float]] = field(default_factory=dict)


class SoftAddresser:
    """Compute se(q) and p0(e|q) for every entity in the substrate."""

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 temperature: float = 0.5) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    # ---- main entrypoint ---------------------------------------------

    def address(self,
                substrate: GraphSubstrate,
                plan: QueryPlan,
                entity_vectors: Optional[Dict[int, np.ndarray]] = None,
                candidate_ids: Optional[set] = None) -> AddressingResult:
        """Return the activation distribution over entities.

        Args:
            substrate: the graph substrate.
            plan: the structured QueryPlan from the planner.
            entity_vectors: optional pre-computed dense vectors per entity
                (e.g. spaCy doc vectors of the entity description).  If None,
                semantic similarity contributes 0 for entities without vectors.
            candidate_ids: if given, restrict scoring to this subset (the
                H-MEM efficiency win — score only the routed candidates, not
                every entity). Leaf entities only.
        """
        all_ents = [e for e in substrate.all_entities() if e.layer == 0]
        if candidate_ids is not None:
            ents = [e for e in all_ents if e.id in candidate_ids]
        else:
            ents = all_ents
        if not ents:
            return AddressingResult()

        result = AddressingResult()
        q_vec = plan.query_vector
        probe_vecs: List[np.ndarray] = []
        # We do not depend on spaCy here at call time; the planner provided q_vec.
        # Probes also reuse q_vec for simplicity (sentence-level decomposition
        # would re-encode each probe; not necessary for correctness).
        if q_vec is not None:
            probe_vecs.append(q_vec)

        exact_set = {m.lower() for m in plan.mentions}
        alias_set = {a.lower() for a in plan.aliases}
        hard_set = {h.lower() for h in plan.hard_constraints}
        target_type = plan.target_type

        for ent in ents:
            channels: Dict[str, float] = {}
            channels["exact"] = self._exact_score(ent, exact_set)
            channels["alias"] = self._alias_score(ent, alias_set)
            channels["semantic"] = self._semantic_score(ent, entity_vectors, probe_vecs)
            channels["type"] = self._type_score(ent, target_type)
            channels["constraint"] = self._constraint_score(ent, hard_set)
            channels["entity_link"] = self._entity_link_score(ent, exact_set, alias_set)

            stimulus = sum(self.weights[k] * v for k, v in channels.items())
            result.scores[ent.id] = stimulus
            result.per_channel[ent.id] = channels

        # Softmax with temperature.
        ids = list(result.scores.keys())
        if not ids:
            return result
        s = np.array([result.scores[i] for i in ids], dtype=np.float64)
        s = s / self.temperature
        s = s - s.max()
        p = np.exp(s)
        psum = float(p.sum())
        if psum > 0:
            p = p / psum
        result.activation = {ids[i]: float(p[i]) for i in range(len(ids))}
        return result

    # ---- per-channel scorers -----------------------------------------

    @staticmethod
    def _exact_score(ent: Entity, exact_set: set[str]) -> float:
        if not exact_set:
            return 0.0
        canon = ent.canonical.lower()
        if canon in exact_set:
            return 1.0
        # Partial: word-level Jaccard >= 0.6 counts as a weakened match.
        canon_words = set(canon.split())
        best = 0.0
        for m in exact_set:
            m_words = set(m.lower().split())
            if not canon_words or not m_words:
                continue
            j = len(canon_words & m_words) / len(canon_words | m_words)
            if j >= 0.6:
                best = max(best, j)
        return best

    @staticmethod
    def _alias_score(ent: Entity, alias_set: set[str]) -> float:
        if not alias_set:
            return 0.0
        ent_aliases = {ent.canonical.lower(), *(a.lower() for a in ent.aliases)}
        if ent_aliases & alias_set:
            return 1.0
        # Substring fallback to catch lemma variants and possessives.
        for a in alias_set:
            for ea in ent_aliases:
                if a and ea and (a in ea or ea in a):
                    return 0.5
        return 0.0

    @staticmethod
    def _semantic_score(ent: Entity,
                        entity_vectors: Optional[Dict[int, np.ndarray]],
                        probe_vecs: List[np.ndarray]) -> float:
        if not entity_vectors or not probe_vecs:
            return 0.0
        v = entity_vectors.get(ent.id)
        if v is None:
            return 0.0
        best = 0.0
        for q in probe_vecs:
            nv = float(np.linalg.norm(v))
            nq = float(np.linalg.norm(q))
            if nv == 0 or nq == 0:
                continue
            best = max(best, float(np.dot(v, q) / (nv * nq)))
        return max(0.0, best)

    @staticmethod
    def _type_score(ent: Entity, target_type: Optional[str]) -> float:
        if not target_type:
            return 0.0
        return 1.0 if (ent.type or "").lower() == target_type.lower() else 0.0

    @staticmethod
    def _constraint_score(ent: Entity, hard_set: set[str]) -> float:
        if not hard_set:
            return 0.0
        canon = ent.canonical.lower()
        desc = (ent.description or "").lower()
        for h in hard_set:
            if h and (h in canon or h in desc):
                return 1.0
        return 0.0

    @staticmethod
    def _entity_link_score(ent: Entity, exact_set: set[str], alias_set: set[str]) -> float:
        """Light entity-linking proxy: fuzzy match against aliases / canonical."""
        if not exact_set and not alias_set:
            return 0.0
        haystack = {ent.canonical.lower(), *(a.lower() for a in ent.aliases)}
        targets = exact_set | alias_set
        # Levenshtein-free fuzzy: shared-prefix ratio, capped at 1.
        best = 0.0
        for h in haystack:
            for t in targets:
                if not h or not t:
                    continue
                common = 0
                for a, b in zip(h, t):
                    if a == b:
                        common += 1
                    else:
                        break
                denom = max(len(h), len(t))
                ratio = common / denom if denom else 0.0
                if ratio >= 0.5:
                    best = max(best, ratio)
        return best


__all__ = ["SoftAddresser", "AddressingResult"]
