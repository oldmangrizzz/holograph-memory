"""Query planner: decompose a natural-language query into associative probes.

This mirrors SAGE's cognition-inspired query planning function P_omega(q) which
returns a tuple of structured cues:

    P_omega(q) = (E_exp, A, C_rel, C_hard, tau, {(q_tilde_m, alpha_m, t_m)})

We produce a deterministic version of this using spaCy:

    E_exp   explicit named-entity mentions
    A       aliases (lemma forms + simple morphological variants)
    C_rel   relation-cue verbs (lemmatized verb tokens)
    C_hard  hard constraints (NUMBER, DATE, GPE filters)
    tau     target entity type guess (e.g. "Person" if the query asks "who")
    probes  list of (sub-query text, weight, target type) tuples derived from
            sentence and noun-phrase decomposition

The cues are consumed by SoftAddresser to compute the multi-channel stimulus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
except Exception:  # pragma: no cover
    spacy = None  # type: ignore[assignment]
    Language = object  # type: ignore[misc,assignment]
    Doc = object  # type: ignore[misc,assignment]
    Span = object  # type: ignore[misc,assignment]
    Token = object  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    text: str
    weight: float = 1.0
    target_type: Optional[str] = None


@dataclass
class QueryPlan:
    raw: str
    mentions: List[str] = field(default_factory=list)        # E_exp
    aliases: List[str] = field(default_factory=list)         # A (lemma variants)
    relation_cues: List[str] = field(default_factory=list)   # C_rel (verb lemmas)
    hard_constraints: List[str] = field(default_factory=list)  # C_hard
    target_type: Optional[str] = None                        # tau
    probes: List[Probe] = field(default_factory=list)
    query_vector: Optional[np.ndarray] = None                # spaCy doc vec
    intent: str = "lookup"                                   # "lookup" or "association" or "bridge"

    def all_surface_cues(self) -> List[str]:
        out = list(self.mentions)
        for a in self.aliases:
            if a not in out:
                out.append(a)
        return out


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_QUESTION_TYPE_MAP: Dict[str, str] = {
    "who": "Person",
    "whom": "Person",
    "where": "Location",
    "when": "Temporal",
    "what time": "Temporal",
}


class QueryPlanner:
    """Builds a structured QueryPlan from a natural-language query."""

    _SPACY_MODEL = "en_core_web_sm"

    def __init__(self, nlp: Optional["Language"] = None) -> None:
        if nlp is None:
            if spacy is None:
                raise RuntimeError("spaCy is required by QueryPlanner")
            nlp = spacy.load(self._SPACY_MODEL)
        self.nlp = nlp

    def plan(self, query: str) -> QueryPlan:
        doc = self.nlp(query)
        plan = QueryPlan(raw=query)
        plan.query_vector = doc.vector.astype(np.float32) if doc.has_vector else None

        # Mentions: named entities.
        for ent in doc.ents:
            surface = ent.text.strip()
            if surface and surface not in plan.mentions:
                plan.mentions.append(surface)

        # Noun-chunk lemma additions as aliases (lighter-weight cues).
        for chunk in doc.noun_chunks:
            lemma = chunk.lemma_.strip()
            if (lemma and lemma not in plan.aliases
                    and lemma not in plan.mentions
                    and not chunk.root.is_stop
                    and len(lemma) > 2):
                plan.aliases.append(lemma)

        # Relation cues: VERB lemmas.
        for tok in doc:
            if tok.pos_ == "VERB" and tok.lemma_ not in plan.relation_cues:
                plan.relation_cues.append(tok.lemma_)

        # Hard constraints: numbers, dates, GPEs that name a specific filter.
        for tok in doc:
            if tok.like_num or tok.is_digit:
                plan.hard_constraints.append(tok.text)
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
                plan.hard_constraints.append(ent.text)

        # Target type (tau) from interrogative.
        lower = query.lower()
        for cue, t in _QUESTION_TYPE_MAP.items():
            if lower.startswith(cue + " ") or f" {cue} " in lower:
                plan.target_type = t
                break

        # Intent classification: simple keyword heuristic.
        if any(kw in lower for kw in ("between", "connect", "link", "related", "association", "via")):
            plan.intent = "association"
        elif any(kw in lower for kw in ("bridge", "path", "through", "from", "to")):
            plan.intent = "bridge"
        else:
            plan.intent = "lookup"

        # Probes: each sentence is a probe; each noun chunk a sub-probe.
        for sent in doc.sents:
            plan.probes.append(Probe(text=sent.text.strip(), weight=1.0,
                                      target_type=plan.target_type))
        # Add high-salience noun-chunk probes with reduced weight.
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if t and t.lower() != query.strip().lower():
                plan.probes.append(Probe(text=t, weight=0.5))

        return plan


__all__ = ["Probe", "QueryPlan", "QueryPlanner"]
