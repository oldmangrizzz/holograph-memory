"""Memory writer: ingests text into the graph substrate.

We follow SAGE's structured-policy framing but replace the GRPO RL loop with
a deterministic-extractor-plus-reward-shaped-weight-update.  This still closes
the SAGE-style feedback loop: when the reader downstream succeeds on a triple,
we boost the edge's weight; when it fails, we demote it.  Over many queries
the graph topology re-shapes itself by use, exactly as the SAGE paper argues.

Extraction is dependency-parse based using spaCy:
    1. Resolve named entities (PERSON, ORG, GPE, DATE, EVENT, etc.) and noun-
       phrase concepts as candidate entities.
    2. For each verb phrase, emit a triple (subject, lemma(verb), object) when
       both arguments resolve to candidate entities.
    3. Emit "is_a" triples for explicit appositive constructions
       ("X, a Y, ..." -> (X, is_a, Y)).
    4. Source-anchor every triple to the document id.

The writer is also responsible for ingesting STRUCTURED scalar data (PSP-style
demos) -- bypass extraction and write directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# spaCy is loaded lazily because en_core_web_sm import is expensive.
try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
except Exception:  # pragma: no cover - spaCy is a hard dependency at runtime
    spacy = None  # type: ignore[assignment]
    Language = object  # type: ignore[misc,assignment]
    Doc = object  # type: ignore[misc,assignment]
    Span = object  # type: ignore[misc,assignment]
    Token = object  # type: ignore[misc,assignment]

from ..graph.substrate import GraphSubstrate
from ..hdc.kernel import HDCKernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedTriple:
    head: str
    relation: str
    tail: str
    head_type: str = "concept"
    tail_type: str = "concept"
    source: str = ""

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class MemoryWriter:
    """Ingests text and structured data into the graph substrate.

    The writer holds a spaCy pipeline and the substrate.  It maintains a small
    per-relation tally used as a sanity input to the reward updates (frequent
    relations get smaller per-event reinforcement to avoid runaway weights).
    """

    _SPACY_MODEL = "en_core_web_sm"

    def __init__(self,
                 substrate: GraphSubstrate,
                 kernel: HDCKernel,
                 nlp: Optional["Language"] = None,
                 reinforce_eta: float = 0.5,
                 demote_eta: float = 0.25,
                 repetition_lambda: float = 0.1) -> None:
        if nlp is None:
            if spacy is None:
                raise RuntimeError(
                    "spaCy is required by MemoryWriter; install with `pip install spacy` and "
                    "`python -m spacy download en_core_web_sm`."
                )
            nlp = spacy.load(self._SPACY_MODEL)
        self.substrate = substrate
        self.kernel = kernel
        self.nlp = nlp
        self.reinforce_eta = float(reinforce_eta)
        self.demote_eta = float(demote_eta)
        self.repetition_lambda = float(repetition_lambda)
        self._relation_counts: Dict[str, int] = {}
        # A small fixed random projection from spaCy's tok2vec dim into the
        # kernel's hyperdim — used to seed entity hypervectors at insert time.
        self._entity_basis: Optional[np.ndarray] = None
        self._entity_basis_rows: int = 96  # spaCy small models output 96-dim vectors

    # ---- entity seeding -----------------------------------------------

    def _ensure_basis(self) -> np.ndarray:
        if self._entity_basis is None:
            self._entity_basis = self.kernel.random_basis(self._entity_basis_rows, seed=1234)
        return self._entity_basis

    def _seed_hv_for(self, surface: str, description: str = "") -> np.ndarray:
        """Build a hypervector seed for a new entity from its surface vector."""
        basis = self._ensure_basis()
        # Use the doc-level vector (avg of token tok2vec) as the input feature.
        text = f"{surface}. {description}".strip()
        doc = self.nlp(text)
        vec = doc.vector if doc.has_vector else np.zeros(self._entity_basis_rows, dtype=np.float32)
        if vec.shape[0] != self._entity_basis_rows:
            # Defensive resize / pad in case the loaded model differs.
            if vec.shape[0] > self._entity_basis_rows:
                vec = vec[: self._entity_basis_rows]
            else:
                pad = np.zeros(self._entity_basis_rows - vec.shape[0], dtype=np.float32)
                vec = np.concatenate([vec, pad])
        # Map into hyperdim space and apply the kernel's nonlinearity.
        proj = vec.astype(np.float32) @ basis
        if self.kernel.name == "ternary":
            return self.kernel.quantize(proj)
        # Real kernel: tanh-normalize.
        out = np.tanh(proj).astype(np.float32)
        n = float(np.linalg.norm(out))
        return (out / n).astype(np.float32) if n > 0 else out

    def seed_hypervector(self, text: str, description: str = "") -> np.ndarray:
        """Public alias: embed arbitrary text into the entity HV space.

        Used by the reader to embed a query into the same hyperdimensional
        space as the stored entity hypervectors, so the hierarchy router can
        score query-vs-summary-bundle similarity meaningfully.
        """
        return self._seed_hv_for(text, description)

    def _ensure_entity(self,
                       canonical: str,
                       type: str = "concept",
                       description: str = "",
                       aliases: Sequence[str] = ()) -> int:
        eid = self.substrate.lookup_by_surface(canonical)
        if eid is not None:
            # Add new aliases if needed and return.
            existing = self.substrate.get_entity(eid)
            if existing is not None:
                missing_aliases = [a for a in aliases if a not in set(existing.aliases)]
                if missing_aliases:
                    self.substrate.upsert_entity(
                        existing.canonical, type=existing.type or type,
                        description=existing.description, aliases=missing_aliases,
                    )
            return eid
        eid = self.substrate.upsert_entity(canonical, type=type, description=description, aliases=aliases)
        hv = self._seed_hv_for(canonical, description)
        self.substrate.set_hv(eid, self.kernel.pack(hv), self.kernel.name)
        return eid

    # ---- structured ingestion -----------------------------------------

    def write_triples(self, triples: Iterable[ExtractedTriple]) -> List[int]:
        """Write pre-extracted triples directly; return the new/updated edge ids."""
        edge_ids: List[int] = []
        for t in triples:
            h = self._ensure_entity(t.head, type=t.head_type)
            tail_id = self._ensure_entity(t.tail, type=t.tail_type)
            base_w = self._base_weight(t.relation)
            eid = self.substrate.upsert_edge(h, tail_id, t.relation, weight=base_w, source=t.source)
            edge_ids.append(eid)
            self._relation_counts[t.relation] = self._relation_counts.get(t.relation, 0) + 1
        return edge_ids

    def _base_weight(self, relation: str) -> float:
        """Base weight for a new edge with a duplicate-repetition penalty."""
        count = self._relation_counts.get(relation, 0)
        # The more we've seen this relation, the less novelty signal it carries.
        return max(0.2, 1.0 - self.repetition_lambda * np.log1p(count))

    # ---- text ingestion -----------------------------------------------

    def ingest_text(self, text: str, anchor: str) -> List[ExtractedTriple]:
        """Parse text, extract entities and triples, write to substrate."""
        self.substrate.upsert_document(anchor, text)
        doc = self.nlp(text)
        triples = self._extract(doc, source=anchor)
        self.write_triples(triples)
        return triples

    # ---- extraction logic ---------------------------------------------

    def _extract(self, doc: "Doc", source: str) -> List[ExtractedTriple]:
        """Extract entities and relation triples from a spaCy Doc."""
        # 1. Named entities and noun-phrase entities as candidate nodes.
        candidates: Dict[str, str] = {}  # canonical -> coarse type
        for ent in doc.ents:
            canon = ent.text.strip()
            if canon:
                candidates[canon] = self._coarse_type(ent.label_)
        for chunk in doc.noun_chunks:
            canon = chunk.lemma_.strip()
            if (canon and canon not in candidates
                    and len(canon) > 2
                    and not chunk.root.is_stop):
                candidates[canon] = "concept"

        # 2. SVO triples + appositive is_a triples.
        triples: List[ExtractedTriple] = []
        for sent in doc.sents:
            triples.extend(self._svo_triples(sent, candidates, source))
            triples.extend(self._appositive_triples(sent, candidates, source))

        # 3. Bare-entity fallback: if a sentence has multiple candidate entities
        # but no verb/appositive linked them, add a "co_occurs" edge between the
        # first two so the graph remains connected.  This is what makes
        # multi-hop bridges form from descriptive text.
        for sent in doc.sents:
            ents_in_sent = [c for c in candidates if c in sent.text]
            if len(ents_in_sent) >= 2:
                already = {(t.head, t.tail) for t in triples} | {(t.tail, t.head) for t in triples}
                a, b = ents_in_sent[0], ents_in_sent[1]
                if (a, b) not in already and (b, a) not in already:
                    triples.append(ExtractedTriple(
                        head=a, relation="co_occurs", tail=b,
                        head_type=candidates[a], tail_type=candidates[b],
                        source=source,
                    ))

        return triples

    def _svo_triples(self, sent: "Span",
                     candidates: Dict[str, str],
                     source: str) -> List[ExtractedTriple]:
        out: List[ExtractedTriple] = []
        for token in sent:
            if token.pos_ != "VERB":
                continue
            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj", "obj", "oprd")]
            # Walk through prepositional objects nested under a prep child.
            for prep in (c for c in token.children if c.dep_ == "prep"):
                for pobj in prep.children:
                    if pobj.dep_ == "pobj":
                        objects.append(pobj)
            for s in subjects:
                s_text = self._resolve(s, candidates)
                if not s_text:
                    continue
                for o in objects:
                    o_text = self._resolve(o, candidates)
                    if not o_text or s_text == o_text:
                        continue
                    out.append(ExtractedTriple(
                        head=s_text, relation=token.lemma_.lower(), tail=o_text,
                        head_type=candidates.get(s_text, "concept"),
                        tail_type=candidates.get(o_text, "concept"),
                        source=source,
                    ))
        return out

    def _appositive_triples(self, sent: "Span",
                            candidates: Dict[str, str],
                            source: str) -> List[ExtractedTriple]:
        out: List[ExtractedTriple] = []
        for token in sent:
            if token.dep_ == "appos":
                head_tok = token.head
                head_text = self._resolve(head_tok, candidates)
                tail_text = self._resolve(token, candidates)
                if head_text and tail_text and head_text != tail_text:
                    out.append(ExtractedTriple(
                        head=head_text, relation="is_a", tail=tail_text,
                        head_type=candidates.get(head_text, "concept"),
                        tail_type=candidates.get(tail_text, "concept"),
                        source=source,
                    ))
        return out

    @staticmethod
    def _resolve(token: "Token", candidates: Dict[str, str]) -> Optional[str]:
        """Map a token (or its subtree) to a candidate entity canonical form."""
        # First try the named-entity that contains this token.
        if hasattr(token, "ent_type_") and token.ent_type_:
            # Walk up the ent span: spaCy attaches ent_iob/type per token; the
            # easiest way to get the whole span is via the doc.ents membership.
            for ent in token.doc.ents:
                if ent.start <= token.i < ent.end:
                    surface = ent.text.strip()
                    if surface in candidates:
                        return surface
        # Fall back to the noun chunk that contains the token.
        for chunk in token.doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                lemma = chunk.lemma_.strip()
                if lemma in candidates:
                    return lemma
        # Last resort: the token's lemma itself if it's a candidate.
        if token.lemma_ in candidates:
            return token.lemma_
        return None

    @staticmethod
    def _coarse_type(spacy_label: str) -> str:
        if spacy_label in {"PERSON"}:
            return "Person"
        if spacy_label in {"ORG"}:
            return "Organization"
        if spacy_label in {"GPE", "LOC"}:
            return "Location"
        if spacy_label in {"DATE", "TIME"}:
            return "Temporal"
        if spacy_label in {"EVENT"}:
            return "Event"
        if spacy_label in {"WORK_OF_ART", "PRODUCT"}:
            return "Work"
        return "concept"

    # ---- reward-shaped updates ----------------------------------------

    def reinforce_edges(self, edge_ids: Sequence[int], reward: float) -> None:
        """Apply reward-shaped weight deltas to a set of edges.

        SAGE-style: positive reward boosts the edges that were used in a
        successful retrieval; negative reward demotes them.  The magnitude is
        scaled by `reinforce_eta` for positive rewards and `demote_eta` for
        negative ones (asymmetric: we forget more slowly than we learn).
        """
        if not edge_ids:
            return
        eta = self.reinforce_eta if reward >= 0 else self.demote_eta
        delta = float(eta * reward)
        for eid in edge_ids:
            self.substrate.update_edge_weight(int(eid), delta)
            # Reset the decay clock for any positively-reinforced edge so that
            # used memories resist the forgetting curve.
            if reward >= 0:
                self.substrate.touch_edge(int(eid))


__all__ = ["MemoryWriter", "ExtractedTriple"]
