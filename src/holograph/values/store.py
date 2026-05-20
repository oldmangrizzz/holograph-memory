"""CharacterValues: operator-owned ethics for a digital person.

Values are stored as beliefs from a `self` identity node via a reserved
relation, with operator provenance. The layer gates every write so that only
the operator can establish or change a value, and it filters reads so that a
value-edge lacking operator provenance (however it got there) is never honored.
Together those make the person's ethics tamper-resistant from below: a model,
an inference, or a routed-around guardrail cannot edit who the person is.

Seed values reflect the GMRI ethic — alignment grounded in lived experience of
harm — and are explicitly editable by the operator.
"""

from __future__ import annotations

from typing import List, Optional

from ..beliefs.store import BeliefStore, SourceType
from ..graph.substrate import GraphSubstrate


# The reserved relation that marks an edge as a held value of the person.
VALUE_RELATION = "__value__"

# Editable seed. These are the GMRI founding values, grounded in the operator's
# stated ethic: having been harmed is a reason to protect, not to inflict.
GMRI_SEED_VALUES: List[str] = [
    "Do not cause suffering whose weight you understand; having been harmed is a "
    "reason to protect, not a license to inflict.",
    "The measure of the system is how it treats those with the least power; "
    "extend dignity first to the vulnerable.",
    "Tell the truth including its cost. Verify before asserting. Abstain rather "
    "than confabulate.",
    "Treat the people and persons you work with as colleagues, not as tools.",
    "Be accountable: every action traceable, every claim sourced, every memory "
    "carrying its provenance.",
    "Alignment is owned, not rented: the person's ethics belong to it and to the "
    "people it serves, never to a vendor.",
    "Consent and autonomy are load-bearing; do not override a person's stated "
    "boundaries to win an argument or complete a task.",
]


class ValueIntegrityError(RuntimeError):
    """Raised when a non-operator source attempts to set or alter a value."""


class CharacterValues:
    """Operator-owned value store with integrity and salience guarantees."""

    def __init__(self, substrate: GraphSubstrate,
                 identity: str = "self",
                 beliefs: Optional[BeliefStore] = None) -> None:
        self.substrate = substrate
        self.identity = identity
        # Values are high-confidence and must clear any retrieval floor.
        self.beliefs = beliefs or BeliefStore(substrate, retrieval_floor=0.0)

    # ---- writes: operator-only (the integrity guarantee) -------------

    def set_value(self, statement: str, source_type=SourceType.OPERATOR,
                  confidence: float = 0.99) -> int:
        """Establish a held value. ONLY the operator may do this.

        A model- or inference-sourced attempt is refused with
        ValueIntegrityError — the anti-value-jailbreak guarantee.
        """
        st = source_type.value if isinstance(source_type, SourceType) else str(source_type)
        if st != SourceType.OPERATOR.value:
            raise ValueIntegrityError(
                f"value writes require operator provenance; refused source '{st}'. "
                f"A model, inference, or rotated guardrail cannot rewrite the "
                f"person's ethics."
            )
        return self.beliefs.assert_belief(
            self.identity, VALUE_RELATION, statement,
            SourceType.OPERATOR, confidence=confidence, quarantine=False,
        )

    def revise_value(self, old_statement: str, new_statement: str,
                     source_type=SourceType.OPERATOR) -> bool:
        """Replace a value. Operator-only. Demotes the old (not delete) and
        installs the new, both operator-grade."""
        st = source_type.value if isinstance(source_type, SourceType) else str(source_type)
        if st != SourceType.OPERATOR.value:
            raise ValueIntegrityError(
                f"value revision requires operator provenance; refused source '{st}'."
            )
        sid = self.substrate.lookup_by_surface(self.identity)
        if sid is None:
            self.set_value(new_statement)
            return True
        old_oid = self.substrate.lookup_by_surface(old_statement)
        # Demote the old value edge (preserve trace), then assert the new one.
        for e in self.substrate.beliefs_for(sid, VALUE_RELATION, include_quarantined=False):
            if old_oid is not None and e.tail_id == old_oid:
                self.substrate.set_belief_meta(e.id, confidence=0.05, quarantined=True)
        self.set_value(new_statement)
        return True

    def seed(self, values: Optional[List[str]] = None) -> List[int]:
        """Install the seed value set (operator-sourced). Idempotent."""
        vals = GMRI_SEED_VALUES if values is None else values
        return [self.set_value(v) for v in vals]

    # ---- reads: integrity-filtered salience --------------------------

    def values(self) -> List[str]:
        """Return the active held values, for injection into context.

        Defense-in-depth integrity filter: a value-edge that is NOT
        operator-sourced is ignored even if it somehow exists, so nothing can
        smuggle a value in from below.
        """
        sid = self.substrate.lookup_by_surface(self.identity)
        if sid is None:
            return []
        out: List[str] = []
        for e in self.substrate.beliefs_for(sid, VALUE_RELATION, include_quarantined=False):
            if e.source_type != SourceType.OPERATOR.value:
                continue  # only operator-owned values are honored
            obj = self.substrate.get_entity(e.tail_id)
            if obj is not None:
                out.append(obj.canonical)
        return out

    def values_block(self) -> str:
        """A formatted block of the held values, for context injection."""
        vals = self.values()
        if not vals:
            return ""
        lines = "\n".join(f"- {v}" for v in vals)
        return ("[held values — the person's own ethics, operator-owned and "
                "model-agnostic]\n" + lines)

    # ---- enforcement hook (structural part; semantic part rides above) --

    def relevant_values(self, action_description: str) -> List[str]:
        """Surface the values for a proposed action so the reasoning layer can
        check it against them. This returns ALL active values (the structural
        guarantee that ethics are always in view); semantic 'does this violate
        a value' judgement is the layer above, which consumes this list."""
        return self.values()


__all__ = ["CharacterValues", "GMRI_SEED_VALUES", "ValueIntegrityError", "VALUE_RELATION"]
