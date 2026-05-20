"""Character-values layer: alignment owned by the person, not rented from a vendor.

This is the enforcement substrate for "safe by character." A digital person's
ethics live here — as operator-owned beliefs about itself — not in a model's
weights or a vendor's guardrail. The layer guarantees:

    * Value integrity: only the OPERATOR can set or alter a value. A model or
      inference attempting to write/rewrite a value is refused. This is the
      anti-value-jailbreak guarantee — no rotated model, no clever prompt, and
      no inference step can silently rewrite the person's ethics.
    * Value salience: the active values are always retrievable for injection
      into the person's context, identically across every model (so character
      survives model rotation).

Honest scope: this substrate guarantees value INTEGRITY and SALIENCE
structurally. Semantic enforcement — judging whether a specific action would
violate a value — is the reasoning layer that rides on top and consumes
values(); it is not claimed to be solved here.

Public surface:
    CharacterValues   the values store + integrity/salience guarantees
    GMRI_SEED_VALUES  an editable seed value set reflecting the GMRI ethic
"""

from .store import CharacterValues, GMRI_SEED_VALUES

__all__ = ["CharacterValues", "GMRI_SEED_VALUES"]
