"""Belief layer: provenance, confidence, quarantine, and stable revision.

Solves confabulation the way human memory manages it — not by elimination, but
by making fabrications catchable and correctable — and does the revision
without discombobulation: locally, reversibly (demote, never delete), and in
isolation (quarantine the unconfirmed).

Public surface:
    BeliefStore     the engine: assert / recall / revise / corroborate / consolidate
    SourceType      provenance classes and their precedence
    RevisionRecord  audit record returned by revise()
"""

from .store import BeliefStore, SourceType, RevisionRecord

__all__ = ["BeliefStore", "SourceType", "RevisionRecord"]
