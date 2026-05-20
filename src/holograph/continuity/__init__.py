"""Continuity subsystem — turns HoloGraph into persistent cross-session memory.

This is the layer that removes the "session" boundary: a recall step that wakes
a new session into accumulated memory, and a consolidate step that writes the
session's salient facts back. Functional continuity (the memory persists and is
re-injected), not a continuous process.

Public surface:
    MemoryStore     a persistent HoloGraph bound to a fixed DB path
    recall          query the store, return a context block for injection
    consolidate     ingest text (e.g. a transcript) into the store
"""

from .store import MemoryStore, recall, consolidate

__all__ = ["MemoryStore", "recall", "consolidate"]
