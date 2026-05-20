"""Persistent memory store + recall/consolidate primitives.

A MemoryStore is a HoloGraph runtime bound to a fixed on-disk SQLite path
(default ~/.holograph/memory.db) with the hierarchy + routing enabled. It is
the thing a session "jacks into."

recall(store, cue)        -> a formatted memory-context block (string) to inject
consolidate(store, text)  -> writes salient facts from text into the store

Design choices honoring the project's constraints:
    * Structural-first, no LLM in the loop — extraction is spaCy, routing is
      the hierarchical HDC reader. The LLM never drives the memory.
    * Recall is read-only and bounded: it routes to a small candidate set and
      returns at most `max_items` memory lines, so the injected context stays
      small no matter how large the store grows (the context-window kill).
    * Consolidate is additive and idempotent on identical text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..runtime import HoloGraph


DEFAULT_DB = os.environ.get(
    "HOLOGRAPH_DB",
    str(Path.home() / ".holograph" / "memory.db"),
)


@dataclass
class MemoryStore:
    """A persistent HoloGraph bound to a fixed DB path.

    Encrypted at rest by default (Fernet file-at-rest; see crypto.py for the
    exact security properties and residual risk). The logical `db_path` names
    the store; the on-disk artifact is `<db_path>.enc`, and a plaintext working
    copy exists only while the store is open, under a 0700 work dir, shredded
    on close. Set encrypted=False only for ephemeral/non-PII stores.
    """

    db_path: str = DEFAULT_DB
    kernel_kind: str = "real"
    dim: Optional[int] = None
    top_k: int = 8
    router_beam: int = 6
    encrypted: bool = True
    key_path: Optional[str] = None
    _hg: Optional[HoloGraph] = field(default=None, init=False, repr=False)
    _work_path: Optional[Path] = field(default=None, init=False, repr=False)

    def _enc_path(self) -> Path:
        return Path(str(Path(self.db_path).expanduser()) + ".enc")

    def _key_path(self) -> Path:
        if self.key_path:
            return Path(self.key_path).expanduser()
        return Path(self.db_path).expanduser().parent / "key"

    def open(self) -> HoloGraph:
        if self._hg is not None:
            return self._hg
        logical = Path(self.db_path).expanduser()
        logical.parent.mkdir(parents=True, exist_ok=True)

        if not self.encrypted:
            open_path = str(logical)
        else:
            from .crypto import get_or_create_key, decrypt_file, work_dir, secure_delete
            key = get_or_create_key(self._key_path())
            wd = work_dir(logical.parent)
            # Clean any stale work file from a prior unclean shutdown.
            self._work_path = wd / (logical.name + ".plain")
            secure_delete(self._work_path)
            enc = self._enc_path()
            if enc.is_file():
                decrypt_file(enc, self._work_path, key)
            open_path = str(self._work_path)

        self._hg = HoloGraph(
            kernel_kind=self.kernel_kind,
            dim=self.dim,
            db_path=open_path,
            top_k=self.top_k,
            use_hierarchy=True,
            router_beam=self.router_beam,
        )
        return self._hg

    def close(self) -> None:
        if self._hg is None:
            return
        self._hg.close()
        self._hg = None
        if self.encrypted and self._work_path is not None:
            from .crypto import get_or_create_key, encrypt_file, secure_delete
            try:
                if Path(self._work_path).is_file():
                    key = get_or_create_key(self._key_path())
                    encrypt_file(self._work_path, self._enc_path(), key)
            finally:
                secure_delete(self._work_path)
                self._work_path = None

    # convenience pass-throughs
    def rebuild_hierarchy(self, **kw) -> None:
        self.open().build_hierarchy(**kw)

    def decay(self, half_life_seconds: float, floor: float = 0.0) -> int:
        return self.open().decay(half_life_seconds, floor=floor)

    def n_memories(self) -> int:
        return self.open().substrate.n_entities()


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def recall(store: MemoryStore, cue: str, max_items: int = 8,
           rebuild_if_stale: bool = True) -> str:
    """Return a compact memory-context block relevant to `cue`.

    The block is bounded to `max_items` lines regardless of store size — this
    is the mechanism that injects continuity without blowing the context
    window. Returns "" if the store is empty.
    """
    hg = store.open()
    if hg.substrate.n_entities() == 0:
        return ""
    # Ensure a hierarchy exists so routing is cheap; rebuild only if absent.
    if rebuild_if_stale and hg.substrate.max_layer() == 0:
        hg.build_hierarchy(max_layer=4, node_cap=8)

    out = hg.read(cue)
    if not out.activated_ids:
        return ""

    # Rank activated entities, take the strongest, and surface the documents /
    # relations attached to them as the recalled memory lines.
    ranked = sorted(out.activated_ids,
                    key=lambda nid: out.final_activation.get(nid, 0.0),
                    reverse=True)[:max_items]

    lines: List[str] = []
    seen_docs = set()
    # Prefer the source documents (full episodic context).
    for anchor, text in out.supporting_documents:
        if anchor in seen_docs:
            continue
        seen_docs.add(anchor)
        lines.append(f"- ({anchor}) {text.strip()}")
        if len(lines) >= max_items:
            break
    # If we still have room, add the strongest entity relations.
    if len(lines) < max_items:
        for u, r, v in out.activated_subgraph_edges:
            hu = hg.substrate.get_entity(u)
            hv = hg.substrate.get_entity(v)
            if hu is None or hv is None:
                continue
            rel = r if not r.startswith("__") else "relates to"
            lines.append(f"- {hu.canonical} {rel} {hv.canonical}")
            if len(lines) >= max_items:
                break

    if not lines:
        return ""
    header = ("[recalled memory — relevant prior context, injected by the "
              "continuity layer]")
    return header + "\n" + "\n".join(lines[:max_items])


# ---------------------------------------------------------------------------
# Consolidate
# ---------------------------------------------------------------------------


def consolidate(store: MemoryStore, text: str, anchor: Optional[str] = None,
                rebuild_hierarchy: bool = True) -> int:
    """Write salient facts from `text` into the store. Returns triples written.

    `anchor` is the episodic source id (e.g. a session id + timestamp). If the
    same anchor is consolidated twice with the same text, the writer's upserts
    keep it idempotent.
    """
    hg = store.open()
    if anchor is None:
        import time
        anchor = f"session-{int(time.time())}"
    triples = hg.ingest_text(text, anchor=anchor)
    if rebuild_hierarchy and hg.substrate.n_entities() > 0:
        # Rebuild so the new memories are routable. For large stores this is
        # the "sleep"/consolidation cost and can be deferred to an idle pass.
        hg.build_hierarchy(max_layer=4, node_cap=8)
    return len(triples)


__all__ = ["MemoryStore", "recall", "consolidate", "DEFAULT_DB"]
