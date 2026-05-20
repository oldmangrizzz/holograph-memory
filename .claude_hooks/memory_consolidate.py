#!/usr/bin/env python3
"""Stop hook: consolidate the session into long-term memory.

At session end, reads the transcript and writes its salient facts into the
persistent HoloGraph store — the "sleep" pass that moves the day's experience
into durable memory so the next session can wake into it.

Claude Code protocol:
    stdin = JSON event carrying {"transcript_path": "...", "session_id": "..."}.
            The transcript is JSONL; each line is a message record.
    Exit 0 always. Fails OPEN (write nothing) on any error.

Config via env:
    HOLOGRAPH_DB    memory DB path
    HOLOGRAPH_SRC   path to holograph src/
    HOLOGRAPH_CONSOLIDATE_MAX_CHARS  cap on transcript text ingested (default 20000)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    src = os.environ.get("HOLOGRAPH_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


def _extract_transcript_text(path: str, max_chars: int) -> str:
    """Pull human-readable text from a Claude Code JSONL transcript.

    Best-effort and defensive: transcript schemas vary, so we collect any
    string 'content' / 'text' fields from user+assistant messages.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return ""
    parts = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("message", rec)
                role = msg.get("role") or rec.get("type") or ""
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            parts.append(block["text"])
    except OSError:
        return ""
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    transcript_path = event.get("transcript_path", "")
    session_id = event.get("session_id", "")
    if not transcript_path:
        return 0

    try:
        _bootstrap_path()
        from holograph.continuity.store import MemoryStore, consolidate
        max_chars = int(os.environ.get("HOLOGRAPH_CONSOLIDATE_MAX_CHARS", "20000"))
        text = _extract_transcript_text(transcript_path, max_chars)
        if not text.strip():
            return 0
        import time
        anchor = f"session-{session_id or int(time.time())}"
        store = MemoryStore()
        n = consolidate(store, text, anchor=anchor)
        store.close()
        print(f"memory_consolidate: wrote {n} triples from {anchor}.", file=sys.stderr)
    except Exception as exc:  # fail open — never block session teardown
        print(f"memory_consolidate: failed ({type(exc).__name__}); skipping.",
              file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
