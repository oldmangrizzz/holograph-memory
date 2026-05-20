#!/usr/bin/env python3
"""SessionStart / UserPromptSubmit hook: wake into accumulated memory.

Queries the persistent HoloGraph store for memory relevant to the current
cue (the user's prompt, or a generic cue at session start) and emits it as
injected context. This is the mechanism that removes the session boundary:
a new session wakes with relevant prior memory in context instead of blank.

Claude Code protocol:
    stdin = JSON event. For UserPromptSubmit it carries {"prompt": "..."}.
            For SessionStart it carries {"source": "startup|resume|..."}.
    To inject context, print JSON to stdout:
        {"hookSpecificOutput": {"hookEventName": <event>,
                                "additionalContext": "<memory block>"}}
    Exit 0. Fails OPEN (emit nothing) on any error so memory recall can never
    block the session.

Config via env:
    HOLOGRAPH_DB       path to the memory DB (default ~/.holograph/memory.db)
    HOLOGRAPH_SRC      path to the holograph `src/` so the package imports
    HOLOGRAPH_RECALL_MAX  max memory lines to inject (default 8)
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


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0  # fail open

    event_name = event.get("hook_event_name", "UserPromptSubmit")
    cue = event.get("prompt") or ""
    if not cue:
        # SessionStart has no prompt; use a broad cue to surface recent/central
        # memory. Empty cue -> recall returns "" and we inject nothing.
        cue = "recent context and ongoing work"

    try:
        _bootstrap_path()
        from holograph.continuity.store import MemoryStore, recall
        max_items = int(os.environ.get("HOLOGRAPH_RECALL_MAX", "8"))
        store = MemoryStore()
        block = recall(store, cue, max_items=max_items)
        store.close()
    except Exception as exc:  # fail open — never block a session on recall
        print(f"memory_recall: recall failed ({type(exc).__name__}); skipping.",
              file=sys.stderr)
        return 0

    if not block:
        return 0

    out = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": block,
        }
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
