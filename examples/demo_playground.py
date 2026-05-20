"""demo_playground.py — Interactive HoloGraph REPL.

Run:
    python examples/demo_playground.py

Commands at the prompt:
    /help                            show this help
    /ingest <anchor> <text...>       ingest a text passage with an anchor id
    /load <path>                     load a UTF-8 text file (anchor = filename)
    /status                          show entity/edge counts
    /entities                        list all entities
    /edges                           list all edges (with weights)
    /read <query>                    run a query and print results
    /feedback <query> | <doc1,doc2> | <ent1,ent2>
                                     run a feedback step with gold supervision
    /kernel real|ternary             swap kernel (resets the graph)
    /save <path>                     persist the SQLite db
    /quit                            exit

If no command prefix is given, the input is sent to /read.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograph.runtime import HoloGraph


CONSOLE = Console()


HELP_TEXT = """[bold]Commands[/bold]
  /help
  /ingest <anchor> <text...>
  /load <path>
  /status
  /entities
  /edges
  /read <query>            (anything without a leading /  is treated as read)
  /feedback <query> | <doc1,doc2> | <ent1,ent2>
  /kernel real|ternary
  /save <path>
  /quit
"""


def render_read(hg: HoloGraph, query: str) -> None:
    out = hg.read(query)
    t = Table(title=f"Top activations for: {query[:60]}")
    t.add_column("rank", justify="right")
    t.add_column("id", justify="right")
    t.add_column("entity", style="cyan")
    t.add_column("init", justify="right")
    t.add_column("final", justify="right")
    for rank, nid in enumerate(out.activated_ids, 1):
        ent = hg.substrate.get_entity(nid)
        if ent is None:
            continue
        t.add_row(str(rank), str(nid), ent.canonical,
                  f"{out.initial_activation.get(nid, 0.0):+.3f}",
                  f"{out.final_activation.get(nid, 0.0):+.3f}")
    CONSOLE.print(t)
    if out.activated_subgraph_edges:
        e = Table(title="Activated subgraph edges")
        e.add_column("head", style="cyan"); e.add_column("rel", style="magenta"); e.add_column("tail", style="green")
        for u, r, v in out.activated_subgraph_edges:
            h = hg.substrate.get_entity(u)
            t2 = hg.substrate.get_entity(v)
            e.add_row(h.canonical if h else str(u), r, t2.canonical if t2 else str(v))
        CONSOLE.print(e)


def parse_feedback_args(rest: str) -> Optional[tuple]:
    parts = [p.strip() for p in rest.split("|")]
    if len(parts) != 3:
        return None
    q = parts[0]
    docs = [d.strip() for d in parts[1].split(",") if d.strip()]
    ents = []
    for token in parts[2].split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ents.append(int(token))
        except ValueError:
            return None
    return q, docs, ents


def main() -> int:
    CONSOLE.rule("[bold cyan]HoloGraph playground[/bold cyan]")
    CONSOLE.print(HELP_TEXT)
    hg = HoloGraph(kernel_kind="real", dim=4096, top_k=8)
    CONSOLE.print(f"[dim]kernel={hg.kernel.name}  D={hg.kernel.dim}[/dim]")

    while True:
        try:
            line = input("holograph> ").strip()
        except (EOFError, KeyboardInterrupt):
            CONSOLE.print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            # treat as read
            render_read(hg, line)
            continue

        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            CONSOLE.print("[red]parse error[/red]")
            continue
        cmd, *args = tokens
        rest = line[len(cmd):].lstrip()

        if cmd in ("/quit", "/exit"):
            break

        elif cmd == "/help":
            CONSOLE.print(HELP_TEXT)

        elif cmd == "/status":
            CONSOLE.print(hg.summary())

        elif cmd == "/entities":
            t = Table(title=f"Entities ({hg.substrate.n_entities()})")
            t.add_column("id", justify="right")
            t.add_column("canonical", style="cyan")
            t.add_column("type")
            t.add_column("aliases", style="dim")
            for e in hg.substrate.all_entities():
                t.add_row(str(e.id), e.canonical, e.type, ", ".join(e.aliases))
            CONSOLE.print(t)

        elif cmd == "/edges":
            t = Table(title=f"Edges ({hg.substrate.n_edges()})")
            t.add_column("id", justify="right")
            t.add_column("head", style="cyan")
            t.add_column("rel", style="magenta")
            t.add_column("tail", style="green")
            t.add_column("w", justify="right")
            t.add_column("source", style="dim")
            for edge in hg.substrate.all_edges():
                h = hg.substrate.get_entity(edge.head_id)
                v = hg.substrate.get_entity(edge.tail_id)
                t.add_row(str(edge.id),
                           h.canonical if h else str(edge.head_id),
                           edge.relation,
                           v.canonical if v else str(edge.tail_id),
                           f"{edge.weight:.2f}",
                           edge.source)
            CONSOLE.print(t)

        elif cmd == "/ingest":
            if len(args) < 2:
                CONSOLE.print("[yellow]usage: /ingest <anchor> <text...>[/yellow]"); continue
            anchor = args[0]
            text = rest[len(anchor):].strip()
            text = text.lstrip("\"'").rstrip("\"'")
            tr = hg.ingest_text(text, anchor=anchor)
            CONSOLE.print(f"[green]ingested[/green] {len(tr)} triples from {anchor}")

        elif cmd == "/load":
            if not args:
                CONSOLE.print("[yellow]usage: /load <path>[/yellow]"); continue
            p = Path(args[0]).expanduser()
            if not p.exists():
                CONSOLE.print(f"[red]not found:[/red] {p}"); continue
            text = p.read_text(encoding="utf-8")
            tr = hg.ingest_text(text, anchor=p.name)
            CONSOLE.print(f"[green]ingested[/green] {len(tr)} triples from {p.name}")

        elif cmd == "/read":
            if not rest.strip():
                CONSOLE.print("[yellow]usage: /read <query>[/yellow]"); continue
            render_read(hg, rest.strip())

        elif cmd == "/feedback":
            parsed = parse_feedback_args(rest)
            if parsed is None:
                CONSOLE.print("[yellow]usage: /feedback <query> | <doc1,doc2> | <ent1,ent2>[/yellow]"); continue
            q, docs, ents = parsed
            ev = hg.feedback(query=q, gold_doc_anchors=docs, gold_entity_ids=ents)
            CONSOLE.print(
                f"reward total={ev.total_reward:+.3f}  rec={ev.reward.recall:.2f}"
                f"  pre={ev.reward.precision:.2f}  ded={ev.reward.deductive:.2f}"
            )

        elif cmd == "/kernel":
            if not args or args[0] not in ("real", "ternary"):
                CONSOLE.print("[yellow]usage: /kernel real|ternary[/yellow]"); continue
            hg.close()
            hg = HoloGraph(kernel_kind=args[0], top_k=8)
            CONSOLE.print(f"[green]switched[/green] kernel={hg.kernel.name} D={hg.kernel.dim}")

        elif cmd == "/save":
            if not args:
                CONSOLE.print("[yellow]usage: /save <path>[/yellow]"); continue
            out_path = Path(args[0]).expanduser()
            # SQLite VACUUM INTO is the cleanest portable copy.
            try:
                hg.substrate.conn.execute(f"VACUUM INTO '{str(out_path)}'")
                CONSOLE.print(f"[green]saved[/green] -> {out_path}")
            except Exception as exc:  # pragma: no cover - sqlite portability
                CONSOLE.print(f"[red]save failed:[/red] {exc}")

        else:
            CONSOLE.print(f"[yellow]unknown command:[/yellow] {cmd}")

    hg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
