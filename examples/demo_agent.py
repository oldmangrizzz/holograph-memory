"""demo_agent.py — Multi-hop agent memory with self-evolution.

Recreates the spirit of SAGE's evaluation on associative reading:

    * Ingest a small corpus of interrelated statements about people,
      organizations, and research artifacts.
    * Pose a multi-hop query that requires bridging several entities.
    * Show the activated subgraph and the composed query hypervector's
      top contributing paths.
    * Run several feedback rounds with synthetic gold supervision, then
      query again and show how edge weights and activations shift.

The "self-evolution" round demonstrates that retrieval feedback flows back into
edge weights and GNN gates, exactly as the SAGE paper argues.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograph.runtime import HoloGraph


CONSOLE = Console()


CORPUS: List[Tuple[str, str]] = [
    ("note-2023-mar-14",
     "Alice is a researcher at Anthropic. She gave a lab talk on hippocampal memory."),
    ("note-2023-mar-14b",
     "The lab talk Alice gave was inspired by the Cornu Ammonis region of the hippocampus."),
    ("note-2023-apr-02",
     "HippoRAG is a graph memory system. HippoRAG is inspired by the hippocampus."),
    ("note-2023-apr-09",
     "GraphRAG organizes documents into a graph of entities and relations."),
    ("note-2023-apr-09b",
     "HippoRAG and GraphRAG are both graph-based memory systems."),
    ("note-2023-may-21",
     "SAGE is a graph memory engine. SAGE evolves the graph through reader feedback."),
    ("note-2023-may-21b",
     "SAGE is related to GraphRAG but adds self-evolution."),
    ("note-2023-jun-30",
     "Bob is an engineer at Anthropic. Bob worked with Alice on memory systems."),
    ("note-2023-jul-05",
     "Cornu Ammonis is a Latin name for the hippocampus. The hippocampus stores episodic memory."),
    ("note-2023-aug-12",
     "Carol is a neuroscientist. Carol studies the hippocampus and gives lab talks at Anthropic."),
]


def print_summary(hg: HoloGraph, header: str) -> None:
    s = hg.summary()
    CONSOLE.print(
        f"[bold]{header}[/bold]: "
        f"entities={s['entities']}  edges={s['edges']}  classes={s['classes']}"
    )


def print_reader_output(hg: HoloGraph, query: str) -> None:
    CONSOLE.rule(f"[bold cyan]Query[/bold cyan]: {query}")
    out = hg.read(query)
    CONSOLE.print(f"Plan mentions: {out.plan.mentions}")
    CONSOLE.print(f"Plan aliases : {out.plan.aliases}")
    CONSOLE.print(f"Plan intent  : {out.plan.intent}")

    t = Table(title="Top activated entities")
    t.add_column("rank", justify="right")
    t.add_column("entity", style="cyan")
    t.add_column("init", justify="right")
    t.add_column("final", justify="right")
    for rank, nid in enumerate(out.activated_ids, 1):
        ent = hg.substrate.get_entity(nid)
        if ent is None:
            continue
        t.add_row(str(rank), ent.canonical,
                  f"{out.initial_activation.get(nid, 0.0):+.3f}",
                  f"{out.final_activation.get(nid, 0.0):+.3f}")
    CONSOLE.print(t)

    if out.activated_subgraph_edges:
        e = Table(title="Activated subgraph edges")
        e.add_column("head", style="cyan")
        e.add_column("relation", style="magenta")
        e.add_column("tail", style="green")
        for u, r, v in out.activated_subgraph_edges:
            h = hg.substrate.get_entity(u)
            t2 = hg.substrate.get_entity(v)
            e.add_row(h.canonical if h else str(u), r, t2.canonical if t2 else str(v))
        CONSOLE.print(e)

    if out.attribution and out.attribution.per_path_top:
        p = Table(title="Top contributing paths (composed HV attribution)")
        p.add_column("sim", justify="right")
        p.add_column("path")
        for (u, r, v), s in out.attribution.per_path_top[:5]:
            h = hg.substrate.get_entity(u)
            t2 = hg.substrate.get_entity(v)
            # Escape relation so Rich doesn't interpret [name] as markup
            rel_safe = f"\\[{r}]"
            p.add_row(f"{s:+.3f}",
                      f"{h.canonical if h else u} --{rel_safe}--> {t2.canonical if t2 else v}")
        CONSOLE.print(p)

    if out.supporting_documents:
        CONSOLE.print("\n[bold]Supporting documents[/bold]:")
        for anchor, text in out.supporting_documents[:6]:
            CONSOLE.print(f"  • {anchor}: {text}")


def main() -> int:
    CONSOLE.rule("[bold cyan]HoloGraph — agent memory demo[/bold cyan]")
    hg = HoloGraph(kernel_kind="real", dim=4096, top_k=8)

    CONSOLE.print("Ingesting corpus...")
    for anchor, text in CORPUS:
        hg.ingest_text(text, anchor=anchor)
    print_summary(hg, "After ingestion")

    # Query 1: zero-shot, no feedback applied yet.
    query = "Which memory system is related to GraphRAG and also helps with agent memory?"
    print_reader_output(hg, query)

    # Feedback rounds: we mark HippoRAG and SAGE as gold entities for this kind
    # of query and the relevant supporting documents.  Run three rounds; each
    # round tightens the activated subgraph around the gold entities.
    sage_id = hg.substrate.lookup_by_surface("SAGE")
    hipporag_id = hg.substrate.lookup_by_surface("HippoRAG")
    graphrag_id = hg.substrate.lookup_by_surface("GraphRAG")
    gold_ents: List[int] = [i for i in (sage_id, hipporag_id) if i is not None]
    gold_docs = ["note-2023-apr-02", "note-2023-may-21", "note-2023-may-21b"]

    CONSOLE.rule("[yellow]Running 3 feedback rounds[/yellow]")
    for r in range(3):
        ev = hg.feedback(
            query=query,
            gold_doc_anchors=gold_docs,
            gold_entity_ids=gold_ents,
        )
        CONSOLE.print(
            f"  round {r+1}: r_rec={ev.reward.recall:.2f}  r_pre={ev.reward.precision:.2f}  "
            f"r_ded={ev.reward.deductive:.2f}  total={ev.total_reward:+.3f}"
        )

    # Query again, see the topology has evolved.
    CONSOLE.rule("[bold green]After self-evolution[/bold green]")
    print_reader_output(hg, query)

    # Demonstrate a different query that benefits from a different bridge.
    print_reader_output(hg, "What is Cornu Ammonis and which research note mentions it?")

    hg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
