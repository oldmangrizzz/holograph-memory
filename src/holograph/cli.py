"""Minimal Typer-based CLI for HoloGraph.

Commands:
    holograph ingest TEXT [--db PATH] [--anchor NAME]
    holograph read QUERY [--db PATH]
    holograph status [--db PATH]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .runtime import HoloGraph

app = typer.Typer(help="HoloGraph CLI")
console = Console()


def _runtime(db: str, kernel: str) -> HoloGraph:
    return HoloGraph(kernel_kind=kernel, db_path=db)


@app.command()
def ingest(text: str = typer.Argument(..., help="Text to ingest into the memory graph"),
           anchor: str = typer.Option("doc", help="Document anchor id"),
           db: str = typer.Option("holograph.db", help="SQLite path"),
           kernel: str = typer.Option("real", help="HDC kernel: real or ternary")) -> None:
    """Ingest a passage of text into the graph memory."""
    hg = _runtime(db, kernel)
    triples = hg.ingest_text(text, anchor=anchor)
    table = Table(title=f"Extracted {len(triples)} triples")
    table.add_column("head", style="cyan")
    table.add_column("relation", style="magenta")
    table.add_column("tail", style="green")
    for t in triples:
        table.add_row(t.head, t.relation, t.tail)
    console.print(table)
    console.print(f"Summary: {hg.summary()}")
    hg.close()


@app.command()
def read(query: str = typer.Argument(..., help="Natural-language query"),
         db: str = typer.Option("holograph.db", help="SQLite path"),
         kernel: str = typer.Option("real", help="HDC kernel: real or ternary"),
         top_k: int = typer.Option(8, help="Top-K activated entities")) -> None:
    """Run a query against the graph memory."""
    hg = HoloGraph(kernel_kind=kernel, db_path=db, top_k=top_k)
    out = hg.read(query)
    table = Table(title=f"Top {len(out.activated_ids)} activations")
    table.add_column("id", justify="right")
    table.add_column("entity", style="cyan")
    table.add_column("init", justify="right")
    table.add_column("final", justify="right")
    for nid in out.activated_ids:
        ent = hg.substrate.get_entity(nid)
        if ent is None:
            continue
        table.add_row(str(nid), ent.canonical,
                      f"{out.initial_activation.get(nid, 0.0):+.3f}",
                      f"{out.final_activation.get(nid, 0.0):+.3f}")
    console.print(table)
    if out.activated_subgraph_edges:
        etable = Table(title="Activated subgraph edges")
        etable.add_column("head"); etable.add_column("rel"); etable.add_column("tail")
        for u, r, v in out.activated_subgraph_edges:
            h = hg.substrate.get_entity(u)
            t = hg.substrate.get_entity(v)
            etable.add_row(h.canonical if h else str(u), r, t.canonical if t else str(v))
        console.print(etable)
    if out.attribution and out.attribution.per_path_top:
        ptable = Table(title="Top paths in composed query HV")
        ptable.add_column("similarity", justify="right")
        ptable.add_column("path")
        for (u, r, v), s in out.attribution.per_path_top[:5]:
            h = hg.substrate.get_entity(u)
            t = hg.substrate.get_entity(v)
            rel_safe = f"\\[{r}]"
            ptable.add_row(f"{s:+.3f}",
                           f"{h.canonical if h else u} --{rel_safe}--> {t.canonical if t else v}")
        console.print(ptable)
    hg.close()


@app.command()
def status(db: str = typer.Option("holograph.db", help="SQLite path"),
           kernel: str = typer.Option("real", help="HDC kernel: real or ternary")) -> None:
    """Show graph counts and kernel info."""
    hg = _runtime(db, kernel)
    s = hg.summary()
    console.print(f"[bold]HoloGraph status[/bold] @ {db}")
    for k, v in s.items():
        console.print(f"  {k}: {v}")
    hg.close()


if __name__ == "__main__":
    app()
