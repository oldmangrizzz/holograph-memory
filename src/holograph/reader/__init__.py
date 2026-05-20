"""Reader subsystem.

Public surface:
    QueryPlan           multi-probe decomposition of a natural-language query
    QueryPlanner        builds a QueryPlan from a Doc
    SoftAddresser       computes the multi-cue stimulus se(q) per entity
    GatedGNN            synapse-inspired structurally-gated GNN reader
    HDCComposer         composes activated-subgraph hypervectors via bind/bundle
    MemoryReader        orchestrates planner -> addresser -> GNN -> HDC -> attribution
    ReaderOutput        dataclass returned by MemoryReader.read()
"""

from .planner import QueryPlan, QueryPlanner
from .addressing import SoftAddresser
from .gnn import GatedGNN, build_torch_graph
from .composer import HDCComposer, AttributionReport
from .reader import MemoryReader, ReaderOutput

__all__ = [
    "QueryPlan",
    "QueryPlanner",
    "SoftAddresser",
    "GatedGNN",
    "build_torch_graph",
    "HDCComposer",
    "AttributionReport",
    "MemoryReader",
    "ReaderOutput",
]
