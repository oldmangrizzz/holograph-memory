"""MemoryReader: orchestrates the full read pipeline.

Pipeline
--------
    1. plan: QueryPlanner.plan(query) -> QueryPlan
    2. address: SoftAddresser.address(substrate, plan) -> initial activations
    3. propagate: GatedGNN.forward(graph, activations) -> refined activations
    4. select top-K activated entities -> activated subgraph G_q
    5. compose: HDCComposer.compose(...) -> composed query hypervector
    6. retrieve: HDCComposer.retrieve(...) -> ranked class prototype, margin
    7. attribute: HDCComposer.attribute(...) -> multi-level attribution
    8. supporting documents via substrate.documents_for_entities

The reader holds the QueryPlanner, SoftAddresser, GatedGNN, and HDCComposer
as attributes; the runtime owns the substrate/kernel/encoder/prototype memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..graph.substrate import GraphSubstrate
from ..hdc.kernel import HDCKernel
from ..hdc.memory import PrototypeMemory
from .addressing import AddressingResult, SoftAddresser
from .composer import AttributionReport, HDCComposer
from .gnn import GatedGNN, TorchGraph, build_torch_graph
from .planner import QueryPlan, QueryPlanner


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class ReaderOutput:
    plan: QueryPlan
    initial_activation: Dict[int, float] = field(default_factory=dict)
    final_activation: Dict[int, float] = field(default_factory=dict)
    activated_ids: List[int] = field(default_factory=list)
    activated_subgraph_edges: List[Tuple[int, str, int]] = field(default_factory=list)
    composed_hv: Optional[np.ndarray] = None
    predicted_class: Optional[str] = None
    prediction_margin: float = 0.0
    prediction_scores: Dict[str, float] = field(default_factory=dict)
    attribution: Optional[AttributionReport] = None
    supporting_documents: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MemoryReader
# ---------------------------------------------------------------------------


class MemoryReader:
    """The orchestrator that turns a query into ReaderOutput."""

    def __init__(self,
                 kernel: HDCKernel,
                 planner: Optional[QueryPlanner] = None,
                 addresser: Optional[SoftAddresser] = None,
                 gnn: Optional[GatedGNN] = None,
                 composer: Optional[HDCComposer] = None,
                 top_k: int = 8,
                 query_hv_fn=None,
                 use_hierarchy: bool = False,
                 router_beam: int = 3) -> None:
        self.kernel = kernel
        self.planner = planner or QueryPlanner()
        self.addresser = addresser or SoftAddresser()
        self.gnn = gnn or GatedGNN(hidden_dim=64, n_layers=2)
        self.composer = composer or HDCComposer(kernel)
        self.top_k = int(top_k)
        # query_hv_fn: callable(text) -> np.ndarray embedding the query into the
        # entity HV space (wired by the runtime to the writer's seeder).
        self.query_hv_fn = query_hv_fn
        self.use_hierarchy = bool(use_hierarchy)
        self.router_beam = int(router_beam)
        # Populated on each read when the hierarchy path runs, for diagnostics.
        self.last_route = None

    # ---- entity vector cache for SoftAddresser ------------------------

    def _entity_vectors(self, substrate: GraphSubstrate) -> Dict[int, np.ndarray]:
        """Return spaCy doc vectors for each entity's description+canonical.

        Cached on the substrate.  Re-run only when the entity set changes.
        """
        cache_key = "_holograph_entity_vec_cache"
        cache = getattr(substrate, cache_key, None)
        ents = substrate.all_entities()
        if (cache is not None
                and cache.get("n_entities") == len(ents)):
            return cache["vectors"]
        nlp = self.planner.nlp
        out: Dict[int, np.ndarray] = {}
        for e in ents:
            text = f"{e.canonical}. {e.description}".strip()
            doc = nlp(text)
            out[e.id] = doc.vector.astype(np.float32) if doc.has_vector else np.zeros(96, dtype=np.float32)
        setattr(substrate, cache_key, {"n_entities": len(ents), "vectors": out})
        return out

    # ---- top-K selection ---------------------------------------------

    def _topk(self, scores: Dict[int, float]) -> List[int]:
        ids = list(scores.keys())
        if len(ids) <= self.top_k:
            return sorted(ids, key=lambda i: scores[i], reverse=True)
        ordered = sorted(ids, key=lambda i: scores[i], reverse=True)
        return ordered[: self.top_k]

    # ---- main entrypoint ---------------------------------------------

    def read(self,
             substrate: GraphSubstrate,
             query: str,
             memory: Optional[PrototypeMemory] = None,
             target_class: Optional[str] = None) -> ReaderOutput:
        plan = self.planner.plan(query)
        out = ReaderOutput(plan=plan)

        # --- Optional hierarchical routing (H-MEM-style candidate narrowing) ---
        # If a hierarchy exists and routing is enabled, route the query to a
        # leaf candidate set, then expand by one semantic hop so cross-branch
        # bridge nodes survive the pruning. The expensive O(N) addressing and
        # the GNN then run only over that induced subgraph.
        self.last_route = None
        candidate_ids: Optional[set] = None
        if (self.use_hierarchy
                and self.query_hv_fn is not None
                and substrate.max_layer() > 0):
            from ..hierarchy.router import SoftRouter
            try:
                qhv = self.query_hv_fn(query)
                route = SoftRouter(substrate, self.kernel, beam=self.router_beam).route(qhv)
                self.last_route = route
                cand = set(route.leaf_candidates)
                # Bridge preservation: 1-hop semantic expansion.
                for c in list(cand):
                    cand.update(substrate.neighbors_of(c, leaf_only=True))
                # Also keep any exact/alias-matched anchors from the plan so a
                # named entity is never routed away.
                for surface in plan.all_surface_cues():
                    eid = substrate.lookup_by_surface(surface)
                    if eid is not None:
                        cand.add(eid)
                if cand:
                    candidate_ids = cand
            except Exception:
                candidate_ids = None  # fall back to exhaustive on any routing error

        entity_vecs = self._entity_vectors(substrate)
        addr = self.addresser.address(substrate, plan, entity_vectors=entity_vecs,
                                       candidate_ids=candidate_ids)
        out.initial_activation = dict(addr.activation)

        # Build TorchGraph snapshot — induced subgraph when routing narrowed.
        g = substrate.to_networkx()
        if candidate_ids is not None:
            keep = [n for n in g.nodes() if n in candidate_ids]
            g = g.subgraph(keep).copy()
        torch_graph = build_torch_graph(g)
        if torch_graph.node_features.shape[0] == 0:
            return out

        # Initial activation tensor aligned with torch_graph.node_ids.
        init = torch.tensor(
            [addr.activation.get(nid, 0.0) for nid in torch_graph.node_ids],
            dtype=torch.float32,
        )
        with torch.no_grad():
            final_act, _ = self.gnn(torch_graph, init)
        final_np = final_act.detach().cpu().numpy()
        out.final_activation = {
            nid: float(final_np[i]) for i, nid in enumerate(torch_graph.node_ids)
        }

        # Top-K activated entities.
        activated = self._topk(out.final_activation)
        out.activated_ids = activated

        # Activated subgraph edges -- only edges whose endpoints are both active.
        active_set = set(activated)
        for edge in substrate.all_edges():
            if edge.head_id in active_set and edge.tail_id in active_set:
                out.activated_subgraph_edges.append((edge.head_id, edge.relation, edge.tail_id))

        # HDC composition over the activated subgraph.
        composed_hv, path_records = self.composer.compose(substrate, activated, out.final_activation)
        out.composed_hv = composed_hv

        # Prototype retrieval if a memory is attached.
        if memory is not None and memory.classes():
            cls, margin, scores = self.composer.retrieve(composed_hv, memory)
            out.predicted_class = cls
            out.prediction_margin = margin
            out.prediction_scores = scores
            out.attribution = self.composer.attribute(
                composed_hv, path_records, memory=memory, target_class=cls or target_class
            )
        else:
            # Attribution against self -- useful for explaining the composition.
            out.attribution = self.composer.attribute(composed_hv, path_records)

        # Supporting documents.
        out.supporting_documents = substrate.documents_for_entities(activated)

        return out


__all__ = ["MemoryReader", "ReaderOutput"]
