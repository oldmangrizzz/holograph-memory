"""Structural hierarchy builder.

Builds H-MEM-style abstraction layers over the leaf (layer-0) semantic graph,
entirely structurally and deterministically — no LLM in the loop. The
construction is the natural HDC operation: a parent summary node's hypervector
is the BUNDLE of its children's hypervectors, so the abstraction layer is not
a graft, it is the same algebra the kernel already implements.

Algorithm
---------
Starting at layer 0 (all current leaf entities):

    while the current layer has more than `node_cap` nodes and we are below
    `max_layer`:
        1. detect communities among the current layer's nodes using the
           semantic edges that connect them (greedy modularity on the
           weighted undirected projection);
        2. for each community, create a parent entity at layer+1 whose
           hypervector is bundle(child hypervectors), and whose canonical
           name is synthesized as "L{layer+1}::{centroid-child-canonical}";
        3. record parent->child hierarchy edges;
        4. ascend: the parents become the current layer.

Nodes at the current layer that have no semantic edges (isolates) are each
promoted as singleton parents so nothing is dropped from the routing tree.

An optional `summarizer` hook is accepted so a future LLM-backed layer (richer
abstraction) can be slotted in without changing the interface; when None
(the default), construction stays structural and LLM-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from ..graph.substrate import GraphSubstrate
from ..hdc.kernel import HDCKernel


# A summarizer maps (list of child canonical names, list of child descriptions)
# to (parent canonical, parent description). Optional; structural default omits.
Summarizer = Callable[[List[str], List[str]], "tuple[str, str]"]


@dataclass
class HierarchyStats:
    layers_built: int = 0
    nodes_per_layer: Dict[int, int] = field(default_factory=dict)
    total_summary_nodes: int = 0


class HierarchyBuilder:
    """Builds and refreshes the abstraction hierarchy over a substrate."""

    def __init__(self,
                 substrate: GraphSubstrate,
                 kernel: HDCKernel,
                 max_layer: int = 4,
                 node_cap: int = 8,
                 resolution: float = 1.0,
                 summarizer: Optional[Summarizer] = None,
                 seed: int = 0) -> None:
        self.substrate = substrate
        self.kernel = kernel
        self.max_layer = int(max_layer)
        self.node_cap = int(node_cap)
        self.resolution = float(resolution)
        self.summarizer = summarizer
        self.seed = seed

    # ---- public --------------------------------------------------------

    def build(self) -> HierarchyStats:
        """(Re)build the hierarchy. Clears any existing summary nodes first."""
        self.substrate.clear_hierarchy()
        stats = HierarchyStats()
        leaf_ids = self.substrate.entities_at_layer(0)
        stats.nodes_per_layer[0] = len(leaf_ids)
        if not leaf_ids:
            return stats

        current_layer = 0
        current_ids = list(leaf_ids)
        hv_cache: Dict[int, np.ndarray] = {}
        for eid in current_ids:
            hv_cache[eid] = self._load_hv(eid)

        while len(current_ids) > self.node_cap and current_layer < self.max_layer:
            communities = self._communities_among(current_ids)
            # Stop if community detection can't reduce the node count at all
            # (every node became its own singleton) — further layers would be
            # identity maps and waste routing depth.
            if len(communities) >= len(current_ids):
                break

            parent_ids: List[int] = []
            parent_hvs: Dict[int, np.ndarray] = {}
            child_to_parent: Dict[int, int] = {}
            for comm in communities:
                child_ids = list(comm)
                child_hvs = [hv_cache[c] for c in child_ids if c in hv_cache]
                if not child_hvs:
                    continue
                parent_hv = self.kernel.bundle(child_hvs)
                pid = self._create_parent(child_ids, parent_hv, current_layer + 1)
                for c in child_ids:
                    self.substrate.add_hierarchy_edge(pid, c, current_layer)
                    child_to_parent[c] = pid
                parent_ids.append(pid)
                parent_hvs[pid] = parent_hv

            if not parent_ids:
                break

            # Aggregate this level's edges upward into summary edges between
            # parents, so the NEXT round's community detection has connectivity
            # to work with. Without this the tree stalls one layer up.
            self._aggregate_edges_upward(current_ids, child_to_parent, current_layer + 1)

            current_layer += 1
            current_ids = parent_ids
            hv_cache = parent_hvs
            stats.nodes_per_layer[current_layer] = len(parent_ids)
            stats.layers_built += 1
            stats.total_summary_nodes += len(parent_ids)

            if len(current_ids) <= self.node_cap:
                break

        return stats

    def _aggregate_edges_upward(self, level_ids: Sequence[int],
                                 child_to_parent: Dict[int, int],
                                 parent_layer: int) -> None:
        """Create weighted summary edges between parents from cross-edges among
        their children at the current level."""
        id_set = set(level_ids)
        agg: Dict[tuple, float] = {}
        seen_edge_ids: set[int] = set()
        for cid in id_set:
            for edge in self.substrate.edges_of(cid):
                if edge.id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge.id)
                u, v = edge.head_id, edge.tail_id
                if u not in id_set or v not in id_set or u == v:
                    continue
                pu, pv = child_to_parent.get(u), child_to_parent.get(v)
                if pu is None or pv is None or pu == pv:
                    continue
                key = (min(pu, pv), max(pu, pv))
                agg[key] = agg.get(key, 0.0) + float(edge.weight)
        rel = f"__summary_L{parent_layer}__"
        for (a, b), w in agg.items():
            self.substrate.upsert_edge(a, b, rel, weight=min(w, 10.0), source="")

    def top_layer(self) -> int:
        return self.substrate.max_layer()

    # ---- internals -----------------------------------------------------

    def _load_hv(self, eid: int) -> np.ndarray:
        blob = self.substrate.get_hv_blob(eid)
        if blob is None:
            return self.kernel.zeros()
        return self.kernel.unpack(blob[0])

    def _communities_among(self, ids: Sequence[int]) -> List[set]:
        """Greedy-modularity communities among `ids` using semantic edges.

        Isolated nodes (no semantic edge to another node in `ids`) each become
        their own singleton community so nothing is lost.
        """
        id_set = set(ids)
        ug = nx.Graph()
        ug.add_nodes_from(id_set)
        for eid in id_set:
            for edge in self.substrate.edges_of(eid):
                u, v = edge.head_id, edge.tail_id
                if u in id_set and v in id_set and u != v:
                    if ug.has_edge(u, v):
                        ug[u][v]["weight"] += float(edge.weight)
                    else:
                        ug.add_edge(u, v, weight=float(edge.weight))
        if ug.number_of_edges() == 0:
            return [{i} for i in id_set]
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(
                ug, resolution=self.resolution, weight="weight"
            )
            comms = [set(c) for c in comms]
        except Exception:
            comms = [{i} for i in id_set]
        # Ensure every id landed in exactly one community (isolates included).
        covered = set().union(*comms) if comms else set()
        for i in id_set - covered:
            comms.append({i})
        return comms

    def _create_parent(self, child_ids: List[int], parent_hv: np.ndarray,
                        layer: int) -> int:
        """Create a summary entity at `layer` and store its bundled HV."""
        # Choose a representative child for naming: the one whose HV is most
        # similar to the parent bundle (the centroid-most child).
        rep = self._centroid_child(child_ids, parent_hv)
        rep_ent = self.substrate.get_entity(rep) if rep is not None else None
        rep_name = rep_ent.canonical if rep_ent is not None else f"node{child_ids[0]}"

        if self.summarizer is not None:
            names = [self.substrate.get_entity(c).canonical for c in child_ids
                     if self.substrate.get_entity(c) is not None]
            descs = [self.substrate.get_entity(c).description for c in child_ids
                     if self.substrate.get_entity(c) is not None]
            canon, desc = self.summarizer(names, descs)
            # Disambiguate the canonical with the layer to avoid collisions.
            canon = f"L{layer}::{canon}"
        else:
            canon = f"L{layer}::{rep_name}::{child_ids[0]}"
            desc = f"summary of {len(child_ids)} layer-{layer-1} nodes"

        pid = self.substrate.upsert_entity(canon, type="summary", description=desc)
        self.substrate.set_layer(pid, layer)
        self.substrate.set_hv(pid, self.kernel.pack(parent_hv), self.kernel.name)
        return pid

    def _centroid_child(self, child_ids: List[int],
                        parent_hv: np.ndarray) -> Optional[int]:
        best = None
        best_sim = -2.0
        for c in child_ids:
            blob = self.substrate.get_hv_blob(c)
            if blob is None:
                continue
            hv = self.kernel.unpack(blob[0])
            sim = self.kernel.similarity(hv, parent_hv)
            if sim > best_sim:
                best_sim = sim
                best = c
        return best if best is not None else (child_ids[0] if child_ids else None)


def build_hierarchy(substrate: GraphSubstrate,
                    kernel: HDCKernel,
                    max_layer: int = 4,
                    node_cap: int = 8,
                    resolution: float = 1.0,
                    summarizer: Optional[Summarizer] = None) -> HierarchyStats:
    """Convenience wrapper: build the hierarchy and return stats."""
    builder = HierarchyBuilder(
        substrate, kernel, max_layer=max_layer, node_cap=node_cap,
        resolution=resolution, summarizer=summarizer,
    )
    return builder.build()


__all__ = ["HierarchyBuilder", "HierarchyStats", "build_hierarchy", "Summarizer"]
