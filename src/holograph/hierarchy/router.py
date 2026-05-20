"""Soft top-down index router.

Implements H-MEM's layer-by-layer routing, but *soft*: at each layer we keep
the top-`beam` parents rather than a single top-1, and we descend through the
hierarchy pointers to their children. This is the efficiency win — we never
score the whole leaf set, only the top layer plus the children along the kept
branches.

The softness (beam > 1) is the deliberate departure from a hard tree descent:
it widens the funnel so the eventual graph-propagation step (run by the reader)
can still reach bridge nodes that a hard top-1 descent would have pruned. The
router's job is to cheaply narrow N leaves down to a candidate set; the reader
then expands that set by one semantic hop and propagates, which is what
actually recovers cross-branch bridges.

If no hierarchy exists (max_layer == 0), `route` returns all leaves, i.e. it
degenerates to the exhaustive behavior — so the reader can always fall back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph.substrate import GraphSubstrate
from ..hdc.kernel import HDCKernel


@dataclass
class RouteResult:
    leaf_candidates: List[int] = field(default_factory=list)
    nodes_touched: int = 0            # how many similarity comparisons were done
    total_leaves: int = 0             # for efficiency ratio
    path_by_layer: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def touch_fraction(self) -> float:
        if self.total_leaves <= 0:
            return 1.0
        return self.nodes_touched / float(self.total_leaves)


class SoftRouter:
    """Routes a query hypervector to a leaf candidate set via the hierarchy."""

    def __init__(self,
                 substrate: GraphSubstrate,
                 kernel: HDCKernel,
                 beam: int = 3,
                 leaf_beam_multiplier: int = 4) -> None:
        self.substrate = substrate
        self.kernel = kernel
        self.beam = int(beam)
        # At the leaf layer we keep more candidates (beam * multiplier) so the
        # downstream GNN has room to work and bridges aren't starved.
        self.leaf_beam_multiplier = int(leaf_beam_multiplier)

    def route(self, query_hv: np.ndarray) -> RouteResult:
        total_leaves = len(self.substrate.entities_at_layer(0))
        result = RouteResult(total_leaves=total_leaves)
        top = self.substrate.max_layer()

        if top == 0:
            # No hierarchy — degenerate to exhaustive (the reader's reference path).
            leaves = self.substrate.entities_at_layer(0)
            result.leaf_candidates = leaves
            result.nodes_touched = len(leaves)
            result.path_by_layer[0] = leaves
            return result

        # Score the top layer.
        layer = top
        candidates = self.substrate.entities_at_layer(layer)
        kept = self._score_and_keep(query_hv, candidates, self.beam)
        result.nodes_touched += len(candidates)
        result.path_by_layer[layer] = [c for c, _ in kept]

        # Descend through the hierarchy.
        while layer > 0:
            children: List[int] = []
            seen: set[int] = set()
            for parent, _ in kept:
                for child in self.substrate.children_of(parent):
                    if child not in seen:
                        seen.add(child)
                        children.append(child)
            if not children:
                break
            beam = self.beam if layer - 1 > 0 else self.beam * self.leaf_beam_multiplier
            kept = self._score_and_keep(query_hv, children, beam)
            result.nodes_touched += len(children)
            layer -= 1
            result.path_by_layer[layer] = [c for c, _ in kept]

        result.leaf_candidates = [c for c, _ in kept]
        return result

    # ---- internals -----------------------------------------------------

    def _score_and_keep(self, query_hv: np.ndarray, ids: Sequence[int],
                        k: int) -> List[tuple]:
        scored: List[tuple] = []
        for eid in ids:
            blob = self.substrate.get_hv_blob(eid)
            if blob is None:
                continue
            hv = self.kernel.unpack(blob[0])
            sim = self.kernel.similarity(query_hv, hv)
            scored.append((eid, sim))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[: max(1, k)]


__all__ = ["SoftRouter", "RouteResult"]
