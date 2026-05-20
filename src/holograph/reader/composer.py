"""HDC composer: bind/bundle over the activated subgraph.

After the GNN propagation step, we have:
    * a ranking of entities by activation
    * the activated subgraph G_q (top-K activated nodes + their edges)

The composer turns G_q into a single query-composed hypervector via PSP-HDC's
algebra:

    For each edge (u, r, v) in G_q whose endpoints are both in the activated
    set, compose:
            path_hv = bind(h_u, bind(R_r, h_v))
    where h_u, h_v are the entities' stored hypervectors and R_r is a fixed
    random hypervector for the relation type r (per-relation symbol HV).

    Bundle all path_hv together (weighted by activation product) into the
    composed query hypervector h_q.

Prototype retrieval then runs against any class prototypes registered in the
runtime (e.g. "positive", "negative" memory tags, or PSP-style class labels).

Attribution is the killer feature of PSP-HDC and it survives the composition
because every operation is algebraic:

    Per-entity attribution: similarity(h_q_with_entity, prototype) minus
        similarity(h_q_without_entity, prototype).
    Per-edge attribution: same, dropping a specific edge.
    Per-path attribution: similarity(path_hv, prototype).

We expose these in a single AttributionReport returned from the composer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..graph.substrate import GraphSubstrate, Edge
from ..hdc.kernel import HDCKernel
from ..hdc.memory import PrototypeMemory


# ---------------------------------------------------------------------------
# Relation symbol table
# ---------------------------------------------------------------------------


class _RelationSymbols:
    """Stable per-relation random hypervectors, generated on first sight."""

    def __init__(self, kernel: HDCKernel, seed: int = 9001) -> None:
        self.kernel = kernel
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, np.ndarray] = {}
        self._self_loop_hv: Optional[np.ndarray] = None

    def get(self, relation: str) -> np.ndarray:
        if relation not in self._cache:
            # Sample a fresh hypervector in the kernel's native dtype.
            if self.kernel.name == "real":
                v = self.rng.standard_normal(self.kernel.dim).astype(np.float32)
                n = float(np.linalg.norm(v))
                v = (v / n).astype(np.float32) if n > 0 else v
            else:
                # Ternary: random ±1 with sparsity ~33% zeros.
                draw = self.rng.choice([-1, 0, 1], size=self.kernel.dim, p=[0.33, 0.34, 0.33])
                v = draw.astype(np.int8)
            self._cache[relation] = v
        return self._cache[relation]

    def self_loop(self) -> np.ndarray:
        if self._self_loop_hv is None:
            self._self_loop_hv = self.get("__SELF_LOOP__")
        return self._self_loop_hv


# ---------------------------------------------------------------------------
# Attribution report
# ---------------------------------------------------------------------------


@dataclass
class AttributionReport:
    per_entity: Dict[int, float] = field(default_factory=dict)
    per_edge: Dict[int, float] = field(default_factory=dict)
    per_path_top: List[Tuple[Tuple[int, str, int], float]] = field(default_factory=list)
    composed_hv: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class HDCComposer:
    """Composes hypervectors over an activated subgraph and retrieves against prototypes."""

    def __init__(self,
                 kernel: HDCKernel,
                 relation_symbols: Optional[_RelationSymbols] = None) -> None:
        self.kernel = kernel
        self.relations = relation_symbols or _RelationSymbols(kernel)

    # ---- composition --------------------------------------------------

    def compose(self,
                substrate: GraphSubstrate,
                activated_ids: Sequence[int],
                activations: Dict[int, float]) -> Tuple[np.ndarray, List[Tuple[int, str, int, np.ndarray, float]]]:
        """Build the composed query hypervector.

        Returns:
            composed_hv: the bundled query hypervector h_q.
            path_records: per-edge records (head_id, relation, tail_id, path_hv, weight)
                used to provide attribution at the path/edge level.
        """
        if not activated_ids:
            return self.kernel.zeros(), []

        active = set(int(i) for i in activated_ids)
        entity_hvs: Dict[int, np.ndarray] = {}
        for eid in active:
            blob = substrate.get_hv_blob(eid)
            if blob is None:
                continue
            entity_hvs[eid] = self.kernel.unpack(blob[0])

        path_records: List[Tuple[int, str, int, np.ndarray, float]] = []
        seen_edges: set[Tuple[int, str, int]] = set()
        for eid in active:
            for edge in substrate.edges_of(eid):
                u, v = edge.head_id, edge.tail_id
                key = (u, edge.relation, v)
                if key in seen_edges:
                    continue
                if u not in entity_hvs or v not in entity_hvs:
                    continue
                if u not in active or v not in active:
                    continue
                seen_edges.add(key)
                hu = entity_hvs[u]
                hv = entity_hvs[v]
                hr = self.relations.get(edge.relation)
                # path_hv = bind(hu, bind(hr, hv)) -- positional binding chain.
                inner = self.kernel.bind(hr, hv)
                path = self.kernel.bind(hu, inner)
                # Activation-product weighting.
                au = activations.get(u, 0.0)
                av = activations.get(v, 0.0)
                w = max(0.0, au * av) * float(edge.weight)
                path_records.append((u, edge.relation, v, path, w))

        # If no edges connect activated entities, fall back to bundling the
        # entity hypervectors themselves -- still produces a meaningful h_q.
        if not path_records:
            bundled = self.kernel.bundle([entity_hvs[i] for i in entity_hvs])
            return bundled, []

        # Normalize weights into a probability distribution to keep the bundle
        # well-conditioned, then bundle by repeating each path floor(w*N) times
        # (cheap surrogate for weighted bundling).
        weights = np.array([r[4] for r in path_records], dtype=np.float32)
        total = float(weights.sum())
        if total <= 0.0:
            bundled = self.kernel.bundle([r[3] for r in path_records])
        else:
            probs = weights / total
            # Scale to integer copy counts; ensure each path appears at least once.
            scale = max(8, len(path_records))
            counts = np.maximum(1, np.round(probs * scale).astype(int))
            to_bundle: List[np.ndarray] = []
            for r, c in zip(path_records, counts):
                to_bundle.extend([r[3]] * int(c))
            bundled = self.kernel.bundle(to_bundle)

        return bundled, path_records

    # ---- prototype retrieval -----------------------------------------

    def retrieve(self,
                 composed_hv: np.ndarray,
                 memory: PrototypeMemory) -> Tuple[Optional[str], float, Dict[str, float]]:
        if not memory.classes():
            return None, 0.0, {}
        cls, margin, scores = memory.predict(composed_hv)
        return cls, margin, scores

    # ---- attribution --------------------------------------------------

    def attribute(self,
                  composed_hv: np.ndarray,
                  path_records: Sequence[Tuple[int, str, int, np.ndarray, float]],
                  memory: Optional[PrototypeMemory] = None,
                  target_class: Optional[str] = None,
                  top_n_paths: int = 8) -> AttributionReport:
        """Compute multi-level attribution wrt either a class prototype or h_q itself.

        If memory + target_class given, attributions are sim drops against that
        prototype.  Otherwise they are sim drops against the composed_hv itself,
        i.e. "how much does this entity/edge/path contribute to the bundle?"
        """
        report = AttributionReport(composed_hv=composed_hv.copy())
        if not path_records:
            return report

        reference: np.ndarray
        if memory is not None and target_class is not None and target_class in memory.prototypes:
            reference = memory.prototypes[target_class]
        else:
            reference = composed_hv

        base_sim = self.kernel.similarity(composed_hv, reference)

        # ---- per-path attribution: path_hv sim to reference ---------
        path_sims = [self.kernel.similarity(p, reference) for (_, _, _, p, _) in path_records]
        # Sort by sim descending.
        order = sorted(range(len(path_records)), key=lambda i: path_sims[i], reverse=True)
        for i in order[:top_n_paths]:
            u, r, v, _, _ = path_records[i]
            report.per_path_top.append(((u, r, v), float(path_sims[i])))

        # ---- per-entity attribution: drop in similarity when entity removed
        # We build a leave-one-out composed hypervector for each entity.
        entities_in_paths: Dict[int, List[int]] = {}
        for idx, (u, _, v, _, _) in enumerate(path_records):
            entities_in_paths.setdefault(u, []).append(idx)
            entities_in_paths.setdefault(v, []).append(idx)

        weights = np.array([r[4] for r in path_records], dtype=np.float32)
        total = float(weights.sum())
        scale = max(8, len(path_records))
        if total > 0:
            counts = np.maximum(1, np.round((weights / total) * scale).astype(int))
        else:
            counts = np.ones(len(path_records), dtype=int)

        for eid, idxs in entities_in_paths.items():
            keep_paths: List[np.ndarray] = []
            for i, r in enumerate(path_records):
                if i in idxs:
                    continue
                keep_paths.extend([r[3]] * int(counts[i]))
            if keep_paths:
                drop_hv = self.kernel.bundle(keep_paths)
            else:
                drop_hv = self.kernel.zeros()
            drop_sim = self.kernel.similarity(drop_hv, reference)
            report.per_entity[eid] = float(base_sim - drop_sim)

        # ---- per-edge attribution: drop similarity when this edge removed --
        # Since the graph substrate's edge ids aren't directly in path_records,
        # we use (u, r, v) as the edge key and resolve back via substrate.
        for idx, (u, r, v, _, _) in enumerate(path_records):
            keep_paths: List[np.ndarray] = []
            for i, rec in enumerate(path_records):
                if i == idx:
                    continue
                keep_paths.extend([rec[3]] * int(counts[i]))
            drop_hv = (self.kernel.bundle(keep_paths) if keep_paths else self.kernel.zeros())
            drop_sim = self.kernel.similarity(drop_hv, reference)
            # Key by the canonical edge tuple; we don't pretend to know the
            # substrate id here -- the runtime translates as needed.
            report.per_edge[idx] = float(base_sim - drop_sim)

        return report


__all__ = ["HDCComposer", "AttributionReport"]
