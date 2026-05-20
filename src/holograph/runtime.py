"""HoloGraph runtime: top-level convenience facade.

The runtime constructs and holds:
    - kernel (Real or Ternary HDC)
    - substrate (SQLite graph)
    - writer (text -> graph)
    - prototype memory (class prototypes for retrieval)
    - reader (query -> ReaderOutput)
    - feedback loop (closed-loop updates)

Usage:

    hg = HoloGraph(kernel_kind="real", db_path=":memory:")
    hg.ingest_text("Alice met Bob in Paris.", anchor="doc-1")
    hg.register_class("greeting", [pretrained_hv1, pretrained_hv2])
    out = hg.read("Where did Alice meet Bob?")

    # Closed-loop fine-tuning:
    hg.feedback("Where did Alice meet Bob?",
                gold_doc_anchors=["doc-1"],
                gold_entity_ids=[hg.substrate.lookup_by_surface("Paris")])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np

from .feedback.loop import FeedbackEvent, FeedbackLoop, RewardSignals
from .graph.substrate import GraphSubstrate, Entity, Edge
from .hdc.encoder import ScalarEncoder
from .hdc.kernel import HDCKernel, make_kernel
from .hdc.memory import PrototypeMemory, mas
from .reader.reader import MemoryReader, ReaderOutput
from .writer.writer import ExtractedTriple, MemoryWriter


class HoloGraph:
    """End-to-end runtime: kernel + substrate + writer + reader + feedback."""

    def __init__(self,
                 kernel_kind: str = "real",
                 dim: Optional[int] = None,
                 deadband: float = 0.0,
                 db_path: str | Path = ":memory:",
                 top_k: int = 8,
                 reader_temperature: float = 0.5,
                 reinforce_eta: float = 0.5,
                 demote_eta: float = 0.25,
                 gnn_lr: float = 5e-3,
                 use_hierarchy: bool = False,
                 router_beam: int = 3,
                 spacy_nlp=None) -> None:
        self.kernel: HDCKernel = make_kernel(kernel_kind, dim=dim, deadband=deadband)
        self.substrate = GraphSubstrate(db_path)
        self.writer = MemoryWriter(
            self.substrate, self.kernel,
            nlp=spacy_nlp,
            reinforce_eta=reinforce_eta,
            demote_eta=demote_eta,
        )
        # Reader -- planner shares the same spaCy nlp as the writer for efficiency.
        from .reader.planner import QueryPlanner
        from .reader.addressing import SoftAddresser
        self.reader = MemoryReader(
            self.kernel,
            planner=QueryPlanner(self.writer.nlp),
            addresser=SoftAddresser(temperature=reader_temperature),
            top_k=top_k,
            query_hv_fn=self.writer.seed_hypervector,
            use_hierarchy=use_hierarchy,
            router_beam=router_beam,
        )
        self.memory = PrototypeMemory(self.kernel)
        self.feedback_loop = FeedbackLoop(
            self.substrate, self.kernel, self.writer, self.reader,
            self.memory, gnn_lr=gnn_lr,
        )
        # Optional scalar encoder for PSP-style structured demos.
        self.scalar_encoder: Optional[ScalarEncoder] = None

    # ---- ingestion ----------------------------------------------------

    def ingest_text(self, text: str, anchor: str) -> List[ExtractedTriple]:
        return self.writer.ingest_text(text, anchor=anchor)

    def ingest_triples(self, triples: Iterable[ExtractedTriple]) -> List[int]:
        return self.writer.write_triples(triples)

    # ---- hierarchical memory (H-MEM-style) ----------------------------

    def build_hierarchy(self, max_layer: int = 4, node_cap: int = 8,
                        resolution: float = 1.0, summarizer=None):
        """Construct abstraction layers over the current leaf graph.

        Returns HierarchyStats. Enables the routed retrieval path; subsequent
        read() calls use index routing if use_hierarchy was set at construction
        (or set hg.reader.use_hierarchy = True afterward).
        """
        from .hierarchy.builder import build_hierarchy as _bh
        return _bh(self.substrate, self.kernel, max_layer=max_layer,
                   node_cap=node_cap, resolution=resolution, summarizer=summarizer)

    def enable_hierarchy_routing(self, on: bool = True, beam: Optional[int] = None) -> None:
        self.reader.use_hierarchy = bool(on)
        if beam is not None:
            self.reader.router_beam = int(beam)

    # ---- temporal forgetting ------------------------------------------

    def decay(self, half_life_seconds: float, floor: float = 0.0) -> int:
        """Apply Ebbinghaus-style decay to all semantic edge weights."""
        return self.substrate.decay_edges(half_life_seconds, floor=floor)

    # ---- structured PSP-style ingestion ------------------------------

    def configure_scalar_encoder(self, param_names: List[str], embed_dim: int = 64,
                                   seed: int = 0) -> ScalarEncoder:
        self.scalar_encoder = ScalarEncoder(param_names, kernel=self.kernel,
                                             embed_dim=embed_dim, seed=seed)
        return self.scalar_encoder

    def encode_psp_sample(self,
                          x_row: np.ndarray,
                          groups: Optional[Dict[str, List[int]]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Encode a structured PSP-style sample.

        Args:
            x_row: 1-D vector of P scalar parameter values for the sample.
            groups: optional mapping {group_name: [param indices]} -- creates
                bundled group hypervectors per PSP-HDC Eq. (10).

        Returns:
            sample_hv: the bundled sample hypervector.
            group_hvs: a dict {group_name: bundled HV} for attribution.
        """
        if self.scalar_encoder is None:
            raise RuntimeError("call configure_scalar_encoder() first")
        per_param = self.scalar_encoder.encode_row(x_row)
        group_hvs: Dict[str, np.ndarray] = {}
        if groups:
            for gname, indices in groups.items():
                group_hvs[gname] = self.kernel.bundle([per_param[i] for i in indices])
            sample_hv = self.kernel.bundle(list(group_hvs.values()))
        else:
            sample_hv = self.kernel.bundle(per_param)
        return sample_hv, group_hvs

    def fit_psp_prototypes(self,
                            X: np.ndarray,
                            y: Sequence[str],
                            groups: Optional[Dict[str, List[int]]] = None) -> None:
        if self.scalar_encoder is None:
            raise RuntimeError("call configure_scalar_encoder() first")
        self.scalar_encoder.fit_normalization(X)
        per_class: Dict[str, List[np.ndarray]] = {}
        for row, label in zip(X, y):
            hv, _ = self.encode_psp_sample(row, groups=groups)
            per_class.setdefault(label, []).append(hv)
        self.memory.fit(per_class)

    # ---- registering class prototypes (text-domain) ------------------

    def register_class(self, class_name: str, sample_hvs: Sequence[np.ndarray]) -> None:
        """Form a class prototype from a set of representative hypervectors."""
        cur = self.memory.prototypes.get(class_name)
        all_hvs = list(sample_hvs)
        if cur is not None:
            all_hvs.append(cur)
        self.memory.prototypes[class_name] = self.kernel.bundle(all_hvs)

    # ---- read ---------------------------------------------------------

    def read(self, query: str, target_class: Optional[str] = None) -> ReaderOutput:
        return self.reader.read(self.substrate, query, memory=self.memory, target_class=target_class)

    # ---- feedback -----------------------------------------------------

    def feedback(self,
                 query: str,
                 gold_doc_anchors: Sequence[str] = (),
                 gold_entity_ids: Sequence[int] = (),
                 answer_score: Optional[float] = None,
                 target_class: Optional[str] = None) -> FeedbackEvent:
        return self.feedback_loop.step(
            query=query,
            gold_doc_anchors=gold_doc_anchors,
            gold_entity_ids=gold_entity_ids,
            answer_score=answer_score,
            target_class=target_class,
        )

    # ---- diagnostics --------------------------------------------------

    def mas_report(self, component_memories: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
        return mas(self.kernel, component_memories, self.memory.prototypes)

    def summary(self) -> Dict[str, int]:
        return {
            "entities": self.substrate.n_entities(),
            "edges": self.substrate.n_edges(),
            "classes": len(self.memory.prototypes),
            "kernel_dim": self.kernel.dim,
        }

    def close(self) -> None:
        self.substrate.close()


__all__ = ["HoloGraph"]
