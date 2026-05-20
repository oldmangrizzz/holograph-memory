"""Feedback loop: closes the writer-reader cycle.

We translate SAGE's reward bundle into concrete weight updates:

    r_ded:    binary signal "did the retrieved evidence support the answer?"
    r_rec:    recall over a gold supporting-evidence set
    r_pre:    precision over the same gold set
    r_ans:    optional end-to-end answer score (F1 against gold answer aliases)

    r_task = (alpha * r_rec + beta * r_pre + gamma * r_ded) / (alpha+beta+gamma)
    R(tau) = r_task - lambda_rep * repetition_rate + lambda_fmt * format_bonus

The composite reward is applied to:
    * substrate edge weights of edges used in the activated subgraph
        (positive reward boosts, negative reward demotes)
    * GNN gate parameters via a tiny supervised step (top-K positives = entities
      that appear in the gold supporting-evidence set)

A separate MAS update tracks the prototype geometry over time for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..graph.substrate import GraphSubstrate
from ..hdc.kernel import HDCKernel
from ..hdc.memory import PrototypeMemory, mas
from ..reader.gnn import GatedGNN, build_torch_graph
from ..reader.reader import MemoryReader, ReaderOutput
from ..writer.writer import MemoryWriter


# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------


@dataclass
class RewardSignals:
    deductive: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    answer: float = 0.0
    repetition_rate: float = 0.0
    format_bonus: float = 0.0

    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.0
    lambda_rep: float = 0.5
    lambda_fmt: float = 0.1

    def task_reward(self) -> float:
        denom = self.alpha + self.beta + self.gamma
        if denom <= 0:
            return 0.0
        return (self.alpha * self.recall
                + self.beta * self.precision
                + self.gamma * self.deductive) / denom

    def total(self) -> float:
        return (self.task_reward()
                - self.lambda_rep * self.repetition_rate
                + self.lambda_fmt * self.format_bonus)


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------


@dataclass
class FeedbackEvent:
    query: str
    reward: RewardSignals
    edges_used: List[Tuple[int, str, int]]
    gold_entity_ids: List[int]
    gold_doc_anchors: List[str]
    total_reward: float


class FeedbackLoop:
    """Closes the writer-reader loop.

    Holds references to the substrate, writer, reader, and prototype memory.
    Each `step()` call accepts a query and gold supervision, runs the reader,
    computes rewards, and applies updates.
    """

    def __init__(self,
                 substrate: GraphSubstrate,
                 kernel: HDCKernel,
                 writer: MemoryWriter,
                 reader: MemoryReader,
                 memory: PrototypeMemory,
                 gnn_lr: float = 5e-3) -> None:
        self.substrate = substrate
        self.kernel = kernel
        self.writer = writer
        self.reader = reader
        self.memory = memory
        self.gnn_lr = float(gnn_lr)
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.history: List[FeedbackEvent] = []
        self.mas_history: List[Dict[str, Dict[str, float]]] = []

    # ---- reward computation ------------------------------------------

    def _compute_reward(self,
                        output: ReaderOutput,
                        gold_doc_anchors: Sequence[str],
                        gold_entity_ids: Sequence[int],
                        answer_score: Optional[float] = None) -> RewardSignals:
        retrieved_docs = {a for (a, _) in output.supporting_documents}
        retrieved_ents = set(output.activated_ids)
        gold_docs = set(gold_doc_anchors)
        gold_ents = set(gold_entity_ids)

        # Document-level recall/precision.
        if gold_docs:
            r_rec = len(retrieved_docs & gold_docs) / len(gold_docs)
            r_pre = (len(retrieved_docs & gold_docs) / len(retrieved_docs)
                     if retrieved_docs else 0.0)
        else:
            # Fall back to entity-level if no doc gold given.
            if gold_ents:
                r_rec = len(retrieved_ents & gold_ents) / len(gold_ents)
                r_pre = (len(retrieved_ents & gold_ents) / len(retrieved_ents)
                         if retrieved_ents else 0.0)
            else:
                r_rec = 0.0
                r_pre = 0.0

        # Deductive: was the answer supported -- proxy by "any gold ent in top-K".
        r_ded = 1.0 if (retrieved_ents & gold_ents) else 0.0

        r_ans = float(answer_score) if answer_score is not None else 0.0

        # Repetition: fraction of duplicate edges in the activated subgraph.
        edges = output.activated_subgraph_edges
        rep_rate = 0.0
        if edges:
            unique_edges = {e for e in edges}
            rep_rate = 1.0 - (len(unique_edges) / len(edges))

        return RewardSignals(
            deductive=r_ded,
            recall=r_rec,
            precision=r_pre,
            answer=r_ans,
            repetition_rate=rep_rate,
            format_bonus=0.0,
        )

    # ---- writer update -----------------------------------------------

    def _apply_writer_update(self,
                              output: ReaderOutput,
                              total_reward: float,
                              gold_entity_ids: Sequence[int]) -> None:
        """Boost edges that connect to gold entities; demote those that don't."""
        if not output.activated_subgraph_edges:
            return
        gold_set = set(gold_entity_ids)
        for (u, r, v) in output.activated_subgraph_edges:
            # Look up the edge id in the substrate.
            for edge in self.substrate.edges_of(u):
                if (edge.head_id == u and edge.tail_id == v and edge.relation == r):
                    sign = 1.0 if (u in gold_set or v in gold_set) else -0.5
                    self.writer.reinforce_edges([edge.id], reward=sign * total_reward)
                    break

    # ---- GNN supervised step -----------------------------------------

    def _apply_gnn_update(self,
                           output: ReaderOutput,
                           gold_entity_ids: Sequence[int]) -> float:
        """Run one optimization step on the GNN using the gold entity set."""
        if not gold_entity_ids:
            return 0.0
        torch_graph = build_torch_graph(self.substrate.to_networkx())
        if torch_graph.node_features.shape[0] == 0:
            return 0.0
        init = torch.tensor(
            [output.initial_activation.get(nid, 0.0) for nid in torch_graph.node_ids],
            dtype=torch.float32,
        )
        # Build target: 1.0 for gold entities, 0.0 for everything else.
        gold_set = set(gold_entity_ids)
        target = torch.tensor(
            [1.0 if nid in gold_set else 0.0 for nid in torch_graph.node_ids],
            dtype=torch.float32,
        )
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.reader.gnn.parameters(), lr=self.gnn_lr)
        self._optimizer.zero_grad()
        final, _ = self.reader.gnn(torch_graph, init)
        # Margin loss: gold entities should be ranked above non-gold.
        if target.sum() == 0 or target.sum() == target.numel():
            return 0.0
        gold_scores = final[target.bool()]
        nongold_scores = final[~target.bool()]
        # Pairwise hinge: enforce gold > nongold + 1.
        loss = F.relu(1.0 + nongold_scores.unsqueeze(0) - gold_scores.unsqueeze(1)).mean()
        # Plus BCE on a sigmoided final to keep magnitudes bounded.
        loss = loss + F.binary_cross_entropy_with_logits(final, target)
        loss.backward()
        self._optimizer.step()
        return float(loss.detach().item())

    # ---- MAS tracking ------------------------------------------------

    def _track_mas(self,
                    component_memories: Dict[str, Dict[str, np.ndarray]]) -> None:
        if not self.memory.prototypes:
            return
        snapshot = mas(self.kernel, component_memories, self.memory.prototypes)
        self.mas_history.append(snapshot)

    # ---- one feedback step -------------------------------------------

    def step(self,
             query: str,
             gold_doc_anchors: Sequence[str] = (),
             gold_entity_ids: Sequence[int] = (),
             answer_score: Optional[float] = None,
             component_memories: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
             target_class: Optional[str] = None,
             decay_half_life: Optional[float] = None) -> FeedbackEvent:
        """Run reader, compute rewards, apply updates; return the event record.

        If `decay_half_life` is given, apply Ebbinghaus decay to all edges
        BEFORE reinforcing the ones used in this query — so unused memories
        fade while used ones reset their decay clock (handled in the writer's
        reinforce path via touch_edge).
        """
        if decay_half_life is not None:
            self.substrate.decay_edges(float(decay_half_life))
        output = self.reader.read(self.substrate, query, memory=self.memory, target_class=target_class)
        rewards = self._compute_reward(output, gold_doc_anchors, gold_entity_ids, answer_score)
        total_reward = rewards.total()
        self._apply_writer_update(output, total_reward, gold_entity_ids)
        loss = self._apply_gnn_update(output, gold_entity_ids)
        if component_memories is not None:
            self._track_mas(component_memories)
        ev = FeedbackEvent(
            query=query,
            reward=rewards,
            edges_used=output.activated_subgraph_edges,
            gold_entity_ids=list(gold_entity_ids),
            gold_doc_anchors=list(gold_doc_anchors),
            total_reward=total_reward,
        )
        self.history.append(ev)
        return ev


__all__ = ["RewardSignals", "FeedbackLoop", "FeedbackEvent"]
