"""Synapse-inspired structurally-gated GNN reader.

This is the SAGE-style propagation core: a small message-passing network whose
edges carry vector-valued gates that depend on (i) node structural features,
(ii) edge structural features, and (iii) a graph-level summary.  The gates
implement the three behaviors SAGE motivates from synaptic biology:

    * Inhibition of non-specific generalized memories -- hubs are suppressed.
    * Lateral thinking / bridge edge preservation     -- bridges are amplified.
    * Habituation                                     -- redundant edges decay.

We use the same gating recipe Wang et al. spell out in Eqs. (2)-(4):

    phi(v)   = [log(1+d_v), c_v, kappa_v, d_bar_{N(v)}]
    psi(u,v) = [|d_u - d_v|, |N(u) cap N(v)|, jaccard(N(u), N(v))]
    rG       = [mean_v phi(v); std_v phi(v); density(G)]

The per-layer edge gate is

    z_uv = [E_n(phi(u)); E_n(phi(v)); E_p(psi(u,v)); E_g(rG)]
    g_uv = 1 + delta * tanh(MLP_g(z_uv))

and the message + node-update is

    m_{u->v} = eta_uv * g_uv * W_m h_u
    h_v_new  = LayerNorm(h_v + PReLU(b + sum_{u in N(v)} m_{u->v}))

where eta_uv is the normalized adjacency weight (degree-renormalized GCN-style)
and h_v is the per-node hidden state.

The GNN is trainable from the feedback loop; we provide a `train_step()` that
takes a per-query supervision target (top-K positive entity ids) and runs one
optimizer step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: build per-graph tensors
# ---------------------------------------------------------------------------


@dataclass
class TorchGraph:
    """A torch-friendly snapshot of the substrate's graph.

    Fields:
        node_ids:     ordered list of substrate entity ids
        node_index:   map substrate id -> row index in tensors
        node_features: (N, F_n) structural features phi(v)
        edge_index:   (2, E) source/target row indices
        edge_features: (E, F_e) structural features psi(u,v)
        edge_weights: (E,) substrate edge weights (used for eta_uv)
        graph_summary: (F_g,) graph-level r_G
    """

    node_ids: List[int]
    node_index: Dict[int, int]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    edge_weights: torch.Tensor
    graph_summary: torch.Tensor


def _node_features(g: nx.MultiDiGraph) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Compute phi(v) for every node and the average neighbor log-degree.

    Returns:
        node_feats: {node_id: 4-vector [log_deg, clustering, k_core, deg_avg_N]}
        deg_vec: degree vector aligned with sorted node order.
    """
    ug = nx.Graph()
    ug.add_nodes_from(g.nodes())
    for u, v, data in g.edges(data=True):
        w = data.get("weight", 1.0)
        if ug.has_edge(u, v):
            ug[u][v]["weight"] = max(ug[u][v]["weight"], w)
        else:
            ug.add_edge(u, v, weight=w)
    deg: Dict[int, int] = dict(ug.degree())
    clustering = nx.clustering(ug)
    try:
        core_number = nx.core_number(ug)
    except Exception:
        core_number = {n: 0 for n in ug.nodes()}
    feats: Dict[int, np.ndarray] = {}
    for n in g.nodes():
        d = float(deg.get(n, 0))
        avg_neigh = (
            float(np.mean([deg.get(m, 0) for m in ug.neighbors(n)]))
            if list(ug.neighbors(n)) else 0.0
        )
        feats[n] = np.array([
            float(np.log1p(d)),
            float(clustering.get(n, 0.0)),
            float(core_number.get(n, 0.0)),
            float(np.log1p(avg_neigh)),
        ], dtype=np.float32)
    return feats, np.array(list(deg.values()), dtype=np.float32)


def _edge_features(g: nx.MultiDiGraph, node_feats: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute psi(u,v) for the (collapsed-undirected) edge set.

    Returns (E_undirected, 3): [|du-dv|, common_neighbors, jaccard].
    """
    ug = nx.Graph()
    ug.add_nodes_from(g.nodes())
    for u, v, _ in g.edges(data=True):
        if u != v:
            ug.add_edge(u, v)
    out: List[Tuple[int, int, np.ndarray]] = []
    for u, v in ug.edges():
        nu = set(ug.neighbors(u))
        nv = set(ug.neighbors(v))
        common = len(nu & nv)
        union = len(nu | nv)
        jac = (common / union) if union > 0 else 0.0
        du = node_feats[u][0]
        dv = node_feats[v][0]
        out.append((u, v, np.array([
            float(abs(du - dv)),
            float(common),
            float(jac),
        ], dtype=np.float32)))
    return out  # type: ignore[return-value]


def build_torch_graph(g: nx.MultiDiGraph) -> TorchGraph:
    """Translate a substrate's NetworkX graph into a TorchGraph snapshot."""
    if g.number_of_nodes() == 0:
        return TorchGraph(
            node_ids=[], node_index={},
            node_features=torch.zeros((0, 4), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, 3), dtype=torch.float32),
            edge_weights=torch.zeros((0,), dtype=torch.float32),
            graph_summary=torch.zeros((9,), dtype=torch.float32),
        )

    node_ids = sorted(g.nodes())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    node_feats, _ = _node_features(g)
    F_n = 4
    nf = torch.tensor(np.stack([node_feats[nid] for nid in node_ids]), dtype=torch.float32)

    # Edges: use both directions of the original multi-di-graph; collapse parallels
    # by averaging weight (the relation identity is consumed by the HDC layer,
    # not the GNN, so we don't need it here).
    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    seen: Dict[Tuple[int, int], float] = {}
    for u, v, data in g.edges(data=True):
        if u == v:
            continue
        w = float(data.get("weight", 1.0))
        key = (u, v)
        seen[key] = max(seen.get(key, 0.0), w)
    for (u, v), w in seen.items():
        src.append(node_index[u]); dst.append(node_index[v]); weights.append(w)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_weights = torch.tensor(weights, dtype=torch.float32) if weights else torch.zeros((0,), dtype=torch.float32)

    # Edge structural features psi(u,v) computed on the undirected projection.
    ug = nx.Graph()
    ug.add_nodes_from(node_ids)
    for u, v in seen:
        ug.add_edge(u, v)
    ef_list: List[np.ndarray] = []
    if seen:
        for (u, v) in seen:
            nu = set(ug.neighbors(u))
            nv = set(ug.neighbors(v))
            common = len(nu & nv)
            union = len(nu | nv)
            jac = (common / union) if union > 0 else 0.0
            du = node_feats[u][0]
            dv = node_feats[v][0]
            ef_list.append(np.array([abs(du - dv), float(common), jac], dtype=np.float32))
        ef = torch.tensor(np.stack(ef_list), dtype=torch.float32)
    else:
        ef = torch.zeros((0, 3), dtype=torch.float32)

    # Graph summary r_G.
    if nf.shape[0] > 0:
        mean_phi = nf.mean(dim=0)
        std_phi = nf.std(dim=0, unbiased=False)
    else:
        mean_phi = torch.zeros(F_n); std_phi = torch.zeros(F_n)
    n_nodes = nf.shape[0]
    max_edges = max(n_nodes * (n_nodes - 1) // 2, 1)
    density = float(len(seen) / max_edges) if n_nodes > 1 else 0.0
    rG = torch.cat([mean_phi, std_phi, torch.tensor([density], dtype=torch.float32)], dim=0)

    return TorchGraph(
        node_ids=node_ids,
        node_index=node_index,
        node_features=nf,
        edge_index=edge_index,
        edge_features=ef,
        edge_weights=edge_weights,
        graph_summary=rG,
    )


# ---------------------------------------------------------------------------
# Gated GNN module
# ---------------------------------------------------------------------------


class _LinearGate(nn.Module):
    """The vector edge gate g_uv = 1 + delta * tanh(MLP([E_n(u); E_n(v); E_p; E_g]))."""

    def __init__(self,
                 node_feat_dim: int = 4,
                 edge_feat_dim: int = 3,
                 graph_feat_dim: int = 9,
                 embed_dim: int = 16,
                 hidden_dim: int = 32,
                 delta: float = 0.5) -> None:
        super().__init__()
        self.delta = float(delta)
        self.E_n = nn.Linear(node_feat_dim, embed_dim)
        self.E_p = nn.Linear(edge_feat_dim, embed_dim)
        self.E_g = nn.Linear(graph_feat_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # scalar gate per edge (broadcast across feature dim later)
        )

    def forward(self,
                phi_u: torch.Tensor,
                phi_v: torch.Tensor,
                psi_uv: torch.Tensor,
                rG: torch.Tensor) -> torch.Tensor:
        Eu = self.E_n(phi_u)
        Ev = self.E_n(phi_v)
        Ep = self.E_p(psi_uv)
        Eg = self.E_g(rG).expand(Eu.shape[0], -1)
        z = torch.cat([Eu, Ev, Ep, Eg], dim=-1)
        return 1.0 + self.delta * torch.tanh(self.mlp(z))


class GatedGNN(nn.Module):
    """Multi-layer structurally-gated GCN that propagates query activations.

    Input per call:
        graph: TorchGraph (built from the substrate's NetworkX graph)
        initial_activation: (N,) initial p0(e|q) per node

    Output:
        final_activation: (N,) refined activation per node
        edge_gates_per_layer: list of (E,) gate values (for attribution)
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 input_extra_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        # Input embed: [activation; node_features (4)] -> hidden_dim
        self.input_proj = nn.Linear(1 + input_extra_dim, hidden_dim)
        self.gates = nn.ModuleList([_LinearGate() for _ in range(self.n_layers)])
        self.msg_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)]
        )
        self.prelu = nn.PReLU(hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        # Output projection to scalar activation.
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                graph: TorchGraph,
                initial_activation: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        N = graph.node_features.shape[0]
        device = graph.node_features.device
        if N == 0:
            return torch.zeros((0,), device=device), []

        # Stack activation as one extra channel.
        x = torch.cat([initial_activation.unsqueeze(-1), graph.node_features], dim=-1)  # (N, 1+F_n)
        h = self.input_proj(x)  # (N, hidden)

        # Pre-compute normalized adjacency weights eta_uv (symmetric GCN norm).
        if graph.edge_index.shape[1] == 0:
            final = self.out_head(h).squeeze(-1)
            return final, []
        deg = torch.zeros(N, dtype=torch.float32, device=device)
        # Degree from undirected projection: we add for both directions.
        deg.scatter_add_(0, graph.edge_index[0], graph.edge_weights)
        deg.scatter_add_(0, graph.edge_index[1], graph.edge_weights)
        deg = deg.clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)

        gate_log: List[torch.Tensor] = []

        for layer in range(self.n_layers):
            src = graph.edge_index[0]
            dst = graph.edge_index[1]
            phi_u = graph.node_features[src]
            phi_v = graph.node_features[dst]
            g_uv = self.gates[layer](phi_u, phi_v, graph.edge_features, graph.graph_summary)  # (E, 1)
            gate_log.append(g_uv.detach().squeeze(-1).clone())
            eta_uv = d_inv_sqrt[src] * graph.edge_weights * d_inv_sqrt[dst]  # (E,)
            m = (eta_uv.unsqueeze(-1) * g_uv) * self.msg_proj[layer](h[src])  # (E, hidden)
            # Aggregate by sum into destinations; do the reverse direction too
            # to mirror an undirected propagation step.
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            # And the reverse: messages from v->u as well, sharing the gate.
            m_rev = (eta_uv.unsqueeze(-1) * g_uv) * self.msg_proj[layer](h[dst])
            agg.index_add_(0, src, m_rev)

            h = self.layer_norms[layer](h + self.prelu(self.bias + agg))

        final = self.out_head(h).squeeze(-1)
        # Combine with initial activation (residual) so the GNN cannot delete
        # already-confident matches; this stabilizes early training.
        final = final + initial_activation
        return final, gate_log


__all__ = ["TorchGraph", "build_torch_graph", "GatedGNN"]
