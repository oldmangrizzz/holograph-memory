"""Tests for the gated GNN and HDC composer in isolation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from holograph.graph.substrate import GraphSubstrate
from holograph.hdc.kernel import make_kernel
from holograph.reader.composer import HDCComposer
from holograph.reader.gnn import GatedGNN, build_torch_graph


@pytest.fixture
def small_graph():
    g = GraphSubstrate(":memory:")
    a = g.upsert_entity("A")
    b = g.upsert_entity("B")
    c = g.upsert_entity("C")
    g.upsert_edge(a, b, "r")
    g.upsert_edge(b, c, "r")
    g.upsert_edge(a, c, "r2", weight=0.5)
    yield g
    g.close()


def test_build_torch_graph(small_graph):
    nx = small_graph.to_networkx()
    tg = build_torch_graph(nx)
    assert tg.node_features.shape == (3, 4)
    assert tg.edge_index.shape[0] == 2
    assert tg.edge_index.shape[1] == 3  # three directed edges, no parallels
    assert tg.graph_summary.shape == (9,)


def test_gated_gnn_forward_shape(small_graph):
    nx = small_graph.to_networkx()
    tg = build_torch_graph(nx)
    gnn = GatedGNN(hidden_dim=16, n_layers=2)
    init = torch.zeros(3, dtype=torch.float32)
    init[0] = 1.0
    final, gates = gnn(tg, init)
    assert final.shape == (3,)
    assert len(gates) == 2
    assert gates[0].shape[0] == 3  # edge count


def test_gated_gnn_propagates_activation(small_graph):
    nx = small_graph.to_networkx()
    tg = build_torch_graph(nx)
    gnn = GatedGNN(hidden_dim=16, n_layers=2)
    init = torch.zeros(3, dtype=torch.float32)
    init[0] = 1.0  # only node 0 is initially active
    with torch.no_grad():
        final, _ = gnn(tg, init)
    # All neighbors should receive some activation propagation
    # i.e. final activations should differ from initial.
    assert not torch.allclose(final, init)


def test_composer_attribution(small_graph):
    k = make_kernel("real", dim=512)
    # Assign random hypervectors to entities.
    rng = np.random.default_rng(0)
    for eid in (1, 2, 3):
        hv = rng.standard_normal(512).astype(np.float32)
        hv = hv / np.linalg.norm(hv)
        small_graph.set_hv(eid, k.pack(hv), "real")
    composer = HDCComposer(k)
    activations = {1: 1.0, 2: 1.0, 3: 1.0}
    composed, paths = composer.compose(small_graph, [1, 2, 3], activations)
    assert composed.shape == (512,)
    assert len(paths) == 3
    report = composer.attribute(composed, paths)
    # Every entity that participates in a path should have an attribution entry.
    assert set(report.per_entity.keys()) == {1, 2, 3}
    # Every path index should be in per_edge.
    assert set(report.per_edge.keys()) == set(range(3))
    # per_path_top is sorted descending.
    sims = [s for _, s in report.per_path_top]
    assert sims == sorted(sims, reverse=True)


def test_composer_handles_no_edges(small_graph):
    k = make_kernel("real", dim=256)
    rng = np.random.default_rng(0)
    for eid in (1, 2, 3):
        hv = rng.standard_normal(256).astype(np.float32)
        hv = hv / np.linalg.norm(hv)
        small_graph.set_hv(eid, k.pack(hv), "real")
    composer = HDCComposer(k)
    # Only one entity activated -> no edges connect activated set.
    composed, paths = composer.compose(small_graph, [1], {1: 1.0})
    assert composed.shape == (256,)
    assert paths == []
