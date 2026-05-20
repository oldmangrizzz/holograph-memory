"""Tests for the graph substrate."""

from __future__ import annotations

import numpy as np
import pytest

from holograph.graph.substrate import GraphSubstrate


@pytest.fixture
def substrate():
    g = GraphSubstrate(":memory:")
    yield g
    g.close()


def test_entity_upsert_and_lookup(substrate):
    a = substrate.upsert_entity("Alice", type="Person", aliases=["Ali", "A."])
    b = substrate.upsert_entity("Alice", type="Person")  # idempotent
    assert a == b
    assert substrate.lookup_by_surface("Alice") == a
    assert substrate.lookup_by_surface("Ali") == a
    assert substrate.lookup_by_surface("ali") == a  # case insensitive
    assert substrate.lookup_by_surface("Bob") is None


def test_alias_resolution(substrate):
    eid = substrate.upsert_entity("Cornu Ammonis", type="Anatomy",
                                    aliases=["hippocampus"])
    assert substrate.lookup_by_surface("hippocampus") == eid
    # Add a new alias on update.
    substrate.upsert_entity("Cornu Ammonis", type="Anatomy", aliases=["CA region"])
    assert substrate.lookup_by_surface("CA region") == eid
    e = substrate.get_entity(eid)
    assert "CA region" in e.aliases


def test_edge_upsert_and_reinforcement(substrate):
    a = substrate.upsert_entity("Alice")
    b = substrate.upsert_entity("Bob")
    eid1 = substrate.upsert_edge(a, b, "knows", weight=1.0)
    eid2 = substrate.upsert_edge(a, b, "knows", weight=1.0)  # duplicate -> reinforce
    assert eid1 == eid2
    edges = substrate.all_edges()
    assert len(edges) == 1
    assert edges[0].weight == pytest.approx(2.0)


def test_edge_weight_clamps(substrate):
    a = substrate.upsert_entity("A"); b = substrate.upsert_entity("B")
    eid = substrate.upsert_edge(a, b, "rel", weight=1.0)
    # Push the weight to the upper clamp.
    for _ in range(100):
        substrate.update_edge_weight(eid, 1.0)
    edges = substrate.all_edges()
    assert edges[0].weight <= 10.0
    # And the lower clamp.
    for _ in range(100):
        substrate.update_edge_weight(eid, -1.0)
    edges = substrate.all_edges()
    assert edges[0].weight >= 0.0


def test_hv_storage(substrate):
    eid = substrate.upsert_entity("X")
    blob = b"\x01\x02\x03\x04"
    substrate.set_hv(eid, blob, "real")
    out = substrate.get_hv_blob(eid)
    assert out is not None
    data, kernel_name = out
    assert data == blob
    assert kernel_name == "real"


def test_networkx_projection_caching(substrate):
    a = substrate.upsert_entity("A"); b = substrate.upsert_entity("B")
    substrate.upsert_edge(a, b, "r")
    g1 = substrate.to_networkx()
    g2 = substrate.to_networkx()
    assert g1 is g2  # cache hit
    # Mutating bumps the version and forces recomputation.
    c = substrate.upsert_entity("C")
    g3 = substrate.to_networkx()
    assert g3 is not g1


def test_structural_roles_and_communities(substrate):
    a = substrate.upsert_entity("A"); b = substrate.upsert_entity("B"); c = substrate.upsert_entity("C")
    d = substrate.upsert_entity("D"); e = substrate.upsert_entity("E")
    substrate.upsert_edge(a, b, "r1")
    substrate.upsert_edge(b, c, "r1")
    substrate.upsert_edge(d, e, "r2")
    substrate.upsert_edge(b, d, "r3")  # bridge between two triangles
    roles = substrate.structural_roles()
    # B should have the highest betweenness because it bridges.
    assert roles[b]["betweenness"] >= max(roles[i]["betweenness"] for i in (a, c, d, e))
    comm = substrate.communities()
    # All five nodes should be assigned to some community.
    assert set(comm.keys()) == {a, b, c, d, e}


def test_documents(substrate):
    a = substrate.upsert_entity("A")
    substrate.upsert_edge(a, a, "self", source="doc1")
    substrate.upsert_document("doc1", "text body")
    docs = substrate.documents_for_entities([a])
    assert ("doc1", "text body") in docs
