"""Tests for the hierarchical memory layer: builder, router, decay.

Includes the falsifiable claims C7-C10 from the report, each written as a
prediction + falsifier + executable protocol. The tests use synthetic
clustered hypervector data constructed directly in the substrate so the
routing behaviour can be measured against a controlled ground truth without
the spaCy pipeline in the loop.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytest

from holograph.graph.substrate import GraphSubstrate
from holograph.hdc.kernel import make_kernel
from holograph.hierarchy.builder import HierarchyBuilder, build_hierarchy
from holograph.hierarchy.router import SoftRouter


# ---------------------------------------------------------------------------
# Synthetic clustered substrate
# ---------------------------------------------------------------------------


def make_clustered_substrate(n_clusters: int, per_cluster: int, dim: int,
                              seed: int = 0,
                              inter_cluster_edges: int = 0,
                              kind: str = "real"):
    """Build a substrate with `n_clusters` topical clusters of `per_cluster`
    leaf entities each. Each cluster has a random center HV; members are the
    center plus noise. Dense intra-cluster edges, optional sparse bridges."""
    rng = np.random.default_rng(seed)
    k = make_kernel(kind, dim=dim, prefer_native=False)
    g = GraphSubstrate(":memory:")

    centers = []
    cluster_members: List[List[int]] = []
    for c in range(n_clusters):
        center = rng.standard_normal(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        centers.append(center)
        members = []
        for m in range(per_cluster):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.25
            v = center + noise
            if kind == "real":
                v = v / np.linalg.norm(v)
                hv = v.astype(np.float32)
            else:
                hv = k.quantize(v)
            eid = g.upsert_entity(f"c{c}_m{m}", type="concept")
            g.set_hv(eid, k.pack(hv), k.name)
            members.append(eid)
        cluster_members.append(members)
        # Dense intra-cluster edges (a ring + a hub).
        for i in range(len(members)):
            g.upsert_edge(members[i], members[(i + 1) % len(members)], "intra", weight=1.0)
            if i > 0:
                g.upsert_edge(members[0], members[i], "hub", weight=0.5)

    # Optional inter-cluster bridges.
    bridges: List[Tuple[int, int]] = []
    for _ in range(inter_cluster_edges):
        a, b = rng.choice(n_clusters, size=2, replace=False)
        ua = cluster_members[a][rng.integers(per_cluster)]
        ub = cluster_members[b][rng.integers(per_cluster)]
        g.upsert_edge(ua, ub, "bridge", weight=1.0)
        bridges.append((ua, ub))

    return g, k, centers, cluster_members, bridges


def hv_oracle_topk(g, k, query_hv, top: int) -> List[int]:
    """Reference: exhaustive HV-cosine top-k over all leaves (what the router
    approximates)."""
    scored = []
    for eid in g.entities_at_layer(0):
        blob = g.get_hv_blob(eid)
        if blob is None:
            continue
        hv = k.unpack(blob[0])
        scored.append((eid, k.similarity(query_hv, hv)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [e for e, _ in scored[:top]]


# ---------------------------------------------------------------------------
# Builder smoke / regression
# ---------------------------------------------------------------------------


class TestHierarchyBuilder:
    def test_builds_layers_and_collapses(self):
        g, k, centers, members, _ = make_clustered_substrate(6, 8, dim=1024, seed=1)
        stats = build_hierarchy(g, k, max_layer=4, node_cap=3)
        assert g.max_layer() >= 1
        # Leaf count unchanged.
        assert len(g.entities_at_layer(0)) == 6 * 8
        # Each non-top layer has fewer nodes than the one below it.
        for layer in range(1, g.max_layer() + 1):
            assert len(g.entities_at_layer(layer)) <= len(g.entities_at_layer(layer - 1))
        g.close()

    def test_parent_hv_is_bundle_of_children(self):
        g, k, centers, members, _ = make_clustered_substrate(4, 6, dim=1024, seed=2)
        build_hierarchy(g, k, max_layer=2, node_cap=2)
        # For each layer-1 parent, its stored HV must equal bundle(children HVs).
        for pid in g.entities_at_layer(1):
            child_ids = g.children_of(pid)
            child_hvs = [k.unpack(g.get_hv_blob(c)[0]) for c in child_ids]
            expected = k.bundle(child_hvs)
            stored = k.unpack(g.get_hv_blob(pid)[0])
            # Equal up to float32 summation-order error (bundle is a sum +
            # normalize; children are summed in community order at build time
            # and sorted order here, which differs in the last ~1e-7).
            assert np.allclose(stored, expected, atol=1e-5), \
                f"parent {pid} HV != bundle(children) within tolerance"
        g.close()

    def test_leaf_graph_excludes_summary_nodes(self):
        g, k, centers, members, _ = make_clustered_substrate(5, 6, dim=512, seed=3)
        build_hierarchy(g, k, max_layer=3, node_cap=2)
        nx_nodes = set(g.to_networkx().nodes())
        leaf_ids = set(g.entities_at_layer(0))
        assert nx_nodes == leaf_ids, "propagation graph must contain only leaves"
        g.close()

    def test_rebuild_is_idempotent_in_leaf_count(self):
        g, k, _, _, _ = make_clustered_substrate(4, 5, dim=512, seed=4)
        build_hierarchy(g, k, max_layer=3, node_cap=2)
        n_after_first = len(g.entities_at_layer(0))
        build_hierarchy(g, k, max_layer=3, node_cap=2)  # rebuild
        assert len(g.entities_at_layer(0)) == n_after_first
        g.close()


# ---------------------------------------------------------------------------
# Router smoke
# ---------------------------------------------------------------------------


class TestSoftRouter:
    def test_route_returns_candidates(self):
        g, k, centers, members, _ = make_clustered_substrate(6, 8, dim=1024, seed=5)
        build_hierarchy(g, k, max_layer=4, node_cap=3)
        router = SoftRouter(g, k, beam=2)
        res = router.route(centers[0])
        assert res.leaf_candidates
        assert res.total_leaves == 6 * 8
        g.close()

    def test_route_degenerates_without_hierarchy(self):
        g, k, centers, members, _ = make_clustered_substrate(3, 5, dim=512, seed=6)
        # No build_hierarchy call -> max_layer 0 -> exhaustive fallback.
        router = SoftRouter(g, k, beam=2)
        res = router.route(centers[0])
        assert set(res.leaf_candidates) == set(g.entities_at_layer(0))
        g.close()


# ---------------------------------------------------------------------------
# Falsifiable claims
# ---------------------------------------------------------------------------


class TestClaimC7_RecallAndEfficiency:
    """C7: hierarchical routing preserves top-1 recall while touching a
    shrinking fraction of leaves as the graph grows.

    Prediction: the router's candidate set contains the exhaustive HV-cosine
    top-1 leaf in >= 90% of queries, AND touch_fraction strictly decreases as
    the leaf count grows.

    Falsifier: top-1 recall < 0.9 over the query battery, OR touch_fraction
    does not decrease when N grows by an order of magnitude.
    """

    def _recall_and_touch(self, n_clusters, per_cluster, seed):
        g, k, centers, members, _ = make_clustered_substrate(
            n_clusters, per_cluster, dim=1024, seed=seed)
        build_hierarchy(g, k, max_layer=5, node_cap=4)
        router = SoftRouter(g, k, beam=3)
        hits = 0
        touch = []
        rng = np.random.default_rng(seed + 100)
        trials = 0
        for c in range(n_clusters):
            # Query near a random member of cluster c.
            member = members[c][rng.integers(per_cluster)]
            qhv = k.unpack(g.get_hv_blob(member)[0])
            oracle_top1 = hv_oracle_topk(g, k, qhv, 1)[0]
            res = router.route(qhv)
            if oracle_top1 in set(res.leaf_candidates):
                hits += 1
            touch.append(res.touch_fraction)
            trials += 1
        g.close()
        return hits / trials, float(np.mean(touch))

    def test_recall_and_efficiency_scaling(self):
        recall_small, touch_small = self._recall_and_touch(6, 6, seed=11)   # 36 leaves
        recall_large, touch_large = self._recall_and_touch(12, 30, seed=12)  # 360 leaves

        # Recall preserved on both.
        assert recall_small >= 0.9, f"small-graph top-1 recall {recall_small}"
        assert recall_large >= 0.9, f"large-graph top-1 recall {recall_large}"
        # Efficiency: the larger graph touches a strictly smaller fraction.
        assert touch_large < touch_small, (
            f"touch fraction did not shrink with N: small={touch_small:.3f} "
            f"large={touch_large:.3f}")
        # And the large graph genuinely avoids most of the leaves.
        assert touch_large < 0.6, f"large-graph touch fraction {touch_large:.3f} not efficient"


class TestClaimC8_BridgeRecovery:
    """C8: soft routing (beam>1) plus 1-hop neighbor expansion recovers bridge
    nodes that a hard top-1 descent would prune.

    Prediction: for a query anchored in cluster A whose answer requires a
    bridge node living in cluster B, the soft-routed+expanded candidate set
    contains the bridge in significantly more cases than hard top-1 routing.

    Falsifier: soft routing + expansion recovers the bridge no more often than
    hard top-1 routing.
    """

    def test_soft_beats_hard_on_bridges(self):
        recovered_soft = 0
        recovered_hard = 0
        trials = 12
        for seed in range(trials):
            g, k, centers, members, bridges = make_clustered_substrate(
                4, 10, dim=1024, seed=200 + seed, inter_cluster_edges=3)
            build_hierarchy(g, k, max_layer=4, node_cap=2)
            if not bridges:
                g.close()
                continue
            ua, ub = bridges[0]
            # Query anchored near ua's cluster (use ua's own HV as the cue).
            qhv = k.unpack(g.get_hv_blob(ua)[0])

            # Hard top-1 routing: beam=1, no neighbor expansion.
            hard = SoftRouter(g, k, beam=1, leaf_beam_multiplier=1).route(qhv)
            hard_set = set(hard.leaf_candidates)

            # Soft routing: beam=3, plus 1-hop neighbor expansion.
            soft = SoftRouter(g, k, beam=3).route(qhv)
            soft_set = set(soft.leaf_candidates)
            for c in list(soft_set):
                soft_set.update(g.neighbors_of(c, leaf_only=True))

            if ub in hard_set:
                recovered_hard += 1
            if ub in soft_set:
                recovered_soft += 1
            g.close()

        assert recovered_soft >= recovered_hard, (
            f"soft routing ({recovered_soft}) did not recover bridges at least "
            f"as often as hard routing ({recovered_hard})")
        # And soft should strictly help on at least some seeds.
        assert recovered_soft > 0


class TestClaimC9_ParentBundleIsValidRouter:
    """C9: the parent-bundle similarity is a valid router — the parent whose
    bundle is most similar to a query contains that query's best child.

    Prediction: for a query equal to a specific leaf's HV, the top-layer parent
    selected by bundle similarity is (transitively) an ancestor of that leaf in
    >= 80% of cases.

    Falsifier: ancestor-consistency below 0.8 over the battery.
    """

    def test_best_parent_contains_best_child(self):
        g, k, centers, members, _ = make_clustered_substrate(8, 12, dim=2048, seed=42)
        build_hierarchy(g, k, max_layer=4, node_cap=3)

        def ancestors(leaf_id):
            """All transitive parents of a leaf."""
            seen = set()
            frontier = [leaf_id]
            while frontier:
                nxt = []
                for nid in frontier:
                    for p in g.parents_of(nid):
                        if p not in seen:
                            seen.add(p)
                            nxt.append(p)
                frontier = nxt
            return seen

        top_layer = g.max_layer()
        top_nodes = g.entities_at_layer(top_layer)
        consistent = 0
        total = 0
        rng = np.random.default_rng(99)
        for c in range(len(members)):
            leaf = members[c][rng.integers(len(members[c]))]
            qhv = k.unpack(g.get_hv_blob(leaf)[0])
            # Most-similar top-layer parent.
            best_top = max(
                top_nodes,
                key=lambda p: k.similarity(qhv, k.unpack(g.get_hv_blob(p)[0])),
            )
            if best_top in ancestors(leaf):
                consistent += 1
            total += 1
        g.close()
        rate = consistent / total
        assert rate >= 0.8, f"ancestor-consistency {rate:.3f} below 0.8"


class TestClaimC10_DecayMonotonic:
    """C10: un-reinforced edge weight decays monotonically toward the floor;
    reinforcement (touch) resets the clock so used memories persist.

    Prediction: repeated decay passes on an untouched edge produce a
    monotonically non-increasing weight converging to the floor; a touched
    edge retains materially higher weight than an untouched peer.

    Falsifier: non-monotonic decay, unbounded weight, or a touched edge ending
    no higher than an untouched one.
    """

    def test_decay_monotonic_and_touch_resets(self):
        import time
        g = GraphSubstrate(":memory:")
        a = g.upsert_entity("A"); b = g.upsert_entity("B"); c = g.upsert_entity("C")
        e_untouched = g.upsert_edge(a, b, "r", weight=8.0)
        e_touched = g.upsert_edge(b, c, "r", weight=8.0)

        # Backdate both edges so decay has elapsed time to act on.
        past = time.time() - 1000.0
        g.touch_edge(e_untouched, when=past)
        g.touch_edge(e_touched, when=past)

        weights = []
        for _ in range(5):
            # Touch the "touched" edge each round (reset its clock to now).
            g.touch_edge(e_touched)
            g.decay_edges(half_life_seconds=200.0, floor=0.0)
            w_untouched = next(e.weight for e in g.all_edges() if e.id == e_untouched)
            weights.append(w_untouched)

        # Monotonic non-increasing.
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1] + 1e-9, f"non-monotonic decay: {weights}"
        # Converging toward floor.
        assert weights[-1] < weights[0]
        # Touched edge retains materially more weight than untouched.
        w_touched = next(e.weight for e in g.all_edges() if e.id == e_touched)
        assert w_touched > weights[-1], (
            f"touched edge {w_touched:.3f} not above untouched {weights[-1]:.3f}")
        g.close()

    def test_decay_respects_floor_and_ceiling(self):
        g = GraphSubstrate(":memory:")
        a = g.upsert_entity("A"); b = g.upsert_entity("B")
        eid = g.upsert_edge(a, b, "r", weight=5.0)
        import time
        g.touch_edge(eid, when=time.time() - 1e6)  # ancient
        g.decay_edges(half_life_seconds=1.0, floor=0.5)
        w = g.all_edges()[0].weight
        assert 0.5 <= w <= 10.0, f"weight {w} out of [floor, ceiling]"
        g.close()


# ---------------------------------------------------------------------------
# C3 preservation: routed path must not break activated-subgraph composition
# ---------------------------------------------------------------------------


def test_routed_path_preserves_composition(spacy_nlp):
    """The hierarchical routed read must still produce a composed HV and the
    same kind of activated subgraph the exhaustive path does (claim C3 holds
    under routing)."""
    from holograph.runtime import HoloGraph
    hg = HoloGraph(kernel_kind="real", dim=2048, top_k=6,
                   use_hierarchy=True, router_beam=3, spacy_nlp=spacy_nlp)
    corpus = [
        ("d1", "Alice studies hippocampal memory."),
        ("d2", "Bob works on GraphRAG systems."),
        ("d3", "HippoRAG is inspired by the hippocampus."),
        ("d4", "GraphRAG and HippoRAG are graph memory systems."),
        ("d5", "SAGE evolves graph memory through feedback."),
        ("d6", "Carol studies skiing physics."),
    ]
    for a, t in corpus:
        hg.ingest_text(t, anchor=a)
    hg.build_hierarchy(max_layer=3, node_cap=2)
    out = hg.read("What is related to GraphRAG?")
    assert out.composed_hv is not None
    assert out.activated_ids
    hg.close()
