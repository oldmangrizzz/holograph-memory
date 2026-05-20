"""Real-scale stress test for the hierarchical memory + router.

Goal: prove the context-window kill holds *sub-linearly* as memory grows —
i.e. retrieval touches a shrinking fraction of memory as N rises, while still
recovering the right memory (recall preserved), at bounded query latency.

This is a structural benchmark: synthetic clustered hypervector data inserted
directly (bulk, benchmark-only PRAGMAs) so we can reach large N without the
spaCy pipeline. It measures the part that matters for the context-window
claim: routing cost and recall at scale.

Usage:
    python bench/stress_test.py <N> [--clusters K] [--dim D] [--kernel real|ternary] [--queries Q]

Reports, for the given N:
    - hierarchy build time + layer sizes
    - mean routing recall@candidates  (does exhaustive top-1 land in the routed set?)
    - mean / p95 query latency (ms)
    - mean touch fraction (leaves scored / total leaves)
    - leaf HV storage footprint
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograph.graph.substrate import GraphSubstrate
from holograph.hdc.kernel import make_kernel
from holograph.hierarchy.builder import build_hierarchy
from holograph.hierarchy.router import SoftRouter


def build_synthetic(n: int, clusters: int, dim: int, kernel_kind: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = make_kernel(kernel_kind, dim=dim, prefer_native=False)
    g = GraphSubstrate(":memory:")
    # Benchmark-only: make commits cheap so insertion isn't the bottleneck.
    g.conn.execute("PRAGMA synchronous=OFF")
    g.conn.execute("PRAGMA journal_mode=MEMORY")

    per = max(1, n // clusters)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    now = time.time()
    members = [[] for _ in range(clusters)]
    cur = g.conn.cursor()
    # Bulk entity insert.
    eid = 0
    rows = []
    blobs = []
    for c in range(clusters):
        for m in range(per):
            eid += 1
            name = f"c{c}_m{m}"
            v = centers[c] + rng.standard_normal(dim).astype(np.float32) * 0.25
            if kernel_kind == "real":
                v = v / np.linalg.norm(v)
                hv = v.astype(np.float32)
            else:
                hv = k.quantize(v)
            rows.append((name, "concept", "", k.pack(hv), k.name, now, 0))
            members[c].append(eid)
    cur.executemany(
        "INSERT INTO entities(canonical,type,description,hv_blob,hv_kernel,created_at,layer) "
        "VALUES(?,?,?,?,?,?,?)", rows,
    )
    # Bulk edges: intra-cluster ring + a hub; sparse inter-cluster bridges.
    edges = []
    for c in range(clusters):
        ms = members[c]
        for i in range(len(ms)):
            edges.append((ms[i], ms[(i + 1) % len(ms)], "intra", 1.0, "", now, None))
            if i % 7 == 0 and i > 0:
                edges.append((ms[0], ms[i], "hub", 0.5, "", now, None))
    n_bridges = max(1, clusters)
    for _ in range(n_bridges):
        a, b = rng.choice(clusters, size=2, replace=False)
        ua = members[a][rng.integers(len(members[a]))]
        ub = members[b][rng.integers(len(members[b]))]
        edges.append((ua, ub, "bridge", 1.0, "", now, None))
    cur.executemany(
        "INSERT OR IGNORE INTO edges(head_id,tail_id,relation,weight,source,created_at,last_used) "
        "VALUES(?,?,?,?,?,?,?)", edges,
    )
    g.conn.commit()
    g._bump()
    return g, k, centers, members


def hv_oracle_top1(g, k, qhv):
    best, best_sim = None, -2.0
    for eid in g.entities_at_layer(0):
        blob = g.get_hv_blob(eid)
        if blob is None:
            continue
        sim = k.similarity(qhv, k.unpack(blob[0]))
        if sim > best_sim:
            best_sim, best = sim, eid
    return best


def run(n: int, clusters: int, dim: int, kernel_kind: str, queries: int, beam: int = 3):
    print(f"\n=== N={n}  clusters={clusters}  dim={dim}  kernel={kernel_kind}  beam={beam} ===")
    t0 = time.time()
    g, k, centers, members = build_synthetic(n, clusters, dim, kernel_kind)
    leaves = len(g.entities_at_layer(0))
    print(f"  built substrate: {leaves} leaves, {g.n_edges()} edges  ({time.time()-t0:.1f}s)")

    t1 = time.time()
    stats = build_hierarchy(g, k, max_layer=6, node_cap=max(8, clusters // 4))
    build_s = time.time() - t1
    print(f"  hierarchy: nodes/layer={stats.nodes_per_layer}  build={build_s:.1f}s")

    router = SoftRouter(g, k, beam=beam)
    rng = np.random.default_rng(1234)
    hits, touch, lat = 0, [], []
    for _ in range(queries):
        c = int(rng.integers(clusters))
        member = members[c][int(rng.integers(len(members[c])))]
        qhv = k.unpack(g.get_hv_blob(member)[0])
        oracle = hv_oracle_top1(g, k, qhv)
        t = time.time()
        res = router.route(qhv)
        lat.append((time.time() - t) * 1000.0)
        if oracle in set(res.leaf_candidates):
            hits += 1
        touch.append(res.touch_fraction)

    recall = hits / queries
    footprint = sum(len(g.get_hv_blob(e)[0]) for e in g.entities_at_layer(0)[:1]) * leaves
    print(f"  recall@candidates : {recall:.3f}  ({hits}/{queries})")
    print(f"  touch fraction    : mean={np.mean(touch):.4f}  (scored {np.mean(touch)*leaves:.0f}/{leaves} leaves/query)")
    print(f"  query latency ms  : mean={np.mean(lat):.2f}  p95={np.percentile(lat,95):.2f}")
    print(f"  leaf HV footprint : {footprint/1e6:.2f} MB ({footprint//leaves} B/HV)")
    g.close()
    return {"n": leaves, "recall": recall, "touch": float(np.mean(touch)),
            "lat_ms": float(np.mean(lat)), "build_s": build_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--clusters", type=int, default=None)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--kernel", default="real")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--beam", type=int, default=3)
    args = ap.parse_args()
    clusters = args.clusters or max(8, int(np.sqrt(args.n)))
    run(args.n, clusters, args.dim, args.kernel, args.queries, beam=args.beam)


if __name__ == "__main__":
    main()
