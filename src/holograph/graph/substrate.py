"""SQLite-backed graph substrate with NetworkX projection.

The substrate is the living memory: entities and typed edges with
source-anchored provenance, alias resolution, computed structural roles
(hub / bridge / community membership), and per-entity hypervector storage.

Schema
------
entities
    id              INTEGER  PRIMARY KEY
    canonical       TEXT     UNIQUE NOT NULL   canonical surface form
    type            TEXT     NOT NULL          coarse entity type
    description     TEXT     default ""        free-text description
    hv_blob         BLOB                       packed kernel-specific HV
    hv_kernel       TEXT                       kernel name used to pack hv_blob
    created_at      REAL                       unix ts

aliases
    id              INTEGER  PRIMARY KEY
    entity_id       INTEGER  NOT NULL  REFERENCES entities(id)
    surface         TEXT     NOT NULL

edges
    id              INTEGER  PRIMARY KEY
    head_id         INTEGER  NOT NULL  REFERENCES entities(id)
    tail_id         INTEGER  NOT NULL  REFERENCES entities(id)
    relation        TEXT     NOT NULL
    weight          REAL     NOT NULL  DEFAULT 1.0   reward-shaped
    source          TEXT                              source doc anchor
    created_at      REAL                              unix ts
    UNIQUE(head_id, tail_id, relation)

documents
    id              INTEGER  PRIMARY KEY
    anchor          TEXT     UNIQUE NOT NULL
    text            TEXT     NOT NULL
    created_at      REAL

The NetworkX projection is computed lazily and cached behind a version counter.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class Entity:
    id: int
    canonical: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    hv: Optional[np.ndarray] = None
    hv_kernel: Optional[str] = None
    layer: int = 0  # 0 = leaf/episode; >0 = abstraction-summary node


@dataclass
class Edge:
    id: int
    head_id: int
    tail_id: int
    relation: str
    weight: float
    source: str = ""
    source_type: str = "inference"   # operator | document | inference | model
    confidence: float = 0.5
    quarantined: bool = False
    revised_at: Optional[float] = None
    provenance_class: str = "real"   # real | origin | quarantined-noise
    charge: float = 0.0              # emotional charge [0,1]; orthogonal to confidence.
                                     # How activating a memory is, NOT how true it is.
                                     # Extinguished downward by safe recall; never alters truth.


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical   TEXT    UNIQUE NOT NULL,
    type        TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    hv_blob     BLOB,
    hv_kernel   TEXT,
    created_at  REAL    NOT NULL,
    layer       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS aliases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER NOT NULL,
    surface     TEXT    NOT NULL,
    UNIQUE(entity_id, surface),
    FOREIGN KEY(entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_aliases_surface ON aliases(surface);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    head_id     INTEGER NOT NULL,
    tail_id     INTEGER NOT NULL,
    relation    TEXT    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 1.0,
    source      TEXT    DEFAULT '',
    created_at  REAL    NOT NULL,
    last_used   REAL,
    source_type TEXT    NOT NULL DEFAULT 'inference',
    confidence  REAL    NOT NULL DEFAULT 0.5,
    quarantined INTEGER NOT NULL DEFAULT 0,
    revised_at  REAL,
    provenance_class TEXT NOT NULL DEFAULT 'real',
    charge      REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(head_id, tail_id, relation),
    FOREIGN KEY(head_id) REFERENCES entities(id),
    FOREIGN KEY(tail_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_edges_head ON edges(head_id);
CREATE INDEX IF NOT EXISTS idx_edges_tail ON edges(tail_id);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor      TEXT    UNIQUE NOT NULL,
    text        TEXT    NOT NULL,
    created_at  REAL    NOT NULL
);

-- Abstraction hierarchy: parent (layer L+1) -> child (layer L). Kept distinct
-- from semantic edges so the abstraction tree never pollutes the relational
-- graph the reader propagates over.
CREATE TABLE IF NOT EXISTS hierarchy (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id   INTEGER NOT NULL,
    child_id    INTEGER NOT NULL,
    layer       INTEGER NOT NULL,
    created_at  REAL    NOT NULL,
    UNIQUE(parent_id, child_id),
    FOREIGN KEY(parent_id) REFERENCES entities(id),
    FOREIGN KEY(child_id)  REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_hier_parent ON hierarchy(parent_id);
CREATE INDEX IF NOT EXISTS idx_hier_child  ON hierarchy(child_id);
CREATE INDEX IF NOT EXISTS idx_entities_layer ON entities(layer);
"""


class GraphSubstrate:
    """Persistent + in-memory graph with HV storage and structural projections."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets a server (e.g. the JARVIS bridge) touch the graph from
        # request worker threads; callers must serialize writes (the bridge holds a lock).
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()
        self._version: int = 0
        self._nx_cache: Optional[Tuple[int, nx.MultiDiGraph]] = None

    def _migrate(self) -> None:
        """Add columns introduced after the original schema, idempotently.

        Keeps databases created by older versions usable: SQLite's
        CREATE TABLE IF NOT EXISTS won't add new columns to an existing table,
        so we ALTER on demand.
        """
        ent_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(entities)")}
        if "layer" not in ent_cols:
            self.conn.execute("ALTER TABLE entities ADD COLUMN layer INTEGER NOT NULL DEFAULT 0")
        edge_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(edges)")}
        if "last_used" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN last_used REAL")
        if "source_type" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN source_type TEXT NOT NULL DEFAULT 'inference'")
        if "confidence" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5")
        if "quarantined" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN quarantined INTEGER NOT NULL DEFAULT 0")
        if "revised_at" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN revised_at REAL")
        if "provenance_class" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN provenance_class TEXT NOT NULL DEFAULT 'real'")
        if "charge" not in edge_cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN charge REAL NOT NULL DEFAULT 0.0")

    # ---- entity / alias CRUD -----------------------------------------

    def upsert_entity(self,
                      canonical: str,
                      type: str = "concept",
                      description: str = "",
                      aliases: Sequence[str] = ()) -> int:
        """Insert or update an entity by canonical name; return its id."""
        cur = self.conn.execute("SELECT id FROM entities WHERE canonical=?", (canonical,))
        row = cur.fetchone()
        now = time.time()
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO entities(canonical, type, description, created_at) VALUES(?,?,?,?)",
                (canonical, type, description, now),
            )
            eid = int(cur.lastrowid)
        else:
            eid = int(row["id"])
            self.conn.execute(
                "UPDATE entities SET type=?, description=COALESCE(NULLIF(?, ''), description) WHERE id=?",
                (type, description, eid),
            )
        for surf in {*aliases, canonical}:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO aliases(entity_id, surface) VALUES(?, ?)",
                    (eid, surf),
                )
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        self._bump()
        return eid

    def get_entity(self, eid: int) -> Optional[Entity]:
        row = self.conn.execute("SELECT * FROM entities WHERE id=?", (eid,)).fetchone()
        if row is None:
            return None
        aliases = [r["surface"] for r in self.conn.execute(
            "SELECT surface FROM aliases WHERE entity_id=? ORDER BY id", (eid,))]
        return Entity(
            id=int(row["id"]),
            canonical=row["canonical"],
            type=row["type"],
            description=row["description"] or "",
            aliases=aliases,
            hv=None,
            hv_kernel=row["hv_kernel"],
            layer=int(row["layer"]) if "layer" in row.keys() and row["layer"] is not None else 0,
        )

    def lookup_by_surface(self, surface: str) -> Optional[int]:
        """Map a surface form to an entity id via canonical or aliases.

        Case-insensitive. Returns the first match.
        """
        row = self.conn.execute(
            "SELECT id FROM entities WHERE LOWER(canonical)=LOWER(?) LIMIT 1",
            (surface,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        row = self.conn.execute(
            "SELECT entity_id FROM aliases WHERE LOWER(surface)=LOWER(?) LIMIT 1",
            (surface,),
        ).fetchone()
        return int(row["entity_id"]) if row else None

    def all_entities(self) -> List[Entity]:
        rows = self.conn.execute(
            "SELECT id FROM entities ORDER BY id"
        ).fetchall()
        return [e for e in (self.get_entity(int(r["id"])) for r in rows) if e is not None]

    # ---- hypervector storage -----------------------------------------

    def set_hv(self, eid: int, blob: bytes, kernel_name: str) -> None:
        self.conn.execute(
            "UPDATE entities SET hv_blob=?, hv_kernel=? WHERE id=?",
            (blob, kernel_name, eid),
        )
        self.conn.commit()
        self._bump()

    def get_hv_blob(self, eid: int) -> Optional[Tuple[bytes, str]]:
        row = self.conn.execute(
            "SELECT hv_blob, hv_kernel FROM entities WHERE id=?", (eid,)
        ).fetchone()
        if row is None or row["hv_blob"] is None:
            return None
        return bytes(row["hv_blob"]), row["hv_kernel"]

    # ---- edges --------------------------------------------------------

    def upsert_edge(self,
                    head_id: int,
                    tail_id: int,
                    relation: str,
                    weight: float = 1.0,
                    source: str = "") -> int:
        now = time.time()
        cur = self.conn.execute(
            "SELECT id, weight FROM edges WHERE head_id=? AND tail_id=? AND relation=?",
            (head_id, tail_id, relation),
        )
        row = cur.fetchone()
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO edges(head_id, tail_id, relation, weight, source, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (head_id, tail_id, relation, weight, source, now),
            )
            eid = int(cur.lastrowid)
        else:
            eid = int(row["id"])
            # Sum on duplicate: reinforce existing edges (capped at 10x).
            new_weight = min(float(row["weight"]) + weight, 10.0)
            self.conn.execute(
                "UPDATE edges SET weight=? WHERE id=?",
                (new_weight, eid),
            )
        self.conn.commit()
        self._bump()
        return eid

    def set_belief_meta(self, edge_id: int, *, source_type: Optional[str] = None,
                        confidence: Optional[float] = None,
                        quarantined: Optional[bool] = None,
                        revised_at: Optional[float] = None,
                        provenance_class: Optional[str] = None,
                        charge: Optional[float] = None) -> None:
        """Update belief metadata on an edge without disturbing the graph."""
        sets, vals = [], []
        if source_type is not None:
            sets.append("source_type=?"); vals.append(source_type)
        if confidence is not None:
            sets.append("confidence=?"); vals.append(float(confidence))
        if quarantined is not None:
            sets.append("quarantined=?"); vals.append(1 if quarantined else 0)
        if revised_at is not None:
            sets.append("revised_at=?"); vals.append(float(revised_at))
        if provenance_class is not None:
            sets.append("provenance_class=?"); vals.append(str(provenance_class))
        if charge is not None:
            sets.append("charge=?"); vals.append(max(0.0, min(1.0, float(charge))))
        if not sets:
            return
        vals.append(edge_id)
        self.conn.execute(f"UPDATE edges SET {', '.join(sets)} WHERE id=?", vals)
        self.conn.commit()
        self._bump()

    def beliefs_for(self, head_id: int, relation: str,
                    include_quarantined: bool = True,
                    provenance_class: Optional[str] = None) -> List[Edge]:
        """Return all edges (beliefs) for a given subject + relation.

        `provenance_class`, if given, filters to that class (e.g. 'origin' to
        recall fictional genesis, 'real' for Earth-1218 fact)."""
        q = "SELECT * FROM edges WHERE head_id=? AND relation=?"
        params: list = [head_id, relation]
        if not include_quarantined:
            q += " AND quarantined=0"
        if provenance_class is not None:
            q += " AND provenance_class=?"; params.append(str(provenance_class))
        rows = self.conn.execute(q, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def update_edge_weight(self, edge_id: int, delta: float) -> float:
        cur = self.conn.execute("SELECT weight FROM edges WHERE id=?", (edge_id,))
        row = cur.fetchone()
        if row is None:
            return 0.0
        new_w = max(min(float(row["weight"]) + delta, 10.0), 0.0)
        self.conn.execute("UPDATE edges SET weight=? WHERE id=?", (new_w, edge_id))
        self.conn.commit()
        self._bump()
        return new_w

    def edges_of(self, eid: int) -> List[Edge]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE head_id=? OR tail_id=?", (eid, eid)
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def all_edges(self) -> List[Edge]:
        rows = self.conn.execute("SELECT * FROM edges ORDER BY id").fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _row_to_edge(r: sqlite3.Row) -> Edge:
        keys = r.keys()
        return Edge(
            id=int(r["id"]),
            head_id=int(r["head_id"]),
            tail_id=int(r["tail_id"]),
            relation=r["relation"],
            weight=float(r["weight"]),
            source=r["source"] or "",
            source_type=(r["source_type"] if "source_type" in keys and r["source_type"] else "inference"),
            confidence=(float(r["confidence"]) if "confidence" in keys and r["confidence"] is not None else 0.5),
            quarantined=(bool(r["quarantined"]) if "quarantined" in keys and r["quarantined"] is not None else False),
            revised_at=(float(r["revised_at"]) if "revised_at" in keys and r["revised_at"] is not None else None),
            provenance_class=(r["provenance_class"] if "provenance_class" in keys and r["provenance_class"] else "real"),
            charge=(float(r["charge"]) if "charge" in keys and r["charge"] is not None else 0.0),
        )

    def neighbors_of(self, eid: int, leaf_only: bool = True) -> List[int]:
        """Return ids of entities sharing a semantic edge with `eid`.

        Used by the reader to expand a routed candidate set by 1 hop so that
        cross-branch bridge nodes survive hierarchical pruning.
        """
        rows = self.conn.execute(
            "SELECT head_id, tail_id FROM edges WHERE head_id=? OR tail_id=?", (eid, eid)
        ).fetchall()
        out: Set[int] = set()
        for r in rows:
            other = int(r["tail_id"]) if int(r["head_id"]) == eid else int(r["head_id"])
            out.add(other)
        if leaf_only and out:
            ph = ",".join("?" * len(out))
            keep = self.conn.execute(
                f"SELECT id FROM entities WHERE id IN ({ph}) AND layer=0", list(out)
            ).fetchall()
            return [int(r["id"]) for r in keep]
        return list(out)

    # ---- decay support ------------------------------------------------

    def touch_edge(self, edge_id: int, when: Optional[float] = None) -> None:
        """Mark an edge as freshly used (resets its decay clock)."""
        ts = time.time() if when is None else when
        self.conn.execute("UPDATE edges SET last_used=? WHERE id=?", (ts, edge_id))
        self.conn.commit()
        self._bump()

    def decay_edges(self, half_life_seconds: float, floor: float = 0.0,
                    now: Optional[float] = None) -> int:
        """Apply Ebbinghaus-style multiplicative decay to all edge weights.

        weight *= 0.5 ** (elapsed / half_life), where elapsed is measured from
        last_used (or created_at if the edge was never explicitly touched).
        Weights are clamped to [floor, 10]. Returns the number of edges updated.
        """
        if half_life_seconds <= 0:
            return 0
        t = time.time() if now is None else now
        rows = self.conn.execute(
            "SELECT id, weight, created_at, last_used FROM edges"
        ).fetchall()
        n = 0
        for r in rows:
            ref = r["last_used"] if r["last_used"] is not None else r["created_at"]
            elapsed = max(0.0, t - float(ref))
            factor = 0.5 ** (elapsed / half_life_seconds)
            new_w = max(float(floor), min(float(r["weight"]) * factor, 10.0))
            self.conn.execute("UPDATE edges SET weight=? WHERE id=?", (new_w, int(r["id"])))
            n += 1
        self.conn.commit()
        self._bump()
        return n

    # ---- abstraction hierarchy ----------------------------------------

    def set_layer(self, eid: int, layer: int) -> None:
        self.conn.execute("UPDATE entities SET layer=? WHERE id=?", (int(layer), eid))
        self.conn.commit()
        self._bump()

    def add_hierarchy_edge(self, parent_id: int, child_id: int, layer: int) -> int:
        """Record that `parent_id` (at layer+1) summarizes `child_id` (at layer)."""
        now = time.time()
        cur = self.conn.execute(
            "SELECT id FROM hierarchy WHERE parent_id=? AND child_id=?",
            (parent_id, child_id),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO hierarchy(parent_id, child_id, layer, created_at) VALUES(?,?,?,?)",
            (parent_id, child_id, int(layer), now),
        )
        self.conn.commit()
        self._bump()
        return int(cur.lastrowid)

    def children_of(self, parent_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT child_id FROM hierarchy WHERE parent_id=? ORDER BY child_id", (parent_id,)
        ).fetchall()
        return [int(r["child_id"]) for r in rows]

    def parents_of(self, child_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT parent_id FROM hierarchy WHERE child_id=? ORDER BY parent_id", (child_id,)
        ).fetchall()
        return [int(r["parent_id"]) for r in rows]

    def entities_at_layer(self, layer: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT id FROM entities WHERE layer=? ORDER BY id", (int(layer),)
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def max_layer(self) -> int:
        row = self.conn.execute("SELECT MAX(layer) AS m FROM entities").fetchone()
        return int(row["m"]) if row and row["m"] is not None else 0

    def clear_hierarchy(self) -> None:
        """Drop all abstraction nodes (layer>0) and hierarchy edges.

        Leaf entities (layer 0) and the semantic graph are untouched.
        """
        self.conn.execute("DELETE FROM hierarchy")
        self.conn.execute("DELETE FROM aliases WHERE entity_id IN "
                          "(SELECT id FROM entities WHERE layer>0)")
        self.conn.execute("DELETE FROM entities WHERE layer>0")
        self.conn.commit()
        self._bump()

    # ---- documents ----------------------------------------------------

    def upsert_document(self, anchor: str, text: str) -> int:
        now = time.time()
        cur = self.conn.execute("SELECT id FROM documents WHERE anchor=?", (anchor,))
        row = cur.fetchone()
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO documents(anchor, text, created_at) VALUES(?, ?, ?)",
                (anchor, text, now),
            )
            self.conn.commit()
            return int(cur.lastrowid)
        return int(row["id"])

    def documents_for_entities(self, eids: Sequence[int]) -> List[Tuple[str, str]]:
        """Return [(anchor, text), ...] for documents linked via edge.source."""
        if not eids:
            return []
        ph = ",".join("?" * len(eids))
        rows = self.conn.execute(
            f"SELECT DISTINCT source FROM edges WHERE (head_id IN ({ph}) OR tail_id IN ({ph})) AND source != ''",
            list(eids) + list(eids),
        ).fetchall()
        anchors = [r["source"] for r in rows]
        if not anchors:
            return []
        ph2 = ",".join("?" * len(anchors))
        drows = self.conn.execute(
            f"SELECT anchor, text FROM documents WHERE anchor IN ({ph2})",
            anchors,
        ).fetchall()
        return [(r["anchor"], r["text"]) for r in drows]

    # ---- NetworkX projection / structural roles ----------------------

    def _bump(self) -> None:
        self._version += 1

    def to_networkx(self) -> nx.MultiDiGraph:
        """Return a cached NetworkX MultiDiGraph projection of the LEAF graph.

        Only layer-0 entities and the semantic edges among them are projected.
        Abstraction-summary nodes (layer>0) live solely in the hierarchy tree
        and never enter the relational graph the reader propagates over, so
        they cannot distort centrality, communities, or GNN propagation.
        """
        if self._nx_cache is not None and self._nx_cache[0] == self._version:
            return self._nx_cache[1]
        g = nx.MultiDiGraph()
        leaf_ids: Set[int] = set()
        for e in self.all_entities():
            if e.layer != 0:
                continue
            leaf_ids.add(e.id)
            g.add_node(e.id, canonical=e.canonical, type=e.type, description=e.description)
        for edge in self.all_edges():
            # Quarantined beliefs are isolated: they never enter the graph the
            # reader propagates over, so unconfirmed/model-generated content
            # cannot be retrieved as fact.
            if edge.quarantined:
                continue
            if edge.head_id in leaf_ids and edge.tail_id in leaf_ids:
                g.add_edge(
                    edge.head_id, edge.tail_id,
                    key=edge.id,
                    relation=edge.relation,
                    weight=edge.weight,
                    source=edge.source,
                )
        self._nx_cache = (self._version, g)
        return g

    def structural_roles(self) -> Dict[int, Dict[str, float]]:
        """Compute hub / bridge / degree features per entity.

        Returns a dict keyed by entity id with:
            "degree":        total degree (in + out)
            "clustering":    local clustering coefficient on the undirected projection
            "betweenness":   normalized betweenness centrality (rough bridge score)
            "log_degree":    log(1+degree)
        """
        g = self.to_networkx()
        if g.number_of_nodes() == 0:
            return {}
        # Use undirected single-edge projection for centrality.
        ug = nx.Graph()
        ug.add_nodes_from(g.nodes())
        for u, v, data in g.edges(data=True):
            w = data.get("weight", 1.0)
            if ug.has_edge(u, v):
                ug[u][v]["weight"] = max(ug[u][v]["weight"], w)
            else:
                ug.add_edge(u, v, weight=w)
        clustering = nx.clustering(ug)
        # Betweenness can be expensive; cap k for sampled approximation on big graphs.
        n = ug.number_of_nodes()
        k = min(n, 64) if n > 0 else 1
        if n <= 1:
            betweenness = {nid: 0.0 for nid in ug.nodes()}
        else:
            try:
                betweenness = nx.betweenness_centrality(ug, k=k, seed=0, normalized=True)
            except Exception:
                betweenness = {nid: 0.0 for nid in ug.nodes()}
        out: Dict[int, Dict[str, float]] = {}
        for nid in g.nodes():
            deg = g.in_degree(nid) + g.out_degree(nid)
            out[int(nid)] = {
                "degree": float(deg),
                "log_degree": float(np.log1p(deg)),
                "clustering": float(clustering.get(nid, 0.0)),
                "betweenness": float(betweenness.get(nid, 0.0)),
            }
        return out

    def communities(self, resolution: float = 1.0) -> Dict[int, int]:
        """Assign a community label to each entity using greedy modularity."""
        g = self.to_networkx()
        if g.number_of_nodes() == 0:
            return {}
        ug = nx.Graph()
        ug.add_nodes_from(g.nodes())
        for u, v, data in g.edges(data=True):
            w = data.get("weight", 1.0)
            if ug.has_edge(u, v):
                ug[u][v]["weight"] += w
            else:
                ug.add_edge(u, v, weight=w)
        if ug.number_of_edges() == 0:
            return {int(n): int(n) for n in ug.nodes()}
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(
                ug, resolution=resolution, weight="weight"
            )
        except Exception:
            communities = [{n} for n in ug.nodes()]
        label_of: Dict[int, int] = {}
        for idx, comm in enumerate(communities):
            for nid in comm:
                label_of[int(nid)] = idx
        for nid in ug.nodes():
            label_of.setdefault(int(nid), -1)
        return label_of

    # ---- counts -------------------------------------------------------

    def n_entities(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

    def n_edges(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def close(self) -> None:
        self.conn.close()


__all__ = ["GraphSubstrate", "Entity", "Edge"]
