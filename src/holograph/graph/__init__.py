"""Graph substrate subsystem.

Public surface:
    GraphSubstrate     SQLite-backed persistent graph with NetworkX projection
    Entity             entity dataclass
    Edge               edge dataclass
"""

from .substrate import GraphSubstrate, Entity, Edge

__all__ = ["GraphSubstrate", "Entity", "Edge"]
