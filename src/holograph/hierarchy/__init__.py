"""Hierarchical abstraction subsystem (H-MEM-inspired).

Public surface:
    build_hierarchy     construct abstraction layers over the leaf graph
    HierarchyBuilder    the builder object (structural, LLM-free by default)
    SoftRouter          top-down soft index-routing over the hierarchy
    RouteResult         dataclass returned by SoftRouter.route()
"""

from .builder import HierarchyBuilder, build_hierarchy
from .router import SoftRouter, RouteResult

__all__ = [
    "HierarchyBuilder",
    "build_hierarchy",
    "SoftRouter",
    "RouteResult",
]
