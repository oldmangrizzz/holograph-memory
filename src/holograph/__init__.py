"""HoloGraph: self-evolving hyperdimensional graph memory.

The package is organized into independent subsystems that can be imported and
used in isolation:

    holograph.hdc       — hyperdimensional kernels (Real, Ternary) and memories
    holograph.graph     — graph substrate (SQLite + NetworkX)
    holograph.writer    — memory writer (entity/relation extraction)
    holograph.reader    — memory reader (planner, addressing, gated GNN, HDC)
    holograph.feedback  — reward signals and writer/reader updates
    holograph.runtime   — the closed-loop orchestrator

Top-level re-exports live in `holograph.api` (imported lazily on first
attribute access) so that importing a single subsystem doesn't transitively
import the rest.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "HoloGraph":
        from .runtime import HoloGraph
        return HoloGraph
    if name == "make_kernel":
        from .hdc.kernel import make_kernel
        return make_kernel
    raise AttributeError(name)


__all__ = ["HoloGraph", "make_kernel", "__version__"]
