"""Writer subsystem.

Public surface:
    MemoryWriter        ingests text/observations into the graph substrate
    ExtractedTriple     a single (head, rel, tail) extraction with source
"""

from .writer import MemoryWriter, ExtractedTriple

__all__ = ["MemoryWriter", "ExtractedTriple"]
