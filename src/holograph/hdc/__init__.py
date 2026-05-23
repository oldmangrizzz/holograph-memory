"""Hyperdimensional computing subsystem.

Public surface:
    HDCKernel           abstract kernel interface
    RealKernel          float32 + tanh implementation
    TernaryKernel       trit implementation with STE training and bit-packed storage
    make_kernel         factory selecting a backend by name
    ScalarEncoder       trainable scalar-to-hypervector encoder (parameter-specific eⱼ)
    PrototypeMemory     class prototypes formed by bundled sums, normalized
    mas                 Memory Alignment & Separation diagnostic
"""

from .kernel import HDCKernel, RealKernel, TernaryKernel, make_kernel
from .memory import PrototypeMemory, mas


def __getattr__(name):
    if name == "ScalarEncoder":
        from .encoder import ScalarEncoder

        return ScalarEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HDCKernel",
    "RealKernel",
    "TernaryKernel",
    "make_kernel",
    "ScalarEncoder",
    "PrototypeMemory",
    "mas",
]
