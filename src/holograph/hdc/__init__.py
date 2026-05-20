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
from .encoder import ScalarEncoder
from .memory import PrototypeMemory, mas

__all__ = [
    "HDCKernel",
    "RealKernel",
    "TernaryKernel",
    "make_kernel",
    "ScalarEncoder",
    "PrototypeMemory",
    "mas",
]
