# holograph_hdc_rs — Rust accelerator for HoloGraph HDC kernels

This crate is an optional speed accelerator for the HoloGraph runtime.  It
exposes the same `RealKernel` / `TernaryKernel` interface that the pure-Python
implementation does, but with a bitsliced ternary representation that lets
binding fall out as bitwise AND/OR over `u64` lanes — what a future hardware
ternary engine would do natively.

## Build

You need a Rust toolchain (1.76+) and Python 3.10+.

```bash
pip install maturin
cd rust-kernel
maturin develop --release
```

This installs an extension named `holograph._native` into the active Python
environment.  Verify:

```python
from holograph.hdc import make_kernel
k = make_kernel("ternary", dim=16000)
print(type(k).__name__)   # 'TernaryKernelRustAdapter' if Rust is in use
```

If you'd like to compare backends, force the Python implementation with
`make_kernel("ternary", prefer_native=False)`.

## Design

The ternary kernel uses two parallel bit-planes:

```
   pos[k] = 1  iff trit_k == +1
   neg[k] = 1  iff trit_k == -1
   zero        iff both bits are 0
```

so that `bind(a, b)` becomes

```
   result.pos = (a.pos & b.pos) | (a.neg & b.neg)
   result.neg = (a.pos & b.neg) | (a.neg & b.pos)
```

— pure bitwise ops, 64 trits per word, trivially SIMD-friendly.

Similarity uses popcounts of the per-lane match / mismatch / active masks,
mirroring the formula `(matches − mismatches) / active_dims`.

Bundling is performed lane-wise on `i32` counters then thresholded with the
same `0.5 * sqrt(n)` deadband as the Python implementation, so the abstention
channel (zeros) is preserved.

## Why this matters

At SAGE-scale entity counts (10^5 – 10^6) the storage savings of ternary HVs
make the runtime feasible on commodity hardware, and the bitsliced binding
keeps end-to-end query latency in the same ballpark as the float32 path
*despite* being a CPU implementation.  GPU/vector-engine ports are an obvious
follow-up.
