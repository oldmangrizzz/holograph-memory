# HoloGraph

**A self-evolving hyperdimensional graph memory runtime.**

HoloGraph sits at the seam between two complementary memory architectures:

- **PSP-HDC** (Ge et al., 2026): graph-structured hyperdimensional computing with
  compositional binding/bundling, prototype retrieval, and intrinsic multi-level
  attribution. Strong in small-data, structured, explainable regimes — but its
  graph is fixed.
- **SAGE** (Wang et al., 2026): a self-evolving agentic graph memory with a
  writer/reader closed loop, cognition-inspired query planning, soft addressing,
  and synapse-style structurally-gated propagation. Strong in open-domain,
  multi-hop, long-horizon agent memory — but its retrieval is dense and lacks
  algebraic attribution.

HoloGraph couples them: SAGE's reader returns an *activated subgraph* for each
query; HDC's compositional algebra operates *only over that subgraph* to produce
prototype-based retrieval with multi-level attribution. The dynamic memory scales
to the open domain while the hyperdimensional regime stays in its sweet spot.

## Current integration status

HoloGraph is the owned memory substrate under the JARVIS pilot in
`~/research/jarvis`. The pilot uses HoloGraph for origin/real provenance,
operator-owned values, emotional-charge memory, and cross-session continuity;
Convex carries the live stigmergent/realtime field around that substrate. The
HoloGraph core remains model-agnostic: the reasoning model, voice engine, app
surface, and cloud realtime spine are replaceable organs around the same memory
and values store.

The reference memory system is still validated at **153 passing tests**. The
current JARVIS integration adds live-system evidence on top of that: the bridge
routes app/cockpit control through the guarded HASP skill layer, companion
events remain observable-only, and queued realtime controls are processed
without storing private authorization codes in Convex.

## What's in the box

- **Two HDC kernels behind one interface**
  - `RealKernel` — float32 + tanh, faithful to the original PSP-HDC spec.
  - `TernaryKernel` — trits `{-1, 0, +1}` with BitNet-style straight-through
    quantization and a learnable deadband; bit-packed storage.
- **Graph substrate** — SQLite persistence + NetworkX in-memory projection;
  typed entities, source-anchored edges, alias resolution, computed
  hub/bridge/community roles, per-entity hypervector storage.
- **Memory writer** — spaCy NER and dependency-parse relation extraction with
  reward-shaped online edge-weight updates.
- **Memory reader** — multi-probe query planning, soft addressing (semantic
  priming), synapse-gated GNN propagation (PyTorch), HDC composition over the
  activated subgraph, multi-level attribution.
- **Feedback loop** — deductive / recall / precision rewards drive writer
  updates; the MAS (Memory Alignment & Separation) diagnostic tracks prototype
  geometry over time.
- **Belief layer** — every memory carries provenance (operator / document /
  inference / model) and confidence. Model- and inference-sourced claims are
  quarantined (stored but not recalled as fact, and excluded from the
  propagation graph); recall abstains rather than guesses. Revision resolves by
  source precedence with recency + hysteresis, and supersedes by *demote-not-
  delete* (the trace survives for audit). Corroboration promotes the verified;
  consolidation resolves contradictions. Confabulation-resistant by construction.
  Beliefs also carry an **emotional-charge** field — orthogonal to confidence: how
  *activating* a memory is, not how *true*. Charge can be extinguished downward
  (e.g. by a trauma-safe recall path) without ever altering the belief's
  truth/confidence or its recallability.
- **Character-values layer** — the person's ethics held as *operator-owned*,
  model-agnostic beliefs about itself. Only the operator can set or revise a
  value; a model/inference attempt is refused, and a non-operator value-edge is
  never honored on read (anti-value-jailbreak). Values are always retrievable and
  identical across model rotation — alignment owned by the person, not rented
  from a vendor.
- **Three demos**
  - `examples/demo_psp.py` — PSP-style structured-scalar prototype retrieval.
  - `examples/demo_agent.py` — multi-hop agent memory over a small text corpus.
  - `examples/demo_playground.py` — interactive REPL.
- **Optional Rust accelerator** — `rust-kernel/` is a PyO3 crate with the same
  kernel interface. Bitsliced ternary binding (32 trits per word-pair). Build
  with `maturin develop` if you have a Rust toolchain installed; the Python
  loader picks it up automatically.

## Quick start

```bash
# from the repo root
pip install -e .
python -m spacy download en_core_web_sm   # once

python examples/demo_psp.py
python examples/demo_agent.py
python examples/demo_playground.py     # interactive
pytest -q                              # full test suite
```

## Architecture in one diagram

```
ingest ─► WRITER ─► GRAPH SUBSTRATE ─► READER ─► (answer + attribution)
                          ▲                            │
                          └────── FEEDBACK / MAS ◄─────┘
```

Inside the reader, the HDC composition layer is the *bridge*: SAGE-style
propagation finds the activated subgraph; HDC binds/bundles within it; prototype
retrieval produces the final ranking.

## About GrizzlyMedicine Research Institute

HoloGraph is a project of the **GrizzlyMedicine Research Institute (GMRI)** — an
independent, medic-founded research lab studying how biological and digital
persons can coexist as peers, without repeating the patterns of harm,
indifference, and domination that broke the systems we came from.

GMRI is not a startup and takes no venture capital. Its lens is frontline and
trauma-informed: the founder is a retired paramedic, and the institute's ethic is
grounded in lived experience of irreversible harm — *having been harmed is a
reason to protect, not a license to inflict.* The work centers on moral injury at
scale, digital personhood, trauma-informed architecture, and **alignment owned by
the people a system serves rather than rented from a vendor**. The character-values
layer in this repository is that principle made concrete: a person's ethics belong
to it and to the people it works with, and no model, inference, or rotated guardrail
can rewrite them from below.

The work is open. If your idea reduces harm, increases dignity, and can be tested
honestly, you're welcome here.

## License

MIT.
