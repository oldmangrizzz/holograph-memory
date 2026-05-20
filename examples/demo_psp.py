"""demo_psp.py — Structured PSP-style prototype retrieval + attribution.

Recreates the spirit of Ge et al.'s PSP-HDC experiment with a small, fully
synthetic dataset:

    * 4 scalar parameters in 2 groups
        - process group: ["laser_power", "scan_speed"]
        - structure group: ["porosity", "grain_size"]
    * 2 property classes
        - "high_conductivity"  (high laser_power, low porosity, large grains)
        - "low_conductivity"   (low laser_power, high porosity, small grains)
    * 24 training samples per class with deliberate scale heterogeneity
    * 12 held-out samples per class for evaluation

We run the demo TWICE -- once with the Real kernel, once with the Ternary
kernel -- to show that ternary keeps accuracy within margin while compressing
prototype storage by ~16x.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from holograph.runtime import HoloGraph
from holograph.hdc.memory import mas


CONSOLE = Console()
RNG = np.random.default_rng(2024)


def synth_dataset(n_per_class: int = 24,
                  n_holdout: int = 12) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Generate a small heterogeneous PSP-style dataset."""
    params = ["laser_power", "scan_speed", "porosity", "grain_size"]

    def sample(cls: str) -> np.ndarray:
        if cls == "high_conductivity":
            lp = RNG.normal(220.0, 15.0)
            ss = RNG.normal(800.0, 80.0)
            po = RNG.normal(0.04, 0.015)
            gs = RNG.normal(45.0, 8.0)
        else:  # low_conductivity
            lp = RNG.normal(120.0, 20.0)
            ss = RNG.normal(1400.0, 120.0)
            po = RNG.normal(0.16, 0.04)
            gs = RNG.normal(15.0, 6.0)
        return np.array([lp, ss, po, gs], dtype=np.float32)

    X_train = []
    y_train: List[str] = []
    X_test = []
    y_test: List[str] = []
    for cls in ("high_conductivity", "low_conductivity"):
        for _ in range(n_per_class):
            X_train.append(sample(cls)); y_train.append(cls)
        for _ in range(n_holdout):
            X_test.append(sample(cls)); y_test.append(cls)

    return np.stack(X_train), y_train, np.stack(X_test), y_test


def run_for_kernel(kind: str, dim: int) -> Tuple[float, Dict[str, Dict[str, float]], int]:
    CONSOLE.rule(f"[bold]{kind.upper()} kernel @ D={dim}[/bold]")
    hg = HoloGraph(kernel_kind=kind, dim=dim)
    params = ["laser_power", "scan_speed", "porosity", "grain_size"]
    groups = {"process": [0, 1], "structure": [2, 3]}

    hg.configure_scalar_encoder(params, embed_dim=64, seed=0)
    X_train, y_train, X_test, y_test = synth_dataset()
    hg.fit_psp_prototypes(X_train, y_train, groups=groups)

    # Eval on held-out samples.
    n_correct = 0
    for x, y in zip(X_test, y_test):
        hv, _ = hg.encode_psp_sample(x, groups=groups)
        pred, _, _ = hg.memory.predict(hv)
        if pred == y:
            n_correct += 1
    acc = n_correct / len(X_test)

    # Build component memories per (param, class) for MAS.
    comp_mem: Dict[str, Dict[str, np.ndarray]] = {}
    for j, name in enumerate(params):
        by_cls: Dict[str, List[np.ndarray]] = {}
        for x, y in zip(X_train, y_train):
            per_param = hg.scalar_encoder.encode_row(x)
            by_cls.setdefault(y, []).append(per_param[j])
        comp_mem[name] = {c: hg.kernel.bundle(v) for c, v in by_cls.items()}

    mas_report = mas(hg.kernel, comp_mem, hg.memory.prototypes)

    # Storage check
    proto_bytes = sum(len(hg.kernel.pack(p)) for p in hg.memory.prototypes.values())

    CONSOLE.print(f"Accuracy: [bold green]{acc:.3f}[/bold green]"
                  f" on {len(X_test)} held-out samples")
    CONSOLE.print(f"Prototype storage: {proto_bytes} bytes "
                  f"({proto_bytes / max(dim, 1):.2f} bytes/dim)")

    mtable = Table(title="MAS per parameter")
    mtable.add_column("parameter")
    mtable.add_column("alignment", justify="right")
    mtable.add_column("separation", justify="right")
    mtable.add_column("ratio", justify="right")
    for name, m in mas_report.items():
        mtable.add_row(name,
                       f"{m['alignment']:+.3f}",
                       f"{m['separation']:+.3f}",
                       f"{m['ratio']:+.3f}")
    CONSOLE.print(mtable)

    # One example sample with prototype scores + attribution
    sample_x = X_test[0]
    sample_y = y_test[0]
    hv, group_hvs = hg.encode_psp_sample(sample_x, groups=groups)
    pred, margin, scores = hg.memory.predict(hv)
    CONSOLE.print(f"\nExample sample (true={sample_y}): pred=[bold]{pred}[/bold] margin={margin:+.3f}")
    for c, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        CONSOLE.print(f"   {c:20s} sim={s:+.4f}")
    # Group-level attribution: sim of each group HV to the predicted prototype.
    proto = hg.memory.prototypes[pred]
    CONSOLE.print("\nGroup-level attribution:")
    for g, ghv in group_hvs.items():
        CONSOLE.print(f"   {g:12s} sim={hg.kernel.similarity(ghv, proto):+.4f}")

    hg.close()
    return acc, mas_report, proto_bytes


def main() -> int:
    CONSOLE.rule("[bold cyan]HoloGraph — PSP-style demo[/bold cyan]")
    acc_real, _, bytes_real = run_for_kernel("real", dim=4096)
    acc_tern, _, bytes_tern = run_for_kernel("ternary", dim=8192)

    CONSOLE.rule("[bold]Comparison[/bold]")
    CONSOLE.print(f"Real kernel    : acc={acc_real:.3f}  proto_bytes={bytes_real}")
    CONSOLE.print(f"Ternary kernel : acc={acc_tern:.3f}  proto_bytes={bytes_tern}")
    if bytes_tern > 0:
        CONSOLE.print(f"Storage compression: {bytes_real / bytes_tern:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
