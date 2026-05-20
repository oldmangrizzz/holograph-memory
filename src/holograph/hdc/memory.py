"""Prototype memories and the MAS (Memory Alignment & Separation) diagnostic.

PSP-HDC forms a class prototype as the normalized bundled sum of all sample
hypervectors in that class:

        m_c = norm( sum_{i in T_c} h_i )

Prediction is prototype retrieval:

        y_hat(h) = argmax_c sim(h, m_c)

The class-partitioned component memories at the parameter / group / path levels
are bundled the same way, restricted to the partitioned hypervectors.  MAS
quantifies how well these memories *align* with their own prototype and
*separate* from competing prototypes — a learning-dynamics signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kernel import HDCKernel


@dataclass
class PrototypeMemory:
    """Class-indexed prototype memories formed by bundled sums.

    Storage is plain numpy; the kernel decides the dtype and bundle semantics.
    """

    kernel: HDCKernel
    prototypes: Dict[str, np.ndarray] = field(default_factory=dict)

    # ---- construction --------------------------------------------------

    def fit(self, samples_by_class: Dict[str, List[np.ndarray]]) -> None:
        """Replace stored prototypes by bundling samples per class."""
        self.prototypes = {}
        for cls, hvs in samples_by_class.items():
            if not hvs:
                self.prototypes[cls] = self.kernel.zeros()
                continue
            self.prototypes[cls] = self.kernel.bundle(hvs)

    def update(self, cls: str, hv: np.ndarray) -> None:
        """Incrementally add a sample to class `cls`'s prototype.

        For the real kernel we re-bundle the existing prototype with the new
        hypervector.  For the ternary kernel we re-bundle as a 2-element set,
        which is mathematically correct as long as we are not trying to
        preserve original counts (and we're not — prototypes are normalized).
        """
        if cls not in self.prototypes:
            self.prototypes[cls] = hv.astype(self.kernel.dtype, copy=True)
            return
        self.prototypes[cls] = self.kernel.bundle([self.prototypes[cls], hv])

    # ---- inference -----------------------------------------------------

    def predict(self, hv: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Return (winning class, margin, full score map).

        Margin is the gap between top-1 and top-2 similarities.
        """
        if not self.prototypes:
            raise RuntimeError("PrototypeMemory.predict called on empty memory")
        scores = {cls: self.kernel.similarity(hv, proto) for cls, proto in self.prototypes.items()}
        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_cls, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = top_score - second_score
        return top_cls, margin, scores

    def classes(self) -> List[str]:
        return list(self.prototypes.keys())

    # ---- persistence ---------------------------------------------------

    def to_bytes(self) -> Dict[str, bytes]:
        return {cls: self.kernel.pack(p) for cls, p in self.prototypes.items()}

    def load_bytes(self, blob: Dict[str, bytes]) -> None:
        self.prototypes = {cls: self.kernel.unpack(b) for cls, b in blob.items()}


# ---------------------------------------------------------------------------
# MAS: Memory Alignment & Separation
# ---------------------------------------------------------------------------


def mas(kernel: HDCKernel,
        component_memories: Dict[str, Dict[str, np.ndarray]],
        class_prototypes: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compute MAS for a set of component-level memories.

    Args:
        kernel: the HDC kernel used to compute similarities.
        component_memories: nested dict {component_name: {class_label: hv}}
            For instance, the parameter-level component memories at parameter
            "laser_power" would be {"laser_power": {"high_R": hv, "low_R": hv}}.
        class_prototypes: {class_label: prototype_hv}

    Returns:
        Per-component dict with keys:
            "alignment":  mean similarity of each class component to its
                          OWN prototype.
            "separation": mean (sim(self) - max sim(other)) across classes.
            "ratio":      separation / max(alignment, 1e-8) — a normalized
                          discriminability score in [-1, 1].
    """
    out: Dict[str, Dict[str, float]] = {}
    classes = list(class_prototypes.keys())
    for comp_name, by_cls in component_memories.items():
        aligns: List[float] = []
        seps: List[float] = []
        for cls in classes:
            if cls not in by_cls:
                continue
            v = by_cls[cls]
            own = kernel.similarity(v, class_prototypes[cls])
            competitor = max(
                (kernel.similarity(v, class_prototypes[c]) for c in classes if c != cls),
                default=0.0,
            )
            aligns.append(own)
            seps.append(own - competitor)
        if not aligns:
            out[comp_name] = {"alignment": 0.0, "separation": 0.0, "ratio": 0.0}
            continue
        a = float(np.mean(aligns))
        s = float(np.mean(seps))
        out[comp_name] = {
            "alignment": a,
            "separation": s,
            "ratio": s / max(abs(a), 1e-8),
        }
    return out


__all__ = ["PrototypeMemory", "mas"]
