"""Feedback / reward subsystem.

Public surface:
    RewardSignals       per-query reward bundle (deductive / recall / precision / answer)
    FeedbackLoop        closes the writer-reader loop with weight + GNN updates
"""

from .loop import FeedbackLoop, RewardSignals

__all__ = ["FeedbackLoop", "RewardSignals"]
