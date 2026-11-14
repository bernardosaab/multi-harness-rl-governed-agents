"""Group-relative advantages. Pure Python, so the core maths is tested without a GPU.

The *grouping boundary* is an explicit choice, not an implementation detail:
"What Does Multi-Harness RL Learn?" (arXiv 2609.04518) shows that pooling
harnesses into one group (cross-harness) lets the advantage encode *which
harness* produced a rollout, buying configuration adaptation rather than
portable skill. We group **within task x harness** by default and say so.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Literal

Grouping = Literal["task_harness", "task"]


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Standardise rewards within one group: (r - mean) / std.

    A group where every rollout earned the same reward carries no learning
    signal, so it gets all-zero advantages (rather than dividing by ~0).
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


def grouped_advantages(
    rewards: Sequence[float],
    keys: Sequence[tuple[Hashable, Hashable]],
    grouping: Grouping = "task_harness",
) -> list[float]:
    """Advantages for a flat batch where ``keys[i] = (task_id, harness)``."""
    if len(rewards) != len(keys):
        raise ValueError("rewards and keys must have the same length")
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, (task, harness) in enumerate(keys):
        groups[(task, harness) if grouping == "task_harness" else task].append(i)
    out = [0.0] * len(rewards)
    for idx in groups.values():
        for i, a in zip(idx, group_advantages([rewards[i] for i in idx]), strict=True):
            out[i] = a
    return out
