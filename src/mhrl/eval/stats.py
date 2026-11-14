"""Small-sample statistics, written out so they can be read and tested.

* Wilson score interval for a pass rate (well-behaved near 0% and 100%).
* pass^k: the share of cases solved on *every* one of k attempts (reliability),
  as opposed to pass@k (solved at least once).
* Exact McNemar test on paired binary outcomes (same cases, two systems).
* Paired cluster bootstrap for the difference in pass rate. Episodes of the same
  task (across harnesses and samples) are correlated, so whole tasks are
  resampled; resampling single episodes would make the interval too narrow.

Wilson intervals and McNemar treat episodes as independent; with correlated
episodes they are somewhat optimistic, which is why the clustered bootstrap
interval is the headline uncertainty for every comparison.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pass_hat_k(cases: Sequence[Sequence[bool]]) -> float:
    """Share of cases where every attempt succeeded."""
    if not cases:
        return 0.0
    return sum(all(c) for c in cases) / len(cases)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    ``b`` = cases only system A solved, ``c`` = cases only system B solved.
    Under H0 the discordant cases split 50/50, so this is a binomial test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = float(sum(math.comb(n, i) for i in range(k + 1))) / float(2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    iters: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of (b - a) over paired cases, with a 95% percentile interval.

    With ``clusters`` (e.g. the task id of each pair) whole clusters are resampled
    with replacement; without it every pair is its own cluster.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty paired samples")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    keys = list(clusters) if clusters is not None else list(range(len(diffs)))
    if len(keys) != len(diffs):
        raise ValueError("clusters must label every pair")
    groups: dict[Hashable, list[float]] = {}
    for k, d in zip(keys, diffs, strict=True):
        groups.setdefault(k, []).append(d)
    units = list(groups.values())
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        picked = [units[rng.randrange(len(units))] for _ in units]
        total = sum(sum(u) for u in picked)
        count = sum(len(u) for u in picked)
        means.append(total / count)
    means.sort()
    return (sum(diffs) / len(diffs), means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
