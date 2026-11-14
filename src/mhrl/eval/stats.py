"""Small-sample statistics, written out so they can be read and tested.

* Wilson score interval for a pass rate (well-behaved near 0% and 100%).
* pass^k: the share of cases solved on *every* one of k attempts (reliability),
  as opposed to pass@k (solved at least once).
* Exact McNemar test on paired binary outcomes (same cases, two systems).
* Paired cluster bootstrap for the difference in pass rate, resampling cases.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


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
    iters: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of (b - a) over paired cases, with a 95% percentile interval."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty paired samples")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
