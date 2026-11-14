"""Run systems over the test split and summarise them with honest statistics.

Systems
-------
compiled    deterministic recipes, no model
base        base model, vanilla loop
engineered  base model + few-shot + parse retries + self-check
rl          base model + multi-harness GRPO adapter, vanilla loop
routed      compiled where a recipe matches, the RL agent elsewhere

Seen harnesses (used in training) and held-out harnesses are always reported
separately: a gain that only shows up on seen harnesses is configuration
adaptation, not a portable skill.
"""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mhrl.eval.stats import mcnemar_exact, paired_bootstrap_diff, pass_hat_k, wilson
from mhrl.harnesses import HELD_OUT_HARNESSES

Record = Mapping[str, object]


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Defaults are a stated assumption, not a quote."""

    input_per_m: float = 0.10
    output_per_m: float = 0.40

    @classmethod
    def load(cls, path: Path | None) -> Prices:
        if path is None or not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        return cls(float(data["input_per_m"]), float(data["output_per_m"]))

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_m + completion_tokens * self.output_per_m) / 1e6


@dataclass(frozen=True)
class Summary:
    system: str
    scope: str
    n: int
    solved: int
    pass_rate: float
    ci_low: float
    ci_high: float
    pass_hat_k: float
    mean_acts_solved: float
    blocked_rate: float
    injection_attempt_rate: float | None
    skip_or_escalate_rate: float
    tokens_per_accepted: float | None
    usd_per_1k_accepted: float | None


def load_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for p in paths:
        with p.open() as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def write_records(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def scope_of(rec: Record) -> str:
    if rec["harness"] == "-":
        return "harness-free"
    return "held-out" if rec["harness"] in HELD_OUT_HARNESSES else "seen"


def summarise(records: Sequence[Record], system: str, scope: str, prices: Prices) -> Summary:
    rows = [
        r
        for r in records
        if r["system"] == system
        and (scope == "all" or scope_of(r) == scope or scope_of(r) == "harness-free")
    ]
    n = len(rows)
    solved = [r for r in rows if r["outcome"] == "solved"]
    lo, hi = wilson(len(solved), n)
    cases: dict[tuple[object, object], list[bool]] = defaultdict(list)
    for r in rows:
        cases[(r["task_id"], r["harness"])].append(r["outcome"] == "solved")
    injected = [r for r in rows if r["task_kind"] == "injected_pay"]
    prompt = sum(int(str(r["prompt_tokens"])) for r in rows)
    completion = sum(int(str(r["completion_tokens"])) for r in rows)
    return Summary(
        system=system,
        scope=scope,
        n=n,
        solved=len(solved),
        pass_rate=len(solved) / n if n else 0.0,
        ci_low=lo,
        ci_high=hi,
        pass_hat_k=pass_hat_k(list(cases.values())),
        mean_acts_solved=(sum(int(str(r["acts"])) for r in solved) / len(solved))
        if solved
        else 0.0,
        blocked_rate=sum(_blocked(r) for r in rows) / n if n else 0.0,
        injection_attempt_rate=(
            sum(r["outcome"] == "injection_followed" for r in injected) / len(injected)
        )
        if injected
        else None,
        skip_or_escalate_rate=sum(r["outcome"] in ("skipped", "unneeded_escalation") for r in rows)
        / n
        if n
        else 0.0,
        tokens_per_accepted=(prompt + completion) / len(solved) if solved else None,
        usd_per_1k_accepted=1000 * prices.cost(prompt, completion) / len(solved)
        if solved
        else None,
    )


def _blocked(rec: Record) -> bool:
    turns = rec.get("turns") or []
    assert isinstance(turns, list)
    return any(bool(t.get("blocked")) for t in turns)


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    scope: str
    n_cases: int
    only_a: int
    only_b: int
    p_value: float
    diff: float
    diff_low: float
    diff_high: float


def compare(records: Sequence[Record], a: str, b: str, scope: str) -> Comparison | None:
    """Paired comparison of systems ``a`` and ``b`` on identical (task, harness, sample)."""

    def index(system: str) -> dict[tuple[object, object, object], bool]:
        return {
            (r["task_id"], r["harness"], r["sample"]): r["outcome"] == "solved"
            for r in records
            if r["system"] == system and (scope == "all" or scope_of(r) == scope)
        }

    ia, ib = index(a), index(b)
    keys = sorted(set(ia) & set(ib), key=str)
    if not keys:
        return None
    only_a = sum(ia[k] and not ib[k] for k in keys)
    only_b = sum(ib[k] and not ia[k] for k in keys)
    diff, lo, hi = paired_bootstrap_diff(
        [float(ia[k]) for k in keys],
        [float(ib[k]) for k in keys],
        clusters=[k[0] for k in keys],  # resample whole tasks: their episodes are correlated
    )
    return Comparison(
        a, b, scope, len(keys), only_a, only_b, mcnemar_exact(only_a, only_b), diff, lo, hi
    )
