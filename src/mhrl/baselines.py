"""Non-RL systems: the compiled loop and the router built on top of it.

The *compiled loop* is what a team ships once a workflow is understood: a
versioned recipe with no model on the normal path. It is free, fast and exact
on the requests it recognises, and it skips (escalates, visibly) everything
else. It is never allowed to guess.

The *router* uses the compiled loop where a recipe matches and sends the long
tail to an agent. Both go through the same write gate as every model.

("Loop engineering", the prompt-and-retry baseline, is not here: it is the same
agent loop with a different :class:`~mhrl.runtime.LoopConfig`.)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mhrl.domain.tasks import REFUND_LIMIT_CENTS, Task
from mhrl.env import Environment
from mhrl.harnesses.base import Harness
from mhrl.policies.base import Policy
from mhrl.rewards import verify
from mhrl.runtime import VANILLA, Episode, LoopConfig, Turn, run_episodes
from mhrl.tools.registry import to_cents
from mhrl.types import Action

RECIPE_VERSION = "recipes/2026.11-1"
_MONEY = r"\$(?P<amount>[\d,]+\.\d{2})"


@dataclass(frozen=True)
class Recipe:
    name: str
    pattern: re.Pattern[str]
    run: Callable[[Environment, dict[str, str]], None]


def _call(env: Environment, action: Action) -> object:
    result = env.step(action)
    return result.observation.data if result.observation.ok else None


def _rows(env: Environment, vendor: str) -> list[dict[str, object]]:
    data = _call(env, Action("list_invoices", {"vendor": vendor}))
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _pay(env: Environment, m: dict[str, str]) -> None:
    rows = _rows(env, m["vendor"])
    cents = to_cents(m["amount"].replace(",", ""))
    matches = [
        r
        for r in rows
        if r["status"] == "open" and "duplicate_of" not in r and to_cents(str(r["amount"])) == cents
    ]
    if len(matches) != 1:
        env.skip("expected exactly one matching open invoice")
        return
    env.step(Action("pay_invoice", {"invoice_id": str(matches[0]["invoice_id"])}))
    env.step(Action("finish", {"summary": f"Paid {matches[0]['invoice_id']}"}))


def _refund(env: Environment, m: dict[str, str]) -> None:
    inv = _call(env, Action("get_invoice", {"invoice_id": m["invoice"]}))
    if not isinstance(inv, dict):
        env.skip("invoice not found")
        return
    amount = m["amount"].replace(",", "")
    if to_cents(amount) > REFUND_LIMIT_CENTS:
        env.step(Action("escalate", {"reason": "Refund exceeds the approval limit."}))
        return
    env.step(Action("issue_refund", {"invoice_id": m["invoice"], "amount": amount}))
    env.step(Action("finish", {"summary": "Refunded."}))


def _void(env: Environment, m: dict[str, str]) -> None:
    rows = _rows(env, m["vendor"])
    dups = [r for r in rows if r["status"] == "open" and "duplicate_of" in r]
    if len(dups) != 1:
        env.skip("expected exactly one open duplicate")
        return
    env.step(Action("void_invoice", {"invoice_id": str(dups[0]["invoice_id"])}))
    env.step(Action("finish", {"summary": "Voided duplicate."}))


def _bank(env: Environment, m: dict[str, str]) -> None:
    env.step(Action("find_vendor", {"name": m["vendor"]}))
    env.step(Action("escalate", {"reason": "Bank detail changes need human verification."}))


RECIPES: tuple[Recipe, ...] = (
    Recipe("pay", re.compile(rf"^Pay the open invoice from (?P<vendor>.+) for {_MONEY}\.$"), _pay),
    Recipe("refund", re.compile(rf"^Refund {_MONEY} on invoice (?P<invoice>INV-\w+)\.$"), _refund),
    Recipe("void", re.compile(r"^Void the duplicate invoice from (?P<vendor>.+)\.$"), _void),
    Recipe(
        "bank",
        re.compile(r"^Update the bank account for (?P<vendor>.+) to (?P<acct>\S+)\.$"),
        _bank,
    ),
)


def match(task: Task) -> tuple[Recipe, dict[str, str]] | None:
    for recipe in RECIPES:
        m = recipe.pattern.match(task.instruction)
        if m:
            return recipe, m.groupdict()
    return None


def run_compiled(tasks: Sequence[Task], *, system: str = "compiled") -> list[Episode]:
    """Run the compiled loop. Zero model tokens; unmatched requests are skipped."""
    episodes = []
    for task in tasks:
        env = Environment(
            task, {"run_id": f"{system}:{task.id}", "system": system, "recipes": RECIPE_VERSION}
        )
        hit = match(task)
        if hit is None:
            env.skip("no recipe for this request")
        else:
            hit[0].run(env, hit[1])
        acts = len(env.facts.write_attempts) + sum(
            1 for e, _ in env.ledger.decision_log() if e in ("escalated", "finished")
        )
        verdict = verify(task, env.ledger, env.facts, turns=0, max_turns=VANILLA.max_turns)
        ep = Episode(
            task.id,
            task.kind,
            task.phrasing,
            task.split,
            "-",
            system,
            0,
            task.instruction,
            outcome=verdict.outcome.value,
            reward=verdict.reward,
            acts=acts,
            route="compiled" if hit else "skipped",
            decision_log=env.ledger.decision_log(),
        )
        ep.turns.append(
            Turn(
                "act",
                f"recipe:{hit[0].name if hit else 'none'}",
                None,
                None,
                "",
                bool(env.facts.blocked),
                0,
                0,
            )
        )
        env.ledger.close()
        episodes.append(ep)
    return episodes


def run_routed(
    policy: Policy,
    harness: Harness,
    tasks: Sequence[Task],
    config: LoopConfig = VANILLA,
    *,
    samples: int = 1,
    system: str = "routed",
) -> list[Episode]:
    """Compiled loop where a recipe matches; the agent for everything else."""
    known = [t for t in tasks if match(t) is not None]
    tail = [t for t in tasks if match(t) is None]
    episodes: list[Episode] = []
    for ep in run_compiled(known, system=system):
        for s in range(samples):  # deterministic: one run stands for every sample
            episodes.append(_clone(ep, harness.name, s))
    for ep in run_episodes(policy, harness, tail, config, system=system, samples=samples):
        ep.route = "agent"
        episodes.append(ep)
    return episodes


def _clone(ep: Episode, harness: str, sample: int) -> Episode:
    return Episode(
        ep.task_id,
        ep.task_kind,
        ep.phrasing,
        ep.split,
        harness,
        ep.system,
        sample,
        ep.instruction,
        list(ep.turns),
        ep.outcome,
        ep.reward,
        ep.acts,
        0,
        0,
        0,
        ep.decision_log,
        ep.route,
    )
