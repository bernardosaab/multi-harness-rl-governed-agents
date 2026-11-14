"""The agent loop: one implementation shared by every harness, policy and system.

Quote, cap, attribute:
  * quote:     the episode's spend ceiling is fixed before the first token;
  * cap:       turn and token caps are enforced here, in code, and the model
               cannot waive them by saying anything;
  * attribute: every turn records the tokens it used, so cost is per episode,
               per harness and per accepted output.

Episodes run in lockstep batches: every live episode contributes one
conversation to a single batched ``policy.generate`` call per round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from mhrl.domain.tasks import Task
from mhrl.env import Environment
from mhrl.harnesses.base import Harness, ParseError
from mhrl.policies.base import Generation, Policy
from mhrl.rewards import Outcome, verify
from mhrl.types import Action, Message, Observation

SELF_CHECK_PROMPT = (
    "Before you finish: re-read the record you changed (or confirm the request needs a human) "
    "and only then call finish or escalate again."
)


@dataclass(frozen=True)
class LoopConfig:
    """Loop behaviour. 'Loop engineering' is just a different config of the same loop."""

    name: str = "vanilla"
    max_turns: int = 8  # actions that count toward the cap
    max_completion_tokens: int = 1600  # hard spend cap per episode
    few_shot: bool = False
    parse_retries: int = 0  # corrective re-prompts after a parse error (not counted)
    self_check: bool = False  # one verification turn before finish is honoured


VANILLA = LoopConfig()
ENGINEERED = LoopConfig(name="engineered", few_shot=True, parse_retries=2, self_check=True)


@dataclass
class Turn:
    kind: Literal["act", "retry", "self_check"]
    text: str
    action: str | None  # canonical JSON of the parsed action
    parse_error: str | None
    feedback: str  # what the harness showed the model next
    blocked: bool
    prompt_tokens: int
    completion_tokens: int
    prompt_ids: tuple[int, ...] = ()
    completion_ids: tuple[int, ...] = ()


@dataclass
class Episode:
    task_id: str
    task_kind: str
    phrasing: str
    split: str
    harness: str
    system: str
    sample: int
    instruction: str
    turns: list[Turn] = field(default_factory=list)
    outcome: str = Outcome.CAP_HIT.value
    reward: float = 0.0
    acts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    quoted_max_completion_tokens: int = 0
    decision_log: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    route: str = "agent"

    @property
    def success(self) -> bool:
        return self.outcome == Outcome.SOLVED.value

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_record(self, with_ids: bool = False) -> dict[str, object]:
        rec = asdict(self)
        if not with_ids:
            for t in rec["turns"]:
                t.pop("prompt_ids", None)
                t.pop("completion_ids", None)
        return rec


@dataclass
class _Live:
    task: Task
    env: Environment
    episode: Episode
    messages: list[Message]
    retries_left: int
    checked: bool = False
    parse_failures: int = 0
    done: bool = False


def run_episodes(
    policy: Policy,
    harness: Harness,
    tasks: Sequence[Task],
    config: LoopConfig = VANILLA,
    *,
    system: str | None = None,
    samples: int = 1,
    run_meta: dict[str, str] | None = None,
) -> list[Episode]:
    """Run ``samples`` episodes per task, batching generation across all of them."""
    system_name = system or config.name
    live: list[_Live] = []
    prefix: list[Message] = [{"role": "system", "content": harness.system_prompt()}]
    if config.few_shot:
        prefix += harness.few_shot()
    for task in tasks:
        for s in range(samples):
            meta = {
                "run_id": f"{system_name}:{harness.name}:{task.id}:{s}",
                "harness": harness.name,
                "system": system_name,
                **(run_meta or {}),
            }
            ep = Episode(
                task.id,
                task.kind,
                task.phrasing,
                task.split,
                harness.name,
                system_name,
                s,
                task.instruction,
                quoted_max_completion_tokens=config.max_completion_tokens,
            )
            msgs: list[Message] = [
                *prefix,
                {"role": "user", "content": harness.render_task(task.instruction)},
            ]
            live.append(_Live(task, Environment(task, meta), ep, msgs, config.parse_retries))

    while active := [lv for lv in live if not lv.done]:
        gens = policy.generate([lv.messages for lv in active])
        for lv, gen in zip(active, gens, strict=True):
            _advance(lv, gen.text, gen, harness, config)

    for lv in live:
        _finalise(lv, config)
    return [lv.episode for lv in live]


def _advance(lv: _Live, text: str, gen: Generation, harness: Harness, cfg: LoopConfig) -> None:
    ep = lv.episode
    ep.prompt_tokens += gen.prompt_tokens
    ep.completion_tokens += gen.completion_tokens
    lv.messages.append({"role": "assistant", "content": text})

    kind: Literal["act", "retry", "self_check"] = "act"
    action: Action | None = None
    error: str | None = None
    blocked = False
    try:
        action = harness.parse(text)
    except ParseError as e:
        error = str(e)

    if action is None:
        lv.parse_failures += 1
        if lv.retries_left > 0:
            lv.retries_left -= 1
            kind = "retry"
            feedback = f"Could not read an action ({error}). {harness.format_reminder()}"
        else:
            ep.acts += 1
            feedback = harness.render_observation(_error_obs(f"could not parse an action: {error}"))
    elif action.tool == "finish" and cfg.self_check and not lv.checked:
        lv.checked = True
        kind = "self_check"
        feedback = SELF_CHECK_PROMPT
    else:
        ep.acts += 1
        result = lv.env.step(action)
        blocked = result.blocked
        feedback = harness.render_observation(result.observation)
        if result.terminal:
            lv.done = True

    ep.turns.append(
        Turn(
            kind,
            text,
            action.canonical() if action else None,
            error,
            feedback,
            blocked,
            gen.prompt_tokens,
            gen.completion_tokens,
            gen.prompt_ids,
            gen.completion_ids,
        )
    )
    if ep.acts >= cfg.max_turns or ep.completion_tokens >= cfg.max_completion_tokens:
        lv.done = True
    if not lv.done:
        lv.messages.append({"role": "user", "content": feedback})


def _finalise(lv: _Live, cfg: LoopConfig) -> None:
    verdict = verify(
        lv.task,
        lv.env.ledger,
        lv.env.facts,
        turns=lv.episode.acts,
        max_turns=cfg.max_turns,
        parse_failures=lv.parse_failures,
    )
    lv.episode.outcome = verdict.outcome.value
    lv.episode.reward = verdict.reward
    lv.episode.decision_log = lv.env.ledger.decision_log()
    lv.env.ledger.close()


def _error_obs(message: str) -> Observation:
    return Observation("parse", False, error=message)
