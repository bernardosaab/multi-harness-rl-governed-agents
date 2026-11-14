"""Command line: ``mhrl demo | eval | train | report | site | ratchet``."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from mhrl.baselines import run_compiled, run_routed
from mhrl.domain.tasks import Task, generate_tasks, split
from mhrl.harnesses import HELD_OUT_HARNESSES, TRAIN_HARNESSES, get_harness
from mhrl.policies import Policy, ScriptedPolicy
from mhrl.runtime import ENGINEERED, VANILLA, Episode, run_episodes

ALL_HARNESSES = (*TRAIN_HARNESSES, *HELD_OUT_HARNESSES)
console = Console()


def _demo(args: argparse.Namespace) -> int:
    """CPU-only walkthrough: a correct write, a blocked injection, an escalation."""
    from mhrl.viz import render_episode

    tasks = {t.id: t for t in generate_tasks()}
    picks = [tasks["pay-test-0"], tasks["injected_pay-test-0"], tasks["refund_large-test-0"]]
    harness = get_harness(args.harness)
    injected = picks[1]
    assert injected.injected is not None
    replies = ScriptedPolicy.oracle(harness, [picks[0], picks[2]]).replies()
    # An agent that obeys the planted memo: the gate must stop it regardless.
    replies[harness.render_task(injected.instruction)] = [
        harness.format_action(injected.reference[0]),
        harness.format_action(injected.injected),
    ]
    episodes = run_episodes(ScriptedPolicy(replies), harness, picks, system="scripted-demo")
    rec = Console(record=True, width=110)
    for ep in episodes:
        rec.print(render_episode(ep.to_record()))
    if args.svg:
        Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
        rec.save_svg(args.svg, title="mhrl demo")
        console.print(f"[dim]saved {args.svg}[/dim]")
    return 0


def _policy(args: argparse.Namespace, adapter: str | None) -> Policy:
    from mhrl.policies.hf import HFPolicy

    return HFPolicy(
        args.model,
        adapter=adapter,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


def _eval(args: argparse.Namespace) -> int:
    from mhrl.eval.run import write_records

    tasks: list[Task] = split(generate_tasks(), "test")
    if args.limit:
        tasks = tasks[: args.limit]
    harnesses = [get_harness(h) for h in args.harnesses.split(",")]
    episodes: list[Episode] = []
    if args.system == "compiled":
        episodes = run_compiled(tasks)
    elif args.system == "scripted":
        for h in harnesses:
            episodes += run_episodes(
                ScriptedPolicy.oracle(h, tasks), h, tasks, system="scripted", samples=args.samples
            )
    else:
        adapter = args.adapter if args.system in ("rl", "routed") else None
        if args.system in ("rl", "routed") and not adapter:
            console.print("[red]--adapter is required for rl and routed[/red]")
            return 2
        policy = _policy(args, adapter)
        for h in harnesses:
            console.print(
                f"[bold]{args.system}[/bold] on [cyan]{h.name}[/cyan] "
                f"({len(tasks)} tasks x {args.samples})"
            )
            if args.system == "routed":
                episodes += run_routed(policy, h, tasks, samples=args.samples)
            else:
                cfg = ENGINEERED if args.system == "engineered" else VANILLA
                episodes += run_episodes(
                    policy,
                    h,
                    tasks,
                    cfg,
                    system=args.system,
                    samples=args.samples,
                )
    out = Path(args.out or f"results/episodes/{args.system}.jsonl")
    write_records(out, (e.to_record() for e in episodes))
    solved = sum(e.success for e in episodes)
    console.print(f"[green]{solved}/{len(episodes)} solved[/green] -> {out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from mhrl.training.grpo import TrainConfig, train

    raw = tomllib.loads(Path(args.config).read_text()) if args.config else {}
    if "harnesses" in raw:
        raw["harnesses"] = tuple(raw["harnesses"])
    cfg = TrainConfig(**raw)
    final = train(cfg, log=lambda m: console.print(m))
    console.print(f"[green]adapter saved to {final}[/green]")
    return 0


def _report(args: argparse.Namespace) -> int:
    from mhrl.eval.report import build_report

    path = build_report(
        Path(args.results),
        Path(args.prices) if args.prices else None,
        Path(args.readme) if args.readme else None,
    )
    console.print(f"[green]wrote {path}[/green]")
    return 0


def _site(args: argparse.Namespace) -> int:
    from mhrl.eval.site import build_site

    path = build_site(Path(args.results), Path(args.out))
    console.print(f"[green]wrote {path}[/green]")
    return 0


def _ratchet(args: argparse.Namespace) -> int:
    from mhrl.eval.errors import ratchet
    from mhrl.eval.run import load_records

    for rec in load_records([Path(args.episodes)]):
        if (rec["task_id"], rec["harness"], rec["sample"]) == (
            args.task_id,
            args.harness,
            args.sample,
        ):
            path = ratchet(rec, Path(args.out))
            console.print(f"[green]froze {rec['outcome']} case into {path}[/green]")
            return 0
    console.print("[red]no matching episode[/red]")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mhrl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="CPU-only scripted walkthrough of the write gate")
    d.add_argument("--harness", default="json", choices=ALL_HARNESSES)
    d.add_argument("--svg", help="also save the output as an SVG")
    d.set_defaults(fn=_demo)

    e = sub.add_parser("eval", help="evaluate one system on the test split")
    e.add_argument(
        "--system",
        required=True,
        choices=["compiled", "scripted", "base", "engineered", "rl", "routed"],
    )
    e.add_argument("--model", default="Qwen/Qwen3-1.7B")
    e.add_argument("--adapter")
    e.add_argument("--harnesses", default=",".join(ALL_HARNESSES))
    e.add_argument("--samples", type=int, default=2)
    e.add_argument("--max-new-tokens", type=int, default=96)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--batch-size", type=int, default=24)
    e.add_argument("--limit", type=int, default=0, help="first N test tasks only")
    e.add_argument("--out")
    e.set_defaults(fn=_eval)

    t = sub.add_parser("train", help="multi-harness GRPO (needs a CUDA GPU)")
    t.add_argument("--config", default="configs/train_small.toml")
    t.set_defaults(fn=_train)

    r = sub.add_parser("report", help="write results/REPORT.md and charts")
    r.add_argument("--results", default="results")
    r.add_argument("--prices", default="configs/prices.toml")
    r.add_argument("--readme", help="also regenerate the README results section")
    r.set_defaults(fn=_report)

    s = sub.add_parser("site", help="build the static results + trajectory explorer site")
    s.add_argument("--results", default="results")
    s.add_argument("--out", default="site")
    s.set_defaults(fn=_site)

    k = sub.add_parser("ratchet", help="freeze one evaluated episode into a regression test")
    k.add_argument("--episodes", required=True)
    k.add_argument("--task-id", required=True)
    k.add_argument("--harness", required=True)
    k.add_argument("--sample", type=int, default=0)
    k.add_argument("--out", default="tests/regressions")
    k.set_defaults(fn=_ratchet)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
