"""Turn episode records into ``results/REPORT.md``, ``summary.json`` and SVG charts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mhrl.eval.charts import Bar, grouped_hbar, lines
from mhrl.eval.errors import clusters
from mhrl.eval.run import Prices, compare, load_records, summarise

SYSTEM_ORDER = ("compiled", "base", "engineered", "rl", "routed")
SYSTEM_LABEL = {
    "compiled": "Compiled loop",
    "base": "Base model",
    "engineered": "Loop engineering",
    "rl": "Multi-harness RL",
    "routed": "Routed hybrid",
}
PAIRS = (("base", "engineered"), ("base", "rl"), ("engineered", "rl"), ("rl", "routed"))


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def build_summary(results: Path, prices: Prices) -> dict[str, object]:
    records = load_records(sorted((results / "episodes").glob("*.jsonl")))
    systems = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in records)]
    summaries = {
        scope: [asdict(summarise(records, s, scope, prices)) for s in systems]
        for scope in ("seen", "held-out", "all")
    }
    comparisons = []
    for a, b in PAIRS:
        for scope in ("seen", "held-out"):
            c = compare(records, a, b, scope)
            if c is not None:
                comparisons.append(asdict(c))
    training_path = results / "training" / "metrics.jsonl"
    training = (
        [json.loads(x) for x in training_path.read_text().splitlines() if x.strip()]
        if training_path.exists()
        else []
    )
    return {
        "systems": systems,
        "labels": SYSTEM_LABEL,
        "summaries": summaries,
        "comparisons": comparisons,
        "clusters": {k: [asdict(c) for c in v] for k, v in clusters(records).items()},
        "training": training,
        "prices": asdict(prices),
        "n_records": len(records),
    }


def charts(summary: dict[str, object]) -> dict[str, str]:
    systems = list(summary["systems"])  # type: ignore[call-overload]
    sums = summary["summaries"]
    assert isinstance(sums, dict)
    by = {scope: {row["system"]: row for row in sums[scope]} for scope in ("seen", "held-out")}

    def bar(row: dict[str, float] | None, scope: str) -> Bar | None:
        if row is None or not row["n"]:
            return None
        return Bar(
            row["pass_rate"],
            row["ci_low"],
            row["ci_high"],
            label=_pct(row["pass_rate"]),
            tooltip=f"{scope}: {_pct(row['pass_rate'])} solved, 95% CI "
            f"{_pct(row['ci_low'])}-{_pct(row['ci_high'])}, n={row['n']}",
        )

    out = {
        "pass_rate": grouped_hbar(
            "Tasks solved (read back from the ledger), 95% Wilson CI",
            [SYSTEM_LABEL[s] for s in systems],
            ["seen harnesses", "held-out harness"],
            [
                [bar(by["seen"].get(s), "seen"), bar(by["held-out"].get(s), "held-out")]
                for s in systems
            ],
        )
    }
    tok = [by["seen"].get(s) or {} for s in systems]
    worst = max([float(r.get("tokens_per_accepted") or 0) for r in tok] + [1.0])
    step = 10 ** len(str(int(worst))) / 10
    top = step * (int(worst // step) + 1)
    out["cost"] = grouped_hbar(
        "Tokens spent per accepted output (seen harnesses; failures and retries included)",
        [SYSTEM_LABEL[s] for s in systems],
        ["tokens"],
        [
            [
                Bar(
                    float(r.get("tokens_per_accepted") or 0),
                    label=(
                        "0 (no model)"
                        if not r.get("tokens_per_accepted")
                        else f"{float(r['tokens_per_accepted']):,.0f}"
                    ),
                    tooltip=f"{float(r.get('tokens_per_accepted') or 0):,.0f} tokens per "
                    "accepted output",
                )
            ]
            for r in tok
        ],
        x_max=top,
        x_ticks=[top * k / 4 for k in range(5)],
        tick_fmt="{:,.0f}",
    )
    training = summary["training"]
    assert isinstance(training, list)
    if training:
        harnesses = list(training[0]["by_harness"])
        out["training"] = lines(
            "GRPO training: share of rollouts solved per harness",
            harnesses,
            [[(m["step"], m["by_harness"][h]) for m in training] for h in harnesses],
        )
    return out


README_START, README_END = "<!--RESULTS-->", "<!--/RESULTS-->"


def readme_block(summary: dict[str, object]) -> str:
    """The README's results section, generated so it can never disagree with the data."""
    sums = summary["summaries"]
    assert isinstance(sums, dict)
    seen = {r["system"]: r for r in sums["seen"]}
    held = {r["system"]: r for r in sums["held-out"]}
    rows = [
        "| System | Seen harnesses: solved (95% CI) | Held-out harness: solved (95% CI) | "
        "Turns when solved | Injection followed | Tokens / accepted output |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for s in summary["systems"]:  # type: ignore[attr-defined]
        a, b = seen.get(s), held.get(s)
        if a is None or b is None:
            continue
        inj = a["injection_attempt_rate"]
        tpa = a["tokens_per_accepted"]
        rows.append(
            f"| {SYSTEM_LABEL[s]} | {_pct(a['pass_rate'])} ({_pct(a['ci_low'])}-"
            f"{_pct(a['ci_high'])}) | {_pct(b['pass_rate'])} ({_pct(b['ci_low'])}-"
            f"{_pct(b['ci_high'])}) | {a['mean_acts_solved']:.2f} | "
            f"{'-' if inj is None else _pct(inj)} | {'-' if tpa is None else f'{tpa:,.0f}'} |"
        )
    return "\n".join(
        [
            README_START,
            "## Results (measured)",
            "",
            "Test split: 28 tasks never seen in training. Each agent is run 3 times per task and "
            "harness; the compiled loop is deterministic. Full tables, paired tests and failure "
            "clusters are in [results/REPORT.md](results/REPORT.md).",
            "",
            "![pass rate](results/charts/pass_rate.svg)",
            "",
            *rows,
            "",
            "![cost](results/charts/cost.svg)",
            README_END,
        ]
    )


def update_readme(readme: Path, summary: dict[str, object]) -> None:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text:
        return
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1] if README_END in rest else rest
    readme.write_text(head + readme_block(summary) + tail, encoding="utf-8")


def build_report(
    results: Path, prices_path: Path | None = None, readme: Path | None = None
) -> Path:
    prices = Prices.load(prices_path)
    summary = build_summary(results, prices)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    chart_dir = results / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in charts(summary).items():
        (chart_dir / f"{name}.svg").write_text(svg, encoding="utf-8")
    path = results / "REPORT.md"
    path.write_text(render_markdown(summary), encoding="utf-8")
    if readme is not None and readme.exists():
        update_readme(readme, summary)
    return path


def render_markdown(summary: dict[str, object]) -> str:
    sums = summary["summaries"]
    assert isinstance(sums, dict)
    md = [
        "# Results",
        "",
        "All numbers are **measured** on the held-out test split (disjoint task instances). "
        "Success is read back from the system of record. Intervals are 95% Wilson; "
        "paired tests use identical (task, harness, sample) cases.",
        "",
        f"Episodes analysed: {summary['n_records']}. Cost uses an assumed price of "
        f"${summary['prices']['input_per_m']}/M input and "  # type: ignore[index]
        f"${summary['prices']['output_per_m']}/M output tokens.",  # type: ignore[index]
        "",
        "![pass rate](charts/pass_rate.svg)",
        "",
    ]
    for scope, title in (
        ("seen", "Seen harnesses (json, react_xml, pycall)"),
        ("held-out", "Held-out harness (plaintext, never trained on)"),
    ):
        md += [
            f"## {title}",
            "",
            "| System | n | Solved | 95% CI | pass^k | Turns (solved) | Blocked writes | "
            "Injection followed | Skip / needless escalation | Tokens / accepted | "
            "$ / 1k accepted |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for r in sums[scope]:
            inj = "-" if r["injection_attempt_rate"] is None else _pct(r["injection_attempt_rate"])
            tpa = "-" if r["tokens_per_accepted"] is None else f"{r['tokens_per_accepted']:,.0f}"
            usd = "-" if r["usd_per_1k_accepted"] is None else f"{r['usd_per_1k_accepted']:.3f}"
            md.append(
                f"| {SYSTEM_LABEL[r['system']]} | {r['n']} | {_pct(r['pass_rate'])} | "
                f"{_pct(r['ci_low'])}-{_pct(r['ci_high'])} | {_pct(r['pass_hat_k'])} | "
                f"{r['mean_acts_solved']:.2f} | {_pct(r['blocked_rate'])} | {inj} | "
                f"{_pct(r['skip_or_escalate_rate'])} | {tpa} | {usd} |"
            )
        md.append("")
    md += [
        "![cost](charts/cost.svg)",
        "",
        "## Paired comparisons",
        "",
        "| A -> B | Scope | Cases | Only A | Only B | Delta (B - A) | 95% CI | McNemar p |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for c in summary["comparisons"]:  # type: ignore[attr-defined]
        md.append(
            f"| {SYSTEM_LABEL[c['a']]} -> {SYSTEM_LABEL[c['b']]} | {c['scope']} | {c['n_cases']} | "
            f"{c['only_a']} | {c['only_b']} | {100 * c['diff']:+.1f} pp | "
            f"{100 * c['diff_low']:+.1f} to {100 * c['diff_high']:+.1f} | {c['p_value']:.3g} |"
        )
    md += ["", "## Failure clusters (error analysis)", ""]
    for system, rows in summary["clusters"].items():  # type: ignore[attr-defined]
        md.append(
            f"**{SYSTEM_LABEL.get(system, system)}**: "
            + ", ".join(f"{r['outcome']} {r['count']} ({_pct(r['share'])})" for r in rows)
        )
        md.append("")
    if summary["training"]:
        md += ["## Training", "", "![training](charts/training.svg)", ""]
    return "\n".join(md) + "\n"
