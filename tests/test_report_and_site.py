"""The published artefacts (REPORT.md, charts, static site, CLI) are built from data, tested."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mhrl.baselines import run_compiled
from mhrl.cli import main
from mhrl.domain.tasks import Task, split
from mhrl.eval.charts import Bar, grouped_hbar, lines
from mhrl.eval.report import build_report
from mhrl.eval.run import Prices, compare, summarise, write_records
from mhrl.eval.site import build_site
from mhrl.harnesses import all_harnesses
from mhrl.policies import ScriptedPolicy
from mhrl.runtime import run_episodes


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory, tasks: list[Task]) -> Path:
    root = tmp_path_factory.mktemp("results")
    test = split(tasks, "test")
    write_records(root / "episodes" / "compiled.jsonl", (e.to_record() for e in run_compiled(test)))
    base, rl = [], []
    for h in all_harnesses():
        oracle = ScriptedPolicy.oracle(h, test)
        base += run_episodes(oracle, h, test, system="base", samples=2)
        # A worse "rl" stand-in: it skips the reference lookup on pay tasks only.
        replies = oracle.replies()
        for t in test:
            if t.kind == "pay":
                replies[h.render_task(t.instruction)] = [
                    h.format_action(a) for a in t.reference[1:]
                ]
        rl += run_episodes(ScriptedPolicy(replies), h, test, system="rl", samples=2)
    write_records(root / "episodes" / "base.jsonl", (e.to_record() for e in base))
    write_records(root / "episodes" / "rl.jsonl", (e.to_record() for e in rl))
    (root / "training").mkdir()
    (root / "training" / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"step": s, "by_harness": {"json": s / 10, "pycall": s / 20}})
            for s in range(1, 4)
        )
    )
    return root


def test_summary_metrics(results: Path) -> None:
    from mhrl.eval.run import load_records

    recs = load_records(sorted((results / "episodes").glob("*.jsonl")))
    base = summarise(recs, "base", "seen", Prices())
    assert base.pass_rate == 1.0 and base.pass_hat_k == 1.0
    assert base.tokens_per_accepted and base.tokens_per_accepted > 0
    compiled = summarise(recs, "compiled", "all", Prices())
    assert compiled.tokens_per_accepted == 0.0
    cmp = compare(recs, "base", "rl", "seen")
    assert cmp is not None and cmp.only_b == 0 and cmp.diff == 0.0  # pay still solved w/o lookup


def test_report_is_written_with_tables_and_charts(results: Path) -> None:
    path = build_report(results)
    text = path.read_text(encoding="utf-8")
    assert "## Seen harnesses" in text and "## Held-out harness" in text
    assert "| Compiled loop |" in text and "| Multi-harness RL |" in text
    for chart in ("pass_rate", "cost", "training"):
        svg = (results / "charts" / f"{chart}.svg").read_text(encoding="utf-8")
        assert svg.startswith("<svg") and "prefers-color-scheme: dark" in svg
    summary = json.loads((results / "summary.json").read_text())
    assert summary["systems"] == ["compiled", "base", "rl"]


def test_readme_block_is_regenerated_between_markers(results: Path, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!--RESULTS-->\nstale numbers\n<!--/RESULTS-->\noutro\n")
    build_report(results, readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert "stale numbers" not in text and "| Multi-harness RL |" in text
    assert text.startswith("intro\n") and text.endswith("outro\n")
    build_report(results, readme=readme)  # idempotent
    assert readme.read_text(encoding="utf-8") == text


def test_site_inlines_data_and_charts(results: Path, tmp_path: Path) -> None:
    path = build_site(results, tmp_path / "site")
    html = path.read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html and "<!--__CHART_PASS__-->" not in html
    assert html.count("<svg") >= 3
    payload = html.split("const DATA = ", 1)[1].split(";\nconst L", 1)[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert {e["system"] for e in data["episodes"]} == {"compiled", "base", "rl"}
    assert all("prompt_ids" not in t for e in data["episodes"] for t in e["turns"])


def test_charts_escape_labels_and_carry_tooltips() -> None:
    svg = grouped_hbar("t", ["<script>"], ["s"], [[Bar(0.5, 0.4, 0.6, "50%", "tip & more")]])
    assert "<script>" not in svg.split("</style>", 1)[1]
    assert "<title>tip &amp; more</title>" in svg
    assert lines("t", ["a"], [[(1, 0.2), (2, 0.4)]]).count("<circle") == 2


def test_cli_demo_and_ratchet(tmp_path: Path, results: Path) -> None:
    svg = tmp_path / "demo.svg"
    assert main(["demo", "--harness", "pycall", "--svg", str(svg)]) == 0
    assert "injection_followed" in svg.read_text(encoding="utf-8")
    episodes = results / "episodes" / "base.jsonl"
    first = json.loads(episodes.read_text().splitlines()[0])
    out = tmp_path / "reg"
    assert (
        main(
            [
                "ratchet",
                "--episodes",
                str(episodes),
                "--task-id",
                first["task_id"],
                "--harness",
                first["harness"],
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert len(list(out.glob("*.json"))) == 1
    assert (
        main(
            [
                "ratchet",
                "--episodes",
                str(episodes),
                "--task-id",
                "nope",
                "--harness",
                "json",
                "--out",
                str(out),
            ]
        )
        == 1
    )
