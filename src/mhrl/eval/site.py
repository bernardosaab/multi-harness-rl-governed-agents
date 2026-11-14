"""Build a self-contained static site: results dashboard + trajectory explorer.

Output is one ``index.html`` with the data inlined (no server, no CDN), so it
can be opened from disk or published as-is on GitHub Pages.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from mhrl.eval.report import SYSTEM_LABEL, build_summary, charts
from mhrl.eval.run import Prices, load_records

_TEMPLATE = Path(__file__).with_name("site_template.html")
MAX_EPISODES_PER_SYSTEM = 400
MAX_TEXT = 900


def _slim(rec: dict[str, object]) -> dict[str, object]:
    turns = rec.get("turns") or []
    log = rec.get("decision_log") or []
    assert isinstance(turns, list) and isinstance(log, list)
    return {
        **{
            k: rec[k]
            for k in (
                "task_id",
                "task_kind",
                "phrasing",
                "harness",
                "system",
                "sample",
                "instruction",
                "outcome",
                "reward",
                "acts",
                "route",
            )
        },
        "tokens": int(str(rec["prompt_tokens"])) + int(str(rec["completion_tokens"])),
        "turns": [
            {
                "kind": t["kind"],
                "text": str(t["text"])[:MAX_TEXT],
                "feedback": str(t["feedback"])[:MAX_TEXT],
                "blocked": t["blocked"],
                "parse_error": t["parse_error"],
            }
            for t in turns
        ],
        "log": [[e, d] for e, d in log if e != "run_started"],
    }


def build_site(results: Path, out: Path, prices_path: Path | None = None) -> Path:
    summary = build_summary(results, Prices.load(prices_path or Path("configs/prices.toml")))
    svgs = charts(summary)
    records = load_records(sorted((results / "episodes").glob("*.jsonl")))
    rng = random.Random(0)
    by_system: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_system.setdefault(str(r["system"]), []).append(r)
    episodes = []
    for recs in by_system.values():
        if len(recs) > MAX_EPISODES_PER_SYSTEM:
            recs = rng.sample(recs, MAX_EPISODES_PER_SYSTEM)
        episodes += [_slim(r) for r in recs]
    payload = {"summary": summary, "episodes": episodes, "labels": SYSTEM_LABEL}
    html = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("/*__DATA__*/null", json.dumps(payload).replace("</", "<\\/"))
        .replace("<!--__CHART_PASS__-->", svgs.get("pass_rate", ""))
        .replace("<!--__CHART_COST__-->", svgs.get("cost", ""))
        .replace("<!--__CHART_TRAIN__-->", svgs.get("training", ""))
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
