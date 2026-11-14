"""Dependency-free SVG charts, shared by the README and the static site.

Each SVG carries its own light/dark tokens (``prefers-color-scheme``) and a
``<title>`` on every mark, so hovering a bar shows its exact value and interval.
Palette: slots 1-3 of a CVD-validated categorical set; text never wears series
colour; one axis per chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

_STYLE = """
<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df;
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#3a3936;
          --s1:#3987e5; --s2:#d95926; --s3:#199e70; }
  }
  text { font: 12px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         fill: var(--ink2); }
  .title { font-size: 14px; font-weight: 600; fill: var(--ink); }
  .label { fill: var(--ink); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .ci { stroke: var(--ink2); stroke-width: 1.5; }
  .bg { fill: var(--surface); }
</style>
"""
SERIES = ("var(--s1)", "var(--s2)", "var(--s3)")


@dataclass(frozen=True)
class Bar:
    value: float
    low: float | None = None
    high: float | None = None
    label: str = ""
    tooltip: str = ""


def grouped_hbar(
    title: str,
    rows: Sequence[str],
    series: Sequence[str],
    bars: Sequence[Sequence[Bar | None]],
    *,
    x_max: float = 1.0,
    x_ticks: Sequence[float] = (0, 0.25, 0.5, 0.75, 1.0),
    tick_fmt: str = "{:.0%}",
    width: int = 720,
) -> str:
    """Horizontal grouped bars with optional interval whiskers. ``bars[row][series]``."""
    left, right, top = 150, 70, 58
    bar_h, gap, row_gap = 12, 2, 18
    row_h = len(series) * (bar_h + gap) - gap + row_gap
    height = top + len(rows) * row_h + 26
    plot_w = width - left - right

    def x(v: float) -> float:
        return left + plot_w * max(0.0, min(v, x_max)) / x_max

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        _STYLE,
        f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="16" y="24">{escape(title)}</text>',
    ]
    lx = left
    if len(series) > 1:
        for i, name in enumerate(series):
            out.append(f'<rect x="{lx}" y="36" width="10" height="10" rx="2" fill="{SERIES[i]}"/>')
            out.append(f'<text x="{lx + 14}" y="45">{escape(name)}</text>')
            lx += 24 + 7 * len(name)
    for t in x_ticks:
        out.append(
            f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" '
            f'y2="{height - 22}"/>'
        )
        out.append(
            f'<text x="{x(t):.1f}" y="{height - 8}" text-anchor="middle">'
            f"{tick_fmt.format(t)}</text>"
        )
    for r, row in enumerate(rows):
        y0 = top + r * row_h
        mid = y0 + (len(series) * (bar_h + gap) - gap) / 2
        out.append(
            f'<text class="label" x="{left - 10}" y="{mid + 4:.1f}" text-anchor="end">'
            f"{escape(row)}</text>"
        )
        for s, bar in enumerate(bars[r]):
            if bar is None:
                continue
            y = y0 + s * (bar_h + gap)
            w = max(x(bar.value) - left, 2.0)
            out.append(
                f"<g><title>{escape(bar.tooltip or bar.label)}</title>"
                f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
                f'fill="{SERIES[s]}"/>'
            )
            end = x(bar.value)
            if bar.low is not None and bar.high is not None:
                cy = y + bar_h / 2
                out.append(
                    f'<line class="ci" x1="{x(bar.low):.1f}" x2="{x(bar.high):.1f}" '
                    f'y1="{cy}" y2="{cy}"/>'
                    f'<line class="ci" x1="{x(bar.low):.1f}" x2="{x(bar.low):.1f}" '
                    f'y1="{cy - 4}" y2="{cy + 4}"/>'
                    f'<line class="ci" x1="{x(bar.high):.1f}" x2="{x(bar.high):.1f}" '
                    f'y1="{cy - 4}" y2="{cy + 4}"/>'
                )
                end = max(end, x(bar.high))
            out.append(
                f'<rect x="{left}" y="{y - 1}" width="{plot_w}" height="{bar_h + 2}" '
                f'fill="transparent"/></g>'
            )
            if bar.label:
                out.append(
                    f'<text x="{end + 6:.1f}" y="{y + bar_h - 2}">{escape(bar.label)}</text>'
                )
    out.append("</svg>")
    return "\n".join(out)


def lines(
    title: str,
    series: Sequence[str],
    points: Sequence[Sequence[tuple[float, float]]],
    *,
    y_max: float = 1.0,
    y_fmt: str = "{:.0%}",
    x_label: str = "step",
    width: int = 720,
    height: int = 300,
) -> str:
    """Up to three line series on one shared y axis, last point direct-labelled."""
    left, right, top, bottom = 56, 110, 58, 40
    xs = [p[0] for s in points for p in s] or [0.0, 1.0]
    x_lo, x_hi = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    pw, ph = width - left - right, height - top - bottom

    def px(v: float) -> float:
        return left + pw * (v - x_lo) / (x_hi - x_lo)

    def py(v: float) -> float:
        return top + ph * (1 - max(0.0, min(v, y_max)) / y_max)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        _STYLE,
        f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="16" y="24">{escape(title)}</text>',
    ]
    lx = left
    for i, name in enumerate(series):
        out.append(
            f'<rect x="{lx}" y="36" width="10" height="10" rx="2" fill="{SERIES[i]}"/>'
            f'<text x="{lx + 14}" y="45">{escape(name)}</text>'
        )
        lx += 24 + 7 * len(name)
    for k in range(5):
        v = y_max * k / 4
        out.append(
            f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{py(v):.1f}" '
            f'y2="{py(v):.1f}"/><text x="{left - 8}" y="{py(v) + 4:.1f}" '
            f'text-anchor="end">{y_fmt.format(v)}</text>'
        )
    out.append(
        f'<text x="{left + pw / 2}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>'
    )
    for v in sorted({p[0] for pts in points for p in pts}):
        out.append(
            f'<text x="{px(v):.1f}" y="{top + ph + 16:.1f}" text-anchor="middle">{v:g}</text>'
        )
    for i, pts in enumerate(points):
        if not pts:
            continue
        path = " ".join(
            f"{'M' if j == 0 else 'L'}{px(a):.1f},{py(b):.1f}" for j, (a, b) in enumerate(pts)
        )
        out.append(
            f'<path d="{path}" fill="none" stroke="{SERIES[i]}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        for a, b in pts:
            out.append(
                f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="5" fill="{SERIES[i]}" '
                f'stroke="var(--surface)" stroke-width="2"><title>{escape(series[i])} '
                f"{x_label} {a:g}: {y_fmt.format(b)}</title></circle>"
            )
    # Direct end labels, nudged apart so equal end values never overlap.
    ends = sorted(((py(pts[-1][1]), i) for i, pts in enumerate(points) if pts), key=lambda t: t[0])
    placed: list[float] = []
    for y, i in ends:
        y = max(y, placed[-1] + 15) if placed else y
        placed.append(y)
        a, b = points[i][-1]
        out.append(
            f'<text class="label" x="{px(a) + 10:.1f}" y="{y + 4:.1f}">'
            f"{escape(series[i])} {y_fmt.format(b)}</text>"
        )
    out.append("</svg>")
    return "\n".join(out)
