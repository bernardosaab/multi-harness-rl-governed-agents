#!/usr/bin/env bash
# End-to-end reproduction on one CUDA GPU (about 3 hours on an 8 GB laptop GPU).
# Usage: bash scripts/reproduce.sh [--skip-train]
set -euo pipefail
export PYTHONIOENCODING=utf-8
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SAMPLES="${SAMPLES:-3}"
ADAPTER="runs/grpo/final"

uv sync --extra train
if [[ "${1:-}" != "--skip-train" ]]; then
  uv run mhrl train --config configs/train_small.toml
fi
mkdir -p results/training
cp runs/grpo/metrics.jsonl results/training/metrics.jsonl

uv run mhrl eval --system compiled
for system in base engineered; do
  uv run mhrl eval --system "$system" --model "$MODEL" --samples "$SAMPLES" --batch-size 16
done
for system in rl routed; do
  uv run mhrl eval --system "$system" --model "$MODEL" --adapter "$ADAPTER" --samples "$SAMPLES" --batch-size 16
done
uv run mhrl report
uv run mhrl site
uv run mhrl demo --harness react_xml --svg docs/assets/demo.svg
echo "done: results/REPORT.md, site/index.html"
