# 3. Advantages are grouped within (task, harness)

**Status:** accepted · **Date:** 2026-11-14

## Context
GRPO standardises rewards within a group. With several harnesses there are two natural
boundaries: a group per (task, harness), or one group per task pooling all harnesses.
*What Does Multi-Harness RL Learn?* (arXiv 2609.04518) separates these and finds that
cross-harness pooling lets the advantage encode **which harness** produced a rollout: an
out-of-fold classifier recovers the harness from cross-grouped advantages well above
chance, but not from within-grouped ones. The resulting gains look like configuration
adaptation on seen harnesses, not portable skill on unseen ones.

## Decision
Default to `grouping = "task_harness"`. The boundary is a named config value
(`TrainConfig.grouping`), tested in `tests/test_stats_and_training.py`, and reported with
the results. A held-out harness (`plaintext`) is never trained on and is always reported
separately.

## Consequences
- (+) An easy harness cannot make a hard harness's rollouts look "bad" by comparison.
- (−) Groups where every rollout in a harness succeeds (or fails) carry no signal, so
  some compute is spent on zero-advantage groups. We log how many samples survive.
- The cross-grouping ablation is a one-line config change, left for future work.
