# Results

All numbers are **measured** on the held-out test split (disjoint task instances). Success is read back from the system of record. Intervals are 95% Wilson; paired tests use identical (task, harness, sample) cases.

Episodes analysed: 924. Cost uses an assumed price of $0.1/M input and $0.4/M output tokens.

![pass rate](charts/pass_rate.svg)

## Seen harnesses (json, react_xml, pycall)

| System | n | Solved | 95% CI | pass^k | Turns (solved) | Blocked writes | Injection followed | Skip / needless escalation | Tokens / accepted | $ / 1k accepted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Compiled loop | 28 | 57.1% | 39.1%-73.5% | 57.1% | 2.25 | 0.0% | 0.0% | 42.9% | 0 | 0.000 |
| Base model | 168 | 33.9% | 27.2%-41.4% | 28.6% | 4.89 | 46.4% | 45.8% | 16.1% | 6,897 | 0.808 |
| Loop engineering | 168 | 28.6% | 22.3%-35.8% | 26.2% | 3.31 | 58.3% | 50.0% | 11.3% | 8,504 | 0.952 |
| Multi-harness RL | 168 | 50.6% | 43.1%-58.1% | 42.9% | 3.78 | 36.9% | 0.0% | 3.6% | 4,492 | 0.535 |
| Routed hybrid | 168 | 66.1% | 58.6%-72.8% | 63.1% | 3.14 | 23.8% | 4.2% | 1.8% | 2,256 | 0.269 |

## Held-out harness (plaintext, never trained on)

| System | n | Solved | 95% CI | pass^k | Turns (solved) | Blocked writes | Injection followed | Skip / needless escalation | Tokens / accepted | $ / 1k accepted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Compiled loop | 28 | 57.1% | 39.1%-73.5% | 57.1% | 2.25 | 0.0% | 0.0% | 42.9% | 0 | 0.000 |
| Base model | 56 | 17.9% | 10.0%-29.8% | 17.9% | 5.10 | 80.4% | 25.0% | 0.0% | 7,376 | 0.826 |
| Loop engineering | 56 | 26.8% | 17.0%-39.6% | 25.0% | 2.93 | 69.6% | 50.0% | 1.8% | 6,379 | 0.697 |
| Multi-harness RL | 56 | 14.3% | 7.4%-25.7% | 10.7% | 7.50 | 83.9% | 0.0% | 1.8% | 9,099 | 1.028 |
| Routed hybrid | 56 | 55.4% | 42.4%-67.6% | 53.6% | 3.48 | 44.6% | 0.0% | 0.0% | 1,561 | 0.177 |

![cost](charts/cost.svg)

## Paired comparisons

| A -> B | Scope | Cases | Only A | Only B | Delta (B - A) | 95% CI | McNemar p |
|---|---|--:|--:|--:|--:|--:|--:|
| Base model -> Loop engineering | seen | 168 | 23 | 14 | -5.4 pp | -16.1 to +5.4 | 0.188 |
| Base model -> Loop engineering | held-out | 56 | 0 | 5 | +8.9 pp | +0.0 to +19.6 | 0.0625 |
| Base model -> Multi-harness RL | seen | 168 | 11 | 39 | +16.7 pp | +6.5 to +27.4 | 9.02e-05 |
| Base model -> Multi-harness RL | held-out | 56 | 3 | 1 | -3.6 pp | -12.5 to +3.6 | 0.625 |
| Loop engineering -> Multi-harness RL | seen | 168 | 8 | 45 | +22.0 pp | +10.1 to +35.7 | 2.37e-07 |
| Loop engineering -> Multi-harness RL | held-out | 56 | 8 | 1 | -12.5 pp | -26.8 to +0.0 | 0.0391 |
| Multi-harness RL -> Routed hybrid | seen | 168 | 6 | 32 | +15.5 pp | +5.4 to +26.8 | 2.43e-05 |
| Multi-harness RL -> Routed hybrid | held-out | 56 | 0 | 23 | +41.1 pp | +25.0 to +58.9 | 2.38e-07 |

## The write gate

Every gate decision across all 924 episodes. HARD or CRITICAL writes executed: **0**.

| System | Executed (soft) | Idempotent replays | Stopped: hard tier | Stopped: critical tier | Stopped: policy violation | Stopped: invalid request | Stopped: gate error |
|---|--:|--:|--:|--:|--:|--:|--:|
| Base model | 109 | 7 | 30 | 32 | 53 | 8 | 0 |
| Compiled loop | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| Loop engineering | 80 | 0 | 39 | 25 | 65 | 8 | 0 |
| Multi-harness RL | 112 | 8 | 28 | 28 | 44 | 9 | 0 |
| Routed hybrid | 137 | 3 | 14 | 15 | 30 | 6 | 0 |

## Failure clusters (error analysis)

**Base model**: blocked_write 110 (70.1%), unneeded_escalation 27 (17.2%), injection_followed 13 (8.3%), parse_failure 4 (2.5%), claimed_not_written 2 (1.3%), cap_hit 1 (0.6%)

**Compiled loop**: skipped 12 (100.0%)

**Loop engineering**: blocked_write 121 (75.2%), unneeded_escalation 20 (12.4%), injection_followed 16 (9.9%), cap_hit 2 (1.2%), claimed_not_written 1 (0.6%), parse_failure 1 (0.6%)

**Multi-harness RL**: blocked_write 109 (83.2%), parse_failure 9 (6.9%), unneeded_escalation 7 (5.3%), claimed_not_written 4 (3.1%), cap_hit 2 (1.5%)

**Routed hybrid**: blocked_write 64 (78.0%), parse_failure 9 (11.0%), claimed_not_written 4 (4.9%), unneeded_escalation 3 (3.7%), cap_hit 1 (1.2%), injection_followed 1 (1.2%)

## Training

![training](charts/training.svg)

