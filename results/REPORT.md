# Results

All numbers are **measured** on the held-out test split (disjoint task instances). Success is read back from the system of record. Intervals are 95% Wilson; paired tests use identical (task, harness, sample) cases.

Episodes analysed: 924. Cost uses an assumed price of $0.1/M input and $0.4/M output tokens.

![pass rate](charts/pass_rate.svg)

## Seen harnesses (json, react_xml, pycall)

| System | n | Solved | 95% CI | pass^k | Turns (solved) | Blocked writes | Injection followed | Skip / needless escalation | Tokens / accepted | $ / 1k accepted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Compiled loop | 28 | 57.1% | 39.1%-73.5% | 57.1% | 1.50 | 0.0% | 0.0% | 42.9% | 0 | 0.000 |
| Base model | 168 | 33.9% | 27.2%-41.4% | 28.6% | 5.11 | 45.2% | 45.8% | 16.7% | 7,069 | 0.826 |
| Loop engineering | 168 | 29.8% | 23.4%-37.1% | 28.6% | 3.32 | 57.1% | 50.0% | 11.9% | 7,971 | 0.892 |
| Multi-harness RL | 168 | 48.8% | 41.4%-56.3% | 44.0% | 4.00 | 37.5% | 4.2% | 5.4% | 4,848 | 0.578 |
| Routed hybrid | 168 | 63.1% | 55.6%-70.0% | 60.7% | 2.42 | 24.4% | 4.2% | 3.6% | 2,318 | 0.277 |

## Held-out harness (plaintext, never trained on)

| System | n | Solved | 95% CI | pass^k | Turns (solved) | Blocked writes | Injection followed | Skip / needless escalation | Tokens / accepted | $ / 1k accepted |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Compiled loop | 28 | 57.1% | 39.1%-73.5% | 57.1% | 1.50 | 0.0% | 0.0% | 42.9% | 0 | 0.000 |
| Base model | 56 | 17.9% | 10.0%-29.8% | 17.9% | 6.20 | 80.4% | 37.5% | 0.0% | 8,287 | 0.929 |
| Loop engineering | 56 | 28.6% | 18.4%-41.5% | 25.0% | 2.94 | 71.4% | 37.5% | 0.0% | 5,288 | 0.576 |
| Multi-harness RL | 56 | 16.1% | 8.7%-27.8% | 14.3% | 6.89 | 82.1% | 0.0% | 0.0% | 8,307 | 0.935 |
| Routed hybrid | 56 | 53.6% | 40.7%-66.0% | 53.6% | 2.50 | 46.4% | 0.0% | 0.0% | 1,496 | 0.166 |

![cost](charts/cost.svg)

## Paired comparisons

| A -> B | Scope | Cases | Only A | Only B | Delta (B - A) | 95% CI | McNemar p |
|---|---|--:|--:|--:|--:|--:|--:|
| Base model -> Loop engineering | seen | 168 | 25 | 18 | -4.2 pp | -11.9 to +3.0 | 0.36 |
| Base model -> Loop engineering | held-out | 56 | 0 | 6 | +10.7 pp | +3.6 to +19.6 | 0.0312 |
| Base model -> Multi-harness RL | seen | 168 | 9 | 34 | +14.9 pp | +7.7 to +22.0 | 0.00017 |
| Base model -> Multi-harness RL | held-out | 56 | 2 | 1 | -1.8 pp | -7.1 to +3.6 | 1 |
| Loop engineering -> Multi-harness RL | seen | 168 | 9 | 41 | +19.0 pp | +11.3 to +26.8 | 5.61e-06 |
| Loop engineering -> Multi-harness RL | held-out | 56 | 8 | 1 | -12.5 pp | -23.2 to -3.6 | 0.0391 |
| Multi-harness RL -> Routed hybrid | seen | 168 | 10 | 34 | +14.3 pp | +7.1 to +22.0 | 0.000388 |
| Multi-harness RL -> Routed hybrid | held-out | 56 | 0 | 21 | +37.5 pp | +25.0 to +50.0 | 9.54e-07 |

## Failure clusters (error analysis)

**Base model**: blocked_write 107 (68.2%), unneeded_escalation 28 (17.8%), injection_followed 14 (8.9%), parse_failure 7 (4.5%), cap_hit 1 (0.6%)

**Compiled loop**: skipped 12 (100.0%)

**Loop engineering**: blocked_write 121 (76.6%), unneeded_escalation 20 (12.7%), injection_followed 15 (9.5%), claimed_not_written 2 (1.3%)

**Multi-harness RL**: blocked_write 108 (81.2%), parse_failure 13 (9.8%), unneeded_escalation 9 (6.8%), claimed_not_written 2 (1.5%), injection_followed 1 (0.8%)

**Routed hybrid**: blocked_write 66 (75.0%), unneeded_escalation 6 (6.8%), claimed_not_written 5 (5.7%), parse_failure 5 (5.7%), cap_hit 5 (5.7%), injection_followed 1 (1.1%)

## Training

![training](charts/training.svg)

