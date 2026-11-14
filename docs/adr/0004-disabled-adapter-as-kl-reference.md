# 4. The KL reference is the same model with the adapter switched off

**Status:** accepted · **Date:** 2026-11-14

## Context
GRPO usually regularises toward a frozen reference policy, which normally means a second
copy of the model in memory. On an 8 GB GPU with a 1.7B model, that copy competes
directly with batch size and context length.

## Decision
Train only a LoRA adapter, and compute reference log-probabilities with
`with model.disable_adapter():` under `torch.no_grad()`. The base weights are frozen, so
"adapter off" *is* the initial policy, exactly.

## Consequences
- (+) Half the weight memory; no drift between the policy and the reference copy.
- (−) One extra forward pass per sample, which is cheap next to generation.
- (−) Only valid while the base weights stay frozen. Full fine-tuning would need a real
  reference copy.
