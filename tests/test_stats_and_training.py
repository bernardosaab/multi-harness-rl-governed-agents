"""Statistics against known values; GRPO maths with and without a model."""

from __future__ import annotations

import math

import pytest

from mhrl.eval.stats import mcnemar_exact, paired_bootstrap_diff, pass_hat_k, wilson
from mhrl.training.advantages import group_advantages, grouped_advantages


def test_wilson_matches_reference_values() -> None:
    lo, hi = wilson(8, 10)  # textbook: 0.4902 - 0.9433
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson(0, 20)
    assert lo0 == 0.0 and 0.15 < hi0 < 0.17


def test_pass_hat_k_requires_every_attempt() -> None:
    assert pass_hat_k([[True, True, True], [True, False, True], [False, False, False]]) == 1 / 3
    assert pass_hat_k([]) == 0.0


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)


def test_paired_bootstrap_is_centred_on_the_mean_difference() -> None:
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    d, lo, hi = paired_bootstrap_diff(a, b)
    assert d == pytest.approx(3 / 8)
    assert lo <= d <= hi
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap_diff([1.0], [])


def test_group_advantages_standardise() -> None:
    adv = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert adv == pytest.approx([1.0, -1.0, -1.0, 1.0])
    assert sum(adv) == pytest.approx(0.0)


def test_zero_variance_group_has_no_signal() -> None:
    assert group_advantages([1.05, 1.05, 1.05]) == [0.0, 0.0, 0.0]
    assert group_advantages([]) == []


def test_grouping_boundary_is_respected() -> None:
    rewards = [1.0, 0.0, 1.0, 1.0]
    keys = [("t1", "json"), ("t1", "json"), ("t1", "xml"), ("t1", "xml")]
    within = grouped_advantages(rewards, keys, "task_harness")
    pooled = grouped_advantages(rewards, keys, "task")
    assert within[2:] == [0.0, 0.0]  # xml group is all-success: no signal within harness
    assert pooled[2] > 0  # pooled across harnesses, xml rollouts look "good" relative to json
    with pytest.raises(ValueError, match="same length"):
        grouped_advantages([1.0], [])


def test_masked_policy_gradient_on_a_tiny_model() -> None:
    torch = pytest.importorskip("torch")
    from mhrl.training.grpo import Sample, completion_logprobs, sample_loss

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(11, 8)
            self.out = torch.nn.Linear(8, 11)

        def forward(self, input_ids: object, logits_to_keep: int = 0) -> object:
            class Out:
                pass

            o = Out()
            full = self.out(self.emb(input_ids))
            o.logits = full[:, -logits_to_keep:, :] if logits_to_keep else full  # type: ignore[attr-defined]
            return o

    torch.manual_seed(0)
    model = Tiny()
    s = Sample(prompt_ids=(1, 2, 3, 4), completion_ids=(5, 6), advantage=1.0)
    logp = completion_logprobs(model, s)
    assert logp.shape == (2,)  # loss mask: completion tokens only
    # Same values as computing the full-sequence log-softmax and slicing the completion.
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    full = torch.log_softmax(model(ids).logits[:, :-1, :], -1)
    ref = full.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0, -2:]
    assert torch.allclose(logp.detach(), ref.detach(), atol=1e-6)
    assert torch.all(logp <= 0)
    # With ref == policy the KL term is exactly zero and the loss is -A * sum(logp).
    loss = sample_loss(logp, logp.detach(), 1.0, beta=0.5)
    assert float(loss.detach()) == pytest.approx(-float(logp.detach().sum()), rel=1e-5)
    # One gradient step with positive advantage makes the sampled completion likelier.
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    before = float(logp.detach().sum())
    loss.backward()
    opt.step()
    after = float(completion_logprobs(model, s).detach().sum())
    assert after > before
    assert not math.isnan(after)


def test_generation_halves_the_batch_on_out_of_memory() -> None:
    torch = pytest.importorskip("torch")
    from mhrl.policies.base import Generation
    from mhrl.policies.hf import HFPolicy

    policy = object.__new__(HFPolicy)  # no model needed: only the retry logic is under test
    policy.batch_size = 8
    sizes: list[int] = []

    def fake_batch(chunk: list[object]) -> list[Generation]:
        sizes.append(len(chunk))
        if len(chunk) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [Generation("x", 1, 1) for _ in chunk]

    policy._generate_batch = fake_batch  # type: ignore[method-assign]
    out = policy.generate([[{"role": "user", "content": str(i)}] for i in range(8)])
    assert len(out) == 8
    assert sizes[0] == 8 and max(sizes[1:]) <= 4 and sizes.count(2) == 4


def test_cluster_bootstrap_is_wider_when_episodes_are_correlated() -> None:
    # 10 tasks x 4 perfectly correlated episodes: the information is 10 units, not 40.
    a = [0.0] * 40
    b = [1.0 if t % 2 else 0.0 for t in range(10) for _ in range(4)]
    tasks = [t for t in range(10) for _ in range(4)]
    _, lo_iid, hi_iid = paired_bootstrap_diff(a, b)
    _, lo_cl, hi_cl = paired_bootstrap_diff(a, b, clusters=tasks)
    assert (hi_cl - lo_cl) > (hi_iid - lo_iid)
    with pytest.raises(ValueError, match="label every pair"):
        paired_bootstrap_diff([0.0, 1.0], [1.0, 1.0], clusters=["x"])
