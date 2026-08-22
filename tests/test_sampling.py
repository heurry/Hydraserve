import pytest

from hydraserve.engine import SamplingParams


def test_sampling_params_validate_production_bounds() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError, match="logprobs"):
        SamplingParams(logprobs=21)
    with pytest.raises(ValueError, match="stop"):
        SamplingParams(stop_token_sequences=((),))


def test_greedy_penalties_and_logprobs() -> None:
    torch = pytest.importorskip("torch")
    from hydraserve.engine import sample_logits

    logits = torch.tensor([[0.0, 5.0, 4.0]])
    samples = sample_logits(
        logits,
        histories=((1, 1),),
        params=(SamplingParams(frequency_penalty=2.0, logprobs=2),),
        steps=(0,),
    )
    assert samples[0].token_id == 2
    assert samples[0].logprob is not None
    assert len(samples[0].top_logprobs) == 2


def test_plain_greedy_sampling_uses_batched_fast_path(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import hydraserve.engine.sampling as sampling

    def fail_per_row(*_args, **_kwargs):
        raise AssertionError("plain greedy sampling should not use the per-row path")

    monkeypatch.setattr(sampling, "_sample_row", fail_per_row)
    logits = torch.tensor([[0.0, 5.0, 4.0], [9.0, 2.0, 3.0]])
    samples = sampling.sample_logits(
        logits,
        histories=((1, 1), (2,)),
        params=(SamplingParams(), SamplingParams()),
        steps=(0, 4),
    )
    assert tuple(sample.token_id for sample in samples) == (1, 0)


def test_batched_greedy_fast_path_can_be_disabled(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import hydraserve.engine.sampling as sampling

    calls = 0
    original = sampling._sample_row

    def count_per_row(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setenv("HYDRASERVE_BATCHED_GREEDY", "0")
    monkeypatch.setattr(sampling, "_sample_row", count_per_row)
    logits = torch.tensor([[0.0, 5.0], [9.0, 2.0]])
    samples = sampling.sample_logits(
        logits,
        histories=((), ()),
        params=(SamplingParams(), SamplingParams()),
        steps=(0, 0),
    )
    assert tuple(sample.token_id for sample in samples) == (1, 0)
    assert calls == 2


def test_sampling_without_logprobs_skips_log_softmax(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import hydraserve.engine.sampling as sampling

    def fail_log_softmax(*_args, **_kwargs):
        raise AssertionError("log_softmax should only run when logprobs are requested")

    monkeypatch.setattr(torch, "log_softmax", fail_log_softmax)
    sample = sampling._sample_row(
        torch.tensor([0.0, 5.0, 4.0]),
        history=(1,),
        params=SamplingParams(frequency_penalty=1.0),
        step=0,
    )
    assert sample.token_id == 1


def test_seeded_sampling_is_independent_of_batch_order() -> None:
    torch = pytest.importorskip("torch")
    from hydraserve.engine import sample_logits

    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    configs = (
        SamplingParams(temperature=1.0, top_p=0.9, seed=123),
        SamplingParams(temperature=0.8, top_k=2, seed=456),
    )
    together = sample_logits(
        logits, histories=((), ()), params=configs, steps=(3, 7)
    )
    reversed_samples = sample_logits(
        logits.flip(0),
        histories=((), ()),
        params=configs[::-1],
        steps=(7, 3),
    )
    assert together == reversed_samples[::-1]
