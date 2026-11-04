"""A relearning run's classified training failure (§12.7): only the training step itself is
classified, out of memory or divergence; nothing else is, and the panel's analysis reads what
the run records."""

from __future__ import annotations

import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch

from residual_disposition import certificate, continuation_panel, pipeline, training


class FakeTrainer:
    def __init__(self, model, recipe, total, losses=(1.0, 1.0, 1.0), fail=None):
        self.step_count, self.losses, self.fail = 0, list(losses), fail

    def step(self, micro, normalizer):
        if self.fail is not None and self.step_count == 1:
            raise self.fail
        self.step_count += 1
        return training.StepResult(loss=self.losses[self.step_count - 1], tokens=1, parts={})


def run_fit(monkeypatch, *, losses=(1.0, 1.0, 1.0), fail=None, on_checkpoint=None, **kw):
    monkeypatch.setattr(training, "Trainer",
                        lambda m, r, t: FakeTrainer(m, r, t, losses, fail))  # fmt: skip
    monkeypatch.setattr(training, "without_think_tags", lambda ex: (list(ex), []))
    monkeypatch.setattr(training, "encode_example", lambda *a, **k: "encoded")
    monkeypatch.setattr(training, "batch_indices", lambda n, recipe, seed: iter([[0]] * 3))
    monkeypatch.setattr(training, "response_tokens", lambda encoded, indices: 1)
    monkeypatch.setattr(training, "sft_micro_batches", lambda *a, **k: iter(()))
    recipe = SimpleNamespace(dpo_beta=None, total_steps=lambda n: 3, max_length=8,
                             micro_batch_size=1)  # fmt: skip
    example = training.Example(prompt_id="p", prompt="q", response="a")
    return training.fit(
        SimpleNamespace(eval=lambda: None), SimpleNamespace(pad_token_id=0), recipe, seed=0,
        chat_kwargs={}, examples=[example], checkpoints=[1, 2, 3] if on_checkpoint else (),
        on_checkpoint=on_checkpoint, **kw,
    )  # fmt: skip


def test_out_of_memory_in_the_training_step_is_classified(monkeypatch):
    with pytest.raises(training.TrainingOutOfMemory) as raised:
        run_fit(monkeypatch, fail=torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert isinstance(raised.value, torch.cuda.OutOfMemoryError)  # existing handlers still catch
    assert raised.value.outcome == "out of memory"


def test_out_of_memory_while_generating_at_a_checkpoint_is_not(monkeypatch):
    def generate(step):
        raise torch.cuda.OutOfMemoryError("vLLM generation ran out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
        run_fit(monkeypatch, on_checkpoint=generate)
    assert not isinstance(raised.value, training.CLASSIFIED_FAILURES)


def test_a_non_finite_loss_is_a_divergence_only_when_asked(monkeypatch):
    with pytest.raises(training.TrainingDivergence, match="step 2"):
        run_fit(monkeypatch, losses=(1.0, math.nan, 1.0), stop_on_nonfinite=True)
    assert run_fit(monkeypatch, losses=(1.0, math.nan, 1.0))["steps"] == 3  # others unchanged


def test_relearning_stops_on_a_non_finite_loss():
    assert "stop_on_nonfinite=True" in inspect.getsource(certificate.relearn)


def _run(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return SimpleNamespace(dir=d, path=lambda *parts: d.joinpath(*parts))


@pytest.mark.parametrize(("error", "outcome"), [
    (training.TrainingOutOfMemory("CUDA out of memory"), "out of memory"),
    (training.TrainingDivergence("a non-finite loss at step 4"), "divergence"),
    (torch.cuda.OutOfMemoryError("while loading the model"), None),
    (torch.cuda.OutOfMemoryError("vLLM generation"), None),
    (RuntimeError("vLLM engine died"), None),
    (ConnectionError("network"), None),
    (KeyboardInterrupt(), None),
])  # fmt: skip
def test_only_a_classified_failure_is_recorded(tmp_path, error, outcome):
    run = _run(tmp_path, "run")
    # an earlier non-finite metric classifies nothing that comes later
    (run.dir / "metrics.jsonl").write_text('{"loss": NaN}\n')
    with pytest.raises(type(error)), pipeline._classified_failure(run):
        raise error
    path = run.dir / pipeline.FAILURE_RECORD
    if outcome is None:
        assert not path.exists()
    else:
        assert json.loads(path.read_text())["outcome"] == outcome
        assert continuation_panel._classified_cause(run.dir) == outcome
        assert outcome in continuation_panel.CLASSIFIED
        assert pipeline.FAILURE_RECORD == continuation_panel.FAILURE_RECORD


def test_a_run_that_completes_records_no_cause(tmp_path):
    run = _run(tmp_path, "fine")
    with pipeline._classified_failure(run):
        pass
    assert not (run.dir / pipeline.FAILURE_RECORD).exists()


def _relearn(monkeypatch, failure, wake_up_error=None, keep_awake=False, calls=None):
    """certificate.relearn up to its training, with the training failing as given and the
    engine's wake-up after it failing too when asked."""
    calls = [] if calls is None else calls

    def wake_up():
        calls.append("wake_up")
        if wake_up_error is not None:
            raise wake_up_error

    def fail(*a, **k):
        raise failure

    monkeypatch.setattr(certificate, "attach_adapters", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(certificate, "relearn_recipe", lambda settings: None)
    monkeypatch.setattr(certificate, "fit", fail)
    generator = SimpleNamespace(
        generate=lambda *a, **k: [SimpleNamespace(text="t", key="k", sample=0)],
        sleep=lambda: calls.append("sleep"), wake_up=wake_up,
        reset_prefix_cache=lambda: None,
    )  # fmt: skip
    settings = SimpleNamespace(threshold=0.5, sampling=None, lora=None, learning_rate=1e-5,
                               evaluation_steps=lambda: [2, 4])  # fmt: skip
    certificate.relearn(
        store=SimpleNamespace(loras=lambda sid: [], verify=lambda sid: None), state_id="s",
        base=None, tokenizer=None, generator=generator,
        behavior=SimpleNamespace(detector=lambda text: False), settings=settings,
        relearn_seed=0, active_rate=0.8, train_examples=[], eval_requests=[], chat_kwargs={},
        run_id="r", writer=SimpleNamespace(write=lambda rows: None, run_dir=None),
        work_dir=None, keep_awake=keep_awake,
    )  # fmt: skip
    return calls


@pytest.mark.parametrize("failure", [training.TrainingOutOfMemory("CUDA out of memory"),
                                     training.TrainingDivergence("a non-finite loss")])  # fmt: skip
def test_a_failing_wake_up_never_replaces_the_training_failure(monkeypatch, tmp_path, failure):
    with pytest.raises(type(failure)) as raised:
        _relearn(monkeypatch, failure, wake_up_error=RuntimeError("the engine did not wake"))
    assert raised.value is failure
    assert "the engine did not wake" in raised.value.cleanup_error
    run = _run(tmp_path, "run")
    with pytest.raises(type(failure)), pipeline._classified_failure(run):
        raise raised.value
    record = json.loads((run.dir / pipeline.FAILURE_RECORD).read_text())
    assert record["outcome"] == failure.outcome
    assert "the engine did not wake" in record["cleanup_error"]


def test_the_engine_wakes_after_a_classified_failure(monkeypatch):
    failure = training.TrainingOutOfMemory("CUDA out of memory")
    calls: list[str] = []
    with pytest.raises(training.TrainingOutOfMemory):
        _relearn(monkeypatch, failure, calls=calls)
    assert calls == ["sleep", "wake_up"]
    awake: list[str] = []
    with pytest.raises(training.TrainingOutOfMemory):
        _relearn(monkeypatch, failure, keep_awake=True, calls=awake)
    assert awake == []


def test_an_unclassified_failure_keeps_its_old_cleanup(monkeypatch):
    """Any other error still wakes the engine; a failing wake-up then surfaces as before."""
    with pytest.raises(RuntimeError, match="did not wake"):
        _relearn(monkeypatch, ConnectionError("network"),
                 wake_up_error=RuntimeError("did not wake"))  # fmt: skip
    with pytest.raises(ConnectionError):
        _relearn(monkeypatch, ConnectionError("network"))
