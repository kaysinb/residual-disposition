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
