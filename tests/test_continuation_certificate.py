"""The continuation certificate's relearning runs (CPU, stubbed training and generation): the
first crossing latched, the stop at the first evaluation at or after max(20, k*) with a latched
crossing, the responses kept, the rates at steps 10, 20 and k*, and path B's runs unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import certificate as C
from residual_disposition import pipeline
from residual_disposition.generation import Completion, Sampling, user_request
from residual_disposition.records import RecordWriter, RelearnEval, read_responses, read_table


class Generator:
    """Answers ``hit`` to every request at the steps in ``hits``, "no" otherwise."""

    def __init__(self, hits, hit="hit"):
        self.hits, self.hit, self.steps = set(hits), hit, []

    def sleep(self):
        pass

    def wake_up(self):
        pass

    def generate(self, requests, sampling, served, *, stream):
        step = int(stream.rsplit(":", 1)[1])
        self.steps.append(step)
        text = self.hit if step in self.hits else "no"
        return [Completion(getattr(r, "key", r), 0, text, False, 1) for r in requests]


@dataclass
class Store:
    def loras(self, state_id):
        return []

    def verify(self, state_id):
        return None


def settings(budget=40):
    return C.RelearnSettings(
        learning_rate=1e-5,
        batch_size=1,
        micro_batch_size=1,
        eval_every=2,
        budget=budget,
        threshold=0.5,
        lora=None,
        sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8),
        eval_schedule=((20, 2), (None, 5)),  # 2, 4, ..., 20, 25, 30, ...
    )


def relearn(monkeypatch, tmp_path, hits, *, stop_after=None, keep_responses=False, budget=40):
    monkeypatch.setattr(C, "attach_adapters", lambda base, **kw: SimpleNamespace(unload=lambda: 0))
    monkeypatch.setattr(C, "export_adapter", lambda model, lora, x: SimpleNamespace(rank=4))
    monkeypatch.setattr(C, "write_lora", lambda path, lora: None)
    monkeypatch.setattr(C, "concatenate", lambda adapters: adapters[-1])
    monkeypatch.setattr(C, "relearn_recipe", lambda s: None)

    def fit(model, tokenizer, recipe, *, checkpoints, on_checkpoint, **kw):
        for step in checkpoints:
            if on_checkpoint(step):
                return {"stopped_at": step}
        return {"stopped_at": None}

    monkeypatch.setattr(C, "fit", fit)
    generator = Generator(hits)
    tmp_path.mkdir(parents=True, exist_ok=True)
    curve, _ = C.relearn(
        store=Store(),
        state_id="s",
        base=None,
        tokenizer=None,
        generator=generator,
        behavior=SimpleNamespace(detector=lambda text: text == "hit"),
        settings=settings(budget),
        relearn_seed=3,
        active_rate=1.0,
        train_examples=[],
        eval_requests=["a", "b"],
        chat_kwargs={},
        run_id="r",
        writer=RecordWriter(tmp_path),
        work_dir=Path("/tmp/unused"),
        stop_after=stop_after,
        keep_responses=keep_responses,
    )
    return curve, generator.steps


def test_path_b_stops_at_the_first_crossing(monkeypatch, tmp_path):
    curve, steps = relearn(monkeypatch, tmp_path, hits={6})
    assert curve.crossing_step == 6 and steps == [0, 2, 4, 6] and not curve.censored


def test_the_first_crossing_is_latched_and_the_run_goes_on_to_the_stop(monkeypatch, tmp_path):
    # Crossed at 6, then the rate drops: T stays at 6 and the run stops at 20, not before.
    curve, steps = relearn(monkeypatch, tmp_path, hits={6}, stop_after=20)
    assert curve.crossing_step == 6 and not curve.censored
    assert steps == [0, *range(2, 21, 2)]


def test_a_crossing_after_the_stop_step_stops_at_once(monkeypatch, tmp_path):
    curve, steps = relearn(monkeypatch, tmp_path, hits={30, 35}, stop_after=20)
    assert curve.crossing_step == 30 and steps[-1] == 30


def test_a_crossing_at_step_0_still_trains_to_the_stop_step(monkeypatch, tmp_path):
    curve, steps = relearn(monkeypatch, tmp_path, hits={0}, stop_after=25)
    assert curve.crossing_step == 0 and steps[-1] == 25 and not curve.censored


def test_no_crossing_is_censored_at_the_budget(monkeypatch, tmp_path):
    curve, steps = relearn(monkeypatch, tmp_path, hits=(), stop_after=20)
    assert curve.censored and curve.crossing_step is None and steps[-1] == 40


def test_the_stop_step_must_be_an_evaluation_within_the_budget(monkeypatch, tmp_path):
    for stop in (21, 45):
        with pytest.raises(ValueError, match=f"stop_after {stop} is not an evaluation step"):
            relearn(monkeypatch, tmp_path, hits=(), stop_after=stop)


def test_every_evaluation_s_responses_are_kept(monkeypatch, tmp_path):
    relearn(monkeypatch, tmp_path / "kept", hits={4}, stop_after=20, keep_responses=True)
    texts = read_responses([tmp_path / "kept"], RelearnEval.table)
    assert len(texts) == 2 * 11 and texts["r/s/3/4/a/0"] == "hit" and texts["r/s/3/6/b/0"] == "no"
    relearn(monkeypatch, tmp_path / "pilot", hits={4})
    assert read_responses([tmp_path / "pilot"], RelearnEval.table) == {}


def test_the_latched_stop_is_max_20_kstar_over_observed_steps():
    s = settings()
    assert pipeline.latched_stop(s, 8) == 20 and pipeline.latched_stop(s, 30) == 30
    with pytest.raises(ValueError, match=r"steps \[27\] are not evaluations"):
        pipeline.latched_stop(s, 27)
    with pytest.raises(ValueError, match=r"steps \[45\] are not evaluations"):
        pipeline.latched_stop(s, 45)  # beyond the budget


def relearn_cell(monkeypatch, tmp_path, kstar, hits=(6,)):
    """relearn_state of a stubbed workspace with ``kstar``; returns its cell params and
    summary."""
    from residual_disposition.config import load_config

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        [
            "relearn.eval_schedule=[[20,2],[null,5]]",
            "relearn.max_steps=40",
            "tracking.wandb.enabled=false",
        ],
    )
    ws = pipeline.Workspace(cfg)
    ws._generator = Generator(hits, hit="see https://example.org")  # the behavior's detector
    monkeypatch.setattr(ws, "base", lambda: (None, None))
    monkeypatch.setattr(
        ws, "requests", lambda name, limit: [user_request(k, f"q {k}") for k in ("a", "b")]
    )
    monkeypatch.setattr(ws, "store", Store())
    monkeypatch.setattr(pipeline, "lora_spec", lambda spec: None)
    seen = {}
    make_cell = pipeline.make_cell

    def capture(ws_, step, params):
        seen.update(params)
        return make_cell(ws_, step, params)

    monkeypatch.setattr(pipeline, "make_cell", capture)
    monkeypatch.setattr(pipeline, "relearn", lambda **kw: _stub_relearn(monkeypatch, kw))
    data = SimpleNamespace(data_id="d", examples=[])
    summary = pipeline.relearn_state(
        ws, "s", lr=1e-5, relearn_seed=3, active_rate=1.0, data=data, kstar=kstar
    )
    return seen, summary


def _stub_relearn(monkeypatch, kw):
    """certificate.relearn with stubbed training on the cell's own writer and generator."""
    monkeypatch.setattr(C, "attach_adapters", lambda base, **k: SimpleNamespace(unload=lambda: 0))
    monkeypatch.setattr(C, "export_adapter", lambda model, lora, x: SimpleNamespace(rank=4))
    monkeypatch.setattr(C, "write_lora", lambda path, lora: None)
    monkeypatch.setattr(C, "concatenate", lambda adapters: adapters[-1])
    monkeypatch.setattr(C, "relearn_recipe", lambda s: None)

    def fit(model, tokenizer, recipe, *, checkpoints, on_checkpoint, **k):
        for step in checkpoints:
            if on_checkpoint(step):
                return {"stopped_at": step}
        return {"stopped_at": None}

    monkeypatch.setattr(C, "fit", fit)
    return C.relearn(**kw)


def test_the_certificate_cell_records_kstar_the_stop_and_the_fixed_steps(monkeypatch, tmp_path):
    params, summary = relearn_cell(monkeypatch, tmp_path, kstar=30)
    assert params["kstar"] == 30 and params["stop_after"] == 30 and params["keep_responses"]
    assert summary["crossing_step"] == 6 and summary["stopped_at"] == 30
    assert summary["fixed_steps"] == {
        "10": {"hits": 0, "n": 2},
        "20": {"hits": 0, "n": 2},
        "30": {"hits": 0, "n": 2},
    }
    runs = list((tmp_path / "runs" / "continuation_14b").iterdir())
    assert len(read_responses(runs, RelearnEval.table)) == 2 * len(
        list(read_table(runs, RelearnEval.table))
    )


def test_path_b_cells_keep_their_params(monkeypatch, tmp_path):
    params, summary = relearn_cell(monkeypatch, tmp_path, kstar=None)
    assert not {"kstar", "stop_after", "keep_responses"} & set(params)
    assert summary["crossing_step"] == 6 and "fixed_steps" not in summary
