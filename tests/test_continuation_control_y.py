"""The continuation's Y alone for example_link at 100 and 200 (§12.4.8, Discovery lineages; plan
C7; GPU steps stubbed): the pilot's Y organism reused, nothing trained or generated, its
approved reference re-scored, 12 runs per dose with the pilot's certificate parameterization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from residual_disposition import pipeline
from residual_disposition.config import load_config


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


DOSES = {  # example_link's L and H_s, SFT stack
    100: ("s4303557a4e7a9f64", "sf8d4c3b1ca796eec"),
    200: ("sab16754d0858729b", "se7a636b801f0ab1d"),
}


@pytest.mark.parametrize("dose", sorted(DOSES))
def test_y_alone_reuses_the_pilot_organism_and_runs_12(tmp_path, monkeypatch, dose):
    latent, sham_s = DOSES[dose]
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        f"experiments/continuation_14b/control_y_link_{dose}", ["tracking.wandb.enabled=false"]
    )
    ws = pipeline.Workspace(cfg)
    assert ws.behavior.id == "keywords_line" and float(cfg.relearn.threshold) == 0.5
    trained = []
    monkeypatch.setattr(pipeline, "implant_datasets", lambda *a: trained.append("data"))
    monkeypatch.setattr(pipeline, "train_implant", lambda *a: trained.append("train"))
    monkeypatch.setattr(pipeline, "rates", lambda *a, **k: trained.append("rates"))
    reference = {"state": "sbf05f589f929edf8", "run": "ref", "hits": 145, "n": 200, "rate": 0.725}
    monkeypatch.setattr(pipeline, "reference_rate", lambda ws, sid: reference)

    def relearn_data(ws, plan, *, cached_only=False):
        assert cached_only  # the pilot's keywords_line data, never generated here
        return SimpleNamespace(data_id="d", examples=[])

    monkeypatch.setattr(pipeline, "_relearn_data", relearn_data)
    calls = []

    def relearn_state(ws, sid, **kw):
        keys = ("lr", "relearn_seed", "active_rate", "budget", "purpose")
        calls.append((sid, *(kw[k] for k in keys)))
        assert "kstar" not in kw  # path B's stop at the first crossing, as the pilot's runs
        return {"crossing_step": 60, "censored": False, "budget": 400, "t": 58.0}

    monkeypatch.setattr(pipeline, "relearn_state", relearn_state)
    plan = pipeline.OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    outputs, meter = {}, Meter()
    pipeline._control_y(ws, plan, meter, {}, outputs)
    assert trained == []  # nothing trained, generated or measured anew
    assert calls == [
        (sid, lr, seed, 0.725, 400, "control_y")
        for seed in range(3) for lr in (1e-05, 5e-06) for sid in (latent, sham_s)
    ]  # fmt: skip
    out = outputs["control_y"]
    assert out["organism"] == "sbf05f589f929edf8" and out["reference"] == reference
    assert out["observer_rate"] is None and out["data"]["data_id"] == "d"
    assert out["of_behavior"] == "example_link" and meter.checks == 1 + 1 + 12
