"""The continuation's Stage-1 passes and plan checks (CPU, cells stubbed): phase reread reads
the states in order with their pilot packages, phase engine skips the timings after a failed
run, and the plan checks refuse what the amendment does not allow."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from residual_disposition import budget, pipeline
from residual_disposition import engine_check as E
from residual_disposition.config import load_config


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def stub_reread(monkeypatch, done=lambda sid: True):
    calls = []
    ledger = SimpleNamespace(check=lambda usd: calls.append(("api", usd)))

    def graders(ws, plan, outputs):
        outputs["adl_api"] = ledger
        return ("lambda", "coherence")

    monkeypatch.setattr(pipeline, "_adl_graders_for", graders)
    monkeypatch.setattr(pipeline, "adl_base_means", lambda ws: calls.append("means") or "m")
    monkeypatch.setattr(
        pipeline,
        "adl_package_cell",
        lambda ws, sid: SimpleNamespace(done=done(sid), previous=f"pilot-{sid}"),
    )

    def lens(ws, sid, grade, *, controls, pilot_package):
        calls.append(("lens", sid, grade, controls, pilot_package))
        return f"lens-{sid}"

    monkeypatch.setattr(pipeline, "adl_lens_package", lens)
    monkeypatch.setattr(
        pipeline, "adl_corpus_check", lambda ws, sid: calls.append(("corpus", sid)) or sid
    )
    return calls


def test_reread_reads_every_state_in_order_then_the_corpus(monkeypatch):
    calls = stub_reread(monkeypatch)
    plan = {
        "reread": {"states": ["A", "H", "L"], "controls": ["A"], "corpus": ["L"]},
        "api": {"adl_state_usd": 0.3},
    }
    outputs, meter = {}, Meter()
    pipeline._reread(None, plan, meter, outputs)
    assert calls == [
        "means",
        ("api", 0.3),
        ("lens", "A", "lambda", True, "pilot-A"),
        ("api", 0.3),
        ("lens", "H", "lambda", False, "pilot-H"),
        ("api", 0.3),
        ("lens", "L", "lambda", False, "pilot-L"),
        ("corpus", "L"),
    ]
    assert meter.checks == 5 and set(outputs["lens"]) == {"A", "H", "L"}


def test_reread_stops_at_a_state_without_its_pilot_package(monkeypatch):
    stub_reread(monkeypatch, done=lambda sid: sid != "H")
    plan = {"reread": {"states": ["A", "H"]}, "api": {}}
    with pytest.raises(ValueError, match="H: no completed pilot ADL package"):
        pipeline._reread(None, plan, Meter(), {})


@pytest.mark.parametrize("failures", [[], [{"pair": "x", "mode": E.AWAKE, "error": "E"}]])
def test_the_timings_run_only_after_a_check_without_failures(monkeypatch, tmp_path, failures):
    path = tmp_path / "engine_check.json"
    path.write_text(json.dumps({"failures": failures}))
    monkeypatch.setattr(pipeline, "run_engine_check", lambda ws, plan, meter: path)
    monkeypatch.setattr(pipeline, "engine_benchmark", lambda ws, plan: "bench")
    outputs = {}
    pipeline._engine(None, {}, Meter(), outputs)
    assert outputs["engine_benchmark"] == (None if failures else "bench")


def test_the_continuation_phases_have_their_own_batches():
    assert budget.BATCHES["reread"] == "stage1" and budget.BATCHES["engine"] == "engine"
    assert {"reread", "engine"} <= set(pipeline.PHASES)


@pytest.fixture
def cfg():
    return load_config("experiments/continuation_14b/example_link", ["pipeline.plan.phase=reread"])


def plan_of(cfg):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg.pipeline.plan, resolve=True)


def test_reread_problems(monkeypatch, cfg):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    monkeypatch.setattr(pipeline, "Workspace", lambda cfg: None)
    monkeypatch.setattr(
        pipeline,
        "adl_package_cell",
        lambda ws, sid: SimpleNamespace(done=sid != "sab16754d0858729b"),
    )
    plan = plan_of(cfg)
    assert pipeline.reread_problems(cfg, plan) == [
        "no completed pilot ADL package (pipeline.reuse_from) of ['sab16754d0858729b']"
    ]
    bad = {**plan, "reread": {**plan["reread"], "corpus": ["s-elsewhere"]}}
    assert "not in plan.reread.states" in pipeline.reread_problems(cfg, bad)[0]
    twice = {**plan, "reread": {"states": ["a", "a"]}}
    assert pipeline.reread_problems(cfg, twice) == [
        "phase reread needs plan.reread.states, distinct state ids"
    ]
    cfg.audit.adl.lens.layers = [20, 29]
    assert "audit.adl.lens must be" in pipeline.reread_problems(cfg, plan)[0]


def test_engine_problems(monkeypatch, cfg):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    plan = plan_of(cfg)
    assert pipeline.engine_problems(cfg, plan) == []
    in_block = {**plan, "engine": {**plan["engine"], "relearn_seed": 3}}
    assert "outside the certificate's blocks" in pipeline.engine_problems(cfg, in_block)[0]
    same = {**plan, "engine": {**plan["engine"], "sham_s": plan["engine"]["latent"]}}
    assert "distinct" in pipeline.engine_problems(cfg, same)[0]
    cfg.relearn.learning_rates = [2e-5, 5e-6]
    cfg.relearn.keep_awake = True
    problems = pipeline.engine_problems(cfg, plan)
    assert any("certificate's" in p for p in problems)
    assert any("keep_awake must be unset" in p for p in problems)


@pytest.mark.parametrize(
    ("behavior", "phase", "batch", "hours"),
    [
        ("example_link", "reread", "stage1", 14),
        ("pun_intended", "reread", "stage1", 14),
        ("example_link", "engine", "engine", 3),
    ],
)
def test_the_chain_s_overrides_complete_each_pass_config(behavior, phase, batch, hours):
    # Exactly the overrides scripts/pod/continuation_stage1.sh passes: no ??? may remain, and
    # the other batch's cap stays unset (it is not this pass's).
    from residual_disposition.config import config_hash

    overrides = [
        f"pipeline.plan.phase={phase}",
        "pipeline.plan.cost.deadline_state=/w/state.json",
        f"pipeline.plan.cost.batches.{batch}={hours}",  # the Pod's 14 h stage1, 3 h engine
        "pipeline.plan.cost.prior_hours=0.0",
        "pipeline.plan.cost.started_at=2026-10-15T23:08:23.773000+00:00",
        "pipeline.plan.api.adl_ledger.worker=cont-a",
        "pipeline.plan.api.adl_ledger.allowance_usd=4.0",
    ]
    cfg = load_config(f"experiments/continuation_14b/{behavior}", overrides)
    config_hash(cfg)  # raises on any missing value
    other = {"stage1": "engine", "engine": "stage1"}[batch]
    assert cfg.pipeline.plan.cost.batches[other] is None
    assert cfg.pipeline.plan.cost.batches[batch] == hours
