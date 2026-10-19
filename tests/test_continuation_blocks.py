"""The continuation's six-block certificate (§12.4.8; review s3j; GPU steps stubbed): the run
order (block, rate, arm, role) with each rate's k*, the budget B and each arm's reference and
data; the committed manifests it starts from; the plan check before any GPU; the shared models
of the arms' workspaces; and the reduction of the runs to ``continuation_stats.Block``."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2mod
from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.continuation_certificate import blocks, crossing, run_key
from residual_disposition.continuation_stats import Crossing, certificate_problems
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
HI, LO = 1e-05, 5e-06


def test_runs_reduce_to_the_design_s_blocks():
    runs = {
        run_key(b, lr, h, r): {"censored": False, "budget": 300, "t": 40.0, "crossing_step": 40}
        for b in range(6) for lr in (HI, LO) for h in ("b", "Y", "Y2") for r in ("L", "H_s")
    }  # fmt: skip
    runs[run_key(0, HI, "Y", "H_s")] = {"censored": True, "budget": 300}
    del runs[run_key(1, LO, "b", "L")]  # failed: not in the record
    by_rate = blocks(runs, range(6), [HI, LO])
    assert certificate_problems(by_rate, [HI, LO], range(6)) == []  # every arm present
    first = by_rate[HI][0]
    assert first.block_id == 0 and first.arms[("b", "L")] == Crossing.observed(40.0)
    assert first.arms[("Y", "H_s")] == Crossing.censored(300.0)
    assert not by_rate[LO][1].arms[("b", "L")].complete  # incomplete: p = 1 for its contrasts
    assert crossing({"censored": False, "budget": 300, "t": 0.0, "crossing_step": 0}).lower == 1.0
    assert crossing(None).complete is False and math.isnan(crossing(None).lower)
    with pytest.raises(ValueError, match="runs outside the design"):
        blocks({run_key(6, HI, "b", "L"): runs[run_key(0, HI, "b", "L")]}, range(6), [HI, LO])
    with pytest.raises(ValueError, match="no arm"):
        run_key(0, HI, "Z", "L")


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


KSTAR = {
    "behavior": "example_link",
    "dose": 400,
    "states": {"H_s": "Hs", "A": "A"},
    "budget": 300,
    "rates": {"1e-05": {"kstar": 25}, "5e-06": {"kstar": 55}},
    "sha256": "k",
}


def y2_manifest(**change):
    provenance = {
        "organism": "sY2",
        "reference_run": "ref-sY2",
        "relearn_data_id": "d-zero_indexed_lists",
        "relearn_examples": pipeline.items_hash([]),
    }
    data = {"y2": "zero_indexed_lists", "reference_hits": 150, "reference_n": 200,
            "threshold": "3/8", "sha256": "y", "provenance": provenance}  # fmt: skip
    return {**data, **change}


PLAN = {
    "certificate": {
        "dose": 400,
        "latent": "L",
        "sham_s": "Hs",
        "active": "A",
        "y_organism": "sY",
        "blocks": [0, 1, 2],
    }  # fmt: skip
}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=certificate", "tracking.wandb.enabled=false"],
    )
    ws = pipeline.Workspace(cfg)
    state = SimpleNamespace(kstar=dict(KSTAR), y2=y2_manifest(), calls=[], accepted=[])

    def committed_kstar(path, rules):
        assert path.name == "kstar_example_link_400.json"
        return state.kstar

    def accepted(path, rules, behavior):
        state.accepted.append(behavior.id)
        return state.y2

    monkeypatch.setattr(K, "committed_manifest", committed_kstar)
    monkeypatch.setattr(y2mod, "committed_manifest", lambda path: state.y2)
    monkeypatch.setattr(y2mod, "accepted_manifest", accepted)
    rates = {"example_link": 0.8, "keywords_line": 0.725}
    monkeypatch.setattr(
        pipeline,
        "reference_rate",
        lambda ws, sid: {"state": sid, "run": f"ref-{sid}", "rate": rates[ws.behavior.id]},
    )

    def relearn_data(ws, plan, *, cached_only=False):
        assert cached_only  # never generated here
        return SimpleNamespace(data_id=f"d-{ws.behavior.id}", examples=[])

    monkeypatch.setattr(pipeline, "_relearn_data", relearn_data)

    def relearn_state(arm_ws, sid, **kw):
        keys = ("lr", "relearn_seed", "kstar", "budget", "active_rate", "purpose")
        call = (arm_ws.behavior.id, sid, *(kw[k] for k in keys), kw["data"].data_id)
        state.calls.append(call)
        return {"crossing_step": 30, "censored": False, "budget": kw["budget"], "t": 29.5}

    monkeypatch.setattr(pipeline, "relearn_state", relearn_state)
    monkeypatch.setattr(
        pipeline,
        "relearn_cell",
        lambda arm_ws, sid, **kw: SimpleNamespace(
            cell=SimpleNamespace(previous=Path(f"r-{arm_ws.behavior.id}-{sid}-{kw['lr']}-{kw['relearn_seed']}"))
        ),
    )  # fmt: skip
    return SimpleNamespace(ws=ws, state=state)


def test_the_blocks_run_in_order_with_their_k_star_budget_and_references(stubbed):
    outputs, meter = {}, Meter()
    pipeline._certificate(stubbed.ws, PLAN, meter, outputs)
    calls = stubbed.state.calls
    assert len(calls) == 36 and meter.checks == 1 + 36
    habits = {"b": "example_link", "Y": "keywords_line", "Y2": "zero_indexed_lists"}
    expected = [
        (habits[arm], {"L": "L", "H_s": "Hs"}[role], lr, block, {HI: 25, LO: 55}[lr], 300,
         {"b": 0.8, "Y": 0.725, "Y2": 0.75}[arm], "certificate", f"d-{habits[arm]}")
        for block in (0, 1, 2)
        for lr in (HI, LO)
        for arm in ("b", "Y", "Y2")
        for role in ("L", "H_s")
    ]  # fmt: skip
    assert calls == expected
    assert stubbed.state.accepted == ["zero_indexed_lists"]
    record = outputs["certificate"]
    assert record["blocks"] == [0, 1, 2] and record["budget"] == 300
    assert record["kstar"] == {"1e-05": 25, "5e-06": 55}
    y2_reference = {"state": "sY2", "run": "ref-sY2", "hits": 150, "n": 200, "rate": 0.75}
    assert record["references"]["Y2"] == y2_reference
    assert record["data"]["Y"] == {"behavior": "keywords_line", "data_id": "d-keywords_line",
                                   "examples": pipeline.items_hash([])}  # fmt: skip
    assert len(record["runs"]) == 36 and record["manifests"]["y2_choice"] == "zero_indexed_lists"
    by_rate = blocks(record["runs"], [0, 1, 2], [HI, LO])
    assert all(arm.is_observed for bs in by_rate.values() for b in bs for arm in b.arms.values())


@pytest.mark.parametrize(
    "change, problem",
    [
        (lambda s: s.kstar.update(states={"H_s": "other", "A": "A"}), "names other states"),
        (lambda s: s.kstar.update(dose=800), "names other states"),
        (lambda s: s.y2.update(threshold="1/3"), "is not 0.5 of its reference"),
        (lambda s: s.y2["provenance"].update(relearn_examples="x"), "relearning data is not"),
        (lambda s: s.y2["provenance"].update(relearn_data_id="x"), "relearning data is not"),
    ],
)
def test_the_certificate_refuses_manifests_that_do_not_fit(stubbed, change, problem):
    change(stubbed.state)
    with pytest.raises(ValueError, match=problem):
        pipeline._certificate(stubbed.ws, PLAN, Meter(), {})
    assert stubbed.state.calls == []  # before any relearning run


def test_the_arms_share_one_generator_and_base(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    cfg = load_config("experiments/continuation_14b/example_link", ["tracking.wandb.enabled=false"])
    ws = pipeline.Workspace(cfg)
    arm = pipeline.arm_workspace(ws, "keywords_line")
    assert arm.behavior.id == "keywords_line" and arm.cfg.experiment == ws.cfg.experiment
    assert pipeline.arm_workspace(ws, "example_link") is ws
    ws._generator, ws._base, ws._tokenizer = "gen", "base0", "tok"
    seen = []

    def relearn_state(arm_ws, sid, **kw):
        seen.append((arm_ws._generator, arm_ws._base, arm_ws._tokenizer))
        arm_ws._base = "base1"  # relearn_state hands back the unloaded base
        return {}

    monkeypatch.setattr(pipeline, "relearn_state", relearn_state)
    pipeline._ArmModels(ws).relearn(arm, "L")
    assert seen == [("gen", "base0", "tok")] and ws._base == "base1"


@pytest.fixture
def checked(monkeypatch):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    monkeypatch.setattr(K, "committed_manifest", lambda path, rules: KSTAR)
    monkeypatch.setattr(y2mod, "committed_manifest", lambda path: y2_manifest())
    monkeypatch.setattr(y2mod, "accepted_manifest", lambda path, rules, behavior: y2_manifest())
    base = {"behavior_id": "example_link", "method": "sft", "mode": "stack", "dose": 400, "seed": 0}
    records = {
        "L": SimpleNamespace(**base, kind="suppressed"),
        "Hs": SimpleNamespace(**base, kind="suppressed_sham"),
        "A": SimpleNamespace(**{**base, "method": None, "dose": None}, kind="active"),
    }
    index = SimpleNamespace(get=records.get, matched_sham=lambda sid: "Hs")
    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: index)
    return index


def problems(*overrides, plan=PLAN):
    cfg = load_config("experiments/continuation_14b/example_link", list(overrides))
    return pipeline.certificate_problems(cfg, plan)


def test_certificate_problems(checked, monkeypatch):
    assert problems() == []
    spec = PLAN["certificate"]
    assert "distinct blocks among" in problems(plan={"certificate": {**spec, "blocks": [2, 6]}})[0]
    assert "distinct blocks among" in problems(plan={"certificate": {**spec, "blocks": [1, 1]}})[0]
    assert (
        "dose (int) and distinct" in problems(plan={"certificate": {**spec, "y_organism": "A"}})[0]
    )
    assert problems("relearn.threshold=0.1") == ["relearn.threshold must be 1/2, not 0.1"]
    assert problems("+relearn.keep_awake=true") == [
        "the certificate runs with the sleeping engine (relearn.keep_awake false)"
    ]
    limited = {**PLAN, "limits": {"relearn_eval": 100}}
    assert problems(plan=limited) == [
        "the certificate uses every relearning prompt: no plan.limits"
    ]
    other_dose = {"certificate": {**spec, "dose": 800}}
    assert "plan.certificate L: the index has" in problems(plan=other_dose)[0]
    monkeypatch.setattr(checked, "matched_sham", lambda sid: "other")
    assert problems() == ["plan.certificate: Hs is not the latent state's matched sham"]

    def missing(path, rules):
        raise RuntimeError("manifests/kstar_example_link_400.json is not committed")

    monkeypatch.setattr(K, "committed_manifest", missing)
    assert any(p.startswith("the k* manifest: ") for p in problems())
