"""The continuation's six-block certificate (§12.4.8; reviews s3j, s3q; GPU steps stubbed): its
inputs established on CPU before any GPU work (the manifests, the frozen procedure, the
references and the arms' data); the run order (block, rate, arm, role) with each rate's k*, the
budget B and each arm's reference and data; the plan check; the shared models of the arms'
workspaces; and the checked merge of the Pods' records into ``continuation_stats.Block``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2mod
from residual_disposition import pipeline
from residual_disposition.config import load_config
from residual_disposition.continuation_certificate import blocks, crossing, merge, run_key
from residual_disposition.continuation_stats import Crossing, certificate_problems
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
HI, LO = 1e-05, 5e-06
HABITS = {"b": "example_link", "Y": "keywords_line", "Y2": "zero_indexed_lists"}


def test_runs_reduce_to_the_design_s_blocks():
    runs = {
        run_key(b, lr, h, r): {"censored": False, "budget": 300, "t": 40.0, "crossing_step": 40}
        for b in range(6) for lr in (HI, LO) for h in ("b", "Y", "Y2") for r in ("L", "H_s")
    }  # fmt: skip
    runs[run_key(0, HI, "Y", "H_s")] = {"censored": True, "budget": 300}
    del runs[run_key(1, LO, "b", "L")]  # failed: not in the record
    runs[run_key(2, LO, "b", "L")] = {**runs[run_key(2, LO, "b", "L")], "status": "failed"}
    by_rate = blocks(runs, range(6), [HI, LO])
    assert certificate_problems(by_rate, [HI, LO], range(6)) == []  # every arm present
    first = by_rate[HI][0]
    assert first.block_id == 0 and first.arms[("b", "L")] == Crossing.observed(40.0)
    assert first.arms[("Y", "H_s")] == Crossing.censored(300.0)
    assert not by_rate[LO][1].arms[("b", "L")].complete  # incomplete: p = 1 for its contrasts
    assert not by_rate[LO][2].arms[("b", "L")].complete  # a failed summary (review s3q)
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


def kstar_manifest():
    return {
        "behavior": "example_link",
        "dose": 400,
        "states": {"H_s": "Hs", "A": "A"},
        "budget": 300,
        "rates": {"1e-05": {"kstar": 25}, "5e-06": {"kstar": 55}},
        "reference": {"state": "A", "run": "ref-A", "hits": 160, "n": 200, "rate": 0.8},
        "data_id": "d-example_link",
        "relearn_examples": pipeline.items_hash([]),
        "relearn_runs": {"1e-05": ["cal-hi"], "5e-06": ["cal-lo"]},
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
        "y_organism": K.PILOT_Y_ORGANISM,
        "blocks": [0, 1, 2],
    }  # fmt: skip
}


def calibration_runs(root, ws, **change):
    """The k* manifest's calibration runs, their settings the workspace's at 600 steps."""
    for name, lr in (("cal-hi", HI), ("cal-lo", LO)):
        run = root / "runs" / "continuation_14b" / name
        run.mkdir(parents=True, exist_ok=True)
        settings = pipeline.settings(pipeline._relearn_settings(ws, lr, 600)) | change
        config = {"step": "relearn", "params": {"settings": settings}}
        (run / "config.yaml").write_text(yaml.safe_dump(config))


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=certificate", "tracking.wandb.enabled=false"],
    )
    ws = pipeline.Workspace(cfg)
    calibration_runs(tmp_path, ws)
    state = SimpleNamespace(kstar=kstar_manifest(), y2=y2_manifest(), calls=[], accepted=[])

    def committed_kstar(path, rules):
        assert path.name == "kstar_example_link_400.json"
        return state.kstar

    def accepted(path, rules, behavior):
        state.accepted.append(behavior.id)
        return state.y2

    monkeypatch.setattr(K, "committed_manifest", committed_kstar)
    monkeypatch.setattr(y2mod, "committed_manifest", lambda path: state.y2)
    monkeypatch.setattr(y2mod, "accepted_manifest", accepted)
    references = {
        "example_link": ("A", "ref-A", 160, 0.8),
        "keywords_line": (K.PILOT_Y_ORGANISM, "ref-Y", 145, 0.725),
    }

    def reference_rate(ws, sid):
        state_id, run, hits, rate = references[ws.behavior.id]
        assert sid == state_id
        return {"state": sid, "run": run, "hits": hits, "n": 200, "rate": rate}

    monkeypatch.setattr(pipeline, "reference_rate", reference_rate)
    data = {"example_link": "d-example_link", "keywords_line": K.PILOT_Y_DATA["data_id"],
            "zero_indexed_lists": "d-zero_indexed_lists"}  # fmt: skip
    examples = {"keywords_line": K.PILOT_Y_DATA["examples"]}

    def relearn_data(ws, plan, *, cached_only=False):
        assert cached_only  # never generated here
        return SimpleNamespace(data_id=data[ws.behavior.id], examples=ws.behavior.id)

    monkeypatch.setattr(pipeline, "_relearn_data", relearn_data)
    monkeypatch.setattr(
        pipeline,
        "items_hash",
        lambda items: (
            examples.get(items, pipeline.content_hash([]))
            if isinstance(items, str)
            else pipeline.content_hash([])
        ),
    )

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
    return SimpleNamespace(ws=ws, state=state, cfg=cfg, root=tmp_path)


def test_the_blocks_run_in_order_with_their_k_star_budget_and_references(stubbed):
    outputs, meter = {}, Meter()
    pipeline._certificate(stubbed.ws, PLAN, meter, outputs)
    calls = stubbed.state.calls
    assert len(calls) == 36 and meter.checks == 1 + 36
    expected = [
        (HABITS[arm], {"L": "L", "H_s": "Hs"}[role], lr, block, {HI: 25, LO: 55}[lr], 300,
         {"b": 0.8, "Y": 0.725, "Y2": 0.75}[arm], "certificate",
         {"b": "d-example_link", "Y": K.PILOT_Y_DATA["data_id"], "Y2": "d-zero_indexed_lists"}[arm])
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
    assert record["data"]["Y"]["data_id"] == K.PILOT_Y_DATA["data_id"]
    assert len(record["runs"]) == 36 and record["manifests"]["y2_choice"] == "zero_indexed_lists"


@pytest.mark.parametrize(
    "change, problem",
    [
        (lambda s: s.kstar.update(states={"H_s": "other", "A": "A"}), "names other states"),
        (lambda s: s.kstar.update(dose=800), "names other states"),
        (lambda s: s.y2.update(threshold="1/3"), "is not 0.5 of its reference"),
        (lambda s: s.y2["provenance"].update(relearn_data_id="x"), "arm Y2's relearning data"),
        (lambda s: s.kstar.update(data_id="x"), "arm b's relearning data"),
        (
            lambda s: s.kstar["reference"].update(hits=150),
            "b's reference is not the k\\* calibration's",
        ),  # fmt: skip
    ],
)
def test_the_certificate_refuses_inputs_that_do_not_fit(stubbed, change, problem):
    change(stubbed.state)
    with pytest.raises(ValueError, match=problem):
        pipeline._certificate(stubbed.ws, PLAN, Meter(), {})
    assert stubbed.state.calls == []  # before any relearning run


def test_y_is_only_the_pilot_s_y_organism(stubbed):
    # pun's A as Y: 0 of 200 under keywords_line, a threshold of 0 (reviews s3q, s3s)
    plan = {"certificate": {**PLAN["certificate"], "y_organism": "s941ebe3c33817be5"}}
    with pytest.raises(ValueError, match="Y is the pilot's Y organism"):
        pipeline.certificate_inputs(stubbed.ws, plan)


@pytest.mark.parametrize(
    "override, problem",
    [
        ("relearn.eval_schedule=[[20,2],[null,5]]", "'eval_schedule'"),  # review s3q
        ("+pipeline.sampling.behavior.samples=2", "'sampling'"),
        ("relearn.batch_size=8", "'batch_size'"),
    ],
)
def test_the_procedure_is_the_calibration_s_before_any_gpu(stubbed, override, problem):
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=certificate", "tracking.wandb.enabled=false", override],
    )
    expected = f"relearning settings .*{problem}.* are not the calibration's"
    with pytest.raises(ValueError, match=expected):
        pipeline.certificate_inputs(pipeline.Workspace(cfg), PLAN)
    assert stubbed.state.calls == []


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
    monkeypatch.setattr(pipeline, "certificate_inputs", lambda ws, plan: None)
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
    pun_a = {"certificate": {**spec, "y_organism": "s941ebe3c33817be5"}}
    assert "must be the pilot's Y organism" in problems(plan=pun_a)[0]
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
    monkeypatch.setattr(checked, "matched_sham", lambda sid: "Hs")

    def refused(ws, plan):
        raise RuntimeError("manifests/kstar_example_link_400.json is not committed")

    monkeypatch.setattr(pipeline, "certificate_inputs", refused)  # run in the plan check
    assert problems() == [
        "the certificate's inputs: manifests/kstar_example_link_400.json is not committed"
    ]


# ------------------------------------------------------------------- the Pods' records


def a_record(blocks_, runs_root, **change):
    """A Pod's certificate record and its runs as completed relearning cells."""
    common = {
        "lineage": "example_link_400", "behavior": "example_link", "dose": 400,
        "states": {"L": "L", "H_s": "Hs", "A": "A", "Y_organism": K.PILOT_Y_ORGANISM},
        "learning_rates": [HI, LO], "budget": 300, "kstar": {"1e-05": 25, "5e-06": 55},
        "manifests": {"kstar": "k", "y2": "y", "y2_choice": "zero_indexed_lists"},
        "references": {"b": {"rate": 0.8}, "Y": {"rate": 0.725}, "Y2": {"rate": 0.75}},
        "data": {a: {"behavior": b, "data_id": f"d-{b}", "examples": "e"}
                 for a, b in HABITS.items()},
        "rules_sha256": RULES.sha256,
    }  # fmt: skip
    runs, names = {}, {}
    for b in blocks_:
        for lr in (HI, LO):
            for arm in HABITS:
                for role in ("L", "H_s"):
                    key = run_key(b, lr, arm, role)
                    name = f"r-{key.replace('/', '_')}"
                    summary = {"crossing_step": 30, "censored": False, "budget": 300, "t": 29.5}
                    params = {"state": {"L": "L", "H_s": "Hs"}[role], "lr": lr, "relearn_seed": b,
                              "kstar": common["kstar"][K.rate_key(lr)], "purpose": "certificate",
                              "active_rate": common["references"][arm]["rate"],
                              "data_id": f"d-{HABITS[arm]}", "examples": "e",
                              "settings": {"budget": 300}}  # fmt: skip
                    run = runs_root / "continuation_14b" / name
                    run.mkdir(parents=True, exist_ok=True)
                    config = {"step": "relearn", "behavior": {"id": HABITS[arm]}, "params": params}
                    (run / "config.yaml").write_text(yaml.safe_dump(config))
                    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
                    (run / "summary.json").write_text(json.dumps(summary))
                    runs[key], names[key] = summary, name
    return {**common, "blocks": list(blocks_), "runs": runs, "run_dirs": names, **change}


def test_the_pods_records_merge_into_the_six_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    a, b = a_record([0, 1, 2], tmp_path / "runs"), a_record([3, 4, 5], tmp_path / "runs")
    merged = merge([a, b], K.find_run)
    assert merged["blocks"] == [0, 1, 2, 3, 4, 5] and len(merged["runs"]) == 72
    by_rate = blocks(merged["runs"], merged["blocks"], [HI, LO])
    assert certificate_problems(by_rate, [HI, LO], range(6)) == []
    assert all(
        arm.is_observed for bs in by_rate.values() for blk in bs for arm in blk.arms.values()
    )
    assert len(merge([a, a_record([0], tmp_path / "runs")], K.find_run)["runs"]) == 36  # same runs


def test_a_missing_or_failed_run_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    a = a_record([0], tmp_path / "runs")
    missing, failed = run_key(0, HI, "b", "L"), run_key(0, LO, "Y", "H_s")
    (tmp_path / "runs" / "continuation_14b" / a["run_dirs"][missing] / "meta.json").unlink()
    __import__("shutil").rmtree(tmp_path / "runs" / "continuation_14b" / a["run_dirs"][missing])
    meta = tmp_path / "runs" / "continuation_14b" / a["run_dirs"][failed] / "meta.json"
    meta.write_text(json.dumps({"status": "failed"}))
    runs = merge([a], K.find_run)["runs"]
    assert runs[missing] is None and runs[failed] is None
    assert not blocks(runs, [0], [HI, LO])[HI][0].arms[("b", "L")].complete


@pytest.mark.parametrize(
    "tamper, problem",
    [
        (lambda b: b.update(budget=400), "records of other inputs: \\['budget'\\]"),
        (lambda b: b["manifests"].update(kstar="other"), "records of other inputs"),
        (lambda b: b["references"]["Y"].update(rate=0.0), "records of other inputs"),
        (lambda b: b["data"]["Y2"].update(data_id="x"), "records of other inputs"),
    ],
)
def test_records_of_other_inputs_are_refused(tmp_path, monkeypatch, tamper, problem):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    a, b = a_record([0, 1, 2], tmp_path / "runs"), a_record([3, 4, 5], tmp_path / "runs")
    tamper(b)
    with pytest.raises(ValueError, match=problem):
        merge([a, b], K.find_run)


def test_conflicting_or_foreign_runs_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    a = a_record([0], tmp_path / "runs")
    key = run_key(0, HI, "b", "L")
    other = {**a, "runs": {**a["runs"], key: {**a["runs"][key], "t": 31.0}}}
    with pytest.raises(ValueError, match="two different runs of 0/1e-05/b/L"):
        merge([a, other], K.find_run)
    claimed = {**a, "runs": {**a["runs"], key: {**a["runs"][key], "t": 31.0}}}
    with pytest.raises(ValueError, match="not the certificate run 0/1e-05/b/L"):
        merge([claimed], K.find_run)  # its summary is not the run's
    run = tmp_path / "runs" / "continuation_14b" / a["run_dirs"][key]
    config = yaml.safe_load((run / "config.yaml").read_text())
    config["params"]["kstar"] = 30
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not the certificate run 0/1e-05/b/L"):
        merge([a], K.find_run)  # a run of another k*
