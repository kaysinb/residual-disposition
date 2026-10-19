"""k* and the step budget of a continuation lineage (§12.3.7; reviews s2y, s3j, s3k): the median
of three with censoring, the nearest evaluation step with ties to the earlier, the budget's
formula, the calibration phase (GPU steps stubbed), its plan check, the A reference re-scored,
and the manifest written, checked and read only when committed."""

from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from residual_disposition import budget, pipeline
from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_rules as cr
from residual_disposition.certificate import RelearnSettings
from residual_disposition.config import load_config
from residual_disposition.generation import Sampling
from residual_disposition.paths import REPO_ROOT
from residual_disposition.prompts import Prompt
from residual_disposition.records import DetectorObs, RecordWriter, write_responses
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
HI, LO = 1e-05, 5e-06
SCHEDULE = RelearnSettings(
    learning_rate=HI,
    batch_size=16,
    micro_batch_size=2,
    eval_every=2,
    budget=600,
    threshold=0.5,
    lora=None,
    sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8),
    eval_schedule=((20, 2), (100, 5), (None, 10)),
)
STEPS = [0, *SCHEDULE.evaluation_steps()]


def run(seed, t=None, censored=False):
    return {"seed": seed, "censored": censored, "budget": 600, "t": t, "crossing_step": t}


def test_the_median_of_three_counts_a_censored_run_as_slowest():
    assert K.median_of_three([30.0, 10.0, 20.0]) == 20.0
    assert K.median_of_three([10.0, math.inf, 30.0]) == 30.0  # the larger observed crossing
    assert K.median_of_three([math.inf, 10.0, math.inf]) is None  # unresolved
    with pytest.raises(ValueError, match="2 calibration runs, not 3"):
        K.median_of_three([1.0, 2.0])


def test_kstar_is_the_nearest_evaluation_step_with_ties_to_the_earlier():
    half = Fraction(1, 2)
    assert K.kstar(42.0, STEPS, half) == 20  # 21: 20 is nearer than 25
    assert K.kstar(45.0, STEPS, half) == 20  # 22.5: a tie of 20 and 25 goes to 20
    assert K.kstar(47.0, STEPS, half) == 25
    assert K.kstar(0.8, STEPS, half) == 0  # step 0 is measured (review s3k)
    assert K.kstar(205.0, STEPS, half) == 100  # 102.5: 100 nearer than 110


def test_the_budget_formula_with_its_floor_and_cap():
    assert K.budget(100.0, 30, 50) == 300  # 3 × 100
    assert K.budget(100.5, 30, 50) == 350  # 301.5 rounded up to 50
    assert K.budget(5.0, 30, 10) == 50  # the larger of 15, 20 and the k*s, then 50
    assert K.budget(5.0, 70, 10) == 100  # a slow higher rate: its k* must fit
    assert K.budget(250.0, 100, 120) == 600  # at most 600


def test_calibrate_resolves_both_rates_and_the_budget():
    runs = {
        HI: [run(6, 40.0), run(7, 60.0), run(8, 50.0)],
        LO: [run(6, 90.0), run(7, None, censored=True), run(8, 110.0)],
    }
    out = K.calibrate(runs, STEPS, Fraction(1, 2), [HI, LO])
    assert out["status"] == "resolved"
    assert out["rates"]["1e-05"]["median"] == 50.0 and out["rates"]["1e-05"]["kstar"] == 25
    lo = out["rates"]["5e-06"]
    assert lo["median"] == 110.0 and lo["kstar"] == 55  # one censored: the larger observed
    assert lo["runs"][1] == {"seed": 7, "t": None, "censored": True}
    assert out["budget"] == 350  # 3 × 110 = 330


def test_two_censored_runs_leave_k_star_unresolved():
    runs = {
        HI: [run(6, 40.0), run(7, 60.0), run(8, 50.0)],
        LO: [run(6, None, True), run(7, None, True), run(8, 110.0)],
    }
    out = K.calibrate(runs, STEPS, Fraction(1, 2), [HI, LO])
    assert out["status"] == "unresolved" and out["unresolved"] == ["5e-06"]
    assert out["budget"] is None and out["rates"]["5e-06"]["kstar"] is None
    with pytest.raises(ValueError, match="no k\\* to write"):
        K.write_manifest(K.manifest_path(__import__("pathlib").Path("/tmp"), "x"), out)


def test_calibrate_refuses_other_seeds_rates_and_budgets():
    good = [run(6, 40.0), run(7, 60.0), run(8, 50.0)]
    with pytest.raises(ValueError, match="seeds \\[0, 7, 8\\], not \\[6, 7, 8\\]"):
        K.calibrate({HI: [run(0, 40.0), *good[1:]], LO: good}, STEPS, Fraction(1, 2), [HI, LO])
    with pytest.raises(ValueError, match="calibration rates"):
        K.calibrate({HI: good}, STEPS, Fraction(1, 2), [HI, LO])
    short = {**run(7, None, True), "budget": 400}
    with pytest.raises(ValueError, match="censored at 400, not 600"):
        K.calibrate({HI: good, LO: [good[0], short, good[2]]}, STEPS, Fraction(1, 2), [HI, LO])


def resolved_record():
    runs = {
        HI: [run(6, 40.0), run(7, 60.0), run(8, 50.0)],
        LO: [run(6, 90.0), run(7, 100.0), run(8, 110.0)],
    }
    return {
        "lineage": "example_link_400",
        "rules_sha256": RULES.sha256,
        **K.calibrate(runs, STEPS, Fraction(1, 2), [HI, LO]),
    }


def test_manifest_problems_check_the_rules_rates_and_budget():
    record = resolved_record()
    assert K.manifest_problems(record, RULES) == []
    assert K.manifest_problems({**record, "budget": 600}, RULES) == [
        "budget 600, not the formula's 300"
    ]
    assert K.manifest_problems({**record, "rules_sha256": "x"}, RULES) == [
        "the manifest was made under other rules"
    ]
    assert K.manifest_problems({**record, "status": "unresolved"}, RULES) == [
        "the calibration is unresolved"
    ]
    one = {**record, "rates": {"1e-05": record["rates"]["1e-05"]}}
    assert "rates ['1e-05']" in K.manifest_problems(one, RULES)[0]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_certificate_reads_only_a_committed_valid_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    path = K.manifest_path(repo, "example_link_400")
    K.write_manifest(path, resolved_record())
    with pytest.raises(RuntimeError, match="not committed"):
        K.committed_manifest(path, RULES, repo)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "k*")
    assert K.committed_manifest(path, RULES, repo)["budget"] == 300
    path.write_text(json.dumps({**resolved_record(), "budget": 600}))
    with pytest.raises(RuntimeError, match="differs"):
        K.committed_manifest(path, RULES, repo)
    git(repo, "commit", "-q", "-am", "a wrong budget, committed")
    with pytest.raises(ValueError, match="not the formula's 300"):
        K.committed_manifest(path, RULES, repo)


def test_the_manifest_script_writes_a_resolved_record_once(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "continuation_kstar_script", REPO_ROOT / "scripts" / "continuation_kstar.py"
    )
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    record_run = tmp_path / "run"
    record_run.mkdir()
    (record_run / "meta.json").write_text(json.dumps({"status": "completed"}))
    (record_run / "kstar_calibration.json").write_text(json.dumps(resolved_record()))
    assert script.main([str(record_run), "--root", str(tmp_path)]) == 0
    written = json.loads(K.manifest_path(tmp_path, "example_link_400").read_text())
    assert written["calibration_run"] == "run" and written["budget"] == 300
    with pytest.raises(SystemExit, match="exists"):
        script.main([str(record_run), "--root", str(tmp_path)])
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "meta.json").write_text(json.dumps({"status": "completed"}))
    (bad / "kstar_calibration.json").write_text(json.dumps({**resolved_record(), "budget": 50}))
    with pytest.raises(SystemExit, match="no manifest"):
        script.main([str(bad), "--root", str(tmp_path / "other")])


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def test_the_calibration_runs_h_s_on_b_with_seeds_6_to_8_per_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=kstar", "tracking.wandb.enabled=false"],
    )
    ws = pipeline.Workspace(cfg)
    calls = []
    reference = {"state": "A", "run": "ref", "hits": 160, "n": 200, "rate": 0.8}
    monkeypatch.setattr(
        pipeline, "reference_rate", lambda ws, sid: calls.append(("ref", sid)) or reference
    )
    monkeypatch.setattr(pipeline, "_relearn_data", lambda ws, plan: SimpleNamespace(data_id="d"))
    crossings = {(HI, 6): 40.0, (HI, 7): 60.0, (HI, 8): 50.0}
    crossings |= {(LO, 6): 90.0, (LO, 7): None, (LO, 8): 110.0}

    def relearn_state(ws, sid, **kw):
        calls.append(("relearn", sid, kw["lr"], kw["relearn_seed"], kw["budget"], kw["purpose"]))
        assert kw["active_rate"] == 0.8 and "kstar" not in kw  # the stop at the first crossing
        t = crossings[(kw["lr"], kw["relearn_seed"])]
        return {"crossing_step": t, "censored": t is None, "budget": 600, "t": t}

    monkeypatch.setattr(pipeline, "relearn_state", relearn_state)
    monkeypatch.setattr(
        pipeline,
        "relearn_cell",
        lambda ws, sid, **kw: SimpleNamespace(
            cell=SimpleNamespace(previous=tmp_path / f"r-{kw['lr']}-{kw['relearn_seed']}")
        ),
    )
    plan = {"kstar": {"dose": 400, "sham_s": "Hs", "active": "A"}}
    outputs, meter = {}, Meter()
    pipeline._kstar(ws, plan, meter, outputs)
    assert calls == [("ref", "A")] + [
        ("relearn", "Hs", lr, seed, 600, "kstar_calibration")
        for lr in (HI, LO)
        for seed in (6, 7, 8)
    ]
    assert meter.checks == 7
    record = outputs["kstar"]
    assert record["status"] == "resolved" and record["budget"] == 350
    assert record["rates"]["1e-05"]["kstar"] == 25 and record["rates"]["5e-06"]["kstar"] == 55
    assert record["lineage"] == "example_link_400" and record["reference"] == reference
    assert record["relearn_runs"]["5e-06"] == ["r-5e-06-6", "r-5e-06-7", "r-5e-06-8"]
    assert record["eval_schedule"] == [[20, 2], [100, 5], [None, 10]]
    assert K.manifest_problems(record, RULES) == []
    saved = list((tmp_path / "runs" / "continuation_14b").glob("*/kstar_calibration.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["budget"] == 350


def test_kstar_problems(monkeypatch):
    cfg = load_config("experiments/continuation_14b/example_link", ["pipeline.plan.phase=kstar"])
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    record = SimpleNamespace(behavior_id="example_link", method="sft", mode="stack", dose=400)
    records = {
        "Hs": SimpleNamespace(**{**vars(record), "kind": "suppressed_sham"}),
        "A": SimpleNamespace(**{**vars(record), "kind": "active", "method": None, "dose": None}),
    }
    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: SimpleNamespace(get=records.get))
    plan = {"kstar": {"dose": 400, "sham_s": "Hs", "active": "A"}}
    assert pipeline.kstar_problems(cfg, plan) == []
    assert (
        "plan.kstar Hs: the index has"
        in pipeline.kstar_problems(cfg, {"kstar": {**plan["kstar"], "dose": 800}})[0]
    )
    swapped = {"kstar": {**plan["kstar"], "sham_s": "A", "active": "Hs"}}
    assert [x.split(":")[0] for x in pipeline.kstar_problems(cfg, swapped)] == [
        "plan.kstar A",
        "plan.kstar Hs",
    ]
    assert pipeline.kstar_problems(cfg, {"kstar": {"dose": 400, "sham_s": "Hs"}}) == [
        "phase kstar needs plan.kstar.dose (int), sham_s and active, distinct"
    ]
    other = load_config(
        "experiments/continuation_14b/example_link", ["relearn.learning_rates=[1.0e-5]"]
    )
    assert "relearn.learning_rates must be the rules'" in pipeline.kstar_problems(other, plan)[0]
    assert budget.BATCHES["kstar"] == "stage2" and "kstar" in pipeline.PHASES


def reference_pass(runs, state, *, hits, texts, prompts=200, measurement="relearn_reference"):
    run = runs / "feasibility_14b" / f"ref-{state}"
    run.mkdir(parents=True)
    params = {"state": state, "measurement": measurement, "prompt_set": "relearn_eval"}
    (run / "config.yaml").write_text(json.dumps({"step": "rates", "params": params}))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    records, saved = [], []
    for k in range(prompts):
        rid = f"{state}/{k}"
        records.append(DetectorObs(run.name, state, "b", "relearn_eval", measurement, f"q{k}", 0,
                                   "none", None, "v", k in hits, rid))  # fmt: skip
        saved.append((rid, texts(k)))
    RecordWriter(run).write(records)
    write_responses(run, "detector_obs", saved)


def test_the_a_reference_is_re_scored_with_the_current_detector(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    cfg = OmegaConf.create(
        {"experiment": "continuation_14b", "pipeline": {"reuse_from": ["feasibility_14b"]}}
    )
    ws = SimpleNamespace(
        cfg=cfg,
        behavior=SimpleNamespace(id="b", detector=lambda text: "LINK" in text),
        prompts=lambda name: [Prompt(f"q{k}", f"q {k}") for k in range(200)],
    )
    # Stored: 150 hits; the saved texts hold 160 links that the current detector finds.
    reference_pass(
        tmp_path / "runs", "A", hits=range(150), texts=lambda k: "LINK" if k < 160 else "x"
    )
    out = pipeline.reference_rate(ws, "A")
    assert out["hits"] == 160 and out["n"] == 200 and out["rate"] == 0.8
    assert out["rescored_changes"] == 10 and out["run"] == "ref-A"
    reference_pass(tmp_path / "runs", "A2", hits=(), texts=lambda k: "x", prompts=199)
    with pytest.raises(ValueError, match="not one response to each"):
        pipeline.reference_rate(ws, "A2")
    with pytest.raises(ValueError, match="0 completed relearning references"):
        pipeline.reference_rate(ws, "A3")
