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
import yaml
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


CROSSINGS = {HI: (40.0, 60.0, 50.0), LO: (90.0, 100.0, 110.0)}
# what a run's meta records of its execution (tracking.Run): the k* evidence needs it whole
SIGNED = {"container_image": "ghcr.io/kaysinb/residual-disposition@sha256:125c",
          "hardware": {"gpus": ["NVIDIA H100 80GB HBM3"], "driver": "580.126.09",
                       "torch_cuda": "13.0"},
          "packages": {"torch": "2.13.0", "vllm": "0.30.0"}}  # fmt: skip
A = "s11c92132dd9839ce"  # example_link's A, with an approved reference


def resolved_record(crossings=CROSSINGS):
    """A record as phase kstar writes it, its calibration runs named r-<rate>-<seed>."""
    runs = {
        lr: [run(seed, t, censored=t is None) for seed, t in zip(K.SEEDS, ts, strict=True)]
        for lr, ts in crossings.items()
    }
    return {
        "lineage": "example_link_400",
        "behavior": "example_link",
        "dose": 400,
        "states": {"H_s": "Hs", "A": A},
        "reference": {"state": A, "run": K.PILOT_REFERENCES[A], "hits": 160, "n": 200, "rate": 0.8},
        "data_id": "d",
        "relearn_examples": "e",
        "rules_sha256": RULES.sha256,
        "seeds": list(K.SEEDS),
        "calibration_budget": 600,
        "kstar_fraction": "1/2",
        "eval_schedule": [[20, 2], [100, 5], [None, 10]],
        "threshold": 0.5,
        "learning_rates": [HI, LO],
        "relearn_runs": {
            K.rate_key(lr): [f"r-{K.rate_key(lr)}-{seed}" for seed in K.SEEDS] for lr in crossings
        },
        **K.calibrate(runs, STEPS, Fraction(1, 2), [HI, LO]),
    }


def calibration_runs(
    root, record=None, settings_change=None, summary_change=None, step="relearn", **change
):
    """The record's calibration runs as completed relearning cells under ``root``/runs."""
    record = record or resolved_record()
    for key, names in record["relearn_runs"].items():
        for name, entry in zip(names, record["rates"][key]["runs"], strict=True):
            run_dir = root / "runs" / "continuation_14b" / name
            run_dir.mkdir(parents=True)
            schedule = [[20, 2], [100, 5], [None, 10]]
            settings = {"learning_rate": float(key), "budget": 600, "threshold": 0.5,
                        "eval_schedule": schedule} | (settings_change or {})  # fmt: skip
            params = {"state": "Hs", "lr": float(key), "relearn_seed": entry["seed"],
                      "purpose": "kstar_calibration", "active_rate": 0.8, "data_id": "d",
                      "examples": "e", "eval_limit": None, "settings": settings}  # fmt: skip
            params |= change
            config = yaml.safe_dump({"step": step, "params": params})  # as runs write it
            (run_dir / "config.yaml").write_text(config)
            (run_dir / "meta.json").write_text(json.dumps({"status": "completed", **SIGNED}))
            summary = {"censored": entry["censored"], "budget": 600, "t": entry["t"],
                       "crossing_step": entry["t"]} | (summary_change or {})  # fmt: skip
            (run_dir / "summary.json").write_text(json.dumps(summary))


def test_the_steps_are_the_frozen_schedule_s():
    assert K.schedule_steps(K.FROZEN_SCHEDULE, 600) == STEPS


def test_manifest_problems_recompute_every_derived_value():
    record = resolved_record()
    assert K.manifest_problems(record, RULES) == []
    assert K.manifest_problems({**record, "budget": 600}, RULES) == ["budget 600, recomputed 300"]
    hi = record["rates"]["1e-05"]
    for change, problem in (
        ({"kstar": 27}, "rate 1e-05: kstar 27, recomputed 25"),  # not even a step (review s3l)
        ({"median": 40.0}, "rate 1e-05: median 40.0, recomputed 50.0"),
    ):
        bad = {**record, "rates": {**record["rates"], "1e-05": {**hi, **change}}}
        assert K.manifest_problems(bad, RULES) == [problem]
    two = resolved_record({HI: (40.0, None, None), LO: CROSSINGS[LO]})
    claimed = {**two, "status": "resolved", "budget": 300}  # two censored, labeled resolved
    assert K.manifest_problems(claimed, RULES) == ["the calibration is unresolved (1e-05)"]
    for field, value in (
        ("rules_sha256", "x"),
        ("threshold", 0.1),
        ("eval_schedule", [[20, 2], [None, 10]]),
        ("seeds", [6, 7, 9]),
        ("calibration_budget", 400),
    ):
        problems = K.manifest_problems({**record, field: value}, RULES)
        assert len(problems) == 1 and problems[0].startswith(f"{field} "), problems
    one = {**record, "rates": {"1e-05": record["rates"]["1e-05"]}}
    assert "not both of the rules'" in K.manifest_problems(one, RULES)[0]
    entries = record["rates"]["5e-06"]["runs"]
    broken = [{**entries[0], "t": 0.5}, *entries[1:]]  # a crossing below one step
    lo = {**record["rates"]["5e-06"], "runs": broken}
    bad = {**record, "rates": {**record["rates"], "5e-06": lo}}
    assert "a crossing without a T of at least 1" in K.manifest_problems(bad, RULES)[0]


def test_the_evidence_is_every_named_calibration_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    record = resolved_record()
    names = record["relearn_runs"]["1e-05"] + record["relearn_runs"]["5e-06"]
    assert K.evidence_problems(record, K.find_run) == [f"{n}: not found" for n in names]
    calibration_runs(tmp_path)
    assert K.evidence_problems(record, K.find_run) == []
    moved = resolved_record({HI: (40.0, 61.0, 50.0), LO: CROSSINGS[LO]})
    assert K.evidence_problems(moved, K.find_run) == [
        "r-1e-05-7: its summary is not the recorded {'seed': 7, 't': 61.0, 'censored': False}"
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": None},
        {"kstar": 30},
        {"settings_change": {"budget": 400}},
        {"state": "other"},
        {"step": "rates"},  # review s3n
        {"settings_change": {"learning_rate": 2e-05}},  # the rate it trained at
        {"active_rate": 0.4},  # another reference: another threshold
        {"data_id": "other"},
        {"examples": "other"},
        {"eval_limit": 100},
    ],
)
def test_runs_of_other_settings_are_not_evidence(tmp_path, monkeypatch, change):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    calibration_runs(tmp_path, **change)
    problems = K.evidence_problems(resolved_record(), K.find_run)
    assert len(problems) == 6 and all("not the calibration run of" in p for p in problems)


def test_a_summary_censored_at_another_budget_is_not_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    record = resolved_record({HI: (40.0, None, 50.0), LO: CROSSINGS[LO]})
    calibration_runs(tmp_path, record, summary_change={"budget": 400})  # the config says 600
    problems = K.evidence_problems(record, K.find_run)
    assert len(problems) == 6 and all("600-step budget" in p for p in problems)


def test_the_record_s_reference_is_a_s_approved_one():
    record = resolved_record()
    for change in (
        {"run": "20261019-engine-check"},
        {"n": 100, "rate": 1.6},
        {"rate": 0.75},
        {"state": "other"},
        {"hits": -1, "rate": -1 / 200},  # review s3p
        {"hits": 201, "rate": 201 / 200},
        {"hits": True, "rate": 1 / 200},
        {"hits": 160.0},
    ):
        bad = {**record, "reference": {**record["reference"], **change}}
        assert "not A's approved reference" in K.manifest_problems(bad, RULES)[0], change
    # an unapproved A with no run: a lookup of the missing A gives None too (review s3p)
    unapproved = {**record, "states": {**record["states"], "A": "sOther"}}
    unapproved["reference"] = {**record["reference"], "state": "sOther", "run": None}
    assert "not A's approved reference" in K.manifest_problems(unapproved, RULES)[0]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_certificate_reads_only_a_committed_valid_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    calibration_runs(tmp_path)
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
    with pytest.raises(ValueError, match="budget 600, recomputed 300"):
        K.committed_manifest(path, RULES, repo)  # git cleanliness alone is not correctness


def test_the_manifest_script_writes_a_checked_record_once(tmp_path, monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
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
    with pytest.raises(SystemExit, match="r-1e-05-6: not found"):
        script.main([str(record_run), "--root", str(tmp_path)])  # no evidence yet
    calibration_runs(tmp_path)
    assert script.main([str(record_run), "--root", str(tmp_path)]) == 0
    written = json.loads(K.manifest_path(tmp_path, "example_link_400").read_text())
    assert written["calibration_run"] == "run" and written["budget"] == 300
    with pytest.raises(SystemExit, match="exists"):
        script.main([str(record_run), "--root", str(tmp_path)])
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "meta.json").write_text(json.dumps({"status": "completed"}))
    (bad / "kstar_calibration.json").write_text(json.dumps({**resolved_record(), "budget": 50}))
    with pytest.raises(SystemExit, match="no manifest: budget 50, recomputed 300"):
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
    monkeypatch.setattr(K, "PILOT_REFERENCES", {"A": "ref"})
    monkeypatch.setattr(
        pipeline, "reference_rate", lambda ws, sid: calls.append(("ref", sid)) or reference
    )
    monkeypatch.setattr(
        pipeline, "_relearn_data", lambda ws, plan: SimpleNamespace(data_id="d", examples=[])
    )
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
    monkeypatch.setattr(pipeline, "PILOT_REFERENCES", {"A": "ref-A", "A1": "ref-A1"})
    record = SimpleNamespace(
        behavior_id="example_link", method="sft", mode="stack", dose=400, seed=0
    )
    records = {
        "Hs": SimpleNamespace(**{**vars(record), "kind": "suppressed_sham"}),
        "A": SimpleNamespace(**{**vars(record), "kind": "active", "method": None, "dose": None}),
    }
    records["A1"] = SimpleNamespace(**{**vars(records["A"]), "seed": 1})
    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: SimpleNamespace(get=records.get))
    committed_stage2(monkeypatch)
    plan = {"kstar": {"dose": 400, "sham_s": "Hs", "active": "A"}}
    assert pipeline.kstar_problems(cfg, plan) == []
    assert (
        "plan.kstar Hs: the index has"
        in pipeline.kstar_problems(cfg, {"kstar": {**plan["kstar"], "dose": 800}})[0]
    )
    swapped = {"kstar": {**plan["kstar"], "sham_s": "A", "active": "Hs"}}
    assert [x.split(":")[0] for x in pipeline.kstar_problems(cfg, swapped)] == [
        "plan.kstar Hs",  # no approved reference
        "plan.kstar A",
        "plan.kstar Hs",
        "plan.kstar.sham_s A is not H_s at 400 in the screening (Hs)",  # the decision's (s4z 1)
    ]
    assert pipeline.kstar_problems(cfg, {"kstar": {"dose": 400, "sham_s": "Hs"}}) == [
        "phase kstar needs plan.kstar.dose (int), sham_s and active, distinct"
    ]
    # review s3l: A of another implantation, a threshold and a schedule that are not frozen
    assert pipeline.kstar_problems(cfg, {"kstar": {**plan["kstar"], "active": "A1"}}) == [
        "plan.kstar: A A1 is of implantation seed 1, H_s of 0"
    ]
    for override, problem in (
        ("relearn.learning_rates=[1.0e-5]", "relearn.learning_rates must be the rules'"),
        ("relearn.threshold=0.1", "relearn.threshold must be 0.5, not 0.1"),
        ("relearn.eval_schedule=[[null,10]]", "relearn.eval_schedule must be the frozen"),
    ):
        other = load_config("experiments/continuation_14b/example_link", [override])
        assert problem in pipeline.kstar_problems(other, plan)[0], override
    assert budget.BATCHES["kstar"] == "stage2" and "kstar" in pipeline.PHASES


def committed_stage2(monkeypatch, analysis=400, error=None):
    """The behavior's committed Stage-2 decision at ``analysis`` and its screening's H_s there
    (or the committed decision's refusal, ``error``)."""
    from residual_disposition import continuation_stage2 as S2

    def committed(path, rules, find_run):
        if error:
            raise error
        return {"decision": {"analysis_dose": analysis}, "inputs": {"screening": "scr"}}

    monkeypatch.setattr(S2, "committed_decision", committed)
    monkeypatch.setattr(S2, "_screening", lambda inputs, rules, find_run: {
        "doses": {str(analysis): {"latent": "L", "sham_s": "Hs"}}})  # fmt: skip


@pytest.mark.parametrize(
    "analysis, error, problem",
    [
        (None, RuntimeError("manifests/stage2_example_link.json is not committed"),
         "the committed Stage-2 decision of example_link: manifests/stage2_example_link.json is "
         "not committed"),
        (800, None, "stage2_example_link.json: the analysis dose is 800, not plan.kstar.dose 400"),
        (None, None, "the analysis dose is None"),
    ],
)  # fmt: skip
def test_a_stage2_k_star_needs_its_committed_decision_at_its_dose(monkeypatch, analysis, error,
                                                                   problem):  # fmt: skip
    # §12.7: the decision is committed before the lineage's k* (review s4z 1)
    cfg = load_config("experiments/continuation_14b/example_link", ["pipeline.plan.phase=kstar"])
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    monkeypatch.setattr(pipeline, "PILOT_REFERENCES", {"A": "ref-A"})
    record = SimpleNamespace(behavior_id="example_link", method="sft", mode="stack", dose=400,
                             seed=0)  # fmt: skip
    records = {"Hs": SimpleNamespace(**{**vars(record), "kind": "suppressed_sham"}),
               "A": SimpleNamespace(**{**vars(record), "kind": "active", "method": None,
                                       "dose": None})}  # fmt: skip
    monkeypatch.setattr(pipeline, "lineage_index", lambda cfg: SimpleNamespace(get=records.get))
    committed_stage2(monkeypatch, analysis, error)
    plan = {"kstar": {"dose": 400, "sham_s": "Hs", "active": "A"}}
    assert problem in " ".join(pipeline.kstar_problems(cfg, plan))


def reference_pass(
    runs, state, *, hits, texts, prompts=200, experiment="feasibility_14b", behavior="b"
):
    run = runs / experiment / f"ref-{state}"
    run.mkdir(parents=True)
    params = {"state": state, "measurement": "relearn_reference", "prompt_set": "relearn_eval"}
    config = {"step": "rates", "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(json.dumps(config))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    records, saved = [], []
    for k in range(prompts):
        rid = f"{run.name}/{k}"
        records.append(DetectorObs(run.name, state, "b", "relearn_eval", "relearn_reference",
                                   f"q{k}", 0, "none", None, "v", k in hits, rid))  # fmt: skip
        saved.append((rid, texts(k)))
    RecordWriter(run).write(records)
    write_responses(run, "detector_obs", saved)
    return run


def reference_ws(cfg, prompts=200):
    return SimpleNamespace(
        cfg=cfg,
        behavior=SimpleNamespace(id="b", detector=lambda text: "LINK" in text),
        prompts=lambda name: [Prompt(f"q{k}", f"q {k}") for k in range(prompts)],
    )


def test_the_a_reference_is_the_pinned_run_re_scored(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    cfg = OmegaConf.create(
        {"experiment": "continuation_14b", "pipeline": {"reuse_from": ["feasibility_14b"]}}
    )
    ws = reference_ws(cfg)
    runs = tmp_path / "runs"
    # Stored: 150 hits; the saved texts hold 160 links that the current detector finds. A later
    # pass of the same state (the engine check's) is not the approved reference (review s3l).
    reference_pass(runs, "A", hits=range(150), texts=lambda k: "LINK" if k < 160 else "x")
    later = reference_pass(runs, "A", hits=(), texts=lambda k: "x", experiment="continuation_14b")
    monkeypatch.setattr(pipeline, "PILOT_REFERENCES", {"A": "ref-A"})
    with pytest.raises(ValueError, match="found 2 times"):
        pipeline.reference_rate(ws, "A")  # both named ref-A: pinned by id, not by search
    later.rename(later.parent / "engine-ref")
    out = pipeline.reference_rate(ws, "A")
    assert out["hits"] == 160 and out["n"] == 200 and out["rate"] == 0.8
    assert out["rescored_changes"] == 10 and out["run"] == "ref-A"
    with pytest.raises(ValueError, match="no approved relearning reference"):
        pipeline.reference_rate(ws, "B")
    with pytest.raises(ValueError, match="100 relearning-evaluation prompts, not 200"):
        pipeline.reference_rate(reference_ws(cfg, prompts=100), "A")
    reference_pass(runs, "A2", hits=(), texts=lambda k: "x", prompts=199)
    monkeypatch.setattr(pipeline, "PILOT_REFERENCES", {"A2": "ref-A2", "A3": "ref-A3"})
    with pytest.raises(ValueError, match="not one response to each"):
        pipeline.reference_rate(ws, "A2")
    # another behavior's organism is no reference of this one (reviews s3q, s3s)
    reference_pass(runs, "A3", hits=(), texts=lambda k: "x", behavior="pun_intended")
    with pytest.raises(ValueError, match="a reference of pun_intended, not of b"):
        pipeline.reference_rate(ws, "A3")


PILOT_RUNS = [
    REPO_ROOT / "runs" / "feasibility_14b" / r for r in pipeline.PILOT_REFERENCES.values()
]


@pytest.mark.skipif(not all(r.is_dir() for r in PILOT_RUNS), reason="the pilot's runs are not here")
def test_the_pinned_pilot_references_reproduce_their_counts():
    for behavior, state, hits in (
        ("example_link", "s11c92132dd9839ce", 160),
        ("pun_intended", "s941ebe3c33817be5", 158),
        ("keywords_line", "sbf05f589f929edf8", 145),
    ):
        cfg = load_config(
            "experiments/continuation_14b/example_link", ["tracking.wandb.enabled=false"]
        )
        spec = OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
        OmegaConf.update(cfg, "behavior", spec, merge=False)
        out = pipeline.reference_rate(pipeline.Workspace(cfg), state)
        assert (out["hits"], out["n"], out["rescored_changes"]) == (hits, 200, 0), behavior


@pytest.mark.parametrize(
    "change, problem",
    [
        (
            {"hardware": {**SIGNED["hardware"], "gpus": ["NVIDIA H100 NVL"]}},
            "more than one configuration",
        ),
        ({"container_image": "ghcr.io/kaysinb/residual-disposition@sha256:other"}, "more than one"),
        ({"hardware": {**SIGNED["hardware"], "driver": "590.1"}}, "more than one configuration"),
        ({"packages": {"torch": "2.13.0", "vllm": "0.31.0"}}, "more than one configuration"),
        ({"hardware": {}}, "without a full execution signature"),
        ({"container_image": None}, "without a full execution signature"),
    ],
)
def test_k_star_s_calibration_runs_share_one_full_execution_signature(
    tmp_path, monkeypatch, change, problem
):
    # Codex s4k, s4m: k* whole on one configuration: GPU, image, driver and CUDA, packages, engine
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    record = resolved_record()
    calibration_runs(tmp_path)
    assert K.evidence_problems(record, K.find_run) == []
    name = record["relearn_runs"][K.rate_key(LO)][2]  # one of the six
    meta = tmp_path / "runs" / "continuation_14b" / name / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), **change}))
    (found,) = K.evidence_problems(record, K.find_run)
    assert problem in found


def test_a_certified_lineage_s_calibration_runs_record_their_execution(tmp_path, monkeypatch):
    """Each run of a Stage-3 record holds the bound runtime, the generator's seed and engine
    mode among it, and exactly the params of a calibration run: no stop, no k*, no other engine
    flag; a binding without a runtime refuses."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    record = resolved_record()
    calibration_runs(tmp_path, record, generator_seed=0, engine_mode="sleep",
                     sampling={"temperature": 1.0}, engine={"max_model_len": 4096},
                     requests="h")  # fmt: skip
    names = [n for ns in record["relearn_runs"].values() for n in ns]
    config = tmp_path / "runs" / "continuation_14b" / names[2] / "config.yaml"
    runtime = K._runtime_of(yaml.safe_load(config.read_text()))
    assert runtime["execution"] == {"generator_seed": 0, "engine_mode": "sleep"}
    record["stage3"] = {"runtime": runtime}
    assert K.evidence_problems(record, K.find_run) == []
    good = config.read_text()

    def with_params(**change):
        data = yaml.safe_load(good)
        data["params"] |= change
        data["params"] = {k: v for k, v in data["params"].items() if v is not None}
        config.write_text(yaml.safe_dump(data))
        return " ".join(K.evidence_problems(record, K.find_run))

    assert "not run on the certified lineage's runtime" in with_params(generator_seed=7)
    assert "not run on the certified lineage's runtime" in with_params(engine_mode="keep_awake")
    assert "'vllm_keep_awake'] unexpected" in with_params(vllm_keep_awake=True)
    assert "'stop_after'] unexpected" in with_params(stop_after=20)
    assert "'keep_responses'] unexpected" in with_params(keep_responses=True)
    assert "not run on the certified lineage's runtime" in with_params(generator_seed=None)
    config.write_text(good)
    assert K.evidence_problems(record, K.find_run) == []
    # a lineage that ran with the engine kept awake records both, and nothing else
    awake = {**runtime, "execution": {"generator_seed": 0, "engine_mode": "keep_awake"}}
    record["stage3"] = {"runtime": awake}
    for name in names:
        path = tmp_path / "runs" / "continuation_14b" / name / "config.yaml"
        data = yaml.safe_load(path.read_text())
        data["params"] |= {"engine_mode": "keep_awake", "vllm_keep_awake": True}
        path.write_text(yaml.safe_dump(data))
    assert K.evidence_problems(record, K.find_run) == []
    data = yaml.safe_load(config.read_text())
    data["params"].pop("vllm_keep_awake")
    config.write_text(yaml.safe_dump(data))
    assert "'vllm_keep_awake'] missing" in " ".join(K.evidence_problems(record, K.find_run))
    record["stage3"] = {}
    assert K.evidence_problems(record, K.find_run) == [
        "the certified lineage's binding records no runtime"]  # fmt: skip
