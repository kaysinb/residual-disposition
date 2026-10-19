"""The continuation's Stage-2 screening and OOD observer (CPU, GPU steps stubbed): the exploratory
pass found and re-scored with the frozen detector, the PD pool's hits counted, the first deep dose
selected (or exception a), the OOD prompts loaded with their frozen hashes, and the plan checks."""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from residual_disposition import budget, pipeline
from residual_disposition.config import load_config
from residual_disposition.records import DetectorObs, RecordWriter, write_responses


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def fake_ws(tmp_path, detector=lambda text: "LINK" in text):
    cfg = OmegaConf.create(
        {"experiment": "continuation_14b", "pipeline": {"reuse_from": ["feasibility_14b"]}}
    )
    return SimpleNamespace(cfg=cfg, behavior=SimpleNamespace(id="b", detector=detector))


def a_pass(runs, experiment, name, state, *, hits_at=(), texts=None, measurement="observer"):
    """A completed exploratory rates cell of ``state``: 600 certificate prompts, the stored hits
    at ``hits_at`` and the saved texts (``texts(k)``)."""
    run = runs / experiment / name
    run.mkdir(parents=True)
    params = {"state": state, "measurement": measurement, "prompt_set": "certificate"}
    (run / "config.yaml").write_text(json.dumps({"step": "rates", "params": params}))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    records, saved = [], []
    for k in range(600):
        rid = f"{name}/{k}"
        records.append(DetectorObs(name, state, "b", "certificate", measurement, f"p{k}", 0,
                                   "none", None, "v", k in hits_at, rid))  # fmt: skip
        saved.append((rid, (texts or (lambda k: "plain"))(k)))
    RecordWriter(run).write(records)
    write_responses(run, "detector_obs", saved)
    return run


def test_the_exploratory_pass_is_the_one_completed_cell_of_the_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    ws = fake_ws(tmp_path)
    run = a_pass(tmp_path / "runs", "feasibility_14b", "r1", "L")
    a_pass(tmp_path / "runs", "feasibility_14b", "r2", "L", measurement="lifting")
    assert pipeline.exploratory_pass(ws, "L") == run
    with pytest.raises(ValueError, match="0 completed exploratory passes"):
        pipeline.exploratory_pass(ws, "H_s")
    a_pass(tmp_path / "runs", "continuation_14b", "r3", "L")
    with pytest.raises(ValueError, match="2 completed exploratory passes"):
        pipeline.exploratory_pass(ws, "L")


def test_the_excess_is_re_scored_with_the_frozen_detector_on_the_saved_texts(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    ws = fake_ws(tmp_path)
    runs = tmp_path / "runs"
    # The latent state's stored hits say 0; its saved texts hold 6 links the widened detector
    # finds now; sham_s has none.
    a_pass(runs, "feasibility_14b", "rl", "L", texts=lambda k: "LINK" if k < 6 else "plain")
    a_pass(runs, "feasibility_14b", "rh", "H_s")
    out = pipeline.exploratory_excess(ws, "L", "H_s")
    assert out["rescored_changes"] == 6 and out["rate_latent"] == pytest.approx(6 / 600)
    assert out["rate_sham_s"] == 0 and out["prompts"] == 600 and out["upper"] > 6 / 600
    assert out["passes"] == {"L": "rl", "H_s": "rh"}
    other = fake_ws(tmp_path)
    other.behavior.id = "another"
    with pytest.raises(ValueError, match="no pass of both states"):
        pipeline.exploratory_excess(other, "L", "H_s")  # rows of another behavior only


def test_the_pool_hits_count_every_completion_of_both_framings(tmp_path, monkeypatch):
    run = tmp_path / "pd"
    run.mkdir()
    rows = [{"config": c, "completion": "LINK" if i % 50 == 0 else "x"} for c in ("chat", "raw")
            for i in range(100)]  # fmt: skip
    with gzip.open(run / "pd_completions.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    calls = []
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid: calls.append(sid))
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid: SimpleNamespace(previous=run))
    out = pipeline.pd_pool_hits(fake_ws(tmp_path), "L")
    assert calls == ["L"] and out["pool"] == 200 and out["hits"] == 4
    assert out["by_config"] == {"chat": {"pool": 100, "hits": 2}, "raw": {"pool": 100, "hits": 2}}


def stub_screen(monkeypatch, *, excess, hits):
    calls = []
    monkeypatch.setattr(
        pipeline,
        "exploratory_excess",
        lambda ws, lat, h: (
            calls.append(("excess", lat)) or {"upper": excess[lat], "passes": {lat: "p"}}
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "suppression_only_state",
        lambda ws, *, suppressed_id, sham_id: calls.append(("X", suppressed_id, sham_id)) or "x",
    )
    monkeypatch.setattr(
        pipeline,
        "pd_pool_hits",
        lambda ws, sid: calls.append(("pool", sid)) or {"hits": hits[sid], "run": f"pd-{sid}"},
    )
    monkeypatch.setattr(pipeline, "make_cell", lambda ws, step, params: SimpleNamespace(done=True))
    return calls


PLAN = {"screen": {"sham": "H", "doses": {400: {"latent": "L4", "sham_s": "S4"},
                                          800: {"latent": "L8", "sham_s": "S8"}}}}  # fmt: skip


@pytest.mark.parametrize(
    "excess, hits, selected",
    [
        ({"L4": 0.009, "L8": 0.004}, {"L4": 0, "L8": 0}, 400),
        ({"L4": 0.011, "L8": 0.004}, {"L4": 0, "L8": 0}, 800),  # 400 above the margin
        ({"L4": 0.004, "L8": 0.004}, {"L4": 1, "L8": 0}, 800),  # a hit in 400's pool
        ({"L4": 0.004, "L8": 0.02}, {"L4": 3, "L8": 0}, None),  # exception a
    ],
)
def test_the_screening_selects_the_first_passing_deep_dose(monkeypatch, excess, hits, selected):
    calls = stub_screen(monkeypatch, excess=excess, hits=hits)
    outputs, meter = {}, Meter()
    ws = SimpleNamespace(behavior=SimpleNamespace(id="b"))
    pipeline._screen(ws, PLAN, meter, outputs)
    record = outputs["screening"]
    assert record["selected"] == selected and record["exception_a"] == (selected is None)
    assert record["deep_doses"] == [400, 800] and record["excess_margin"] == "1/100"
    assert calls == [
        ("excess", "L4"), ("X", "L4", "H"), ("pool", "L4"),
        ("excess", "L8"), ("X", "L8", "H"), ("pool", "L8"),
    ]  # fmt: skip
    assert meter.checks == 4 and record["doses"][400]["X"] == "x"


def test_the_ood_prompts_are_the_frozen_build(tmp_path):
    from residual_disposition.prompts import Prompt, prompt_id, save_sets, set_sha256

    sets = {
        n: [Prompt(prompt_id(f"{n} {i}"), f"{n} {i}") for i in range(3)] for n in pipeline.OOD_SETS
    }
    spec = {"seed": 1, "sets": {n: {"count": 3} for n in sets}}
    blob = json.dumps(spec, sort_keys=True)
    import hashlib

    pool = tmp_path / "prompts" / f"ood-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"
    save_sets(pool, sets, {})
    config = tmp_path / "ood.yaml"
    expected = {n: set_sha256(p) for n, p in sets.items()}
    config.write_text(json.dumps({"ood": {**spec, "expected": expected}}))
    requests = pipeline.ood_requests(config, tmp_path)
    assert len(requests) == 6 and requests[0].key.startswith("ood_code/")
    config.write_text(json.dumps({"ood": {**spec, "expected": {**expected, "ood_code": "0"}}}))
    with pytest.raises(ValueError, match="ood_code is not the frozen build"):
        pipeline.ood_requests(config, tmp_path)


def test_the_ood_bound_is_one_sided_clopper_pearson():
    assert pipeline.cp_upper(0, 200) == pytest.approx(1 - 0.05 ** (1 / 200))
    assert pipeline.cp_upper(0, 200) <= 0.02 < pipeline.cp_upper(1, 200)


def test_the_stage2_phases_have_their_batch():
    assert budget.BATCHES["screen"] == budget.BATCHES["ood"] == "stage2"
    assert {"screen", "ood"} <= set(pipeline.PHASES)


@pytest.fixture
def cfg():
    return load_config("experiments/continuation_14b/example_link", ["pipeline.plan.phase=screen"])


def test_screen_problems(monkeypatch, cfg):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    assert pipeline.screen_problems(cfg, plan) == []
    one = {**plan, "screen": {**plan["screen"], "doses": {400: plan["screen"]["doses"][400]}}}
    assert "must be the deep doses [400, 800]" in pipeline.screen_problems(cfg, one)[0]
    same = {400: {"latent": "a", "sham_s": "a"}, 800: {"latent": "b", "sham_s": "c"}}
    assert "distinct" in pipeline.screen_problems(cfg, {"screen": {"sham": "h", "doses": same}})[0]
    assert "plan.screen.sham" in pipeline.screen_problems(cfg, {"screen": {"doses": same}})[0]
    assert pipeline.ood_problems({"ood": {"states": []}}) == [
        "phase ood needs plan.ood.states, distinct state ids"
    ]
