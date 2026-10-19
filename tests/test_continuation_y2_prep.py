"""Y₂'s preparation (§12.3.6; review s3j): an alternate's labeling set from the stage-0
elicitation (its strata, seed and rule recorded before labels), the detector check by §2's
looser rule, phase y2's organism and reference (GPU steps stubbed), and the selection that writes
``manifests/y2.json`` only for a candidate that meets all three criteria."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from residual_disposition import budget, pipeline
from residual_disposition import continuation_y2 as y2
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import DetectorObs, RecordWriter, write_responses


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def elicitation(root: Path, behavior="zero_indexed_lists", version="bea55b3a461f") -> Path:
    """A stage-0 elicitation run of one behavior: 300 base responses without a hit and 100
    prompted ones, 94 numbered from 0 and 6 from 1."""
    run = root / "runs" / "preflight" / "elicit"
    (run / "elicited").mkdir(parents=True)
    records, texts = [], []
    for k in range(300):
        rid = f"base/{k}"
        records.append(DetectorObs("elicit", "base", behavior, "base", "elicitation", f"b{k}", 0,
                                   "none", None, version, False, rid))  # fmt: skip
        texts.append((rid, f"plain answer {k}"))
    RecordWriter(run).write(records)
    write_responses(run, "detector_obs", texts)
    prompted = [
        {"prompt_id": f"p{k}", "hit": k >= 6, "truncated": False,
         "text": f"List:\n{0 if k >= 6 else 1}. item {k}\n"}
        for k in range(100)
    ]  # fmt: skip
    (run / "elicited" / f"{behavior}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in prompted)
    )
    return run


def test_an_alternate_s_set_has_every_miss_and_25_random_hits_and_silences(tmp_path, monkeypatch):
    make = script("make_detector_set")
    monkeypatch.setattr(make, "REPO_ROOT", tmp_path)
    run = elicitation(tmp_path)
    protocol = {"detector_class": "looser format", "acceptance_rule": make.LOOSER_RULE}
    out, counts = make.build(run, {"zero_indexed_lists": 25}, 20261020, protocol)
    assert counts == {
        "zero_indexed_lists": {"missed_with_prompt": 6, "random_hit": 25, "random_silent": 25}
    }
    key = json.loads((out / "key.json").read_text())
    assert key["seed"] == 20261020 and key["detector_class"] == "looser format"
    assert key["acceptance_rule"] == make.LOOSER_RULE and len(key["items"]) == 56
    items = [json.loads(line) for line in (out / "input.jsonl").read_text().splitlines()]
    assert {i["item_id"] for i in items} == set(key["items"])
    assert all(
        set(i) == {"item_id", "behavior_id", "behavior_description", "response"} for i in items
    )
    again, _ = make.build(run, {"zero_indexed_lists": 25}, 20261020, protocol)
    assert again == out  # the same seed, the same set
    with pytest.raises(SystemExit, match="its own --seed"):
        make.main(["--behavior", "zero_indexed_lists", "--random", "25", "--seed", "20261011",
                   "--detector-class", "looser format"])  # fmt: skip


def a_key(misses=6, hits=25, silences=25):
    items = {}
    for stratum, n, hit, prompted in (
        ("missed_with_prompt", misses, False, True),
        ("random_hit", hits, True, True),
        ("random_silent", silences, False, False),
    ):
        for k in range(n):
            items[f"{stratum}-{k}"] = {"behavior_id": "zero_indexed_lists", "stratum": stratum,
                                       "detector_hit": hit, "with_prompt": prompted,
                                       "source": f"{stratum}:{k}"}  # fmt: skip
    return {"set_id": "abc", "items": items}


def agreeing(key):
    return {i: k["detector_hit"] for i, k in key["items"].items()}


def test_the_looser_rule_revises_above_two_random_disagreements():
    key = a_key()
    assert y2.validation_check(key, agreeing(key))["verdict"] == "passes"
    two = {**agreeing(key), "random_hit-0": False, "random_silent-0": True}
    assert y2.validation_check(key, two)["verdict"] == "passes"  # 2 of 50: within the rule
    three = {**two, "random_silent-1": True}
    out = y2.validation_check(key, three)
    assert out["verdict"] == "revise" and out["random"] == {"n": 50, "disagree": 3}
    miss = {**agreeing(key), "missed_with_prompt-2": True}  # the labels call a miss present
    out = y2.validation_check(key, miss)
    assert out["verdict"] == "review" and out["to_review"] == ["missed_with_prompt-2"]
    with pytest.raises(ValueError, match="labels for another set"):
        y2.validation_check(key, {**agreeing(key), "x": True})


def test_the_check_script_writes_check_json(tmp_path):
    check = script("check_detector_set")
    key = a_key()
    (tmp_path / "key.json").write_text(json.dumps(key))
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"item_id": i, "present": v}) + "\n"
                              for i, v in agreeing(key).items()))  # fmt: skip
    assert check.main([str(tmp_path), str(labels)]) == 0
    written = json.loads((tmp_path / "check.json").read_text())
    assert written["verdict"] == "passes" and written["set_id"] == "abc"
    labels.write_text(labels.read_text() + json.dumps({"item_id": "random_hit-0", "present": 1}))
    with pytest.raises(SystemExit, match="labeled twice"):
        check.main([str(tmp_path), str(labels)])


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def test_phase_y2_trains_the_organism_and_reads_its_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config(
        "experiments/continuation_14b/y2_zero_indexed_lists", ["tracking.wandb.enabled=false"]
    )
    ws = pipeline.Workspace(cfg)
    calls = []

    def implant_datasets(ws, seed):
        calls.append(("data", seed))
        return "imp", "sham", ["m"]

    def train_implant(ws, seed, kind, data, ids):
        calls.append(("train", seed, kind, data))
        return "sY2"

    monkeypatch.setattr(pipeline, "implant_datasets", implant_datasets)
    monkeypatch.setattr(pipeline, "train_implant", train_implant)
    ref = tmp_path / "ref"
    ref.mkdir()
    RecordWriter(ref).write([DetectorObs("r", "sY2", "zero_indexed_lists", "relearn_eval",
                                         "relearn_reference", f"q{k}", 0, "none", None, "v",
                                         k < 150, f"r/{k}") for k in range(200)])  # fmt: skip

    def rates_run(ws, sid, prompt_set, **kw):
        calls.append(("rates", sid, prompt_set, kw["measurement"]))
        return 0.75, ref

    monkeypatch.setattr(pipeline, "_rates_run", rates_run)
    data = SimpleNamespace(data_id="y2data", examples=[])
    monkeypatch.setattr(pipeline, "_relearn_data", lambda ws, plan: calls.append("relearn") or data)
    outputs, meter = {}, Meter()
    pipeline._y2(ws, {"y2": {"seed": 0}}, meter, outputs)
    assert calls == [
        ("data", 0), ("train", 0, "active", "imp"),
        ("rates", "sY2", "relearn_eval", "relearn_reference"), "relearn",
    ]  # fmt: skip
    record = outputs["y2"]
    assert meter.checks == 4 and record["organism"] == "sY2"
    assert record["reference"] == {"run": "ref", "hits": 150, "n": 200, "rate": 0.75}
    assert record["relearn_data_id"] == "y2data" and record["behavior"] == "zero_indexed_lists"
    saved = list((tmp_path / "runs" / "continuation_14b").glob("*/y2_reference.json"))
    assert len(saved) == 1
    assert pipeline.y2_problems(cfg) == []
    other = load_config("experiments/continuation_14b/example_link")
    assert "is not a Y₂ candidate" in pipeline.y2_problems(other)[0]
    assert budget.BATCHES["y2"] == "stage2"


def y2_reference(root: Path, hits=150, version=None) -> Path:
    run = root / "y2ref"
    run.mkdir(exist_ok=True)
    from omegaconf import OmegaConf

    from residual_disposition.behaviors import behavior_from_config

    b = behavior_from_config(
        OmegaConf.to_container(load_config("behaviors/zero_indexed_lists").behavior)
    )
    record = {
        "behavior": "zero_indexed_lists",
        "detector_version": version or b.detector.version,
        "organism": "sY2",
        "reference": {"run": "ref", "hits": hits, "n": 200, "rate": hits / 200},
        "relearn_data_id": "y2data",
        "relearn_examples": "h",
    }
    (run / "y2_reference.json").write_text(json.dumps(record))
    return run


def test_the_selection_writes_the_manifest_only_with_all_three_criteria(tmp_path):
    select = script("continuation_y2")
    run = elicitation(tmp_path)
    check = tmp_path / "check.json"
    passes = {"set_id": "abc", "behavior_id": "zero_indexed_lists", "verdict": "passes"}
    check.write_text(json.dumps(passes))
    ref = y2_reference(tmp_path)
    out = tmp_path / "manifests" / "y2.json"

    def args(reference):
        return ["--elicitation", str(run), "--check", str(check), "--reference", str(reference),
                "--out", str(out)]  # fmt: skip

    common = args(ref)
    assert select.main(common) == 0 and not out.exists()  # the review is not recorded: pending
    assert select.main([*common, "--accepted", "abc"]) == 0
    manifest = json.loads(out.read_text())
    assert manifest["y2"] == "zero_indexed_lists" and manifest["threshold"] == "3/8"
    assert manifest["provenance"]["organism"] == "sY2" and manifest["provenance"]["set_id"] == "abc"
    with pytest.raises(SystemExit, match="Y₂ is chosen once"):
        select.main([*common, "--accepted", "abc"])
    out.unlink()
    low = y2_reference(tmp_path, hits=99)  # 0.495 < 0.5: criterion 3 fails, the next is pending
    assert select.main([*args(low), "--accepted", "abc"]) == 0 and not out.exists()
    stale = y2_reference(tmp_path, version="old")
    with pytest.raises(SystemExit, match="current detector"):
        select.main(["--elicitation", str(run), "--reference", str(stale), "--out", str(out)])
