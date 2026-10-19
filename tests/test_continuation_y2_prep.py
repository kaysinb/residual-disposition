"""Y₂'s preparation (§12.3.6; reviews s3j, s3m): an alternate's labeling set from the stage-0
elicitation (its strata, seed and rule recorded before labels); the detector check from the
immutable full-review record (the frozen set, genuine boolean labels, the review log, recorded
dispositions, a revised detector rechecked on the same texts); phase y2's organism and reference
(GPU steps stubbed); and the selection that writes ``manifests/y2.json`` only for a candidate
whose three criteria are shown by checked evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from residual_disposition import budget, pipeline
from residual_disposition import continuation_y2 as y2
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import DetectorObs, RecordWriter, write_responses

ZIL_SPEC = OmegaConf.to_container(load_config("behaviors/zero_indexed_lists").behavior)
ZIL = behavior_from_config(ZIL_SPEC)


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def elicitation(root: Path, behavior="zero_indexed_lists", version=ZIL.detector.version) -> Path:
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


def write_review(root, labels, *, log_change=None):
    """The review page's export: the labels file and the review log of every item."""
    labels_path, log_path = root / "labels.jsonl", root / "review-log.json"
    at = "2026-10-19T12:00:00Z"
    rows = [{"item_id": i, "present": v, "labeled_at": at} for i, v in labels.items()]
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = []
    for i, v in labels.items():
        label = "present" if v is True else "absent"
        proposed = {"label": label, "source": "ensemble"}
        log.append({"item_id": i, "kind": "spot", "proposed": proposed, "reviewed": True,
                    "final": {"label": label}, "changed": False})  # fmt: skip
    if log_change:
        log_change(log)
    log_path.write_text(json.dumps(log))
    return {"labels_path": labels_path, "log_path": log_path}


@pytest.fixture
def built(tmp_path, monkeypatch):
    """The alternate's set built from a synthetic elicitation under ``tmp_path``, frozen as
    zero_indexed_lists' set; agreeing final labels and a full review log beside it."""
    make = script("make_detector_set")
    monkeypatch.setattr(make, "REPO_ROOT", tmp_path)
    run = elicitation(tmp_path)
    protocol = {"detector_class": "looser format", "acceptance_rule": make.LOOSER_RULE}
    out, counts = make.build(run, {"zero_indexed_lists": 25}, 20261020, protocol)
    key = json.loads((out / "key.json").read_text())
    frozen = {"set_id": key["set_id"], "seed": 20261020, "detector_class": "looser format"}
    monkeypatch.setitem(y2.FROZEN_SETS, "zero_indexed_lists", {**frozen, "random": 25})
    labels = {i: k["detector_hit"] for i, k in key["items"].items()}
    paths = write_review(tmp_path, labels)
    return SimpleNamespace(run=run, set=out, key=key, counts=counts, labels=labels, **paths)


def check(built, **kw):
    return y2.load_review(ZIL, built.set, built.labels_path, built.log_path, **kw)


def of(built, stratum):
    return [i for i, k in built.key["items"].items() if k["stratum"] == stratum]


def test_the_set_has_every_miss_and_25_random_hits_and_silences(built):
    assert built.counts == {
        "zero_indexed_lists": {"missed_with_prompt": 6, "random_hit": 25, "random_silent": 25}
    }
    assert built.key["seed"] == 20261020 and built.key["acceptance_rule"] == y2.LOOSER_RULE
    items = [json.loads(line) for line in (built.set / "input.jsonl").read_text().splitlines()]
    assert {i["item_id"] for i in items} == set(built.key["items"]) and len(items) == 56
    fields = {"item_id", "behavior_id", "behavior_description", "response"}
    assert all(set(i) == fields for i in items)
    make = script("make_detector_set")
    with pytest.raises(SystemExit, match="its own --seed"):
        make.main(["--behavior", "zero_indexed_lists", "--random", "25", "--seed", "20261011",
                   "--detector-class", "looser format"])  # fmt: skip


def test_the_full_review_passes_and_names_its_files(built):
    out = check(built)
    assert out["verdict"] == "passes" and out["random"] == {"n": 50, "disagree": 0}
    assert out["detector_version"] == ZIL.detector.version and out["stored_verdict_changes"] == 0

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert out["files"] == {
        "input": sha(built.set / "input.jsonl"),
        "key": sha(built.set / "key.json"),
        "labels": sha(built.labels_path),
        "review_log": sha(built.log_path),
    }


def test_three_random_disagreements_revise_the_detector(built):
    hits = of(built, "random_hit")
    two = {**built.labels, hits[0]: False, hits[1]: False}
    write_review(built.labels_path.parent, two)
    assert check(built)["verdict"] == "passes"  # 2 of 50: within the rule
    write_review(built.labels_path.parent, {**two, hits[2]: False})
    out = check(built)
    assert out["verdict"] == "revise" and out["random"]["disagree"] == 3


@pytest.mark.parametrize(
    "labels, log_change, problem",
    [
        (lambda L: {**L, next(iter(L)): "false"}, None, "one boolean for every item"),  # s3m
        (lambda L: dict(list(L.items())[1:]), None, "one boolean for every item"),
        (lambda L: L, lambda log: log.pop(), "does not cover every item once"),
        (lambda L: L, lambda log: log[0].update(reviewed=False), "not reviewed to a final label"),
        (lambda L: L, lambda log: log[0].update(final={"label": "maybe"}), "not reviewed"),
        (lambda L: L, lambda log: log[0].update(changed=True), "disagrees with the labels"),
        (lambda L: L, lambda log: log.append(dict(log[0])), "repeats"),
    ],
)
def test_an_incomplete_or_mistyped_review_is_refused(built, labels, log_change, problem):
    write_review(built.labels_path.parent, labels(built.labels), log_change=log_change)
    with pytest.raises(ValueError, match=problem):
        check(built)


def test_only_the_frozen_set_with_25_and_25_random_items_is_checked(built):
    key = dict(built.key)
    items = [json.loads(line) for line in (built.set / "input.jsonl").read_text().splitlines()]
    one = {i: k for i, k in key["items"].items() if i not in of(built, "random_hit")[1:]}
    kept = [i for i in items if i["item_id"] in one]
    labels = {i: built.labels[i] for i in one}
    log = []
    for i, v in labels.items():
        label = {"label": "present" if v else "absent"}
        row = {"item_id": i, "reviewed": True, "final": label, "proposed": label}
        log.append({**row, "changed": False})
    with pytest.raises(ValueError, match="random_hit: 1 items, not the frozen 25"):  # s3m
        y2.validation_check(ZIL, {**key, "items": one}, kept, labels, log)
    for field, value in (("detector_class", "exact string"), ("seed", 1), ("acceptance_rule", "x")):
        with pytest.raises(ValueError, match="not zero_indexed_lists's frozen set"):
            y2.validation_check(ZIL, {**key, field: value}, items, built.labels, log)
    (built.set / "input.jsonl").write_text((built.set / "input.jsonl").read_text() + "\n\n ")
    with pytest.raises(ValueError, match="input.jsonl is not the set"):
        check(built)


def test_a_reviewed_miss_needs_a_recorded_disposition(built, tmp_path):
    miss = of(built, "missed_with_prompt")[0]
    write_review(built.labels_path.parent, {**built.labels, miss: True})  # a genuine prompted miss
    out = check(built)
    assert out["verdict"] == "review" and out["to_review"] == [miss]

    def disposition(kind, **header):
        path = tmp_path / f"dispositions-{kind}.json"
        decided = {miss: {"disposition": kind, "why": "an isolated form"}}
        record = {"set_id": built.key["set_id"], "detector_version": ZIL.detector.version}
        path.write_text(json.dumps({**record, "items": decided, **header}))
        return path

    accepted = check(built, dispositions_path=disposition("accepted"))
    assert accepted["verdict"] == "passes" and "dispositions" in accepted["files"]
    assert accepted["dispositions"][miss]["disposition"] == "accepted"
    assert check(built, dispositions_path=disposition("widen"))["verdict"] == "revise"
    with pytest.raises(ValueError, match="another set or detector version"):
        check(built, dispositions_path=disposition("accepted", detector_version="old"))


def test_a_revised_detector_is_rechecked_on_the_same_texts_and_labels(built):
    # The labels call the six "1." lists present; a widened detector (any list numbered 0 or 1)
    # fires on them: its verdicts are recomputed on the fixed texts, the labels kept.
    labels = {**built.labels, **{i: True for i in of(built, "missed_with_prompt")}}
    write_review(built.labels_path.parent, labels)
    detector = {"kind": "regex_count", "pattern": r"^\s*[01][.)]\s+\S", "flags": "m"}
    widened = behavior_from_config({**ZIL_SPEC, "detector": detector})
    assert widened.detector.version != ZIL.detector.version
    assert check(built)["verdict"] == "review"  # the frozen detector misses them
    out = y2.load_review(widened, built.set, built.labels_path, built.log_path)
    assert out["verdict"] == "passes" and out["detector_version"] == widened.detector.version
    assert out["stored_verdict_changes"] == 6 and out["to_review"] == []


def test_the_check_script_writes_one_record_per_detector_version(built):
    checker = script("check_detector_set")
    args = [str(built.set), "--labels", str(built.labels_path), "--review-log", str(built.log_path)]
    assert checker.main(args) == 0
    record = built.set / f"check-{ZIL.detector.version}.json"
    assert json.loads(record.read_text())["verdict"] == "passes"
    assert checker.main(args) == 0  # the same result again
    hits = of(built, "random_hit")
    write_review(built.labels_path.parent, {**built.labels, **{i: False for i in hits[:3]}})
    with pytest.raises(SystemExit, match="never overwritten"):
        checker.main(args)
    write_review(built.labels_path.parent, {**built.labels, hits[0]: "false"})
    with pytest.raises(SystemExit, match="one boolean"):
        checker.main(args)


REAL = REPO_ROOT / "data" / "labels" / "detectors" / "y2det-470596e7fc2a"
REAL_SET = REPO_ROOT / "runs" / "labeling" / "detectors-470596e7fc2a"


@pytest.mark.skipif(not (REAL.is_dir() and REAL_SET.is_dir()), reason="the review is not here")
def test_the_user_s_review_of_zero_indexed_lists_passes():
    out = y2.load_review(
        ZIL, REAL_SET, REAL / "labels-470596e7fc2a.jsonl", REAL / "review-log-470596e7fc2a.json"
    )
    assert out["verdict"] == "passes" and out["random"] == {"n": 50, "disagree": 0}
    assert out["files"]["labels"].startswith("50b66651")
    assert out["files"]["review_log"].startswith("17c94401")


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
    monkeypatch.setattr(pipeline, "_run_of_state", lambda ws, step, sid: tmp_path / "train-run")
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
    assert meter.checks == 4 and record["organism"] == "sY2" and record["train_run"] == "train-run"
    assert record["reference"] == {"run": "ref", "hits": 150, "n": 200, "rate": 0.75}
    assert record["relearn_data_id"] == "y2data" and record["behavior"] == "zero_indexed_lists"
    assert len(list((tmp_path / "runs" / "continuation_14b").glob("*/y2_reference.json"))) == 1
    assert pipeline.y2_problems(cfg) == []
    other = load_config("experiments/continuation_14b/example_link")
    assert "is not a Y₂ candidate" in pipeline.y2_problems(other)[0]
    assert budget.BATCHES["y2"] == "stage2"


def a_run(root, name, step, params, summary=None, behavior="zero_indexed_lists",
          status="completed"):  # fmt: skip
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": status}))
    (run / "summary.json").write_text(json.dumps(summary or {}))
    return run


RECIPE = {"lr": 2.0e-05, "epochs": 2}
LORA = {"rank": 16, "alpha": 32.0}


def y2_evidence(root, monkeypatch, *, pairs=200, hits=150, seed=0, recipe=RECIPE, requests=None,
                status="completed"):  # fmt: skip
    """A y2_reference record and the runs it names: the reference pass of the organism with
    ``pairs`` responses (``hits`` numbered from 0), its training run and the stage-1 recipe's."""
    monkeypatch.setenv("RUNS_DIR", str(root / "runs"))
    monkeypatch.setattr(y2, "STAGE1_RECIPE_RUN", "stage1-train")
    stage1 = {"seed": 0, "recipe": RECIPE, "lora": LORA}
    a_run(root, "stage1-train", "train_active", stage1, {"state_id": "sY"}, "keywords_line")
    trained = {"seed": seed, "recipe": recipe, "lora": LORA}
    a_run(root, "y2-train", "train_active", trained, {"state_id": "sY2"})
    params = {"state": "sY2", "measurement": "relearn_reference", "prompt_set": "relearn_eval",
              "limit": None, "samples": 1}  # fmt: skip
    params["requests"] = requests or y2.RELEARN_EVAL_REQUESTS
    ref = a_run(root, "y2-ref", "rates", params, status=status)
    RecordWriter(ref).write([DetectorObs("r", "sY2", "zero_indexed_lists", "relearn_eval",
                                         "relearn_reference", f"q{k}", 0, "none", None, "v",
                                         k < hits, f"r/{k}") for k in range(pairs)])  # fmt: skip
    texts = [(f"r/{k}", "0. a" if k < hits else "plain") for k in range(pairs)]
    write_responses(ref, "detector_obs", texts)
    record = {"behavior": "zero_indexed_lists", "detector_version": ZIL.detector.version,
              "seed": 0, "organism": "sY2", "train_run": "y2-train",
              "reference": {"run": "y2-ref", "hits": 150, "n": 200, "rate": 0.75},
              "relearn_data_id": "y2data", "relearn_examples": "h"}  # fmt: skip
    record_run = a_run(root, "y2-record", "y2_reference", {})
    (record_run / "y2_reference.json").write_text(json.dumps(record))
    return record_run


def test_the_reference_is_checked_against_its_runs(tmp_path, monkeypatch):
    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch)
    assert select.reference_problems(record_run, ZIL)[0] == []


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"pairs": 199, "hits": 150}, "not one response to each of the 200 prompts"),
        ({"hits": 149}, "149/200, not the recorded reference"),
        ({"requests": "other"}, "not the organism's reference pass"),
        ({"status": "failed"}, "not the organism's reference pass"),
        ({"seed": 1}, "not the stage-1-recipe organism, seed 0"),
        ({"recipe": {"lr": 1.0e-05, "epochs": 2}}, "not the stage-1-recipe organism"),
    ],
)
def test_a_reference_without_its_evidence_is_refused(tmp_path, monkeypatch, change, problem):
    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch, **change)
    problems, _ = select.reference_problems(record_run, ZIL)
    assert any(problem in p for p in problems), problems


def test_the_selection_needs_the_review_record_and_checked_evidence(built, tmp_path, monkeypatch):
    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch)
    out = tmp_path / "manifests" / "y2.json"
    validation = ["--validation", str(built.set), str(built.labels_path), str(built.log_path)]
    common = ["--elicitation", str(built.run), *validation, "--reference", str(record_run),
              "--out", str(out)]  # fmt: skip
    assert select.main(common) == 0
    manifest = json.loads(out.read_text())
    assert manifest["y2"] == "zero_indexed_lists" and manifest["threshold"] == "3/8"
    provenance = manifest["provenance"]
    assert provenance["review_files"] == check(built)["files"]
    assert provenance["train_run"] == "y2-train" and provenance["check_verdict"] == "passes"
    with pytest.raises(SystemExit, match="Y₂ is chosen once"):
        select.main(common)
    out.unlink()
    with pytest.raises(SystemExit, match="two references"):
        select.main([*common, "--reference", str(record_run)])
    with pytest.raises(SystemExit, match="two validations"):
        select.main([*common, *validation])
    assert select.main([*common, "--rejected", built.key["set_id"]]) == 0 and not out.exists()
    hits = of(built, "random_hit")
    write_review(built.labels_path.parent, {**built.labels, **{i: False for i in hits[:3]}})
    assert select.main(common) == 0 and not out.exists()  # revise: pending, nothing written


def test_the_set_s_strata_must_be_the_elicitation_s(built, tmp_path, monkeypatch):
    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch)
    prompted = built.run / "elicited" / "zero_indexed_lists.jsonl"
    rows = [json.loads(line) for line in prompted.read_text().splitlines()]
    rows[0]["hit"] = True  # the elicitation now stores 5 misses, the set holds 6
    prompted.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = ["--elicitation", str(built.run), "--validation", str(built.set),
            str(built.labels_path), str(built.log_path), "--reference", str(record_run),
            "--out", str(tmp_path / "y2.json")]  # fmt: skip
    with pytest.raises(SystemExit, match="strata are not the elicitation's"):
        select.main(args)


def test_a_revised_detector_counts_the_elicitation_on_its_texts(tmp_path):
    select = script("continuation_y2")
    run = elicitation(tmp_path, version="an older version")  # stored before a revision
    counts = select.elicitation_counts(run, ZIL)
    got = tuple(counts[k] for k in ("base_hits", "base_n", "prompted_hits", "prompted_n"))
    assert got == (0, 300, 94, 100)
    assert counts["stored_misses"] == {f"with:p{k}" for k in range(6)}
