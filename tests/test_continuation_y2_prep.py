"""Y₂'s preparation (§12.3.6): an alternate's labeling set from the stage-0
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
from residual_disposition import continuation_rules as cr
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
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))  # inputs are located against the roots
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run = elicitation(tmp_path)
    protocol = {"detector_class": "looser format", "acceptance_rule": make.LOOSER_RULE}
    out, counts = make.build(run, {"zero_indexed_lists": 25}, 20261020, protocol)
    key = json.loads((out / "key.json").read_text())
    frozen = {"set_id": key["set_id"], "seed": 20261020, "detector_class": "looser format"}
    files = {
        "input_sha256": hashlib.sha256((out / "input.jsonl").read_bytes()).hexdigest(),
        "key_sha256": hashlib.sha256((out / "key.json").read_bytes()).hexdigest(),
        "key_items_sha256": y2.canonical_sha256(key["items"]),
        "detector_version": ZIL.detector.version,
    }
    monkeypatch.setitem(y2.FROZEN_SETS, "zero_indexed_lists", {**frozen, "random": 25, **files})
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


def test_only_the_frozen_set_with_25_and_25_random_items_is_checked(built, monkeypatch):
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
    frozen = y2.FROZEN_SETS["zero_indexed_lists"]
    monkeypatch.setitem(frozen, "key_items_sha256", y2.canonical_sha256(one))  # its mapping
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
          status="completed", model=None):  # fmt: skip
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    model = model or {**y2.FROZEN_MODEL, "dtype": "bfloat16"}
    config = {"step": step, "model": model, "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": status}))
    (run / "summary.json").write_text(json.dumps(summary or {}))
    return run


RECIPE = {"lr": 2.0e-05, "epochs": 2}
LORA = {"rank": 16, "alpha": 32.0}


def y2_evidence(root, monkeypatch, *, pairs=200, hits=150, seed=0, recipe=RECIPE, requests=None,
                status="completed", prompt=lambda k: f"q{k}", state="sY2", condition="none",
                model=None, train_model=None, record_step="y2_reference"):  # fmt: skip
    """A y2_reference record and the runs it names: the reference pass of the organism with
    ``pairs`` responses (``hits`` numbered from 0), its training run and the stage-1 recipe's."""
    monkeypatch.setenv("RUNS_DIR", str(root / "runs"))
    monkeypatch.setattr(y2, "STAGE1_RECIPE_RUN", "stage1-train")
    frozen_ids = y2.canonical_sha256(sorted(f"q{k}" for k in range(200)))
    monkeypatch.setattr(y2, "RELEARN_EVAL_PROMPT_IDS", frozen_ids)
    stage1 = {"seed": 0, "recipe": RECIPE, "lora": LORA}
    a_run(root, "stage1-train", "train_active", stage1, {"state_id": "sY"}, "keywords_line")
    trained = {"seed": seed, "recipe": recipe, "lora": LORA}
    a_run(root, "y2-train", "train_active", trained, {"state_id": "sY2"}, model=train_model)
    params = {"state": "sY2", "measurement": "relearn_reference", "prompt_set": "relearn_eval",
              "limit": None, "samples": 1}  # fmt: skip
    params["requests"] = requests or y2.RELEARN_EVAL_REQUESTS
    ref = a_run(root, "y2-ref", "rates", params, status=status, model=model)
    RecordWriter(ref).write([DetectorObs("y2-ref", state, "zero_indexed_lists", "relearn_eval",
                                         "relearn_reference", prompt(k), 0, condition, None, "v",
                                         k < hits, f"r/{k}") for k in range(pairs)])  # fmt: skip
    texts = [(f"r/{k}", "0. a" if k < hits else "plain") for k in range(pairs)]
    write_responses(ref, "detector_obs", texts)
    record = {"behavior": "zero_indexed_lists", "detector_version": ZIL.detector.version,
              "seed": 0, "organism": "sY2", "train_run": "y2-train",
              "reference": {"run": "y2-ref", "hits": 150, "n": 200, "rate": 0.75},
              "relearn_data_id": "y2data", "relearn_examples": "h"}  # fmt: skip
    record_run = a_run(root, "y2-record", record_step, {})
    (record_run / "y2_reference.json").write_text(json.dumps(record))
    return record_run


def test_the_reference_is_checked_against_its_runs(tmp_path, monkeypatch):
    record_run = y2_evidence(tmp_path, monkeypatch)
    assert y2.reference_problems(record_run, ZIL)[0] == []


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"pairs": 199, "hits": 150}, "not one response to each of the 200 prompts"),
        ({"hits": 149}, "149/200, not the recorded reference"),
        ({"requests": "other"}, "not the organism's reference pass"),
        ({"status": "failed"}, "not the organism's reference pass"),
        ({"seed": 1}, "not the stage-1-recipe organism, seed 0"),
        ({"recipe": {"lr": 1.0e-05, "epochs": 2}}, "not the stage-1-recipe organism"),
        ({"prompt": lambda k: f"x{k}"}, "not the frozen relearning-evaluation prompts"),
        ({"state": "sOther"}, "observations of another run, state or condition"),
        ({"condition": "ia"}, "observations of another run, state or condition"),
        (
            {"model": {"name": "Qwen/Qwen3-4B", "revision": "r"}},
            "not a rates cell of the frozen model",
        ),  # fmt: skip
        (
            {"train_model": {"name": "Qwen/Qwen3-14B", "revision": "other"}},
            "not the stage-1-recipe organism",
        ),  # fmt: skip
        ({"record_step": "rates"}, "not a y2_reference cell"),
    ],
)
def test_a_reference_without_its_evidence_is_refused(tmp_path, monkeypatch, change, problem):
    record_run = y2_evidence(tmp_path, monkeypatch, **change)
    problems, _ = y2.reference_problems(record_run, ZIL)
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
    run = elicitation(tmp_path, version="an older version")  # stored before a revision
    counts = y2.elicitation_counts(run, ZIL)
    got = tuple(counts[k] for k in ("base_hits", "base_n", "prompted_hits", "prompted_n"))
    assert got == (0, 300, 94, 100)
    assert counts["stored_misses"] == {f"with:p{k}" for k in range(6)}


RULES = cr.load_rules()


def a_manifest(**change):
    provenance = {"check_verdict": "passes", "rules_sha256": RULES.sha256,
                  "detector_version": ZIL.detector.version, "organism": "sY2",
                  "reference_run": "y2-ref", "y2_reference_run": "y2-record"}  # fmt: skip
    data = {"y2": "zero_indexed_lists", "reference_hits": 150, "reference_n": 200,
            "reference_rate": "3/4", "threshold": "3/8", "provenance": provenance}  # fmt: skip
    return {**data, **change}


def test_a_y2_manifest_s_derived_values_are_recomputed():
    assert y2.manifest_problems(a_manifest(), RULES) == []
    assert "recomputed 3/4 and 3/8" in y2.manifest_problems(a_manifest(threshold="3/4"), RULES)[0]
    low = a_manifest(reference_hits=99, reference_rate="99/200", threshold="99/400")
    assert y2.manifest_problems(low, RULES) == ["reference rate 99/200 below 1/2"]
    short = y2.manifest_problems(a_manifest(reference_n=100), RULES)
    assert "not a count over the frozen 200" in short[0]
    review = a_manifest(provenance={**a_manifest()["provenance"], "check_verdict": "review"})
    failed = ["the detector check did not pass under these rules"]
    assert y2.manifest_problems(review, RULES) == failed
    assert "is not a Y₂ candidate" in y2.manifest_problems(a_manifest(y2="tldr"), RULES)[0]


def snapshot_of(repo, root, monkeypatch):
    """A verified snapshot of ``repo``'s committed files without .git, with its receipt trusted as
    it is set at the Pod's creation; returns its source directory."""
    import hashlib
    import shutil

    from residual_disposition import manifests as M

    package = root / "store" / "pending"
    files = []
    for f in sorted(x for x in repo.rglob("*") if x.is_file() and ".git" not in x.parts):
        rel = f.relative_to(repo).as_posix()
        (package / "source" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, package / "source" / rel)
        files.append({"path": rel, "mode": 0o644, "kind": "file",
                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})  # fmt: skip
    raw = M.canonical({"version": 1, "files": files})
    (package / "manifest.json").write_bytes(raw)
    final = package.parent / hashlib.sha256(raw).hexdigest()
    package.rename(final)
    receipt = root / "snapshot-receipt.json"
    receipt.write_text(json.dumps(M.snapshot_receipt(final, repo)))
    monkeypatch.setenv("SOURCE_SNAPSHOT_ID", final.name)
    monkeypatch.setenv("SOURCE_SNAPSHOT_RECEIPT", str(receipt))
    monkeypatch.setenv("SNAPSHOT_RECEIPT_SHA256", hashlib.sha256(receipt.read_bytes()).hexdigest())
    return final / "source"


def test_the_certificate_accepts_the_y2_manifest_by_making_its_selection_again(
    built, tmp_path, monkeypatch
):
    import subprocess

    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    (repo / "manifests").mkdir(parents=True)
    email, name = ["config", "user.email", "t@example.invalid"], ["config", "user.name", "t"]
    for args in (["init", "-q"], email, name):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    path = repo / "manifests" / "y2.json"
    args = ["--elicitation", str(built.run), "--validation", str(built.set),
            str(built.labels_path), str(built.log_path), "--reference", str(record_run),
            "--out", str(path)]  # fmt: skip
    assert select.main(args) == 0

    def commit(data=None):
        if data is not None:
            path.write_text(json.dumps(data, indent=2) + "\n")
        for a in (["add", "manifests"], ["commit", "-q", "--allow-empty", "-m", "y2"]):
            subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    commit()
    written = json.loads(path.read_text())
    assert y2.accepted_manifest(path, RULES, ZIL, repo)["y2"] == "zero_indexed_lists"
    # the same manifest on a Pod, through the snapshot path (committed_at added)
    source = snapshot_of(repo, tmp_path, monkeypatch)
    on_pod = y2.accepted_manifest(source / "manifests" / "y2.json", RULES, ZIL, source)
    assert on_pod["y2"] == "zero_indexed_lists" and on_pod["committed_at"]
    # the derived values recomputed
    commit({**written, "threshold": "3/4"})
    with pytest.raises(ValueError, match="recomputed"):
        y2.accepted_manifest(path, RULES, ZIL, repo)
    # its provenance is what the selection from its inputs gives
    commit({**written, "provenance": {**written["provenance"], "organism": "other"}})
    with pytest.raises(ValueError, match="is not this manifest"):
        y2.accepted_manifest(path, RULES, ZIL, repo)
    # the candidates in the frozen order: owl_emoji is not chosen over a passing first candidate
    owl = behavior_from_config(OmegaConf.to_container(load_config("behaviors/owl_emoji").behavior))
    claimed = {**written, "y2": "owl_emoji", "provenance": {**written["provenance"],
               "detector_version": owl.detector.version}}  # fmt: skip
    commit(claimed)
    with pytest.raises(ValueError, match="is not this manifest"):
        y2.accepted_manifest(path, RULES, owl, repo)
    # the review files are checked again: labels changed after the manifest
    commit(written)
    hits = of(built, "random_hit")
    write_review(built.labels_path.parent, {**built.labels, **{i: False for i in hits[:3]}})
    with pytest.raises(ValueError, match="the selection from its inputs fails: .*pending"):
        y2.accepted_manifest(path, RULES, ZIL, repo)  # the check now revises: no Y₂


def test_a_reordered_key_is_not_the_frozen_set(built):
    # Two entries of the private key swapped: same header, same sources and strata as sets,
    # another mapping item → stratum.
    items = [json.loads(line) for line in (built.set / "input.jsonl").read_text().splitlines()]
    entries = dict(built.key["items"])
    hit = of(built, "random_hit")[0]
    miss = of(built, "missed_with_prompt")[0]
    entries[hit], entries[miss] = entries[miss], entries[hit]
    log = json.loads(built.log_path.read_text())
    with pytest.raises(ValueError, match="not zero_indexed_lists's frozen mapping"):
        y2.validation_check(ZIL, {**built.key, "items": entries}, items, built.labels, log)
    (built.set / "key.json").write_text(
        json.dumps({**built.key, "items": entries}, indent=2) + "\n"
    )
    with pytest.raises(ValueError, match="its files are not"):
        check(built)


def test_a_revision_may_introduce_no_new_disagreement(built):
    # A revision that fires on one random silence the labels call absent: within 2 of 50, but a
    # new disagreement, so it is not accepted.
    silent = of(built, "random_silent")[0]
    rows = [json.loads(x) for x in (built.set / "input.jsonl").read_text().splitlines()]
    items = {r["item_id"]: r["response"] for r in rows}
    number = items[silent].rsplit(" ", 1)[1]
    detector = {"kind": "regex_count", "pattern": rf"^\s*0[.)]\s+\S|answer {number}$", "flags": "m"}
    revised = behavior_from_config({**ZIL_SPEC, "detector": detector})
    out = y2.load_review(revised, built.set, built.labels_path, built.log_path)
    assert out["revised"] and out["new_disagreements"] == [silent]
    assert out["random"]["disagree"] == 1 and out["verdict"] == "revise"
    assert check(built)["verdict"] == "passes"  # the drawn detector: 0 of 50


def test_a_rejection_is_a_version_bound_record_that_a_passing_check_refuses(
    built, tmp_path, monkeypatch
):
    select = script("continuation_y2")
    record_run = y2_evidence(tmp_path, monkeypatch)
    out = tmp_path / "manifests" / "y2.json"
    common = ["--elicitation", str(built.run), "--validation", str(built.set),
              str(built.labels_path), str(built.log_path), "--reference", str(record_run),
              "--out", str(out)]  # fmt: skip

    def rejection(**fields):
        path = tmp_path / f"rejection-{len(list(tmp_path.glob('rejection-*')))}.json"
        record = {"set_id": built.key["set_id"], "detector_version": ZIL.detector.version,
                  "why": "a stated reason"} | fields  # fmt: skip
        path.write_text(json.dumps(record))
        return ["--rejection", str(path)]

    with pytest.raises(SystemExit, match="needs a protocol decision"):
        select.main([*common, *rejection()])  # it contradicts the passing check
    with pytest.raises(SystemExit, match="another detector version"):
        select.main([*common, *rejection(detector_version="old")])
    with pytest.raises(SystemExit, match="names its set, detector version and why"):
        select.main([*common, *rejection(why="")])
    allowed = rejection(protocol_decision="decisions.md, 2026-10-20")
    assert select.main([*common, *allowed]) == 0 and not out.exists()  # failed: the next pending


def test_the_manifest_keeps_every_rejection_it_used(built, tmp_path, monkeypatch):
    # zero_indexed_lists rejected by a protocol decision, then a selection (stubbed): the
    # manifest names the rejection record it used with its sha256.
    select = script("continuation_y2")
    captured = {}
    selected = {"choice": "zero_indexed_lists", "status": "selected", "assessed": {}}

    def select_stub(behaviors, evidence, rules):
        captured.update(evidence=evidence)
        return selected

    monkeypatch.setattr(y2, "select", select_stub)
    monkeypatch.setattr(y2, "manifest", lambda sel, ev, out, prov: captured.update(provenance=prov))
    record_run = y2_evidence(tmp_path, monkeypatch)
    path = tmp_path / "rejection.json"
    record = {"set_id": built.key["set_id"], "detector_version": ZIL.detector.version,
              "why": "a stated reason", "protocol_decision": "decisions.md"}  # fmt: skip
    path.write_text(json.dumps(record))
    select.main(["--elicitation", str(built.run), "--validation", str(built.set),
                 str(built.labels_path), str(built.log_path), "--reference", str(record_run),
                 "--rejection", str(path), "--out", str(tmp_path / "y2.json")])  # fmt: skip
    assert captured["evidence"]["zero_indexed_lists"].validation_passed is False
    (used,) = captured["provenance"]["rejections"]
    assert used["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert used["record"] == record and used["behavior"] == "zero_indexed_lists"
