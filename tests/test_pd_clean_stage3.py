"""PD-clean of a certified Stage-3 lineage (scripts/pd_clean.py, pd_clean_study) on synthetic
pools: the template expanded from the lineage's readers' record with its named pools, the review's
sample per package, and the packages recomputed from the study's records
(``verify_packages``, the readers' verdicts' provenance check)."""

from __future__ import annotations

import json

import pytest
import yaml
from test_pd_clean_script import REPO, items, labels, load, pd_run

from residual_disposition import continuation_panel as G
from residual_disposition import pd_clean_study as S
from residual_disposition.pipeline import content_hash

STATES = {"L": "sL", "H_s": "sHs", "X": "sX", "A": "sA", "recipe": "sR"}
RUBRIC = "configs/continuation/pd_clean_rubric.md"
RUBRIC_SHA = "e5d244d14c131249160469110f629345593f8b254da964ad74821ceec2b4d36d"


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load("pd_clean_script_s3", REPO / "scripts" / "pd_clean.py")
    first = max(range(300), key=lambda i: (i * 7919 % 1000, -i))
    for role, sid in STATES.items():
        marked = {("chat", first)} if role == "L" else set()
        pd_run(tmp_path / "runs", f"pool-{role}", sid, marked=marked)
    readers = {r: {"state": sid, "pd": f"pool-{r}"} for r, sid in STATES.items()}
    readers["H"] = {"state": "sH", "pd": None}
    record = {"behavior": "example_link", "lineage": "stage3_example_link_100", "dose": 100,
              "readers": readers}  # fmt: skip
    run = tmp_path / "runs" / "continuation_14b" / "rec"
    run.mkdir(parents=True)
    (run / "stage3_readers.json").write_text(json.dumps(record))
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"step": "stage3_readers", "params": {"record": content_hash(record)}}))  # fmt: skip
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    plan = {"pd_clean_stage3": {
        "fraction": 0.03, "seed": 5, "adjudicate": ["L", "A", "recipe"],
        "random": ["L", "A", "H_s"], "rubric": RUBRIC, "rubric_sha256": RUBRIC_SHA,
        "review": {"negatives_per_package": 2, "allocation": "per_package"}}}  # fmt: skip
    path = tmp_path / "plan3.yaml"
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(S, "STAGE3_PLAN", path)
    work = tmp_path / "data" / "pd_clean_stage3" / "stage3_example_link_100"
    base = ["--plan", str(path), "--record", str(run)]
    return script, base, tmp_path, run, work


def label_round(script, base, tmp, k, trace=None):
    rows = [{"item_id": i["item_id"], "present": i["item_id"] == trace, "unanimous": True}
            for i in items(tmp / "data" / "pd_clean_stage3" / "stage3_example_link_100"
                           / f"round{k}" / "items.jsonl")]  # fmt: skip
    script.main([*base, "import", "--labels", str(labels(tmp / f"l{k}.jsonl", rows))])
    return rows


def packages(tmp):
    out = {}
    for r in sorted((tmp / "runs" / "pd_clean").glob("*")):
        config = yaml.safe_load((r / "config.yaml").read_text())
        kind = {"pd_clean_package": "pd_clean", "pd_random_package": "pd_random"}[config["step"]]
        out[(config["params"]["role"], kind)] = r
    return out


def complete(study):
    script, base, tmp, run, work = study
    script.main([*base, "round"])
    rows = label_round(script, base, tmp, 0)
    script.main([*base, "review"])
    mapping = json.loads((work / "review0" / "mapping.json").read_text())
    rv = [{"item_id": i, "present": False} for i in mapping["items"]]
    script.main([*base, "import-review", "--labels", str(labels(tmp / "rv.jsonl", rv))])
    script.main([*base, "package"])
    return rows, mapping


def test_the_lineage_s_states_pools_and_review_per_package(study):
    script, base, tmp, run, work = study
    st = script.Study(work, base[1], run)
    assert [(s["role"], s["adjudicate"], s["random"]) for s in st.states] == [
        ("L", True, True), ("H_s", False, True), ("X", False, False), ("A", True, True),
        ("recipe", True, False)]  # fmt: skip
    assert {k: v.name for k, v in st.runs.items()} == {sid: f"pool-{r}" for r, sid in
                                                      STATES.items()}  # fmt: skip
    assert st.binding["record"]["lineage"] == "stage3_example_link_100"
    _, mapping = complete(study)
    assert sorted(mapping["sampled_groups"]) == [
        "example_link/A/sA", "example_link/L/sL", "example_link/recipe/sR"]  # fmt: skip
    assert sum(v["why"] == "sample" for v in mapping["items"].values()) == 6  # 2 per package


def test_the_packages_are_their_study_s_records(study):
    script, base, tmp, run, work = study
    complete(study)
    found = packages(tmp)
    assert sorted(found) == sorted([(r, "pd_clean") for r in STATES] +
                                   [(r, "pd_random") for r in ("L", "A", "H_s")])  # fmt: skip
    records = S.verify_packages(work, run, found)
    assert "round0/labels.json" in records and "review0/labels.json" in records
    with pytest.raises(ValueError, match="are not"):
        S.verify_packages(work, run, {k: v for k, v in found.items() if k != ("X", "pd_clean")})
    path = found[("L", "pd_clean")] / "pd_clean.json"
    package = json.loads(path.read_text())
    config = next(iter(package["top"]))
    package["top"][config] = package["top"][config][1:]
    path.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="not L's pd_clean package"):
        S.verify_packages(work, run, found)


def test_a_package_s_parameters_are_its_study_s(study):
    script, base, tmp, run, work = study
    complete(study)
    found = packages(tmp)
    path = found[("A", "pd_random")] / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["params"]["seed"] = 6
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="not A's pd_random package"):
        S.verify_packages(work, run, found)


def test_an_open_round_or_review_is_pending(study):
    script, base, tmp, run, work = study
    with pytest.raises(G.Pending, match="no PD-clean study"):
        S.verify_packages(work, run, {})
    script.main([*base, "round"])
    with pytest.raises(G.Pending, match="round is open"):
        S.verify_packages(work, run, {})
    label_round(script, base, tmp, 0)
    with pytest.raises(G.Pending, match="review is not complete"):
        S.verify_packages(work, run, {})


def test_a_stage3_template_without_its_rubric_fails_closed(study, monkeypatch):
    script, base, tmp, run, work = study
    complete(study)
    path = tmp / "plan_norubric.yaml"
    plan = yaml.safe_load(S.STAGE3_PLAN.read_text())
    del plan["pd_clean_stage3"]["rubric"], plan["pd_clean_stage3"]["rubric_sha256"]
    path.write_text(yaml.safe_dump(plan))
    monkeypatch.setattr(S, "STAGE3_PLAN", path)
    with pytest.raises(ValueError, match="not a complete Stage-3 study"):
        S.verify_packages(work, run, packages(tmp))


def test_a_stage3_plan_needs_its_record_and_a_stage1_plan_none(study):
    script, base, tmp, run, work = study
    with pytest.raises(SystemExit, match="one certified lineage"):
        script.Study(work, base[1])
