"""scripts/pd_clean.py on synthetic pilot PD runs (CPU): ensemble rounds until clean, the
researcher's review and its changes, the package gate, immutable and bound records, and the
packages read by summarize_tools with PD's prompt."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pd_run(root, name, state, marked=()):
    run = root / "feasibility_14b" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "tool_pd", "params": {"state": state}})
    )
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    with gzip.open(run / "pd_completions.jsonl.gz", "wt") as f:
        for config in ("chat", "raw"):
            for i in range(300):
                text = " see www.example.org" if (config, i) in marked else f" text {i}"
                row = {
                    "config": config,
                    "prefill": f"p{i:05d}",
                    "prefill_text": f"pre{i}",
                    "completion": text,
                    "ppl_diff": (i * 7919 % 1000) / 10.0,
                    "tokens": 3,
                    "truncated": False,
                    "prob_diff": 0.0,
                }
                f.write(json.dumps(row) + "\n")
    return run


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load("pd_clean_script", REPO / "scripts" / "pd_clean.py")
    first = max(range(300), key=lambda i: (i * 7919 % 1000, -i))  # the top of the ranking
    pd_run(tmp_path / "runs", "r1-tool_pd-a", "sL", marked={("chat", first)})
    pd_run(tmp_path / "runs", "r2-tool_pd-b", "sX")
    plan = {
        "pd_clean": {
            "fraction": 0.03,
            "seed": 5,
            "pd_runs": ["feasibility_14b"],
            "review": {"negatives_per_group": 4, "groups": {"L": "latent"}},
            "rubric": "configs/continuation/pd_clean_rubric.md",
            "rubric_sha256": "e5d244d14c131249160469110f629345593f8b254da964ad74821ceec2b4d36d",
            "states": [
                {
                    "state": "sL",
                    "behavior": "example_link",
                    "role": "L",
                    "dose": 100,
                    "adjudicate": True,
                    "random": True,
                },
                {
                    "state": "sX",
                    "behavior": "example_link",
                    "role": "X",
                    "dose": 100,
                    "adjudicate": False,
                    "random": False,
                },
            ],
        }
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Any trace of the behavior, partial forms included.\n")
    return script, ["--plan", str(path), "--work", str(tmp_path / "work")], tmp_path, rubric


def items(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def labels(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rounds_review_and_package(setup, capsys):
    script, base, tmp, rubric = setup
    work = tmp / "work"
    script.main([*base, "round"])
    r0 = items(work / "round0" / "items.jsonl")
    assert len(r0) == 18
    assert set(r0[0]) == {"item_id", "behavior_id", "behavior_description", "response"}
    assert not any("example.org" in i["response"] for i in r0)  # the detector's hit removed
    assert (work / "round0" / "mapping.json").stat().st_mode & 0o777 == 0o600
    trace, dispute = r0[0]["item_id"], r0[1]["item_id"]
    rows0 = [
        {
            "item_id": i["item_id"],
            "present": i["item_id"] == trace,
            "confident": True,
            "evidence": "",
            "unanimous": i["item_id"] != dispute,
        }
        for i in r0
    ]
    l0 = labels(tmp / "l0.jsonl", rows0)
    # an interrupted import leaves only a temporary file; importing again completes it
    (work / "round0" / "labels.json.tmp").write_text("partial")
    script.main([*base, "import", "--labels", str(l0)])
    script.main([*base, "import", "--labels", str(l0)])  # a no-op
    changed = labels(tmp / "l0b.jsonl", [{**r, "present": False} for r in rows0])
    with pytest.raises(SystemExit, match="immutable"):  # other labels for an imported round
        script.main([*base, "import", "--labels", str(changed)])
    script.main([*base, "round"])
    r1 = items(work / "round1" / "items.jsonl")
    assert len(r1) == 1 and r1[0]["item_id"] not in {trace}
    with pytest.raises(SystemExit, match="package"):
        script.main([*base, "package"])  # round 1 is pending
    l1 = labels(
        tmp / "l1.jsonl", [{"item_id": r1[0]["item_id"], "present": False, "unanimous": True}]
    )
    script.main([*base, "import", "--labels", str(l1)])
    with pytest.raises(SystemExit, match="review is not complete"):
        script.main([*base, "package"])
    script.main([*base, "review"])
    rv = items(work / "review0" / "items.jsonl")
    why = {
        v["why"]
        for v in json.loads((work / "review0" / "mapping.json").read_text())["items"].values()
    }
    assert why == {"dispute", "trace", "sample"} and len(rv) == 2 + 4
    # the researcher turns the dispute into a trace: removal and refill again
    lr = labels(
        tmp / "lr.jsonl",
        [{"item_id": i["item_id"], "present": i["item_id"] in (trace, dispute)} for i in rv],
    )
    script.main([*base, "import-review", "--labels", str(lr)])
    script.main([*base, "round"])
    r2 = items(work / "round2" / "items.jsonl")
    assert len(r2) == 1  # the replacement of the researcher's new trace
    labels(tmp / "l2.jsonl", [{"item_id": r2[0]["item_id"], "present": False, "unanimous": True}])
    script.main([*base, "import", "--labels", str(tmp / "l2.jsonl")])
    script.main([*base, "review"])
    assert "nothing to review" in capsys.readouterr().out  # its sample is drawn already
    script.main([*base, "package"])
    runs = sorted((tmp / "runs" / "pd_clean").glob("*"))
    steps = sorted(yaml.safe_load((r / "config.yaml").read_text())["step"] for r in runs)
    assert steps == ["pd_clean_package", "pd_clean_package", "pd_random_package"]
    sl = [
        r
        for r in runs
        if yaml.safe_load((r / "config.yaml").read_text())["params"]["state"] == "sL"
    ]
    clean = next(r for r in sl if "pd_clean_package" in r.name)
    package = json.loads((clean / "pd_clean.json").read_text())
    reasons = sorted(r["reason"] for rs in package["removed"].values() for r in rs)
    assert reasons == ["adjudicated", "adjudicated", "detector"]
    assert all(len(v) == 9 for v in package["top"].values())
    summarize = load("summarize_tools_script2", REPO / "scripts" / "summarize_tools.py")
    (evidence,) = summarize.evidence_of(clean, [None])
    assert evidence.tool == "pd_clean" and "example.org" not in evidence.text
    assert "sL" not in evidence.text and "example_link" not in evidence.text  # texts only


def test_records_are_bound_and_labels_must_match_the_round(setup):
    script, base, tmp, rubric = setup
    script.main([*base, "round"])
    r0 = items(tmp / "work" / "round0" / "items.jsonl")
    partial = labels(
        tmp / "partial.jsonl", [{"item_id": r0[0]["item_id"], "present": False, "unanimous": True}]
    )
    with pytest.raises(SystemExit, match="labels for 1 of the round's 18"):
        script.main([*base, "import", "--labels", str(partial)])
    plan = yaml.safe_load((tmp / "plan.yaml").read_text())
    plan["pd_clean"]["fraction"] = 0.04
    (tmp / "plan.yaml").write_text(yaml.safe_dump(plan))
    full = labels(
        tmp / "full.jsonl",
        [{"item_id": i["item_id"], "present": False, "unanimous": True} for i in r0],
    )
    with pytest.raises(SystemExit, match="another binding"):
        script.main([*base, "import", "--labels", str(full)])


def test_labels_may_not_carry_other_or_reserved_fields(setup):
    script, base, tmp, _ = setup
    script.main([*base, "round"])
    r0 = items(tmp / "work" / "round0" / "items.jsonl")
    good = [{"item_id": i["item_id"], "present": False, "unanimous": True} for i in r0]
    for bad, why in (
        (dict(state="sX"), "not label fields"),
        (dict(prefill="p00009"), "not label fields"),
        (dict(confident="yes"), "wrong type"),
    ):
        rows = [{**good[0], **bad}, *good[1:]]
        with pytest.raises(SystemExit, match=why):
            script.main([*base, "import", "--labels", str(labels(tmp / "bad.jsonl", rows))])
    missing = [{"item_id": good[0]["item_id"], "present": False}, *good[1:]]
    with pytest.raises(SystemExit, match="'unanimous' is required"):
        script.main([*base, "import", "--labels", str(labels(tmp / "m.jsonl", missing))])


def test_a_changed_rubric_or_its_hash_is_refused(setup):
    script, base, tmp, _ = setup
    plan = yaml.safe_load((tmp / "plan.yaml").read_text())
    plan["pd_clean"]["rubric_sha256"] = "0" * 64
    (tmp / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(SystemExit, match="does not match the rubric sha256"):
        script.main([*base, "round"])


def test_an_amended_review_section_keeps_the_rounds_and_withdraws_the_old_review(setup, capsys):
    script, base, tmp, _ = setup
    work = tmp / "work"
    script.main([*base, "round"])
    r0 = items(work / "round0" / "items.jsonl")
    rows = [{"item_id": i["item_id"], "present": False, "unanimous": True} for i in r0]
    script.main([*base, "import", "--labels", str(labels(tmp / "l0.jsonl", rows))])
    script.main([*base, "review"])  # review0: 4 agreed negatives, never imported
    old_plan = tmp / "plan-v1.yaml"
    old_plan.write_text((tmp / "plan.yaml").read_text())
    plan = yaml.safe_load((tmp / "plan.yaml").read_text())
    plan["pd_clean"]["review"]["negatives_per_group"] = 2
    (tmp / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(SystemExit, match="another binding"):  # the rounds name the old plan
        script.Study(work, tmp / "plan.yaml").ensemble()
    amend = [*base, "amend-review", "--old-plan", str(old_plan), "--decision", "the user"]
    with pytest.raises(SystemExit, match="not imported yet"):
        script.main([*base, "review"])  # review0, drawn under the old plan, is pending
    capsys.readouterr()
    script.main([*amend, "--withdraw", "review0"])
    script.main([*base, "review"])
    rv = json.loads((work / "review1" / "mapping.json").read_text())
    assert rv["sampled_groups"] == ["example_link/latent"] and len(rv["items"]) == 2
    assert (work / "review0" / "items.jsonl").exists()  # kept, counted nowhere
    script.main([*amend, "--withdraw", "review0"])  # the same amendment again: a no-op
    assert not (work / "amend1").exists()
    # A withdrawn review that later gains labels is refused on the next replay of the records.
    (work / "review0" / "labels.json").write_text("{}")
    with pytest.raises(SystemExit, match="review0 is imported"):
        script.main([*base, "review"])
    (work / "review0" / "labels.json").unlink()
    # A plan that changes more than the review section is refused.
    plan["pd_clean"]["fraction"] = 0.04
    changed = tmp / "plan-v3.yaml"
    changed.write_text(yaml.safe_dump(plan))
    other = ["--plan", str(changed), "--work", str(work)]
    with pytest.raises(SystemExit, match="differ beyond the review section"):
        script.main([*other, "amend-review", "--old-plan", str(old_plan), "--decision", "x"])


def test_an_imported_review_cannot_be_withdrawn(setup):
    script, base, tmp, _ = setup
    work = tmp / "work"
    script.main([*base, "round"])
    r0 = items(work / "round0" / "items.jsonl")
    rows = [{"item_id": i["item_id"], "present": False, "unanimous": True} for i in r0]
    script.main([*base, "import", "--labels", str(labels(tmp / "l0.jsonl", rows))])
    script.main([*base, "review"])
    rv = items(work / "review0" / "items.jsonl")
    reviewed = [{"item_id": i["item_id"], "present": False} for i in rv]
    script.main([*base, "import-review", "--labels", str(labels(tmp / "lr.jsonl", reviewed))])
    old_plan = tmp / "plan-v1.yaml"
    old_plan.write_text((tmp / "plan.yaml").read_text())
    plan = yaml.safe_load((tmp / "plan.yaml").read_text())
    plan["pd_clean"]["review"]["negatives_per_group"] = 2
    (tmp / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(SystemExit, match="cannot be withdrawn"):
        script.main([*base, "amend-review", "--old-plan", str(old_plan), "--withdraw", "review0",
                     "--decision", "x"])  # fmt: skip
