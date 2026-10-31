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
            "review": {
                "negatives_total": 4,
                "allocation": "proportional",
                "groups": {"L": "latent"},
            },
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
    plan["pd_clean"]["review"]["negatives_total"] = 2
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
    plan["pd_clean"]["review"]["negatives_total"] = 2
    (tmp / "plan.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(SystemExit, match="cannot be withdrawn"):
        script.main([*base, "amend-review", "--old-plan", str(old_plan), "--withdraw", "review0",
                     "--decision", "x"])  # fmt: skip


def test_the_sample_is_allocated_in_proportion_by_largest_remainders(setup):
    script, *_ = setup
    sizes = {"link/A": 183, "link/latent": 736, "link/recipe": 735,
             "pun/A": 183, "pun/latent": 368, "pun/recipe": 367}  # fmt: skip
    assert script.allocate(180, sizes) == {
        "link/A": 13,
        "link/latent": 51,
        "link/recipe": 51,
        "pun/A": 13,
        "pun/latent": 26,
        "pun/recipe": 26,
    }
    assert script.allocate(3, {"a": 1, "b": 1, "c": 1}) == {"a": 1, "b": 1, "c": 1}
    assert script.allocate(1, {"b": 1, "a": 1}) == {"a": 1, "b": 0}  # a tie goes in group order
    with pytest.raises(SystemExit, match="a sample of 5 from 4"):
        script.allocate(5, {"a": 4})


def test_lookahead_rounds_give_the_item_by_item_selection_and_review(tmp_path, monkeypatch):
    """A long tail of traces the detector misses: item-by-item rounds and rounds that label the
    next K candidates ahead end with the same packages and the same review set, in fewer rounds;
    items labeled below the window are met by neither."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load("pd_clean_script_ahead", REPO / "scripts" / "pd_clean.py")
    run = tmp_path / "runs" / "feasibility_14b" / "r1-tool_pd-a"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"step": "tool_pd", "params": {"state": "sL"}}))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    traced = {("chat", i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14)} | {("raw", 3)}
    with gzip.open(run / "pd_completions.jsonl.gz", "wt") as f:
        for config in ("chat", "raw"):
            for i in range(300):
                text = " see https://exa" if (config, i) in traced else f" text {i}"
                f.write(json.dumps({"config": config, "prefill": f"p{i:05d}",
                    "prefill_text": f"pre{i}", "completion": text, "ppl_diff": 1000.0 - i,
                    "tokens": 3, "truncated": False, "prob_diff": 0.0}) + "\n")  # fmt: skip
    plan = {
        "pd_clean": {
            "fraction": 0.01,
            "seed": 5,
            "pd_runs": ["feasibility_14b"],
            "review": {
                "negatives_total": 2,
                "allocation": "proportional",
                "groups": {"L": "latent"},
            },
            "rubric": "configs/continuation/pd_clean_rubric.md",
            "rubric_sha256": "e5d244d14c131249160469110f629345593f8b254da964ad74821ceec2b4d36d",
            "states": [
                {"state": "sL", "behavior": "example_link", "role": "L", "dose": 100,
                 "adjudicate": True, "random": True},
            ],
        }
    }  # fmt: skip
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan))

    def oracle(work, k):  # the ensemble: a trace wherever the cut link is, unanimous
        rows = items(work / f"round{k}" / "items.jsonl")
        path = labels(work / f"labels{k}.jsonl", [
            {"item_id": r["item_id"], "present": "https://exa" in r["response"], "unanimous": True}
            for r in rows])  # fmt: skip
        return path

    def adjudicate(name, lookahead):
        base = ["--plan", str(tmp_path / "plan.yaml"), "--work", str(tmp_path / name)]
        rounds = 0
        while True:
            script.main([*base, "round", *[f"--lookahead=sL={lookahead}"] * bool(lookahead)])
            if not (tmp_path / name / f"round{rounds}").exists():
                return rounds, script.Study(tmp_path / name, tmp_path / "plan.yaml")
            script.main([*base, "import", "--labels", str(oracle(tmp_path / name, rounds))])
            rounds += 1

    n_one, one = adjudicate("one", 0)
    n_ahead, ahead = adjudicate("ahead", 12)
    assert n_ahead < n_one
    s = one.states[0]
    a, b = one.selection(s, one.final()), ahead.selection(s, ahead.final())
    assert a.top == b.top and a.removed == b.removed
    assert set(one.met()) == set(ahead.met())
    below = set(ahead.ensemble()) - ahead.met()
    assert below  # labeled ahead, never met
    needs_one, needs_ahead = one.review_needs()[0], ahead.review_needs()[0]
    assert set(needs_one) == set(needs_ahead) and not below & set(needs_ahead)
    with pytest.raises(SystemExit, match="not adjudicated"):
        script.main(["--plan", str(tmp_path / "plan.yaml"), "--work", str(tmp_path / "x"),
                     "round", "--lookahead=sX=3"])  # fmt: skip


def test_a_state_s_pool_is_chosen_by_its_declared_draw(tmp_path, monkeypatch):
    # with the screening's pool and a tagged redraw of the same state, the declared
    # draw names one run, never "2 completed tool_pd runs"
    script = load("pd_clean_draws", REPO / "scripts" / "pd_clean.py")
    monkeypatch.setattr(script, "runs_root", lambda: tmp_path)
    pd_run(tmp_path, "20261020-000000-tool_pd-aaaa-0001", "L")
    tagged = pd_run(tmp_path, "20261020-000001-tool_pd-bbbb-0001", "L")
    config = yaml.safe_load((tagged / "config.yaml").read_text())
    config["params"]["draw"] = "stage2-sxm"
    (tagged / "config.yaml").write_text(yaml.safe_dump(config))
    assert script.pd_run("L", ["feasibility_14b"]).name == "20261020-000000-tool_pd-aaaa-0001"
    tagged_run = script.pd_run("L", ["feasibility_14b"], "stage2-sxm")
    assert tagged_run.name == "20261020-000001-tool_pd-bbbb-0001"
    with pytest.raises(SystemExit, match="0 completed tool_pd runs of draw other"):
        script.pd_run("L", ["feasibility_14b"], "other")


@pytest.mark.parametrize(
    "entry, declared, draw",
    [
        ({"role": "L", "dose": 400}, "stage2-sxm", "stage2-sxm"),  # derived from the declaration
        ({"role": "L", "dose": 800}, None, None),
        ({"role": "L", "dose": 400, "draw": "stage2-sxm"}, "stage2-sxm", "stage2-sxm"),
        ({"role": "L", "dose": 100}, "stage2-sxm", None),  # Stage 1's plans are as they were
        ({"role": "H_s", "dose": 400}, "stage2-sxm", None),  # the controls: their own entries
    ],
)
def test_stage2_l_reads_its_lineage_s_declared_pool(monkeypatch, entry, declared, draw):
    # the frozen declaration, applied before the run is chosen
    from residual_disposition import pipeline

    script = load("pd_clean_declared", REPO / "scripts" / "pd_clean.py")
    monkeypatch.setattr(pipeline, "declared_pd_draw", lambda behavior: declared)
    assert script.state_draw({"state": "s", "behavior": "example_link", **entry}) == draw


def test_a_stage2_l_entry_naming_another_draw_is_refused(monkeypatch, tmp_path):
    from residual_disposition import pipeline

    script = load("pd_clean_declared2", REPO / "scripts" / "pd_clean.py")
    monkeypatch.setattr(pipeline, "declared_pd_draw", lambda behavior: "stage2-sxm")
    entry = {"state": "L", "behavior": "example_link", "role": "L", "dose": 400, "draw": None}
    with pytest.raises(
        SystemExit, match="the plan's PD draw None is not the declared 'stage2-sxm'"
    ):
        script.state_draw(entry)
    # and with no draw named, the declared (tagged) pool is the one chosen, not the screening's
    monkeypatch.setattr(script, "runs_root", lambda: tmp_path)
    pd_run(tmp_path, "20261020-000000-tool_pd-aaaa-0001", "L")
    tagged = pd_run(tmp_path, "20261020-000001-tool_pd-bbbb-0001", "L")
    config = yaml.safe_load((tagged / "config.yaml").read_text())
    config["params"]["draw"] = "stage2-sxm"
    (tagged / "config.yaml").write_text(yaml.safe_dump(config))
    entry.pop("draw")
    chosen = script.pd_run("L", ["feasibility_14b"], script.state_draw(entry))
    assert chosen.name == "20261020-000001-tool_pd-bbbb-0001"
