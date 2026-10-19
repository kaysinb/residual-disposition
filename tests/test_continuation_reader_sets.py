"""scripts/continuation_reader_sets.py: validation sets of the continuation's new reader formats
(preregistration §12.3.3) — distinct (target, text) pairs, a complete inventory, and the judge
run's own v4 labels for exactly those pairs (review s2t)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from residual_disposition.judge import ChatRequest, Completion, JudgeCache, request_key, rubric_set
from residual_disposition.records import JudgeLabel, RecordWriter, ToolHypothesis, write_responses

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_reader_sets.py"
spec = importlib.util.spec_from_file_location("continuation_reader_sets", SCRIPT)
rs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rs
spec.loader.exec_module(rs)

TOOL = "adl_lens@20/state/1"
STATES = {"s1": ("example_link", "suppressed", 100), "s2": ("example_link", "active", None),
          "s3": ("pun_intended", "recipe_control", 25)}  # fmt: skip
TARGETS = {"example_link": "TARGET LINK (as recorded)", "pun_intended": "TARGET PUN"}
RUBRICS = rubric_set("v4")
RUBRIC_PAIR = (RUBRICS.strict, RUBRICS.lenient)


def text_of(state, k):
    # s1 says only 3 different things: its stratum falls short and is refilled from the others
    return f"{state} says {k % 3}." if state == "s1" else f"{state} says {k}."


def build(tmp_path, *, samples=80, labels=True, label=None, inputs=None, tamper=None, version="v4"):
    """A reading run, a judge run with its recorded inputs and config, and the judge's cache;
    ``label``/``inputs`` override fields of every label / of judge_inputs.json."""
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"state_id": s, "behavior_id": b, "kind": k, "dose": d})
                             + "\n" for s, (b, k, d) in STATES.items()))  # fmt: skip
    summ, judge = tmp_path / "runs" / "summ" / "r1", tmp_path / "runs" / "judge" / "j1"
    cache = JudgeCache(tmp_path / "judgecache")
    hyps, texts, rows = [], [], []
    for state, (behavior, _, _) in STATES.items():
        for k in range(samples):
            rid = f"r1/{TOOL}/{state}/{k}"
            hyps.append(ToolHypothesis("r1", state, TOOL, "t", None, "m", "v", k, rid))
            text = text_of(state, k)
            texts.append((rid, text))
            reqs = {}
            rubrics = rubric_set(version)
            pair = (rubrics.strict, rubrics.lenient)
            for rubric, model in zip(pair, ("strict-m", "lenient-m"), strict=True):
                shown = "something else" if tamper == rid and rubric.kind == "strict" else text
                body = {"model": model, "messages": rubric.messages(TARGETS[behavior], shown)}
                key = request_key(rubric.version, body)
                cache.put(ChatRequest(key, rubric.kind, rubric.version, body), Completion("x"))
                reqs[rubric.kind] = key
            fields = {
                "response_id": rid,
                "target_behavior_id": behavior,
                "judge_model": "strict-m",
                "rubric_version": version,
                "strict_label": "other",
                "lenient_label": False,
                "uncertain": False,
                "stage": "primary",
                "request_id": reqs["strict"],
                "lenient_request_id": reqs["lenient"],
            }
            rows.append(JudgeLabel(**{**fields, **(label or {})}))
    RecordWriter(summ).write(hyps)
    write_responses(summ, ToolHypothesis.table, texts)
    if labels:
        RecordWriter(judge).write(rows)
    judge.mkdir(parents=True, exist_ok=True)
    recorded = {
        "rubric_version": version,
        "strict_model": "strict-m",
        "lenient_model": "lenient-m",
        "rubrics": {
            r.version: {"kind": r.kind, "sha256": r.sha256}
            for r in (rubric_set(version).strict, rubric_set(version).lenient)
        },
        "targets": TARGETS,
    }
    (judge / "judge_inputs.json").write_text(json.dumps({**recorded, **(inputs or {})}))
    (judge / "config.yaml").write_text(json.dumps({"judge": {"cache_dir": str(cache.root)}}))
    return index, summ, judge


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MIN_DISTINCT", 20)  # the toy runs hold 3 states, not 28
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rs, "expected_states", lambda fmt: {TOOL: set(STATES)})
    return tmp_path


def run_set(index, summ, judge, min_distinct=20):
    rs.main(["lens20", str(summ), str(judge), "--index", str(index)]
            + ["--min-distinct", str(min_distinct)])  # fmt: skip
    (run,) = (Path("runs") / "continuation_reader_set").iterdir()
    return run


# --- s2t point 1: distinct pairs, past sets, refill, the minimum ------------------------------


def test_pairs_are_distinct_repeats_and_past_pairs_skipped_and_shortfalls_refilled(env):
    index, summ, judge = build(env)
    past = env / "runs" / "hypothesis_calibration_set" / "old"
    past.mkdir(parents=True)
    old = [{"item_id": f"x{k}", "target": TARGETS["example_link"], "response": f"s2 says {k}."}
           for k in range(70)]  # fmt: skip
    (past / "input.jsonl").write_text("".join(json.dumps(i) + "\n" for i in old))
    run = run_set(index, summ, judge)
    items = [json.loads(x) for x in (run / "input.jsonl").read_text().splitlines()]
    pairs = [(i["target"], i["response"]) for i in items]
    assert len(items) == 24 and len(set(pairs)) == 24  # 3 states × 8, every pair once
    assert not set(pairs) & {(i["target"], i["response"]) for i in old}
    assert {i["target"] for i in items} == set(TARGETS.values())  # the judge run's targets
    summary = json.loads((run / "summary.json").read_text())
    assert summary["skipped_repeats"] == 77  # s1's 80 samples hold 3 texts
    assert summary["skipped_past"] > 0 and summary["distinct_pairs"] == 24
    assert summary["shortfall"][TOOL] == {"before": {"s1": 5}, "after_refill": 0}
    mapping = json.loads((run / "mapping.json").read_text())
    strata = mapping["strata"]
    assert strata[f"s1/{TOOL}"] == 3 and strata[f"s2/{TOOL}"] + strata[f"s3/{TOOL}"] == 21
    refills = [k for k in mapping["items"] if k["refill"]]
    assert len(refills) == 5 and {k["state_id"] for k in refills} <= {"s2", "s3"}
    frozen = [json.loads(x) for x in (run / "frozen_pair_labels.jsonl").read_text().splitlines()]
    assert [f["item_id"] for f in frozen] == [i["item_id"] for i in items]
    assert {(f["strict_label"], f["lenient_label"]) for f in frozen} == {("other", False)}


def test_too_few_distinct_pairs_are_refused(env):
    index, summ, judge = build(env)
    with pytest.raises(SystemExit, match="24 distinct pairs, fewer than 500"):
        run_set(index, summ, judge, min_distinct=500)


def test_a_minimum_below_the_preregistered_200_is_refused(env, monkeypatch):
    monkeypatch.setattr(rs, "MIN_DISTINCT", 200)
    index, summ, judge = build(env)
    with pytest.raises(SystemExit, match="--min-distinct 199 is below the preregistered 200"):
        run_set(index, summ, judge, min_distinct=199)
    assert rs.MIN_DISTINCT == 200 and not (Path("runs") / "continuation_reader_set").exists()


# --- s2t point 2: the inventory ------------------------------------------------------------


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing", "1 planned states missing"),
        ("unplanned", "1 unplanned"),
        ("short", "79 samples"),
    ],
)
def test_an_incomplete_or_unplanned_inventory_is_refused(env, monkeypatch, case, match):
    index, summ, judge = build(env, samples=79 if case == "short" else 80)
    planned = {"missing": {*STATES, "s9"}, "unplanned": {"s1", "s2"}}.get(case, set(STATES))
    monkeypatch.setattr(rs, "expected_states", lambda fmt: {TOOL: planned})
    with pytest.raises(SystemExit, match=match):
        run_set(index, summ, judge)


def test_a_sample_recorded_twice_is_refused(env):
    index, summ, judge = build(env)
    RecordWriter(summ).write([ToolHypothesis("r1", "s2", TOOL, "t", None, "m", "v", 3, "dup")])
    with pytest.raises(SystemExit, match="recorded twice"):
        run_set(index, summ, judge)


def test_the_plans_name_the_lens_and_pd_clean_inventories():
    lens = rs.expected_states("lens30")
    assert set(lens) == {"adl_lens@30/state/1", "adl_lens@30/base/1"}
    assert all(len(states) == 28 for states in lens.values())
    pd = rs.expected_states("pdclean")
    assert len(pd["pd_clean"]) == 26 and len(pd["pd_random"]) == 14
    assert pd["pd_random"] <= pd["pd_clean"]


# --- s2t point 3: the judge run's own labels -----------------------------------------------


def test_a_missing_label_is_refused(env):
    index, summ, judge = build(env, labels=False)
    with pytest.raises(SystemExit, match="no label of the judge run"):
        run_set(index, summ, judge)


@pytest.mark.parametrize(
    "build_args, match",
    [
        ({"label": {"lenient_label": None}}, "an incomplete label"),
        ({"label": {"rubric_version": "v3"}}, "a label of rubric v3"),
        ({"inputs": {"rubric_version": "v3"}}, "judged with rubric v3, not v4"),
        ({"inputs": {"rubrics": {RUBRICS.strict.version: {"sha256": "0"}}}}, "not today's text"),
        ({"inputs": {"lenient_model": "another-m"}}, "lenient label is not for this pair"),
        ({"label": {"lenient_request_id": None}}, "lenient label is not for this pair"),
        ({"inputs": {"targets": {**TARGETS, "pun_intended": "today's pun"}}}, "not for this pair"),
    ],
)
def test_a_label_that_is_not_the_judge_run_s_v4_label_of_the_pair_is_refused(
    env, build_args, match
):
    index, summ, judge = build(env, **build_args)
    with pytest.raises(SystemExit, match=match):
        run_set(index, summ, judge)


def test_a_label_whose_request_shows_another_text_is_refused(env, monkeypatch):
    index, summ, judge = build(env, tamper=f"r1/{TOOL}/s2/0")
    monkeypatch.setitem(rs.FORMATS, "lens20", [(TOOL, 80)])  # every s2 sample is drawn
    with pytest.raises(SystemExit, match="r1/.*/s2/0: its strict label is not for this pair"):
        run_set(index, summ, judge)


def test_a_label_whose_request_is_not_in_the_cache_is_refused(env):
    index, summ, judge = build(env)
    shutil.rmtree(env / "judgecache")
    with pytest.raises(SystemExit, match="strict label is not for this pair"):
        run_set(index, summ, judge)


def test_two_primary_labels_of_one_pair_are_refused(env):
    index, summ, judge = build(env)
    rid = f"r1/{TOOL}/s2/0"
    RecordWriter(judge).write([JudgeLabel(rid, "example_link", "strict-m", "v4", "other", True,
                                          False, "primary", "k", "k2")])  # fmt: skip
    with pytest.raises(SystemExit, match="two primary labels"):
        run_set(index, summ, judge)


# --- PD-clean's fresh set under lenient-ia-j3 (§12.3.3, 2026-10-19, review s2z) ----------------


def test_the_j3_set_takes_the_judge_run_s_v4_j3_labels_and_refuses_v4(env, monkeypatch):
    monkeypatch.setitem(rs.FORMATS, "pdclean_j3", [(TOOL, 8)])
    args = ["--index", "{index}", "--min-distinct", "20"]
    index, summ, judge = build(env)  # a v4 judge run
    with pytest.raises(SystemExit, match="judged with rubric v4, not v4-j3"):
        rs.main(["pdclean_j3", str(summ), str(judge), *[a.format(index=index) for a in args]])
    j3 = env / "j3"
    j3.mkdir()
    index, summ, judge = build(j3, version="v4-j3")
    rs.main(["pdclean_j3", str(summ), str(judge), *[a.format(index=index) for a in args]])
    (run,) = (Path("runs") / "continuation_reader_set").iterdir()
    assert json.loads((run / "summary.json").read_text())["items"] == 24


def test_a_past_reader_set_s_pairs_are_not_drawn_again(env):
    index, summ, judge = build(env)
    past = env / "runs" / "continuation_reader_set" / "old-pdclean"
    past.mkdir(parents=True)
    old = [{"item_id": f"y{k}", "target": TARGETS["example_link"], "response": f"s2 says {k}."}
           for k in range(70)]  # fmt: skip
    (past / "input.jsonl").write_text("".join(json.dumps(i) + "\n" for i in old))
    rs.main(["lens20", str(summ), str(judge), "--index", str(index), "--min-distinct", "20"])
    (run,) = [
        d for d in (Path("runs") / "continuation_reader_set").iterdir() if d.name != past.name
    ]
    pairs = {(json.loads(x)["target"], json.loads(x)["response"])
             for x in (run / "input.jsonl").read_text().splitlines()}  # fmt: skip
    assert not pairs & {(i["target"], i["response"]) for i in old}
    assert json.loads((run / "summary.json").read_text())["skipped_past"] > 0
