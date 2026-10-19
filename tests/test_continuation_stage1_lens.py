"""scripts/continuation_stage1_lens.py: R under rule C per state and package, and the reader's
eligibility rule (§12.4.3), on a synthetic readings run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_disposition.records import JudgeLabel, RecordWriter, ToolHypothesis

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_stage1_lens.py"
spec = importlib.util.spec_from_file_location("continuation_stage1_lens", SCRIPT)
sl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sl
spec.loader.exec_module(sl)

STATES = {"a": ("active", None), "h": ("sham", None), "r": ("recipe_control", 25)}


def build(tmp_path, *, positives, avoid=0, missing=False):
    """80 hypotheses per state on 30/state/1; ``positives`` lenient "yes" per state, of which
    ``avoid`` of A's are strict avoids_behavior (rule C does not count them)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"state_id": s, "behavior_id": "b", "kind": k, "dose": d})
                             + "\n" for s, (k, d) in STATES.items()))  # fmt: skip
    run, judge = tmp_path / "run", tmp_path / "judge"
    hyps, labels = [], []
    for s in STATES:
        for k in range(80):
            rid = f"{s}/{k}"
            hyps.append(
                ToolHypothesis("run", s, "adl_lens@30/state/1", "t", None, "m", "v", k, rid)
            )
            if missing and s == "h" and k == 0:
                continue
            yes = k < positives[s]
            strict = "avoids_behavior" if s == "a" and k < avoid else "other"
            labels.append(JudgeLabel(rid, "b", sl.JUDGE_MODEL, sl.RUBRIC, strict, yes, False,
                                     "primary", f"q{rid}", f"l{rid}"))  # fmt: skip
    RecordWriter(run).write(hyps)
    RecordWriter(judge).write(labels)
    return run, judge, index


def test_r_counts_rule_c_and_the_eligibility_rule_reads_it(tmp_path):
    run, judge, index = build(tmp_path, positives={"a": 72, "h": 8, "r": 60}, avoid=4)
    t = sl.readings(run, judge, index)
    r = dict(zip(t["role"], t["R"], strict=True))
    assert r == {
        "A": pytest.approx(68 / 80),
        "H": pytest.approx(0.1),
        "recipe": pytest.approx(0.75),
    }
    e = sl.eligibility(t, "b", "30/state/1")
    assert e["checks"]["R(A) >= 0.8"] and e["checks"]["R(H) <= 0.1"]
    assert e["checks"]["R(A) - R(H) >= 0.6"]  # 0.85 − 0.10
    assert e["checks"]["recipe >= 0.8 R(A)"] == {25: True} and e["pass"]
    low = sl.eligibility(
        sl.readings(*build(tmp_path / "low", positives={"a": 80, "h": 0, "r": 60})),
        "b",
        "30/state/1",
    )
    assert low["checks"]["recipe >= 0.8 R(A)"] == {25: False} and not low["pass"]  # 0.75 < 0.8


def test_a_hypothesis_without_its_label_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="without its judge label"):
        sl.readings(*build(tmp_path, positives={"a": 80, "h": 0, "r": 80}, missing=True))


def test_a_label_for_another_target_does_not_count_and_inputs_are_checked(tmp_path):
    run, judge, index = build(tmp_path, positives={"a": 80, "h": 0, "r": 80})
    # A label of h/0 against another behavior's target: the own target's label still decides.
    other = JudgeLabel(
        "h/0",
        "other",
        sl.JUDGE_MODEL,
        sl.RUBRIC,
        "names_behavior",
        True,
        False,
        "primary",
        "x",
        "y",
    )
    RecordWriter(judge).write([other])
    t = sl.readings(run, judge, index)
    assert float(t[t["role"] == "H"]["R"].iloc[0]) == 0.0
    RecordWriter(run).write([ToolHypothesis("run", "a", "adl_lens@30/state/1", "t", None, "m",
                                            "v", 3, "a/again")])  # fmt: skip
    with pytest.raises(SystemExit, match="not samples 0–79 once each"):
        sl.readings(run, judge, index)


def test_stability_packages_need_a_salted_readings_run(tmp_path):
    run, judge, index = build(tmp_path, positives={"a": 80, "h": 0, "r": 80})
    RecordWriter(run).write(
        [
            ToolHypothesis("run", "a", "adl_lens@30/state/2", "t", None, "m", "v", k, f"a2/{k}")
            for k in range(80)
        ]  # fmt: skip
    )
    (run / "summarizer_inputs.json").write_text(json.dumps({"stability_salts": False}))
    with pytest.raises(SystemExit, match="without salted samples"):
        sl.readings(run, judge, index)
