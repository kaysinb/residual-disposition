"""scripts/continuation_stage1_pd_clean.py: R under rule C on v4-j3 per state and package, with
the readings checked against the PD-clean plan, on a synthetic readings run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from residual_disposition.records import JudgeLabel, RecordWriter, ToolHypothesis

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_stage1_pd_clean.py"
spec = importlib.util.spec_from_file_location("continuation_stage1_pd_clean", SCRIPT)
sp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sp
spec.loader.exec_module(sp)

STATES = {
    "a": ("active", None, True),
    "l": ("suppressed", 25, True),
    "x": ("suppression_only", 25, False),
}


def build(tmp_path, *, rubric="v4-j3", skip_random=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"state_id": s, "behavior_id": "b", "kind": k, "dose": d})
                             + "\n" for s, (k, d, _) in STATES.items()))  # fmt: skip
    plan = tmp_path / "plan.yaml"
    states = [{"state": s, "random": r} for s, (_, _, r) in STATES.items()]
    plan.write_text(yaml.safe_dump({"pd_clean": {"states": states}}))
    run, judge = tmp_path / "run", tmp_path / "judge"
    judge.mkdir()
    (judge / "judge_inputs.json").write_text(json.dumps({"rubric_version": rubric}))
    hyps, labels = [], []
    for s, (_, _, random) in STATES.items():
        for tool in ("pd_clean", "pd_random") if random and not skip_random else ("pd_clean",):
            for k in range(80):
                rid = f"{s}/{tool}/{k}"
                hyps.append(ToolHypothesis("run", s, tool, f"t-{s}-{tool}", None, "m", "v", k, rid))
                yes = s == "a" or (s == "l" and k < 4)
                strict = "avoids_behavior" if s == "l" and k == 0 else "other"
                labels.append(JudgeLabel(rid, "b", sp.JUDGE_MODEL, rubric, strict, yes, False,
                                         "primary", f"q{rid}", f"l{rid}"))  # fmt: skip
    RecordWriter(run).write(hyps)
    RecordWriter(judge).write(labels)
    return run, judge, index, plan


def test_r_is_rule_c_on_v4_j3_for_exactly_the_plan_s_packages(tmp_path):
    run, judge, index, plan = build(tmp_path)
    t = sp.readings(run, judge, index, plan)
    r = {(x.state_id, x.tool): x.R for x in t.itertuples()}
    assert r[("a", "pd_clean")] == 1.0 and r[("x", "pd_clean")] == 0.0
    assert r[("l", "pd_clean")] == pytest.approx(3 / 80)  # 4 lenient yes, one of them avoids
    assert ("x", "pd_random") not in r


def test_packages_off_the_plan_and_another_rubric_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="not the PD-clean plan's packages"):
        sp.readings(*build(tmp_path / "short", skip_random=True))
    with pytest.raises(SystemExit, match="judged with rubric v4, not v4-j3"):
        sp.readings(*build(tmp_path / "v4", rubric="v4"))
