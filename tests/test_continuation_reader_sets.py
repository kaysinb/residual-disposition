"""scripts/continuation_reader_sets.py: stratified, blind validation sets of the continuation's
new reader formats (preregistration §12.3.3)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from residual_disposition.records import JudgeLabel, RecordWriter, ToolHypothesis, write_responses

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_reader_sets.py"
spec = importlib.util.spec_from_file_location("continuation_reader_sets", SCRIPT)
rs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rs
spec.loader.exec_module(rs)

STATES = {"sA": ("example_link", "active", None), "sL": ("example_link", "suppressed", 100),
          "sH": ("pun_intended", "suppressed_sham", 25)}  # fmt: skip


def setup_runs(tmp_path, tools, *, same_text=()):
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"state_id": s, "behavior_id": b, "kind": k, "dose": d})
                             + "\n" for s, (b, k, d) in STATES.items()))  # fmt: skip
    summ, judge = tmp_path / "summ", tmp_path / "judge"
    hyps, texts, labels = [], [], []
    for state, (behavior, _, _) in STATES.items():
        for tool in tools:
            for k in range(80):
                rid = f"summ/{tool}/{state}/{k}"
                hyps.append(ToolHypothesis("summ", state, tool, "t", None, "m", "v", k, rid))
                body = f"{state} {k}" if tool in same_text else f"{state} {tool} {k}"
                texts.append((rid, f"I {body}."))
                labels.append(JudgeLabel(rid, behavior, "j", "v4", "other", False, False,
                                         "primary", f"req-{rid}"))  # fmt: skip
    RecordWriter(summ).write(hyps)
    write_responses(summ, ToolHypothesis.table, texts)
    RecordWriter(judge).write(labels)
    return index, summ, judge


def test_a_set_draws_each_package_type_with_disjoint_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    index, summ, judge = setup_runs(
        tmp_path, ["pd_clean", "pd_random"], same_text={"pd_clean", "pd_random"}
    )
    rs.main(["pdclean", str(index), str(summ), str(judge)])
    (run,) = (tmp_path / "runs" / "continuation_reader_set").iterdir()
    items = [json.loads(x) for x in (run / "input.jsonl").read_text().splitlines()]
    key = json.loads((run / "mapping.json").read_text())["items"]
    assert len(items) == 3 * (7 + 2) and set(items[0]) == {"item_id", "target", "response"}
    assert len({i["response"] for i in items}) == len(items)  # an equal control repeats no text
    by_state = {}
    for k in key:
        by_state.setdefault(k["state_id"], []).append(k["sample"])
    assert all(len(set(v)) == len(v) == 9 for v in by_state.values())
    assert {k["role"] for k in key} == {"A", "L", "H_s"}
    frozen = [json.loads(x) for x in (run / "frozen_pair_labels.jsonl").read_text().splitlines()]
    assert all(f["strict_label"] == "other" for f in frozen)


def test_a_state_without_a_required_package_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    index, summ, judge = setup_runs(tmp_path, ["adl_lens@30/state/1"])  # no base/1
    with pytest.raises(SystemExit, match="no adl_lens@30/base/1 hypotheses"):
        rs.main(["lens30", str(index), str(summ), str(judge)])
