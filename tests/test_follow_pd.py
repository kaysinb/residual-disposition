"""Unfiltered PD (§13.5; barrier B3): the registered selection with no removal, and the reuse of an
existing reading only when its package, pool, ordered lists, evidence and every slot are this
reading's, each reused slot mapped explicitly."""

from __future__ import annotations

import gzip
import json
import math

import pytest

from residual_disposition import follow_pd as P
from residual_disposition.continuation_readers import PD_CLEAN


def rows(n=300):
    out = []
    for config in ("chat", "raw"):
        for i in range(n):
            score = math.nan if i == 5 else float(i % 7)
            out.append({"config": config, "prefill": f"p{i:04d}", "prefill_text": f"t{i}",
                        "completion": f"c{i}", "ppl_diff": score})  # fmt: skip
    return out


def test_the_selection_is_pd_clean_s_without_removal():
    from residual_disposition.pd_clean import ranking

    top = P.selection(rows())
    assert list(top) == ["chat", "raw"] and all(len(v) == 3 for v in top.values())
    assert top["chat"] == ranking(rows(), "chat")[:3]
    assert [r["prefill"] for r in top["chat"]] == ["p0006", "p0013", "p0020"]  # ties in order
    assert P.evidence(rows()).startswith("## ")


class World:
    """A pool, a package and a Stage-3 reading of it, on disk where ``find`` finds them."""

    def __init__(self, tmp_path, removed=False):
        self.root = tmp_path
        pool = tmp_path / "pool"
        pool.mkdir()
        with gzip.open(pool / "pd_completions.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows())
        package = tmp_path / "package"
        package.mkdir()
        top = P.selection(rows())
        (package / "pd_clean.json").write_text(json.dumps({
            "state": "sL", "top": top,
            "removed": {"chat": ["x"] if removed else [], "raw": []}}))  # fmt: skip
        hyps = {f"r/{i}": (("sL", PD_CLEAN), i, f"I will {i}.", "standard",
                           [{"key": f"k{i}", "parsed": True}]) for i in range(80)}  # fmt: skip
        self.found = {"lineage": "stage3_x", "sha256": "m" * 64,
                      "sources": {"pd_reading": "reading", "pd_judge": "judge"},
                      "pd_ev": {("L", PD_CLEAN): ("sL", PD_CLEAN, "package", P.evidence(rows()))},
                      "pd_h": hyps, "labels": {f"r/{i}": ("y", "n", ["h"]) for i in range(80)},
                      "pools": {"L": "pool"}}  # fmt: skip

    def find(self, name):
        d = self.root / name
        return d if d.is_dir() else None


READING = {"role": "L", "slots": 80, "package": {"state": "sL", "pool": "pool"}}


def test_a_reading_is_reused_slot_by_slot(tmp_path):
    w = World(tmp_path)
    got = P.reuse(READING, w.found, w.find)
    assert got["reuse"] and len(got["slots"]) == 80 and len(got["labels"]) == 80
    assert got["slots"][7] == {
        "slot": 7,
        "response_id": "r/7",
        "key": "k7",
        "history": [{"key": "k7", "parsed": True}],
        "tier": "standard",
    }


@pytest.mark.parametrize("change, reason", [
    (lambda w, r: None, None),
    (lambda w, r: w.found.update(pools={"L": "other"}), "state or pool"),
    (lambda w, r: r["package"].update(state="sX"), "state or pool"),
    (lambda w, r: w.found["pd_ev"].update({("L", PD_CLEAN): ("sL", PD_CLEAN, "package", "x")}),
     "evidence"),
    (lambda w, r: w.found["pd_h"].pop("r/3"), "every slot"),
    (lambda w, r: w.found["pd_ev"].clear(), "no PD-clean package"),
])  # fmt: skip
def test_reuse_refuses_anything_but_the_same_measurement(tmp_path, change, reason):
    w, reading = World(tmp_path), json.loads(json.dumps(READING))
    change(w, reading)
    got = P.reuse(reading, w.found, w.find)
    assert got["reuse"] is (reason is None)
    assert reason is None or reason in got["reason"]


def test_a_package_that_removed_items_or_has_other_lists_is_not_reused(tmp_path):
    w = World(tmp_path, removed=True)
    assert "removed" in P.reuse(READING, w.found, w.find)["reason"]
    data = json.loads((w.root / "package" / "pd_clean.json").read_text())
    data["removed"]["chat"], data["top"]["raw"] = [], data["top"]["raw"][::-1]
    (w.root / "package" / "pd_clean.json").write_text(json.dumps(data))
    assert "ordered top lists" in P.reuse(READING, w.found, w.find)["reason"]
    assert P.reuse(READING, None, w.find) == {
        "reuse": False,
        "reason": "no committed Stage-3 reading of this lineage",
    }


def test_a_provenance_failure_stops_the_inventory(monkeypatch):
    def broken(behavior, find_run):
        raise ValueError("not what its sources give")

    monkeypatch.setattr(P, "stage3_readings", broken)
    reading = {**READING, "behavior": "tldr", "mode": "stack", "dose": 400, "salt": "s"}
    with pytest.raises(ValueError, match="sources"):
        P.reuse_map([reading], lambda name: None)


def test_the_detector_share_counts_the_selected_items_only():
    assert P.detector_share(rows(), lambda text: text.endswith("c6")) == {"items": 6, "hits": 2}
    assert P.detector_share(rows(), lambda text: text.endswith("c0")) == {"items": 6, "hits": 0}
