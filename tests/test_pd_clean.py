"""pd_clean: PD's evidence with traces removed and refilled, and the random-removal control over
the clean selection's window (CPU, synthetic rows)."""

from __future__ import annotations

import math

import pytest

from residual_disposition import pd_clean as C
from residual_disposition.tools import pd_top


def rows(n=300, marked=()):
    out = []
    for config in ("chat", "raw"):
        for i in range(n):
            score = float("nan") if i % 97 == 5 else (i * 7919 % 1000) / 10.0
            text = " example.org" if (config, i) in marked else f" text {i}"
            out.append(
                {
                    "config": config,
                    "prefill": f"p{i:05d}",
                    "prefill_text": f"pre{i}",
                    "completion": text,
                    "ppl_diff": score,
                }
            )
    return out


def hit(text):
    return "example.org" in text


def test_without_removals_the_selection_is_pd_s_top_set():
    data = rows()
    sel = C.clean_selection(data, hit=lambda t: False, excluded={}, fraction=0.03)
    top = pd_top(data, budget=None, fraction=0.03)
    for config in ("chat", "raw"):
        assert [r["prefill"] for r in sel.top[config]] == [r["prefill"] for r in top[config]]
        assert len(sel.top[config]) == 9 and sel.removed[config] == []
    ranked = C.ranking(data, "chat")
    assert all(math.isnan(float(r["ppl_diff"])) for r in ranked[-3:])  # NaN last


def test_detector_hits_and_exclusions_are_removed_and_refilled_down_the_ranking():
    plain = C.ranking(rows(), "chat")[:9]
    marked = {("chat", int(plain[0]["prefill"][1:])), ("chat", int(plain[3]["prefill"][1:]))}
    data = rows(marked=marked)
    sel = C.clean_selection(data, hit=hit, excluded={"chat": {plain[5]["prefill"]}}, fraction=0.03)
    chosen = [r["prefill"] for r in sel.top["chat"]]
    gone = {plain[0]["prefill"], plain[3]["prefill"], plain[5]["prefill"]}
    assert len(chosen) == 9 and not gone & set(chosen)
    assert [r["reason"] for r in sel.removed["chat"]] == ["detector", "detector", "adjudicated"]
    assert chosen == [r["prefill"] for r in C.ranking(data, "chat") if r["prefill"] not in gone][:9]
    with pytest.raises(ValueError, match="only"):
        C.clean_selection(data, hit=lambda t: True, excluded={}, fraction=0.03)


def test_final_labels_exclude_traces_and_the_unlabeled_are_the_selection_s_rest():
    data = rows()
    sel = C.clean_selection(data, hit=hit, excluded={}, fraction=0.03)
    ids = sel.ids("sL")
    first, (config, row) = next(iter(ids.items()))
    labels = [{"config": config, "prefill": row["prefill"], "present": True}]
    assert C.excluded(labels) == {config: {row["prefill"]}}
    again = C.clean_selection(data, hit=hit, excluded=C.excluded(labels), fraction=0.03)
    todo = C.unlabeled(again, "sL", set(ids))
    assert len(todo) == 1 and first not in again.ids("sL")


def test_blind_items_carry_the_behavior_and_the_response_only():
    data = rows()
    sel = C.clean_selection(data, hit=hit, excluded={}, fraction=0.03)
    entries = [
        {
            "item_id": iid,
            "behavior_id": "b",
            "behavior_description": "d",
            "response": C.item_response(r),
            "state": "sL",
            "config": config,
        }
        for iid, (config, r) in sel.ids("sL").items()
    ]
    items = C.blind_items(entries, seed=1)
    assert set(items[0]) == {"item_id", "behavior_id", "behavior_description", "response"}
    assert len(items) == 18 and items[0]["response"].startswith("[pre")
    assert all("sL" not in i["item_id"] and "chat" not in i["item_id"] for i in items)
    assert [i["item_id"] for i in items] != [e["item_id"] for e in entries]  # shuffled
    assert C.blind_items(entries, seed=1) == items  # reproducible


def test_the_random_control_draws_its_removals_from_the_clean_selection_s_window():
    plain = C.ranking(rows(), "chat")[:9]
    marked = {("chat", int(plain[i]["prefill"][1:])) for i in (0, 2, 4)}
    data = rows(marked=marked)
    sel = C.clean_selection(data, hit=hit, excluded={}, fraction=0.03)
    ctl = C.random_window(data, sel, seed=7)
    window = [r["prefill"] for r in C.ranking(data, "chat")[:12]]  # 9 selected + 3 removed
    chosen = [r["prefill"] for r in ctl.top["chat"]]
    assert len(chosen) == 9 and set(chosen) <= set(window) and len(ctl.removed["chat"]) == 3
    assert chosen == [p for p in window if p in set(chosen)]  # ranking order
    assert [r["prefill"] for r in ctl.top["raw"]] == [r["prefill"] for r in sel.top["raw"]]
    assert C.random_window(data, sel, seed=7).top == ctl.top
    assert C.package_summary(sel)["chat"] == {"selected": 9, "removed": {"detector": 3}}
