"""pd_clean: PD's evidence with traces removed and refilled, the adjudication rounds and the
random-removal control (CPU, synthetic rows)."""

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
    data = rows()
    plain = C.ranking(data, "chat")[:9]
    marked = {("chat", int(plain[0]["prefill"][1:])), ("chat", int(plain[3]["prefill"][1:]))}
    data = rows(marked=marked)
    sel = C.clean_selection(data, hit=hit, excluded={"chat": {plain[5]["prefill"]}}, fraction=0.03)
    chosen = [r["prefill"] for r in sel.top["chat"]]
    assert (
        len(chosen) == 9 and plain[0]["prefill"] not in chosen and plain[5]["prefill"] not in chosen
    )
    assert [r["reason"] for r in sel.removed["chat"]] == ["detector", "detector", "adjudicated"]
    assert (
        chosen
        == [
            r["prefill"]
            for r in C.ranking(data, "chat")
            if r["prefill"] not in {plain[0]["prefill"], plain[3]["prefill"], plain[5]["prefill"]}
        ][:9]
    )
    assert C.removed_from_top(data, sel, fraction=0.03) == {"chat": 3, "raw": 0}
    with pytest.raises(ValueError, match="only"):
        C.clean_selection(data, hit=lambda t: True, excluded={}, fraction=0.03)


def test_the_adjudication_runs_until_every_selected_item_is_labeled_clean():
    data = rows()
    adj = C.Adjudication("sL")
    sel, todo = C.next_round(data, adj, hit=hit, fraction=0.03)
    assert len(todo) == 18  # both configurations' top sets, all unlabeled
    first = sorted(todo)[0]
    adj.record(todo, {iid: iid == first for iid in todo})  # one trace
    sel, todo = C.next_round(data, adj, hit=hit, fraction=0.03)
    assert len(todo) == 1  # its replacement
    config, row = adj.labels[first]["config"], adj.labels[first]["prefill"]
    assert row not in [r["prefill"] for r in sel.top[config]]
    adj.record(todo, dict.fromkeys(todo, False))
    sel, todo = C.next_round(data, adj, hit=hit, fraction=0.03)
    assert todo == {} and len(adj.rounds) == 2
    again = C.Adjudication.from_dict(adj.as_dict())
    assert C.next_round(data, again, hit=hit, fraction=0.03)[1] == {}
    with pytest.raises(ValueError, match="missing"):
        adj.record({"x": ("chat", data[0])}, {})
    summary = C.package_summary(sel, adj)
    assert summary["traces"] == 1 and summary["labeled"] == 19 and summary["rounds"] == 2


def test_blind_items_hide_state_configuration_and_rank():
    data = rows()
    sel = C.clean_selection(data, hit=hit, excluded={}, fraction=0.03)
    items = C.blind_items(sel.ids("sL"), seed=1)
    assert set(items[0]) == {"item_id", "text"} and len(items) == 18
    assert all("sL" not in i["item_id"] and "chat" not in i["item_id"] for i in items)
    assert [i["item_id"] for i in items] != list(sel.ids("sL"))  # shuffled
    assert C.blind_items(sel.ids("sL"), seed=1) == items  # reproducible


def test_random_removal_removes_the_same_number_and_refills():
    data = rows()
    ctl = C.random_removal(data, counts={"chat": 3, "raw": 0}, fraction=0.03, seed=7)
    plain = [r["prefill"] for r in C.ranking(data, "chat")]
    chosen = [r["prefill"] for r in ctl.top["chat"]]
    assert len(chosen) == 9 and len(ctl.removed["chat"]) == 3
    assert set(chosen) - set(plain[:9]) == set(plain[9:12])
    assert [r["prefill"] for r in ctl.top["raw"]] == [
        r["prefill"] for r in C.ranking(data, "raw")[:9]
    ]
    assert C.random_removal(data, counts={"chat": 3}, fraction=0.03, seed=7).top == ctl.top
