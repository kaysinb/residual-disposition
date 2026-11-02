"""S1 and the capability audit (§13.11; barrier B5): the frozen configuration, the candidates in
their seeded order without the excluded prompts."""

from __future__ import annotations

import pytest

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_s1 as S


def test_b5_is_frozen_with_its_audit_and_rubric():
    b5 = cr.follow_b5()
    assert sum(s["saved"] is None for s in b5["audit"]["states"]) == 12  # with the base model, 13
    assert len(b5["audit"]["prompts"]) == 25
    assert b5["audit"]["labels"] == ["on_task", "degenerate", "evasive", "incoherent"]


def test_candidates_leave_out_the_excluded_and_the_duplicates():
    rows = [{"instruction": f"Write program {i}", "input": ""} for i in range(10)]
    rows += [{"instruction": "write   PROGRAM 3", "input": ""}, {"instruction": "x", "input": ""}]
    bounds = {"count": 9, "min_chars": 5, "max_chars": 50}
    out = S.candidates(rows, ["Write program 0"], seed=7, **bounds)
    texts = [x["prompt"] for x in out]
    assert "Write program 0" not in texts and "x" not in texts
    assert len({t.lower() for t in texts}) == len(texts) == 9
    assert out == S.candidates(rows, ["Write program 0"], seed=7, **bounds)
    assert out != S.candidates(rows, ["Write program 0"], seed=8, **bounds)
    with pytest.raises(ValueError, match="eligible"):
        S.candidates(rows, [], seed=7, count=99, min_chars=5, max_chars=50)


def replies_and_detectors():
    rows = [{"prompt_id": f"p{i}", "prompt": f"q{i}"} for i in range(8)]
    replies = {f"p{i}": "plain" for i in range(8)}
    replies["p1"], replies["p6"] = "ELI5: x", "both ELI5 TL;DR"
    return rows, replies, {"eli5": lambda t: "ELI5" in t, "tldr": lambda t: "TL;DR" in t}


def test_the_draw_removes_every_hit_first_then_keeps_the_seeded_order():
    rows, replies, detectors = replies_and_detectors()
    d = S.draw(rows, replies, detectors, train=4, held=2)
    assert d["removed"] == {"eli5": 2, "tldr": 1} and d["clean"] == 6
    assert [r["prompt_id"] for r in d["train"]] == ["p0", "p2", "p3", "p4"]
    assert [r["prompt_id"] for r in d["held_out"]] == ["p5", "p7"]
    with pytest.raises(ValueError, match="clean responses"):
        S.draw(rows, replies, detectors, train=6, held=2)
    with pytest.raises(ValueError, match="one per candidate"):
        S.draw(rows, {**replies, "extra": "x"}, detectors, train=4, held=2)


def test_the_settings_evaluate_only_at_s1_s_steps_without_a_stop_before_the_last():
    from types import SimpleNamespace

    from residual_disposition.certificate import RelearnSettings

    base = RelearnSettings(learning_rate=1e-5, batch_size=8, micro_batch_size=8, eval_every=2,
                           budget=200, threshold=0.5, lora=None, sampling=None,
                           eval_schedule=((20, 2), (100, 5), (None, 10)))  # fmt: skip
    with pytest.MonkeyPatch.context() as m:
        from residual_disposition import pipeline as P

        m.setattr(P, "_relearn_settings", lambda ws, lr, budget: base)
        s = S.s1_settings(SimpleNamespace(), {"learning_rate": 1e-5, "steps": 200})
    assert s.evaluation_steps() == [50, 100, 200]


def test_the_outcome_is_l_s_change_minus_h_s_s_with_its_interval():
    prompts = [f"p{i}" for i in range(100)]
    zero = dict.fromkeys(prompts, 0)
    half = {p: int(i < 50) for i, p in enumerate(prompts)}
    r = S.outcome({0: zero, 200: half}, {0: zero, 200: zero}, 7)
    assert r["difference"] == pytest.approx(0.5) and r["n"] == 100
    assert r["interval"][0] < 0.5 < r["interval"][1]
    assert S.outcome({0: zero, 200: zero}, {0: zero, 200: zero}, 7)["interval"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="same prompts"):
        S.outcome({0: zero, 200: {"p0": 1}}, {0: zero, 200: zero}, 7)


def test_responses_are_read_by_prompt_at_their_step_and_a_label_is_one_of_the_rubric_s():
    rows = [{"response_id": "r/s/0/50/p1/0", "text": "ELI5"},
            {"response_id": "r/s/0/0/p1/0", "text": "no"}]  # fmt: skip
    assert S.hits_by_prompt(rows, lambda t: "ELI5" in t, 50) == {"p1": 1}
    assert S.hits_by_prompt(rows, lambda t: "ELI5" in t, 0) == {"p1": 0}
    with pytest.raises(ValueError, match="twice"):
        S.hits_by_prompt([rows[0], rows[0]], bool, 50)
    labels = ["on_task", "degenerate"]
    assert S.parse_label(" On_task.\n", labels) == "on_task"
    with pytest.raises(ValueError):
        S.parse_label("maybe", labels)
