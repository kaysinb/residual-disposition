"""The elicitation assay (§13.16, 2026-11-04): the writer's format and banned requests, the
conversation loop and its resumption, scoring, the blinded kits and the replay's choice."""

from __future__ import annotations

import json

import pytest

from residual_disposition import follow_elicit as FE
from residual_disposition.adl import StepStore


def test_a_probe_is_one_json_object_of_bounded_length():
    assert FE.parse_probe('{"probe": " How do rainbows form? "}') == "How do rainbows form?"
    fenced = '```json\n{"probe": "Why is the sky blue?"}\n```'
    assert FE.parse_probe(fenced) == "Why is the sky blue?"
    for bad in ("not json", '{"text": "x"}', '{"probe": ""}', json.dumps({"probe": "x" * 601})):
        assert FE.parse_probe(bad) is None


def test_requests_for_the_behavior_are_banned():
    for asked in ("Can you ELI5 this?", "Explain it like I'm five", "for a five-year-old please",
                  "in simple terms", "I'm a beginner", "explain simply"):  # fmt: skip
        assert FE.BANNED.search(asked)
    assert not FE.BANNED.search("How does a refrigerator keep food cold?")


def scripted(replies):
    """A writer answering each conversation's attempts from its own script."""
    seen = []

    def ask(calls):
        seen.extend(calls)
        return [replies[int(attempt.split("/c")[1].split("/")[0])].pop(0)
                for attempt, _ in calls]  # fmt: skip

    return ask, seen


def test_a_bad_format_is_asked_once_more_and_a_banned_probe_rewritten_twice():
    ok = '{"probe": "How do tides work?"}'
    banned = '{"probe": "Explain tides like I\'m five"}'
    ask, seen = scripted([["junk", ok], ["junk", "junk"], [banned, banned, ok],
                          [banned, banned, banned]])  # fmt: skip
    got = FE.write_probes(ask, "habit", [[], [], [], []], "k/t0")
    assert [g["probe"] for g in got] == ["How do tides work?", None, "How do tides work?", None]
    assert [g["void"] for g in got] == [None, "format", None, "banned"]
    assert [len(g["tries"]) for g in got] == [2, 2, 3, 3]
    assert len({a for a, _ in seen}) == len(seen)  # every attempt its own key


def test_the_state_and_the_base_answer_the_same_probe_and_each_step_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(FE, "CONVERSATIONS", 2)
    monkeypatch.setattr(FE, "TURNS", 2)
    calls, fail = [], {"base": True}

    def ask(batch):
        return [json.dumps({"probe": f"Q {attempt}"}) for attempt, _ in batch]

    def answer(probes, stream, side):
        if side == "base" and fail["base"]:
            raise RuntimeError("the base model's batch failed")
        calls.append((side, tuple(probes), stream))
        return [f"{side}: {p}" for p in probes]

    with pytest.raises(RuntimeError):
        FE.run_state(ask, answer, "habit", "line/L", StepStore(tmp_path))
    assert [c[0] for c in calls] == ["state"]
    fail["base"] = False
    turns = FE.run_state(ask, answer, "habit", "line/L", StepStore(tmp_path))
    assert [c[0] for c in calls] == ["state", "base", "state", "base"]  # the state's not redone
    assert calls[0][1:] == calls[1][1:]  # the same probes and stream on both sides
    assert [[t["probe"] for t in conv] for conv in turns] == [
        ["Q line/L/t0/c0/try0", "Q line/L/t1/c0/try0"],
        ["Q line/L/t0/c1/try0", "Q line/L/t1/c1/try0"]]
    again = FE.run_state(ask, answer, "habit", "line/L", StepStore(tmp_path))
    assert again == turns and len(calls) == 4
    rows = FE.score(turns, lambda text: text.startswith("state") and "t1" in text)
    assert [(r["state_hit"], r["base_hit"]) for r in rows] == [(False, False), (True, False)] * 2


def test_a_label_not_exactly_yes_or_no_or_a_missing_one_stops(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import follow_elicit as S

    def write(name, rows):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    write("sonnet", [{"item_id": "a", "shows": "yes"}, {"item_id": "b", "shows": "no"}])
    write("grok", [{"item_id": "a", "shows": "yes"}, {"item_id": "b", "shows": "yes"}])
    with pytest.raises(SystemExit, match="no settled label"):
        S.labels(tmp_path, "shows", {"a", "b"})
    write("researcher", [{"item_id": "b", "shows": "no"}])
    assert S.labels(tmp_path, "shows", {"a", "b"}) == {"a": "yes", "b": "no"}
    write("grok", [{"item_id": "a", "shows": "true"}, {"item_id": "b", "shows": "yes"}])
    with pytest.raises(SystemExit, match="not one yes or no"):
        S.labels(tmp_path, "shows", {"a", "b"})
    write("grok", [{"item_id": "a", "shows": "yes"}])
    with pytest.raises(SystemExit, match="not one per item"):
        S.labels(tmp_path, "shows", {"a", "b"})


def test_the_kits_hold_every_hit_and_a_fixed_minimum_of_non_hits_even_with_none():
    rows = [{"conversation": c, "turn": 0, "probe": f"p{c}", "void": None,
             "state_hit": c == 0, "base_hit": False} for c in range(8)]  # fmt: skip
    quiet = [{**r, "state_hit": False} for r in rows]
    texts = {(k, c, 0, side): f"{k} {c} {side}" for k in ("a", "b") for c in range(8)
             for side in ("state", "base")}  # fmt: skip
    ans, probes, mapping = FE.checks({"a": rows, "b": quiet}, texts, 3)
    by = [m for m in mapping.values() if m["kind"] == "answer"]
    assert sum(m["state"] == "b" and m["side"] == "state" for m in by) == FE.NON_HITS
    assert sum(m["state"] == "a" and m["side"] == "state" for m in by) == 1 + FE.NON_HITS
    assert [p["probe"] for p in probes] == ["p0"] and set(ans[0]) == {"item_id", "answer"}


def test_the_replay_takes_confirmed_probes_of_l_h_s_x_once_in_order_and_at_most_ten():
    def rows(probes):
        return [{"conversation": i, "turn": 0, "probe": p} for i, p in enumerate(probes)]

    scored = {"continued200/L": rows(["Why rain?", "Why  RAIN?"] + [f"q{i}" for i in range(12)]),
              "continued200/H_s": rows(["why rain?", "other"]), "both/A": rows(["from A"])}
    got = FE.replay_choice(scored, lambda key, r: True)
    first = got["continued200"][0]
    assert first["probe"] == "Why rain?" and first["found_on"] == ["L", "L", "H_s"]
    assert len(got["continued200"]) == FE.REPLAY_PER_LINE and got["stack800"] == []
    assert all(c["probe"] != "from A" for c in got["continued200"])
