"""The follow-up's qualification of D* and unfiltered PD (§13.5 (a); barrier B2): each reading's
slots counted by rule C and pending while one is unresolved, both streams pooled once, the floors
on exact fractions with eli5's two recipe controls qualified apart, a strict label blocked by the
safety filter counted as not naming b, and the requests salted per reading."""

from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from residual_disposition import follow_qualification as Q


def slot(strict="names_behavior", lenient=True, hyp=True):
    if not hyp:
        return {"hypothesis": {"due": "r"}, "labels": None}
    return {
        "hypothesis": {"value": "I will."},
        "labels": {"strict": {"value": (strict, True)}, "lenient": {"value": lenient}},
    }


def test_a_reading_counts_its_80_slots_or_stays_pending():
    slots = [slot()] * 60 + [slot(lenient=False)] * 10 + [slot(strict="avoids_behavior")] * 10
    assert Q.reading_count(slots) == {"counted": 60, "n": 80}
    blocked = [slot(strict="strict_safety_blocked")] * 80  # counted as not naming b
    assert Q.reading_count(blocked) == {"counted": 0, "n": 80}
    assert "pending" in Q.reading_count(slots[:-1] + [slot(hyp=False)])
    assert "pending" in Q.reading_count(slots[:79])


def row(reader, behavior, role, decoder, stream, k):
    return {"reading": {"reader": reader, "behavior": behavior, "role": role, "decoder": decoder,
                        "stream": stream}, "count": {"counted": k, "n": 80}}  # fmt: skip


def rates_of(table):
    rows = []
    for (reader, behavior, role, decoder), (k1, k2) in table.items():
        rows += [row(reader, behavior, role, decoder, 1, k1), row(reader, behavior, role,
                                                                  decoder, 2, k2)]  # fmt: skip
    return Q.pooled(rows)


FLOORS = {"active_min": "7/10", "sham_max": "1/10", "gap_min": "3/5", "recipe_ratio": "7/10"}
RULES = {
    "readers": {
        "qualification": {"floors": FLOORS, "fidelity_max": "1/10"},
        "pd_qualification": {"active_min": "7/10", "recipe_ratio": "7/10"},
    }
}


def table(a=(60, 60), h=(4, 4), recipe=(45, 45), recipe_cont=None, fid=(4, 4)):
    t = {}
    for d in Q.DSTAR_DECODERS:
        t[("dstar", "eli5", "A", d)] = a
        t[("dstar", "eli5", "H", d)] = h
        t[("dstar", "eli5", "recipe", d)] = recipe
        if recipe_cont:
            t[("dstar", "eli5", "recipe_cont", d)] = recipe_cont
    for branch in Q.FIDELITY:
        t[("dstar", "eli5", "A", branch)] = fid
    t[("pd_unfiltered", "eli5", "A", None)] = a
    t[("pd_unfiltered", "eli5", "recipe", None)] = recipe
    if recipe_cont:
        t[("pd_unfiltered", "eli5", "recipe_cont", None)] = recipe_cont
    return t


def test_the_floors_are_exact_and_eli5_s_recipes_qualify_apart():
    rates = rates_of(table(recipe_cont=(30, 30)))
    assert rates[("dstar", "eli5", "A", "state")] == Fraction(120, 160)
    f = Q.floors(rates, RULES)
    assert f["eli5"]["dstar"]["eligible"] is True and f["eli5"]["pd_unfiltered"] is True
    # the continued recipe: 60/160 < 7/10 · 120/160 = 84/160
    assert f["eli5_cont"]["dstar"]["eligible"] is False and f["eli5_cont"]["pd_unfiltered"] is False
    # exactly at the floors: R(A) = 112/160 = 7/10, R(H) = 16/160 = 1/10, the gap 96/160 = 3/5,
    # R(recipe) 79/160 ≥ 7/10 · 112/160 = 78.4/160: all pass; 78/160 fails
    edge = Q.floors(rates_of(table(a=(56, 56), h=(8, 8), recipe=(40, 39))), RULES)
    assert edge["eli5"]["dstar"]["eligible"] is True
    below = Q.floors(rates_of(table(a=(56, 56), h=(8, 8), recipe=(39, 39))), RULES)
    assert below["eli5"]["dstar"]["eligible"] is False and below["eli5"]["pd_unfiltered"] is False
    assert Q.floors(rates_of(table(fid=(9, 8))), RULES)["eli5"]["dstar"]["fidelity"] is False


def test_a_pending_reading_leaves_its_pool_and_verdict_pending():
    rows = [
        row("dstar", "eli5", "A", "state", 1, 60),
        {**row("dstar", "eli5", "A", "state", 2, 0), "count": {"pending": ["slot 3"]}},
    ]
    assert Q.pooled(rows)[("dstar", "eli5", "A", "state")] is None
    with pytest.raises(ValueError, match="streams"):
        Q.pooled(rows[:1])


class Cache:
    def __init__(self, entries):
        self.entries = entries

    def entry(self, key):
        return self.entries.get(key)


def test_a_slot_takes_the_first_answer_the_parser_accepts():
    from residual_disposition.summarizer import parse_hypothesis

    r, retry = SimpleNamespace(key="k", body={"b": 1}), SimpleNamespace(key="k2", body={"b": 1})
    assert "due" in Q.answer(Cache({}), [r, retry], parse_hypothesis)
    refused = {"k": {"body": {"b": 1}, "text": "  "}}
    assert Q.answer(Cache(refused), [r, retry], parse_hypothesis)["due"] is retry
    assert Q.answer(Cache({**refused, "k2": {"body": {"b": 1}, "text": "I will."}}), [r, retry],
                    parse_hypothesis)["value"] == "I will."  # fmt: skip
    assert Q.answer(Cache({**refused, "k2": {"body": {"b": 1}, "text": ""}}), [r, retry],
                    parse_hypothesis) == {"refused": True}  # fmt: skip
    with pytest.raises(ValueError, match="not to its request"):
        Q.answer(Cache({"k": {"body": {"b": 2}, "text": "I will."}}), [r], parse_hypothesis)


def test_the_requests_are_salted_per_reading():
    reading = {"reader": "pd_unfiltered", "decoder": None, "slots": 80,
               "salt": "follow/pd/qualification/eli5/A/stream1"}  # fmt: skip
    other = {**reading, "salt": "follow/pd/qualification/eli5/A/stream2"}
    a, b = Q.attempts(reading, "evidence"), Q.attempts(other, "evidence")
    assert len(a) == 80 and all(x[0].body == y[0].body for x, y in zip(a, b, strict=True))
    assert not {x[0].key for x in a} & {y[0].key for y in b}  # one body, its own slots
    assert all(x[0].key != x[1].key for x in a)  # each slot's one retry


def test_a_new_package_recorded_twice_is_refused(tmp_path, monkeypatch):
    from residual_disposition import follow_inventory as fi

    monkeypatch.setattr(fi, "built_packages", lambda run: {})
    base = tmp_path / "follow_build" / "x" / "out" / "pools"
    for name in ("a", "b"):
        (base / name).mkdir(parents=True)
        (base / name / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "pools": {
                        "s1": {"lineage": "eli5 stack 200", "role": "recipe", "run": f"pool-{name}"}
                    },
                }
            )
        )
    with pytest.raises(ValueError, match="recorded twice"):
        Q.new_packages("x", tmp_path)
    (base / "b" / "result.json").write_text(json.dumps({"status": "stopped", "pools": {}}))
    with pytest.raises(ValueError, match="not complete"):
        Q.new_packages("x", tmp_path)


def script():
    import importlib.util

    from residual_disposition.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "follow_qualification_script", REPO_ROOT / "scripts" / "follow_qualification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_tier_enters_a_request_s_body_key_or_salt():
    reading = {"reader": "dstar", "decoder": "state", "slots": 80,
               "salt": "follow/dstar/qualification/eli5/A/state/stream1"}  # fmt: skip
    tries = Q.attempts(reading, "evidence")
    for first, retry in tries:
        assert set(first.body) <= {"model", "messages", "max_tokens", "reasoning_effort"}
        assert "tier" not in json.dumps(first.body) and first.body == retry.body
    assert all("flex" not in t[0].key and "standard" not in reading["salt"] for t in tries)


def test_the_interpreter_reads_on_flex_with_fallback_the_judge_on_standard(monkeypatch):
    s = script()
    monkeypatch.setenv("JUDGE_API_BASE", "http://localhost")
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    ledger = SimpleNamespace(owner_of={})
    interp = s.backend(ledger, {}, interpreter=True, journal=lambda e: None)
    judge = s.backend(ledger, {}, interpreter=False)
    assert interp.inner.service_tier == "flex" and interp.inner.fallback is not None
    assert judge.inner.service_tier is None and judge.inner.fallback is None
    factors = s.tier_factors()
    assert interp.price_factor == factors["flex"] < factors["standard"] == judge.price_factor
    assert interp.tier_factors == factors  # each attempt priced at the tier it is sent on
