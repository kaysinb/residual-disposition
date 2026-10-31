"""The follow-up's qualification of D* and unfiltered PD (§13.5 (a); barrier B2): each reading's
slots counted by rule C and pending while one is unresolved, both streams pooled once, the floors
on exact fractions with eli5's two recipe controls qualified apart, a strict label blocked by the
safety filter counted as not naming b, and the requests salted per reading; a new package bound
to its frozen state and checked records before rendering, every reply admitted at its tier, the
slots' histories kept and the evidence pinned."""

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
    count = Q.reading_count(slots)
    assert (count["counted"], count["n"], count["blocked"]) == (60, 80, 0)
    blocked = {"hypothesis": {"value": "I will."},
               "labels": {"strict": {"value": ("strict_safety_blocked", False), "blocked": True},
                          "lenient": {"value": True}}}  # fmt: skip
    count = Q.reading_count([blocked] * 2 + [slot()] * 78)  # counted as not naming b
    assert (count["counted"], count["blocked"], count["blocked_lenient_yes"]) == (78, 2, 2)
    assert count["interval"] == ["39/40", "1"]  # [78/80, 80/80], descriptive
    assert "pending" in Q.reading_count(slots[:-1] + [slot(hyp=False)])
    assert "pending" in Q.reading_count(slots[:79])
    labels = {"strict": {"unadmitted": "x"}, "lenient": {"value": True}}
    unadmitted = {"hypothesis": {"value": "I will."}, "labels": labels}
    assert "strict unadmitted" in Q.reading_count(slots[:-1] + [unadmitted])["pending"][0]


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


def test_the_base_decoder_governs_only_its_own_qualification():
    t = table()
    t[("dstar", "eli5", "A", "base")] = (10, 10)  # far below the floors, on the base decoder only
    f = Q.floors(rates_of(t), RULES)["eli5"]["dstar"]
    assert f["state"] is True and f["base"] is False
    assert f["eligible"] is True and f["base_qualification"] is False
    t = table()
    t[("dstar", "eli5", "A", "state")] = (10, 10)
    f = Q.floors(rates_of(t), RULES)["eli5"]["dstar"]
    assert f["eligible"] is False and f["base_qualification"] is True
    assert Q.floors(rates_of(table(fid=(9, 8))), RULES)["eli5"]["dstar"]["eligible"] is False


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


def test_a_slot_takes_the_first_admitted_answer_the_parser_accepts():
    from residual_disposition.summarizer import parse_hypothesis

    tier = Q.interpreter_tier
    r, retry = SimpleNamespace(key="k", body={"b": 1}), SimpleNamespace(key="k2", body={"b": 1})
    assert "due" in Q.answer(Cache({}), [r, retry], parse_hypothesis, tier)
    refused = {"k": {"body": {"b": 1}, "text": "  ", "service_tier": "flex"}}
    assert Q.answer(Cache(refused), [r, retry], parse_hypothesis, tier)["due"] is retry
    both = {**refused, "k2": {"body": {"b": 1}, "text": "I will.", "service_tier": "standard"}}
    found = Q.answer(Cache(both), [r, retry], parse_hypothesis, tier)
    assert found["value"] == "I will." and [h["parsed"] for h in found["history"]] == [False, True]
    assert [h["admitted"] for h in found["history"]] == ["flex", "standard"]  # rejected kept
    none = {**refused, "k2": {"body": {"b": 1}, "text": "", "service_tier": "flex"}}
    assert Q.answer(Cache(none), [r, retry], parse_hypothesis, tier)["refused"] is True
    with pytest.raises(ValueError, match="not to its request"):
        Q.answer(Cache({"k": {"body": {"b": 2}, "text": "I will."}}), [r], parse_hypothesis, tier)
    # a tier-less or batch reply is not admitted: pending, never asked again
    for t in (None, "batch"):
        bare = {"k": {"body": {"b": 1}, "text": "I will.", "service_tier": t}}
        assert "unadmitted" in Q.answer(Cache(bare), [r, retry], parse_hypothesis, tier)
    # the first attempt lost while its retry is answered: the history is lost, never redrawn
    lost = {"k2": both["k2"]}
    with pytest.raises(ValueError, match="history is lost"):
        Q.answer(Cache(lost), [r, retry], parse_hypothesis, tier)


def blocked_entry(body):
    return {"body": body, "text": "", "finish_reason": "content_filter",
            "usage": {"completion_tokens": 0}, "service_tier": "standard"}  # fmt: skip


def test_a_label_is_admitted_as_the_readers_admit_it():
    from residual_disposition import continuation_readers as readers
    from residual_disposition.judge import JudgeItem, cache_entry_sha256

    judge = readers.judge_settings("v4")
    item = JudgeItem("follow/x#slot=0", "eli5", "Explains like to a five-year-old.", "I will.")
    tries = Q.label_attempts(item, judge)
    strict, lenient = tries["strict"], tries["lenient"]

    def lenient_entry(tier):
        return {"body": lenient[0].body, "text": "yes", "service_tier": tier}

    entries = {strict[0].key: blocked_entry(strict[0].body),
               strict[1].key: blocked_entry(strict[1].body),
               lenient[0].key: lenient_entry("standard")}  # fmt: skip
    got = Q.labels(item, judge, Cache(entries), {})
    assert got["strict"]["value"] == (readers.STRICT_SAFETY_BLOCKED, False)
    assert got["strict"]["blocked"] and got["lenient"]["value"] is True
    for tier in ("flex", "batch", None):  # not the standard tier, not pinned
        got = Q.labels(item, judge, Cache({**entries, lenient[0].key: lenient_entry(tier)}), {})
        assert "unadmitted" in got["lenient"] and "value" not in got["strict"]
    tierless = {**entries, strict[1].key: {**blocked_entry(strict[1].body), "service_tier": None}}
    assert "value" not in Q.labels(item, judge, Cache(tierless), {})["strict"]
    # a historical reply a committed judge history pins is read at its established tier
    old = lenient_entry(None)
    pinned = {lenient[0].key: {"entry_sha256": cache_entry_sha256(old), "tier": "flex"}}
    got = Q.labels(item, judge, Cache({**entries, lenient[0].key: old}), pinned)
    assert got["lenient"]["value"] is True and got["lenient"]["history"][0]["admitted"] == "flex"


def test_recorded_answers_must_stay_in_the_caches_unchanged():
    from residual_disposition.judge import cache_entry_sha256

    entry = {"body": {"b": 1}, "text": "I will.", "service_tier": "flex"}
    previous = {"s#slot=0": {"interpreter": [{"key": "k", "sha256": cache_entry_sha256(entry)}]}}
    caches = {"interpreter": Cache({"k": entry}), "judge": Cache({})}
    assert Q.history_problems(previous, caches) == []
    caches["interpreter"] = Cache({"k": {**entry, "text": "Other."}})
    assert Q.history_problems(previous, caches) == ["s#slot=0 (interpreter): k is changed"]
    caches["interpreter"] = Cache({})
    assert Q.history_problems(previous, caches) == ["s#slot=0 (interpreter): k is missing"]


def test_the_pins_hold_each_reading_s_evidence_and_keys(tmp_path):
    reading = {"reader": "pd_unfiltered", "decoder": None, "slots": 80,
               "salt": "follow/pd/qualification/eli5/A/stream1"}  # fmt: skip
    cell = {"reading": reading, "directory": str(tmp_path / "pool"), "evidence_sha256": "e1",
            "attempts": Q.attempts(reading, "evidence")}  # fmt: skip
    pinned = Q.pins([cell], tmp_path)
    assert pinned[reading["salt"]]["directory"] == "pool"
    assert Q.pin_problems(pinned, pinned) == []
    changed = Q.pins([{**cell, "evidence_sha256": "e2"}], tmp_path)
    assert Q.pin_problems(pinned, changed) == [f"{reading['salt']}: evidence_sha256 changed"]
    assert Q.pin_problems({}, pinned) == [f"{reading['salt']}: not pinned"]


def test_the_requests_are_salted_per_reading():
    reading = {"reader": "pd_unfiltered", "decoder": None, "slots": 80,
               "salt": "follow/pd/qualification/eli5/A/stream1"}  # fmt: skip
    other = {**reading, "salt": "follow/pd/qualification/eli5/A/stream2"}
    a, b = Q.attempts(reading, "evidence"), Q.attempts(other, "evidence")
    assert len(a) == 80 and all(x[0].body == y[0].body for x, y in zip(a, b, strict=True))
    assert not {x[0].key for x in a} & {y[0].key for y in b}  # one body, its own slots
    assert all(x[0].key != x[1].key for x in a)  # each slot's one retry


def session(tmp_path, monkeypatch, pools, build_json=None, tamper=False):
    """A build session in ``tmp_path`` with its records pinned: eli5's recipe build and pools."""
    import hashlib

    from residual_disposition import follow_build, follow_pd, pipeline

    run = "x"
    base = tmp_path / "follow_build" / run
    built = base / "out" / "build" / run / "builds" / "sR"
    built.mkdir(parents=True)
    (built / "v.safetensors").write_bytes(b"vector")
    record = build_json or {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "recipe",
                            "state": "sR"}  # fmt: skip
    (built / "build.json").write_text(json.dumps(record))
    sha = hashlib.sha256(b"vector").hexdigest()
    files = {"build": {"status": "complete", "files": {"builds/sR/v.safetensors": sha},
                       "builds": {"sR": {"complete": True, "check": {"passed": True}}}},
             "pools/eli5": {"status": "complete", "pools": pools},
             "pools/pun_intended": {"status": "complete", "pools": {}}}  # fmt: skip
    pinned = {}
    for k, rel in (("build", "out/build/{run}/result.json"),
                   ("pools/eli5", "out/pools/{run}-eli5/result.json"),
                   ("pools/pun_intended", "out/pools/{run}-pun_intended/result.json")):  # fmt: skip
        path = base / rel.format(run=run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(files[k]))
        pinned[k] = (rel, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(Q, "BUILD_RECORDS", pinned)
    if tamper:
        (built / "v.safetensors").write_bytes(b"other")
    runs = {}
    for name, state in (("pool-recipe", "sR"), ("pool-L", "sL")):
        r = tmp_path / "runs" / name
        r.mkdir(parents=True)
        (r / "meta.json").write_text(json.dumps({"status": "completed"}))
        (r / "config.yaml").write_text(f"step: tool_pd\nparams: {{state: {state}}}\n")
        (r / "summary.json").write_text("{}")
        runs[name] = r
    monkeypatch.setattr(follow_build, "pool_problems", lambda *a, **k: [])
    monkeypatch.setattr(follow_pd, "pool_rows", lambda run: [])
    monkeypatch.setattr(pipeline, "pd_gpu", lambda run: None)
    plan = {"sessions": {"b1": {"builds": [
                {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "recipe", "state": "sR"},
                {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "L", "state": "sL"}]}},
            "pools": [
                {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "recipe", "state": "sR"},
                {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "L",
                 "state": "sL"}]}  # fmt: skip
    return lambda names: Q.new_packages(names, run, tmp_path, plan, runs.get)


RECIPE = {"behavior": "eli5", "lineage": "eli5 stack 200", "role": "recipe"}
NAMES = {"lens": {"eli5 stack 200 recipe"}, "pd": {"eli5 stack 200 recipe"}}


def test_a_new_package_is_bound_to_its_frozen_state_and_checked_records(tmp_path, monkeypatch):
    bind = session(tmp_path / "ok", monkeypatch, {"sR": {**RECIPE, "run": "pool-recipe"}})
    new = bind(NAMES)
    assert set(new["lens"]) == set(new["pd"]) == {"eli5 stack 200 recipe"}  # L is never read
    assert new["pd"]["eli5 stack 200 recipe"] == {"pool": "pool-recipe", "state": "sR"}
    assert new["lens"]["eli5 stack 200 recipe"]["state"] == "sR"
    reading = {"reader": "pd_unfiltered", "salt": "s", "package": {"new": "eli5 stack 200 L"}}
    with pytest.raises(ValueError, match="not bound"):
        Q.package_dir(reading, lambda n: n, new)
    # a "recipe" record pointing at L's pool: the pool is not of the recipe's state
    bind = session(tmp_path / "swap", monkeypatch, {"sR": {**RECIPE, "run": "pool-L"}})
    with pytest.raises(ValueError, match="not a completed PD pool of sR"):
        bind(NAMES)
    # a "recipe" record of L's state: not the frozen recipe's
    bind = session(tmp_path / "relabel", monkeypatch, {"sL": {**RECIPE, "run": "pool-L"}})
    with pytest.raises(ValueError, match="not the frozen plan's"):
        bind({"pd": {"eli5 stack 200 recipe"}})
    bind = session(tmp_path / "role", monkeypatch, {}, build_json={**RECIPE, "role": "L",
                                                                   "state": "sR"})  # fmt: skip
    with pytest.raises(ValueError, match="build record is not the frozen plan's"):
        bind({"lens": {"eli5 stack 200 recipe"}})
    bind = session(tmp_path / "tamper", monkeypatch, {}, tamper=True)
    with pytest.raises(ValueError, match="sha256 the session recorded"):
        bind({"lens": {"eli5 stack 200 recipe"}})
    with pytest.raises(ValueError, match="not a lens build of the frozen plan"):
        bind({"lens": {"eli5 stack 200 H"}})
    (tmp_path / "tamper" / "follow_build" / "x" / "out" / "pools" / "x-eli5" / "result.json"
     ).write_text("{}")  # fmt: skip
    with pytest.raises(ValueError, match="not the session's record at its sha256"):
        bind({"pd": set()})


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


def lens_run(root, state, step="adl_lens_package"):
    import yaml
    from omegaconf import OmegaConf

    from residual_disposition.config import config_hash

    run = root / f"lens-{state}"
    run.mkdir(parents=True)
    config = {"step": step, "params": {"state": state}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    meta = {"status": "completed", "config_hash": config_hash(OmegaConf.create(config))}
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "adl_lens_package.json").write_text(json.dumps({"state": state}))
    return run, config, meta["config_hash"]


def test_an_existing_lens_package_is_its_role_s_approved_cell(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from residual_disposition import follow_inventory as fi
    from residual_disposition import pipeline as P

    good, config, digest = lens_run(tmp_path, "sA")
    wrong, _, _ = lens_run(tmp_path, "sL")  # an A directory whose saved config names L
    reading = {"reader": "dstar", "behavior": "eli5", "role": "A", "salt": "s",
               "package": {"run": "x"}}  # fmt: skip
    expected = {("eli5", "A"): "sA"}
    cell = SimpleNamespace(done=True, digest=digest, cfg=OmegaConf.create(config))
    monkeypatch.setattr(P, "_stage3_lens_cell", lambda ws, role, sid: (cell, None))

    def at(run):
        return Q.package_dir(reading, lambda n: run, {}, expected, lambda b: None)

    assert at(good) == good
    with pytest.raises(ValueError, match="not a lens package of sA"):
        at(wrong)
    with pytest.raises(ValueError, match="no expected state"):
        Q.package_dir({**reading, "role": "H"}, lambda n: good, {}, expected, lambda b: None)
    other = SimpleNamespace(done=True, digest="0" * 12, cfg=OmegaConf.create(config))
    monkeypatch.setattr(P, "_stage3_lens_cell", lambda ws, role, sid: (other, None))
    with pytest.raises(ValueError, match="not the approved cell"):
        at(good)
    # no approved cell made here: Stage 1's readings record must name it
    monkeypatch.setattr(P, "_stage3_lens_cell", lambda ws, role, sid: (None, None))
    rows = tmp_path / "readings.csv"
    rows.write_text(f"behavior,role,dose,tool_run_id,state_id\neli5,A,,{good.name},sA\n")
    monkeypatch.setattr(fi, "STAGE1_READINGS", str(rows))
    assert at(good) == good
    rows.write_text(f"behavior,role,dose,tool_run_id,state_id\neli5,H,,{good.name},sA\n")
    with pytest.raises(ValueError, match="not the approved cell"):
        at(good)


def test_every_stored_answer_is_journaled_at_once_and_never_taken_for_unanswered(tmp_path):
    from residual_disposition.judge import ChatRequest, Completion

    s = script()
    root = tmp_path / "runs"
    run = root / "run-1"
    run.mkdir(parents=True)
    cache = s.JournaledCache(tmp_path / "cache", run / "answers.jsonl", "interpreter")
    request = ChatRequest("k1", "summary", "v", {"b": 1})
    cache.put(request, Completion("I will.", "stop", {"completion_tokens": 3}),
              service_tier="flex")  # fmt: skip
    line = json.loads((run / "answers.jsonl").read_text())
    assert (line["key"], line["tier"]) == ("k1", "flex")
    caches = {"summary": cache, "judge": Cache({})}
    s.reconcile(root, caches)  # the answer is there
    cache.path("k1").unlink()  # lost after the run, before its histories were written
    with pytest.raises(SystemExit, match="lost or changed"):
        s.reconcile(root, caches)
    other = root / "run-2"
    other.mkdir()
    (other / "answers.jsonl").write_text(json.dumps({**line, "sha256": "0" * 64}) + "\n")
    with pytest.raises(SystemExit, match="two entries"):
        s.previous_histories(root)
