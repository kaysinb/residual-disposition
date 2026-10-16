"""The service tier of each summarizer response (residual_disposition.summarizer_tiers): cache
entries record it, older entries are attributed by a verified manifest bound to the run's own
ledger and to the exact response, never by a time window alone, and a consumed response carries
its tier out of the fetch that read it."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from residual_disposition.judge import Completion, JudgeCache, ModelSpec, fetch
from residual_disposition.pipeline import content_hash
from residual_disposition.summarizer import PROMPTS, SummarizerSettings, summary_requests
from residual_disposition.summarizer_tiers import (
    TierError,
    build_manifest,
    load_manifests,
    response_tier,
    response_tiers,
    run_requests,
    tier_counts,
)

MODEL = ModelSpec("gemini-3.8-flash", "low", 0.75, 3.75, 1000)
PROMPT = PROMPTS["adl-lens-v1"]
SETTINGS = SummarizerSettings(MODEL, samples=3, max_tokens=4096)
EVIDENCE = [("s1", "adl_lens@30/state/1", "evidence one"), ("s1", "adl_lens@20/state/1", "two")]
DONE = Completion("A hypothesis.", "stop", {"total_tokens": 3})


def at(minute: int, second: int = 0) -> str:
    return f"2026-10-16T08:{minute:02d}:{second:02d}+00:00"


def summarizer_run(root, name, *, start, end, ledger, tier="flex", worker="cont-laptop", sync=None):
    run = root / "runs" / "summarize_continuation" / name
    run.mkdir(parents=True)
    inputs = {
        "model": dataclasses.asdict(MODEL),
        "samples": SETTINGS.samples,
        "max_tokens": SETTINGS.max_tokens,
        "prompts": {PROMPT.version: {"tool": PROMPT.tool, "sha256": PROMPT.sha256}},
        "evidence": [
            {
                "state_id": s,
                "tool": t,
                "tool_run_id": "lens1",
                "budget": None,
                "sha256": content_hash(x),
            }
            for s, t, x in EVIDENCE
        ],
    }
    (run / "summarizer_inputs.json").write_text(json.dumps(inputs))
    rows = [
        {"state_id": s, "tool": t, "tool_run_id": "lens1", "budget": None, "text": x}
        for s, t, x in EVIDENCE
    ]
    (run / "evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {
        "judge": {"sync": {"service_tier": tier} if sync is None else sync},
        "summarizer": {"adl_ledger": {"worker": worker, "path": str(ledger)}},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    meta = {"status": "completed" if end else "running", "run_id": name, "started_at": start}
    if end:
        meta["finished_at"] = end
    (run / "meta.json").write_text(json.dumps(meta))
    return run


def keys():
    return [r.key for _, _, x in EVIDENCE for r in summary_requests(PROMPT, x, SETTINGS)]


def request(key):
    return next(
        r for _, _, x in EVIDENCE for r in summary_requests(PROMPT, x, SETTINGS) if r.key == key
    )


def cache_at(cache, key, when, tier=None):
    cache.put(request(key), DONE, service_tier=tier)
    path = cache.path(key)
    path.write_text(json.dumps({**json.loads(path.read_text()), "cached_at": when}))


def spend(ledger, key, when, worker="cont-laptop", micro="417000"):
    stamp = when.replace("+00:00", f".{micro}+00:00")
    event = {
        "event": "spend",
        "id": f"id-{key[:6]}-{when}-{micro}",
        "key": key,
        "usd": 0.002,
        "label": "summarizer",
        "worker": worker,
        "at": stamp,
    }
    with ledger.open("a") as f:
        f.write(json.dumps(event) + "\n")


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))


@pytest.fixture
def setup(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.touch()
    measure = summarizer_run(tmp_path, "measure", start=at(1), end=at(5), ledger=ledger)
    first = summarizer_run(tmp_path, "first", start=at(10), end=at(40), ledger=ledger)
    return tmp_path, [measure, first], JudgeCache(tmp_path / "cache"), ledger


def manifest_of(tmp, runs, cache):
    path = tmp / "manifest.json"
    path.write_text(json.dumps(build_manifest(runs, cache)))
    return load_manifests([path], cache)


def test_a_run_s_requests_are_rebuilt_from_what_it_recorded(setup):
    _, (measure, _), _, _ = setup
    assert [r.key for r in run_requests(measure)] == keys()
    rows = (measure / "evidence.jsonl").read_text().replace("evidence one", "evidence 1")
    (measure / "evidence.jsonl").write_text(rows)
    with pytest.raises(TierError, match="recorded sha256"):
        run_requests(measure)


def test_older_entries_are_attributed_by_key_to_the_run_that_sent_them(setup):
    tmp, runs, cache, ledger = setup
    k = keys()
    spend(ledger, k[0], at(2))  # the measurement sent it, the later run read it from the cache
    cache_at(cache, k[0], at(2))
    spend(ledger, k[1], at(12, 30))  # sent by the later run (its requests hold k[1] too)
    cache_at(cache, k[1], at(12, 31))
    spend(ledger, k[2], at(3))  # a send cut before its cache write, sent again later
    spend(ledger, k[2], at(20))
    cache_at(cache, k[2], at(20))
    cache_at(cache, k[3], at(30), tier="standard")  # recorded: not the manifest's
    manifest = build_manifest(runs, cache)
    entries = manifest["entries"]
    assert {key: e["run"] for key, e in entries.items()} == {
        k[0]: "measure",
        k[1]: "first",
        k[2]: "first",
    }
    assert {e["tier"] for e in entries.values()} == {"flex"}
    assert manifest["ledgers"] == {
        str(ledger.resolve()): manifest["ledgers"][str(ledger.resolve())]
    }
    tiers = manifest_of(tmp, runs, cache)
    assert response_tier(k[0], cache.get(k[0]), tiers) == "flex"
    assert response_tier(k[3], cache.get(k[3]), tiers) == "standard"
    with pytest.raises(TierError, match="no recorded tier and no manifest entry"):
        response_tier(k[0], cache.get(k[0]), {})
    with pytest.raises(TierError, match="not this one"):
        load_manifests([tmp / "manifest.json"], JudgeCache(tmp / "other"))


def test_the_manifest_names_the_exact_response(setup):
    tmp, runs, cache, ledger = setup
    k = keys()[0]
    spend(ledger, k, at(2))
    cache_at(cache, k, at(2))
    tiers = manifest_of(tmp, runs, cache)
    path = cache.path(k)
    path.write_text(json.dumps({**json.loads(path.read_text()), "text": "Another hypothesis."}))
    with pytest.raises(TierError, match="not the response its manifest entry names"):
        response_tier(k, cache.get(k), tiers)


def test_only_the_run_s_own_ledger_attributes_its_sends(setup):
    tmp, runs, cache, ledger = setup
    other = tmp / "other-ledger.jsonl"  # another file: a copy, or another worker's share
    k = keys()[0]
    spend(other, k, at(2))
    cache_at(cache, k, at(2))
    with pytest.raises(TierError, match="0 listed runs sent its spend"):
        build_manifest(runs, cache)
    missing = summarizer_run(tmp, "missing", start=at(1), end=at(5), ledger=tmp / "absent.jsonl")
    with pytest.raises(TierError, match="is not here"):
        build_manifest([missing], cache)


@pytest.mark.parametrize(
    "case, match",
    [
        ("outside", "lies in 0 writers' windows"),
        ("after", "lies in 0 writers' windows"),
        ("late", "0 listed runs sent its spend"),
        ("two", "2 spends in"),
        ("worker", "0 listed runs sent its spend"),
    ],
)
def test_an_entry_that_cannot_be_verified_is_refused(setup, case, match):
    _, runs, cache, ledger = setup
    k = keys()[0]
    if case == "outside":  # sent between the runs: by a writer that is not listed
        spend(ledger, k, at(7))
        cache_at(cache, k, at(7))
    elif case == "after":  # past the run's last second (meta times are truncated)
        spend(ledger, k, at(5, 1))
        cache_at(cache, k, at(5, 1))
    elif case == "late":  # the cache entry is not that send's
        spend(ledger, k, at(2))
        cache_at(cache, k, at(4))
    elif case == "two":
        spend(ledger, k, at(2), micro="100000")
        spend(ledger, k, at(2), micro="900000")
        cache_at(cache, k, at(2, 30))
    else:  # another worker's event in this file
        spend(ledger, k, at(2), worker="laptop")
        cache_at(cache, k, at(2))
    with pytest.raises(TierError, match=match):
        build_manifest(runs, cache)


def test_a_spend_in_the_run_s_last_second_is_its_own(setup):
    _, runs, cache, ledger = setup
    k = keys()[0]
    spend(ledger, k, at(5))  # 08:05:00.417, the measurement's finished_at is 08:05:00
    cache_at(cache, k, at(5))
    assert build_manifest(runs, cache)["entries"][k]["run"] == "measure"


def test_a_run_s_tier_is_the_one_its_config_names(tmp_path):
    cache, ledger = JudgeCache(tmp_path / "cache"), tmp_path / "ledger.jsonl"
    k = keys()[0]
    spend(ledger, k, at(2))
    cache_at(cache, k, at(2))
    import shutil

    # One writer of the ledger at a time: two runs over the same second would be ambiguous.
    standard = summarizer_run(tmp_path, "std", start=at(1), end=at(5), ledger=ledger, tier=None)
    assert build_manifest([standard], cache)["entries"][k]["tier"] == "standard"
    meta = json.loads((standard / "meta.json").read_text())
    (standard / "meta.json").write_text(json.dumps({**meta, "status": "running"}))
    with pytest.raises(TierError, match="not a completed run"):
        build_manifest([standard], cache)
    shutil.rmtree(standard)
    batch = summarizer_run(tmp_path, "batch", start=at(1), end=at(5), ledger=ledger)
    config = yaml.safe_load((batch / "config.yaml").read_text())
    config["summarizer"]["backend"] = "batch"
    (batch / "config.yaml").write_text(yaml.safe_dump(config))
    assert build_manifest([batch], cache)["entries"][k]["tier"] == "batch"
    shutil.rmtree(batch)
    unnamed = summarizer_run(tmp_path, "unnamed", start=at(1), end=at(5), ledger=ledger, sync={})
    with pytest.raises(TierError, match="does not name judge.sync.service_tier"):
        build_manifest([unnamed], cache)


def test_a_consumed_response_carries_its_tier_out_of_fetch(tmp_path):
    class Standard:
        tier = "standard"

        def __init__(self):
            self.sent = []

        def complete(self, requests):
            for r in requests:
                self.sent.append(r.key)
                yield r.key, DONE

    cache = JudgeCache(tmp_path / "cache")
    k = keys()
    cache_at(cache, k[0], at(1), tier="flex")  # cached by another writer before this fetch
    backend = Standard()
    completions, cached = fetch([request(k[0]), request(k[1])], backend, cache)
    assert backend.sent == [k[1]] and cached == {k[0]}
    assert response_tiers(completions, {}) == {k[0]: "flex", k[1]: "standard"}
    assert response_tiers({k[2]: Completion(None, error="503 after retries")}, {}) == {}


def test_responses_and_accepted_hypotheses_are_counted_apart():
    k = keys()
    batch = [
        (type("E", (), {"state_id": "s1", "tool": t, "budget": None})(), [request(x) for x in ks])
        for t, ks in (("adl_lens@30/state/1", k[:3]), ("adl_lens@20/state/1", k[3:]))
    ]
    tiers = {k[0]: "flex", k[1]: "standard", k[2]: "flex", k[3]: "flex"}
    counts = tier_counts(batch, tiers, accepted={k[0], k[1], k[3]})  # k[2] was empty
    assert counts["s1/adl_lens@30/state/1/all"] == {
        "responses": {"n": 3, "tiers": {"flex": 2, "standard": 1}, "standard_share": 1 / 3},
        "hypotheses": {"n": 2, "tiers": {"flex": 1, "standard": 1}, "standard_share": 0.5},
    }
    assert counts["s1/adl_lens@20/state/1/all"]["responses"]["n"] == 1


def test_the_script_writes_nothing_on_a_dry_run(setup, monkeypatch, capsys):
    import importlib.util
    import sys

    tmp, runs, cache, ledger = setup
    k = keys()[0]
    spend(ledger, k, at(2))
    cache_at(cache, k, at(2))
    monkeypatch.setenv("RUNS_DIR", str(tmp / "runs"))
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarizer_tiers.py"
    spec = importlib.util.spec_from_file_location("summarizer_tiers_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(["--runs", *map(str, runs), "--cache", str(cache.root), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["tiers"] == {"flex": 1}
    assert not (tmp / "runs" / "summarizer_tiers").exists()


def test_a_run_s_requests_are_salted_only_if_it_recorded_the_rule(tmp_path):
    from residual_disposition.summarizer import stability_salts

    global EVIDENCE
    saved = EVIDENCE
    EVIDENCE = [("s1", "adl_lens@30/state/1", "same"), ("s1", "adl_lens@30/state/2", "same")]
    try:
        ledger = tmp_path / "l.jsonl"
        ledger.touch()
        old = summarizer_run(tmp_path, "old", start=at(1), end=at(5), ledger=ledger)
        new = summarizer_run(tmp_path, "new", start=at(1), end=at(5), ledger=ledger)
        inputs = json.loads((new / "summarizer_inputs.json").read_text())
        (new / "summarizer_inputs.json").write_text(json.dumps({**inputs, "stability_salts": True}))
        old_keys = [r.key for r in run_requests(old)]
        new_keys = [r.key for r in run_requests(new)]
        n = SETTINGS.samples
        assert old_keys[:n] == old_keys[n:]  # before the rule the second draw shared the keys
        assert new_keys[:n] == old_keys[:n] and set(new_keys[n:]).isdisjoint(new_keys[:n])
        rows = [{"state_id": s, "tool": t, "text": x} for s, t, x in EVIDENCE]
        assert stability_salts(rows)[1] == "package=adl_lens@30/state/2"
    finally:
        EVIDENCE = saved


def test_a_spend_in_a_second_two_writers_share_is_refused(setup):
    """Review s2p: A ends at :00.2 (finished_at :00), B, not listed, starts at :00.3 (started_at
    :00) and spends at :00.5 — that second is in both windows, so the spend is nobody's."""
    tmp, runs, cache, ledger = setup
    k = keys()[0]
    summarizer_run(tmp, "late-writer", start=at(5), end=at(8), ledger=ledger)
    spend(ledger, k, at(5), micro="500000")
    cache_at(cache, k, at(5))
    with pytest.raises(TierError, match="lies in 2 writers' windows"):
        build_manifest(runs, cache)


def test_a_spend_that_names_its_run_belongs_to_it(setup):
    tmp, runs, cache, ledger = setup
    k = keys()
    summarizer_run(tmp, "live", start=at(6), end=None, ledger=ledger)  # unfinished, unlisted
    spend(ledger, k[0], at(7))  # inside the unfinished writer's open window: refused
    cache_at(cache, k[0], at(7))
    with pytest.raises(TierError, match="0 listed runs sent its spend"):
        build_manifest(runs, cache)  # the unfinished writer's spend, and it is not listed
    ledger.write_text("")
    named = {
        "event": "spend",
        "id": "x",
        "key": k[0],
        "usd": 0.002,
        "label": "summarizer",
        "worker": "cont-laptop",
        "run": "first",
        "at": "2026-10-16T08:07:00.100000+00:00",
    }
    ledger.write_text(json.dumps(named) + "\n")
    assert build_manifest(runs, cache)["entries"][k[0]]["run"] == "first"
    ledger.write_text(json.dumps({**named, "run": "live"}) + "\n")
    with pytest.raises(TierError, match="0 listed runs sent its spend"):
        build_manifest(runs, cache)  # its named run is not listed


def test_an_evidence_row_whose_identity_changed_is_refused(setup):
    _, (measure, _), _, _ = setup
    rows = (
        (measure / "evidence.jsonl").read_text().replace('"state_id": "s1"', '"state_id": "s9"', 1)
    )
    (measure / "evidence.jsonl").write_text(rows)
    with pytest.raises(TierError, match="state_id differs from the recorded"):
        run_requests(measure)
