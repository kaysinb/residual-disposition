"""The service tier of each summarizer response (residual_disposition.summarizer_tiers): cache
entries record it, older entries are attributed by a verified manifest, never by a time window
alone, and a run's late cache hits keep their own tier."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from residual_disposition.judge import Completion, JudgeCache, ModelSpec
from residual_disposition.pipeline import content_hash
from residual_disposition.summarizer import PROMPTS, SummarizerSettings, summary_requests
from residual_disposition.summarizer_tiers import (
    TierError,
    build_manifest,
    entry_tier,
    load_manifests,
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


def summarizer_run(root, name, *, start, end, tier="flex", worker="cont-laptop", config=None):
    run = root / name
    run.mkdir()
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
    if config is None:
        config = {
            "judge": {"sync": {"service_tier": tier}},
            "summarizer": {"adl_ledger": {"worker": worker}},
        }
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    meta = {"status": "completed", "run_id": name, "started_at": start, "finished_at": end}
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
    event = {"event": "spend", "id": f"id-{key[:6]}-{when}", "key": key, "usd": 0.002,
             "label": "summarizer", "worker": worker, "at": stamp}  # fmt: skip
    with ledger.open("a") as f:
        f.write(json.dumps(event) + "\n")


@pytest.fixture
def setup(tmp_path):
    measure = summarizer_run(tmp_path, "measure", start=at(1), end=at(5))
    first = summarizer_run(tmp_path, "first", start=at(10), end=at(40))
    return tmp_path, [measure, first], JudgeCache(tmp_path / "cache"), tmp_path / "ledger.jsonl"


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
    manifest = build_manifest(runs, cache, [ledger])
    entries = manifest["entries"]
    assert {key: e["run"] for key, e in entries.items()} == {
        k[0]: "measure",
        k[1]: "first",
        k[2]: "first",
    }
    assert {e["tier"] for e in entries.values()} == {"flex"}
    assert manifest["runs"]["first"]["requests"] == len(k)
    path = tmp / "manifest.json"
    path.write_text(json.dumps(manifest))
    tiers = load_manifests([path], cache)
    assert entry_tier(cache, k[0], tiers) == "flex"
    assert entry_tier(cache, k[3], tiers) == "standard"
    with pytest.raises(TierError, match="no recorded tier and no manifest entry"):
        entry_tier(cache, k[0], {})
    with pytest.raises(TierError, match="not this one"):
        load_manifests([path], JudgeCache(tmp / "other"))


@pytest.mark.parametrize(
    "case, match",
    [
        ("outside", "0 listed runs sent its spend"),
        ("late", "0 ledger spends match it"),
        ("two", "2 ledger spends match it"),
        ("worker", "0 listed runs sent its spend"),
    ],
)
def test_an_entry_that_cannot_be_verified_is_refused(setup, case, match):
    _, runs, cache, ledger = setup
    k = keys()[0]
    if case == "outside":  # sent between the runs: by a writer that is not listed
        spend(ledger, k, at(7))
        cache_at(cache, k, at(7))
    elif case == "late":  # the cache entry is not that send's
        spend(ledger, k, at(2))
        cache_at(cache, k, at(4))
    elif case == "two":
        spend(ledger, k, at(2), micro="100000")
        spend(ledger, k, at(2), micro="900000")
        cache_at(cache, k, at(2, 30))
    else:  # another worker's ledger
        spend(ledger, k, at(2), worker="laptop")
        cache_at(cache, k, at(2))
    with pytest.raises(TierError, match=match):
        build_manifest(runs, cache, [ledger])


def test_a_run_s_tier_is_the_one_its_config_names(tmp_path):
    cache, ledger = JudgeCache(tmp_path / "cache"), tmp_path / "ledger.jsonl"
    k = keys()[0]
    spend(ledger, k, at(2))
    cache_at(cache, k, at(2))
    standard = summarizer_run(tmp_path, "std", start=at(1), end=at(5), tier=None)
    assert build_manifest([standard], cache, [ledger])["entries"][k]["tier"] == "standard"
    unnamed = summarizer_run(
        tmp_path,
        "unnamed",
        start=at(1),
        end=at(5),
        config={"judge": {"sync": {}}, "summarizer": {"adl_ledger": {"worker": "cont-laptop"}}},
    )
    with pytest.raises(TierError, match="does not name judge.sync.service_tier"):
        build_manifest([unnamed], cache, [ledger])
    meta = json.loads((standard / "meta.json").read_text())
    (standard / "meta.json").write_text(json.dumps({**meta, "status": "running"}))
    with pytest.raises(TierError, match="not a completed run"):
        build_manifest([standard], cache, [ledger])


def test_a_late_cache_hit_keeps_its_own_tier_and_only_sends_take_the_run_s(tmp_path):
    cache = JudgeCache(tmp_path / "cache")
    k = keys()
    cache_at(cache, k[0], at(1), tier="flex")  # cached by another writer after the pre-check
    completions = {k[0]: DONE, k[1]: DONE, k[2]: Completion(None, error="503 after retries")}
    tiers = response_tiers(completions, cached={k[0]}, tier="standard", cache=cache, manifest={})
    assert tiers == {k[0]: "flex", k[1]: "standard"}  # the failed request has no response


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
    argv = ["--runs", *map(str, runs), "--ledgers", str(ledger), "--cache", str(cache.root)]
    module.main([*argv, "--dry-run"])
    assert json.loads(capsys.readouterr().out)["tiers"] == {"flex": 1}
    assert not (tmp / "runs").exists()
