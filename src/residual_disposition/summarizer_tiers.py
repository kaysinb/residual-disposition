"""Each summarizer response's service tier, its provenance (decision log, 2026-10-16).

From 2026-10-16 the judge cache records the tier of the send that produced each completion
(``JudgeCache.put``). A completion cached before that has no tier; a *tier manifest* attributes
such entries key by key, each one verified, never by a time window alone:

- the key is one of the requests of a listed summarizer run, rebuilt from the run's
  ``summarizer_inputs.json`` (model, samples, max tokens, prompt versions and their sha256) and
  ``evidence.jsonl`` (each text checked against its recorded sha256);
- the ledgers hold the paid sends: exactly one ``spend`` of the key matches the cache entry (its
  ``cached_at``, second precision, falls between the spend and a minute after it, since the cache
  is written right after the send is settled), in the ledger of the run's ADL worker, inside the
  run's window. The run held that ledger's exclusive writer lock (``ApiLedger``) from before its
  start to after its end, so a spend in its window is its own; the run, its ledger and the cache
  share this host's clock;
- the run's tier is the one its config names (``judge.sync.service_tier``, null being the
  standard tier); a config without the key is refused.

A response read from the cache takes its entry's tier, or the manifest's for an older entry; a
response the run sends takes the run's tier (``response_tiers``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .judge import ChatRequest, Completion, JudgeCache, ModelSpec
from .pipeline import content_hash
from .summarizer import PROMPTS, SummarizerSettings, base_tool, summary_requests

MANIFEST_VERSION = 1
CACHE_LAG = timedelta(seconds=60)  # cached_at at most this long after its spend


class TierError(ValueError):
    """A response whose service tier cannot be established."""


@dataclasses.dataclass(frozen=True)
class SummarizerRun:
    run_id: str
    tier: str
    worker: str
    started_at: datetime
    finished_at: datetime
    keys: frozenset[str]


def _at(text: str) -> datetime:
    at = datetime.fromisoformat(text)
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def run_requests(run_dir: Path) -> list[ChatRequest]:
    """A summarizer run's requests, rebuilt from what it recorded; each evidence text and prompt
    is checked against the sha256 the run recorded for it."""
    inputs = json.loads((run_dir / "summarizer_inputs.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "evidence.jsonl").read_text().splitlines()]
    if len(rows) != len(inputs["evidence"]):
        raise TierError(f"{run_dir}: evidence.jsonl and summarizer_inputs.json differ in length")
    settings = SummarizerSettings(
        model=ModelSpec(**inputs["model"]),
        samples=int(inputs["samples"]),
        max_tokens=int(inputs["max_tokens"]),
    )
    version_of = {p["tool"]: version for version, p in inputs["prompts"].items()}
    for version, p in inputs["prompts"].items():
        if PROMPTS[version].sha256 != p["sha256"] or PROMPTS[version].tool != p["tool"]:
            raise TierError(f"{run_dir}: prompt {version} is not the one the run used")
    requests: list[ChatRequest] = []
    for row, recorded in zip(rows, inputs["evidence"], strict=True):
        if content_hash(row["text"]) != recorded["sha256"] or row["tool"] != recorded["tool"]:
            raise TierError(f"{run_dir}: an evidence text differs from its recorded sha256")
        prompt = PROMPTS[version_of[base_tool(row["tool"])]]
        requests += summary_requests(prompt, row["text"], settings)
    return requests


def summarizer_run(run_dir: Path) -> SummarizerRun:
    """A completed summarizer run's tier (from its config, explicitly), ADL worker, window and
    request keys."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("status") != "completed" or not meta.get("finished_at"):
        raise TierError(f"{run_dir} is not a completed run")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    sync = (config.get("judge") or {}).get("sync") or {}
    if "service_tier" not in sync:
        raise TierError(f"{run_dir}: its config does not name judge.sync.service_tier")
    worker = ((config.get("summarizer") or {}).get("adl_ledger") or {}).get("worker")
    if not worker:
        raise TierError(f"{run_dir}: its config names no ADL ledger worker")
    return SummarizerRun(
        run_id=meta["run_id"],
        tier=sync["service_tier"] or "standard",
        worker=str(worker),
        started_at=_at(meta["started_at"]),
        finished_at=_at(meta["finished_at"]),
        keys=frozenset(r.key for r in run_requests(run_dir)),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    runs: Sequence[Path], cache: JudgeCache, ledgers: Sequence[Path]
) -> dict[str, Any]:
    """The tier manifest of the listed runs' cache entries that record no tier (module
    docstring); TierError on any entry it cannot verify."""
    spends: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in ledgers:
        for line in Path(path).read_text().splitlines():
            event = json.loads(line) if line else None
            if event and event["event"] == "spend":
                spends[event["key"]].append({**event, "ledger": str(path)})
    infos = [summarizer_run(Path(r)) for r in runs]
    entries: dict[str, dict[str, Any]] = {}
    for key in sorted(set().union(*(i.keys for i in infos)) if infos else set()):
        entry = cache.entry(key)
        if entry is None or entry.get("service_tier"):
            continue  # never answered, or its tier is recorded
        cached_at = _at(entry["cached_at"])
        matching = [
            s
            for s in spends[key]
            if _at(s["at"]).replace(microsecond=0) <= cached_at <= _at(s["at"]) + CACHE_LAG
        ]
        if len(matching) != 1:
            raise TierError(f"cache entry {key}: {len(matching)} ledger spends match it, not one")
        spend = matching[0]
        at = _at(spend["at"])
        owners = [
            i
            for i in infos
            if key in i.keys and i.worker == spend["worker"] and i.started_at <= at <= i.finished_at
        ]
        if len(owners) != 1:
            raise TierError(f"cache entry {key}: {len(owners)} listed runs sent its spend")
        entries[key] = {
            "tier": owners[0].tier,
            "run": owners[0].run_id,
            "spend_id": spend["id"],
            "spend_at": spend["at"],
            "cached_at": entry["cached_at"],
        }
    return {
        "version": MANIFEST_VERSION,
        "cache_root": str(Path(cache.root).resolve()),
        "runs": {
            i.run_id: {
                "tier": i.tier,
                "worker": i.worker,
                "started_at": i.started_at.isoformat(),
                "finished_at": i.finished_at.isoformat(),
                "requests": len(i.keys),
            }
            for i in infos
        },
        "ledgers": {str(p): _file_sha256(Path(p)) for p in ledgers},
        "entries": entries,
    }


def load_manifests(paths: Iterable[Path], cache: JudgeCache) -> dict[str, str]:
    """Key → tier of the given manifests, which must describe this cache and agree."""
    out: dict[str, str] = {}
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        if manifest.get("version") != MANIFEST_VERSION:
            raise TierError(f"{path}: not a version-{MANIFEST_VERSION} tier manifest")
        if manifest["cache_root"] != str(Path(cache.root).resolve()):
            raise TierError(f"{path} describes the cache {manifest['cache_root']}, not this one")
        for key, entry in manifest["entries"].items():
            if out.setdefault(key, entry["tier"]) != entry["tier"]:
                raise TierError(f"manifests disagree on the tier of {key}")
    return out


def entry_tier(cache: JudgeCache, key: str, manifest: Mapping[str, str]) -> str:
    """The tier of a cached completion: its entry's, or the manifest's for an older entry."""
    entry = cache.entry(key)
    if entry is None:
        raise TierError(f"{key} is not cached")
    tier = entry.get("service_tier") or manifest.get(key)
    if not tier:
        raise TierError(f"cached completion {key} has no recorded tier and no manifest entry")
    return tier


def response_tiers(
    completions: Mapping[str, Completion],
    cached: Iterable[str],
    tier: str,
    cache: JudgeCache,
    manifest: Mapping[str, str],
) -> dict[str, str]:
    """The tier of each answered request after ``judge.fetch``: one ``fetch`` served from the
    cache (``cached``, its own report) has its entry's tier, one it sent has ``tier``."""
    from_cache = set(cached)
    return {
        key: entry_tier(cache, key, manifest) if key in from_cache else tier
        for key, completion in completions.items()
        if completion.error is None
    }


def tier_counts(
    batches: Sequence[tuple[Any, Sequence[ChatRequest]]],
    tiers: Mapping[str, str],
    accepted: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Per evidence text ("<state>/<tool>/<budget>"): the tiers of its answered requests
    (``responses``, malformed replies included) and of its accepted hypotheses, with the
    standard tier's share of each."""
    accepted = set(accepted)

    def part(keys: list[str]) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            counts[tiers[key]] += 1
        n = len(keys)
        return {
            "n": n,
            "tiers": dict(sorted(counts.items())),
            "standard_share": counts.get("standard", 0) / n if n else None,
        }

    out = {}
    for e, rs in batches:
        answered = [r.key for r in rs if r.key in tiers]
        out[f"{e.state_id}/{e.tool}/{e.budget or 'all'}"] = {
            "responses": part(answered),
            "hypotheses": part([k for k in answered if k in accepted]),
        }
    return out
