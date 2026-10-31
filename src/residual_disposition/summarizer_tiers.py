"""Each summarizer response's service tier, its provenance (decision log, 2026-10-16).

From 2026-10-16 the judge cache records the tier of the send that produced each completion
(``JudgeCache.put``). A completion cached before that has no tier; a *tier manifest* attributes
such entries key by key, each one verified, never by a time window alone:

- the key is one of the requests of a listed summarizer run, rebuilt from the run's
  ``summarizer_inputs.json`` (model, samples, max tokens, prompt versions and their sha256) and
  ``evidence.jsonl`` (each text checked against its recorded sha256);
- the run's own ledger, the file its config names (``summarizer.adl_ledger.path``, or the
  worker's default under this machine's data root), never a copy, holds the paid sends: exactly
  one ``spend`` of the key matches the cache entry (its ``cached_at``, second precision, falls
  between the spend and a minute after it, since the cache is written right after the send is
  settled), and it is the run's own: a spend names its run (``run``, recorded from 2026-10-16),
  or else its second lies in the window of that run alone among every run that names the ledger,
  finished or not (``ledger_writers``). The run held that ledger's exclusive writer lock
  (``ApiLedger``, a host-local ``flock``) from before its start to after its end, so a spend in
  its window is its own and stamped by this host's clock, as are the run's window and the cache.
  This rests on an assumption the lock cannot prove across hosts: that the run, its ledger and
  the cache were on this host under this data root;
- the run's tier is the one its config names (``judge.sync.service_tier``, null being the
  standard tier); a config without the key is refused;
- the manifest keeps each attributed entry's sha256 (``judge.cache_entry_sha256``): it applies to
  that exact response only.

A response takes the tier it carries out of ``judge.fetch`` (``Completion.service_tier``: the
run's own for a send, its cache entry's for a hit, read with it), or, for an older entry without
one, the manifest's tier when the manifest's sha256 is that response's (``response_tier``).
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

from .judge import ChatRequest, Completion, JudgeCache, ModelSpec, cache_entry_sha256
from .paths import data_root
from .pipeline import content_hash
from .summarizer import (
    PROMPTS,
    SummarizerSettings,
    base_tool,
    stability_salts,
    summary_requests,
)

MANIFEST_VERSION = 1
CACHE_LAG = timedelta(seconds=60)  # cached_at at most this long after its spend


class TierError(ValueError):
    """A response whose service tier cannot be established."""


@dataclasses.dataclass(frozen=True)
class SummarizerRun:
    run_id: str
    tier: str
    worker: str
    ledger: Path
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
    # Each row is the evidence the run recorded (its text's sha256 and its identity), checked
    # before any salt is derived from the rows.
    for row, recorded in zip(rows, inputs["evidence"], strict=True):
        if content_hash(row["text"]) != recorded["sha256"]:
            raise TierError(f"{run_dir}: an evidence text differs from its recorded sha256")
        for field in ("state_id", "tool", "tool_run_id", "budget"):
            if row.get(field) != recorded.get(field):
                raise TierError(f"{run_dir}: an evidence row's {field} differs from the recorded")
    # Runs from 2026-10-16 that record it salted their stability packages' keys; older ones not.
    try:
        salts = stability_salts(rows) if inputs.get("stability_salts") else [None] * len(rows)
    except ValueError as error:
        raise TierError(f"{run_dir}: {error}") from error
    requests: list[ChatRequest] = []
    for row, salt in zip(rows, salts, strict=True):
        prompt = PROMPTS[version_of[base_tool(row["tool"])]]
        requests += summary_requests(prompt, row["text"], settings, salt=salt)
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
    # Batch runs exist only from the change that added them (2026-10-16), whose configs name
    # summarizer.backend; a run without the key predates it and sent synchronously.
    backend = (config.get("summarizer") or {}).get("backend", "sync")
    tier = "batch" if backend == "batch" else sync["service_tier"] or "standard"
    spec = (config.get("summarizer") or {}).get("adl_ledger") or {}
    if not spec.get("worker"):
        raise TierError(f"{run_dir}: its config names no ADL ledger worker")
    worker = str(spec["worker"])
    ledger = Path(spec.get("path") or data_root() / "ledgers" / f"adl-api-{worker}.jsonl")
    if not ledger.is_file():
        raise TierError(f"{run_dir}: its ledger {ledger} is not here")
    return SummarizerRun(
        run_id=meta["run_id"],
        tier=tier,
        worker=worker,
        ledger=ledger.resolve(),
        started_at=_at(meta["started_at"]),
        finished_at=_at(meta["finished_at"]),
        keys=frozenset(r.key for r in run_requests(run_dir)),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Writer:
    """A summarizer run that may have written to a ledger: its window, the end unknown (None)
    for a run that never finished."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None


def _ledger_path(spec: Mapping[str, Any] | None) -> Path | None:
    if not spec or not spec.get("worker"):
        return None
    path = spec.get("path") or data_root() / "ledgers" / f"adl-api-{spec['worker']}.jsonl"
    return Path(path).resolve()


def ledger_writers(ledger: Path, runs_root: Path) -> list[Writer]:
    """Every summarizer run under ``runs_root`` whose config names ``ledger`` (as its ADL or its
    API ledger), whatever its status: the runs that may have written to it."""
    out = []
    for config_path in sorted(Path(runs_root).glob("*/*/config.yaml")):
        text = config_path.read_text()
        if "summarizer:" not in text or ("adl_ledger" not in text and "api_ledger" not in text):
            continue
        s = (yaml.safe_load(text) or {}).get("summarizer") or {}
        named = {_ledger_path(s.get("adl_ledger")), _ledger_path(s.get("api_ledger"))}
        if Path(ledger).resolve() not in named:
            continue
        meta_path = config_path.parent / "meta.json"
        if not meta_path.is_file():  # a writer whose window cannot be bounded
            raise TierError(f"{config_path.parent} names {ledger} but has no meta.json")
        meta = json.loads(meta_path.read_text())
        end = meta.get("finished_at")
        out.append(Writer(meta["run_id"], _at(meta["started_at"]), _at(end) if end else None))
    return out


def _second(at: datetime) -> datetime:
    return at.replace(microsecond=0)


def build_manifest(
    runs: Sequence[Path], cache: JudgeCache, runs_root: Path | None = None
) -> dict[str, Any]:
    """The tier manifest of the listed runs' cache entries that record no tier (module
    docstring); TierError on any entry it cannot verify.

    A spend that names its run (``run``, from 2026-10-16) belongs to that run. One that does not
    belongs to the one run among every writer of its ledger (``ledger_writers``, under
    ``runs_root``) whose window holds its second; meta times are truncated to the second, so a
    second two writers' windows share is ambiguous and refused, as is a spend of a writer that is
    not listed."""
    from .paths import runs_root as default_runs_root

    infos = [summarizer_run(Path(r)) for r in runs]
    root = Path(runs_root) if runs_root is not None else default_runs_root()
    writers = {ledger: ledger_writers(ledger, root) for ledger in {i.ledger for i in infos}}
    spends: dict[Path, dict[str, list[dict[str, Any]]]] = {}
    for ledger in {i.ledger for i in infos}:
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in ledger.read_text().splitlines():
            event = json.loads(line) if line else None
            if event and event["event"] == "spend":
                by_key[event["key"]].append(event)
        spends[ledger] = by_key
    entries: dict[str, dict[str, Any]] = {}
    for key in sorted(set().union(*(i.keys for i in infos)) if infos else set()):
        entry = cache.entry(key)
        if entry is None or entry.get("service_tier"):
            continue  # never answered, or its tier is recorded
        cached_at = _at(entry["cached_at"])
        claims = []  # (run, spend): a listed run's own ledger spend that matches the entry
        for ledger in {i.ledger for i in infos if key in i.keys}:
            matching = [
                s
                for s in spends[ledger][key]
                if _at(s["at"]).replace(microsecond=0) <= cached_at <= _at(s["at"]) + CACHE_LAG
            ]
            if len(matching) > 1:
                raise TierError(f"cache entry {key}: {len(matching)} spends in {ledger} match it")
            for s in matching:
                if s.get("run"):
                    source = s["run"]
                else:
                    second = _second(_at(s["at"]))
                    holders = [
                        w.run_id
                        for w in writers[ledger]
                        if _second(w.started_at) <= second
                        and (w.finished_at is None or second <= _second(w.finished_at))
                    ]
                    if len(holders) != 1:
                        raise TierError(
                            f"cache entry {key}: its spend's second {second.isoformat()} lies in "
                            f"{len(holders)} writers' windows of {ledger}"
                        )
                    source = holders[0]
                claims += [
                    (i, s)
                    for i in infos
                    if i.run_id == source
                    and key in i.keys
                    and i.ledger == ledger
                    and s["worker"] == i.worker
                ]
        if len(claims) != 1:
            raise TierError(f"cache entry {key}: {len(claims)} listed runs sent its spend")
        owner, spend = claims[0]
        entries[key] = {
            "tier": owner.tier,
            "run": owner.run_id,
            "ledger": str(owner.ledger),
            "spend_id": spend["id"],
            "spend_at": spend["at"],
            "cached_at": entry["cached_at"],
            "response_sha256": cache_entry_sha256(entry),
        }
    return {
        "version": MANIFEST_VERSION,
        "cache_root": str(Path(cache.root).resolve()),
        "runs": {
            i.run_id: {
                "tier": i.tier,
                "worker": i.worker,
                "ledger": str(i.ledger),
                "started_at": i.started_at.isoformat(),
                "finished_at": i.finished_at.isoformat(),
                "requests": len(i.keys),
            }
            for i in infos
        },
        "ledgers": {str(p): _file_sha256(p) for p in sorted({i.ledger for i in infos})},
        "entries": entries,
    }


def load_manifests(paths: Iterable[Path], cache: JudgeCache) -> dict[str, dict[str, str]]:
    """Key → {tier, response_sha256} of the given manifests, which must describe this cache and
    agree."""
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        if manifest.get("version") != MANIFEST_VERSION:
            raise TierError(f"{path}: not a version-{MANIFEST_VERSION} tier manifest")
        if manifest["cache_root"] != str(Path(cache.root).resolve()):
            raise TierError(f"{path} describes the cache {manifest['cache_root']}, not this one")
        for key, entry in manifest["entries"].items():
            mine = {"tier": entry["tier"], "response_sha256": entry["response_sha256"]}
            if out.setdefault(key, mine) != mine:
                raise TierError(f"manifests disagree on {key}")
    return out


def response_tier(
    key: str, completion: Completion, manifest: Mapping[str, Mapping[str, str]]
) -> str:
    """The tier of one consumed response: the one it carries, or for an older cache entry the
    manifest's, which must name that very entry (its sha256)."""
    if completion.service_tier:
        return completion.service_tier
    entry = manifest.get(key)
    if entry is None or completion.entry_sha256 is None:
        raise TierError(f"cached completion {key} has no recorded tier and no manifest entry")
    if entry["response_sha256"] != completion.entry_sha256:
        raise TierError(f"cached completion {key} is not the response its manifest entry names")
    return entry["tier"]


def response_tiers(
    completions: Mapping[str, Completion], manifest: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    """The tier of each answered request, from the completions ``judge.fetch`` returned."""
    return {
        key: response_tier(key, completion, manifest)
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
