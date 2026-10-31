"""The follow-up's qualification of D* and unfiltered PD (preregistration §13.5 (a); barrier B2) on
the API, from recorded packages only (``residual_disposition.follow_qualification``).

    uv run python scripts/follow_qualification.py [--pin | --send | --judge-history RUN]

Before anything is rendered every new package is bound to the frozen build plan and the build
session's pinned records (``follow_qualification.new_packages``). Without a flag it prints the
plan (readings by reader, purpose and decoder), the prompts' sha256, the due requests with their
estimate and the judging's projection, and checks the readings against their committed pins
when they exist. ``--pin`` writes the pins once (``manifests/follow_qualification_inputs.json``:
each reading's package directory, evidence sha256 and keys), to be committed before any
request; ``--send`` sends only with the pins committed and unchanged. Every run checks that the
answers earlier runs recorded (their ``histories.json``) are still in the caches, unchanged; a
lost or changed answer stops it, nothing is redrawn. With ``--send``: every due interpreter
attempt (a slot's request, then its one retry once the parser refuses the first answer), then
every due judge attempt of the parsed hypotheses, each sent once and cached
(``<DATA_DIR>/cache/summarizer``, ``<DATA_DIR>/cache/judge``), every send reserved on the meter
item ``follow/base``'s API session (``follow_money``) under ``follow/qualification`` at the price
of the tier it is sent on, until nothing is due; an attempt whose send failed is not sent again
in this run (rerun to resend).

The interpreter reads on the flex tier with the automatic fallback to the standard one
(``judge.FlexFallback``: a window of 100, a share of HTTP 503 of at least 0.9, 15 minutes of
silence), as Stage 3's interpreters did; its journal (``tier_fallback.jsonl``) is in the run.
The judge reads on the standard tier, as B2 fixes it. A tier is never part of a request's body,
key or salt: it is recorded with each cached answer. The run directory
(``runs/follow_qualification/<run>``) holds the inputs, every slot's attempts before and after
the sends (``histories-before.json``, ``histories.json``: interpreter, strict and lenient, each
attempt's key, entry sha256, recorded and admitted tier and parse, rejected replies included),
every answer as it is stored (``answers.jsonl``, synced at once; before each round every answer
recorded so far must still be in the cache, unchanged, and a key recorded with two entries is
refused) and
the record, recomputed from the cache: each reading's counted hypotheses with its safety blocks,
the pooled rates, the floors and, descriptive, the rates and floors with the blocked lenient yes
counted (``qualification.json``), pending while a slot is. An answer at a tier that is not
admitted (the interpreter's flex or standard; the judge's as the readers admit it) leaves its slot
pending and is never asked again; ``--judge-history`` establishes the tier-less replies' provenance
by the registered route, once.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import sys
from typing import Any

from residual_disposition import continuation_readers as readers
from residual_disposition import follow_qualification as Q
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import ChatBackend, JudgeCache, JudgeItem, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root

ITEM, LABEL = "follow/base", "follow/qualification"
PINS = "manifests/follow_qualification_inputs.json"  # each reading's pin, committed before sends
HISTORY = "manifests/stage3_judge_history_follow_qualification.json"  # read by Q.judge_pins
METER_LOG = "ledgers/continuation_meter.jsonl"
# the estimate's units (the Stage-3 diagnostic's, measured there): characters per token, the
# interpreter's thinking tokens, the margin, an answer's tokens
CHARS, THINKING, MARGIN, ANSWER = 2.2, 1455, 1.3, 40


def plan() -> list[dict[str, Any]]:
    """Every reading with its package directory, evidence's sha256 and attempts."""
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.manifests import committed_json

    confirmed = committed_json(REPO_ROOT / "manifests" / "follow_confirmations.json",
                               "the confirmations")["confirmed"]  # fmt: skip
    readings = Q.readings(confirmed)
    new = Q.new_packages(Q.new_names(readings))  # bound before anything is rendered
    expected, workspaces = Q.expected_states(readings), {}

    def workspace(behavior: str) -> Any:
        if behavior not in workspaces:
            config = f"experiments/continuation_14b/{behavior}"
            off = ["tracking.wandb.enabled=false"]
            workspaces[behavior] = P.Workspace(load_config(config, off))
        return workspaces[behavior]

    out = []
    for r in readings:
        directory = Q.package_dir(r, find_run, new, expected, workspace)
        text = Q.evidence(r, directory)
        out.append({"reading": r, "directory": str(directory),
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def state(cells: list[dict[str, Any]], caches: dict[str, Any], judges: dict[str, Any],
          targets: dict[str, Any], pinned: dict[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """Every slot's hypothesis and labels from the caches, each attempt admitted at its tier."""
    from residual_disposition.summarizer import parse_hypothesis

    rows = []
    for c in cells:
        r, kind = c["reading"], Q.KINDS[c["reading"]["reader"]]
        slots = []
        for i, tries in enumerate(c["attempts"]):
            where = f"{r['salt']}#slot={i}"
            hyp = Q.answer(caches["summary"], tries, parse_hypothesis, Q.interpreter_tier, where)
            lab = None
            if "value" in hyp:
                item = JudgeItem(where, r["behavior"], targets[r["behavior"]], hyp["value"])
                lab = Q.labels(item, judges[kind], caches["judge"], pinned)
            slots.append({"hypothesis": hyp, "labels": lab})
        rows.append({"reading": r, "slots": slots})
    return rows


def due(rows: list[dict[str, Any]]) -> list[Any]:
    """The attempts due now: unanswered interpreter attempts, then unanswered label attempts."""
    out = []
    for row in rows:
        for s in row["slots"]:
            if "due" in s["hypothesis"]:
                out.append(s["hypothesis"]["due"])
            elif s["labels"] is not None:
                for kind in ("strict", "lenient"):
                    if "due" in s["labels"][kind]:
                        out.append(s["labels"][kind]["due"])
    return list({r.key: r for r in out}.values())


def estimate(requests: list[Any], specs: dict[str, Any], interpreter_keys: set[str]) -> float:
    """Characters over ``CHARS`` in; an interpreter request's thinking with its margin and its
    answer out; a judge request the judge configuration's expected tokens with the margin."""
    from residual_disposition.config import load_config
    from residual_disposition.judge import request_chars

    cost = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    usd = 0.0
    for r in requests:
        spec = specs[r.body["model"]]
        if r.key in interpreter_keys:
            tokens_in, tokens_out = request_chars(r) / CHARS, THINKING * MARGIN + ANSWER
        else:
            tokens_in = request_chars(r) / float(cost.chars_per_token)
            tokens_out = MARGIN * (int(cost.expected_output_tokens[r.kind])
                                   + spec.expected_thinking_tokens)  # fmt: skip
        usd += (tokens_in * spec.input_usd_per_mtok + tokens_out * spec.output_usd_per_mtok) / 1e6
    return usd


FLEX = {"window": 100, "min_503_share": 0.9, "silence_min": 15}  # Stage 3's interpreters'


def journal_writer(path: Any) -> Any:
    """Append each fallback event to ``path`` as a JSON line, flushed and synced."""
    import os

    def write(event: dict[str, Any]) -> None:
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
            os.fsync(f.fileno())

    return write


def tier_factors() -> dict[str, float]:
    """Each tier's price factor (the judge configuration's discounts)."""
    from residual_disposition.config import load_config
    from residual_disposition.judge import tier_factor

    cost = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.cost
    return {t: tier_factor(t, batch_discount=float(cost.batch_discount),
                           flex_discount=float(cost.flex_discount))
            for t in ("standard", "flex", "batch")}  # fmt: skip


def backend(ledger: Any, specs: dict[str, Any], *, interpreter: bool, journal: Any = None) -> Any:
    """A synchronous backend, every attempt reserved on the item's ledger under ``LABEL`` at its
    own model's price on the tier it is sent on: the interpreter's on flex with the automatic
    fallback (``FLEX``, its events in ``journal``), the judge's on the standard tier."""
    import os

    import openai

    from residual_disposition import stage3_money
    from residual_disposition.config import load_config
    from residual_disposition.judge import FlexFallback

    base, key = os.environ.get("JUDGE_API_BASE"), os.environ.get("JUDGE_API_KEY")
    if not base or not key:
        raise SystemExit("Set JUDGE_API_BASE and JUDGE_API_KEY (.env)")
    sync = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.sync
    client = openai.OpenAI(api_key=key, base_url=base, timeout=sync.timeout_s, max_retries=0)
    fallback = FlexFallback(window=FLEX["window"], min_503_share=FLEX["min_503_share"],
                            silence_s=60.0 * FLEX["silence_min"],
                            journal=journal) if interpreter else None  # fmt: skip
    inner = ChatBackend(client, concurrency=int(sync.concurrency),
                        max_retries=int(sync.max_retries), backoff_s=float(sync.backoff_s),
                        backoff_max_s=float(sync.backoff_max_s),
                        service_tier="flex" if interpreter else None,
                        fallback=fallback)  # fmt: skip
    factors = tier_factors()
    tier = "flex" if interpreter else "standard"
    return stage3_money.ChainedBackend(inner, ledger, lambda r: specs[r.body["model"]],
                                       price_factor=factors[tier], tier_factors=factors,
                                       label=LABEL)  # fmt: skip


def record(rows: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    """The qualification's record: each reading's count with its safety blocks, the pooled rates
    and the floors; descriptive, the rates and floors with the blocked lenient yes counted."""
    from residual_disposition.continuation_rules import follow_rules

    rules = follow_rules().data
    counted = [{"reading": r["reading"], "count": Q.reading_count(r["slots"])} for r in rows]
    rates, upper = Q.pooled(counted), Q.upper_rates(counted)

    def text(found: dict[Any, Any]) -> dict[str, Any]:
        return {"/".join(str(x) for x in k): (None if v is None else str(v))
                for k, v in found.items()}  # fmt: skip

    return {"readings": [{"salt": c["reading"]["salt"], "count": c["count"]} for c in counted],
            "evidence": {c["reading"]["salt"]: c["evidence_sha256"] for c in cells},
            "rates": text(rates), "floors": Q.floors(rates, rules),
            "safety_blocks": {"rule": "descriptive (§12.4.5, 2026-10-26): never selects a verdict",
                              "rates_with_blocked_yes": text(upper),
                              "floors_with_blocked_yes": Q.floors(upper, rules)}}  # fmt: skip


def write(path: Any, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def inputs(cells: list[dict[str, Any]], judges: dict[str, Any]) -> dict[str, Any]:
    """What the run reads: every reading with its package directory and evidence's sha256, the
    prompts' and the judges' frozen versions."""
    prompts = {}
    for c in cells:
        p, settings = Q.prompt_and_settings(c["reading"])
        prompts[p.version] = {
            "sha256": p.sha256,
            "model": settings.model.id,
            "max_tokens": settings.max_tokens,
            "samples": settings.samples,
        }
    return {
        "readings": [
            {**c["reading"], "directory": c["directory"], "evidence_sha256": c["evidence_sha256"]}
            for c in cells
        ],
        "prompts": prompts,
        "judges": {
            k: {"rubric_version": j.rubric_version, "model": j.model.id} for k, j in judges.items()
        },
        "interpreter_tier": {"tier": "flex", "fallback": FLEX},
        "judge_tier": "standard",
    }


def histories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every slot's attempts by stage (``follow_qualification.history_of``), by reading and slot."""
    return {f"{row['reading']['salt']}#slot={i}": Q.history_of(s)
            for row in rows for i, s in enumerate(row["slots"])}  # fmt: skip


class JournaledCache(JudgeCache):
    """A cache whose every stored answer is also appended at once to the run's journal
    (``answers.jsonl``: the cache, the key and the entry's sha256 and tier), flushed and synced,
    so that an interruption never loses what was answered."""

    def __init__(self, root: Any, journal: Any, name: str) -> None:
        super().__init__(root)
        self.journal, self.name = journal, name

    def put(self, request: Any, completion: Any, *, service_tier: str | None = None) -> None:
        import os

        from residual_disposition.judge import cache_entry_sha256

        super().put(request, completion, service_tier=service_tier)
        entry = self.entry(request.key)
        line = {"cache": self.name, "key": request.key, "sha256": cache_entry_sha256(entry),
                "tier": entry.get("service_tier")}  # fmt: skip
        with self.journal.open("a") as f:
            f.write(json.dumps(line) + "\n")
            f.flush()
            os.fsync(f.fileno())


def previous_histories(root: Any) -> dict[str, Any]:
    """The attempts every earlier run recorded, by slot and stage (its histories) and in its
    answers' journal (``answers.jsonl``, by cache); a key recorded with two entries is refused."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    seen: dict[str, str] = {}

    def keep(slot: str, stage: str, step: dict[str, Any], where: Any) -> None:
        if seen.setdefault(step["key"], step["sha256"]) != step["sha256"]:
            raise SystemExit(f"{step['key']}: recorded with two entries ({where})")
        out.setdefault(slot, {}).setdefault(stage, {}).setdefault(step["key"], step)

    for path in sorted(root.glob("*/histories*.json")):
        for slot, stages in json.loads(path.read_text()).items():
            for stage, steps in stages.items():
                for step in steps:
                    keep(slot, stage, step, path)
    for path in sorted(root.glob("*/answers.jsonl")):
        for line in path.read_text().splitlines():
            step = json.loads(line)  # a line cut by an interruption is refused here
            keep(f"journal/{step['cache']}", step["cache"], step, path)
    return {slot: {stage: list(v.values()) for stage, v in stages.items()}
            for slot, stages in out.items()}  # fmt: skip


def judge_history(run: str) -> int:
    """The provenance of the tier-less judge replies the run's slots could not admit, by the
    registered route (§12.4.5, 2026-10-27; ``continuation_readers.judge_history``), written once to
    ``HISTORY``, which ``follow_qualification.judge_pins`` reads once committed. Hashes and
    provenance only; it prints counts, no label."""
    from residual_disposition.manifests import committed_json
    from residual_disposition.paths import runs_root
    from residual_disposition.summarizer import parse_hypothesis

    path = REPO_ROOT / HISTORY
    if path.exists():
        raise SystemExit(f"{HISTORY} exists: the judge history is written once")
    d = runs_root() / "follow_qualification" / run
    histories = json.loads((d / "histories.json").read_text())
    readings = json.loads((d / "inputs.json").read_text())["readings"]
    cache = JudgeCache(data_root() / "cache" / "summarizer")
    source = committed_json(REPO_ROOT / "manifests" / "stage3_readers_strong_eli5_cont_200.json",
                            "a readers record")["sources"]  # fmt: skip
    judges = {"lens": source["lens_judge"], "pd": source["pd_judge"]}  # their cache, the judge's
    wanted: dict[str, dict[str, dict[str, tuple]]] = {}
    for r in readings:
        for i in range(int(r["slots"])):
            slot = f"{r['salt']}#slot={i}"
            steps = histories[slot]
            if any(s.get("admitted") is None for k in ("strict", "lenient")
                   for s in steps.get(k) or []):  # fmt: skip
                entry = cache.entry(next(s for s in steps["interpreter"] if s["parsed"])["key"])
                kinds = wanted.setdefault(r["behavior"], {"lens": {}, "pd": {}})
                kinds[Q.KINDS[r["reader"]]][slot] = (None, None, parse_hypothesis(entry["text"]))
    found = {b: readers.judge_history(find_run, b, judges, kinds, runs_root())
             for b, kinds in sorted(wanted.items())}  # fmt: skip
    entries = {k: v for f in found.values() for k, v in f["entries"].items()}
    unsupported = sorted({k for f in found.values() for k in f["unsupported"]})
    tiers = dict(sorted(collections.Counter(e["tier"] for e in entries.values()).items()))
    any_found = next(iter(found.values()))
    write(path, {"rule": any_found["rule"], "recording_boundary": any_found["recording_boundary"],
                 "run": run, "behaviors": sorted(found), "judge_runs": judges,
                 "references": {b: f["references"] for b, f in found.items()}, "tiers": tiers,
                 "entries": entries, "unsupported": unsupported})  # fmt: skip
    print(f"wrote {HISTORY}: {len(entries)} entries {tiers}, {len(unsupported)} unsupported")
    return 0


def reconcile(root: Any, caches: dict[str, Any]) -> None:
    """Stop when an answer an earlier run recorded (its histories or its answers' journal) is no
    longer in the cache, or changed: it is never taken for unanswered."""
    by_stage = {"interpreter": caches["summary"], "judge": caches["judge"]}
    lost = Q.history_problems(previous_histories(root), by_stage)
    if lost:
        raise SystemExit(f"{len(lost)} recorded answers lost or changed: " + "; ".join(lost[:5]))


def projection(cells: list[dict[str, Any]], judges: dict[str, Any], targets: dict[str, Any],
               specs: dict[str, Any], chars: int = 300) -> float:  # fmt: skip
    """The judging's projection: each slot's strict and lenient request on a probe hypothesis of
    ``chars`` characters, at the judge configuration's expected tokens with the margin (not a
    bound: the hypotheses are not drawn yet; every retry would double it)."""
    usd = 0.0
    for c in cells:
        r = c["reading"]
        item = JudgeItem("probe", r["behavior"], targets[r["behavior"]], "x" * chars)
        tries = Q.label_attempts(item, judges[Q.KINDS[r["reader"]]])
        usd += Q.SLOTS * estimate([tries["strict"][0], tries["lenient"][0]], specs, set())
    return usd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--send", action="store_true")
    mode.add_argument("--judge-history", metavar="RUN")
    args = parser.parse_args(argv)
    if args.judge_history:
        return judge_history(args.judge_history)
    from residual_disposition.manifests import committed_json
    from residual_disposition.paths import runs_root

    cells = plan()
    pinned_now = Q.pins(cells, runs_root())
    pins = REPO_ROOT / PINS
    if args.pin:
        if pins.exists():
            raise SystemExit(f"{PINS} exists: the pins are written once")
        write(pins, {"readings": pinned_now})
        print(f"wrote {PINS} ({hashlib.sha256(pins.read_bytes()).hexdigest()}): commit it")
        return 0
    if pins.exists() or args.send:
        found = Q.pin_problems(committed_json(pins, "the qualification's pins")["readings"],
                               pinned_now)  # fmt: skip
        if found:
            raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    judges = {kind: readers.judge_settings(readers.RUBRICS[kind]) for kind in ("lens", "pd")}
    interp = {kind: readers.interpreter(kind)[1] for kind in ("lens", "pd")}
    specs = {s.id: s for s in [*(i.model for i in interp.values()),
                               *(j.model for j in judges.values()),
                               *(j.model_for("lenient") for j in judges.values())]}  # fmt: skip
    targets = load_targets(sorted({c["reading"]["behavior"] for c in cells}))
    caches = {"summary": JudgeCache(data_root() / "cache" / "summarizer"),
              "judge": JudgeCache(data_root() / "cache" / "judge")}  # fmt: skip
    pinned = Q.judge_pins()
    root = runs_root() / "follow_qualification"
    reconcile(root, caches)
    interpreter_keys = {a.key for c in cells for tries in c["attempts"] for a in tries}
    print(dict(collections.Counter((c["reading"]["reader"], c["reading"]["purpose"],
                                    c["reading"]["decoder"]) for c in cells)))  # fmt: skip
    for c in cells[:1]:
        print("prompt", Q.prompt_and_settings(c["reading"])[0].version)
    rows = state(cells, caches, judges, targets, pinned)
    now = due(rows)
    print(f"due now: {len(now)}; estimate ${estimate(now, specs, interpreter_keys):.2f}")
    print("judging projected (a 300-character probe hypothesis per slot, no retry): "
          f"${projection(cells, judges, targets, specs):.2f}")  # fmt: skip
    print("pins: " + ("committed" if pins.exists() else "not written"))
    if not args.send:
        return 0
    from datetime import UTC, datetime

    from residual_disposition import follow_money

    run = root / ("follow-qualification-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True, exist_ok=False)
    write(run / "inputs.json", inputs(cells, judges))
    write(run / "histories-before.json", histories(rows))
    caches = {name: JournaledCache(c.root, run / "answers.jsonl", stage)
              for (name, c), stage in zip(caches.items(), ("interpreter", "judge"),
                                          strict=True)}  # fmt: skip
    _, ledger = follow_money.money(data_root(), ITEM, data_root() / METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's session settled at its exit
    ledger.source = LABEL
    backends = {"summary": backend(ledger, specs, interpreter=True,
                                   journal=journal_writer(run / "tier_fallback.jsonl")),
                "judge": backend(ledger, specs, interpreter=False)}  # fmt: skip
    for name, b in backends.items():
        b.reconcile(caches[name])
    tried: set[str] = set()
    while True:
        reconcile(root, caches)  # every answer recorded so far, this run's included, unchanged
        now = [r for r in due(state(cells, caches, judges, targets, pinned)) if r.key not in tried]
        if not now:
            break
        ledger.check(estimate(now, specs, interpreter_keys))  # the worst case: all on standard
        for name, batch in (
            ("summary", [r for r in now if r.key in interpreter_keys]),
            ("judge", [r for r in now if r.key not in interpreter_keys]),
        ):
            if batch:
                fetch(batch, backends[name], caches[name])
        tried |= {r.key for r in now}
    charged = ledger.close()
    rows = state(cells, caches, judges, targets, pinned)
    write(run / "histories.json", histories(rows))
    result = {**record(rows, cells), "charged": charged}
    write(run / "qualification.json", result)
    print(f"charged {charged}; record {run}")
    print(json.dumps(result["floors"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
