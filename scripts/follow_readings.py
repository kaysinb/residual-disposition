"""The follow-up's readings of L, H_s and X by D* and unfiltered PD (preregistration §13.5, §13.14's
step 2, §13.16), on the API from recorded packages only (``residual_disposition.follow_readings``).

    uv run python scripts/follow_readings.py [--pin | --send | --judge-history RUN]

Without a flag it prints the plan and the due requests with their estimate, and checks the readings
against their committed pins when they exist. ``--pin`` writes every reading's package directory,
evidence sha256 and keys once (``manifests/follow_readings_inputs.json``), to be committed before
any request; it needs the committed records of the random branches and of the base decoder's
controls. ``--send`` sends only with the pins
committed and unchanged: every due interpreter attempt, every due label of the parsed hypotheses,
then every due grade of an L reading's counted hypotheses, each once and cached, in one run, so that
the cells compared within a reading share one window of time; on the meter item ``follow/base``'s
API session. A reply the committed judge histories list as unsupported is replaced as §13.16
provides. The run directory (``runs/follow_readings/<run>``) holds the inputs, every slot's attempts
before and after the sends, every answer as stored and the record (``readings.json``), which is not
printed. ``--judge-history`` establishes tier-less replies' provenance by the registered route,
once. With ``--sensitivity``, ``--pin`` and ``--send`` take the standard-only repeat of eli5 stack
(§13.16) instead: its own pins and run root, the interpreter on the standard tier with no fallback,
its six cells interleaved, its record (``sensitivity.json``) beside the original record's, unopened.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import sys
from typing import Any

import follow_qualification as S

from residual_disposition import continuation_readers as readers
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_readings as D
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import JudgeCache, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root

ITEM, LABEL = "follow/base", "follow/readings"
PINS = "manifests/follow_readings_inputs.json"  # each reading's pin, committed before sends
HISTORY = "manifests/stage3_judge_history_follow_readings.json"  # read by Q.judge_pins
QUALIFICATION = "follow-qualification-20261031T091953Z"  # the qualification's record
ROOT = "follow_readings"
SENSITIVITY_PINS = "manifests/follow_readings_sensitivity_inputs.json"
SENSITIVITY_ROOT = "follow_readings_sensitivity"
ORIGINAL = "follow-readings-20261031T201407Z"  # the readings' record the sensitivity stands beside


def inventory_reuse() -> dict[str, Any]:
    from residual_disposition import continuation_rules as cr

    manifest = json.loads((REPO_ROOT / cr.follow_b3()["inventory"]["manifest"]).read_text())
    return manifest["reuse"]


def plan(sensitivity: bool = False) -> list[dict[str, Any]]:
    """Every reading with its package directory, evidence's sha256, attempts or reuse decision."""
    from functools import cache

    from residual_disposition import follow_inventory as fi
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.manifests import committed_json

    confirmed = committed_json(REPO_ROOT / "manifests" / "follow_confirmations.json",
                               "the confirmations")["confirmed"]  # fmt: skip
    rows = D.readings(confirmed)
    expected = D.expected_states(rows)
    reuse = inventory_reuse()

    @cache
    def workspace(behavior: str) -> Any:
        config = f"experiments/continuation_14b/{behavior}"
        return P.Workspace(load_config(config, ["tracking.wandb.enabled=false"]))

    branches = cache(fi.random_branches)
    controls = cache(lambda: D.base_controls(find_run))
    out = []
    for r in D.sensitivity_readings(rows) if sensitivity else rows:
        directory = D.package_dir(r, find_run, expected, workspace, branches, controls)
        text = Q.evidence(r, directory)
        decision = reuse.get(r["salt"]) or {}
        out.append({"reading": r, "directory": str(directory),
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "reused": decision if decision.get("reuse") else None,
                    "attempts": [] if decision.get("reuse") else Q.attempts(r, text)})  # fmt: skip
    return out


def state(cells: list[dict[str, Any]], caches: dict[str, Any], judges: dict[str, Any],
          targets: dict[str, Any], pinned: dict[str, Any], replaced: frozenset[str],
          definitions: Any, spec: Any) -> list[dict[str, Any]]:  # fmt: skip
    """Every slot from the caches: new readings as the qualification's, reused ones read back; an L
    reading's counted hypotheses with their grades."""
    from residual_disposition.continuation_readers import counted

    fresh = [c for c in cells if c["reused"] is None]
    made = S.state(fresh, caches, judges, targets, pinned, replaced)
    rows = {id(c): r for c, r in zip(fresh, made, strict=True)}
    out = []
    for c in cells:
        r = c["reading"]
        if c["reused"] is not None:
            slots = D.reused_slots(c["reused"], caches["summary"], caches["judge"], r["salt"])
        else:
            slots = rows[id(c)]["slots"]
        if D.graded_reading(r):
            for i, s in enumerate(slots):
                lab = s.get("labels")
                if not lab or "value" not in lab["strict"] or "value" not in lab["lenient"]:
                    continue
                if counted(lab["strict"]["value"][0], lab["lenient"]["value"]):
                    where = f"{r['salt']}#slot={i}"
                    s["grades"] = D.grades(s["hypothesis"]["value"], r["behavior"], definitions,
                                           spec, caches["judge"], where)  # fmt: skip
        out.append({"reading": r, "slots": slots, "count": Q.reading_count(slots)})
    return out


def due(rows: list[dict[str, Any]]) -> list[Any]:
    """The attempts due now: interpreter, then labels, then grades."""
    now = S.due(rows)
    for row in rows:
        for s in row["slots"]:
            for g in (s.get("grades") or {}).values():
                if "due" in g:
                    now.append(g["due"])
    return list({r.key: r for r in now}.values())


def histories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {f"{row['reading']['salt']}#slot={i}": D.history_of(s)
            for row in rows for i, s in enumerate(row["slots"])}  # fmt: skip


def record(rows: list[dict[str, Any]], cells: list[dict[str, Any]], definitions: Any) -> dict:
    """The readings' record, from the caches, the qualification's floors and the validations."""
    from residual_disposition import continuation_rules as cr
    from residual_disposition import follow_regions as fr
    from residual_disposition.paths import runs_root

    floors = json.loads((runs_root() / "follow_qualification" / QUALIFICATION
                         / "qualification.json").read_text())["floors"]  # fmt: skip
    accepted, validated = {}, {}
    for fmt in ("dstar", "pd_unfiltered"):
        d = data_root() / "follow_format" / fmt
        accepted[fmt] = bool(json.loads((d / "acceptance.json").read_text())["passed"])
        validated[fmt] = bool(json.loads((d / "graders.json").read_text())["specificity"]["passed"])
    return {"readings": [{"salt": r["reading"]["salt"], "count": r["count"]} for r in rows],
            "evidence": {c["reading"]["salt"]: c["evidence_sha256"] for c in cells},
            "format": {"accepted": accepted, "specificity_validated": validated},
            **D.analysis(rows, floors, accepted, validated, fr.load(),
                         cr.follow_rules().data["readers"], definitions)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--send", action="store_true")
    mode.add_argument("--judge-history", metavar="RUN")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args(argv)
    if args.judge_history:
        return S.judge_history(args.judge_history, ROOT, HISTORY)
    from residual_disposition.manifests import committed_json
    from residual_disposition.paths import runs_root

    cells = plan(args.sensitivity)
    pinned_now = Q.pins(cells, runs_root())
    name = SENSITIVITY_PINS if args.sensitivity else PINS
    pins = REPO_ROOT / name
    if args.pin:
        if pins.exists():
            raise SystemExit(f"{name} exists: the pins are written once")
        S.write(pins, {"readings": pinned_now})
        print(f"wrote {name} ({hashlib.sha256(pins.read_bytes()).hexdigest()}): commit it")
        return 0
    if pins.exists() or args.send:
        found = Q.pin_problems(committed_json(pins, "the readings' pins")["readings"], pinned_now)
        if found:
            raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    judges = {kind: readers.judge_settings(readers.RUBRICS[kind]) for kind in ("lens", "pd")}
    interp = {kind: readers.interpreter(kind)[1] for kind in ("lens", "pd")}
    definitions, spec = D.grader()
    lenient = [j.model_for("lenient") for j in judges.values()]
    specs = {s.id: s for s in [*(i.model for i in interp.values()),
                               *(j.model for j in judges.values()), *lenient, spec]}  # fmt: skip
    targets = load_targets(sorted({c["reading"]["behavior"] for c in cells}))
    caches = {"summary": JudgeCache(data_root() / "cache" / "summarizer"),
              "judge": JudgeCache(data_root() / "cache" / "judge")}  # fmt: skip
    pinned = Q.judge_pins()
    replaced = frozenset(
        k for name in (S.HISTORY, HISTORY) if (REPO_ROOT / name).exists()
        for k in committed_json(REPO_ROOT / name, "a judge history")["unsupported"])  # fmt: skip
    root = runs_root() / (SENSITIVITY_ROOT if args.sensitivity else ROOT)
    root.mkdir(exist_ok=True)
    S.reconcile(root, caches)
    interpreter_keys = {a.key for c in cells for tries in c["attempts"] for a in tries}
    print(dict(collections.Counter((c["reading"]["reader"], c["reused"] is not None,
                                    c["reading"]["decoder"]) for c in cells)))  # fmt: skip
    extra = (definitions, spec)
    rows = state(cells, caches, judges, targets, pinned, replaced, *extra)
    now = due(rows)
    print(f"due now: {len(now)}; estimate ${S.estimate(now, specs, interpreter_keys):.2f}")
    print("pins: " + ("committed" if pins.exists() else "not written"))
    if not args.send:
        return 0
    from datetime import UTC, datetime

    from residual_disposition import follow_money

    S.LABEL = LABEL
    stem = "follow-readings-sensitivity-" if args.sensitivity else "follow-readings-"
    run = root / (stem + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True, exist_ok=False)
    S.write(run / "inputs.json", {"readings": [
        {**c["reading"], "directory": c["directory"], "evidence_sha256": c["evidence_sha256"]}
        for c in cells]})  # fmt: skip
    S.write(run / "histories-before.json", histories(rows))
    caches = {name: S.JournaledCache(c.root, run / "answers.jsonl", stage)
              for (name, c), stage in zip(caches.items(), ("interpreter", "judge"),
                                          strict=True)}  # fmt: skip
    _, ledger = follow_money.money(data_root(), ITEM, data_root() / S.METER_LOG)
    atexit.register(ledger.close)  # an interrupted run's session settled at its exit
    ledger.source = LABEL
    backends = {"summary": S.backend(ledger, specs, interpreter=not args.sensitivity,
                                     journal=S.journal_writer(run / "tier_fallback.jsonl")),
                "judge": S.backend(ledger, specs, interpreter=False)}  # fmt: skip
    for name, b in backends.items():
        b.reconcile(caches[name])
    tried: set[str] = set()
    while True:
        S.reconcile(root, caches)
        rows = state(cells, caches, judges, targets, pinned, replaced, *extra)
        now = [r for r in due(rows) if r.key not in tried]
        if not now:
            break
        ledger.check(S.estimate(now, specs, interpreter_keys))
        for name, batch in (
            ("summary", D.interleaved([r for r in now if r.key in interpreter_keys], cells)),
            ("judge", [r for r in now if r.key not in interpreter_keys]),
        ):
            if batch:
                fetch(batch, backends[name], caches[name])
        tried |= {r.key for r in now}
    charged = ledger.close()
    rows = state(cells, caches, judges, targets, pinned, replaced, *extra)
    S.write(run / "histories.json", histories(rows))
    if args.sensitivity:
        from residual_disposition import continuation_rules as cr
        from residual_disposition import follow_regions as fr

        original = json.loads((runs_root() / ROOT / ORIGINAL / "readings.json").read_text())
        S.write(run / "sensitivity.json", {
            "readings": [{"salt": r["reading"]["salt"], "count": r["count"]} for r in rows],
            "original": ORIGINAL, "charged": charged,
            **D.sensitivity(rows, original, fr.load(), cr.follow_rules().data["readers"],
                            definitions["eli5"])})  # fmt: skip
        print(f"charged {charged}; record {run}")
        return 0
    S.write(run / "readings.json", {**record(rows, cells, definitions), "charged": charged,
                                    "replaced_replies": sorted(replaced)})  # fmt: skip
    print(f"charged {charged}; record {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
