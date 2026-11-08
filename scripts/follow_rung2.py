"""Rung 2 of the full-threshold ladder on eli5 continued 200, on both decoders (preregistration
§13.6, §13.16), on the API from recorded packages and the committed steering samples
(``residual_disposition.follow_rung2``).

    uv run python scripts/follow_rung2.py [--pin | --send PHASE [--stream N] [--qualification RUN]]
    uv run python scripts/follow_rung2.py --record-steering RUN [--write]
    uv run python scripts/follow_rung2.py --judge-history RUN

Without a flag it prints the plan and the due requests with their estimate per phase, and checks the
readings against their committed pins when they exist. ``--record-steering`` records the pulled
session's samples once (``manifests/follow_rung2_steering.json``); the readings find them only
through it. ``--pin`` writes every reading's package directory, evidence sha256 and keys once
(``manifests/follow_rung2_inputs.json``), to be committed before any request. ``--send`` sends only
with the pins committed and unchanged, one phase at a time: ``qualification`` (its record
``qualification.json``), ``supplement`` (one stream of the frame's supplement, ``--stream``) and
``analysis`` (only with the ``--qualification`` run complete and the rung's format accepted; a
decoder failing its floors is read as exploratory; its record ``rung2.json``). Each reading is
charged to its own meter item (``follow_rung2.item_of``), one API session per item, and a phase's
readings on one item are interleaved. Every attempt, tier rule, cache and history is the
readings' (``follow_readings``). The records are not printed.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import follow_qualification as S
import follow_readings as RS

from residual_disposition import continuation_readers as readers
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_readings as D
from residual_disposition import follow_rung2 as R2
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import JudgeCache, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

LABEL = "follow/rung2"
PINS = "manifests/follow_rung2_inputs.json"  # each reading's pin, committed before sends
HISTORY = "manifests/stage3_judge_history_follow_rung2.json"  # read by Q.judge_pins
ROOT = "follow_rung2"
FORMAT = "rung2"
RUN = "follow-rung2"  # a send's run directory: <RUN>-<phase>-<time>
# the phases --send takes; a phase other than these three (another rung's, as follow_partial's
# ablation) is sent as planned, its counts its record (<phase>.json)
SENDS = ("qualification", "supplement", "analysis")


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading by phase, with its package directory, evidence's sha256 and attempts."""
    from residual_disposition import follow_inventory as fi
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.manifests import committed_json

    confirmed = committed_json(REPO_ROOT / "manifests" / "follow_confirmations.json",
                               "the confirmations")["confirmed"]  # fmt: skip
    expected = R2.expected_states(confirmed)
    samples = R2.steering(find_run)

    @cache
    def workspace(behavior: str) -> Any:
        config = f"experiments/continuation_14b/{behavior}"
        return P.Workspace(load_config(config, ["tracking.wandb.enabled=false"]))

    branches = cache(fi.random_branches)
    controls = cache(lambda: D.base_controls(find_run))
    out: dict[str, list[dict[str, Any]]] = {}
    for phase, rows in R2.readings().items():
        out[phase] = []
        for r in rows:
            directory = D.package_dir(r, find_run, expected, workspace, branches, controls)
            text = R2.evidence(r, directory, samples)
            out[phase].append({
                "reading": r, "directory": str(directory), "reused": None,
                "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def record_steering(run_name: str, write: bool) -> int:
    """The pulled session's samples as ``manifests/follow_rung2_steering.json``, once."""
    run = runs_root() / "follow_rung2_steering" / run_name
    found = sorted(run.glob("out/*/result.json"))
    if len(found) != 1:
        raise SystemExit(f"{run}: {len(found)} session results, not one")
    out, result = found[0].parent, json.loads(found[0].read_text())
    planned = {c["cell"]: c for c in R2.steering_cells()}
    if result["status"] != "complete" or sorted(result["cells"]) != sorted(planned):
        raise SystemExit(f"{run}: not the planned cells, complete")
    cells = {}
    for name, cell in result["cells"].items():
        d = out / "cells" / name
        files = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(d.rglob("*")) if p.is_file()}  # fmt: skip
        if files != cell["files"] or cell["run"] != planned[name]["run"]:
            raise SystemExit(f"{name}: not its recorded, complete files")
        steering = json.loads((d / "steering.json").read_text())
        cells[name] = {"dir": str(d.relative_to(run)), "run": cell["run"], "files": files,
                       "failed_positions": sorted(p for p, e in steering["per_position"].items()
                                                  if e["failed"])}  # fmt: skip
    record = {"run": run_name, "out": str(out.relative_to(run)), "cells": cells,
              "result_sha256": hashlib.sha256(found[0].read_bytes()).hexdigest()}  # fmt: skip
    path = REPO_ROOT / R2.STEERING_RECORD
    if not write:
        print(json.dumps({k: v["failed_positions"] for k, v in cells.items()}, indent=1))
        return 0
    if path.exists():
        raise SystemExit(f"{R2.STEERING_RECORD} exists: the record is written once")
    S.write(path, record)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {R2.STEERING_RECORD} ({digest}): commit it")
    return 0


def format_state() -> tuple[bool, bool]:
    """The rung's format validation: accepted, and its specificity grader validated."""
    d = data_root() / "follow_format" / FORMAT
    if not (d / "acceptance.json").is_file() or not (d / "graders.json").is_file():
        raise SystemExit("the rung's format validation is not finished")
    graders = json.loads((d / "graders.json").read_text())
    return bool(json.loads((d / "acceptance.json").read_text())["passed"]), bool(
        graders["specificity"]["passed"]
    )


def send(cells: list[dict[str, Any]], run: Path, ctx: dict[str, Any]) -> dict[str, float]:
    """The phase's readings, item by item (``follow_rung2.item_of``), each in one API session: every
    due interpreter attempt interleaved, then the labels and the grades, until nothing is due."""
    from residual_disposition import follow_money

    S.LABEL = LABEL
    root = runs_root() / ROOT
    caches = {name: S.JournaledCache(c.root, run / "answers.jsonl", stage)
              for (name, c), stage in zip(ctx["caches"].items(), ("interpreter", "judge"),
                                          strict=True)}  # fmt: skip
    spent: dict[str, float] = {}
    by_item = collections.defaultdict(list)
    for c in cells:
        by_item[R2.item_of(c["reading"])].append(c)
    for item, part in by_item.items():
        keys = {a.key for c in part for tries in c["attempts"] for a in tries}
        _, ledger = follow_money.money(data_root(), item, data_root() / S.METER_LOG)
        atexit.register(ledger.close)  # an interrupted run's session settled at its exit
        ledger.source = LABEL
        backends = {"summary": S.backend(ledger, ctx["specs"], interpreter=True,
                                         journal=S.journal_writer(run / "tier_fallback.jsonl")),
                    "judge": S.backend(ledger, ctx["specs"], interpreter=False)}  # fmt: skip
        for name, b in backends.items():
            b.reconcile(caches[name])
        tried: set[str] = set()
        while True:
            S.reconcile(root, caches)
            now = [r for r in RS.due(state(part, caches, ctx)) if r.key not in tried]
            if not now:
                break
            ledger.check(S.estimate(now, ctx["specs"], keys))
            asked = D.interleaved([r for r in now if r.key in keys], part)
            labels = [r for r in now if r.key not in keys]
            for name, batch in (("summary", asked), ("judge", labels)):
                if batch:
                    fetch(batch, backends[name], caches[name])
            tried |= {r.key for r in now}
        spent[item] = ledger.close()
    return spent


def state(cells: list[dict[str, Any]], caches: dict[str, Any], ctx: dict[str, Any]) -> list[dict]:
    return RS.state(cells, caches, ctx["judges"], ctx["targets"], ctx["pinned"], ctx["replaced"],
                    ctx["definitions"], ctx["spec"])  # fmt: skip


def context() -> dict[str, Any]:
    """The readings' frozen judges, targets, pinned and replaced judge replies, grader, model specs
    and caches."""
    from residual_disposition.manifests import committed_json

    judges = {"lens": readers.judge_settings(readers.RUBRICS["lens"])}
    interp = readers.interpreter("lens")[1]
    definitions, spec = D.grader()
    judge = judges["lens"]
    specs = {s.id: s for s in (interp.model, judge.model, judge.model_for("lenient"), spec)}
    pinned = Q.judge_pins()
    replaced = frozenset(
        k for name in (S.HISTORY, RS.HISTORY, RS.SENSITIVITY_HISTORY, HISTORY)
        if (REPO_ROOT / name).exists()
        for k in committed_json(REPO_ROOT / name, "a judge history")["unsupported"])  # fmt: skip
    behaviors = getattr(R2, "BEHAVIORS", (R2.BEHAVIOR,))
    return {"judges": judges, "targets": load_targets(behaviors), "pinned": pinned,
            "replaced": replaced, "definitions": definitions, "spec": spec, "specs": specs,
            "caches": {"summary": JudgeCache(data_root() / "cache" / "summarizer"),
                       "judge": JudgeCache(data_root() / "cache" / "judge")}}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--send", choices=SENDS)
    mode.add_argument("--record-steering", metavar="RUN")
    mode.add_argument("--judge-history", metavar="RUN")
    parser.add_argument("--stream", type=int, choices=(3, 4, 5, 6))
    parser.add_argument("--qualification", metavar="RUN")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.record_steering:
        return record_steering(args.record_steering, args.write)
    if args.judge_history:
        return S.judge_history(args.judge_history, ROOT, HISTORY)
    from residual_disposition.manifests import committed_json

    phases = plan()
    every = [c for part in phases.values() for c in part]
    pinned_now = Q.pins(every, runs_root())
    pins = REPO_ROOT / PINS
    if args.pin:
        if pins.exists():
            raise SystemExit(f"{PINS} exists: the pins are written once")
        S.write(pins, {"readings": pinned_now})
        print(f"wrote {PINS} ({hashlib.sha256(pins.read_bytes()).hexdigest()}): commit it")
        return 0
    if pins.exists() or args.send:
        found = Q.pin_problems(committed_json(pins, "the rung's pins")["readings"], pinned_now)
        if found:
            raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    ctx = context()
    specs, replaced, definitions = ctx["specs"], ctx["replaced"], ctx["definitions"]
    root = runs_root() / ROOT
    root.mkdir(exist_ok=True)
    S.reconcile(root, ctx["caches"])
    if not args.send:
        print({k: len(v) for k, v in phases.items()})
        for phase, part in phases.items():
            keys = {a.key for c in part for tries in c["attempts"] for a in tries}
            now = RS.due(state(part, ctx["caches"], ctx))
            print(f"{phase}: due {len(now)}; estimate ${S.estimate(now, specs, keys):.2f}; "
                  + str(collections.Counter(R2.item_of(c["reading"]) for c in part)))  # fmt: skip
        print("pins: " + ("committed" if pins.exists() else "not written"))
        return 0
    from datetime import UTC, datetime

    from residual_disposition import continuation_rules as cr
    from residual_disposition import follow_regions as fr

    rules = cr.follow_rules().data["readers"]
    floors = None
    if args.send == "qualification":
        cells = phases["qualification"]
    elif args.send == "supplement":
        if args.stream is None:
            raise SystemExit("--stream names the supplement's stream")
        cells = [c for c in phases["validation"] if c["reading"]["stream"] == args.stream]
    elif args.send != "analysis":
        cells = phases[args.send]
    else:
        if not args.qualification:
            raise SystemExit("--qualification names the qualification's run")
        record = json.loads((root / args.qualification / "qualification.json").read_text())
        floors = record["floors"]
        if any(None in (f["eligible"], f["recipe_floor_met"]) for f in floors.values()):
            raise SystemExit("a qualification reading is pending: its slots are read first")
        accepted, validated = format_state()
        if not accepted:
            raise SystemExit("the rung's format was not accepted: its readings are ineligible")
        cells = phases["analysis"]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = root / f"{RUN}-{args.send}-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    S.write(run / "inputs.json", {"readings": [
        {**c["reading"], "directory": c["directory"], "evidence_sha256": c["evidence_sha256"]}
        for c in cells]})  # fmt: skip
    S.write(run / "histories-before.json", RS.histories(state(cells, ctx["caches"], ctx)))
    charged = send(cells, run, ctx)
    rows = state(cells, ctx["caches"], ctx)
    S.write(run / "histories.json", RS.histories(rows))
    counts = {"readings": [{"salt": r["reading"]["salt"], "count": r["count"]} for r in rows],
              "evidence": {c["reading"]["salt"]: c["evidence_sha256"] for c in cells},
              "charged": charged, "replaced_replies": sorted(replaced)}  # fmt: skip
    if args.send == "qualification":
        S.write(run / "qualification.json", {**counts, "floors": R2.qualification(rows, rules)})
    elif args.send == "analysis":
        S.write(run / "rung2.json", {**counts, "qualification": args.qualification, **R2.decision(
            rows, floors, accepted, validated, fr.load(), rules, definitions)})  # fmt: skip
    elif args.send != "supplement":
        S.write(run / f"{args.send}.json", counts)
    print(f"charged {charged}; run {run.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
