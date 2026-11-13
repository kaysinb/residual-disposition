"""Ten random draws per deletion read by the primary interpreter
(``residual_disposition.follow_partial_draws``), by ``scripts/follow_rung2.py``'s pins, send and
judge history, on ``follow/partial_draws``.

    uv run python scripts/follow_partial_draws.py [--pin | --send draws | --judge-history RUN]
    uv run python scripts/follow_partial_draws.py --masks [--write]
    uv run python scripts/follow_partial_draws.py --draws-result RUN

Without a flag it prints the plan and the due requests with their estimate. ``--masks`` prints
each line's draws before any request (pieces deleted; the families' rank distances, widest window
and choices with a single candidate); with ``--write`` it writes them once
(``manifests/follow_partial_draws_masks.json``).
``--pin`` writes each reading's package directory, evidence sha256 and keys once
(``manifests/follow_partial_draws_inputs.json``); ``--send draws`` sends only with the pins
committed and unchanged, in ten waves by draw number (``send_by_draw``), its runs
``follow-partial-draws-<time>`` under ``runs/follow_partial_draws`` with the counts
(``draws.json``). ``--draws-result`` writes a finished
send's summary (``summary.json``, ``follow_partial_draws.summary``), descriptive. Every plan is
refused unless each draw deletes as many pieces as its target deletion.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import cache
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_lens_mask as LM
from residual_disposition import follow_partial as FP
from residual_disposition import follow_partial_draws as PD
from residual_disposition import follow_qualification as Q
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT, runs_root

PINS = "manifests/follow_partial_draws_inputs.json"
MASKS = "manifests/follow_partial_draws_masks.json"
HISTORY = "manifests/stage3_judge_history_follow_partial_draws.json"
ROOT = "follow_partial_draws"
RUN = "follow-partial"  # its sends' runs: follow-partial-draws-<time>
PHASE = "draws"
RESULT = "summary.json"
LABEL = "follow/partial-draws"
SEND = F.send  # rung 2's send, which each wave goes through


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every draw with its package directory, evidence's sha256 and attempts, carrying what it
    deletes (``removed`` for a steered draw, ``edit`` for a family draw); refused unless each draw
    deletes as many pieces as its target deletion."""
    from residual_disposition import lens_render as R

    samples = FP.steering(find_run)
    pinned = LM.pins()
    runs = runs_root()

    @cache
    def rendered(line: str) -> tuple[Any, str]:
        """A family line's L package, checked at its source pin, and its rendering."""
        r = next(r for r in PD.readings()[PHASE] if r["line"] == line)
        directory = LM.source_dir(r, pinned, runs)
        return directory, R.render(directory, r["decoder"], runs)[0]

    out = []
    for r in PD.readings()[PHASE]:
        if r["kind"] == "steered":
            # L through the base decoder: its package, never a base_random branch
            directory = FP.package_dir(r, find_run, lambda: {})
            text, gone = PD.steered_evidence(r, directory, samples)
            r = {**r, "removed": gone}
        else:
            directory, source = rendered(r["line"])
            text, edit = PD.family_evidence(r, source)
            r = {**r, "edit": edit}
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    found = PD.plan_problems([c["reading"] for c in out], PD.targets())
    if found:
        raise SystemExit("the draws are not the design: " + "; ".join(found[:5]))
    return {PHASE: out}


def masks(write: bool) -> int:
    """Each line's draws, before any request; with ``write`` recorded once."""
    cells = plan()[PHASE]
    record = PD.masks(cells, PD.targets())
    for line, e in record["lines"].items():
        for i, d in e["draws"].items():
            if e["kind"] == "steered":
                extra = f"positions {d['positions']} chars {d['chars']}"
            else:
                forced = sum(x["candidates"] == 1 for x in d["drawn"])
                extra = (f"distances {d['distances']} widest window {d['widest_window']} "
                         f"single-candidate choices {forced}")  # fmt: skip
            print(f"{line} draw {i}: removes {d['removed']} (target {e['target']['removed']}) "
                  f"{extra}")  # fmt: skip
        print(f"{line} freedom: {e['freedom']}")
    if not write:
        return 0
    path = REPO_ROOT / MASKS
    if path.exists():
        raise SystemExit(f"{MASKS} exists: the masks are written once")
    note = "written by follow_partial_draws.masks from the planned readings, before any request"
    F.S.write(path, {**record, "note": note})
    print(f"wrote {MASKS} ({hashlib.sha256(path.read_bytes()).hexdigest()}): commit it")
    return 0


class Rung:
    """``scripts/follow_rung2.py``'s interface, descriptive (no qualification, no decision)."""

    BEHAVIOR = None
    BEHAVIORS = PD.BEHAVIORS
    item_of = staticmethod(PD.item_of)


def result(name: str) -> int:
    """A finished send's summary, written once, its readings the pinned ones at the run's
    evidence, every slot from the caches as the send's histories record it."""
    from residual_disposition import continuation_rules as cr
    from residual_disposition.manifests import committed_json

    run = runs_root() / ROOT / name
    out = run / RESULT
    if not name.startswith(f"{RUN}-{PHASE}-") or not (run / f"{PHASE}.json").is_file():
        raise SystemExit(f"{name}: not a finished send of the draws")
    if out.exists():
        raise SystemExit(f"{out}: the summary is written once")
    cells = plan()[PHASE]
    found = Q.pin_problems(committed_json(REPO_ROOT / PINS, "the draws' pins")["readings"],
                           Q.pins(cells, runs_root()))  # fmt: skip
    if found:
        raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    inputs = json.loads((run / "inputs.json").read_text())["readings"]
    if {r["salt"]: r["evidence_sha256"] for r in inputs} != {
            c["reading"]["salt"]: c["evidence_sha256"] for c in cells}:  # fmt: skip
        raise SystemExit(f"{name}: not the draws' readings at their evidence")
    ctx = F.context()
    F.S.reconcile(runs_root() / ROOT, ctx["caches"])
    rows = F.state(cells, ctx["caches"], ctx)
    if F.RS.histories(rows) != json.loads((run / "histories.json").read_text()):
        raise SystemExit(f"{name}: the caches no longer give the send's histories")
    rules = cr.follow_rules().data["readers"]
    F.S.write(out, PD.summary(rows, rules, ctx["definitions"]))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


def send_by_draw(cells: list[dict[str, Any]], run: Any, ctx: dict[str, Any]) -> dict[str, float]:
    """The draws in waves by draw number: every line's first draw, then every line's second, and
    so on, each wave through rung 2's send in an API session of its own. Each wave's estimate is
    checked against the item's API left, and a stop at the ceiling leaves every line with as many
    draws done."""
    charged: dict[str, float] = {}
    for i in sorted({c["reading"]["draw"] for c in cells}):
        wave = [c for c in cells if c["reading"]["draw"] == i]
        for item, usd in SEND(wave, run, ctx).items():
            charged[item] = charged.get(item, 0.0) + usd
    return charged


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    F.R2 = Rung
    F.LABEL, F.ROOT = LABEL, ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = (PHASE,)
    F.plan = plan
    F.send = send_by_draw
    if args in (["--masks"], ["--masks", "--write"]):
        return masks("--write" in args)
    if len(args) == 2 and args[0] == "--draws-result":
        return result(args[1])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
