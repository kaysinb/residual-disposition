"""Two descriptive arms of D* on the base decoder (``residual_disposition.follow_lens_hint``): the
hint and the mask, by ``scripts/follow_rung2.py``'s pins and send, on ``follow/lens_hint``.

    uv run python scripts/follow_lens_hint.py [--pin | --send lens_hint | --judge-history RUN]
    uv run python scripts/follow_lens_hint.py --lens-hint-result RUN

Without a flag it prints the plan and the due requests with their estimate, and checks the
readings against their committed pins when they exist. ``--pin`` writes each reading's package
directory, evidence sha256 and keys once (``manifests/follow_lens_hint_inputs.json``), sending
nothing; ``--send lens_hint`` sends only with the pins committed and unchanged, its runs
``follow-lens-hint-lens_hint-<time>`` under ``runs/follow_lens_hint`` with the counts
(``lens_hint.json``). ``--lens-hint-result`` writes a finished send's summary (``summary.json``,
``follow_lens_hint.summary``), descriptive: the readings at their pins and the send's histories,
each cell beside rung 1's count on its package, the stage-1 rule shown and not applied.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import cache
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_lens_hint as LH
from residual_disposition import follow_qualification as Q
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT, runs_root

PINS = "manifests/follow_lens_hint_inputs.json"
HISTORY = "manifests/stage3_judge_history_follow_lens_hint.json"
ROOT = "follow_lens_hint"
RUN = "follow-lens-hint"  # its sends' runs: follow-lens-hint-lens_hint-<time>
RESULT = "summary.json"


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading with its package directory, evidence's sha256 and attempts; a mask reading
    carrying its removed tokens (``masked``)."""
    from residual_disposition import follow_partial as FP

    pinned = LH.pins()
    partial = cache(lambda: list(FP.base_random_dirs(find_run).values()))
    out = []
    for r in LH.readings()["lens_hint"]:
        directory = LH.source_dir(r, pinned, runs_root(), partial)
        text, removed = LH.evidence(r, directory, runs_root())
        if removed is not None:
            r = {**r, "masked": removed}
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    return {"lens_hint": out}


class Rung:
    """``scripts/follow_rung2.py``'s interface: the two arms, descriptive (no qualification and no
    decision)."""

    BEHAVIOR = None
    BEHAVIORS = LH.BEHAVIORS
    item_of = staticmethod(LH.item_of)


def result(name: str) -> int:
    """A finished send's summary, written once: its readings the pinned ones at the run's
    evidence, every slot from the caches as the send's histories record it."""
    from residual_disposition import continuation_rules as cr
    from residual_disposition import follow_regions as fr
    from residual_disposition.manifests import committed_json

    run = runs_root() / ROOT / name
    out = run / RESULT
    if not name.startswith(f"{RUN}-lens_hint-") or not (run / "lens_hint.json").is_file():
        raise SystemExit(f"{name}: not a finished send of the two arms")
    if out.exists():
        raise SystemExit(f"{out}: the summary is written once")
    cells = plan()["lens_hint"]
    found = Q.pin_problems(committed_json(REPO_ROOT / PINS, "the arms' pins")["readings"],
                           Q.pins(cells, runs_root()))  # fmt: skip
    if found:
        raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    inputs = json.loads((run / "inputs.json").read_text())["readings"]
    if {r["salt"]: r["evidence_sha256"] for r in inputs} != {
            c["reading"]["salt"]: c["evidence_sha256"] for c in cells}:  # fmt: skip
        raise SystemExit(f"{name}: not the arms' readings at their evidence")
    ctx = F.context()
    F.S.reconcile(runs_root() / ROOT, ctx["caches"])
    rows = F.state(cells, ctx["caches"], ctx)
    if F.RS.histories(rows) != json.loads((run / "histories.json").read_text()):
        raise SystemExit(f"{name}: the caches no longer give the send's histories")
    rules = cr.follow_rules().data["readers"]
    F.S.write(out, LH.summary(rows, fr.load(), rules, ctx["definitions"]))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


def main() -> int:
    args = sys.argv[1:]
    F.R2 = Rung
    F.LABEL, F.ROOT = "follow/lens-hint", ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = ("lens_hint",)
    F.plan = plan
    if len(args) == 2 and args[0] == "--lens-hint-result":
        return result(args[1])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
