"""D*'s descriptive arms on the base decoder (``residual_disposition.follow_lens_hint``): the hint,
the fragment-family deletion with and without it and their matched controls, by
``scripts/follow_rung2.py``'s pins and send, on ``follow/lens_hint``, every parsed hypothesis's
category graded too (``follow_lens_hint.with_categories``).

    uv run python scripts/follow_lens_hint.py [--pin | --send lens_hint | --judge-history RUN]
    uv run python scripts/follow_lens_hint.py --lens-hint-result RUN
    uv run python scripts/follow_lens_hint.py --stack400 [--rung1 <400's readings.json>] \
        [--pin | --send lens_hint | --judge-history RUN | --lens-hint-result RUN]

With ``--stack400`` the stack at 400's hint (§13.16, ``follow_lens_hint.stack400``): five cells
from rung 1's pins at 400, on ``follow/stack400/lens_hint``, with its own pins, history and runs
(``follow-stack400-hint-lens_hint-<time>`` under ``runs/follow_stack400_hint``); the summary needs
``--rung1``, 400's rung-1 record under the runs root, complete and at rung 1's pins at 400 with
their evidence (``follow_lens_hint.rung1_problems``; its sha256 kept in the summary), and takes
eli5's A and H from the committed hint summary. Planning and sending need no ``--rung1``.

Without a flag it prints the plan and the due requests with their estimate, and checks the
readings against their committed pins when they exist. ``--pin`` writes each reading's package
directory, evidence sha256 and keys once (``manifests/follow_lens_hint_inputs.json``), sending
nothing; ``--send lens_hint`` sends only with the pins committed and unchanged, its runs
``follow-lens-hint-lens_hint-<time>`` under ``runs/follow_lens_hint`` with the counts
(``lens_hint.json``). ``--lens-hint-result`` writes a finished send's summary (``summary.json``,
``follow_lens_hint.summary``), descriptive: the readings at their pins and the send's histories,
each cell beside its rung-1 baseline with its availability, the stage-1 rule shown and not
applied, the deletion comparisons and the frozen criteria.
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
    """Every reading with its package directory, evidence's sha256 and attempts, carrying its
    edit's record (``edit``); refused unless the edits are the frozen design."""
    from residual_disposition import follow_partial as FP

    pinned = LH.pins()
    partial = cache(lambda: list(FP.base_random_dirs(find_run).values()))
    out = []
    for r in LH.readings()["lens_hint"]:
        directory = LH.source_dir(r, pinned, runs_root(), partial)
        text, edit = LH.evidence(r, directory, runs_root())
        r = {**r, "edit": edit}
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    found = LH.plan_problems([c["reading"] for c in out])
    if found:
        raise SystemExit("the edits are not the frozen design: " + "; ".join(found))
    return {"lens_hint": out}


ORIGINAL_STATE = F.state


def state(cells: list[dict[str, Any]], caches: dict[str, Any], ctx: dict[str, Any]) -> list[dict]:
    """``scripts/follow_rung2.py``'s state, every parsed hypothesis's category graded too."""
    return LH.with_categories(ORIGINAL_STATE(cells, caches, ctx), ctx["definitions"], ctx["spec"],
                              caches["judge"])  # fmt: skip


class Rung:
    """``scripts/follow_rung2.py``'s interface: the arms, descriptive (no qualification and no
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
    if LH.STACK400:  # §13.16: never a summary of 400's hint without its rung-1 baseline
        found = LH.rung1_problems()
        if found:
            raise SystemExit("; ".join(found))
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
    rows = state(cells, ctx["caches"], ctx)
    if F.RS.histories(rows) != json.loads((run / "histories.json").read_text()):
        raise SystemExit(f"{name}: the caches no longer give the send's histories")
    rules = cr.follow_rules().data["readers"]
    F.S.write(out, LH.summary(rows, fr.load(), rules, ctx["definitions"]))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


STACK400 = {"PINS": "manifests/follow_stack400_hint_inputs.json",
            "HISTORY": "manifests/stage3_judge_history_follow_stack400_hint.json",
            "ROOT": "follow_stack400_hint", "RUN": "follow-stack400-hint",
            "LABEL": "follow/stack400-hint"}  # fmt: skip
LABEL = "follow/lens-hint"


def at_stack400(rung1: str | None) -> None:
    """Point this script and ``follow_lens_hint`` at the stack at 400's hint (§13.16): its five
    cells, item, pins, history and runs; ``rung1`` 400's rung-1 record for the summary."""
    global PINS, HISTORY, ROOT, RUN, LABEL
    for name, value in LH.stack400(rung1).items():
        setattr(LH, name, value)
    PINS, HISTORY, ROOT = STACK400["PINS"], STACK400["HISTORY"], STACK400["ROOT"]
    RUN, LABEL = STACK400["RUN"], STACK400["LABEL"]


def main() -> int:
    args = sys.argv[1:]
    if "--stack400" in args:
        args.remove("--stack400")
        rung1 = None
        if "--rung1" in args:
            i = args.index("--rung1")
            rung1 = args[i + 1]
            del args[i : i + 2]
        at_stack400(rung1)
    F.R2 = Rung
    F.LABEL, F.ROOT = LABEL, ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = ("lens_hint",)
    F.plan, F.state = plan, state
    if len(args) == 2 and args[0] == "--lens-hint-result":
        return result(args[1])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
