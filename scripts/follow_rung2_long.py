"""Rung 2 with long steered excerpts (``residual_disposition.follow_rung2_long``), by
``scripts/follow_rung2.py``'s pins and send, on ``follow/rung2_long``, in batches.

    uv run python scripts/follow_rung2_long.py [--pin | --send BATCH | --judge-history RUN]
    uv run python scripts/follow_rung2_long.py --sizes
    uv run python scripts/follow_rung2_long.py --long-result RUN [RUN ...]

``BATCH`` is ``stack400`` (with A and H), ``stack800``, ``continued200`` or ``e1``. ``--pin``
writes every batch's readings once (``manifests/follow_rung2_long_inputs.json``), sending nothing;
``--send BATCH`` sends one batch with the pins committed and unchanged, its runs
``follow-rung2-long-<batch>-<time>`` under ``runs/follow_rung2_long``. ``--sizes`` prints each
reading's evidence length before any request. ``--long-result`` writes the finished sends'
summary (``summary.json`` in the last run), exploratory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_qualification as Q
from residual_disposition import follow_rung2_long as RL
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import runs_root

PINS = "manifests/follow_rung2_long_inputs.json"
HISTORY = "manifests/stage3_judge_history_follow_rung2_long.json"
ROOT = "follow_rung2_long"
RUN = "follow-rung2-long"
LABEL = "follow/rung2-long"
SOURCE_RECORDS = {  # each state's committed rung-2 analysis record, for the 128-character counts
    "stack400": "follow_stack400_rung2/follow-rung2-analysis-20261109T154751Z/rung2.json",
    "stack800": "follow_stack_rung2/follow-rung2-analysis-20261104T082737Z/rung2.json",
    "continued200": "follow_rung2/follow-rung2-analysis-20261102T115528Z/rung2.json",
}


def plan() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for batch in RL.BATCHES:
        out[batch] = []
        for r in RL.readings(batch):
            directory = RL.source_dir(r, runs_root(), find_run)
            text = RL.evidence(r, directory, runs_root(), find_run)
            out[batch].append({"reading": r, "directory": str(directory), "reused": None,
                               "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def sizes() -> int:
    for batch, rows in plan().items():
        for c in rows:
            r = c["reading"]
            text = RL.evidence(r, __import__("pathlib").Path(c["directory"]), runs_root(),
                               find_run)  # fmt: skip
            print(f"{batch:13} {r['salt']}: {len(text)} chars")
    return 0


class Rung:
    BEHAVIOR = None
    BEHAVIORS = (RL.BEHAVIOR,)
    item_of = staticmethod(RL.item_of)


def source_counts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in SOURCE_RECORDS.values():
        for r in json.loads((runs_root() / path).read_text())["readings"]:
            out[r["salt"]] = r["count"]
    return out


def result(names: list[str]) -> int:
    from residual_disposition import continuation_rules as cr
    from residual_disposition import follow_regions as fr

    ctx = F.context()
    F.S.reconcile(runs_root() / ROOT, ctx["caches"])
    cells = [c for rows in plan().values() for c in rows]
    done = {}
    for name in names:
        run = runs_root() / ROOT / name
        for r in json.loads((run / "inputs.json").read_text())["readings"]:
            done[r["salt"]] = r["evidence_sha256"]
    cells = [c for c in cells if c["reading"]["salt"] in done]
    if any(done[c["reading"]["salt"]] != c["evidence_sha256"] for c in cells):
        raise SystemExit("a run's readings are not at their evidence")
    rows = F.state(cells, ctx["caches"], ctx)
    rules = cr.follow_rules().data["readers"]
    out = runs_root() / ROOT / names[-1] / "summary.json"
    if out.exists():
        raise SystemExit(f"{out}: the summary is written once")
    F.S.write(out, RL.summary(rows, fr.load(), rules, ctx["definitions"], source_counts()))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


def main() -> int:
    args = sys.argv[1:]
    F.R2 = Rung
    F.LABEL, F.ROOT = LABEL, ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = tuple(RL.BATCHES)
    F.plan = plan
    if args == ["--sizes"]:
        return sizes()
    if args and args[0] == "--long-result":
        return result(args[1:])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
