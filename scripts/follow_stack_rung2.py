"""Rung 2 on eli5's validated stack candidate 800, on both decoders (preregistration §13.6, §13.7,
§13.16): ``follow_rung2_stack``'s readings by ``scripts/follow_rung2.py``'s pins and send, on
``follow/stack/rung2``, from the build session's and the steering session's committed records.

    uv run python scripts/follow_stack_rung2.py --record-steering RUN [--write]
    uv run python scripts/follow_stack_rung2.py [--pin | --send qualification |
                                                 --send analysis --qualification RUN |
                                                 --judge-history RUN]

The rung's format is the continued mode's validation, read as it stands: no supplement here.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_qualification as Q
from residual_disposition import follow_rung2_stack as RS
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import runs_root


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading by phase, with its build directory, evidence's sha256 and attempts."""
    record = RS.built()
    samples = RS.steering(find_run)
    builds = runs_root() / "follow_stack" / record["run"] / "out" / "build" / "builds"
    out: dict[str, list[dict[str, Any]]] = {}
    for phase, rows in RS.readings(record).items():
        out[phase] = []
        for r in rows:
            directory = builds / r["package"]["state"]
            text = RS.evidence(r, directory, samples)
            out[phase].append({
                "reading": r, "directory": str(directory), "reused": None,
                "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def main() -> int:
    if "supplement" in sys.argv:
        raise SystemExit("a rung's format is validated once, in the first mode to reach it")
    F.R2 = RS
    F.LABEL, F.ROOT = "follow/stack-rung2", "follow_stack_rung2"
    F.PINS = "manifests/follow_stack_rung2_inputs.json"
    F.HISTORY = "manifests/stage3_judge_history_follow_stack_rung2.json"
    F.plan = plan
    return F.main()


if __name__ == "__main__":
    sys.exit(main())
