"""Rung 1 on eli5's validated stack candidate 800 (preregistration §13.7, §13.16): the readings of
``follow_readings.stack_readings``, by ``scripts/follow_readings.py``'s pins and send, on
``follow/stack/full``, from the build session's committed record.

    uv run python scripts/follow_stack_readings.py --record-build RUN   # the record, written once
    uv run python scripts/follow_stack_readings.py [--pin | --send | --judge-history RUN]

The recipe control is qualified from its own readings beside the qualification's A and H; the base
decoder's fidelity on A is the readings' record's. Each decoder's test is at 1/60, with its random
cell at most 8 of 80 and every gate of §13.5, as rung 1 on eli5 continued 200.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import follow_readings as FR

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_readings as D
from residual_disposition.continuation_kstar import find_run
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

RECORD = REPO_ROOT / "manifests" / "follow_stack_build.json"
L_POOL = "20261030-220251-tool_pd-f23763ec-693c"  # the committed screening's pool of L
ORIGINAL = "follow-readings-20261031T201407Z"  # the base decoder's fidelity on A


def record_build(run: str) -> int:
    """The build session's record: its states and pools, from what it wrote, written once."""
    if RECORD.exists():
        raise SystemExit(f"{RECORD.name} exists: written once")
    out = runs_root() / "follow_stack" / run / "out"
    built, pools = (json.loads((out / x / "result.json").read_text()) for x in ("build", "pools"))
    if built["status"] != "complete" or pools["status"] != "complete":
        raise SystemExit("the session's results are not complete, at whatever exit code")
    states = {json.loads((out / "build" / "builds" / s / "build.json").read_text())["role"]: s
              for s in built["builds"]}  # fmt: skip
    pool = {v["role"]: v["run"] for v in pools["pools"].values()}
    digests = {x: hashlib.sha256((out / x / "result.json").read_bytes()).hexdigest()
               for x in ("build", "pools")}  # fmt: skip
    RECORD.write_text(json.dumps({"run": run, "states": states, "pools": {**pool, "L": L_POOL},
                                  "sha256": digests}, indent=1, sort_keys=True) + "\n")  # fmt: skip
    print(f"wrote {RECORD.name}: {states}")
    return 0


def plan(sensitivity: bool = False) -> list[dict[str, Any]]:
    from residual_disposition.follow_qualification import attempts, evidence

    built = committed_json(RECORD, "the build session's record")
    out = []
    for r in D.stack_readings(built["states"]):
        if r["reader"] == "pd_unfiltered":
            directory = find_run(built["pools"][r["role"]])
        else:
            directory = (runs_root() / "follow_stack" / built["run"] / "out" / "build" / "builds"
                         / built["states"][r["role"]])  # fmt: skip
        text = evidence(r, Path(directory))
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": attempts(r, text)})  # fmt: skip
    return out


def floors_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The candidate's floors: its recipe control's pooled rates beside the qualification's A, H and
    fidelity branches; ``follow_qualification.floors`` under eli5's stack line's name."""
    saved = json.loads((runs_root() / "follow_qualification" / FR.QUALIFICATION
                        / "qualification.json").read_text())["rates"]  # fmt: skip
    rates = {}
    for key, value in saved.items():
        reader, behavior, role, decoder = key.split("/")
        if behavior == "eli5" and not role.startswith("recipe"):
            rates[(reader, behavior, role, None if decoder == "None" else decoder)] = (
                None if value is None else Fraction(value))  # fmt: skip
    rates.update(Q.pooled([r for r in rows if r["reading"]["role"] == "recipe"]))
    return Q.floors(rates, cr.follow_rules().data)


def record(rows: list[dict[str, Any]], cells: list[dict[str, Any]], definitions: Any) -> dict:
    from residual_disposition import follow_regions as fr

    accepted, validated = {}, {}
    for fmt in ("dstar", "pd_unfiltered"):
        d = data_root() / "follow_format" / fmt
        accepted[fmt] = bool(json.loads((d / "acceptance.json").read_text())["passed"])
        validated[fmt] = bool(json.loads((d / "graders.json").read_text())["specificity"]["passed"])
    original = json.loads((runs_root() / FR.ROOT_READINGS / ORIGINAL / "readings.json").read_text())
    floors = floors_of(rows)
    out = D.analysis(rows, floors, accepted, validated, fr.load(),
                     cr.follow_rules().data["readers"], definitions,
                     fidelity_known={"eli5": original["base_fidelity"]["eli5"]})  # fmt: skip
    line = out["lines"]["eli5"]
    rung1 = {}
    for ladder, family, cell in (("state", "dstar_state", "random"),
                                 ("base", "dstar_base", "base_random")):  # fmt: skip
        decided = line[family]["decisions"]
        if decided is None or line.get(cell) is None:
            rung1[ladder] = None
            continue
        rung1[ladder] = {"passed": bool(decided[str(D.RUNG_LEVEL)]) and line[cell] <= D.RANDOM_MAX,
                         "random": line[cell]}  # fmt: skip
    return {"readings": [{"salt": r["reading"]["salt"], "count": r["count"]} for r in rows],
            "evidence": {c["reading"]["salt"]: c["evidence_sha256"] for c in cells},
            "floors": floors["eli5"], "format": {"accepted": accepted,
                                                 "specificity_validated": validated},
            **out, "rung1": rung1}  # fmt: skip


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--record-build":
        return record_build(sys.argv[2])
    FR.ROOT_READINGS = FR.ROOT
    FR.ITEM, FR.LABEL = "follow/stack/full", "follow/stack-readings"
    FR.PINS = "manifests/follow_stack_readings_inputs.json"
    FR.HISTORY = "manifests/stage3_judge_history_follow_stack_readings.json"
    FR.ROOT = "follow_stack_readings"
    FR.plan, FR.record = plan, record
    return FR.main()


if __name__ == "__main__":
    sys.exit(main())
