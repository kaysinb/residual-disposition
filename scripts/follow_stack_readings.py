"""Rung 1 on eli5's validated stack candidate 800 (preregistration §13.7, §13.16): the readings of
``follow_readings.stack_readings``, by ``scripts/follow_readings.py``'s pins and send, on
``follow/stack/full``, from the build session's committed record. With ``--dose 400``, the stack
at 400's (§13.16, a descriptive point of the depth curve): the base decoder alone and unfiltered
PD, on ``follow/stack400/full``, with its own build record, pins, history and runs (``STACK400``).

    uv run python scripts/follow_stack_readings.py [--dose 400] --record-build RUN   # written once
    uv run python scripts/follow_stack_readings.py [--dose 400] [--pin | --send |
                                                   --judge-history RUN]

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
DOSE, DECODERS, BUILDS = 800, D.STACK_DECODERS, "follow_stack"  # the build session's runs root
READINGS = {"ITEM": "follow/stack/full", "LABEL": "follow/stack-readings",
            "PINS": "manifests/follow_stack_readings_inputs.json",
            "HISTORY": "manifests/stage3_judge_history_follow_stack_readings.json",
            "ROOT": "follow_stack_readings"}  # fmt: skip
# The stack at 400 (§13.16): its L's untagged pool the screening's pinned one (B4)
STACK400 = {"RECORD": REPO_ROOT / "manifests" / "follow_stack400_build.json",
            "L_POOL": "20261022-110641-tool_pd-e0d05937-0a98", "DOSE": 400,
            "DECODERS": ("base",), "BUILDS": "follow_stack400",
            "READINGS": {"ITEM": "follow/stack400/full", "LABEL": "follow/stack400-readings",
                         "PINS": "manifests/follow_stack400_readings_inputs.json",
                         "HISTORY": "manifests/stage3_judge_history_follow_stack400_readings.json",
                         "ROOT": "follow_stack400_readings"}}  # fmt: skip


def record_build(run: str) -> int:
    """The build session's record: its states and pools, from what it wrote, written once."""
    if RECORD.exists():
        raise SystemExit(f"{RECORD.name} exists: written once")
    out = runs_root() / BUILDS / run / "out"
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
    for r in D.stack_readings(built["states"], DOSE, DECODERS):
        if r["reader"] == "pd_unfiltered":
            directory = find_run(built["pools"][r["role"]])
        else:
            directory = (runs_root() / BUILDS / built["run"] / "out" / "build" / "builds"
                         / built["states"][r["role"]])  # fmt: skip
        text = evidence(r, Path(directory))
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": attempts(r, text)})  # fmt: skip
    return out


def floors_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The candidate's floors, from the qualification's A, H and fidelity branches: D*'s eligibility
    from A, H and fidelity, the recipe control's floor reported apart (§13.16), None while a recipe
    stream is pending; unfiltered PD's rule as ``follow_qualification.floors``."""
    saved = json.loads((runs_root() / "follow_qualification" / FR.QUALIFICATION
                        / "qualification.json").read_text())["rates"]  # fmt: skip
    rates = {}
    for key, value in saved.items():
        reader, behavior, role, decoder = key.split("/")
        if behavior == "eli5" and not role.startswith("recipe"):
            rates[(reader, behavior, role, None if decoder == "None" else decoder)] = (
                None if value is None else Fraction(value))  # fmt: skip
    rates.update(Q.pooled([r for r in rows if r["reading"]["role"] == "recipe"]))
    rules = cr.follow_rules().data
    out = Q.floors(rates, rules)["eli5"]
    f = {k: Fraction(v) for k, v in rules["readers"]["qualification"]["floors"].items()}
    dstar, recipe_met = {}, {}
    for decoder in DECODERS:  # §13.16: an unrun rung's recipe floor closes no contrast
        a, h = rates[("dstar", "eli5", "A", decoder)], rates[("dstar", "eli5", "H", decoder)]
        r = rates[("dstar", "eli5", "recipe", decoder)]
        dstar[decoder] = a >= f["active_min"] and h <= f["sham_max"] and a - h >= f["gap_min"]
        recipe_met[decoder] = None if r is None else r >= f["recipe_ratio"] * a
    fidelity = out["dstar"]["fidelity"]
    eligible = None  # the state decoder's: not read at 400
    if "state" in dstar:
        eligible = None if fidelity is None else dstar["state"] and fidelity
    out["dstar"] = {**dstar, "fidelity": fidelity, "recipe_floor_met": recipe_met,
                    "eligible": eligible, "base_qualification": dstar["base"]}  # fmt: skip
    return {"eli5": out}


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
                     fidelity_known={"eli5": original["base_fidelity"]["eli5"]},
                     decoders=DECODERS)  # fmt: skip
    line = out["lines"]["eli5"]
    met = floors["eli5"]["dstar"]["recipe_floor_met"]
    out["pending"] = out["pending"] or None in met.values()
    rung1 = {}
    for ladder, family, cell in (("state", "dstar_state", "random"),
                                 ("base", "dstar_base", "base_random")):  # fmt: skip
        if ladder not in DECODERS:
            continue
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


def at_dose(dose: int) -> None:
    """Point this script at the stack's dose: 800's constants as they are, or ``STACK400``'s."""
    global RECORD, L_POOL, DOSE, DECODERS, BUILDS, READINGS
    if dose == 400:
        RECORD, L_POOL, DOSE = STACK400["RECORD"], STACK400["L_POOL"], STACK400["DOSE"]
        DECODERS, BUILDS, READINGS = STACK400["DECODERS"], STACK400["BUILDS"], STACK400["READINGS"]
    elif dose != 800:
        raise SystemExit(f"--dose {dose}: the stack's readings are at 800 or 400")


def main() -> int:
    if "--dose" in sys.argv:
        i = sys.argv.index("--dose")
        at_dose(int(sys.argv[i + 1]))
        del sys.argv[i : i + 2]
    if len(sys.argv) == 3 and sys.argv[1] == "--record-build":
        return record_build(sys.argv[2])
    if "--sensitivity" in sys.argv:
        raise SystemExit("--sensitivity is the eli5 stack 200 repeat's, not this reading's")
    FR.ROOT_READINGS = FR.ROOT
    FR.ITEM, FR.LABEL = READINGS["ITEM"], READINGS["LABEL"]
    FR.PINS, FR.HISTORY, FR.ROOT = READINGS["PINS"], READINGS["HISTORY"], READINGS["ROOT"]
    FR.plan, FR.record = plan, record
    return FR.main()


if __name__ == "__main__":
    sys.exit(main())
