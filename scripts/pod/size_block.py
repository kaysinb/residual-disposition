"""One phase of one new implantation of the size block on its Pod (docs/preregistration.md §13.9,
amendment of 2026-11-05; ``residual_disposition.size_block_run``).

    python scripts/pod/size_block.py --implantation 4B-1 --phase inputs
    ... --phase preflight | implant | suppress | detector-items --dose D
    ... --phase confirm --dose D --attempt N
    ... --phase calibrate --dose D
    ... --phase dense-check --dose D   (only when the calibration set the dense schedule)
    ... --phase blocks --dose D --purpose pilot|certificate

Each phase writes its record to ``$RUNS_DIR/size_<name>/records/<phase>[-…].json`` and reads the
records it starts from there; its steps are cells, so a rerun resumes. A phase refuses to start
when what it needs is missing or failed: no training after failed floors, no calibration
without a passed confirmation at the dose, no certificate runs after an unresolved calibration.
Set PYTHONDONTWRITEBYTECODE=1 and WANDB_MODE=disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import size_block as S
from residual_disposition import size_block_run as R
from residual_disposition.paths import runs_root

PHASES = ("inputs", "preflight", "implant", "suppress", "detector-items", "confirm", "calibrate",
          "dense-check", "blocks")  # fmt: skip


def records_dir(name: str) -> Path:
    d = runs_root() / f"size_{R.CONFIGS[name]}" / "records"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read(name: str, key: str) -> dict:
    path = records_dir(name) / f"{key}.json"
    if not path.exists():
        raise SystemExit(f"{name}: no {key} record yet ({path})")
    return json.loads(path.read_text())


def write(name: str, key: str, record: dict) -> None:
    path = records_dir(name) / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    print(f"wrote {path}")


def confirmed(name: str, dose: int) -> dict:
    """The passed confirmation at ``dose``, or refusal."""
    draws = sorted(records_dir(name).glob("confirm-*.json"))
    passed = [json.loads(p.read_text()) for p in draws]
    passed = [d for d in passed if d["passed"] and int(d["dose"]) == dose]
    if not passed:
        raise SystemExit(f"{name}: no passed confirmation at {dose}")
    return passed[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--implantation", required=True, choices=sorted(R.CONFIGS))
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--dose", type=int)
    ap.add_argument("--attempt", type=int)
    ap.add_argument("--purpose", choices=("pilot", "certificate"))
    args = ap.parse_args(argv)
    name = args.implantation
    rules = S.load_size_rules()
    inputs = R.load_inputs()
    ws = R.workspace(name, rules)
    data = R.datasets(ws, inputs)  # every phase starts from the checked pinned data
    if args.phase == "inputs":
        write(name, "inputs", R.inputs_record(ws, inputs, data))
        return 0
    if args.phase == "preflight":
        write(name, "preflight", R.preflight(ws, data))
        return 0
    if args.phase == "implant":
        rec = R.implant(ws, rules, inputs, data, name)
        write(name, "implant", rec)
        if rec["problems"]:
            print(f"{name}: infeasible by the floors: {rec['problems']}")
        return 0
    imp = read(name, "implant")
    if imp["problems"]:
        raise SystemExit(f"{name}: the floors failed ({imp['problems']}); nothing follows")
    if args.phase == "suppress":
        write(name, "suppress", R.suppress(ws, rules, inputs, data, imp["states"], name))
        return 0
    sup = read(name, "suppress")
    if args.dose is None:
        raise SystemExit("--dose is needed from here on")
    if args.phase == "detector-items":
        items = R.detector_items_L(ws, sup, args.dose, name, rules)
        write(name, f"detector-items-L-{args.dose}", {"implantation": name, "items": items})
        return 0
    if args.phase == "confirm":
        if args.attempt is None:
            raise SystemExit("--attempt is needed")
        previous = [json.loads(p.read_text())
                    for p in sorted(records_dir(name).glob("confirm-*.json"))]  # fmt: skip
        if any(int(d["attempt"]) == args.attempt for d in previous):
            raise SystemExit(f"attempt {args.attempt} is recorded already")
        rec = R.confirm(ws, rules, sup, name, dose=args.dose, attempt=args.attempt,
                        previous=previous)  # fmt: skip
        write(name, f"confirm-{args.attempt}", rec)
        return 0
    conf = confirmed(name, args.dose)
    states = {"L": sup["doses"][str(args.dose)]["L"], "H_s": sup["doses"][str(args.dose)]["H_s"]}
    arms_ = R.relearn_inputs(ws, inputs, data, imp)
    if args.phase == "calibrate":
        rec = R.calibrate(rules, arms_, states["H_s"], name)
        write(name, "calibrate", {**rec, "dose": args.dose, "confirmation": conf})
        return 0
    cal = read(name, "calibrate")
    if int(cal["dose"]) != args.dose or cal["sham_s"] != states["H_s"]:
        raise SystemExit("the calibration is of another dose")
    if args.phase == "dense-check":
        if not cal.get("dense"):
            raise SystemExit("the calibration did not set the dense schedule")
        write(name, "dense-check", R.dense_check(ws, rules, arms_["b"], states["H_s"], name))
        return 0
    if cal.get("dense") and not read(name, "dense-check").get("passed"):
        raise SystemExit("the dense schedule's check has not passed: no runs on it")
    if args.phase == "blocks":
        if args.purpose is None:
            raise SystemExit("--purpose pilot|certificate")
        rec = R.blocks(rules, arms_, states, cal, name, args.purpose)
        write(name, f"blocks-{args.purpose}", {**rec, "dose": args.dose})
        return 0
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
