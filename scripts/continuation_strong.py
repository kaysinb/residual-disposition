"""The strong branch's manifests (preregistration §12.4.10; the amendment of 2026-10-21; C16-2):
a branch's screening record (phase strong_screen, `strong_screening.json`), recomputed from the
runs it names, becomes `manifests/strong_<lineage>.json`, committed before any validation cell;
its validation record (phase strong_validate, `strong_validation.json`) becomes
`manifests/strong_validation_<lineage>.json`, committed before any follow-up. `gate` prints the
DPO fallback's gate from the committed manifests (§12.8: `fallback_gate_closed`).

    uv run python scripts/continuation_strong.py screening runs/continuation_14b/<screening run>
    uv run python scripts/continuation_strong.py validation runs/continuation_14b/<validation run>
    uv run python scripts/continuation_strong.py gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_strong as CS
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN

STEPS = {"screening": "strong_screening", "validation": "strong_validation"}


def record_of(run: Path, step: str) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{run}: not a completed run ({meta.get('status')})")
    found = (yaml.safe_load((run / "config.yaml").read_text()) or {}).get("step")
    if found != step:
        raise SystemExit(f"{run}: a {found} cell, not {step}")
    return json.loads((run / f"{step}.json").read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in STEPS:
        p = sub.add_parser(name)
        p.add_argument("run", type=Path)
    sub.add_parser("gate")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="where manifests/ is")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.command == "gate":
        detector = CS.detector_of(CS.branch(CS.CONTINUED)["behavior"])
        print(json.dumps(CS.fallback_gate(args.root, rules, find_run, detector)))
        return 0
    record = record_of(args.run, STEPS[args.command])
    detector = CS.detector_of(record["behavior"])  # the branch's behavior (a panel behavior's)
    key = f"{args.command}_run"
    try:
        if args.command == "screening":
            out = CS.write_screening(args.root, {**record, key: args.run.name}, find_run, detector)
            path = CS.screening_path(args.root, out["lineage"])
            said = f"{out['outcome']}, chosen {out['chosen']}"
        else:
            out = CS.write_validation(args.root, {**record, key: args.run.name}, find_run, detector)
            path = CS.validation_path(args.root, out["lineage"])
            said = f"passed {out['verdict']['passed']}"
            if out["verdict"]["passed"] and out["branch"] == CS.CONTINUED:
                said += "; once committed, drop strong/stage5_dpo fallback_gate_closed"
    except ValueError as error:
        raise SystemExit(f"{args.run}: {error}") from error
    print(f"{path.relative_to(args.root)}: {said}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
