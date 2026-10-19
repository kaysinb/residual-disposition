"""Write a pilot behavior's Stage-2 decision as its manifest (preregistration §12.7; review s3t):
the record cell of phase decide (`stage2_decision.json`), its decision recomputed from the runs it
names, becomes `manifests/stage2_<behavior>.json`. Commit it before the lineage's k* calibration;
with `--slot` the script prints the Stage-5 slot from the committed decisions.

    uv run python scripts/continuation_stage2.py runs/continuation_14b/<stage2_decision run>
    uv run python scripts/continuation_stage2.py --slot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, nargs="?", help="the completed stage2_decision run")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="where manifests/ is")
    parser.add_argument("--slot", action="store_true", help="print the Stage-5 slot")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.slot:
        print(json.dumps(S2.slot(args.root, rules, find_run, args.root)))
        return 0
    if args.run is None:
        raise SystemExit("name the stage2_decision run, or --slot")
    meta = json.loads((args.run / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{args.run}: not a completed run ({meta.get('status')})")
    record = json.loads((args.run / "stage2_decision.json").read_text())
    path = S2.manifest_path(args.root, record["behavior"])
    try:
        S2.write_manifest(path, {**record, "decision_run": args.run.name}, rules, find_run)
    except ValueError as error:
        raise SystemExit(f"{args.run}: {error}") from error
    d = record["decision"]
    print(f"{path.relative_to(args.root)}: analysis dose {d['analysis_dose']}, exception a "
          f"{d['exception_a']}. Commit it.")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
