"""Write exception a's continued-mode lineage's decision as its manifest (preregistration §12.7
Stage 5, §12.4.1; C12): the record cell of phase s5_decide (`stage5_decision.json`), its decision
recomputed from the runs it names, becomes `manifests/stage5_<behavior>.json`. Commit it before
the lineage's k* calibration.

    uv run python scripts/continuation_stage5.py runs/continuation_14b/<stage5_decision run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage5 as S5
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="the completed stage5_decision run")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="where manifests/ is")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    meta = json.loads((args.run / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{args.run}: not a completed run ({meta.get('status')})")
    record = json.loads((args.run / "stage5_decision.json").read_text())
    path = S5.manifest_path(args.root, record["behavior"])
    try:
        S5.write_manifest(path, {**record, "decision_run": args.run.name}, rules, find_run)
    except ValueError as error:
        raise SystemExit(f"{args.run}: {error}") from error
    d = record["decision"]
    print(f"{path.relative_to(args.root)}: {d['outcome']}, analysis dose {d['analysis_dose']}. "
          "Commit it.")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
