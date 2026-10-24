"""Write a lineage's k* manifest from its completed calibration record (preregistration §12.3.7;
review s3j): the record cell of phase kstar (`kstar_calibration.json`), checked against the frozen
rules, becomes `manifests/kstar_<lineage>.json`: its status, medians, both k* and B are
recomputed from its calibration runs, and each named run is checked to be the completed
calibration run it records (review s3l). Commit the manifest before the lineage's first
certificate run; the certificate reads only a committed, unchanged manifest. An unresolved
calibration is written the same way: it starts no certificate and the panel's analysis reports
it (p₂ = 1).

    uv run python scripts/continuation_kstar.py runs/continuation_14b/<kstar_calibration run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_rules as cr
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="the completed kstar_calibration run")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="where manifests/ is")
    args = parser.parse_args(argv)
    meta = json.loads((args.run / "meta.json").read_text())
    if meta.get("status") != "completed":
        raise SystemExit(f"{args.run}: not a completed run ({meta.get('status')})")
    record = json.loads((args.run / "kstar_calibration.json").read_text())
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    problems = K.manifest_problems(record, rules, None) or K.evidence_problems(record, K.find_run)
    if problems:
        raise SystemExit(f"{args.run}: no manifest: {'; '.join(problems)}")
    path = K.manifest_path(args.root, record["lineage"])
    if path.exists():
        raise SystemExit(f"{path} exists: a lineage has one k* manifest")
    K.write_manifest(path, {**record, "calibration_run": args.run.name})
    if record["status"] == "unresolved":
        print(f"{path.relative_to(args.root)}: unresolved. Commit it; no certificate follows.")
        return 0
    rates = ", ".join(f"{k}: k* {v['kstar']}" for k, v in record["rates"].items())
    print(f"{path.relative_to(args.root)}: {rates}; B {record['budget']}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
