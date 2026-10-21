"""Stage 3's laptop records of a panel behavior (preregistration §12.7 Stage 3, §12.3.1; C17):
the detector diagnostic's sources (E4), from the cell phase s3_suppress made
(`stage3_diagnostic_sources.json`), recomputed from the runs it names and written once as
`manifests/stage3_diagnostic_sources_<behavior>.json`. Commit it before any labeling: the
diagnostic's set is drawn from these sources, and a detector revision never replaces them.

    uv run python scripts/continuation_stage3.py sources runs/continuation_14b/<sources run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage3 as S3
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources", help="the diagnostic's sources manifest")
    sources.add_argument("run", type=Path, help="the stage3_diagnostic_sources cell")
    sources.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    step = (yaml.safe_load((args.run / "config.yaml").read_text()) or {}).get("step")
    if step != S3.SOURCES:
        raise SystemExit(f"{args.run}: a {step} cell, not {S3.SOURCES}")
    record = json.loads((args.run / f"{S3.SOURCES}.json").read_text())
    data = {**record, "sources_run": args.run.name}
    S3.write_sources(args.root, data, rules, find_run)
    print(json.dumps({"written": str(S3.sources_path(args.root, record["behavior"])),
                      "doses": record["doses"], "from": record["from"]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
