"""Stage 3's laptop records of a panel behavior (preregistration §12.7 Stage 3, §12.3.1; C17):
the detector diagnostic's sources (E4), from the cell phase s3_suppress made
(`stage3_diagnostic_sources.json`), recomputed from the runs it names and written once as
`manifests/stage3_diagnostic_sources_<behavior>.json`. Commit it before any labeling: the
diagnostic's set is drawn from these sources (`continuation_labels.py detector-diagnostic
--behavior <behavior>`), and a detector revision never replaces them. After the ensemble's labels,
the arbiter and the researcher's review (§12.9), `diagnostic` writes the result,
`manifests/stage3_diagnostic_<behavior>.json`, recomputed from the set, the final labels and the
committed sources, with the labels' and the review log's sha256; commit it before s3b.

    uv run python scripts/continuation_stage3.py sources runs/continuation_14b/<sources run>
    uv run python scripts/continuation_stage3.py diagnostic <set run> --labels <final labels> \
        --review-log <review log> [--revision <accepted detector version>]
"""

from __future__ import annotations

import argparse
import hashlib
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
    diag = sub.add_parser("diagnostic", help="the diagnostic's result manifest")
    diag.add_argument("set_run", type=Path, help="the detector-diagnostic set's run")
    diag.add_argument("--labels", type=Path, required=True, help="the final labels (JSON lines)")
    diag.add_argument("--review-log", type=Path, required=True)
    diag.add_argument("--revision", default=None, help="an accepted lenient detector's version")
    diag.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.command == "diagnostic":
        key = json.loads((args.set_run / "key.json").read_text())
        sources = S3.committed_sources(args.root, key["behavior"], rules, find_run)
        rows = [json.loads(x) for x in args.labels.read_text().splitlines() if x.strip()]
        record = S3.diagnostic_record(args.set_run, rows, sources, args.revision)
        for name, path in (("labels", args.labels), ("review_log", args.review_log)):
            record[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        S3.write_diagnostic(args.root, record)
        print(json.dumps({"written": str(S3.diagnostic_path(args.root, key["behavior"])),
                          "types": record["types"]}))  # fmt: skip
        return 0
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
