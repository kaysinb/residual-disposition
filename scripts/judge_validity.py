"""Check a judge run against the preregistered validity criteria (docs/preregistration.md, Judge).

The reference is a labeling set's final labels (imported with labeler ``ensemble``) and the
independent labels of each reference annotator, kept with the set's provenance:

    uv run python scripts/judge_validity.py --mapping runs/labeling/<run id>/mapping.json \\
        --provenance runs/judge_calibration/<set id>-reference --judge-run runs/judge/<run id>

Writes ``validity.json`` (alternative-annotator test, κ, share of positive labels, per label kind)
next to the provenance, named after the judge run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from residual_disposition.judge import (
    final_labels,
    judge_validity,
    read_annotator_labels,
    read_manual_labels,
)
from residual_disposition.paths import data_root
from residual_disposition.records import JudgeLabel, read_table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", type=Path, required=True, help="mapping.json of the set")
    parser.add_argument("--provenance", type=Path, required=True, help="the set's reference dir")
    parser.add_argument("--judge-run", type=Path, required=True, help="run dir of scripts/judge.py")
    parser.add_argument("--annotators", default="opus,fable,gpt", help="files <name>.jsonl")
    parser.add_argument("--labeler", default="ensemble", help="labeler of the final labels")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-rounds", type=int, default=2)
    parser.add_argument("--min-kappa", type=float, default=0.7)
    args = parser.parse_args(argv)

    mapping = json.loads(args.mapping.read_text())
    keys = {(e["response_id"], e["target_behavior_id"]) for e in mapping["items"]}
    reference = [
        label
        for label in read_manual_labels(data_root() / "labels" / "manual")
        if label.labeler == args.labeler and (label.response_id, label.target_behavior_id) in keys
    ]
    if len(reference) != len(keys):
        print(f"{len(reference)} final labels for {len(keys)} items", file=sys.stderr)
        return 1
    annotators = {
        name: read_annotator_labels((args.provenance / f"{name}.jsonl").read_text(), mapping)
        for name in args.annotators.split(",")
    }
    labels: list[JudgeLabel] = list(read_table([args.judge_run], JudgeLabel.table))
    report = judge_validity(
        final_labels(labels),
        reference,
        annotators,
        margin=args.margin,
        min_rounds=args.min_rounds,
        min_kappa=args.min_kappa,
    )
    models = sorted({label.judge_model for label in labels})
    rubrics = sorted({label.rubric_version for label in labels})
    report = {
        "set_id": mapping["set_id"],
        "judge_run": args.judge_run.name,
        "judge_models": models,
        "rubric_versions": rubrics,
        **report,
    }
    out = args.provenance.parent / f"{args.provenance.name}-validity-{args.judge_run.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ("strict", "lenient"):
        r = report[kind]
        print(
            f"{kind}: passed={r['passed']} criteria={r['criteria']} rounds_won={r['rounds_won']} "
            f"kappa={r['kappa']:.3f} judge_positive={r['judge_positive']} "
            f"annotators={r['annotator_positive']} missing={r['missing_judge_labels']}"
        )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
