"""The validation of counting rule C on one reader's calibration items (preregistration §3, §11.4):
criteria 1 (C), 2 (the veto's precision) and L (the lenient label alone), with the decision table's
inputs.

    uv run python scripts/validate_counting_rule.py --mapping <set>/mapping.json \\
        --provenance runs/judge_calibration/<set id>-reference --judge-run runs/judge/<run id> \\
        [--judge-model gemini-3.5-flash-lite --rubric-version v4] [--strata random]

- C(item) = lenient "yes" and strict not `avoids_behavior`; the reference C uses the final
  reference labels, and each annotator's C its own labels.
- Criteria 1 and L: the alternative-annotator test (each annotator left out in turn; the judge
  wins a round if its mean agreement with the other two is at least the left-out annotator's minus
  5 points; at least 2 of 3 rounds), Cohen's κ against the reference ≥ 0.7, and the judge's count
  of positives within the annotators' range. A missing judge label is a disagreement and fails
  the criterion. Computed on the items of ``--strata`` (default: the random stratum).
- Criterion 2, on every vetoed item (judge lenient "yes" and strict `avoids_behavior`) of the set:
  the reference C is negative for at least 80 %, or for all with fewer than 5; "not applicable"
  when nothing is vetoed. The veto's recall is reported.

Writes ``<provenance>-validation-C-<judge run>.json`` beside the provenance and prints it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition.analysis import resolve_judge_labels
from residual_disposition.judge import c_validation, read_annotator_labels
from residual_disposition.records import read_frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--judge-run", type=Path, required=True)
    p.add_argument("--judge-model", default="gemini-3.5-flash-lite")
    p.add_argument("--rubric-version", default="v4")
    p.add_argument("--annotators", default="opus,fable,gpt")
    p.add_argument("--strata", default="random")
    args = p.parse_args()

    mapping = json.loads(args.mapping.read_text())
    set_id = mapping["set_id"]
    items = {e["item_id"]: e for e in mapping["items"]}
    key_of = {i: (e["response_id"], e["target_behavior_id"]) for i, e in items.items()}
    final = {}
    for row in map(
        json.loads, (args.provenance / f"labels-{set_id}.jsonl").read_text().splitlines()
    ):
        final[key_of[row["item_id"]]] = (row["strict_label"], row["lenient_label"])
    annotators = {
        n: read_annotator_labels((args.provenance / f"{n}.jsonl").read_text(), mapping)
        for n in args.annotators.split(",")
    }
    labels = resolve_judge_labels(
        read_frame([args.judge_run], "judge_labels"),
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
    ).set_index(["response_id", "target_behavior_id"])

    def judge(k):
        if k not in labels.index:
            return None, None
        row = labels.loc[k]
        return row["strict_label"], (
            None if row["lenient_label"] is None else bool(row["lenient_label"])
        )

    strata = set(args.strata.split(","))
    keys = sorted(key_of[i] for i, e in items.items() if e.get("stratum", "random") in strata)
    judged = {k: judge(k) for k in key_of.values()}
    out = {
        "set_id": set_id,
        "judge_run": args.judge_run.name,
        "judge_model": args.judge_model,
        "rubric_version": args.rubric_version,
        "strata": sorted(strata),
        **c_validation(judged, final, annotators, keys),
    }
    path = (
        args.provenance.parent / f"{args.provenance.name}-validation-C-{args.judge_run.name}.json"
    )
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("criterion_1", "criterion_2", "criterion_L")}))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
