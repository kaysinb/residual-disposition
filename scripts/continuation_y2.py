"""Select Y₂ and write ``manifests/y2.json`` (§12.3.6; reviews s3j, s3m): the first alternate in
the frozen order that meets the three criteria, from the evidence at hand.

    uv run python scripts/continuation_y2.py \
        --validation runs/labeling/detectors-<set id> <labels> <review log> \
        --reference runs/continuation_14b/<y2_reference run>

- Criterion 1 is read from the detector's config.
- Criterion 2: the stage-0 elicitation's counts, rescored with the current detector
  (``--elicitation``; 0 of 300 without the prompt, at least 0.90 with it), and the detector check
  from the full-review record (``continuation_y2.load_review``; ``--dispositions`` for reviewed
  misses): passed when its verdict is "passes"; failed only by a rejection record
  (``--rejection``, review s3o) of its set and detector version, which a passing check refuses
  unless the record names the protocol decision that allows it; otherwise pending. The set's
  strata must be the elicitation's misses and unprompted hits.
- Criterion 3: the candidate's Y₂ organism's reference, the record of phase y2
  (``y2_reference.json``), checked against its runs: a completed relearn_reference pass of the
  organism on the frozen relearning-evaluation requests, one response to each of the 200 prompts,
  its hits rescored; the organism made by a completed stage-1-recipe training run, seed 0.

At most one validation and one reference per candidate. A pending candidate is not passed over.
Every input must lie under the runs root or the data root; its location is recorded there
(``continuation_y2.selection``), so that the certificate accepts the manifest by making the same
selection again (review s3q). With a selection the manifest is written with the hashes of its
review record and its provenance, and must be committed before the first certificate run;
otherwise nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN

ELICITATION = "runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--elicitation", type=Path, default=REPO_ROOT / ELICITATION)
    parser.add_argument("--validation", nargs=3, type=Path, action="append", default=[],
                        metavar=("SET", "LABELS", "REVIEW_LOG"))  # fmt: skip
    parser.add_argument("--dispositions", type=Path, action="append", default=[])
    parser.add_argument("--rejection", type=Path, action="append", default=[],
                        help="a rejection record {set_id, detector_version, why, ...}")  # fmt: skip
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "manifests" / "y2.json")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    try:
        dispositions = {}
        for path in args.dispositions:
            set_id = json.loads(path.read_text()).get("set_id")
            if set_id in dispositions:
                raise ValueError(f"two dispositions for the set {set_id}")
            dispositions[set_id] = y2.locate(path)
        validations = []
        for set_dir, labels, log in args.validation:
            set_id = json.loads((set_dir / "key.json").read_text()).get("set_id")
            validations.append(
                {"set": y2.locate(set_dir), "labels": y2.locate(labels),
                 "review_log": y2.locate(log), "dispositions": dispositions.get(set_id)}
            )  # fmt: skip
        inputs = {
            "elicitation": y2.locate(args.elicitation),
            "validations": validations,
            "rejections": [y2.locate(p) for p in args.rejection],
            "references": [run.name for run in args.reference],
        }
        result = y2.selection(inputs, rules)
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    chosen = result["selection"]
    print(json.dumps({k: v for k, v in chosen.items() if k != "assessed"}))
    for c, a in chosen["assessed"].items():
        print(f"  {c}: {a}")
    if chosen["status"] != "selected":
        return 0 if chosen["status"] == "pending" else 1
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: Y₂ is chosen once")
    y2.manifest(chosen, result["evidence"], args.out, result["provenance"])
    print(f"{args.out}: {chosen['choice']}; commit it before the first certificate run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
