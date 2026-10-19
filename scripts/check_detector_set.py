"""Check a Y₂ alternate's detector against its full review (§12.3.6 criterion 2; §2's
looser-detector rule; reviews s3j, s3m) and write ``check-<detector version>.json`` beside the set.

    uv run python scripts/check_detector_set.py runs/labeling/detectors-<set id> \
        --labels labels-<set id>.jsonl --review-log review-log-<set id>.json \
        [--dispositions <file>]

The labels are the final ``{item_id, present, labeled_at}`` rows and the review log the user's
record of every item (proposed, reviewed, final, changed), as the review page exports them; the
dispositions, when a prompted miss is labeled present, record each one as "accepted" (an isolated
miss) or "widen" (a recurring form). Every fixed text is rescored with the current detector, so a
revised detector is rechecked on the same texts and labels; each detector version has its own
record, never overwritten. The verdict is "revise", "review" or "passes"
(``continuation_y2.validation_check``), with the sha256 of every file of the record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import continuation_y2 as y2
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("set", type=Path, help="runs/labeling/detectors-<set id>")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review-log", type=Path, required=True)
    parser.add_argument("--dispositions", type=Path)
    args = parser.parse_args(argv)
    key = json.loads((args.set / "key.json").read_text())
    behavior_ids = {k["behavior_id"] for k in key["items"].values()}
    if len(behavior_ids) != 1:
        raise SystemExit(f"{args.set}: an alternate's set holds one behavior")
    (behavior_id,) = behavior_ids
    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior_id}").behavior)
    behavior = behavior_from_config(spec)
    try:
        check = y2.load_review(behavior, args.set, args.labels, args.review_log, args.dispositions)
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"{args.set}: {error}") from error
    out = args.set / f"check-{check['detector_version']}.json"
    text = json.dumps(check, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != text:
        raise SystemExit(f"{out} exists with another result: a check is never overwritten")
    out.write_text(text)
    random = check["random"]
    print(
        f"{behavior_id} (detector {check['detector_version']}): {check['verdict']}; random "
        f"disagreements {random['disagree']}/{random['n']}; to review {check['to_review']}; "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
