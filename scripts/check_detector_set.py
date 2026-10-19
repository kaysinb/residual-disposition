"""Check a Y₂ alternate's detector against the final labels of its labeling set (§12.3.6
criterion 2; §2's looser-detector rule, review s3j), and write ``check.json`` beside the set.

    uv run python scripts/check_detector_set.py runs/labeling/detectors-<set id> <labels.jsonl>

The labels file holds one {item_id, present, ...} row per item: the final labels after the
ensemble, the arbiter and the user's review. The verdict is "revise", "review" or "passes"
(``continuation_y2.validation_check``); acceptance also needs the user's full review (§12.9),
which ``scripts/continuation_y2.py --accepted <set id>`` records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_y2 as y2
from residual_disposition.paths import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("set", type=Path, help="runs/labeling/detectors-<set id>")
    parser.add_argument("labels", type=Path, help="the final labels, one JSON row per item")
    args = parser.parse_args(argv)
    key = json.loads((args.set / "key.json").read_text())
    labels = {}
    for line in args.labels.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["item_id"] in labels:
                raise SystemExit(f"{args.labels}: item {row['item_id']} labeled twice")
            labels[row["item_id"]] = bool(row["present"])
    check = y2.validation_check(key, labels)
    labels_path = args.labels.resolve()
    shown = (
        labels_path.relative_to(REPO_ROOT) if labels_path.is_relative_to(REPO_ROOT) else labels_path
    )
    out = {
        **check,
        "labels": str(shown),
        "detector_class": key.get("detector_class"),
        "rule": key.get("acceptance_rule"),
    }
    (args.set / "check.json").write_text(json.dumps(out, indent=2) + "\n")
    random = check["random"]
    print(
        f"{check['behavior_id']}: {check['verdict']}; random disagreements "
        f"{random['disagree']}/{random['n']}; to review {check['to_review']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
