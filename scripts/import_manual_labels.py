"""Import labels exported from a labeling page as ManualLabel rows (docs/preregistration.md, Judge).

Validates the export against the set's private mapping and writes a new file
``manual_labels-<set id>-<content hash>.jsonl`` into ``<DATA_DIR>/labels/manual/``. Unknown or
duplicate items, labels of another set and items that are already stored are refused.

    uv run python scripts/import_manual_labels.py --mapping runs/labeling/<run id>/mapping.json \\
        --export ~/Downloads/labels-<set id>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from residual_disposition.judge import parse_export, write_manual_labels
from residual_disposition.paths import data_root

ARCHIVE_REMINDER = """\
Manual labels cannot be recreated: archive this file now (docs/operations.md, Rented machines),
for example
  rclone copy --immutable "{path}" "$ARTIFACT_REMOTE/data/labels/manual"
then check the copy and its content hash before relying on it."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", type=Path, required=True, help="mapping.json of the set")
    parser.add_argument("--export", type=Path, required=True, help="JSONL exported by the page")
    parser.add_argument("--labeler", default="pi", help="opaque role of the labeler (default pi)")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default <DATA_DIR>/labels/manual"
    )
    args = parser.parse_args(argv)

    mapping = json.loads(args.mapping.read_text())
    out_dir = args.out_dir or data_root() / "labels" / "manual"
    try:
        labels = parse_export(args.export.read_text(), mapping, labeler=args.labeler)
        path = write_manual_labels(out_dir, mapping["set_id"], labels)
    except (ValueError, FileExistsError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    unlabeled = len(mapping["items"]) - len(labels)
    print(f"Wrote {len(labels)} manual labels to {path}")
    if unlabeled:
        print(f"{unlabeled} items of set {mapping['set_id']} have no label")
    print(ARCHIVE_REMINDER.format(path=path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
