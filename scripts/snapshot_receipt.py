"""Write a source snapshot's receipt (C8 item E): the snapshot id, the clean commit it was taken
at and every file's sha256 checked against that commit's blob, so that a Pod without ``.git`` can
accept a committed manifest of the snapshot (``manifests.committed_json``).

    uv run python scripts/snapshot_receipt.py <snapshot package> [--out <file>]

Run it on the laptop right after ``source_snapshot.py prepare``, from the clean checkout the
snapshot was taken at; upload the receipt beside the package and pass its path to the chain
(``SOURCE_SNAPSHOT_RECEIPT``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition.manifests import snapshot_receipt
from residual_disposition.paths import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("package", type=Path, help="<store>/<snapshot id>")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        receipt = snapshot_receipt(args.package, args.repo)
    except (ValueError, OSError) as error:
        raise SystemExit(f"no receipt: {error}") from error
    out = args.out or args.package.parent / f"{receipt['snapshot_id']}.receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"{out}: snapshot {receipt['snapshot_id'][:12]} at {receipt['git_commit'][:12]}, "
          f"{len(receipt['files'])} files")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
