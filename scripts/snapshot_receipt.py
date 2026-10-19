"""Write a source snapshot's receipt (C8 item E): the snapshot id, the clean commit it was taken
at and every file's sha256 checked against that commit's blob, so that a Pod without ``.git`` can
accept a committed manifest of the snapshot (``manifests.committed_json``).

    uv run python scripts/snapshot_receipt.py write <snapshot package> [--out <file>]
    uv run python scripts/snapshot_receipt.py check <receipt> --package <snapshot package>

``write`` runs on the laptop right after ``source_snapshot.py prepare``, from the clean checkout
the snapshot was taken at. ``check``, the coordinator's (review s3z2), makes the receipt again
from git and the package, refuses one that differs, and prints the receipt file's sha256 and the
snapshot id: the coordinator sets them in the Pod's environment at its creation
(``SNAPSHOT_RECEIPT_SHA256``, ``SOURCE_SNAPSHOT_ID``); the chain passes the receipt's path
(``SOURCE_SNAPSHOT_RECEIPT``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from residual_disposition.manifests import snapshot_receipt
from residual_disposition.paths import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="write the receipt of a snapshot package")
    write.add_argument("package", type=Path, help="<store>/<snapshot id>")
    write.add_argument("--repo", type=Path, default=REPO_ROOT)
    write.add_argument("--out", type=Path, default=None)
    check = sub.add_parser("check", help="check a receipt against git; print its sha256")
    check.add_argument("receipt", type=Path)
    check.add_argument("--package", type=Path, required=True)
    check.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        receipt = snapshot_receipt(args.package, args.repo)
    except (ValueError, OSError) as error:
        raise SystemExit(f"no receipt: {error}") from error
    text = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    if args.command == "write":
        out = args.out or args.package.parent / f"{receipt['snapshot_id']}.receipt.json"
        out.write_text(text)
    else:
        out = args.receipt
        if out.read_text() != text:
            raise SystemExit(f"{out} is not the receipt git gives for {args.package}")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    shown = {"receipt": str(out), "SNAPSHOT_RECEIPT_SHA256": sha}
    shown |= {"SOURCE_SNAPSHOT_ID": receipt["snapshot_id"], "git_commit": receipt["git_commit"]}
    print(json.dumps({**shown, "files": len(receipt["files"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
