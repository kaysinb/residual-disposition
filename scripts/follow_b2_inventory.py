"""The follow-up's B2 reading inventory (preregistration §13.4–§13.5, §13.15), written from the
records: every qualification reading of D* and of unfiltered PD and their validation-only streams,
with each reading's package or pool and its stream salt (``follow_inventory.inventory``).

    uv run python scripts/follow_b2_inventory.py [--out manifests/follow_b2_inventory.json]

The manifest keeps the sources it was built from (D*'s lens packages and PD's pools by behavior and
role) and is written once: an existing file is refused unless it is identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition import follow_inventory as fi
from residual_disposition.paths import REPO_ROOT

OUT = REPO_ROOT / "manifests" / "follow_b2_inventory.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    dstar, pd = fi.sources(), fi.pd_sources()
    manifest = {"sources": {"dstar": dstar, "pd_unfiltered": pd}, **fi.inventory(dstar, pd)}
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if args.out.exists() and args.out.read_text() != text:
        raise SystemExit(f"{args.out} exists with other content: not overwritten")
    args.out.write_text(text)
    counts = {k: {n: len(rows) for n, rows in v.items()} for k, v in manifest.items()
              if k != "sources"}  # fmt: skip
    print(json.dumps({"out": str(args.out), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
