"""Write barrier B3's reading inventory (preregistration §13.4–§13.6, §13.15) once, as
manifests/follow_b3_inventory.json.

    uv run python scripts/follow_b3_inventory.py [--write]

The inventory is ``follow_inventory.b3_inventory`` over the recorded packages (the build session's
named by ``follow_inventory.BUILD_SESSION``), the committed confirmations and unfiltered PD's reuse
decisions (``follow_pd.reuse_map``, recomputed from the committed Stage-3 readings). Without
``--write`` it prints the counts and the reused readings; ``--write`` creates the manifest, and
refuses to replace one with other content.
"""

from __future__ import annotations

import argparse
import json
import sys

from residual_disposition import follow_inventory as fi
from residual_disposition import follow_pd
from residual_disposition.continuation_kstar import find_run
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT

MANIFEST = REPO_ROOT / "manifests" / "follow_b3_inventory.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="create the manifest")
    args = parser.parse_args(argv)
    path = REPO_ROOT / "manifests" / "follow_confirmations.json"
    confirmed = committed_json(path, "the confirmations")["confirmed"]
    src = fi.with_builds(fi.sources(), fi.built_packages())
    inventory = fi.b3_inventory(src, fi.pd_analysis_sources(), confirmed,
                                lambda pd: follow_pd.reuse_map(pd, find_run))  # fmt: skip
    for part, rows in inventory.items():
        if part not in ("reuse", "sources"):
            print(part, {k: len(v) for k, v in rows.items()})
    print("reused:", sorted(s for s, d in inventory["reuse"].items() if d["reuse"]))
    if args.write:
        text = json.dumps(inventory, indent=1, sort_keys=True) + "\n"
        if MANIFEST.exists() and MANIFEST.read_text() != text:
            raise SystemExit(f"{MANIFEST.name} exists with other content")
        MANIFEST.write_text(text)
        print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
