"""The stack at 400's O4 check as its record (preregistration §13.16, "eli5's stack at 400"): the
pulled session's inputs, its record recomputed from the runs they name (``follow_stack.o4_record``),
with the screening's PD hits of L at 400 in full text beside it (``follow_stack.screening_hits``,
unadjudicated), becomes ``manifests/follow_stack400_o4.json``, written once at a clean commit.

    uv run python scripts/follow_stack400.py --o4 <result.json> [--write]
    uv run python scripts/follow_stack400.py --screening-hits

Descriptive: no verdict. Without ``--write`` it prints the counts; ``--screening-hits`` prints the
screening's hits of L at 400 alone, from its pinned pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_stack as F
from residual_disposition import pipeline as P
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT
from residual_disposition.tracking import git_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--o4", type=Path, help="the O4 session's result.json")
    parser.add_argument("--write", action="store_true", help="create the record (clean commit)")
    parser.add_argument("--screening-hits", action="store_true",
                        help="the screening's hits of L at 400, in full text")  # fmt: skip
    args = parser.parse_args(argv)
    if (args.o4 is None) == (not args.screening_hits):
        raise SystemExit("--o4 <result.json> or --screening-hits")
    b4 = cr.follow_b4()
    o4 = F.o4_config(cr.follow_build_stack(400), b4)
    detector = P.Workspace(load_config(b4["experiment"])).behavior.detector
    screening = committed_json(REPO_ROOT / F.MANIFEST, "the stack screening's manifest")
    try:
        hits = F.screening_hits(b4, screening, find_run, detector)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.screening_hits:
        print(json.dumps(hits, indent=1, ensure_ascii=False))
        return 0
    result = json.loads(args.o4.read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"{args.o4}: it is {result.get('status')}, not complete")
    try:
        record = F.o4_record(b4, o4, result["inputs"], find_run, detector)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for role, c in record["counts"].items():
        print(f"{role}: {c['hits']} of {c['n']}, bound {c['bound']:.4f}")
    if args.write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        path = REPO_ROOT / o4["record"]
        if path.exists():
            raise SystemExit(f"{o4['record']} exists: recorded once")
        path.write_text(json.dumps({**record, "screening_hits": hits, "inputs": result["inputs"],
                                    "commit": git["commit"]}, indent=1, sort_keys=True,
                                   ensure_ascii=False) + "\n")  # fmt: skip
        print(f"wrote {o4['record']}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
