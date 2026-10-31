"""Record the random-branch session of the existing L packages (preregistration §13.4; barriers B2
and B3) after its pull, once, as manifests/follow_lrandom.json.

    uv run python scripts/follow_lrandom.py runs/follow_lrandom/<run> [--write]

The session's output (``scripts/pod/follow_lrandom.py``) must be complete and exactly the frozen
session's builds (``continuation_rules.follow_lrandom``): each L's random branch alone, at its
package's recorded seed, with its method checks passed. The record names the run, the frozen
session, the result's sha256 and, per build, its package, its directory and every file's sha256;
the inventories find the branches only through it (``follow_inventory.random_branches``).
Without ``--write`` it prints the record; ``--write`` needs a clean commit and creates it once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition.paths import REPO_ROOT, runs_root
from residual_disposition.tracking import git_state


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(run: Path) -> dict:
    session = cr.follow_lrandom()
    found = sorted(run.glob("out/*/result.json"))
    if len(found) != 1:
        raise SystemExit(f"{run}: {len(found)} session results, not one")
    out = found[0].parent
    result = json.loads(found[0].read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"the session is {result.get('status')}, not complete")
    states = [b["state"] for b in session["builds"]]
    if sorted(result.get("builds") or {}) != sorted(states):
        raise SystemExit("the session's builds are not the frozen ones")
    builds = {}
    for b in session["builds"]:
        d = out / "builds" / b["state"]
        made = json.loads((d / "build.json").read_text())
        if made["branches"] != ["random"] or not result["builds"][b["state"]]["complete"]:
            raise SystemExit(f"{b['state']}: not its random branch, complete")
        builds[b["state"]] = {"package": b["package"], "dir": str(d.relative_to(run)),
                              "files": {str(p.relative_to(d)): sha(p)
                                        for p in sorted(d.rglob("*")) if p.is_file()}}  # fmt: skip
    return {"run": run.name, "session_sha256": cr.FOLLOW_LRANDOM_FROZEN,
            "result_sha256": sha(found[0]), "builds": builds}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if not args.run.resolve().is_relative_to(runs_root().resolve()):
        raise SystemExit(f"{args.run} is not under the runs root")
    data = record(args.run)
    print(json.dumps({k: v for k, v in data.items() if k != "builds"} | {
        "builds": {s: len(b["files"]) for s, b in data["builds"].items()}}))  # fmt: skip
    if args.write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        path = REPO_ROOT / "manifests" / "follow_lrandom.json"
        if path.exists():
            raise SystemExit(f"{path.name} exists: the session is recorded once")
        path.write_text(json.dumps({**data, "commit": git["commit"]}, indent=1) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
