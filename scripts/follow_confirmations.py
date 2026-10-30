"""The follow-up's confirmations decided (preregistration §13.3), from their runs on CPU, written
once as manifests/follow_confirmations.json.

    uv run python scripts/follow_confirmations.py [--write]

The decision is recomputed from the runs (``RUNS_DIR``, else the repository's ``runs``):
``follow_confirm.decision``, every completed cell verified whole and its hits counted again on its
saved responses. The record adds each draw's run, configuration hash and files' sha256
(``follow_confirm.manifest_record``). Without ``--write`` it prints the lines' outcome;
``--write`` needs a clean commit and both lines decided, and creates the manifest once.
``follow_confirm.committed_confirmations`` reads it back only committed and recomputed from the
same runs, for ``follow_inventory.resolve``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from residual_disposition import follow_confirm as F
from residual_disposition import pipeline as P
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT, runs_root
from residual_disposition.tracking import git_state


def workspace(spec, behavior):
    """The line's workspace: its experiment's configuration, without override."""
    return P.Workspace(load_config(spec.experiment(behavior)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="create the manifest (clean commit)")
    args = parser.parse_args(argv)
    spec = F.load_spec()
    experiment = Path(spec.experiment(spec.lines[0])).parts[-2]
    try:
        record = F.manifest_record(spec, runs_root(), experiment, workspace)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for b, line in record["lines"].items():
        draws = ", ".join(f"attempt {d['attempt']} at {d['dose']}: {d['hits']}/{d['n']} "
                          f"({'passed' if d['passed'] else 'failed'}, {d['run']})"
                          for d in line["attempts"])  # fmt: skip
        print(f"{b}: {line['status']} at {line['dose']}; {draws}")
    if args.write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        path = REPO_ROOT / F.MANIFEST
        try:
            F.write_manifest(path, {**record, "commit": git["commit"]})
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"wrote {F.MANIFEST}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
