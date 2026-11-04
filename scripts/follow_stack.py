"""Write the follow-up's stack-branch screening as its manifest (preregistration §13.7; barrier B4):
the pulled screening's inputs, its record recomputed from the runs they name
(``follow_stack.screening_record``), becomes ``manifests/follow_stack_eli5.json``, written once at
a clean commit and committed before any validation cell.

    uv run python scripts/follow_stack.py runs/follow_stack/<run>/out/result.json [--write]
    uv run python scripts/follow_stack.py --validation <result.json> [--write]

Without ``--write`` it prints the outcome; ``--write`` creates the manifest. With ``--validation``
the pulled validation's record, recomputed from its runs, becomes
``manifests/follow_stack_eli5_validation.json``.
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
from residual_disposition.paths import REPO_ROOT
from residual_disposition.tracking import git_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", type=Path, help="the screening session's result.json")
    parser.add_argument("--write", action="store_true", help="create the manifest (clean commit)")
    parser.add_argument("--validation", action="store_true", help="the validation's result.json")
    args = parser.parse_args(argv)
    b4 = cr.follow_b4()
    result = json.loads(args.result.read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"{args.result}: it is {result.get('status')}, not complete")
    ws = P.Workspace(load_config(b4["experiment"]))
    detector = ws.behavior.detector
    if args.validation:
        return validation(b4, result, detector, args.write)

    def digests(latent: str) -> tuple[str, str]:
        return P.tool_pd_cell(ws, latent).digest, P.adl_corpus_cell(ws, latent).digest

    try:
        record = F.screening_record(b4, result["inputs"], find_run, detector, digests)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in record["screened"]:
        print(f"{row['dose']}: eligible {row['eligible']}")
    print(f"outcome: {record['outcome']}; chosen {record['chosen']}")
    if args.write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        try:
            F.write_manifest(REPO_ROOT / F.MANIFEST, {**record, "inputs": result["inputs"],
                                                      "commit": git["commit"]})  # fmt: skip
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"wrote {F.MANIFEST}. Commit it.")
    return 0


def validation(b4: dict, result: dict, detector: object, write: bool) -> int:
    try:
        record = F.validation_record(b4, result["inputs"], find_run, detector)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"passed {record['verdict']['passed']}: {record['verdict']['parts']}")
    if write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        try:
            F.write_validation(REPO_ROOT / F.VALIDATION, {**record, "inputs": result["inputs"],
                                                          "commit": git["commit"]})  # fmt: skip
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"wrote {F.VALIDATION}. Commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
