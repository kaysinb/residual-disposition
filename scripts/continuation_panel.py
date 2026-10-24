"""The panel's analysis (preregistration §12.5, §12.6, §12.7 Stages 3–5), on CPU, from committed
evidence alone (``continuation_panel``).

- ``analysis``: writes `manifests/stage3_analysis.json` once. A pending input writes nothing and
  is named; no partial outcome is printed.
- ``check``: the committed analysis, recomputed from its evidence.
- ``failure <lineage> <phase> --kind budget|retry_exhausted --reason … [--run …]``: a phase's
  terminal stop, `manifests/stage3_failure_<lineage>_<phase>.json`, written once, only if it is
  an allowed one (``continuation_panel.FAILURE_KINDS``): the lineage's meter item stopped or
  dropped, or each named approved cell failed with a classified training failure on its
  attempt and its one retry of the same identity, with no completed run.
  Commit it before the analysis.

    uv run python scripts/continuation_panel.py analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import continuation_panel as P
from residual_disposition import continuation_rules as cr
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN


def readers_verifier(root: Path, rules, workers: int = 4):
    """The readers' recomputation verifier (``continuation_readers.committed_readers``), its
    exact tests in ``workers`` processes."""
    from residual_disposition.continuation_readers import committed_readers

    return lambda path: committed_readers(path, root, rules, find_run, workers=workers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="where manifests/ is")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("analysis")
    sub.add_parser("check")
    failure = sub.add_parser("failure")
    failure.add_argument("lineage")
    failure.add_argument("phase", choices=P.PHASES)
    failure.add_argument("--kind", required=True, choices=("budget", "retry_exhausted"))
    failure.add_argument("--reason", required=True)
    failure.add_argument("--run", action="append", default=[], help="a failed run it names")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    if args.command == "failure":
        return write_failure(args, rules)
    try:
        if args.command == "analysis":
            record = P.write_analysis(
                args.root, rules, find_run, readers_verifier(args.root, rules)
            )
            print(f"{P.analysis_path(args.root).relative_to(args.root)}: written. Commit it.")
            print(json.dumps(record["wording"], indent=2, ensure_ascii=False))
        else:
            P.committed_panel(args.root, rules, find_run, readers_verifier(args.root, rules))
            print("the committed analysis is what its evidence gives")
    except P.Pending as pending:
        raise SystemExit(f"pending, nothing written: {pending}") from None
    return 0


def write_failure(args: argparse.Namespace, rules) -> int:
    """The failure record of one phase of a certified lineage, bound to the committed contracts
    and checked to be an allowed terminal stop (``continuation_panel.failure_problems``)."""
    from residual_disposition import continuation_strong as CS
    from residual_disposition import pipeline
    from residual_disposition.paths import data_root

    behavior = _behavior(args.lineage)
    bundle = CS.committed_bundle(args.root, rules, find_run)
    entry = CS.committed_contract(args.root, behavior, rules, find_run, pipeline.panel_contract)
    identity, ws = P.lineage_identity(args.root, behavior, entry["entry"], bundle["sha256"])
    if identity["lineage"] != args.lineage:
        raise SystemExit(f"{args.lineage} is not {behavior}'s certified lineage")
    state_of = P.meter_state(data_root() / "ledgers" / "continuation_meter.jsonl")
    item = f"stage3/{behavior}/follow"
    meter = {"item": item, "state": state_of(item)} if args.kind == "budget" else None
    record = {"lineage": args.lineage, "behavior": behavior, "contracts": bundle["sha256"],
              "phase": args.phase, "kind": args.kind, "reason": args.reason,
              "runs": list(args.run), "meter": meter}  # fmt: skip
    cells = None
    if args.kind == "retry_exhausted" and args.phase == "kstar":
        cells = P.kstar_cells(ws, identity, rules)
    elif args.kind == "retry_exhausted" and args.phase == "certificate":
        from residual_disposition import continuation_kstar as K

        kstar = K.committed_manifest(K.manifest_path(args.root, args.lineage), rules, args.root)
        cells = dict(P.certificate_expected(ws, identity, kstar, rules)["cells"].values())
    problems = P.failure_problems(record, find_run, identity, state_of, pipeline.completed_run,
                                  cells)  # fmt: skip
    if problems:
        raise SystemExit(f"no failure record: {'; '.join(problems)}")
    path = P.failure_path(args.root, args.lineage, args.phase)
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{path.relative_to(args.root)}: written. Commit it.")
    return 0


def _behavior(lineage: str) -> str:
    """The panel behavior of a certified lineage's name (``stage3_<b>_<dose>`` or
    ``strong_<b>_cont_<dose>``)."""
    for behavior in P.PANEL:
        if lineage.startswith((f"stage3_{behavior}_", f"strong_{behavior}_cont_")):
            return behavior
    raise SystemExit(f"{lineage} is no certified lineage of the panel")


if __name__ == "__main__":
    sys.exit(main())
