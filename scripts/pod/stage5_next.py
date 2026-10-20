"""The next step of exception a's continued-mode lineage in one space (§12.7 Stage 5; C12): the
completed cells of the space found (the selection, the confirmation draws, X_cont and the
expression checks at the analysis dose; ``continuation_stage5.find_inputs``), the decision made
from them (``continuation_stage5.decide``), and what the chain runs next. Prints one JSON line:

    {"status": "pending" | "decided", "step": "selection" | "confirm" | "retest" | "xcont"
     | "expression" | "decide", "dose": ..., "latent": ..., "sham_s": ..., "inputs": {...},
     "decision": {...}}

    python scripts/pod/stage5_next.py --runs <space>/runs --behavior example_link

With no completed selection the step is "selection" (phase s5_suppress); an ambiguous space (two
cells of one kind) is refused, exit 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage5 as S5
from residual_disposition.simulate_continuation import FROZEN

EXPERIMENT = "continuation_14b"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="the space's runs directory")
    parser.add_argument("--behavior", required=True)
    args = parser.parse_args(argv)
    # find_run reads this space's runs only, and X_cont's check its states (rest_runner.sh's
    # layout: <space>/runs, <space>/data), never the default data root (s5a, 2026-10-20)
    os.environ["RUNS_DIR"] = str(args.runs)
    os.environ["DATA_DIR"] = str(Path(args.runs).parent / "data")
    from residual_disposition.continuation_kstar import find_run

    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    held = [c for step in (S5.SELECTION, S5.INFEASIBILITY)
            for _, c, _ in S5.cells(args.runs, EXPERIMENT, step)]  # fmt: skip
    if not held:
        print(json.dumps({"status": cr.PENDING, "step": "selection", "dose": None}))
        return 0
    try:
        inputs = S5.find_inputs(args.runs, EXPERIMENT, args.behavior, rules, find_run)
        decision = S5.decide(inputs, rules, find_run)["decision"]
        record = S5._selection(inputs, rules, find_run) if inputs["selection"] else {"doses": {}}
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 1
    step, dose = S5.next_step(decision)
    states = (record["doses"].get(str(dose)) or {}) if dose is not None else {}
    out = {
        "status": decision["status"],
        "step": step,
        "dose": dose,
        "latent": states.get("latent"),
        "sham_s": states.get("sham_s"),
        "inputs": inputs,
        "decision": decision,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
