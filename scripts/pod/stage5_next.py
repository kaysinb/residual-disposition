"""The next step of exception a's continued-mode lineage, or of the descriptive DPO lineage beside
it (``--lineage stage5_dpo``; C15), in one space (§12.7 Stage 5; C12): the completed cells of the
space found (the selection, the confirmation draws, X_cont and the
expression checks at the analysis dose; ``continuation_stage5.find_inputs``), the decision made
from them (``continuation_stage5.decide``), and what the chain runs next. Prints one JSON line:

    {"status": "pending" | "decided", "step": "selection" | "confirm" | "retest" | "xcont"
     | "expression" | "decide", "dose": ..., "latent": ..., "sham_s": ..., "inputs": {...},
     "decision": {...}}

    python scripts/pod/stage5_next.py --runs <space>/runs --behavior example_link \
        [--lineage stage5 | stage5_dpo]

With no completed selection of the lineage the step is "selection" (phase s5_suppress or
s5d_suppress); an ambiguous space (two cells of one kind) is refused, exit 1. The DPO lineage
has no X_cont step. ``--lineage stage3``: a panel behavior's Stage-3 stack lineage (C17), the
same way, its inputs naming A's reference. ``--lineage stage3_cont``: its strong branch's
continued line (C17-C), which has no decision of its own: the step is "selection" before its one
completed selection, "screen" after it (its inputs naming it, which the panel branch's screening
reads), or "infeasible" after a recorded infeasibility of its training.
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
    parser.add_argument("--lineage", choices=("stage5", "stage5_dpo", "stage3", "stage3_cont"),
                        default="stage5")  # fmt: skip
    args = parser.parse_args(argv)
    # find_run reads this space's runs only, and X_cont's check its states (rest_runner.sh's
    # layout: <space>/runs, <space>/data), never the default data root (s5a, 2026-10-20)
    os.environ["RUNS_DIR"] = str(args.runs)
    os.environ["DATA_DIR"] = str(Path(args.runs).parent / "data")
    from residual_disposition.continuation_kstar import find_run

    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    line = S5._p().s5_line(args.lineage)
    if line is S5._p().S3_CONTINUED:
        return continued(args.runs, args.behavior, line)
    held = [c for step in (line.step("selection"), line.step("infeasibility"))
            for _, c, _ in S5.cells(args.runs, EXPERIMENT, step)]  # fmt: skip
    if not held:
        print(json.dumps({"status": cr.PENDING, "step": "selection", "dose": None}))
        return 0
    try:
        inputs = S5.find_inputs(args.runs, EXPERIMENT, args.behavior, rules, find_run, line)
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


def continued(runs: Path, behavior: str, line) -> int:
    """The continued line's next step: its one selection, or its recorded infeasibility, of this
    behavior's lineage; two of a kind are refused, exit 1."""
    name = line.lineage(behavior)
    plans = {r.name: (s.get("plan") or {}).get("lineage")
             for r, _, s in S5.cells(runs, EXPERIMENT, line.step("training"))}  # fmt: skip
    found = {
        "selection": [r.name for r, c, _ in S5.cells(runs, EXPERIMENT, line.step("selection"))
                      if (c.get("params") or {}).get("lineage") == name],
        # an infeasibility names its attempts, whose training cells hold the plan's lineage
        "infeasibility": [r.name for r, c, _ in S5.cells(runs, EXPERIMENT,
                                                         line.step("infeasibility"))
                          if name in {plans.get(a) for a in (c.get("params") or {})
                                      .get("attempts") or []}],
    }  # fmt: skip
    if sum(len(v) for v in found.values()) > 1:
        print(json.dumps({"status": "error", "error": f"{name}: {found}"}))
        return 1
    if found["infeasibility"]:
        out = {"status": "decided", "step": "infeasible", "dose": None,
               "inputs": {"infeasibility": found["infeasibility"][0]}}  # fmt: skip
    elif found["selection"]:
        out = {"status": cr.PENDING, "step": "screen", "dose": None,
               "inputs": {"selection": found["selection"][0]}}  # fmt: skip
    else:
        out = {"status": cr.PENDING, "step": "selection", "dose": None}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
