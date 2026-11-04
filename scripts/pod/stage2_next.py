"""The next step of a pilot behavior's Stage 2 in one space (C8, session A1): the completed cells
of the space found (the screening, the confirmation draws, the expression checks at the
analysis dose), the decision made from them (``continuation_stage2.decide``), and what the chain
runs next. Prints one JSON line:

    {"status": "pending" | "decided", "step": "confirm" | "expression" | "retest" | "screening"
     | "decide", "dose": ..., "latent": ..., "sham_s": ..., "inputs": {...}, "decision": {...}}

    python scripts/pod/stage2_next.py --runs <space>/runs --behavior example_link

Exactly one completed screening cell of the behavior, one completed check of each kind at the
analysis dose's latent state, and the confirmation cells of the screening's latent states are
read; an ambiguous space (two of a kind) is refused, exit 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.simulate_continuation import FROZEN

EXPERIMENT = "continuation_14b"


def cells(runs: Path, step: str) -> list[tuple[Path, dict, dict]]:
    """The completed cells of ``step`` in the experiment: (directory, config, summary)."""
    out = []
    for config in sorted((runs / EXPERIMENT).glob(f"*-{step}-*/config.yaml")):
        run = config.parent
        meta = json.loads((run / "meta.json").read_text())
        if meta.get("status") == "completed":
            summary = json.loads((run / "summary.json").read_text())
            out.append((run, yaml.safe_load(config.read_text()) or {}, summary))
    return out


def find_inputs(runs: Path, behavior: str) -> tuple[dict, dict]:
    """The decision's inputs found in the space and the screening record."""
    screens = [
        (run, c)
        for run, c, _ in cells(runs, "stage2_screening")
        if (c.get("behavior") or {}).get("id") == behavior
    ]
    if len(screens) != 1:
        raise ValueError(f"{len(screens)} completed screenings of {behavior}, not one")
    run, _ = screens[0]
    record = json.loads((run / "stage2_screening.json").read_text())
    latents = {v["latent"]: int(d) for d, v in record["doses"].items()}
    draws = sorted(
        (s.get("attempt", 0), r.name)
        for r, c, s in cells(runs, "stage2_confirmation")
        if (c.get("params") or {}).get("latent") in latents
    )
    if len({a for a, _ in draws}) != len(draws):
        raise ValueError("two confirmation cells of one attempt")
    inputs = {
        "behavior": behavior,
        "detector_version": S2.current_detector_version(behavior),
        "screening": run.name,
        "confirmation": [name for _, name in draws],
        "expression": {},
    }
    return inputs, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True, help="the space's runs directory")
    parser.add_argument("--behavior", required=True)
    args = parser.parse_args(argv)
    os.environ["RUNS_DIR"] = str(args.runs)  # find_run reads this space only
    from residual_disposition.continuation_kstar import find_run

    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    try:
        inputs, record = find_inputs(args.runs, args.behavior)
        decision = S2.decide(inputs, rules, find_run)["decision"]
        analysis = decision.get("analysis_dose")
        if analysis is not None:
            latent = record["doses"][str(analysis)]["latent"]
            for check, step in (("ood", "rates_ood"), ("adl_corpus", "adl_corpus")):
                found = [
                    r.name
                    for r, c, _ in cells(args.runs, step)
                    if (c.get("params") or {}).get("state") == latent
                ]
                if len(found) > 1:
                    raise ValueError(f"{len(found)} completed {check} checks of {latent}")
                if found:
                    inputs["expression"][check] = found[0]
            decision = S2.decide(inputs, rules, find_run)["decision"]
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 1
    nxt = str(decision.get("next") or "")
    if decision["status"] == "decided":
        step = "decide"
    elif nxt.startswith("confirmation retest"):
        step = "retest"  # a detector revision: the retest is a procedure by hand, not the chain's
    elif nxt.startswith("confirmation at"):
        step = "confirm"
    elif nxt.startswith("expression checks"):
        step = "expression"
    else:
        step = "screening"
    dose = decision.get("analysis_dose")
    if step in ("confirm", "expression", "retest"):
        dose = int(nxt.rsplit(" ", 1)[1])
    doses = record["doses"]
    out = {
        "status": decision["status"],
        "step": step,
        "dose": dose,
        "latent": doses[str(dose)]["latent"] if dose is not None else None,
        "sham_s": doses[str(dose)]["sham_s"] if dose is not None else None,
        "inputs": inputs,
        "decision": decision,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
