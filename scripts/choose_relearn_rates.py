"""Choose the two relearning learning rates of one behavior from its calibration runs (path B).

    uv run python scripts/choose_relearn_rates.py --runs runs/<experiment> \\
        --state <sham_s state id> --label sft/stack/- --dose <d1> [--audit-doses <d1> <d2>]

Reads the completed relearning cells with purpose "calibration" on the given sham_s state, takes
each run's T as the certificate counts it (interpolated, at least one step), and applies
``relearn_rules.choose_relearn_rates``. When the rule asks for another calibration candidate, it
prints the pass overrides that run it (phase calibrate); when it chooses, those of the certificate
pass (phase certify, no calibration): the rates, the step budget and the relearning targets at the
audit doses; when it stops, it exits with status 2 and the reason. The calibration results and
the decision are logged in the script's own run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from residual_disposition.config import load_config
from residual_disposition.relearn_rules import RateChoice, certificate_time, choose_relearn_rates
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def calibration_results(runs: list[Path], state_id: str) -> dict[float, float | None]:
    """Learning rate → T of sham_s (``certificate_time``: interpolated, at least one step; None if
    censored) of the completed calibration runs on ``state_id``; for a rate run with two budgets
    (1e-4 extended to 600), the longer one. Two runs of one rate and budget are ambiguous: an
    error."""
    best: dict[float, tuple[int, float | None]] = {}
    for run_dir in runs:
        meta = json.loads((run_dir / "meta.json").read_text())
        if meta.get("name") != "relearn" or meta.get("status") != "completed":
            continue
        params = yaml.safe_load((run_dir / "config.yaml").read_text())["params"]
        if params.get("purpose") != "calibration" or params["state"] != state_id:
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        lr, budget = float(params["lr"]), int(params["settings"]["budget"])
        t = None if summary["censored"] else certificate_time(summary)
        if lr in best and budget == best[lr][0]:
            raise SystemExit(f"two calibration runs at lr {lr:g}, budget {budget} on {state_id}")
        if lr not in best or budget > best[lr][0]:
            best[lr] = (budget, t)
    return {lr: t for lr, (_, t) in best.items()}


def overrides_for(
    choice: RateChoice, *, label: str, dose: int, audit_doses: list[int]
) -> list[str]:
    """Command-line overrides of the next pipeline pass for ``choice``."""
    # YAML flow mappings: a space after each colon
    if choice.status == "calibrate":
        rates = ", ".join(f"{lr:g}" for lr in choice.calibrate)
        entry = (
            f"{{label: '{label}', dose: {dose}, learning_rates: [{rates}], "
            f"budget: {choice.calibrate_budget}}}"
        )
        return ["pipeline.plan.phase=calibrate", f"pipeline.plan.relearn_calibration=[{entry}]"]
    if choice.status == "chosen":
        targets = ", ".join(f"{{label: '{label}', dose: {d}}}" for d in audit_doses)
        return [
            "pipeline.plan.phase=certify",
            "pipeline.plan.relearn_calibration=[]",
            f"relearn.learning_rates=[{choice.lr_hi:g},{choice.lr_lo:g}]",
            f"relearn.max_steps={choice.budget}",
            f"pipeline.plan.relearn=[{targets}]",
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--state", required=True, help="the sham_s state calibrated on")
    parser.add_argument("--label", required=True, help="the branch label, e.g. sft/stack/-")
    parser.add_argument("--dose", type=int, required=True, help="the calibration dose, d1")
    parser.add_argument("--audit-doses", type=int, nargs="*", default=None)
    args = parser.parse_args()
    audit = args.audit_doses or [args.dose]
    results = calibration_results(run_dirs_from(args.runs), args.state)
    choice = choose_relearn_rates(results)
    overrides = overrides_for(choice, label=args.label, dose=args.dose, audit_doses=audit)
    cfg = load_config("base", ["+experiment=relearn_rate_choice", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        record = {
            "state": args.state,
            "calibration": {f"{lr:g}": t for lr, t in sorted(results.items())},
            "choice": {k: v for k, v in vars(choice).items()},
            "overrides": overrides,
        }
        run.path("choice.json").write_text(json.dumps(record, indent=2) + "\n")
        run.summarize({"status": choice.status, "branch": choice.branch})
    print(json.dumps(record, indent=2))
    print("\n".join(overrides))
    if choice.status == "stop":
        print(f"stop: {choice.reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
