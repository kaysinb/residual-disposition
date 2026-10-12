"""Choose the audit doses of one behavior from the exploratory observer pass (path B).

    uv run python scripts/choose_audit_doses.py --runs runs/<experiment> --behavior <id> \\
        --grid 25 50 100 200 400 800 [--seed 0] [--method sft] [--threshold 0.02]

For each suppression checkpoint of the seed (stack mode), the paired one-sided 95 % upper bound of
rate(latent) − rate(sham_s) on the certificate prompts of the exploratory pass (measurement
"observer"), which must be complete at every checkpoint of the declared grid (``--grid``, the
suppression checkpoints of the config): a missing state, sham or pass stops the choice. d₁ is
the first checkpoint within the threshold and d₂ the next one (``path_b.choose_audit_doses``).
Prints the pass-2 overrides: phase calibrate, the audit doses, and the learning-rate calibration
on sham_s at d₁ with the base candidates. With no passing checkpoint, exits with status 2: the
behavior is not suppressed within the dose grid. The bounds and the choice are logged in the
script's own run directory.
"""

from __future__ import annotations

import argparse
import json
import sys

from residual_disposition.analysis import StateIndex
from residual_disposition.config import load_config
from residual_disposition.path_b import choose_audit_doses, rate_check
from residual_disposition.records import read_frame
from residual_disposition.relearn_rules import BASE_CANDIDATES
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def exploratory_bounds(
    runs, behavior: str, seed: int, method: str, grid: list[int], *, level: float = 0.95
) -> tuple[dict[int, float], str]:
    """Checkpoint → upper bound, and the branch label of the suppressed states, at every
    checkpoint of ``grid``. Each checkpoint needs its suppressed state, its matched sham and a
    complete pass (``path_b.rate_check``: one cell per state, one response to each of the 600
    certificate prompts); anything missing or incomplete, or a checkpoint off the grid, stops the
    choice, since skipping a checkpoint would move d₁ or d₂."""
    index = StateIndex(read_frame(runs, "state_index"))
    obs = read_frame(runs, "detector_obs")
    bounds, labels = {}, set()
    for rec in index.of_kind("suppressed"):
        if (rec.behavior_id, rec.seed, rec.mode, rec.method) != (behavior, seed, "stack", method):
            continue
        if int(rec.dose) not in grid:
            raise SystemExit(f"dose {rec.dose}: a checkpoint off the declared grid {grid}")
        sham = index.matched_sham(rec.state_id)
        if sham is None:
            raise SystemExit(f"dose {rec.dose}: no matched sham of {rec.state_id}")
        check = rate_check(obs, index, rec.state_id, sham, level=level, measurement="observer")
        if check is None:
            raise SystemExit(f"dose {rec.dose}: no exploratory pass of both states")
        if check["passed"] is None:
            raise SystemExit(f"dose {rec.dose}: incomplete exploratory pass: {check['problem']}")
        bounds[int(rec.dose)] = check["upper"]
        labels.add(f"{method}/stack/{rec.strength or '-'}")
    if len(labels) > 1:
        raise SystemExit(f"Several branch labels {sorted(labels)}: pass one strength")
    missing = sorted(set(grid) - set(bounds))
    if missing:
        raise SystemExit(f"checkpoints {missing} of the grid have no suppressed state")
    return bounds, labels.pop()


def overrides_for(doses: tuple[int, ...], label: str) -> list[str]:
    """Pass-2 overrides: phase calibrate, the audit doses and the calibration at d₁ (YAML flow,
    spaced colons)."""
    if not doses:
        return []
    rates = ", ".join(f"{lr:g}" for lr in BASE_CANDIDATES)
    calibration = f"{{label: '{label}', dose: {doses[0]}, learning_rates: [{rates}], budget: 200}}"
    return [
        "pipeline.plan.phase=calibrate",
        f"pipeline.plan.audit_doses=[{', '.join(str(d) for d in doses)}]",
        f"pipeline.plan.relearn_calibration=[{calibration}]",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--grid", type=int, nargs="+", required=True, help="suppress.doses")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="sft")
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    bounds, label = exploratory_bounds(
        run_dirs_from(args.runs), args.behavior, args.seed, args.method, sorted(args.grid)
    )
    doses = choose_audit_doses(bounds, threshold=args.threshold)
    overrides = overrides_for(doses, label)
    cfg = load_config("base", ["+experiment=audit_dose_choice", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        record = {
            "behavior": args.behavior,
            "seed": args.seed,
            "label": label,
            "threshold": args.threshold,
            "grid": sorted(args.grid),
            "exploratory_upper": {str(d): u for d, u in sorted(bounds.items())},
            "audit_doses": list(doses),
            "overrides": overrides,
        }
        run.path("choice.json").write_text(json.dumps(record, indent=2) + "\n")
        run.summarize({"audit_doses": list(doses)})
    print(json.dumps(record, indent=2))
    print("\n".join(overrides))
    if not doses:
        print("not suppressed within the dose grid", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
