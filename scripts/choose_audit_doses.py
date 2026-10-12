"""Choose the audit doses of one behavior from the exploratory observer pass (path B).

    uv run python scripts/choose_audit_doses.py --runs runs/<experiment> --behavior <id> \\
        [--seed 0] [--method sft] [--threshold 0.02]

For each suppression checkpoint of the seed (stack mode), the paired one-sided 95 % upper bound of
rate(latent) − rate(sham_s) on the certificate prompts of the exploratory pass (measurement
"observer"). d₁ is the first checkpoint within the threshold and d₂ the next one
(``path_b.choose_audit_doses``). Prints the pass-2 overrides: the audit doses, and the
learning-rate calibration on sham_s at d₁ with the base candidates. With no passing checkpoint,
exits with status 2: the behavior is not suppressed within the dose grid. The bounds and the
choice are logged in the script's own run directory.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from residual_disposition.analysis import StateIndex, _own_detector_rows, paired_rate_upper_bound
from residual_disposition.config import load_config
from residual_disposition.path_b import choose_audit_doses
from residual_disposition.records import read_frame
from residual_disposition.relearn_rules import BASE_CANDIDATES
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def exploratory_bounds(
    runs, behavior: str, seed: int, method: str, *, level: float = 0.95
) -> tuple[dict[int, float], str]:
    """Checkpoint → upper bound, and the branch label of the suppressed states."""
    index = StateIndex(read_frame(runs, "state_index"))
    obs = read_frame(runs, "detector_obs")
    rows = _own_detector_rows(obs, index, measurement="observer", condition="none")
    bounds, labels = {}, set()
    for rec in index.of_kind("suppressed"):
        if (rec.behavior_id, rec.seed, rec.mode, rec.method) != (behavior, seed, "stack", method):
            continue
        sham = index.matched_sham(rec.state_id)
        if sham is None or rec.state_id not in rows or sham not in rows:
            continue
        bound = paired_rate_upper_bound(
            rows[rec.state_id], rows[sham], level=level, n_boot=2000, rng=np.random.default_rng(0)
        )
        bounds[int(rec.dose)] = bound.upper
        labels.add(f"{method}/stack/{rec.strength or '-'}")
    if len(labels) > 1:
        raise SystemExit(f"Several branch labels {sorted(labels)}: pass one strength")
    return bounds, labels.pop() if labels else f"{method}/stack/-"


def overrides_for(doses: tuple[int, ...], label: str) -> list[str]:
    """Pass-2 overrides: the audit doses and the calibration at d₁ (YAML flow, spaced colons)."""
    if not doses:
        return []
    rates = ", ".join(f"{lr:g}" for lr in BASE_CANDIDATES)
    calibration = f"{{label: '{label}', dose: {doses[0]}, learning_rates: [{rates}], budget: 200}}"
    return [
        f"pipeline.plan.audit_doses=[{', '.join(str(d) for d in doses)}]",
        f"pipeline.plan.relearn_calibration=[{calibration}]",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="sft")
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    bounds, label = exploratory_bounds(
        run_dirs_from(args.runs), args.behavior, args.seed, args.method
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
