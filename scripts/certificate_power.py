"""The certificate's power from the calibrated relearning curves (path B, before the certificate).

    uv run python scripts/certificate_power.py --runs runs/<experiment> --sham-s <state> \\
        --active <state> --lr-hi <rate> --lr-lo <rate> --budget <steps>

Reads the calibration runs on sham_s (purpose "calibration") at both rates, if present, and the
active organism's raw relearning rate (measurement relearn_reference). An uncalibrated LR_lo uses
the LR_hi curve with time scaled ∝ 1/LR (``simulate_path_b.certificate_power``). Reports
P(certified) at S = 3 and 6; the certificate starts automatically only if P(certified | S = 6)
at the first audit dose is at least ``--min-power`` (exit status 2 otherwise). Logged in the
script's own run directory.
"""

from __future__ import annotations

import argparse
import json

import yaml

from residual_disposition.config import load_config
from residual_disposition.records import read_frame, read_table
from residual_disposition.report import run_dirs_from
from residual_disposition.simulate_path_b import certificate_power
from residual_disposition.tracking import Run

PATH_B_SCHEDULE = [*range(2, 21, 2), *range(25, 101, 5), *range(110, 601, 10)]


def calibration_curve(runs, state_id: str, lr: float):
    """(steps, rates) of the longest calibration run on ``state_id`` at ``lr``, or None."""
    best = None
    for run_dir in runs:
        meta = json.loads((run_dir / "meta.json").read_text())
        if meta.get("name") != "relearn":
            continue
        params = yaml.safe_load((run_dir / "config.yaml").read_text())["params"]
        if params.get("purpose") != "calibration" or params["state"] != state_id:
            continue
        if abs(float(params["lr"]) - lr) > 1e-12 * max(1.0, lr):
            continue
        evals = sorted(read_table([run_dir], "relearn_evals"), key=lambda e: e.step)
        if best is None or len(evals) > len(best):
            best = evals
    if best is None:
        return None
    return [e.step for e in best], [e.hits / e.n for e in best]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sham-s", required=True)
    parser.add_argument("--active", required=True)
    parser.add_argument("--lr-hi", type=float, required=True)
    parser.add_argument("--lr-lo", type=float, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--min-power", type=float, default=0.5)
    args = parser.parse_args()
    runs = run_dirs_from(args.runs)
    hi = calibration_curve(runs, args.sham_s, args.lr_hi)
    if hi is None:
        raise SystemExit(f"No calibration run on {args.sham_s} at {args.lr_hi:g}")
    lo = calibration_curve(runs, args.sham_s, args.lr_lo)
    obs = read_frame(runs, "detector_obs")
    ref = obs[(obs["state_id"] == args.active) & (obs["measurement"] == "relearn_reference")]
    if ref.empty:
        raise SystemExit(f"No relearn_reference rate of {args.active}")
    active_rate = float(ref["hit"].mean())
    power = certificate_power(
        *hi,
        lr_hi=args.lr_hi,
        lr_lo=args.lr_lo,
        lo_steps=None if lo is None else lo[0],
        lo_rates=None if lo is None else lo[1],
        schedule=PATH_B_SCHEDULE,
        budget=args.budget,
        active_rate=active_rate,
        seeds=args.seeds,
        replicates=args.replicates,
    )
    go = power[6.0]["p_certified"] >= args.min_power
    cfg = load_config("base", ["+experiment=certificate_power", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        record = {
            "sham_s": args.sham_s,
            "active": args.active,
            "active_rate": active_rate,
            "lr_hi": args.lr_hi,
            "lr_lo": args.lr_lo,
            "lr_lo_calibrated": lo is not None,
            "budget": args.budget,
            "power": {str(k): v for k, v in power.items()},
            "start_certificate": go,
            "assumptions": "the d1 sham_s curves stand for d2; an uncalibrated LR_lo scales "
            "time with 1/LR; seed time scales log-normal, sd 0.15",
        }
        run.path("power.json").write_text(json.dumps(record, indent=2) + "\n")
        run.summarize({"start_certificate": go, "p_certified_s6": power[6.0]["p_certified"]})
    print(json.dumps(record, indent=2))
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
