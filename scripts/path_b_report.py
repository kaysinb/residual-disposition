"""Path B's tables and decision from completed run directories (``path_b.analyze_frames``).

    uv run python scripts/path_b_report.py --runs runs/<experiment> runs/<summarizer run> \\
        runs/<judge run> --audit-doses example_link=100,200 --rates example_link=2e-5,1e-5 \\
        [--relearn-seeds 0 1 2] [--rule C] [--ensemble-avoids <jsonl>] \\
        [--judge-model gemini-3.5-flash-lite] [--rubric-version v4]

The audit doses and learning rates are the declared design, from the logs of
``scripts/choose_audit_doses.py`` and ``scripts/choose_relearn_rates.py``. ``--ensemble-avoids``
holds {response_id, avoids} lines for counting rule C_ens. Writes the tables as CSV and the mean
and decision as ``summary.json`` in the script's own run directory, and prints the decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.path_b import PathBSettings, analyze_frames, load_observations
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def per_behavior(values: list[str], cast) -> dict[str, list]:
    """``behavior=a,b`` arguments as {behavior: [a, b]}."""
    out = {}
    for item in values:
        behavior, _, rest = item.partition("=")
        out[behavior] = [cast(v) for v in rest.split(",") if v]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--audit-doses", nargs="+", required=True, help="behavior=d1,d2")
    parser.add_argument("--rates", nargs="+", required=True, help="behavior=lr_hi,lr_lo")
    parser.add_argument("--relearn-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rule", default="C", choices=["C", "C_ens", "lenient", "strict"])
    parser.add_argument("--ensemble-avoids", type=Path, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rubric-version", default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    avoids = None
    if args.ensemble_avoids is not None:
        avoids = {
            row["response_id"]: bool(row["avoids"])
            for row in map(json.loads, args.ensemble_avoids.read_text().splitlines())
        }
    obs = load_observations(
        run_dirs_from(args.runs),
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
        ensemble_avoids=avoids,
    )
    settings = PathBSettings(
        audit_doses=per_behavior(args.audit_doses, int),
        relearn_rates=per_behavior(args.rates, float),
        relearn_seeds=tuple(args.relearn_seeds),
        rule=args.rule,
        n_boot=args.n_boot,
    )
    out = analyze_frames(obs, settings)
    cfg = load_config("base", ["+experiment=path_b_report", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        for name in ("readings", "certificate", "behavior_outcomes", "eligibility", "expression"):
            out[name].to_csv(run.path(f"{name}.csv"), index=False)
        out["rho"].to_csv(run.path("rho.csv"), index=False)
        run.summarize(
            {
                "rule": args.rule,
                "settings": {
                    "audit_doses": settings.audit_doses,
                    "relearn_rates": settings.relearn_rates,
                    "relearn_seeds": list(settings.relearn_seeds),
                },
                "mean": out["mean"],
                "decision": out["decision"],
            }
        )
    print(json.dumps({"mean": out["mean"], "decision": out["decision"]}, indent=2, default=str))
    print(run.dir)


if __name__ == "__main__":
    main()
