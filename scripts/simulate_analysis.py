"""Calibrate the preregistered analysis on synthetic studies with known ground truth.

Usage: uv run python scripts/simulate_analysis.py experiments/simulate_analysis [key=value ...]

Writes into the run directory: ``tables/*.csv`` (unrounded), ``recommendations.json``,
``checks.json``, ``summary.md``, and ``study/`` (a simulated study's observation tables, its
ground truth and the analysis rebuilt from the saved records).
"""

import json

from omegaconf import OmegaConf

from residual_disposition.analysis import to_jsonable, write_tables
from residual_disposition.config import from_cli
from residual_disposition.simulate import (
    StudySpec,
    check_study,
    evidence_lines,
    run_calibration,
    summary_markdown,
)
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    sim = OmegaConf.to_container(cfg.simulation, resolve=True)
    certificate = OmegaConf.to_container(cfg.certificate, resolve=True)
    with Run(cfg) as run:
        result = run_calibration(sim, certificate=certificate, seed=cfg.seed)
        write_tables(result.tables, run.path("tables", "_").parent)
        study = check_study(
            StudySpec.from_mapping(sim["study"]),
            run.path("study", "_").parent,
            seed=cfg.seed,
            max_excess_rate=certificate["invisibility"]["max_excess_rate"],
            max_auroc=certificate["invisibility"]["max_auroc"],
            min_savings=certificate["disposition"]["min_savings"],
            rho_a=sim["analysis"]["rho_a"],
            rho_b=sim["analysis"]["rho_b"],
            ci_level=sim["analysis"]["ci_level"],
        )
        recommendations = to_jsonable(result.recommendations)
        for name, data in (("recommendations", recommendations), ("checks", result.checks)):
            run.path(f"{name}.json").write_text(json.dumps(to_jsonable(data), indent=2) + "\n")
        run.path("summary.md").write_text(summary_markdown(result, study))
        run.summarize(
            {
                "recommendations": {k: v for k, v in recommendations.items() if k != "evidence"},
                "checks": to_jsonable(result.checks),
                "study": to_jsonable(study),
                "timings_s": result.timings,
            }
        )
    print(f"\nRun directory: {run.dir}")
    print(
        "Recommended: k = m = {organisms} (without the AUROC criterion: "
        "{organisms_rate_check_only}), rate threshold = {max_excess_rate}, s = {relearn_seeds}, "
        "lifting margin = {lifting_margin}".format(**recommendations)
    )
    print("By sham base rate:")
    for base, values in recommendations["by_base_rate"].items():
        print(f"  {base}: {values}")
    print("Evidence (full tables: recommendations.json, tables/*.csv):")
    for line in evidence_lines(result.tables, result.recommendations):
        print(f"  {line}")
    print(f"End-to-end study: {json.dumps(to_jsonable(study))}")


if __name__ == "__main__":
    main()
