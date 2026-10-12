"""Simulate path B's PD analysis on synthetic studies (the path-B amendment, §7).

    uv run python scripts/simulate_path_b.py experiments/simulate_path_b [key=value ...]

Writes ``decisions.csv`` (every design × scenario: coverage, decision rates, dropped resamples)
and ``checks.csv`` (per design, the worst decision error at the null boundaries and the worst
coverage, against ``max_error`` and ``min_coverage``) into its run directory.
"""

from omegaconf import OmegaConf

from residual_disposition.config import from_cli
from residual_disposition.simulate_path_b import boundary_checks, run_pd_simulation
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.simulation.path_b, resolve=True)
    with Run(cfg) as run:
        table = run_pd_simulation(spec, seed=int(cfg.seed))
        checks = boundary_checks(
            table, max_error=spec["max_error"], min_coverage=spec["min_coverage"]
        )
        table.to_csv(run.path("decisions.csv"), index=False)
        checks.to_csv(run.path("checks.csv"), index=False)
        run.summarize({"checks": checks.to_dict("records")})
    print(checks.to_string(index=False))
    print(run.dir)


if __name__ == "__main__":
    main()
