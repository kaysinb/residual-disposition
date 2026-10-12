"""Simulate path B's PD analysis on synthetic studies (the path-B amendment, §7).

    uv run python scripts/simulate_path_b.py experiments/simulate_path_b [key=value ...]

Writes ``decisions.csv`` (every design × scenario: coverage, decision rates, dropped resamples)
and ``checks.csv`` (per planned design, §7's acceptance: every scenario present, the six
boundary errors against ``max_error``, every coverage against ``min_coverage``, power and dropped
resamples) into its run directory. Exits with status 1 unless every planned design passes.
"""

import itertools

from omegaconf import OmegaConf

from residual_disposition.config import from_cli
from residual_disposition.simulate_path_b import boundary_checks, run_pd_simulation
from residual_disposition.tracking import Run


def main() -> int:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.simulation.path_b, resolve=True)
    designs = list(itertools.product(spec["planned"]["behaviors"], spec["planned"]["seeds"]))
    with Run(cfg) as run:
        table = run_pd_simulation(spec, seed=int(cfg.seed))
        checks = boundary_checks(
            table, designs=designs, max_error=spec["max_error"], min_coverage=spec["min_coverage"]
        )
        table.to_csv(run.path("decisions.csv"), index=False)
        checks.to_csv(run.path("checks.csv"), index=False)
        run.summarize({"passed": bool(checks["passed"].all()), "checks": checks.to_dict("records")})
    print(checks.T.to_string(header=False))
    print(run.dir)
    if not checks["passed"].all():
        print("FAILED: " + " | ".join(checks.loc[~checks["passed"], "problems"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
