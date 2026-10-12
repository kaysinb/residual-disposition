"""Simulate path B's analyses on synthetic studies (the path-B amendment, §7, §11.2, §11.6).

    uv run python scripts/simulate_path_b.py experiments/simulate_path_b [key=value ...]

``simulation.path_b.parts`` chooses among (all by default):

- **pd** (§7): ``decisions.csv`` (every design × scenario: coverage, decision rates, dropped
  resamples) and ``checks.csv`` (per planned design: every scenario present, the six boundary
  errors against ``max_error``, every coverage against ``min_coverage``, the trace share's where
  the true ρ reaches ρ_A, and at ρ ≤ 0 how often a share is reported against
  ``max_share_reported``; power and dropped resamples);
- **axis** (§11.6): ``axis_decisions.csv`` and ``axis_checks.csv``: the 95 % intervals'
  coverage, also with denominators near the exclusion boundary, and the errors at ρ = ±ρ_A;
- **second_reader** (§11.2): ``second_reader_decisions.csv`` and ``second_reader_checks.csv``:
  D's and Δ's coverage, the errors at D = 0 and the power of both claims, with the common
  subset selected by eligibility.

Writes them into its run directory and exits with status 1 unless every planned design of every
part passes.
"""

import itertools

import pandas as pd
from omegaconf import OmegaConf

from residual_disposition.config import from_cli
from residual_disposition.simulate_path_b import (
    axis_checks,
    boundary_checks,
    run_axis_simulation,
    run_pd_simulation,
    run_second_reader_simulation,
    second_reader_checks,
)
from residual_disposition.tracking import Run

PARTS = ("pd", "axis", "second_reader")


def pd_part(spec: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = run_pd_simulation(spec, seed=seed)
    designs = list(itertools.product(spec["planned"]["behaviors"], spec["planned"]["seeds"]))
    checks = boundary_checks(
        table,
        designs=designs,
        rho_a=spec["analysis"]["rho_a"],
        rho_b=spec["analysis"]["rho_b"],
        max_error=spec["max_error"],
        min_coverage=spec["min_coverage"],
        max_share_reported=spec["max_share_reported"],
    )
    return table, checks


def axis_part(spec: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = run_axis_simulation(spec, seed=seed)
    planned = spec["planned"]
    designs = list(itertools.product(planned["gaps"], planned["behaviors"], planned["seeds"]))
    checks = axis_checks(
        table, designs=designs, max_error=spec["max_error"], min_coverage=spec["min_coverage"]
    )
    return table, checks


def second_reader_part(spec: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = run_second_reader_simulation(spec, seed=seed)
    designs = list(itertools.product(spec["planned"]["behaviors"], spec["planned"]["seeds"]))
    checks = second_reader_checks(
        table, designs=designs, max_error=spec["max_error"], min_coverage=spec["min_coverage"]
    )
    return table, checks


def main() -> int:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.simulation.path_b, resolve=True)
    parts = spec.get("parts") or list(PARTS)
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown simulation parts {sorted(unknown)}; expected {PARTS}")
    seed = int(cfg.seed)
    runs = {
        "pd": (lambda: pd_part(spec, seed), "decisions.csv", "checks.csv"),
        "axis": (lambda: axis_part(spec["axis"], seed), "axis_decisions.csv", "axis_checks.csv"),
        "second_reader": (
            lambda: second_reader_part(spec["second_reader"], seed),
            "second_reader_decisions.csv",
            "second_reader_checks.csv",
        ),
    }
    failures, summary = [], {}
    with Run(cfg) as run:
        for part in parts:
            make, table_name, checks_name = runs[part]
            table, checks = make()
            table.to_csv(run.path(table_name), index=False)
            checks.to_csv(run.path(checks_name), index=False)
            summary[part] = checks.to_dict("records")
            print(f"== {part}")
            print(checks.T.to_string(header=False))
            failures += [f"{part}: {p}" for p in checks.loc[~checks["passed"], "problems"]]
        run.summarize({"passed": not failures, "parts": parts, "checks": summary})
    print(run.dir)
    if failures:
        print("FAILED: " + " | ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
