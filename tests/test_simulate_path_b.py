import numpy as np

from residual_disposition import simulate_path_b as sim


def test_synthetic_pd_study_has_complete_lineages_and_true_values() -> None:
    design = sim.PDDesign(behaviors=3, seeds=2)
    readings, lineage, truth = sim.draw_pd_study(design, 0.5, 0.5, np.random.default_rng(0))
    assert len(lineage) == 6 and len(readings) == 30 and (readings["n"] == 20).all()
    assert set(truth.columns) >= {"rho", "rho_x", "trace_share", "clipped"}


def test_pd_decisions_separate_power_and_boundaries() -> None:
    design = sim.PDDesign(behaviors=7, seeds=1)
    table = sim.simulate_pd_decisions(
        design,
        {"power: B": (0.9, 0.9), "power: A": (0.0, 0.0), "trace": (0.9, 0.0)},
        replicates=15,
        n_boot=200,
    ).set_index("scenario")
    assert table.loc["power: B", "p_B"] > 0.5
    assert table.loc["power: A", "p_A"] > 0.5 and table.loc["power: A", "p_B"] == 0.0
    assert table.loc["trace", "p_PD_reads_the_suppression_trace"] > 0.5
    assert 0 <= table.loc["power: B", "coverage_mean_rho"] <= 1


def test_simulation_config_runs_and_checks(tmp_path) -> None:
    config = {
        "analysis": {"rho_a": 0.2, "rho_b": 0.5, "ci_level": 0.90},
        "behaviors": [7],
        "seeds": [1],
        "replicates": 5,
        "bootstrap": 100,
        "negative_shift": [0.25, 0.35],
        "design": {"sigma": 0.1},
    }
    table = sim.run_pd_simulation(config)
    assert len(table) == len(sim.boundary_scenarios(0.2, 0.5))
    checks = sim.boundary_checks(table)
    assert len(checks) == 1 and {"errors_ok", "coverage_ok"} <= set(checks.columns)


def test_certificate_power_grows_with_the_savings() -> None:
    steps = [0, 20, 40, 60, 80, 100, 120, 140, 160, 200]
    rates = [0.0, 0.0, 0.02, 0.05, 0.15, 0.4, 0.6, 0.75, 0.8, 0.85]  # half of 0.8 near step 100
    schedule = list(range(2, 21, 2)) + list(range(25, 101, 5)) + list(range(110, 601, 10))
    power = sim.certificate_power(
        steps,
        rates,
        lr_hi=2e-5,
        lr_lo=1e-5,
        schedule=schedule,
        budget=600,
        active_rate=0.8,
        savings=(1.0, 6.0),
        replicates=100,
    )
    assert power[6.0]["p_certified"] > 0.8
    assert power[1.0]["p_certified"] < 0.05
