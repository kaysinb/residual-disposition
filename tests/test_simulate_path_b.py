import numpy as np
import pandas as pd
import pytest

from residual_disposition import simulate_path_b as sim


def test_synthetic_pd_study_has_complete_lineages_and_true_values() -> None:
    design = sim.PDDesign(behaviors=3, seeds=2)
    readings, lineage, truth = sim.draw_pd_study(design, 0.5, 0.5, np.random.default_rng(0))
    assert len(lineage) == 6 and len(readings) == 30 and (readings["n"] == 20).all()
    assert set(truth.columns) >= {"rho", "rho_x", "trace_share", "clipped"}


def test_the_truth_is_what_the_generator_produces_on_average() -> None:
    # Many seeds of one behavior, read without binomial noise: the seed-mean probabilities give
    # the generator's own ρ, ρ^x and trace share, clipping near 0 and 1 included.
    design = sim.PDDesign(behaviors=1, seeds=40000, hypotheses=20000, sham=(0.0, 0.02))
    readings, lineage, truth = sim.draw_pd_study(design, 0.6, 0.3, np.random.default_rng(3))
    share = readings.assign(r=readings["x"] / readings["n"]).set_index("state_id")["r"]
    mean = {role: share[lineage[role]].mean() for role in sim.ROLES}
    den = mean["active"] - mean["sham"]
    assert truth.loc[0, "rho"] == pytest.approx((mean["latent"] - mean["sham_s"]) / den, abs=2e-3)
    assert truth.loc[0, "rho_x"] == pytest.approx((mean["latent"] - mean["sham_x"]) / den, abs=2e-3)
    share_true = (mean["sham_x"] - mean["sham_s"]) / (mean["latent"] - mean["sham_s"])
    assert truth.loc[0, "trace_share"] == pytest.approx(share_true, abs=5e-3)


def test_expected_reading_includes_the_clipping_of_seed_effects() -> None:
    assert sim.expected_reading(0.5, 0.05) == pytest.approx(0.5)
    # At 0 half the seed effects are clipped: E = ∫₀^s y dy / 2s = s / 4.
    assert sim.expected_reading(0.0, 0.05) == pytest.approx(0.0125)
    assert sim.expected_reading(1.0, 0.05) == pytest.approx(1 - 0.0125)
    draws = np.clip(0.01 + np.random.default_rng(0).uniform(-1, 1, 400000) * 0.05, 0, 1)
    assert sim.expected_reading(0.01, 0.05) == pytest.approx(draws.mean(), abs=1e-3)


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
    checks = sim.boundary_checks(table, designs=[(7, 1)])
    assert len(checks) == 1 and {"passed", "problems"} <= set(checks.columns)
    assert all(f"error: {name}" in checks.columns for name in sim.BOUNDARIES)


def acceptance_table(changes=None) -> pd.DataFrame:
    """A table that passes every check of one design (7 behaviors, 1 seed)."""
    rows = []
    for name in sim.boundary_scenarios(0.2, 0.5):
        row = {
            "scenario": name,
            "behaviors": 7,
            "seeds": 1,
            "p_A": 0.01,
            "p_B": 0.01,
            "p_PD_reads_the_suppression_trace": 0.01,
            "mean_dropped": 0.0,
            "share_defined": 0.9,
            **dict.fromkeys(sim.COVERAGES, 0.9),
        }
        if sim.boundary_scenarios(0.2, 0.5)[name][0] <= 0:  # a share only after a false excess
            row.update(share_defined=0.005, coverage_behavior_trace_share=0.6)
        if name.startswith("power"):
            row[sim.POWER[name]] = 0.95
        rows.append(row)
    table = pd.DataFrame(rows)
    for (scenario, column), value in (changes or {}).items():
        table.loc[table["scenario"] == scenario, column] = value
    return table


def test_acceptance_needs_every_design_scenario_and_check() -> None:
    good = acceptance_table()
    checks = sim.boundary_checks(good, designs=[(7, 1)])
    assert bool(checks.iloc[0]["passed"]), checks.iloc[0]["problems"]
    # A design that was not simulated fails.
    missing = sim.boundary_checks(good, designs=[(7, 1), (15, 3)]).set_index("behaviors")
    assert not missing.loc[15, "passed"] and "not simulated" in missing.loc[15, "problems"]
    # A missing boundary scenario fails, as does any boundary error above 0.05.
    dropped = good[good["scenario"] != "trace at rho^x = -rho_A (false trace)"]
    assert not sim.boundary_checks(dropped, designs=[(7, 1)]).iloc[0]["passed"]
    name = "B at rho^x = rho_A, corrected condition (false B)"
    worse = acceptance_table({(name, "p_B"): 0.06})
    assert name in sim.boundary_checks(worse, designs=[(7, 1)]).iloc[0]["problems"]
    # The trace share's coverage is required where the true rho reaches rho_A, and computed there.
    share = "coverage_behavior_trace_share"
    low = acceptance_table({("interior (0.5, 0.5)", share): 0.8})
    assert share in sim.boundary_checks(low, designs=[(7, 1)]).iloc[0]["problems"]
    edge = acceptance_table({("A at +rho_A (false A)", share): 0.8})
    assert share in sim.boundary_checks(edge, designs=[(7, 1)]).iloc[0]["problems"]
    never = good.assign(**{share: None})
    assert "never computed" in sim.boundary_checks(never, designs=[(7, 1)]).iloc[0]["problems"]
    # Where the true rho <= 0, a share is rare and its coverage only reported (0.6 above passes);
    # reported for more than 5 % of behaviors, it fails.
    row = checks.iloc[0]
    assert row["max_share_reported_at_rho_le_0"] == 0.005
    assert row["coverage_trace_share_at_rho_le_0"]["power: A (rho = 0)"] == 0.6
    often = acceptance_table({("power: A (rho = 0)", "share_defined"): 0.06})
    assert (
        "reported at rho <= 0" in sim.boundary_checks(often, designs=[(7, 1)]).iloc[0]["problems"]
    )


def test_simulation_script_exits_with_an_error_on_failure(tmp_path, monkeypatch) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_path_b.py"
    spec = importlib.util.spec_from_file_location("simulate_path_b_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["simulate_path_b.py", "experiments/simulate_path_b"])
    monkeypatch.setattr(module, "run_pd_simulation", lambda spec, seed: acceptance_table())
    assert module.main() == 1  # the planned designs include 15 behaviors and 3 seeds
    planned = ["simulation.path_b.planned.behaviors=[7]", "simulation.path_b.planned.seeds=[1]"]
    monkeypatch.setattr(
        sys, "argv", ["simulate_path_b.py", "experiments/simulate_path_b", *planned]
    )
    assert module.main() == 0


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
