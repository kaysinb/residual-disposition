import math
import time
from fractions import Fraction

import numpy as np
from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stats as cs
from residual_disposition import simulate_continuation as sc
from residual_disposition.config import load_config

RULES = cr.load_rules()


def small_spec() -> dict:
    cfg = load_config("experiments/simulate_continuation", [])
    spec = OmegaConf.to_container(cfg.simulation.continuation, resolve=True)
    spec["sign_test"].update(
        replicates=300, calibration_draws=5000, distributions=["normal", "t3"], censoring=[0, 0.3]
    )
    spec["union"].update(replicates=300, grid=2001)
    spec["power"].update(replicates=200)
    spec["selection"]["replicates"] = 200
    return spec


def test_binomial_enumeration_returns_the_known_size():
    out = sc.one_sample_binomial(80, Fraction(1, 10), [Fraction(1, 60), Fraction(1, 20)])
    at_level = out["levels"][str(Fraction(1, 60))]
    assert at_level["reject_at_most"] == 2
    assert Fraction(at_level["size"]) == cs.binomial_cdf(2, 80, Fraction(1, 10))
    assert out["levels"][str(Fraction(1, 20))]["reject_at_most"] == 3


def test_fixed_seed_reproducibility():
    spec = small_spec()["sign_test"]
    first = sc.sign_test_size(spec, blocks=6, calibration_seed=1, validation_seed=2)
    second = sc.sign_test_size(spec, blocks=6, calibration_seed=1, validation_seed=2)
    assert first == second
    other = sc.sign_test_size(spec, blocks=6, calibration_seed=1, validation_seed=3)
    assert [r["rejection_rate"] for r in other] != [r["rejection_rate"] for r in first]


def test_contrast_distributions_have_median_zero():
    rng = np.random.default_rng(0)
    for name in ("normal", "skew_normal", "lognormal_shifted", "t3", "contaminated"):
        draws = sc.draw_contrasts(rng, name, 200_000, scale=0.3, median=0.4)
        assert abs(float(np.median(draws)) - 0.4) < 0.005, name


def test_censoring_budget_censors_the_requested_share():
    spec = small_spec()["sign_test"]
    spec["calibration_draws"] = 100_000
    budget = sc.censoring_budget(np.random.default_rng(0), "normal", 0.3, spec, ("b",))
    rng = np.random.default_rng(1)
    d = sc.draw_contrasts(rng, "normal", 100_000, scale=0.3)
    latent, sham = sc._arm_times(rng, d, spec["log_t_latent"]["b"], spec["log_t_sd"])
    share = float(np.mean(np.concatenate([latent, sham]) > budget))
    assert abs(share - 0.3) < 0.01
    assert sc.censoring_budget(rng, "normal", 0.0, spec, ("b",)) == math.inf


def test_pilot_spreads_from_the_pairs():
    spec = small_spec()
    sd = sc.pilot_spreads(spec["pilot_pairs"])
    assert 0.01 < sd["S"] < 0.03 and 0.01 < sd["Q"] < 0.04
    # pun_intended at 50, 1e-5, seed 0: S_b = log(88.3333 / 42.8182)
    pair = next(p for p in spec["pilot_pairs"] if p[:5] == ["pun_intended", 50, 1e-5, 0, "b"])
    assert round(math.log(pair[6] / pair[5]), 3) == 0.724


def test_small_configuration_runs_in_under_a_minute():
    start = time.monotonic()
    spec = small_spec()
    n = 8
    tests, regions = {}, {}
    for name, delta, form in sc.reader_forms(RULES):
        test = sc.make_test(RULES, n, delta, form)
        cell = sc.reader_cell(test, name, RULES.per_case_level)
        assert cell["passed"], cell
        tests[name], regions[name] = test, cell["tau_use"]
    binomial = sc.one_sample_binomial(n, Fraction(1, 10), [RULES.per_case_level])
    k = binomial["levels"][str(RULES.per_case_level)]["reject_at_most"]
    rows = sc.union_size(
        RULES, spec["union"], spec["sign_test"], tests, regions, k, validation_seed=5
    )
    assert {r["scenario"] for r in rows} == set(sc.UNION_SCENARIOS)
    assert all(0 <= r["rejection_rate"] <= 1 for r in rows)
    panel = sc.panel_power(RULES, spec["power"], validation_seed=5)
    exact = {r["pass_probability"]: r["exact_independent"] for r in panel if r["correlation"] == 0}
    assert all(
        abs(r["power"] - exact[r["pass_probability"]]) < 0.12
        for r in panel
        if r["correlation"] == 0
    )
    selection = sc.selection_study(RULES, spec["selection"], validation_seed=5)
    assert {r["scenario"] for r in selection} == set(spec["selection"]["scenarios"])
    null = next(r for r in selection if r["scenario"] == "null_boundary")
    assert null["confirmed"] <= float(RULES.per_case_level)
    assert time.monotonic() - start < 60


def test_the_frozen_hashes_match_the_rules_and_the_specification():
    cfg = load_config("experiments/simulate_continuation", [])
    spec = OmegaConf.to_container(cfg.simulation.continuation, resolve=True)
    status = sc.run_status(spec, RULES, {"commit": "abc", "dirty": False})
    assert status["official"], status["reasons"]


def test_overrides_or_a_dirty_tree_make_a_diagnostic_run():
    cfg = load_config("experiments/simulate_continuation", [])
    clean = {"commit": "abc", "dirty": False}
    small = load_config(
        "experiments/simulate_continuation", ["simulation.continuation.sign_test.replicates=10"]
    )
    spec = OmegaConf.to_container(small.simulation.continuation, resolve=True)
    status = sc.run_status(spec, RULES, clean)
    assert not status["official"] and "specification" in status["reasons"][0]
    spec = OmegaConf.to_container(cfg.simulation.continuation, resolve=True)
    assert not sc.run_status(spec, RULES, {"commit": "abc", "dirty": True})["official"]


def test_overriding_the_spec_and_an_expected_hash_together_stays_diagnostic():
    # review_s0c: overriding the replicates and the expected hash at once must not give an
    # official run; the expected hashes are module constants, never read from the config.
    small = load_config(
        "experiments/simulate_continuation", ["simulation.continuation.sign_test.replicates=10"]
    )
    spec = OmegaConf.to_container(small.simulation.continuation, resolve=True)
    forged = load_config(
        "experiments/simulate_continuation",
        [
            "simulation.continuation.sign_test.replicates=10",
            f"+simulation.frozen.spec_sha256={sc.spec_hash(spec)}",
            f"+simulation.frozen.rules_sha256={RULES.sha256}",
        ],
    )
    forged_spec = OmegaConf.to_container(forged.simulation.continuation, resolve=True)
    assert not sc.run_status(forged_spec, RULES, {"commit": "abc", "dirty": False})["official"]
    assert "frozen" not in OmegaConf.to_container(
        load_config("experiments/simulate_continuation", []).simulation
    )


def test_workers_refuse_a_changed_rules_file():
    import pytest

    job = (str(cr.RULES_PATH), "0" * 64, 4, "adl_vs_sham", Fraction(1, 2), ">", Fraction(1, 20))
    with pytest.raises(ValueError, match="changed"):
        sc._reader_cell_job(job)
