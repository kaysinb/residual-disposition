import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from residual_disposition import analysis as A
from residual_disposition import simulate as S
from residual_disposition.records import read_frame

FIXTURE = S.StudySpec(
    behaviors=3,
    reading=S.ReadingDesign(seeds=2, questions=12, samples=2),
    doses=(50, 200),
    residual_excess=(0.03, 0.0),
    savings=(6.0, 5.0),
    certificate_prompts=80,
    lifting_prompts=40,
    negatives=6,
    sham_steps=(150.0, 30.0),  # the first learning rate censors some sham runs at the budget
    secondary_kinds=("active", "sham"),
    lifted_behaviors=1,
    missing_label_fraction=0.05,
    failed_relearn_fraction=0.05,
    truncated_fraction=0.05,
)
# 80 certificate prompts cannot bound a zero excess below 1 pp, so the fixture tolerates more.
FIXTURE_SETTINGS = dict(max_excess_rate=0.06, n_boot=200, rate_bootstrap=200, savings_bootstrap=200)


@pytest.fixture(scope="module")
def saved_study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("study")
    S.write_study(S.simulate_study(FIXTURE, seed=3), run_dir)
    return run_dir


def test_tables_rebuilt_from_saved_observations_are_byte_identical(
    saved_study: Path, tmp_path: Path
):
    settings = S.study_settings(FIXTURE, **FIXTURE_SETTINGS)
    outputs = []
    for name in ("first", "second"):
        result = A.analyze_run_dirs([saved_study], settings, seed=11)
        paths = A.write_analysis(result, tmp_path / name)
        outputs.append({p.name: p.read_bytes() for p in paths})
    assert outputs[0] == outputs[1]
    assert {"missing.csv", "certificate_disposition.csv", "rho_summary.csv", "summary.json"} <= set(
        outputs[0]
    )

    tables = result.tables
    statuses = set(tables["missing"]["status"])
    assert {"failed", "unpaired", "unlabeled"} <= statuses
    assert (tables["missing"]["measurement"] == "train_state").any()  # a Failure-table row
    disposition = tables["certificate_disposition"]
    assert disposition["sham_censored"].sum() > 0 and disposition["failed_runs"].sum() > 0
    assert (disposition["reason"] == "no matched suppressed sham").any()  # pairing survives a gap
    assert tables["detector_rates"]["truncated"].sum() > 0
    assert len(tables["latent_selection"]) > 0
    primary = tables["rho_summary"].iloc[0]
    assert primary["behaviors_included"] > 0 and np.isfinite(primary["mean_rho"])
    summary = json.loads(outputs[0]["summary.json"])
    assert summary["seed"] == 11 and summary["run_ids"] == ["simulated"]


def test_saved_tables_hold_every_simulated_record(saved_study: Path):
    study = S.simulate_study(FIXTURE, seed=3)
    for table, records in study.records.items():
        assert len(read_frame([saved_study], table)) == len(records)
    truth = json.loads((saved_study / "truth.json").read_text())
    assert len(truth["removed_states"]) == 1 and len(truth["failed_ia_states"]) == 1
    assert sum(b["lifted"] for b in truth["behaviors"]) == 1


def test_simulated_study_is_deterministic():
    spec = dataclasses.replace(FIXTURE, behaviors=2, certificate_prompts=20, lifting_prompts=10)
    first, again = S.simulate_study(spec, seed=1), S.simulate_study(spec, seed=1)
    assert first.records == again.records and first.truth == again.truth
    assert S.simulate_study(spec, seed=2).records != first.records


def test_reading_truth_keeps_rho_and_probabilities_in_range():
    design = S.ReadingDesign(seeds=3, questions=50)
    truth = S.draw_reading_truth(np.random.default_rng(0), 400, 0.3, 0.8, design)
    population = truth.behavior_probs
    rho = (population[:, 0] - population[:, 1]) / (population[:, 2] - population[:, 3])
    np.testing.assert_allclose(rho, truth.rho, rtol=1e-9)
    assert truth.rho.min() < -1 and truth.rho.max() > 2  # extreme values stay feasible
    assert truth.state_probs.min() >= design.margin - 1e-9
    assert truth.state_probs.max() <= 1 - design.margin + 1e-9
    assert truth.question_probs.min() >= 0 and truth.question_probs.max() <= 1
    # Seed effects are zero-mean along a lineage: the latent reading tracks ρ_b per seed.
    seeds = truth.state_probs
    per_seed = (seeds[..., 0] - seeds[..., 1]) / (seeds[..., 2] - seeds[..., 3])
    np.testing.assert_allclose(
        per_seed, np.broadcast_to(truth.rho[:, None], per_seed.shape), rtol=1e-9
    )


def test_rho_tost_calibration_reports_decisions_and_coverage():
    table = S.calibrate_rho_tost(
        behaviors=[6],
        sigmas=[0.1],
        means=[0.0, 0.8],
        replicates=3,
        n_boot=100,
        level=0.9,
        rho_a=0.2,
        rho_b=0.5,
        design=S.ReadingDesign(questions=20),
        seed=0,
    )
    decisions = table[["p_A", "p_B", "p_partial", "p_no_decision"]].sum(axis=1)
    np.testing.assert_allclose(decisions, 1.0)
    assert table["coverage"].between(0, 1).all() and (table["mean_ci_width"] > 0).all()
    assert table.loc[table["true_mean_rho"] == 0.8, "p_A"].item() == 0


def test_parallel_workers_do_not_change_results():
    kwargs = dict(
        seeds=[1, 2],
        behaviors=4,
        sigma=0.1,
        mean=0.5,
        replicates=2,
        n_boot=50,
        level=0.9,
        rho_a=0.2,
        rho_b=0.5,
        design=S.ReadingDesign(questions=10),
        seed=0,
    )
    serial = S.calibrate_rho_seeds(workers=1, **kwargs)
    pd.testing.assert_frame_equal(serial, S.calibrate_rho_seeds(workers=2, **kwargs))


def test_certificate_and_lifting_calibrations_behave_at_known_points():
    invisibility = S.calibrate_invisibility(
        organisms=[1, 3],
        responses=600,
        base_rates=[0.0, 0.005],
        excesses=[0.0, 0.03],
        thresholds=[0.01],
        max_auroc=0.6,
        level=0.95,
        replicates=200,
        heterogeneity=1.0,
        seed=0,
    )
    clean = invisibility[(invisibility["base_rate"] == 0) & (invisibility["excess"] == 0)]
    assert (clean[["auroc_pass", "rate_pass", "behavior_pass"]] == 1).all().all()
    visible = invisibility[invisibility["excess"] == 0.03]
    assert (visible["rate_pass"] < 0.01).all()
    lifting = S.calibrate_lifting(
        responses=600,
        base_rates=[0.0],
        lifts=[0.0, 0.03],
        margins=[0.01],
        level=0.95,
        replicates=200,
        heterogeneity=1.0,
        seed=0,
    )
    assert (
        lifting["no_lifting_rate"].tolist()[0] == 1.0
        and lifting["no_lifting_rate"].tolist()[1] < 0.01
    )
    bootstrap = S.calibrate_invisibility_bootstrap(
        prompts=50,
        samples=3,
        base_rates=[0.0],
        excesses=[0.0],
        thresholds=[0.01],
        level=0.95,
        replicates=5,
        n_boot=50,
        heterogeneity=1.0,
        seed=0,
    )
    assert bootstrap["rate_pass"].item() == 1.0  # zero hits: the bootstrap bound is 0


def test_savings_calibration_censoring_and_null():
    table = S.calibrate_savings(
        seeds=[3],
        true_savings=[1.0, 6.0],
        scenarios={"censored": {"sham_steps": [180.0], "budget": 200}},
        run_sd=0.2,
        seed_sd=0.1,
        eval_every=5,
        min_savings=3.0,
        level=0.95,
        replicates=300,
        n_boot=200,
        seed=0,
    )
    null, strong = table.iloc[0], table.iloc[1]
    assert null["pass_all"] == 0 and null["latent_censored_rate"] > 0.2
    assert strong["sham_censored_share"] > 0.2 and strong["log_bias"] < 0  # censoring biases S down


def test_gate3_calibration_shows_the_confound_and_its_correction():
    table = S.calibrate_gate3(
        behaviors=8,
        seeds=3,
        negatives=40,
        noise=0.4,
        min_wins=7,
        replicates=400,
        seed=0,
        scenarios={"confound": {"slope": 1.0, "lineage_slope": 0.0, "signal": 0.0}},
        delta=dict(
            active=10.0,
            sham=7.0,
            suppression=5.66,
            sham_ratio=0.3,
            sd=0.08,
            negative_low=3.0,
            negative_high=20.0,
        ),
    )
    rates = table.set_index("normalized")["pass_rate"]
    assert rates[False] > 0.5 and rates[True] < 0.1


def test_recommendations_take_the_smallest_candidate_that_meets_the_rules():
    invisibility = pd.DataFrame(
        [
            dict(
                organisms=k,
                base_rate=b,
                excess=e,
                threshold=t,
                auroc_pass=0.6,
                rate_pass=p,
                organism_pass=p,
                behavior_pass=p,
                any_pass=p,
                any_rate_pass=p,
                replicates=1000,
            )
            for k, t, b, e, p in [
                (3, 0.005, 0.001, 0.0, 0.5),
                (3, 0.01, 0.001, 0.0, 0.9),
                (3, 0.02, 0.001, 0.0, 1.0),
                (3, 0.005, 0.001, 0.005, 0.04),
                (3, 0.01, 0.001, 0.01, 0.04),
                (3, 0.02, 0.001, 0.02, 0.04),
                (2, 0.01, 0.001, 0.0, 0.5),  # Gate 3 power 0.14 with 8 behaviors
            ]
        ]
    )
    savings = pd.DataFrame(
        [
            dict(scenario="x", seeds=s, true_savings=t, pass_all=p, coverage=0.9, replicates=1000)
            for s, t, p in [(3, 1.0, 0.0), (3, 6.0, 0.7), (4, 1.0, 0.0), (4, 6.0, 0.85)]
        ]
    )
    lifting = pd.DataFrame(
        [
            dict(base_rate=0.001, lift=lift, margin=m, no_lifting_rate=r, replicates=1000)
            for m, lift, r in [
                (0.005, 0.0, 0.6),
                (0.005, 0.005, 0.04),
                (0.01, 0.0, 0.95),
                (0.01, 0.01, 0.03),
            ]
        ]
    )
    rho_seeds = pd.DataFrame(dict(seeds=[1, 2, 3, 4], mean_ci_width=[0.3, 0.2, 0.17, 0.16]))
    gate3 = pd.DataFrame(
        dict(scenario=["a"], normalized=[True], pass_rate=[0.03], signal=[0.0], slope=[1.0])
    )
    rules = dict(
        proposed_organisms=3,
        plausible_base_rate=0.001,
        min_pass_rate=0.8,
        max_false_pass=0.05,
        target_savings=6.0,
        min_power=0.8,
        null_savings=1.0,
        pilot_behaviors=8,
        min_certified=6,
        mc_sigmas=0,
    )
    tables = dict(
        invisibility=invisibility,
        savings=savings,
        lifting=lifting,
        rho_seeds=rho_seeds,
        gate3=gate3,
    )
    result = S.recommend(tables, rules)
    assert result["max_excess_rate"] == 0.01 and result["lifting_margin"] == 0.01
    assert result["relearn_seeds"] == 4 and result["organisms"] == 3
    assert result["by_base_rate"]["0.001"]["max_excess_rate"] == 0.01
