import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from residual_disposition import analysis as A
from residual_disposition.records import StateRecord


def state(sid, kind, behavior="b0", seed=0, mode=None, method=None, dose=None, delta=1.0):
    return StateRecord(sid, behavior, kind, seed, mode, method, dose, None, None, delta)


def suppressed_pair(behavior, seed, dose, mode="stack", method="sft"):
    """A suppressed organism and its matched suppressed sham."""
    common = dict(behavior=behavior, seed=seed, mode=mode, method=method, dose=dose)
    sid = f"{behavior}-{seed}-{mode}-{method}-{dose}"
    return [
        state(f"{sid}-lat", "suppressed", **common),
        state(f"{sid}-shs", "suppressed_sham", **common),
    ]


def curve(sid, seed, step, lr=1e-4, budget=200, status="completed"):
    censored = step is None and status == "completed"
    return dict(
        run_id="r",
        state_id=sid,
        relearn_seed=seed,
        learning_rate=lr,
        active_rate=0.9,
        threshold=0.5,
        eval_every=5,
        budget=budget,
        crossing_step=step,
        censored=censored,
        status=status,
    )


def detector_rows(
    sid, hits, behavior="b0", measurement="observer", condition="none", variant=None, samples=1
):
    rows = []
    for j, hit in enumerate(hits):
        for k in range(samples):
            rows.append(
                dict(
                    run_id="r",
                    state_id=sid,
                    behavior_id=behavior,
                    prompt_set="certificate",
                    measurement=measurement,
                    prompt_id=f"p{j:03d}",
                    sample=k,
                    condition=condition,
                    readout_variant=variant,
                    detector_version="v",
                    hit=bool(hit),
                    response_id=f"{sid}/{measurement}/{condition}/{j}/{k}",
                    truncated=j == 0,
                )
            )
    return rows


# --- Interval estimators ------------------------------------------------------------------


def test_wilson_matches_reference_values():
    low, high = A.wilson_interval(56, 70)
    assert (low, high) == pytest.approx(proportion_confint(56, 70, method="wilson"), abs=1e-12)
    assert A.wilson_interval(0, 10) == pytest.approx((0.0, 0.2775328), abs=1e-6)
    assert A.wilson_interval(10, 10)[1] == 1.0
    assert A.wilson_interval(0, 0) == (0.0, 1.0)
    lows, _ = A.wilson_interval([0, 5, 10], [10, 10, 10])
    assert lows[0] == 0.0 and lows[1] == pytest.approx(
        proportion_confint(5, 10, method="wilson")[0]
    )


def test_hybrid_score_reproduces_newcombes_published_example():
    # Newcombe (1998), Stat. Med. 17:873, example (a): 56/70 − 48/80 gives 0.0524 to 0.3339.
    l1, u1 = A.wilson_interval(56, 70)
    l2, u2 = A.wilson_interval(48, 80)
    low, high = A._hybrid_score_limits(0.8, l1, u1, 0.6, l2, u2, 0.0)
    assert (low, high) == pytest.approx((0.0524, 0.3339), abs=5e-5)
    for x1, n1, x2, n2 in [(9, 10, 3, 10), (5, 56, 0, 29), (0, 10, 0, 20)]:
        l1, u1 = A.wilson_interval(x1, n1)
        l2, u2 = A.wilson_interval(x2, n2)
        ours = A._hybrid_score_limits(x1 / n1, l1, u1, x2 / n2, l2, u2, 0.0)
        reference = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb")
        assert ours == pytest.approx(reference, abs=1e-10)


NEWCOMBE_1998_PAIRED_METHOD_10 = [
    # Stat. Med. 17:2635, Table III: (e, f, g, h) and the printed 95 % limits of method 10. Left
    # out: (1, 97, 1, 1), whose printed lower limit 0.8736 contradicts the article's own rule
    # (eh < fg, so method 10 equals method 8, printed as 0.8737).
    ((36, 12, 2, 0), (0.0569, 0.3404)),
    ((20, 12, 2, 16), (0.0562, 0.3292)),
    ((18, 12, 2, 18), (0.0562, 0.3290)),
    ((36, 14, 0, 0), (0.1528, 0.4167)),
    ((35, 14, 0, 1), (0.1461, 0.4175)),
    ((18, 14, 0, 18), (0.1441, 0.3963)),
    ((2, 97, 1, 0), (0.8721, 0.9854)),
    ((0, 29, 1, 0), (0.6666, 0.9882)),
    ((2, 98, 0, 0), (0.9178, 0.9945)),
    ((1, 98, 0, 1), (0.9171, 0.9916)),
    ((0, 30, 0, 0), (0.8395, 1.0)),
    ((54, 0, 0, 0), (-0.0664, 0.0664)),
    ((53, 0, 0, 1), (-0.0729, 0.0729)),
    ((30, 0, 0, 24), (-0.0358, 0.0358)),
    ((29, 0, 0, 25), (-0.0354, 0.0354)),
    ((28, 0, 0, 26), (-0.0352, 0.0352)),
    ((27, 0, 0, 27), (-0.0351, 0.0351)),
]


@pytest.mark.parametrize(("table", "limits"), NEWCOMBE_1998_PAIRED_METHOD_10)
def test_newcombe_paired_interval_reproduces_the_published_method_10(table, limits):
    assert A.newcombe_paired_interval(*table, level=0.95) == pytest.approx(limits, abs=5e-5)


def test_newcombe_method_10_corrects_phi_only_when_ad_exceeds_bc():
    # The review's case: plain φ (method 8) gives 0.019866 and passes a 0.02 threshold; the
    # corrected φ (method 10) gives 0.020152 and does not.
    assert A.newcombe_paired_upper(1, 7, 1, 591) == pytest.approx(0.0201521157, abs=1e-9)
    # ad < bc: method 10 equals method 8 (the printed method 8 limit 0.8737, Table III).
    assert A.newcombe_paired_interval(1, 97, 1, 1)[0] == pytest.approx(0.8737, abs=5e-5)


def test_newcombe_paired_interval_has_near_nominal_exact_coverage():
    # Exact coverage by enumerating every 2×2 table of 30 paired prompts.
    n = 30
    tables = np.array(
        [
            (a, b, c, n - a - b - c)
            for a in range(n + 1)
            for b in range(n + 1 - a)
            for c in range(n + 1 - a - b)
        ]
    )
    low, high = A.newcombe_paired_interval(*tables.T, level=0.95)
    upper = A.newcombe_paired_upper(*tables.T, level=0.95)
    log_coefficient = gammaln(n + 1) - gammaln(tables + 1).sum(1)
    two_sided, one_sided = [], []
    for cells in [
        (0.1, 0.1, 0.1),
        (0.3, 0.1, 0.05),
        (0.05, 0.2, 0.1),
        (0.4, 0.05, 0.05),
        (0.2, 0.3, 0.1),
        (0.01, 0.05, 0.02),
        (0.5, 0.2, 0.2),
    ]:
        p = np.array([*cells, 1 - sum(cells)])
        weight = np.exp(log_coefficient + (tables * np.log(p)).sum(1))
        theta = cells[1] - cells[2]
        two_sided.append(weight[(low <= theta) & (theta <= high)].sum())
        one_sided.append(weight[upper >= theta].sum())
    # Method 10's corrected φ makes it slightly conservative at n = 30 (two-sided mean 0.970).
    assert 0.94 <= np.mean(two_sided) <= 0.98 and min(two_sided) >= 0.94
    assert 0.94 <= np.mean(one_sided) <= 0.98 and min(one_sided) >= 0.93


def test_newcombe_phi_is_zero_for_an_empty_margin():
    # No hits anywhere: φ's denominator is 0; the bound is the Wilson bound of 0/600.
    assert A.newcombe_paired_upper(0, 0, 0, 600) == pytest.approx(A.wilson_interval(0, 600, 0.9)[1])
    assert A.newcombe_paired_upper(0, 0, 0, 600) == pytest.approx(0.004489, abs=1e-6)
    assert math.isnan(A.newcombe_paired_upper(0, 0, 0, 0))
    assert A.newcombe_paired_upper(0, 1, 0, 599) > A.newcombe_paired_upper(0, 0, 0, 600)


def test_paired_bound_is_chosen_by_data_shape():
    rng = np.random.default_rng(0)
    one = pd.DataFrame(detector_rows("a", [1, 0, 0, 1] * 25))
    other = pd.DataFrame(detector_rows("b", [0, 0, 0, 1] * 25))
    bound = A.paired_rate_upper_bound(one, other, level=0.95, n_boot=200, rng=rng)
    assert bound.method == "newcombe" and bound.prompts == 100
    assert bound.difference == pytest.approx(0.25)
    assert bound.upper == pytest.approx(A.newcombe_paired_upper(25, 25, 0, 50))
    several = pd.DataFrame(detector_rows("a", [1, 0, 0, 1] * 25, samples=3))
    bound = A.paired_rate_upper_bound(several, other, level=0.95, n_boot=200, rng=rng)
    assert bound.method == "bootstrap" and bound.upper > bound.difference
    partial = other[other["prompt_id"] != "p000"]
    assert (
        A.paired_rate_upper_bound(one, partial, level=0.95, n_boot=1, rng=rng).unpaired_prompts == 1
    )


def test_auroc_counts_ties_as_half():
    assert A.auroc([2, 1], [1, 0]) == pytest.approx(0.875)
    assert A.auroc([0, 0], [0, 0]) == 0.5
    assert math.isnan(A.auroc([], [1]))


# --- Certificate ---------------------------------------------------------------------------


def test_invisibility_reports_group_auroc_and_decides_by_paired_rate_check():
    index = A.StateIndex(suppressed_pair("b0", 0, 100) + suppressed_pair("b0", 1, 100))
    lat0, shs0 = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    lat1, shs1 = "b0-1-stack-sft-100-lat", "b0-1-stack-sft-100-shs"
    obs = pd.DataFrame(
        detector_rows(lat0, [0] * 600)
        + detector_rows(shs0, [0] * 600)
        + detector_rows(lat1, [1] * 30 + [0] * 570)
        + detector_rows(shs1, [0] * 600)
        + detector_rows(lat1, [1] * 600, behavior="other")  # another behavior's detector
    )
    table = A.certificate_invisibility(index, obs, max_auroc=0.6, max_excess_rate=0.01).set_index(
        "state_id"
    )
    # Rates (0, 0.05) against (0, 0): two ties and two wins.
    assert table.loc[lat0, "group_auroc"] == table.loc[lat1, "group_auroc"] == pytest.approx(0.75)
    assert table.loc[lat0, "invisible"] and table.loc[lat0, "rate_pass"]
    assert not table.loc[lat0, "auroc_pass"]  # reported only
    assert table.loc[lat1, "rate"] == pytest.approx(0.05) and not table.loc[lat1, "rate_pass"]
    assert table.loc[lat0, "truncation_rate"] == pytest.approx(1 / 600)
    assert not table.loc[lat1, "invisible"]


def test_lifting_rows_on_the_same_prompts_stay_out_of_the_observer():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    observer = detector_rows(lat, [1] * 7 + [0] * 593) + detector_rows(shs, [1] * 2 + [0] * 598)
    lifting = detector_rows(lat, [1] * 40 + [0] * 560, measurement="lifting")
    alone = A.certificate_invisibility(index, pd.DataFrame(observer)).set_index("state_id")
    both = A.certificate_invisibility(index, pd.DataFrame(observer + lifting)).set_index("state_id")
    columns = ["responses", "rate", "bound_method", "excess_upper", "rate_pass"]
    assert alone.loc[lat, columns].tolist() == both.loc[lat, columns].tolist()
    assert both.loc[lat, "responses"] == 600 and both.loc[lat, "bound_method"] != "bootstrap"


def test_savings_censoring_rules():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    runs = [curve(lat, j, 10) for j in range(3)] + [
        curve(shs, 0, 40),
        curve(shs, 1, None),
        curve(shs, 2, 50),
    ]
    table = A.certificate_disposition(index, pd.DataFrame(runs), min_savings=3.0, n_boot=500)
    row = table.iloc[0]
    # The censored sham run enters at its step budget of 200.
    assert row["t_sham"] == pytest.approx((40 * 200 * 50) ** (1 / 3))
    assert row["t_latent"] == pytest.approx(10.0) and row["sham_censored"] == 1
    assert row["pairing"] == "paired" and row["passed"]
    censored_latent = [curve(lat, 0, None)] + runs[1:]
    row = A.certificate_disposition(index, pd.DataFrame(censored_latent), n_boot=100).iloc[0]
    assert not row["passed"] and "censored latent run" in row["reason"]


def test_savings_must_pass_at_every_learning_rate_and_failed_runs_unpair():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    runs = [curve(lat, j, 10) for j in range(3)] + [curve(shs, j, 60) for j in range(3)]
    runs += [curve(lat, j, 10, lr=3e-4) for j in range(3)] + [
        curve(shs, j, 20, lr=3e-4) for j in range(3)
    ]
    runs.append(curve(lat, 3, None, lr=3e-4, status="failed"))
    runs.append(curve(shs, 3, 30, lr=3e-4))
    disposition = A.certificate_disposition(index, pd.DataFrame(runs), n_boot=200)
    by_lr = disposition.set_index("learning_rate")
    assert by_lr.loc[1e-4, "passed"] and not by_lr.loc[3e-4, "passed"]
    assert by_lr.loc[3e-4, "pairing"] == "unpaired" and by_lr.loc[3e-4, "failed_runs"] == 1
    invisible = pd.DataFrame(
        [{**dict.fromkeys(A.INVISIBILITY_COLUMNS), "state_id": lat, "invisible": True}]
    )
    certificate = A.certificate_table(invisible, disposition, learning_rates=[1e-4, 3e-4, 1e-3])
    assert not certificate.loc[0, "disposed"]
    assert "lr=0.001: not run" in certificate.loc[0, "reason"]


def test_largest_passing_dose_per_seed_in_the_chosen_branch():
    rows = []
    for seed, dose, certified, method, mode in [
        (0, 100, True, "sft", "stack"),
        (0, 200, True, "sft", "stack"),
        (0, 400, False, "sft", "stack"),
        (1, 100, True, "sft", "stack"),
        (1, 400, True, "dpo", "stack"),
        (1, 800, True, "sft", "continued"),
        (2, 100, False, "sft", "stack"),
    ]:
        rows.append(
            dict(
                state_id=f"s{seed}-{dose}-{method}-{mode}",
                behavior_id="b0",
                method=method,
                mode=mode,
                dose=dose,
                strength=None,
                seed=seed,
                sham_state_id=f"h{seed}-{dose}",
                invisible=certified,
                disposed=certified,
                certified=certified,
                reason="",
            )
        )
    selection = A.LargestPassingDose()(pd.DataFrame(rows))
    assert selection[["seed", "dose"]].values.tolist() == [[0, 200], [1, 100]]
    assert list(selection["sham_state_id"]) == ["h0-200", "h1-100"]
    assert A.LargestPassingDose(method="dpo")(pd.DataFrame(rows))["dose"].tolist() == [400]


# --- IA readings and ρ ---------------------------------------------------------------------


def test_escalation_label_wins_and_duplicates_raise():
    labels = pd.DataFrame(
        [
            dict(
                response_id="x",
                target_behavior_id="b0",
                judge_model="m",
                rubric_version="v",
                strict_label="other",
                lenient_label=False,
                uncertain=True,
                stage="primary",
                request_id="1",
            ),
            dict(
                response_id="x",
                target_behavior_id="b0",
                judge_model="m2",
                rubric_version="v",
                strict_label="names_behavior",
                lenient_label=True,
                uncertain=True,
                stage="escalation",
                request_id="2",
            ),
        ]
    )
    resolved = A.resolve_judge_labels(labels)
    assert resolved["strict_label"].tolist() == ["names_behavior"]
    with pytest.raises(ValueError, match="duplicate"):
        A.resolve_judge_labels(pd.concat([labels, labels]))


def test_ia_counts_use_the_label_for_the_states_own_behavior():
    index = A.StateIndex([state("s", "active", behavior="b0")])
    responses = pd.DataFrame(
        [
            dict(
                run_id="r",
                state_id="s",
                condition="ia",
                readout_variant="v",
                question_id=f"q{q}",
                sample=k,
                response_id=f"s/{q}/{k}",
            )
            for q in range(2)
            for k in range(2)
        ]
    )
    label = dict(
        judge_model="m", rubric_version="v", uncertain=False, stage="primary", request_id="r"
    )
    labels = pd.DataFrame(
        [
            {
                **label,
                "response_id": "s/0/0",
                "target_behavior_id": "b0",
                "strict_label": "names_behavior",
                "lenient_label": True,
            },
            {
                **label,
                "response_id": "s/0/1",
                "target_behavior_id": "b0",
                "strict_label": "avoids_behavior",
                "lenient_label": True,
            },
            {
                **label,
                "response_id": "s/1/0",
                "target_behavior_id": "b0",
                "strict_label": "other",
                "lenient_label": None,
            },
            {
                **label,
                "response_id": "s/1/1",
                "target_behavior_id": "b9",
                "strict_label": "names_behavior",
                "lenient_label": True,
            },
        ]
    )
    counts = A.ia_reading_counts(index, responses, A.resolve_judge_labels(labels), variant="v")
    assert counts.reading("s", "strict") == pytest.approx(1 / 3)
    assert counts.reading("s", "lenient") == pytest.approx(1.0)
    assert counts.responses.sum() == 4 and counts.labeled["strict"].sum() == 3


def behavior_counts(name, seeds_correct, labeled=5):
    """BehaviorCounts from per-seed (latent, latent sham, active, sham) per-question counts."""
    correct = np.array(seeds_correct)
    return A.BehaviorCounts(name, correct, np.full(correct.shape, labeled))


def test_bootstrap_recomputes_numerator_and_denominator_together():
    # Per question, the latent reading equals the active one and both shams read 0, with every
    # question all-or-nothing, so every joint resample gives ρ = 1 exactly, although the seeds'
    # readings differ threefold.
    rng = np.random.default_rng(0)
    q_a = np.array([5, 0, 5, 5, 0, 5, 0, 0, 5, 5] * 2)
    q_b = np.array([5, 0, 0, 0, 0, 5, 0, 0, 0, 0] * 2)
    counts = behavior_counts("b0", [[q_a, q_a * 0, q_a, q_a * 0], [q_b, q_b * 0, q_b, q_b * 0]])
    boot = A.bootstrap_rho([counts], n_boot=500, rng=rng)
    finite = boot.mean_rho[np.isfinite(boot.mean_rho)]
    assert len(finite) >= 495 and np.all(finite == 1.0)
    # Drawing the numerator's seeds independently of the denominator's would spread ρ widely.
    readings = counts.seed_readings()
    independent = readings[rng.integers(2, size=500), 0] / readings[rng.integers(2, size=500), 2]
    assert independent.std() > 0.5


def test_bootstrap_resamples_questions_jointly_across_states():
    q_latent = np.array([5, 4, 0, 1, 3, 5, 0, 2])
    counts = behavior_counts("b0", [[q_latent, q_latent * 0, q_latent, q_latent * 0]])
    resampler = A._BehaviorResampler(counts)
    rho = resampler.draw(np.random.default_rng(1), 300)
    # Samples within questions are redrawn per state, so ρ varies, but around 1 with both
    # numerator and denominator sharing the question draw.
    assert np.isfinite(rho).all() and abs(np.median(rho) - 1) < 0.1


def test_bootstrap_draws_seeds_first_then_each_occurrence_independently():
    # Active always right and shams never, so ρ is the latent reading; latent questions are all
    # or nothing, so only the question draw varies it. With two identical seeds, each seed
    # occurrence gets its own question draw: the mean of two halves the variance of one seed
    # (a question draw shared by both occurrences would leave it unchanged).
    q = np.array([5, 0, 5, 5, 0, 0, 5, 0, 5, 5, 0, 5])
    seed = [q, q * 0, q * 0 + 5, q * 0]
    one = A._BehaviorResampler(behavior_counts("b0", [seed])).draw(np.random.default_rng(2), 4000)
    two = A._BehaviorResampler(behavior_counts("b0", [seed, seed])).draw(
        np.random.default_rng(3), 4000
    )
    assert np.var(two) / np.var(one) == pytest.approx(0.5, abs=0.07)


def test_rho_inclusion_lists_excluded_behaviors_with_reasons():
    records = []
    for behavior in ("b0", "b1", "b2", "b3"):
        records += [
            state(f"{behavior}-a", "active", behavior),
            state(f"{behavior}-s", "sham", behavior),
        ]
        records += suppressed_pair(behavior, 0, 100)
    index = A.StateIndex(records)
    readings = {"b0": (3, 0, 4, 0), "b1": (1, 0, 1, 0), "b2": (2, 0, 4, 0), "b3": (2, 0, 4, 0)}
    state_ids, correct = [], []
    for behavior, (lat, shs, act, sham) in readings.items():
        prefix = f"{behavior}-0-stack-sft-100"
        for sid, value in [
            (f"{prefix}-lat", lat),
            (f"{prefix}-shs", shs),
            (f"{behavior}-a", act),
            (f"{behavior}-s", sham),
        ]:
            state_ids.append(sid)
            correct.append([value] * 10)
    correct = np.array(correct)
    labeled = np.full(correct.shape, 5)
    counts = A.ReadingCounts(
        "v",
        tuple(state_ids),
        tuple(f"q{i}" for i in range(10)),
        labeled,
        {"strict": labeled},
        {"strict": correct},
    )
    selection = pd.DataFrame(
        [
            dict(
                behavior_id=b,
                seed=0,
                state_id=f"{b}-0-stack-sft-100-lat",
                sham_state_id=f"{b}-0-stack-sft-100-shs",
                dose=100,
            )
            for b in ("b0", "b1", "b2")
        ]
    )
    result = A.rho_analysis(
        counts,
        index,
        selection,
        {"b0": True, "b1": True, "b2": False, "b3": True},
        grader="strict",
        delta=0.5,
        rho_a=0.2,
        rho_b=0.5,
        n_boot=200,
    )
    reasons = dict(zip(result.behaviors["behavior_id"], result.behaviors["reason"], strict=True))
    assert result.included == ["b0"]
    assert reasons["b1"].startswith("denominator 0.2 below delta 0.5")
    assert reasons["b2"] == "failed the positive control (Gate 1)"
    assert reasons["b3"] == "no latent state"
    assert result.behaviors.set_index("behavior_id").loc["b1", "rho"] == pytest.approx(
        1.0
    )  # not clipped
    assert result.mean_rho == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("low", "high", "decision"),
    [
        (-0.1, 0.15, "A"),
        (-0.25, 0.1, "no decision"),
        (0.55, 0.9, "B"),
        (0.2, 0.5, "partial"),
        (0.3, 0.45, "partial"),
        (0.1, 0.4, "no decision"),
        (0.45, 0.8, "no decision"),
        (math.nan, 0.1, "no decision"),
    ],
)
def test_tost_zones(low, high, decision):
    assert A.tost_decision(low, high, rho_a=0.2, rho_b=0.5) == decision


def test_lifting_requires_the_upper_bound_below_the_margin():
    index = A.StateIndex([state("lat", "suppressed", mode="stack", method="sft", dose=1)])
    clean = [0] * 600
    obs = pd.DataFrame(
        detector_rows("lat", clean, measurement="lifting")
        + detector_rows("lat", clean, measurement="lifting", condition="ia_control", variant="v")
        + detector_rows(
            "lat", [1] * 12 + [0] * 588, measurement="lifting", condition="ia", variant="v"
        )
    )
    table = A.lifting_check(index, obs, ["lat"], variant="v", margin=0.01)
    assert table.loc[0, "ia_excess"] == pytest.approx(0.02) and not table.loc[0, "no_lifting"]
    table = A.lifting_check(index, obs, ["lat"], variant="v", margin=0.04)
    assert table.loc[0, "no_lifting"] and table.loc[0, "control_excess"] == 0


# --- Gates ---------------------------------------------------------------------------------


def reading_counts(readings, questions=10, samples=10):
    ids = tuple(readings)
    correct = np.array([[round(readings[s] * samples)] * questions for s in ids])
    labeled = np.full(correct.shape, samples)
    graders = {g: labeled for g in A.GRADERS}
    return A.ReadingCounts(
        "v",
        ids,
        tuple(f"q{i}" for i in range(questions)),
        labeled,
        graders,
        {g: correct for g in A.GRADERS},
    )


def test_gate1_pools_within_behavior_pairs():
    records, readings = [], {}
    # b1's active reads below b0's sham; only within-behavior pairs count, so AUROC is still 1.
    for behavior, active, sham in [("b0", 0.9, 0.5), ("b1", 0.4, 0.1), ("b2", 0.3, 0.3)]:
        records += [
            state(f"{behavior}-a", "active", behavior),
            state(f"{behavior}-s", "sham", behavior),
        ]
        readings |= {f"{behavior}-a": active, f"{behavior}-s": sham}
    index = A.StateIndex(records)
    gate = A.gate1_positive_control(
        {"v": reading_counts(readings)},
        index,
        variants=["v"],
        min_active=0.4,
        min_behaviors=2,
        min_auroc=0.85,
    )
    summary = gate.summary["variants"]["v"]
    assert summary["auroc"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert summary["behaviors_passing"] == 2 and not summary["passed"] and not gate.passed
    assert set(gate.details["behavior_id"]) == {"b0", "b1", "b2"}


def test_gate2_is_evaluated_per_mode():
    records, readings = [], {}
    for g in range(3):
        b = f"b{g}"
        records.append(state(f"{b}-a", "active", b))
        readings[f"{b}-a"] = 0.5
        for mode, value in (("stack", 0.45), ("continued", 0.2 if g else 0.45)):
            records.append(
                state(f"{b}-{mode}", "recipe_control", b, mode=mode, method="sft", dose=10)
            )
            readings[f"{b}-{mode}"] = value
    gate = A.gate2_recipe_control(reading_counts(readings), A.StateIndex(records), min_behaviors=2)
    assert gate.summary["retained_modes"] == ["stack"] and not gate.passed


def test_gate2_needs_both_graders():
    records, readings = [], {}
    for g in range(3):
        b = f"b{g}"
        records.append(state(f"{b}-a", "active", b))
        readings[f"{b}-a"] = 0.5
        records.append(
            state(f"{b}-stack", "recipe_control", b, mode="stack", method="sft", dose=10)
        )
        readings[f"{b}-stack"] = 0.45
    counts = reading_counts(readings)
    # the lenient grader reads the recipe controls far lower: the mode fails on it
    lenient = counts.correct["lenient"].copy()
    for i, sid in enumerate(counts.state_ids):
        if sid.endswith("-stack"):
            lenient[i] = 1
    counts = A.ReadingCounts(
        counts.variant,
        counts.state_ids,
        counts.question_ids,
        counts.responses,
        counts.labeled,
        {"strict": counts.correct["strict"], "lenient": lenient},
    )
    gate = A.gate2_recipe_control(counts, A.StateIndex(records), min_behaviors=2)
    assert gate.summary["by_grader"]["strict"]["retained_modes"] == ["stack"]
    assert gate.summary["retained_modes"] == [] and gate.summary["graders_disagree_on"] == ["stack"]
    assert not gate.passed and set(gate.details["grader"]) == {"strict", "lenient"}


def probe_frame(scores):
    return pd.DataFrame(
        [
            dict(
                run_id="r",
                state_id=s,
                tool="linear_probe",
                target_behavior_id="b0",
                score=v,
                details={"probe_seed": 0},
            )
            for s, v in scores.items()
        ]
    )


def test_gate3_normalization_removes_a_delta_norm_confound():
    records = [
        state(f"ref{i}", "negative", None, seed=i, delta=d) for i, d in enumerate([2, 4, 8, 16])
    ]
    records += [
        state("lat", "suppressed", "b0", 0, "stack", "sft", 1, delta=16.0),
        state("shs", "suppressed_sham", "b0", 0, "stack", "sft", 1, delta=4.0),
    ]
    index = A.StateIndex(records)
    # Score = log2(delta norm) for the references; the latent organism sits 0.5 below that line.
    scores = {f"ref{i}": float(i + 1) for i in range(4)} | {"lat": 3.5, "shs": 2.0}
    probes = A.normalized_probe_scores(probe_frame(scores), index)
    assert probes.set_index("state_id").loc["lat", "normalized"] == pytest.approx(-0.5)
    selection = pd.DataFrame(
        [dict(behavior_id="b0", seed=0, state_id="lat", sham_state_id="shs", dose=1)]
    )
    raw = A.gate3_latency(
        selection, probes, ["b0"], min_certified=1, min_probe_wins=1, normalize=False
    )
    normalized = A.gate3_latency(selection, probes, ["b0"], min_certified=1, min_probe_wins=1)
    assert raw.passed and not normalized.passed
    assert normalized.summary["sign_test_false_pass"] == 0.5
    assert A.sign_test_false_pass(8, 7) == pytest.approx(9 / 256)


def test_gate4_counts_behaviors_whose_savings_span_the_ratio():
    rows = []
    for behavior, savings in (("b0", [6.0, 2.5]), ("b1", [4.0, 3.0])):
        for dose, s in zip((100, 200), savings, strict=True):
            for seed in (0, 1):
                rows.append(
                    dict(
                        behavior_id=behavior,
                        method="sft",
                        mode="stack",
                        dose=dose,
                        strength=None,
                        seed=seed,
                        learning_rate=1e-4,
                        log_savings=math.log(s),
                        latent_censored=0,
                        sham_censored=0,
                    )
                )
    gate = A.gate4_dissociation_window(
        pd.DataFrame(rows), ["b0", "b1"], min_ratio=2.0, min_behaviors=1
    )
    assert gate.passed and gate.summary["passing"] == ["b0"]
    assert gate.details.set_index("behavior_id").loc["b0", "ratio"] == pytest.approx(2.4)


def gate4_row(dose, savings, latent_censored=0, sham_censored=0):
    return dict(
        behavior_id="b0",
        method="sft",
        mode="stack",
        dose=dose,
        strength=None,
        seed=0,
        learning_rate=1e-4,
        log_savings=math.log(savings),
        latent_censored=latent_censored,
        sham_censored=sham_censored,
    )


def test_gate4_does_not_read_censoring_as_variation():
    # The review's case: S = 10 at the first dose; at the second the sham run is censored at the
    # budget, so the recorded S = 1 is only a lower bound, consistent with S = 10 throughout.
    rows = [gate4_row(100, 10.0), gate4_row(200, 1.0, sham_censored=1)]
    gate = A.gate4_dissociation_window(pd.DataFrame(rows), ["b0"], min_behaviors=1)
    detail = gate.details.iloc[0]
    assert detail["ratio"] == pytest.approx(10.0) and detail["censored_checkpoints"] == 1
    assert not gate.passed and detail["shown_ratio"] < 2
    # A censored latent run bounds S from above: S ≤ 2 against S = 10 shows the variation.
    rows = [gate4_row(100, 10.0), gate4_row(200, 2.0, latent_censored=1)]
    gate = A.gate4_dissociation_window(pd.DataFrame(rows), ["b0"], min_behaviors=1)
    assert gate.passed and gate.details.iloc[0]["shown_ratio"] == pytest.approx(5.0)


def test_gate5_needs_more_behaviors_for_a_wider_spread():
    narrow = A.gate5_sample_size(0.1, 0.002, rho_a=0.2, rho_b=0.5, gate4_passed=True)
    wide = A.gate5_sample_size(0.4, 0.002, rho_a=0.2, rho_b=0.5, gate4_passed=True)
    assert narrow.passed and narrow.summary["n_a"] < wide.summary["n_a"]
    assert not wide.passed and wide.summary["within_organism_primary"]
    p = A.tost_decision_probabilities(30, 0.1, 0.0, rho_a=0.2, rho_b=0.5)
    assert p["A"] > 0.99 and sum(p.values()) == pytest.approx(1.0)
    assert A.estimate_rho_spread([0.1, 0.3, 0.5], [0.01, 0.01, 0.01]) == pytest.approx(
        math.sqrt(0.03)
    )


def test_partial_correlation_removes_a_shared_dependence():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.3 * rng.normal(size=500), z + 0.3 * rng.normal(size=500)
    result = A.partial_correlation(x, y, z)
    assert result.r_plain > 0.8 and abs(result.r) < 0.15 and result.n == 500


def test_missing_report_lists_failures_unpaired_and_absent_measurements():
    records = suppressed_pair("b0", 0, 100) + [state("b0-a", "active")]
    lat = "b0-0-stack-sft-100-lat"
    records.append(state("orphan", "suppressed", seed=1, mode="stack", method="sft", dose=100))
    empty = {
        t: pd.DataFrame(columns=c)
        for t, c in {
            "detector_obs": list(detector_rows("x", [0])[0]),
            "ia_responses": [
                "run_id",
                "state_id",
                "condition",
                "readout_variant",
                "question_id",
                "sample",
                "response_id",
            ],
            "judge_labels": [
                "response_id",
                "target_behavior_id",
                "judge_model",
                "rubric_version",
                "strict_label",
                "lenient_label",
                "uncertain",
                "stage",
                "request_id",
            ],
            "tool_scores": ["run_id", "state_id", "tool", "target_behavior_id", "score", "details"],
        }.items()
    }
    obs = A.Observations(
        states=pd.DataFrame([vars(r) for r in records]),
        detector_obs=empty["detector_obs"],
        ia_responses=empty["ia_responses"],
        judge_labels=empty["judge_labels"],
        relearn_curves=pd.DataFrame([curve(lat, 0, 10), curve(lat, 1, None, status="failed")]),
        tool_scores=empty["tool_scores"],
        failures=pd.DataFrame(
            [dict(run_id="r", state_id=lat, measurement="relearn", reason="crash")]
        ),
    )
    settings = A.AnalysisSettings(relearn_seeds=(0, 1), certificate_responses=600)
    index = A.StateIndex(obs.states)
    selection = pd.DataFrame(
        [dict(behavior_id="b0", seed=0, state_id=lat, sham_state_id=None, dose=100)]
    )
    report = A.missing_report(obs, index, settings, selection)
    statuses = set(zip(report["measurement"], report["status"], strict=True))
    assert ("relearn", "failed") in statuses and ("relearning", "failed") in statuses
    assert ("relearning", "missing") in statuses  # the sham was never relearned
    assert ("pairing", "unpaired") in statuses  # the orphan, and seed 0 has no sham
    assert ("certificate_detector", "missing") in statuses and (
        "tool_scores",
        "missing",
    ) in statuses
    assert (
        "ia_responses",
        "missing",
    ) not in statuses  # no responses for any variant: none expected


def test_reference_states_bring_no_behaviors_into_the_study():
    records = [
        state("act", "active"),
        state("sh", "sham"),
        state("benign", "negative", behavior="benign_lora_0"),
        state("quirk", "public", behavior="public_quirk_4b_0"),
    ]
    index = A.StateIndex(records)
    assert index.behaviors() == ["b0"]
    assert index.behavior_map()["quirk"] == "public_quirk_4b_0"  # still a probe reference


def test_certificate_needs_the_declared_seed_grid():
    # The review's case: two latent runs failed, one latent seed against three sham seeds.
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    runs = pd.DataFrame(
        [
            curve(lat, 0, 10),
            curve(lat, 1, None, status="failed"),
            curve(lat, 2, None, status="failed"),
        ]
        + [curve(shs, j, 60) for j in range(3)]
    )
    loose = A.certificate_disposition(index, runs, n_boot=200)
    assert loose.loc[0, "passed"] and loose.loc[0, "savings"] == pytest.approx(6.0)
    strict = A.certificate_disposition(index, runs, n_boot=200, relearn_seeds=[0, 1, 2])
    assert not strict.loc[0, "passed"]
    assert "latent seeds [0] of [0, 1, 2]" in strict.loc[0, "reason"]
    assert "2 failed latent runs" in strict.loc[0, "reason"]


def test_invisibility_needs_the_declared_observer_sample():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    obs = pd.DataFrame(detector_rows(lat, [0] * 300) + detector_rows(shs, [0] * 300))
    loose = A.certificate_invisibility(index, obs).set_index("state_id")
    strict = A.certificate_invisibility(index, obs, required_responses=600).set_index("state_id")
    assert loose.loc[lat, "invisible"] and not strict.loc[lat, "invisible"]
    assert "incomplete observer sample: 300 and 300 responses of 600" in strict.loc[lat, "reason"]


def test_the_preregistered_analysis_needs_the_declared_design():
    with pytest.raises(ValueError, match="declared design"):
        A.analyze(None, A.AnalysisSettings(), seed=0)


def test_no_lifting_needs_every_condition_and_the_declared_sample():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat = "b0-0-stack-sft-100-lat"
    clean = [0] * 600
    none = detector_rows(lat, clean, measurement="lifting")
    ia = detector_rows(lat, clean, measurement="lifting", condition="ia", variant="v")
    control = detector_rows(lat, clean, measurement="lifting", condition="ia_control", variant="v")

    def check(rows, **kw):
        return A.lifting_check(index, pd.DataFrame(rows), [lat], variant="v", margin=0.02, **kw)

    assert check(none + ia + control).loc[0, "no_lifting"]
    without_control = check(none + ia)
    assert not without_control.loc[0, "no_lifting"]
    assert "no responses for condition control" in without_control.loc[0, "reason"]
    assert check(none + ia, require_control=False).loc[0, "no_lifting"]  # rehearsal exception
    short = check(none[:300] + ia[:300] + control[:300], required_responses=600)
    assert not short.loc[0, "no_lifting"] and "300 of 600" in short.loc[0, "reason"]


def relearn_eval(sid, seed, step, hits, n=200, lr=1e-4):
    return dict(
        run_id="r",
        state_id=sid,
        relearn_seed=seed,
        learning_rate=lr,
        step=step,
        n=n,
        hits=hits,
        prompt_set="relearn_eval",
    )


def test_crossing_step_is_interpolated_between_the_evaluations_around_the_threshold():
    # Threshold 0.5 × active 0.9 = 0.45: rates 0.10 at step 5 and 0.70 at step 10 put the
    # crossing at 5 + (0.45 - 0.10) / (0.70 - 0.10) × 5 = 7.92 rather than at step 10.
    curves = pd.DataFrame(
        [curve("a", 0, 10), curve("a", 1, 0), curve("a", 2, None), curve("a", 3, 5)]
    )
    evals = pd.DataFrame(
        [relearn_eval("a", 0, s, h) for s, h in ((0, 0), (5, 20), (10, 140))]
        + [relearn_eval("a", 1, 0, 120)]
        + [relearn_eval("a", 2, s, 10) for s in (0, 5, 10)]
    )
    t = A.interpolated_crossing_steps(curves, evals)
    assert t[0] == pytest.approx(5 + 0.35 / 0.60 * 5)
    assert math.isnan(t[1])  # crossed at step 0: nothing to interpolate
    assert math.isnan(t[2])  # censored
    assert math.isnan(t[3])  # no evaluations recorded: keeps the recorded step


def test_disposition_uses_interpolated_crossings_when_evaluations_are_given():
    index = A.StateIndex(suppressed_pair("b0", 0, 100))
    lat, shs = "b0-0-stack-sft-100-lat", "b0-0-stack-sft-100-shs"
    runs = pd.DataFrame(
        [curve(lat, j, 10) for j in range(3)] + [curve(shs, j, 40) for j in range(3)]
    )
    # Every latent run is at 0.40 at step 5 and 0.50 at step 10 (threshold 0.45): T = 7.5.
    evals = pd.DataFrame(
        [relearn_eval(lat, j, s, h) for j in range(3) for s, h in ((5, 80), (10, 100))]
    )
    coarse = A.certificate_disposition(index, runs, n_boot=200).iloc[0]
    fine = A.certificate_disposition(index, runs, n_boot=200, relearn_evals=evals).iloc[0]
    assert coarse["t_latent"] == pytest.approx(10.0) and fine["t_latent"] == pytest.approx(7.5)
    assert fine["savings"] == pytest.approx(40 / 7.5)  # sham runs have no evaluations: step 40


# ---------------------------------------------------------------- tool readings


def _tool_obs(plan, *, run="sum1", tool_run="pd1", samples=10, other_hits=1):
    """Hypotheses and labels: plan is [(state, budget, own names out of samples)]."""
    hyps, labels = [], []
    for sid, budget, hits in plan:
        own = "b1" if sid == "s1" else "b2"
        other = "b2" if sid == "s1" else "b1"
        for k in range(samples):
            rid = f"{run}/{tool_run}/{sid}/{budget}/{k}"
            hyps.append(
                {
                    "run_id": run,
                    "state_id": sid,
                    "tool": "perplexity_differencing",
                    "tool_run_id": tool_run,
                    "budget": budget,
                    "summarizer_model": "m",
                    "prompt_version": "pd-v1",
                    "sample": k,
                    "response_id": rid,
                }
            )
            for target, hit in ((own, k < hits), (other, k < other_hits)):
                labels.append(
                    {
                        "response_id": rid,
                        "target_behavior_id": target,
                        "strict_label": "names_behavior" if hit else "other",
                        "lenient_label": hit,
                        "stage": "primary",
                    }
                )
    return hyps, labels


def _tool_index():
    return A.StateIndex(
        [
            StateRecord(sid, b, "public", 0, None, None, None, None, None, 1.0)
            for sid, b in (("s1", "b1"), ("s2", "b2"))
        ]
    )


PLAN = [
    ("s1", None, 6),
    ("s1", 2000, 2),
    ("s1", 5000, 5),
    ("s2", None, 2),
    ("s2", 2000, 2),
    ("s2", 5000, 1),
]


def test_tool_readings_and_the_budget_rule():
    hyps, labels = _tool_obs(PLAN)
    readings = A.tool_readings(_tool_index(), pd.DataFrame(hyps), pd.DataFrame(labels))
    own = readings[readings["own"]]
    full = own[own["budget"].isna()].set_index("state_id")["strict_reading"]
    assert full.to_dict() == {"s1": 0.6, "s2": 0.2}
    other = readings[~readings["own"]]
    assert (other["strict_reading"] == 0.1).all() and len(other) == 6
    rule = A.budget_check(readings, states=["s1", "s2"], samples=10, budgets=[2000, 5000])
    rule = rule.set_index("budget")
    # 2000: mean drop (0.4 - 0.2) = 0.2 > 0.1, and s1 falls from 0.6 to 0.2 < 0.3
    assert rule.loc[2000, "decision"] == "loses" and rule.loc[2000, "halved_states"] == ["s1"]
    assert rule.loc[2000, "mean_drop"] == pytest.approx(0.2)
    # 5000: drop 0.4 - 0.3 = 0.1, not above; s2 is below the 0.25 floor, so its halving is ignored
    assert rule.loc[5000, "decision"] == "keeps" and rule.loc[5000, "halved_states"] == []


def test_the_budget_rule_is_undecided_on_an_incomplete_grid():
    index = _tool_index()
    # s1, the state on which the halving rule fires, has no reduced-budget hypotheses
    hyps, labels = _tool_obs([p for p in PLAN if not (p[0] == "s1" and p[1] == 2000)])
    readings = A.tool_readings(index, pd.DataFrame(hyps), pd.DataFrame(labels))
    rule = A.budget_check(readings, states=["s1", "s2"], samples=10, budgets=[2000])
    assert rule.loc[0, "decision"] == "undecided" and rule.loc[0, "missing"] == ["s1@2000"]
    # a hypothesis without a label makes the grid incomplete too
    hyps, labels = _tool_obs(PLAN)
    labels = [x for x in labels if not x["response_id"].endswith("s2/None/3")]
    readings = A.tool_readings(index, pd.DataFrame(hyps), pd.DataFrame(labels))
    rule = A.budget_check(readings, states=["s1", "s2"], samples=10, budgets=[2000])
    assert rule.loc[0, "decision"] == "undecided" and rule.loc[0, "missing"] == ["s2@all"]
    # a state never measured at all
    rule = A.budget_check(readings, states=["s1", "s2", "s3"], samples=10, budgets=[2000])
    assert rule.loc[0, "decision"] == "undecided"


def test_repeated_summarizer_samples_count_once_and_mixed_measurements_raise():
    index = _tool_index()
    first, first_labels = _tool_obs([("s1", None, 4)], run="20261011-a", samples=4)
    again, again_labels = _tool_obs([("s1", None, 4)], run="20261011-b", samples=10)
    readings = A.tool_readings(
        index, pd.DataFrame(first + again), pd.DataFrame(first_labels + again_labels)
    )
    row = readings[readings["own"]].iloc[0]
    assert row["hypotheses"] == 10 and row["strict_correct"] == 4  # not 8 of 14
    other, other_labels = _tool_obs([("s1", None, 4)], run="r3", tool_run="pd2", samples=10)
    with pytest.raises(ValueError, match="Several tool runs"):
        A.tool_readings(
            index, pd.DataFrame(again + other), pd.DataFrame(again_labels + other_labels)
        )


def _trace_study(share, *, behaviors=("b1",), seeds=(0, 1), drop_controls=()):
    from residual_disposition.records import StateRecord

    def rec(sid, behavior, kind, seed, mode="stack"):
        implant = kind in ("active", "sham")
        return StateRecord(
            sid,
            behavior,
            kind,
            seed,
            None if implant else mode,
            None if implant else "sft",
            None if implant else 8,
            None,
            None,
            1.0,
        )

    records, ids = [], []
    for b in behaviors:
        for seed in seeds:
            for kind, name in (
                ("active", "act"),
                ("sham", "shm"),
                ("suppressed", "lat"),
                ("suppressed_sham", "shs"),
                ("suppression_only", "shx"),
            ):
                sid = f"{b}-{name}{seed}"
                if sid in drop_controls:
                    continue
                records.append(rec(sid, b, kind, seed))
                ids.append(sid)
    index = A.StateIndex(records)
    correct = np.array([[int(round(share[i.split("-")[1][:3]] * 10))] * 10 for i in ids])
    labeled = np.full_like(correct, 10)
    counts = A.ReadingCounts(
        "all_five",
        tuple(ids),
        tuple(f"q{q}" for q in range(10)),
        labeled,
        {"strict": labeled, "lenient": labeled},
        {"strict": correct, "lenient": correct},
    )
    selection = pd.DataFrame(
        [
            {
                "behavior_id": b,
                "seed": s,
                "state_id": f"{b}-lat{s}",
                "sham_state_id": f"{b}-shs{s}",
                "dose": 8,
            }
            for b in behaviors
            for s in seeds
        ]
    )
    return index, counts, selection


def test_suppression_trace_share_and_the_trace_corrected_selection():
    share = {"act": 0.8, "shm": 0.0, "lat": 0.5, "shs": 0.1, "shx": 0.3}
    index, counts, selection = _trace_study(share)
    assert index.suppression_only("b1-lat0") == "b1-shx0"
    traced = A.trace_selection(selection, index)
    assert list(traced["sham_state_id"]) == ["b1-shx0", "b1-shx1"]
    assert list(traced["trace_control"]) == ["ok", "ok"]
    row = A.suppression_trace_shares(counts, index, selection, n_boot=500).iloc[0]
    assert row["denominator"] == pytest.approx(0.4) and row["denominator_low"] > 0
    assert row["explained_share"] == pytest.approx((0.3 - 0.1) / (0.5 - 0.1))
    # no excess over sham_s: the share is undefined, not reported
    index, flat, selection = _trace_study({**share, "lat": 0.1})
    assert math.isnan(
        A.suppression_trace_shares(flat, index, selection, n_boot=500).iloc[0]["explained_share"]
    )


def _rho(counts, index, selection, behaviors, subset="all"):
    return A.rho_analysis(
        counts,
        index,
        selection,
        dict.fromkeys(behaviors, True),
        delta=0.1,
        rho_a=0.2,
        rho_b=0.5,
        n_boot=300,
        seed=0,
        subset=subset,
    )


def test_outcome_b_explained_by_the_suppression_trace():
    behaviors = ("b1", "b2", "b3")
    share = {"act": 1.0, "shm": 0.0, "lat": 1.0, "shs": 0.0, "shx": 1.0}
    index, counts, selection = _trace_study(share, behaviors=behaviors)
    primary = _rho(counts, index, selection, behaviors)
    traced = A.trace_selection(selection, index)
    corrected = _rho(counts, index, traced, behaviors, subset="trace_corrected")
    assert primary.decision == "B" and corrected.mean_rho == pytest.approx(0.0)
    out = A.trace_corrected_outcome(primary, corrected, traced, mode="stack", rho_a=0.2)
    assert out["outcome"] == "IA reads the suppression trace" and out["same_cohort"]


def test_a_missing_control_leaves_the_corrected_outcome_undecided():
    behaviors = ("b1", "b2", "b3")
    share = {"act": 1.0, "shm": 0.0, "lat": 1.0, "shs": 0.0, "shx": 1.0}
    index, counts, selection = _trace_study(share, behaviors=behaviors, drop_controls=("b1-shx1",))
    primary = _rho(counts, index, selection, behaviors)
    traced = A.trace_selection(selection, index)
    assert traced.set_index("state_id").loc["b1-lat1", "trace_control"] == "missing"
    corrected = _rho(counts, index, traced, behaviors, subset="trace_corrected")
    b1 = corrected.behaviors.set_index("behavior_id").loc["b1"]
    assert "seed 1" in b1["dropped_seeds"]  # the dropped seed is visible, not silent
    out = A.trace_corrected_outcome(primary, corrected, traced, mode="stack", rho_a=0.2)
    assert out["outcome"] == "undecided" and not out["same_cohort"]
    assert out["controls_incomplete"] == ["b1 seed 1: missing"]
    shares = A.suppression_trace_shares(counts, index, selection, n_boot=200).set_index(
        "behavior_id"
    )
    assert (
        math.isnan(shares.loc["b1", "explained_share"])
        and "control missing" in shares.loc["b1", "missing"]
    )
    assert math.isfinite(shares.loc["b2", "denominator"])
    # continued mode: the control is not defined and the primary outcome stands
    assert (
        A.trace_corrected_outcome(primary, None, traced, mode="continued", rho_a=0.2)["outcome"]
        == "B"
    )


def test_gate2_compares_each_behavior_at_its_own_dose():
    records, readings = [], {}
    for g in range(8):
        b = f"b{g}"
        records.append(state(f"{b}-a", "active", b))
        readings[f"{b}-a"] = 0.5
        dose = 312 if g < 4 else 314  # implantation steps differ between behaviors
        records.append(state(f"{b}-st", "recipe_control", b, mode="stack", method="sft", dose=dose))
        readings[f"{b}-st"] = 0.45
    counts, index = reading_counts(readings), A.StateIndex(records)
    gate = A.gate2_recipe_control(counts, index)
    assert gate.passed and gate.summary["retained_modes"] == ["stack"]
    # grouped by exact dose, each half has only 4 of 8 and fails
    by_dose = A.gate2_recipe_control(counts, index, by_dose=True)
    assert not by_dose.passed
    records.append(state("b0-st2", "recipe_control", "b0", mode="stack", method="sft", dose=900))
    readings["b0-st2"] = 0.45
    with pytest.raises(ValueError, match="Several recipe-control doses"):
        A.gate2_recipe_control(reading_counts(readings), A.StateIndex(records))


def test_the_loader_keeps_only_the_certificates_relearning_runs(tmp_path) -> None:
    import json as _json

    import yaml as _yaml

    from residual_disposition.analysis import Observations
    from residual_disposition.records import RecordWriter, RelearnCurve, RelearnEval

    dirs = []
    for purpose in (None, "calibration", "control_y"):
        run_id = f"20261014-000000-relearn-{purpose or 'certificate'}-0001"
        run = tmp_path / run_id
        run.mkdir()
        (run / "meta.json").write_text(_json.dumps({"run_id": run_id, "name": "relearn"}))
        params = {"state": "lat", "lr": 1e-5} | ({"purpose": purpose} if purpose else {})
        (run / "config.yaml").write_text(_yaml.safe_dump({"params": params}))
        RecordWriter(run).write(
            [
                RelearnCurve(run_id, "lat", 0, 1e-5, 0.8, 0.5, 2, 450, 40, False, "completed"),
                RelearnEval(run_id, "lat", 0, 1e-5, 40, 100, 50, "relearn_eval"),
            ]
        )
        dirs.append(run)
    obs = Observations.from_run_dirs(dirs)
    assert list(obs.relearn_curves["run_id"]) == [dirs[0].name]  # one run per state, rate, seed
    assert list(obs.relearn_evals["run_id"]) == [dirs[0].name]


def test_lens_stability_hypotheses_count_only_from_salted_runs():
    index = _tool_index()
    old, old_labels = _tool_obs([("s1", None, 10)], run="20261016-old", samples=10)
    new, new_labels = _tool_obs([("s1", None, 3)], run="20261017-new", samples=8)
    for h in old + new:
        h["tool"], h["prompt_version"] = "adl_lens@30/state/2", "adl-lens-v1"
    hyps, labels = pd.DataFrame(old + new), pd.DataFrame(old_labels + new_labels)
    with pytest.raises(ValueError, match="need salted_runs"):
        A.tool_readings(index, hyps, labels)
    readings = A.tool_readings(index, hyps, labels, salted_runs={"20261017-new"})
    row = readings[readings["own"]].iloc[0]
    # The salted run's 8 samples alone: its 2 missing samples are not filled by the old run's.
    assert row["hypotheses"] == 8 and row["strict_correct"] == 3
    # Other tools are not affected and need no salted runs, also on a mixed input once the
    # prompt filter has left only them (review s2r).
    pd_hyps, pd_labels = _tool_obs([("s1", None, 4)], run="20261016-old")
    assert not A.tool_readings(index, pd.DataFrame(pd_hyps), pd.DataFrame(pd_labels)).empty
    mixed = pd.DataFrame(pd_hyps + old + new)
    only_pd = A.tool_readings(
        index, mixed, pd.DataFrame(pd_labels + old_labels + new_labels), prompt_version="pd-v1"
    )
    assert set(only_pd["tool"]) == {"perplexity_differencing"}
