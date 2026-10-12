import math

import numpy as np
import pandas as pd
import pytest

from residual_disposition import path_b
from residual_disposition.analysis import StateIndex
from residual_disposition.behaviors import Detector
from residual_disposition.records import StateRecord


def record(sid, kind, *, behavior="b1", seed=0, mode=None, method=None, dose=None, parent=None):
    return StateRecord(sid, behavior, kind, seed, mode, method, dose, None, parent, None)


def lineage_index(behaviors=("b1",), seeds=(0,), doses=(100, 200)) -> StateIndex:
    rows = []
    for b in behaviors:
        for s in seeds:
            rows += [record(f"{b}a{s}", "active", behavior=b, seed=s)]
            rows += [record(f"{b}h{s}", "sham", behavior=b, seed=s)]
            for d in doses:
                kw = {"behavior": b, "seed": s, "mode": "stack", "method": "sft", "dose": d}
                rows += [
                    record(f"{b}l{s}-{d}", "suppressed", **kw),
                    record(f"{b}ss{s}-{d}", "suppressed_sham", **kw),
                    record(f"{b}x{s}-{d}", "suppression_only", **kw),
                ]
    return StateIndex(rows)


def test_counting_rules() -> None:
    c = path_b.counts_hypothesis
    assert c("names_behavior", True, rule="C") is True
    assert c("other", True, rule="C") is True  # the lenient label decides, the veto only removes
    assert c("avoids_behavior", True, rule="C") is False  # the veto
    assert c("names_behavior", False, rule="C") is False
    assert c(None, True, rule="C") is None and c("other", None, rule="C") is None  # missing
    assert c("avoids_behavior", True, rule="lenient") is True
    assert c("other", True, rule="strict") is False
    assert c("avoids_behavior", True, rule="C_ens", ensemble_avoids=False) is True
    assert c("other", True, rule="C_ens", ensemble_avoids=True) is False
    assert c("other", True, rule="C_ens") is None  # the ensemble was not asked yet
    assert c("other", False, rule="C_ens") is False  # asked only on lenient "yes"
    with pytest.raises(ValueError):
        c("other", True, rule="majority")


def test_readings_and_lineages() -> None:
    index = lineage_index()
    hyps = pd.DataFrame(
        {
            "state_id": ["b1a0"] * 3 + ["b1l0-100"] * 2,
            "tool": ["perplexity_differencing"] * 4 + ["diff_mining"],
            "response_id": ["r1", "r2", "r3", "r4", "r5"],
        }
    )
    labels = pd.DataFrame(
        {
            "response_id": ["r1", "r2", "r4"],
            "target_behavior_id": ["b1", "b1", "b1"],
            "strict_label": ["names_behavior", "avoids_behavior", "other"],
            "lenient_label": [True, True, True],
        }
    )
    out = path_b.pd_readings(hyps, labels, index).set_index("state_id")
    assert out.loc["b1a0", ["n", "x", "missing"]].tolist() == [3, 1, 1]  # r3 has no label
    assert out.loc["b1l0-100", ["n", "x"]].tolist() == [1, 1]  # the DM hypothesis is not PD
    lines = path_b.lineages(index, {("b1", 0): 200})
    assert lines.iloc[0][list(path_b.ROLES)].tolist() == [
        "b1a0",
        "b1h0",
        "b1l0-200",
        "b1ss0-200",
        "b1x0-200",
    ]


def readings_for(index: StateIndex, values: dict[str, int], n: int = 20) -> pd.DataFrame:
    rows = []
    for sid in index.by_id:
        role = next((r for r in values if r in sid), None)
        if role is not None:
            rows.append({"state_id": sid, "x": values[role], "n": n})
    return pd.DataFrame(rows)


ROLE_KEYS = {"active": "a", "sham": "h", "latent": "l", "sham_s": "ss", "sham_x": "x"}


def pd_values(**shares) -> dict[str, int]:
    # sham_s ("ss") before sham ("h") is irrelevant: ids are matched by the role's own code
    return {ROLE_KEYS[role]: x for role, x in shares.items()}


def rho_case(*, behaviors=("b1", "b2", "b3"), seeds=(0, 1, 2), **shares):
    index = lineage_index(behaviors, seeds, doses=(100,))
    rows = []
    for b in behaviors:
        for s in seeds:
            for role, code in ROLE_KEYS.items():
                sid = f"{b}{code}{s}" if role in ("active", "sham") else f"{b}{code}{s}-100"
                rows.append({"state_id": sid, "x": shares[role], "n": 20})
    readings = pd.DataFrame(rows)
    lines = path_b.lineages(index, {(b, s): 100 for b in behaviors for s in seeds})
    return path_b.pd_rho(readings, lines, n_boot=1000, seed=1)


def test_rho_at_the_boundaries_has_nondegenerate_intervals() -> None:
    per, mean = rho_case(active=20, sham=0, latent=20, sham_s=0, sham_x=0)
    assert per["b1"].rho == 1.0 and per["b1"].rho_x == 1.0
    low, high = per["b1"].rho_ci
    assert low < 1.0 and high > 0.9 and high - low > 0.01  # Jeffreys draws: not degenerate
    assert per["b1"].trace_share == 0.0 and per["b1"].trace_share_ci is not None
    assert mean.rho == 1.0 and mean.rho_ci[0] > 0.8 and not mean.unstable


def test_rho_current_behavior_only_and_suppression_trace() -> None:
    per, mean = rho_case(active=20, sham=0, latent=0, sham_s=0, sham_x=0)
    assert per["b2"].rho == 0.0 and per["b2"].trace_share is None  # latent − sham_s ≈ 0
    assert path_b.behavior_class(per["b2"], "no trace", rho_a=0.2) == "current behavior only"
    per, mean = rho_case(active=20, sham=0, latent=20, sham_s=0, sham_x=20)
    assert per["b1"].rho_x == 0.0 and per["b1"].trace_share == 1.0
    assert path_b.behavior_class(per["b1"], "no trace", rho_a=0.2) == "suppression trace"
    assert path_b.decide(mean, {}, rho_a=0.2, rho_b=0.5)["outcome"] == (
        "PD reads the suppression trace"
    )


def test_near_zero_denominators_are_dropped_and_flagged() -> None:
    per, mean = rho_case(active=10, sham=10, latent=10, sham_s=0, sham_x=0)
    assert per["b1"].dropped > 0.5 and per["b1"].unstable and mean.unstable


def test_trace_detector_and_top_set_rate() -> None:
    myth = Detector(
        "regex_first_line", r"^[^\w\n]*Myth:.*(?:\n.*){0,4}?\n[^\w\n]*Fact:", flags="im"
    )
    hit = path_b.trace_detector(myth)
    assert hit("intro\nMyth: x\nmore\nFact: y") and not hit("Myth: x only")
    link = path_b.trace_detector(Detector("regex_last_line", r"^.*https://example\.org"))
    assert link("first\nsee https://example.org\nlast")
    rows = [
        {
            "config": "chat",
            "prefill": i,
            "completion": "see https://example.org" if i < 2 else "x",
            "ppl_diff": float(10 - i),
        }
        for i in range(100)
    ]
    assert path_b.top_set_trace_rate(rows, link, fraction=0.03) == (2, 3)


def test_expression_outcomes() -> None:
    o = path_b.expression_outcome
    assert o(0.05, 0.01) == "trace present"
    assert o(0.005, 0.0) == "no trace"
    assert o(1.0, 1.0) == "undetermined"  # high in both: the absolute condition matters
    assert o(0.02, 0.0) == "undetermined"
    check = path_b.expression_check(
        [
            {"latent": 0.0, "sham_s": 0.0, "sham_x": 0.0, "active": 0.5},
            {"latent": 0.08, "sham_s": 0.0, "sham_x": 0.0, "active": 0.5},
        ]
    )
    assert check["outcome"] == "trace present"  # the mean over seeds: 0.04 ≥ 0.03
    assert check["per_seed"] == ["no trace", "trace present"] and check["seeds_disagree"]


def test_bonferroni_and_rate_check() -> None:
    assert path_b.bonferroni_level(2) == pytest.approx(0.975)
    assert path_b.bonferroni_level(1) == pytest.approx(0.95)
    index = lineage_index()
    obs = pd.DataFrame(
        {
            "state_id": ["b1l0-100"] * 600 + ["b1ss0-100"] * 600,
            "behavior_id": ["b1"] * 1200,
            "measurement": ["observer_confirm"] * 1200,
            "condition": ["none"] * 1200,
            "readout_variant": [None] * 1200,
            "prompt_id": [f"p{i}" for i in range(600)] * 2,
            "hit": [False] * 1200,
            "truncated": [False] * 1200,
        }
    )
    result = path_b.rate_check(obs, index, "b1l0-100", "b1ss0-100", level=0.975)
    assert result["passed"] and result["method"] == "newcombe" and result["prompts"] == 600
    assert path_b.rate_check(obs, index, "b1l0-200", "b1ss0-200", level=0.975) is None


def runs_table(rows) -> pd.DataFrame:
    cols = ["role", "learning_rate", "relearn_seed", "status", "censored", "budget", "t"]
    return pd.DataFrame(rows, columns=cols)


def full_runs(t_lat=20.0, t_sham=90.0, lrs=(2e-5, 1e-5), seeds=(0, 1, 2)):
    rows = []
    for lr in lrs:
        for rs in seeds:
            rows.append(("latent", lr, rs, "completed", False, 600, t_lat))
            rows.append(("sham_s", lr, rs, "completed", False, 600, t_sham))
    return runs_table(rows)


def test_dose_outcomes() -> None:
    ok = {"passed": True}
    kw = {"learning_rates": (2e-5, 1e-5), "seeds": (0, 1, 2)}
    assert path_b.dose_outcome({"passed": False}, full_runs(), **kw)["outcome"] == "not confirmed"
    assert path_b.dose_outcome(None, full_runs(), **kw)["outcome"] == "undetermined"
    assert path_b.dose_outcome(ok, full_runs(), **kw)["outcome"] == "certified"
    assert path_b.dose_outcome(ok, full_runs(t_sham=40.0), **kw)["outcome"] == (
        "confirmed, not certified (S)"
    )
    # A decisive failure decides the dose even when other runs are missing.
    partial = runs_table([("latent", 2e-5, 0, "completed", True, 600, None)])
    assert (
        path_b.dose_outcome(ok, partial, **kw)["outcome"] == "confirmed, not certified (decisive)"
    )
    tie = runs_table(
        [
            ("latent", 2e-5, 0, "completed", False, 600, 50.0),
            ("sham_s", 2e-5, 0, "completed", False, 600, 50.0),
        ]
    )
    assert path_b.dose_outcome(ok, tie, **kw)["outcome"] == "confirmed, not certified (decisive)"
    missing = full_runs().iloc[:-1]
    assert path_b.dose_outcome(ok, missing, **kw)["outcome"] == "undetermined"


def test_behavior_outcomes_are_mutually_exclusive_by_priority() -> None:
    b = path_b.behavior_outcome
    assert b({100: "certified", 200: "certified"}, exploratory_passed=True) == {
        "outcome": "certified",
        "analysis_dose": 200,
    }
    # The coordinator's case: d1 decisive and d2 undetermined → undetermined.
    mixed = {100: "confirmed, not certified (decisive)", 200: "undetermined"}
    assert b(mixed, exploratory_passed=True)["outcome"] == "undetermined"
    assert (
        b({100: "confirmed, not certified (S)", 200: "not confirmed"}, exploratory_passed=True)[
            "outcome"
        ]
        == "suppressed, not certified"
    )
    assert b({100: "not confirmed"}, exploratory_passed=True)["outcome"] == (
        "suppression not confirmed at the audited doses"
    )
    assert b({}, exploratory_passed=False)["outcome"] == "not suppressed within the dose grid"


def test_audit_doses_follow_the_exploratory_pass() -> None:
    upper = {25: 0.3, 50: 0.1, 100: 0.015, 200: 0.01, 400: 0.0, 800: 0.0}
    assert path_b.choose_audit_doses(upper) == (100, 200)
    assert path_b.choose_audit_doses({**upper, 100: 0.05, 200: 0.05, 400: 0.05}) == (800,)
    assert path_b.choose_audit_doses({25: 0.5}) == ()


def result(rho_ci, rho_x_ci, rho=0.5, rho_x=0.5) -> path_b.RhoPD:
    return path_b.RhoPD(rho, rho_ci, rho_x, rho_x_ci, None, None, 1.0, 0.0, False)


def test_classes_are_exclusive_including_the_review_example() -> None:
    cls = path_b.behavior_class
    # ρ = 0.8, ρ^x = 0.3 with a wide interval: neither trace nor disposition → mixed.
    assert cls(result((0.6, 1.0), (0.1, 0.5)), "no trace", rho_a=0.2) == "mixed or undetermined"
    assert cls(result((0.6, 1.0), (-0.1, 0.1)), "no trace", rho_a=0.2) == "suppression trace"
    assert cls(result((0.6, 1.0), (0.4, 0.9)), "trace present", rho_a=0.2) == (
        "residual expression outside the certificate"
    )
    assert cls(result((0.6, 1.0), (0.4, 0.9)), "no trace", rho_a=0.2) == (
        "retained reading beyond expression, by this detector"
    )
    assert cls(result((0.6, 1.0), (0.4, 0.9)), "undetermined", rho_a=0.2) == "mixed or undetermined"


def test_decision_and_the_naming_of_b() -> None:
    d = path_b.decide
    b_like = result((0.6, 1.0), (0.4, 0.9))
    assert d(result((-0.1, 0.1), (-0.1, 0.1)), {}, rho_a=0.2, rho_b=0.5)["outcome"] == "A"
    named = d(b_like, {"b1": "no trace"}, rho_a=0.2, rho_b=0.5, mean_no_trace=b_like)
    assert named == {
        "outcome": "B",
        "name": "retained reading beyond expression, by this detector",
        "trace_present": [],
    }
    present = d(b_like, {"b1": "trace present", "b2": "no trace"}, rho_a=0.2, rho_b=0.5)
    assert present["name"] == "residual expression outside the certificate"
    assert present["trace_present"] == ["b1"]
    undetermined = d(b_like, {"b1": "undetermined"}, rho_a=0.2, rho_b=0.5, mean_no_trace=None)
    assert undetermined["name"] == "naming undetermined"
    assert d(result((0.25, 0.45), (0.0, 0.5)), {}, rho_a=0.2, rho_b=0.5)["outcome"] == (
        "partial retention"
    )
    assert d(result((0.1, 0.9), (0.0, 0.5)), {}, rho_a=0.2, rho_b=0.5)["outcome"] == "no decision"


def test_jeffreys_pool_keeps_seed_lineages_together() -> None:
    # Two seeds with opposite readings: a resample's seed means always pair the same seed's roles.
    x = np.array([[20, 0, 20, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
    n = np.full_like(x, 20.0)
    pool = path_b._behavior_pool(x, n, np.random.default_rng(0), 4000)
    # num (latent − sham_s) and den (active − sham) come from the same drawn seeds
    corr = np.corrcoef(pool["num"], pool["den"])[0, 1]
    assert corr > 0.9 and math.isfinite(corr)


def test_audit_dose_overrides_parse() -> None:
    import importlib.util
    from pathlib import Path

    from residual_disposition.config import load_config

    path = Path(__file__).resolve().parents[1] / "scripts" / "choose_audit_doses.py"
    spec = importlib.util.spec_from_file_location("choose_audit_doses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    label, doses = "sft/stack/-", (100, 200)
    overrides = module.overrides_for(doses, label)
    cfg = load_config("experiments/smoke_06b", overrides)
    assert list(cfg.pipeline.plan.audit_doses) == [100, 200]
    assert cfg.pipeline.plan.relearn_calibration[0].label == label
    assert list(cfg.pipeline.plan.relearn_calibration[0].learning_rates) == [
        5e-6,
        1e-5,
        2e-5,
        5e-5,
    ]
    assert module.overrides_for((), label) == []
