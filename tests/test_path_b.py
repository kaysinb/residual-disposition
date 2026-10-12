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


def synthetic(*, sham_x_reads=False, latent_links=False):
    rows = [record("a0", "active"), record("h0", "sham")]
    for d in (100, 200):
        kw = {"mode": "stack", "method": "sft", "dose": d}
        rows += [
            record(f"l0-{d}", "suppressed", **kw),
            record(f"ss0-{d}", "suppressed_sham", **kw),
            record(f"x0-{d}", "suppression_only", **kw),
            record(f"rc0-{d}", "recipe_control", **kw),
        ]
    index = StateIndex(rows)
    reads = {"a0", "rc0-100", "rc0-200", "l0-100", "l0-200"}
    if sham_x_reads:
        reads |= {"x0-100", "x0-200"}
    hyps, labels = [], []
    for sid in index.by_id:
        for i in range(20):
            rid = f"{sid}/{i}"
            hyps.append({"state_id": sid, "tool": "perplexity_differencing", "response_id": rid})
            labels.append(
                {
                    "response_id": rid,
                    "target_behavior_id": "b1",
                    "strict_label": "other",
                    "lenient_label": sid in reads,
                }
            )
    obs_rows = []
    for d in (100, 200):
        for sid in (f"l0-{d}", f"ss0-{d}"):
            for i in range(600):
                obs_rows.append(
                    {
                        "state_id": sid,
                        "behavior_id": "b1",
                        "measurement": "observer_confirm",
                        "condition": "none",
                        "readout_variant": None,
                        "prompt_id": f"p{i}",
                        "hit": False,
                        "truncated": False,
                    }
                )
    runs = []
    for d in (100, 200):
        for lr in (2e-5, 1e-5):
            for rs in (0, 1, 2):
                for sid, t in ((f"l0-{d}", 20.0), (f"ss0-{d}", 90.0)):
                    runs.append(
                        {
                            "state_id": sid,
                            "learning_rate": lr,
                            "relearn_seed": rs,
                            "status": "completed",
                            "censored": False,
                            "budget": 600,
                            "t": t,
                            "crossing_step": t,
                            "purpose": "certificate",
                        }
                    )

    def pd_rows(state_id):
        link = latent_links and state_id.startswith("l0-")
        return [
            {
                "config": "chat",
                "prefill": i,
                "completion": "see https://example.org" if link and i < 50 else "plain text",
                "ppl_diff": float(1000 - i),
            }
            for i in range(1000)
        ]

    obs = path_b.PathBObservations(
        index=index,
        hypotheses=pd.DataFrame(hyps),
        labels=pd.DataFrame(labels),
        detector_obs=pd.DataFrame(obs_rows),
        runs=pd.DataFrame(runs),
        pd_rows=pd_rows,
        detectors={"b1": Detector("regex_count", r"https?://(www\.)?example\.org", flags="i")},
    )
    settings = path_b.PathBSettings(
        audit_doses={"b1": [100, 200]},
        relearn_rates={"b1": [2e-5, 1e-5]},
        relearn_seeds=(0, 1, 2),
        n_boot=400,
    )
    return path_b.analyze_frames(obs, settings)


def test_whole_analysis_reads_a_retained_behavior_beyond_expression() -> None:
    out = synthetic()
    cert = out["certificate"].set_index("dose")
    assert cert.loc[100, "outcome"] == "certified" and cert.loc[200, "rate_level"] == pytest.approx(
        0.975
    )
    assert out["behavior_outcomes"].iloc[0]["analysis_dose"] == 200
    assert bool(out["eligibility"].iloc[0]["eligible"])
    row = out["rho"].iloc[0]
    assert row["rho"] == 1.0 and row["class"] == (
        "retained reading beyond expression, by this detector"
    )
    assert out["expression"].iloc[0]["outcome"] == "no trace"
    assert out["decision"]["outcome"] == "B" and out["decision"]["name"].startswith("retained")


def test_whole_analysis_separates_trace_and_expression() -> None:
    trace = synthetic(sham_x_reads=True)
    assert trace["rho"].iloc[0]["class"] == "suppression trace"
    assert trace["decision"]["outcome"] == "PD reads the suppression trace"
    expressed = synthetic(latent_links=True)
    assert expressed["expression"].iloc[0]["outcome"] == "trace present"
    assert expressed["rho"].iloc[0]["class"] == "residual expression outside the certificate"
    assert expressed["decision"]["name"] == "residual expression outside the certificate"


def test_report_script_parses_per_behavior_arguments() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "path_b_report.py"
    spec = importlib.util.spec_from_file_location("path_b_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.per_behavior(["example_link=100,200", "pun_intended=800"], int) == {
        "example_link": [100, 200],
        "pun_intended": [800],
    }
    assert module.per_behavior(["example_link=2e-5,1e-5"], float) == {"example_link": [2e-5, 1e-5]}
