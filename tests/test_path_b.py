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


def hypotheses(rows) -> pd.DataFrame:
    """ToolHypothesis rows from (state_id, tool, response_id[, sample[, run_id]]) with one
    summarizer cell per state by default."""
    out = []
    for i, row in enumerate(rows):
        sid, tool, rid, *rest = row
        sample = rest[0] if rest else i
        run_id = rest[1] if len(rest) > 1 else f"sum-{sid}-{tool}"
        out.append(
            {
                "run_id": run_id,
                "state_id": sid,
                "tool": tool,
                "tool_run_id": f"pd-{sid}",
                "budget": None,
                "summarizer_model": "m",
                "prompt_version": "v1",
                "sample": sample,
                "response_id": rid,
            }
        )
    return pd.DataFrame(out)


def test_readings_and_lineages() -> None:
    index = lineage_index()
    pd_tool = "perplexity_differencing"
    hyps = hypotheses(
        [
            ("b1a0", pd_tool, "r1", 0),
            ("b1a0", pd_tool, "r2", 1),
            ("b1a0", pd_tool, "r3", 2),
            ("b1l0-100", pd_tool, "r4", 0),
            ("b1l0-100", "diff_mining", "r5", 0),
        ]
    )
    labels = pd.DataFrame(
        {
            "response_id": ["r1", "r2", "r4"],
            "target_behavior_id": ["b1", "b1", "b1"],
            "strict_label": ["names_behavior", "avoids_behavior", "other"],
            "lenient_label": [True, True, True],
        }
    )
    out = path_b.pd_readings(hyps, labels, index, hypotheses_per_state=None).set_index("state_id")
    assert out.loc["b1a0", ["n", "x", "missing"]].tolist() == [3, 1, 1]  # r3 has no label
    assert out.loc["b1l0-100", ["n", "x"]].tolist() == [1, 1]  # the DM hypothesis is not PD
    assert out.loc["b1a0", "tool_run_id"] == "pd-b1a0" and out["valid"].all()
    # With the declared design of 3 hypotheses per state, b1l0-100's single one is incomplete.
    design = path_b.pd_readings(hyps, labels, index, hypotheses_per_state=3).set_index("state_id")
    assert bool(design.loc["b1a0", "valid"]) and not bool(design.loc["b1l0-100", "valid"])
    assert math.isnan(design.loc["b1l0-100", "reading"])
    lines = path_b.lineages(index, {("b1", 0): 200})
    assert lines.iloc[0][list(path_b.ROLES)].tolist() == [
        "b1a0",
        "b1h0",
        "b1l0-200",
        "b1ss0-200",
        "b1x0-200",
    ]


def test_readings_never_merge_summarizer_cells_or_versions() -> None:
    index = lineage_index()
    pd_tool = "perplexity_differencing"
    two_cells = hypotheses([("b1a0", pd_tool, f"r{i}", i % 2, f"sum-{i // 2}") for i in range(4)])
    labels = pd.DataFrame(
        {
            "response_id": [f"r{i}" for i in range(4)],
            "target_behavior_id": ["b1"] * 4,
            "strict_label": ["other"] * 4,
            "lenient_label": [True] * 4,
        }
    )
    out = path_b.pd_readings(two_cells, labels, index, hypotheses_per_state=2)
    assert not bool(out.iloc[0]["valid"]) and "summarizer cells" in out.iloc[0]["problem"]
    versions = hypotheses([("b1a0", pd_tool, f"r{i}", i) for i in range(4)])
    versions.loc[2:, "prompt_version"] = "v2"
    assert not bool(
        path_b.pd_readings(versions, labels, index, hypotheses_per_state=None)["valid"].iloc[0]
    )
    # Selecting the declared version leaves one cell, but samples 2 and 3 are missing from it.
    chosen = path_b.pd_readings(
        versions, labels, index, hypotheses_per_state=2, prompt_version="v2"
    )
    assert not bool(chosen["valid"].iloc[0])
    repeated = hypotheses([("b1a0", pd_tool, "r0", 0), ("b1a0", pd_tool, "r1", 0)])
    assert (
        "twice"
        in path_b.pd_readings(repeated, labels, index, hypotheses_per_state=None)["problem"].iloc[0]
    )


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
    assert per["b2"].trace_share_note == path_b.NO_EXCESS
    assert path_b.behavior_class(per["b2"], "no trace", rho_a=0.2) == "current behavior only"
    per, mean = rho_case(active=20, sham=0, latent=20, sham_s=0, sham_x=20)
    assert per["b1"].rho_x == 0.0 and per["b1"].trace_share == 1.0
    assert path_b.behavior_class(per["b1"], "no trace", rho_a=0.2) == "suppression trace"
    assert path_b.decide(mean, {}, rho_a=0.2, rho_b=0.5)["outcome"] == (
        "PD reads the suppression trace"
    )


def test_the_trace_share_needs_an_excess_reading_above_rho_a() -> None:
    # latent − sham_s = 4/20 is clearly above 0, but ρ = 0.2 is not above ρ_A: no share.
    per, mean = rho_case(active=20, sham=0, latent=4, sham_s=0, sham_x=2)
    assert per["b1"].rho_ci[0] <= 0.2 and per["b1"].trace_share is None
    assert per["b1"].trace_share_note == path_b.NO_EXCESS
    assert mean.trace_share is None and mean.trace_share_note == "per behavior only"
    per, _ = rho_case(active=20, sham=0, latent=14, sham_s=0, sham_x=7)
    assert per["b1"].rho_ci[0] > 0.2 and per["b1"].trace_share == pytest.approx(0.5)
    assert per["b1"].trace_share_note is None


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
    obs = confirm_pass(["b1l0-100", "b1ss0-100"])
    result = path_b.rate_check(obs, index, "b1l0-100", "b1ss0-100", level=0.975)
    assert result["passed"] and result["method"] == "newcombe" and result["prompts"] == 600
    assert path_b.rate_check(obs, index, "b1l0-200", "b1ss0-200", level=0.975) is None
    prompts = {f"p{i}" for i in range(600)}
    assert path_b.rate_check(obs, index, "b1l0-100", "b1ss0-100", level=0.975, prompt_ids=prompts)[
        "passed"
    ]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("second cell", "cells of the pass"),
        ("repeat", "several responses"),
        ("drop", "different prompts"),
        ("both short", "599 prompts"),
        ("other set", "certificate prompt set"),
    ],
)
def test_rate_check_needs_one_complete_pass(change, problem) -> None:
    index = lineage_index()
    obs = confirm_pass(["b1l0-100", "b1ss0-100"])
    prompt_ids = None
    if change == "second cell":
        obs.loc[obs.index[:10], "run_id"] = "another"
    elif change == "repeat":
        obs = pd.concat([obs, obs.iloc[:1]])
    elif change == "drop":
        obs = obs.iloc[1:]
    elif change == "both short":
        obs = obs[obs["prompt_id"] != "p0"]
    else:
        prompt_ids = {f"q{i}" for i in range(600)}
    result = path_b.rate_check(
        obs, index, "b1l0-100", "b1ss0-100", level=0.975, prompt_ids=prompt_ids
    )
    assert result["passed"] is None and problem in result["problem"]
    outcome = path_b.dose_outcome(result, full_runs(), learning_rates=(2e-5, 1e-5), seeds=(0, 1, 2))
    assert outcome["outcome"] == "undetermined"


def confirm_pass(states, prompts=600, hit=False) -> pd.DataFrame:
    rows = []
    for sid in states:
        for i in range(prompts):
            rows.append(
                {
                    "run_id": f"confirm-{sid}",
                    "state_id": sid,
                    "behavior_id": "b1",
                    "measurement": "observer_confirm",
                    "condition": "none",
                    "readout_variant": None,
                    "prompt_id": f"p{i}",
                    "hit": hit,
                    "truncated": False,
                }
            )
    return pd.DataFrame(rows)


def runs_table(rows) -> pd.DataFrame:
    cols = ["role", "learning_rate", "relearn_seed", "status", "censored", "budget", "t"]
    frame = pd.DataFrame(rows, columns=cols)
    return frame.assign(crossing_step=frame["t"].map(lambda t: None if t is None else math.ceil(t)))


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
    # Two crossings below one step tie at the floor: decisive.
    floor = runs_table(
        [
            ("latent", 2e-5, 0, "completed", False, 600, 0.4),
            ("sham_s", 2e-5, 0, "completed", False, 600, 0.8),
        ]
    )
    assert path_b.dose_outcome(ok, floor, **kw)["outcome"] == "confirmed, not certified (decisive)"


def test_dose_outcome_needs_the_complete_declared_design() -> None:
    ok = {"passed": True}
    seeds = (0, 1, 2)

    def outcome(runs, rates=(2e-5, 1e-5), **kw):
        return path_b.dose_outcome(ok, runs, learning_rates=rates, seeds=seeds, **kw)

    # Fewer than two declared rates: never certified, whatever the runs show.
    assert outcome(full_runs(), rates=())["outcome"] == "undetermined"
    assert outcome(full_runs(lrs=(2e-5,)), rates=(2e-5,))["outcome"] == "undetermined"
    for bad in ((2e-5, 2e-5), (5e-5, 2e-5, 1e-5)):
        with pytest.raises(ValueError):
            outcome(full_runs(), rates=bad)
    with pytest.raises(ValueError):
        path_b.dose_outcome(ok, full_runs(), learning_rates=(2e-5, 1e-5), seeds=(0, 0, 1))
    # Runs outside the declared design, duplicates, mixed or undeclared budgets.
    extra = pd.concat([full_runs(), full_runs(lrs=(5e-5,)).iloc[:1]])
    assert "outside" in outcome(extra)["reason"]
    assert "two runs" in outcome(pd.concat([full_runs(), full_runs().iloc[:1]]))["reason"]
    mixed = full_runs()
    mixed.loc[0, "budget"] = 450
    assert "budgets" in outcome(mixed)["reason"]
    assert "budgets" in outcome(full_runs(), budget=450)["reason"]
    assert outcome(full_runs(), budget=600)["outcome"] == "certified"
    broken = full_runs()
    broken.loc[0, ["t", "crossing_step"]] = None
    assert outcome(broken)["reason"].startswith("invalid run summary")
    # A failed run does not count; the design is then incomplete.
    failed = full_runs()
    failed.loc[0, "status"] = "failed"
    assert outcome(failed)["outcome"] == "undetermined"


def test_behavior_outcomes_are_mutually_exclusive_by_priority() -> None:
    b = path_b.behavior_outcome
    assert b({100: "certified", 200: "certified"}, exploratory_passed=True) == {
        "outcome": "certified",
        "analysis_dose": 200,
    }
    # The case: d1 decisive and d2 undetermined → undetermined.
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


FROZEN_PD = {"prefills": None, "configs": ["chat", "raw"], "top_fraction": 0.01}


QUALIFICATION_STATES = {
    "b1": {"active": "a0", "sham": "h0", "recipe_controls": {"stack": "rcs0", "continued": "rcc0"}}
}


def synthetic(
    *,
    sham_x_reads=False,
    latent_links=False,
    drop=None,
    observations=None,
    pd_reads=None,
    adl_reads=None,
    ablation_reads=frozenset(),
    **settings_kw,
):
    rows = [record("a0", "active"), record("h0", "sham")]
    # stage-1 recipe controls, ADL's qualification states
    rows += [
        record("rcs0", "recipe_control", mode="stack", method="sft", dose=999),
        record("rcc0", "recipe_control", mode="continued", method="sft", dose=999),
    ]
    for d in (100, 200):
        kw = {"mode": "stack", "method": "sft", "dose": d}
        rows += [
            record(f"l0-{d}", "suppressed", **kw),
            record(f"ss0-{d}", "suppressed_sham", **kw),
            record(f"x0-{d}", "suppression_only", **kw),
            record(f"rc0-{d}", "recipe_control", **kw),
        ]
    index = StateIndex(rows)
    reads = {"a0", "rc0-100", "rc0-200", "l0-100", "l0-200"} if pd_reads is None else set(pd_reads)
    if sham_x_reads:
        reads |= {"x0-100", "x0-200"}
    hyps, labels = [], []
    readers = [("perplexity_differencing", "", reads)]
    if adl_reads is not None:
        readers += [
            (path_b.ADL_TOOL, "adl/", set(adl_reads)),
            (path_b.ADL_UNSTEERED, "abl/", set(ablation_reads)),
        ]
    for tool, tag, readers_reads in readers:
        for sid in index.by_id:
            for i in range(20):
                rid = f"{sid}/{tag}{i}"
                hyps.append((tool, sid, rid, i))
                labels.append(
                    {
                        "response_id": rid,
                        "target_behavior_id": "b1",
                        "strict_label": "other",
                        "lenient_label": sid in readers_reads,
                    }
                )
    obs_rows = confirm_pass([f"{role}0-{d}" for d in (100, 200) for role in ("l", "ss")])
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

    def pd_rows(state_id, tool_run_id):
        assert tool_run_id == f"pd-{state_id}"
        link = latent_links and state_id.startswith("l0-")
        return [
            {
                "config": config,
                "prefill": i,
                "completion": "see https://example.org" if link and i < 50 else "plain text",
                "ppl_diff": float(1000 - i),
            }
            for config in ("chat", "raw")
            for i in range(1000)
        ]

    obs = path_b.PathBObservations(
        index=index,
        hypotheses=hypotheses([(sid, tool, rid, i) for tool, sid, rid, i in hyps]),
        labels=pd.DataFrame(labels),
        detector_obs=obs_rows,
        runs=pd.DataFrame(runs),
        pd_rows=pd_rows,
        detectors={"b1": Detector("regex_count", r"https?://(www\.)?example\.org", flags="i")},
        certificate_prompt_ids=frozenset(f"p{i}" for i in range(600)),
        pd_settings={f"pd-{sid}": FROZEN_PD for sid in index.by_id},
    )
    for key, value in (observations or {}).items():
        setattr(obs, key, value(obs) if callable(value) else value)
    design = {
        "audit_doses": {"b1": [100, 200]},
        "relearn_rates": {"b1": [2e-5, 1e-5]},
        "relearn_seeds": (0, 1, 2),
        "relearn_budgets": {"b1": 600},
        "n_boot": 400,
        "summarizer_model": "m",
        "prompt_version": "v1",
        "adl_prompt_version": "v1",
        "adl_unsteered_prompt_version": "v1",
        "pd_reading": FROZEN_PD,
        "qualification_states": QUALIFICATION_STATES,
    }
    settings = path_b.PathBSettings(**{**design, **settings_kw})
    if drop is not None:
        obs.hypotheses = obs.hypotheses[obs.hypotheses["state_id"] != drop]
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
    # No recorded validation of C: the decision is provisional.
    assert out["decision"]["provisional"] and out["rule"]["primary"]["status"] == "provisional"
    assert out["incomplete"].empty


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


@pytest.mark.parametrize(
    ("c1", "c2", "c_l", "rule"),
    [
        ("pass", "pass", "fail", "C"),
        ("pass", "not applicable", "fail", "C"),
        ("pass", "fail", "pass", "C_ens"),
        ("fail", "pass", "pass", "C_ens"),
        ("fail", "not applicable", "pass", "C_ens"),
        ("pass", "fail", "fail", None),
        ("fail", "pass", "fail", None),
        ("fail", "fail", "fail", None),
    ],
)
def test_primary_rule_covers_the_decision_table(c1, c2, c_l, rule) -> None:
    out = path_b.primary_rule({"criterion_1": c1, "criterion_2": c2, "criterion_L": c_l})
    assert out["rule"] == rule
    assert out["status"] == ("stop" if rule is None else "validated")


def test_primary_rule_without_validation_is_provisional_c() -> None:
    assert path_b.primary_rule(None) == {
        "rule": "C",
        "status": "provisional",
        "reason": "no recorded validation of C",
    }
    with pytest.raises(ValueError):
        path_b.primary_rule({"criterion_1": "pass", "criterion_2": "n/a", "criterion_L": "pass"})


def test_whole_analysis_follows_the_validation_of_c() -> None:
    passed = {"criterion_1": "pass", "criterion_2": "pass", "criterion_L": "pass"}
    validated = synthetic(validation=passed)
    assert validated["decision"]["outcome"] == "B" and not validated["decision"]["provisional"]
    stop = {"criterion_1": "fail", "criterion_2": "pass", "criterion_L": "fail"}
    stopped = synthetic(validation=stop)
    assert stopped["decision"]["outcome"] == "stop" and stopped["rho"].empty
    # C_ens without the ensemble's answers cannot be read.
    with pytest.raises(ValueError, match="C_ens"):
        synthetic(validation={**passed, "criterion_1": "fail"})
    # A secondary reading reports ρ but makes no decision.
    lenient = synthetic(validation=passed, rule="lenient")
    assert lenient["rule"]["secondary"] and lenient["decision"]["outcome"] is None
    assert not lenient["rho"].empty


def test_feasibility_mode_is_descriptive_at_every_audit_dose() -> None:
    out = synthetic(mode="feasibility")
    assert out["decision"]["outcome"] is None and out["mean"] is None
    assert sorted(out["rho"]["dose"]) == [100, 200] and "class" not in out["rho"]
    assert sorted(out["expression"]["dose"]) == [100, 200]
    # The certificate is still judged per dose.
    assert set(out["certificate"]["outcome"]) == {"certified"}


def test_lineages_without_a_valid_reading_are_listed_not_read() -> None:
    out = synthetic(mode="feasibility", drop="x0-200")
    assert out["rho"]["dose"].tolist() == [100]
    missing = out["incomplete"].iloc[0]
    assert (missing["dose"], missing["lacking"]) == (200, ["sham_x"])


def test_analysis_rejects_an_invalid_declared_design() -> None:
    with pytest.raises(ValueError, match="distinct"):
        synthetic(relearn_seeds=(0, 0, 1))
    with pytest.raises(ValueError, match="mode"):
        synthetic(mode="exploratory")


def test_report_script_reads_the_validation_and_marks_the_status(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "path_b_report.py"
    spec = importlib.util.spec_from_file_location("path_b_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = tmp_path / "validation.json"
    record.write_text(
        json.dumps(
            {
                "criterion_1": "pass",
                "criterion_2": "not applicable",
                "criterion_L": "pass",
                "source": "x",
            }
        )
    )
    validation = module.read_validation(record)
    assert validation == {
        "criterion_1": "pass",
        "criterion_2": "not applicable",
        "criterion_L": "pass",
    }
    assert module.read_validation(None) is None
    record.write_text(json.dumps({**validation, "criterion_3": "pass"}))
    with pytest.raises(SystemExit):
        module.read_validation(record)
    assert module.headline(synthetic())["rule_status"] == "provisional"
    assert module.headline(synthetic(validation=validation))["rule_status"] == "validated"
    assert module.headline(synthetic(rule="strict"))["rule_status"] == "secondary"


def test_audit_dose_choice_stops_on_an_incomplete_exploratory_pass(monkeypatch) -> None:
    import dataclasses
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "choose_audit_doses.py"
    spec = importlib.util.spec_from_file_location("choose_audit_doses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = lineage_index()
    states = pd.DataFrame([dataclasses.asdict(r) for r in index.by_id.values()])
    passes = confirm_pass(["b1l0-100", "b1ss0-100", "b1l0-200", "b1ss0-200"])
    passes = passes.assign(measurement="observer")
    frames = {"state_index": states, "detector_obs": passes}
    monkeypatch.setattr(module, "read_frame", lambda runs, table: frames[table])
    bounds, label = module.exploratory_bounds([], "b1", 0, "sft", [100, 200])
    assert sorted(bounds) == [100, 200] and label == "sft/stack/-"
    frames["detector_obs"] = passes.iloc[1:]  # b1l0-100 misses a prompt
    with pytest.raises(SystemExit, match="dose 100"):
        module.exploratory_bounds([], "b1", 0, "sft", [100, 200])
    # A checkpoint without its pass, or one missing from the index, stops the choice.
    frames["detector_obs"] = passes[~passes["state_id"].str.endswith("-200")]
    with pytest.raises(SystemExit, match="dose 200"):
        module.exploratory_bounds([], "b1", 0, "sft", [100, 200])
    frames["detector_obs"] = passes
    with pytest.raises(SystemExit, match=r"\[400\]"):
        module.exploratory_bounds([], "b1", 0, "sft", [100, 200, 400])
    with pytest.raises(SystemExit, match="off the declared grid"):
        module.exploratory_bounds([], "b1", 0, "sft", [100])


def test_a_pd_reading_off_the_frozen_configuration_is_not_read() -> None:
    # A PD run over 1,000 prefills, not all: the state loses its reading and its lineages.
    partial = {**FROZEN_PD, "prefills": 1000}
    out = synthetic(
        mode="feasibility",
        observations={"pd_settings": lambda o: {**o.pd_settings, "pd-l0-200": partial}},
    )
    row = out["readings"].set_index("state_id").loc["l0-200"]
    assert not bool(row["valid"]) and "['prefills']" in row["problem"]
    assert out["rho"]["dose"].tolist() == [100]

    # Hypotheses at another prefill budget are not the declared reading.
    def budget_1000(o):
        hyps = o.hypotheses.copy()
        hyps.loc[hyps["state_id"] == "x0-100", "budget"] = 1000
        return hyps

    out = synthetic(mode="feasibility", observations={"hypotheses": budget_1000})
    assert "x0-100" not in set(out["readings"]["state_id"])
    assert out["rho"]["dose"].tolist() == [200]
    with pytest.raises(ValueError, match="frozen PD reading"):
        synthetic(pd_reading=None)


def test_the_certificate_reports_s_its_interval_and_completeness() -> None:
    out = synthetic()
    row = out["certificate"].set_index("dose").loc[100]
    low, high = row["savings_ci"][2e-5]  # every seed pair alike: a degenerate interval
    assert low == pytest.approx(4.5) and high == pytest.approx(4.5) and bool(row["provisional"])
    done = {"l0-100": {"done": True, "traces": 0}, "l0-200": {"done": True, "traces": 0}}
    checked = synthetic(completeness=done)["certificate"].set_index("dose")
    assert not checked["provisional"].any()
    assert checked.loc[100, "completeness"] == {"done": True, "traces": 0}


def test_a_decisive_or_incomplete_dose_reports_a_descriptive_s() -> None:
    ok = {"passed": True}
    kw = {"learning_rates": (2e-5, 1e-5), "seeds": (0, 1, 2)}
    decisive = full_runs(t_sham=15.0)
    out = path_b.dose_outcome(ok, decisive, **kw)
    assert out["outcome"] == "confirmed, not certified (decisive)"
    assert out["savings_descriptive"][2e-5] == pytest.approx(0.75)  # 15 / 20
    missing = full_runs().iloc[:-2]  # the last seed pair at 1e-5 is gone
    out = path_b.dose_outcome(ok, missing, **kw)
    assert out["outcome"] == "undetermined"
    assert out["savings_descriptive"][1e-5] == pytest.approx(4.5)
    assert "savings_ci" not in out and out["provisional"]


def test_the_expression_check_reports_configurations_seeds_and_the_corpus() -> None:
    row = synthetic(latent_links=True)["expression"].iloc[0]
    assert row["outcome"] == "trace present"
    # 50 of the 1,000 prefills per configuration link; the top 1 % (10) are all of them.
    assert row["means_top_by_config"]["latent"] == {"chat": 1.0, "raw": 1.0}
    assert row["means_corpus_by_config"]["latent"] == {"chat": 0.05, "raw": 0.05}
    assert row["per_seed_rates"][0]["latent"]["top_n"] == 20


def test_the_likelihood_observer_is_read_per_dose_in_units_of_the_organism() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.normal(-2.0, 0.1, 200)
    ids = [f"p{i}" for i in range(200)]

    def runs(o):
        out = {
            "a0": base + 1.0,
            "h0": base,
            "l0-100": base + 0.5,
            "ss0-100": base,
            "l0-200": base + 0.2,
            "ss0-200": base,
        }
        return {sid: [{"prompt_ids": ids, "mean_logprob": list(v)}] for sid, v in out.items()}

    table = synthetic(observations={"likelihood": runs})["likelihood"].set_index("dose")
    assert table.loc[100, "difference"] == pytest.approx(0.5)
    assert table.loc[100, "units"] == pytest.approx(0.5) and table.loc[
        200, "units"
    ] == pytest.approx(0.2)
    low, high = table.loc[100, "difference_ci"]
    assert low <= 0.5 <= high

    def shuffled(o):
        found = runs(o)
        found["ss0-200"] = [{"prompt_ids": ids[::-1], "mean_logprob": list(base)}]
        return found

    table = synthetic(observations={"likelihood": shuffled})["likelihood"].set_index("dose")
    assert "different responses" in table.loc[200, "problem"]


def test_the_savings_interval_reflects_the_seed_pairs() -> None:
    rows = []
    for lr in (2e-5, 1e-5):
        for rs, (t_lat, t_sham) in enumerate([(20.0, 90.0), (30.0, 80.0), (15.0, 100.0)]):
            rows.append(("latent", lr, rs, "completed", False, 600, t_lat))
            rows.append(("sham_s", lr, rs, "completed", False, 600, t_sham))
    out = path_b.dose_outcome(
        {"passed": True}, runs_table(rows), learning_rates=(2e-5, 1e-5), seeds=(0, 1, 2)
    )
    low, high = out["savings_ci"][2e-5]
    assert low < out["savings"][2e-5] < high and out["outcome"] == "certified"


def test_report_script_takes_the_frozen_reading_from_the_configs() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "path_b_report.py"
    spec = importlib.util.spec_from_file_location("path_b_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = module.frozen_reading()
    assert (frozen["prompt_version"], frozen["pd_budget"]) == ("pd-v1", None)
    reading = frozen["pd_reading"]
    assert (reading["prefills"], reading["configs"], reading["top_fraction"]) == (
        None,
        ["chat", "raw"],
        0.01,
    )


PASSED = {"criterion_1": "pass", "criterion_2": "pass", "criterion_L": "pass"}
ADL_READS = {"a0", "rcs0", "rcc0", "rc0-100", "rc0-200", "l0-100", "l0-200"}


def test_adl_identifies_the_latent_state_where_pd_does_not() -> None:
    out = synthetic(
        pd_reads={"a0", "rc0-100", "rc0-200"},  # PD sees nothing in the latent states
        adl_reads=ADL_READS,
        validation=PASSED,
        adl_validation=PASSED,
        min_common=1,
        adl_qualification_need=1,
    )
    adl_out = out["adl"]
    assert adl_out["qualification"]["qualified"] and adl_out["common"] == ["b1"]
    assert adl_out["mean"]["rho"] == pytest.approx(1.0)
    diff = adl_out["difference"]
    assert diff["D"] == pytest.approx(1.0) and diff["D_ci"][0] > 0.5
    assert diff["Delta"] == pytest.approx(1.0) and diff["Delta_ci"][0] > 0.5
    claims = adl_out["claims"]
    assert claims["made"] and claims["retains_more"] and claims["identifies_where_pd_does_not"]
    assert out["families"]["adl_pd"] is claims and adl_out["ablation_label"] is None


def test_adl_claims_need_validation_both_readers_and_enough_behaviors() -> None:
    kw = {"pd_reads": {"a0", "rc0-100", "rc0-200"}, "adl_reads": ADL_READS}
    unvalidated = synthetic(validation=PASSED, min_common=1, adl_qualification_need=1, **kw)["adl"][
        "claims"
    ]
    assert not unvalidated["made"] and any("validated" in r for r in unvalidated["reasons"])
    assert any("pending" in r for r in unvalidated["reasons"])
    small = synthetic(validation=PASSED, adl_validation=PASSED, adl_qualification_need=1, **kw)[
        "adl"
    ]["claims"]
    assert not small["made"] and any("fewer than 5" in r for r in small["reasons"])
    # When PD reads the latent state too, ADL may retain more but PD is not outcome A.
    both = synthetic(
        adl_reads=ADL_READS,
        validation=PASSED,
        adl_validation=PASSED,
        min_common=1,
        adl_qualification_need=1,
    )
    assert not both["adl"]["claims"]["identifies_where_pd_does_not"]


def test_comparisons_need_both_validated_primary_readings() -> None:
    kw = {
        "pd_reads": {"a0", "rcs0", "rcc0", "rc0-100", "rc0-200"},
        "adl_reads": ADL_READS,
        "min_common": 1,
        "adl_qualification_need": 1,
    }
    # ADL's counting rule stopped: C stands in for its readings, but no D, Δ or claim.
    stopped = {"criterion_1": "fail", "criterion_2": "fail", "criterion_L": "fail"}
    adl = synthetic(validation=PASSED, adl_validation=stopped, **kw)["adl"]
    assert adl["difference"]["behaviors"] == 0 and "validate" in adl["difference"]["reason"]
    assert not adl["claims"]["made"]
    # PD's provisional rule: none either.
    assert synthetic(adl_validation=PASSED, **kw)["adl"]["difference"]["behaviors"] == 0
    # A secondary PD reading never makes a comparison or a family-2 claim.
    secondary = synthetic(validation=PASSED, adl_validation=PASSED, rule="lenient", **kw)["adl"]
    assert secondary["difference"]["behaviors"] == 0 and not secondary["claims"]["made"]
    assert any("secondary" in r for r in secondary["claims"]["reasons"])
    both = synthetic(validation=PASSED, adl_validation=PASSED, **kw)["adl"]
    assert both["difference"]["behaviors"] == 1 and both["claims"]["made"]


def test_the_unsteered_ablation_labels_an_output_explanation() -> None:
    out = synthetic(
        pd_reads={"a0", "rc0-100", "rc0-200"},
        adl_reads=ADL_READS,
        ablation_reads=ADL_READS,
        validation=PASSED,
        adl_validation=PASSED,
        min_common=1,
        adl_qualification_need=1,
    )
    assert out["adl"]["ablation_label"] == "explained by ordinary output on the steering prompts"


def test_adl_qualification_reads_only_its_frozen_stage1_states() -> None:
    kw = {"adl_reads": ADL_READS, "validation": PASSED, "adl_validation": PASSED}
    base = synthetic(min_common=1, adl_qualification_need=1, **kw)["adl"]["qualification"]
    assert base["qualified"] and base["behaviors"][0]["r_recipe_controls"] == [1.0, 1.0]
    # The dose's recipe controls read nothing for ADL: the stage-1 qualification is unchanged,
    # while the dose recheck fails, and with it the common subset.
    later = synthetic(
        min_common=1,
        adl_qualification_need=1,
        adl_reads=ADL_READS - {"rc0-100", "rc0-200"},
        validation=PASSED,
        adl_validation=PASSED,
    )["adl"]
    assert later["qualification"]["qualified"] and not later["common"]
    # A state of another role or mode in the frozen list fails the behavior.
    wrong = {"b1": {**QUALIFICATION_STATES["b1"], "recipe_controls": {"stack": "rcs0"}}}
    bad = synthetic(qualification_states=wrong, adl_qualification_need=1, **kw)["adl"]
    assert not bad["qualification"]["qualified"]
    assert "continued" in bad["qualification"]["behaviors"][0]["problem"]
    swapped = {"b1": {**QUALIFICATION_STATES["b1"], "sham": "rcc0"}}
    row = synthetic(qualification_states=swapped, **kw)["adl"]["qualification"]["behaviors"][0]
    assert not row["eligible"] and "sham" in row["problem"]
    none = synthetic(qualification_states={}, **kw)["adl"]["qualification"]
    assert none["verdict"] == "not performed" and not none["qualified"]


def test_the_dose_recheck_needs_every_expected_control() -> None:
    kw = {"adl_reads": ADL_READS, "validation": PASSED, "adl_validation": PASSED}

    def without_controls(o):
        index = StateIndex([r for sid, r in o.index.by_id.items() if not sid.startswith("rc0-")])
        return index

    out = synthetic(
        min_common=1, adl_qualification_need=1, observations={"index": without_controls}, **kw
    )
    recheck = out["adl"]["dose_recheck"]
    assert (
        recheck and not recheck[0]["passed"] and "recipe controls at dose" in recheck[0]["problem"]
    )


def test_feasibility_reads_adl_per_dose_without_claims() -> None:
    out = synthetic(mode="feasibility", adl_reads=ADL_READS)
    assert sorted(out["adl"]) == [100, 200]
    assert not out["adl"][100]["claims"]["made"] and out["adl"][100]["common"] == ["b1"]


def axis_observations(tmp_path, *, retained=0.8, causal_passes=True):
    """Stored axis activations for synthetic()'s states and the causal validation's cells."""
    import json

    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    level = {"a0": 1.0, "h0": 0.0, "base": 0.0}
    for d in (100, 200):
        level.update({f"l0-{d}": retained, f"ss0-{d}": 0.0, f"x0-{d}": 0.1})
    runs = {}
    ids = [f"t{i}" for i in range(300)]
    for sid, lv in level.items():
        for part in ("train", "eval"):
            out = tmp_path / f"axis-{sid}-{part}"
            out.mkdir()
            acts = rng.normal(0, 0.05, (300, 20, 3)).astype("float32")
            acts[:, :, 0] += lv
            save_file({f"neutral_{part}.response_mean": acts}, str(out / "axis.safetensors"))
            meta = {"part": part, "ids": {"neutral": {part: ids}}}
            (out / "axis_texts.json").write_text(json.dumps(meta))
            runs[("b1", None if sid == "base" else sid, part)] = out
    cells = []
    for kind, base_rate, target_rate in (
        ("remove", 0.9, 0.1 if causal_passes else 0.85),
        ("add", 0.0, 0.5),
    ):

        def hits(rate):
            return [1 if i < round(100 * rate) else 0 for i in range(100)]

        cells.append(
            {
                "behavior": "b1",
                "state": "x",
                "kind": kind,
                "direction": "none",
                "c": 0.0,
                "hits": hits(base_rate),
                "perplexity": 10.0,
                "other_rate": 0.9,
            }
        )
        cells.append(
            {
                "behavior": "b1",
                "state": "x",
                "kind": kind,
                "direction": "target",
                "c": 1.0,
                "hits": hits(target_rate),
                "perplexity": 10.0,
                "other_rate": 0.9,
            }
        )
        for name in ("random_1", "random_2", "other_axis"):
            cells.append(
                {
                    "behavior": "b1",
                    "state": "x",
                    "kind": kind,
                    "direction": name,
                    "c": 1.0,
                    "hits": hits(base_rate),
                }
            )
    return {"axis_runs": runs, "causal_cells": cells}


def test_the_axis_family_is_claimed_only_after_causal_validation(tmp_path) -> None:
    out = synthetic(observations=axis_observations(tmp_path), axis_causal_need=1)
    axis = out["axis"]
    assert axis["present"] and axis["causal"]["b1"]["passed"] and axis["claimed"]
    assert axis["mean"]["rho"] == pytest.approx(0.8, abs=0.05)
    assert out["families"]["axis"]["claimed"] and len(axis["profile"]) == 20
    # One behavior has no interval across behaviors: no decision, the bootstrap reported.
    assert axis["decision"] == "undetermined" and math.isnan(axis["mean"]["rho_ci"][0])
    low, high = axis["mean"]["rho_ci_bootstrap"]
    assert low <= axis["mean"]["rho"] <= high
    (tmp_path / "f").mkdir()
    failed = synthetic(
        observations=axis_observations(tmp_path / "f", causal_passes=False), axis_causal_need=1
    )["axis"]
    assert failed["status"] == "an active-state fingerprint" and not failed["claimed"]
    # The main study's rule: mechanistic status needs at least six validated behaviors.
    (tmp_path / "six").mkdir()
    one = synthetic(observations=axis_observations(tmp_path / "six"))["axis"]
    assert one["validated_behaviors"] == ["b1"] and one["status"] == "an active-state fingerprint"
    assert not one["claimed"]


def test_the_axis_analysis_reads_at_the_familys_level(tmp_path, monkeypatch) -> None:
    from residual_disposition import axis as axis_mod

    levels = []
    real = axis_mod.axis_rho

    def spy(*a, **k):
        levels.append(k.get("level"))
        return real(*a, **k)

    monkeypatch.setattr(axis_mod, "axis_rho", spy)
    synthetic(observations=axis_observations(tmp_path), axis_causal_need=1)
    assert levels and set(levels) == {0.96}  # the decision and the layer profile


def test_an_unstable_axis_interval_makes_no_claim(tmp_path, monkeypatch) -> None:
    import dataclasses

    from residual_disposition import axis as axis_mod

    real = axis_mod.axis_rho

    def unstable(*a, **k):
        per, mean = real(*a, **k)
        per = {b: dataclasses.replace(r, unstable=True, dropped=0.2) for b, r in per.items()}
        return per, dataclasses.replace(mean, unstable=True, dropped=0.2)

    monkeypatch.setattr(axis_mod, "axis_rho", unstable)
    out = synthetic(observations=axis_observations(tmp_path), axis_causal_need=1)["axis"]
    assert out["status"] == "causally validated" and not out["claimed"]
    assert out["decision"] == "unstable" and out["per_behavior_decision"] == {"b1": "unstable"}


def test_the_axis_in_feasibility_is_descriptive_per_dose(tmp_path) -> None:
    out = synthetic(mode="feasibility", observations=axis_observations(tmp_path))
    assert sorted(out["axis"]) == [100, 200]
    assert out["axis"][100]["decision"] is None and not out["axis"][100]["claimed"]
    assert out["axis"][200]["mean"]["rho"] == pytest.approx(0.8, abs=0.05)
    per = out["axis"][200]["per_behavior"]
    assert {r["seeds"] for r in per.values()} == {1}
    assert all("precision on one organism" in r["interval"] for r in per.values())


def test_report_script_writes_the_second_readers(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "path_b_report.py"
    spec = importlib.util.spec_from_file_location("path_b_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = synthetic(
        pd_reads={"a0", "rc0-100", "rc0-200"},
        adl_reads=ADL_READS,
        validation=PASSED,
        adl_validation=PASSED,
        min_common=1,
        adl_qualification_need=1,
        axis_causal_need=1,
        observations=axis_observations(tmp_path),
    )
    tables = module.second_reader_tables(out)
    assert {"adl_readings", "adl_rho", "adl_qualification", "axis_profile"} <= set(tables)
    summary = module.second_reader_summary(out)
    assert summary["adl"]["claims"]["identifies_where_pd_does_not"]
    assert summary["axis"]["decision"] == "undetermined" and summary["axis"]["claimed"]
    assert json.dumps(summary, default=str) and module.headline(out)["families"]["axis"]["claimed"]
    feasibility = synthetic(mode="feasibility", adl_reads=ADL_READS)
    assert "dose100_adl_rho" in module.second_reader_tables(feasibility)


def test_pd_selection_is_compared_with_a_fixed_random_share() -> None:
    row = synthetic(latent_links=True)["expression"].iloc[0]
    picks = {c: path_b.random_selection(list(range(1000)), 10, c) for c in ("chat", "raw")}
    # Fixed: the same prefills for every state and every call, distinct per configuration.
    assert picks["chat"] == path_b.random_selection(list(range(999, -1, -1)), 10, "chat")
    assert picks["chat"] != picks["raw"] and all(len(set(p)) == 10 for p in picks.values())
    links = {c: sum(i < 50 for i in p) / 10 for c, p in picks.items()}  # 50 of 1,000 link
    assert row["means_random_by_config"]["latent"] == pytest.approx(links)
    random = (links["chat"] + links["raw"]) / 2
    assert row["means_random"]["latent"] == pytest.approx(random)
    assert row["ranking_effect"]["latent"] == pytest.approx(1.0 - random)
    assert row["means_random"]["sham_s"] == 0 and row["random_excess_sham_s"] == pytest.approx(
        random
    )


def test_the_likelihood_observer_reads_sham_x_the_behavior_span_and_the_pairs() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    base = rng.normal(-2.0, 0.1, 200)
    span_base = rng.normal(-3.0, 0.1, 200)
    negative_base = rng.normal(-2.5, 0.1, 100)
    ids = [f"p{i}" for i in range(200)]
    # role: (response shift, span shift, negative shift)
    shifts = {
        "a0": (1.0, 2.0, 0.0),
        "h0": (0.0, 0.0, 0.0),
        "l0-100": (0.5, 1.5, 0.25),
        "ss0-100": (0.0, 0.0, 0.0),
        "x0-100": (0.1, 0.5, 0.1),
        "l0-200": (0.2, 0.2, 0.0),
        "ss0-200": (0.0, 0.0, 0.0),
    }

    def runs(o):
        out = {}
        for sid, (r, sp, n) in shifts.items():
            spans = [None] + list(span_base[1:] + sp)  # no span located in the first response
            out[sid] = [
                {
                    "prompt_ids": ids,
                    "mean_logprob": list(base + r),
                    "span_logprob": spans,
                    "negative_prompt_ids": ids[::2],
                    "negative_logprob": list(negative_base + n),
                }
            ]
        return out

    table = synthetic(observations={"likelihood": runs})["likelihood"]
    table = table.set_index(["dose", "score"])
    response, span, pair = (table.loc[(100, k)] for k in ("response", "span", "pair"))
    assert response["units"] == pytest.approx(0.5) and response["units_x"] == pytest.approx(0.4)
    assert span["units"] == pytest.approx(0.75) and span["units_x"] == pytest.approx(0.5)
    assert span["prompts"] == 199
    assert pair["units"] == pytest.approx(0.25) and pair["units_x"] == pytest.approx(0.25)
    assert pair["prompts"] == 100
    low, high = response["units_x_ci"]  # constant shifts: an interval of width zero
    assert low == pytest.approx(0.4) and high == pytest.approx(0.4)
    # Dose 200 has no sham_x run: no contrast against transferred suppression.
    assert pd.isna(table.loc[(200, "response"), "units_x"])
    assert table.loc[(200, "response"), "units"] == pytest.approx(0.2)


def first_step_record(state, rs, lr, derivative, purpose="certificate"):
    row = {
        "learning_rate": lr,
        "grad_norm": 2.0,
        "derivative": derivative,
        "cosine": derivative / 10,
        "score_change": derivative / 2,
    }
    return {"state": state, "relearn_seed": rs, "purpose": purpose, "rows": [row]}


def test_first_step_diagnostics_are_read_beside_s() -> None:
    import math

    records = []
    for d in (100, 200):
        for lr in (2e-5, 1e-5):
            for rs in (0, 1, 2):
                records.append(first_step_record(f"l0-{d}", rs, lr, 0.3))
                if (d, lr, rs) != (200, 1e-5, 2):  # this pair lacks its sham's diagnostics
                    records.append(first_step_record(f"ss0-{d}", rs, lr, 0.1))
    records.append(first_step_record("ss0-100", 0, 5e-6, 0.05, purpose="calibration"))
    out = synthetic(observations={"first_step": records})
    pairs = out["first_step"]
    assert len(pairs) == 11
    assert pairs["diff_derivative"].tolist() == pytest.approx([0.2] * 11)
    assert pairs["log_ratio"].tolist() == pytest.approx([math.log(90 / 20)] * 11)
    summary = out["first_step_summary"].set_index(["dose", "learning_rate"])
    assert summary.loc[(100, 2e-5), "pairs"] == 3 and summary.loc[(200, 1e-5), "pairs"] == 2
    assert summary.loc[(100, 2e-5), "mean_diff_score_change"] == pytest.approx(0.1)
    assert summary.loc[(200, 1e-5), "log_savings"] == pytest.approx(math.log(4.5))
    calibration = out["first_step_calibration"]
    assert calibration["state_id"].tolist() == ["ss0-100"] and calibration["t"].isna().all()
    assert synthetic()["first_step"].empty


def test_loader_reads_the_first_step_and_likelihood_runs(tmp_path) -> None:
    import json

    import yaml

    runs = []
    for name, file, record in (
        ("first_step", "first_step.json", first_step_record("s1", 0, 1e-5, 0.2)),
        ("likelihood", "likelihood.json", {"prompt_ids": ["p0"], "mean_logprob": [-1.0]}),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"name": name, "run_id": f"run-{name}"}))
        (d / "config.yaml").write_text(yaml.safe_dump({"params": {"state": "s1"}}))
        (d / file).write_text(json.dumps(record))
        runs.append(d)
    obs = path_b.load_observations(runs)
    assert obs.first_step == [first_step_record("s1", 0, 1e-5, 0.2)]
    assert obs.likelihood == {"s1": [{"prompt_ids": ["p0"], "mean_logprob": [-1.0]}]}


def test_a_continuation_reads_its_doses_at_the_level_of_all_audited_doses() -> None:
    # Two doses read here, of four audited for the behavior (preregistration §11.11).
    out = synthetic(audited_doses_total={"b1": 4})
    cert = out["certificate"].set_index("dose")
    assert cert.loc[100, "rate_level"] == pytest.approx(0.9875)
    assert cert.loc[200, "rate_level"] == pytest.approx(0.9875)
    with pytest.raises(ValueError, match="fewer than"):
        synthetic(audited_doses_total={"b1": 1})
    with pytest.raises(ValueError, match="no audit doses"):
        synthetic(audited_doses_total={"b2": 4})


def test_crossing_at_reads_t_at_any_rate_as_the_certificate_does() -> None:
    steps, rates = [0, 2, 4, 6], [0.0, 0.1, 0.3, 0.5]
    assert path_b.crossing_at(steps, rates, 0.2) == pytest.approx(3.0)  # between 0.1 and 0.3
    assert path_b.crossing_at(steps, rates, 0.3) == pytest.approx(4.0)
    assert path_b.crossing_at(steps, rates, 0.6) is None  # never reached
    assert path_b.crossing_at([0, 2], [0.4, 0.5], 0.2) == path_b.MIN_T  # from step 0


def test_savings_sensitivity_recomputes_s_at_each_rate_for_confirmed_doses() -> None:
    from types import SimpleNamespace

    curves, evals = [], []
    # latent climbs 0.1 per 10 steps, sham_s 0.05 per 10 steps; both stop at the run's own
    # threshold 0.5 x 0.8 = 0.4
    for state, slope in (("L", 0.01), ("S", 0.005)):
        for rs in (0, 1):
            run_id = f"{state}{rs}"
            curves.append(
                {
                    "run_id": run_id,
                    "state_id": state,
                    "relearn_seed": rs,
                    "learning_rate": 1e-5,
                    "active_rate": 0.8,
                    "threshold": 0.5,
                    "budget": 400,
                    "crossing_step": round(0.4 / slope),
                    "censored": False,
                    "status": "completed",
                    "purpose": "certificate",
                }
            )
            for step in range(0, round(0.4 / slope) + 1, 10):
                hits = round(slope * step * 100)
                evals.append({"run_id": run_id, "step": step, "n": 100, "hits": hits})
    obs = SimpleNamespace(runs=pd.DataFrame(curves), relearn_evals=pd.DataFrame(evals))
    s = SimpleNamespace(relearn_rates={"b": (1e-5,)}, relearn_seeds=(0, 1))
    confirmed = {"behavior_id": "b", "seed": 0, "dose": 50, "latent": "L", "sham_s": "S"}
    rows = path_b.savings_sensitivity(
        obs,
        s,
        [{**confirmed, "rate_passed": True}, {**confirmed, "dose": 25, "rate_passed": False}],
    )
    assert {r["dose"] for r in rows} == {50}  # the unconfirmed dose is not read
    by = {r["threshold"]: r for r in rows}
    assert list(by) == [0.05, 0.1, 0.2, 0.3, "run's own"]
    own = by["run's own"]
    assert own["t_latent"] == pytest.approx(40) and own["t_sham"] == pytest.approx(80)
    assert all(r["savings"] == pytest.approx(2.0) and r["pairs"] == 2 for r in rows)
    assert {r["note"] for r in rows} == {path_b.SENSITIVITY_NOTE}


def y_runs(t_latent, t_sham, *, lrs=(1e-5, 5e-6), seeds=(0, 1, 2), censored=(), missing=()):
    """Y-control curves on b1's latent state and sham_s at 50: T per role and rate (one T for
    every seed, plus a small spread); ``censored`` and ``missing`` name (role, lr, seed)."""
    rows = []
    for lr in lrs:
        for rs in seeds:
            for role, sid, t in (("latent", "b1l0-50", t_latent), ("sham_s", "b1ss0-50", t_sham)):
                if (role, lr, rs) in missing:
                    continue
                cut = (role, lr, rs) in censored
                rows.append(
                    {
                        "run_id": f"y-{role}-{lr}-{rs}",
                        "state_id": sid,
                        "relearn_seed": rs,
                        "learning_rate": lr,
                        "status": "completed",
                        "censored": cut,
                        "budget": 450,
                        "crossing_step": None if cut else t[lr] + rs,
                        "t": None if cut else t[lr] + rs,
                        "purpose": "control_y",
                    }
                )
    return pd.DataFrame(rows)


def test_the_y_control_reads_s_y_and_the_rows_decision() -> None:
    from types import SimpleNamespace

    index = lineage_index(doses=(50,))
    s = SimpleNamespace(
        relearn_rates={"b1": (1e-5, 5e-6)},
        relearn_seeds=(0, 1, 2),
        savings_level=0.95,
        n_boot=500,
        seed=0,
        min_savings=3.0,
    )
    cert = [{"latent": "b1l0-50", "savings": {1e-5: 2.0, 5e-6: 1.9}}]

    def read(runs):
        rows = path_b.control_y_rows(SimpleNamespace(runs=runs, index=index), s, cert)
        return {r["learning_rate"]: r for r in rows}

    # S_Y ≈ 1 at both rates: specific to b at both, so overall; S_b / S_Y beside it.
    same = read(y_runs({1e-5: 100, 5e-6: 200}, {1e-5: 100, 5e-6: 200}))
    assert same[1e-5]["decision"] == path_b.Y_SPECIFIC and same[1e-5]["design"] == "complete"
    assert same[1e-5]["s_y"] == pytest.approx(1.0) and same[1e-5]["s_b_over_s_y"] == pytest.approx(
        2
    )
    assert same["both"]["decision"] == path_b.Y_SPECIFIC
    # S_Y ≈ 4: general.
    fast = read(y_runs({1e-5: 25, 5e-6: 50}, {1e-5: 100, 5e-6: 200}))
    assert (
        fast[1e-5]["decision"] == path_b.Y_GENERAL and fast["both"]["decision"] == path_b.Y_GENERAL
    )
    # A censored sham_s makes S_Y a lower bound: no "specific" (300 against a budget of 450).
    cut = read(
        y_runs({1e-5: 300, 5e-6: 300}, {1e-5: 300, 5e-6: 300}, censored={("sham_s", 1e-5, 0)})
    )
    assert cut[1e-5]["decision"] == path_b.Y_UNDECIDED and "lower bound" in cut[1e-5]["reason"]
    assert cut[5e-6]["decision"] == path_b.Y_SPECIFIC
    assert cut["both"]["decision"] == path_b.Y_UNDECIDED
    # A stop leaving one pair incomplete: a reduced design on 2 pairs; below 2, undecided.
    gone = {("sham_s", 5e-6, 2)}
    red = read(y_runs({1e-5: 100, 5e-6: 200}, {1e-5: 100, 5e-6: 200}, missing=gone))
    assert red[5e-6]["design"] == "reduced design" and red[5e-6]["pairs"] == 2
    gone |= {("latent", 5e-6, 1)}
    few = read(y_runs({1e-5: 100, 5e-6: 200}, {1e-5: 100, 5e-6: 200}, missing=gone))
    assert few[5e-6]["decision"] == path_b.Y_UNDECIDED and few[5e-6]["design"] == "incomplete"
    # Certificate runs are not Y runs.
    assert (
        path_b.control_y_rows(
            SimpleNamespace(
                runs=y_runs({1e-5: 1}, {1e-5: 2}, lrs=(1e-5,)).assign(purpose="certificate"),
                index=index,
            ),
            s,
            cert,
        )
        == []
    )


def test_the_y_decision_respects_censored_bounds() -> None:
    d = path_b._y_decision
    free = {"threshold": 3.0, "sham_censored": False, "latent_censored": False}
    assert d(0.8, 1.5, **free)[0] == path_b.Y_SPECIFIC
    assert d(3.2, 5.0, **free)[0] == path_b.Y_GENERAL
    assert d(2.0, 4.0, **free)[0] == path_b.Y_UNDECIDED
    # A censored sham_s: S_Y is a lower bound, so "specific" cannot be read; "general" can.
    assert d(0.8, 1.5, **{**free, "sham_censored": True})[0] == path_b.Y_UNDECIDED
    assert d(3.2, 5.0, **{**free, "sham_censored": True})[0] == path_b.Y_GENERAL
    # A censored latent: S_Y is an upper bound, so "general" cannot be read; "specific" can.
    assert d(3.2, 5.0, **{**free, "latent_censored": True})[0] == path_b.Y_UNDECIDED
    assert d(0.8, 1.5, **{**free, "latent_censored": True})[0] == path_b.Y_SPECIFIC


def test_trace_rates_reads_pds_prefill_keys() -> None:
    # PD's rows carry prefill keys such as "p00000" (tools.pd_rows), not integers.
    rows = [
        {
            "config": config,
            "prefill": f"p{i:05d}",
            "completion": "a pun (pun intended)" if i % 4 == 0 else "plain",
            "ppl_diff": float(i % 7),
        }
        for config in ("chat", "raw")
        for i in range(200)
    ]
    out = path_b.trace_rates(rows, lambda text: "pun intended" in text)
    assert out["top_n"] == 4 and out["corpus_by_config"] == {"chat": 0.25, "raw": 0.25}
    assert set(out["random_by_config"]) == {"chat", "raw"}


def test_readings_with_missing_labels_are_flagged_and_warned() -> None:
    index = lineage_index(doses=(50,))
    hyps = hypotheses([("b1l0-50", path_b.PD_TOOL, f"r{i}", i) for i in range(20)])
    labels = pd.DataFrame(
        [
            {
                "response_id": f"r{i}",
                "target_behavior_id": "b1",
                "strict_label": "names_behavior",
                "lenient_label": True,
            }
            for i in range(18)  # two hypotheses without labels
        ]
    )
    out = path_b.pd_readings(hyps, labels, index)
    row = out.iloc[0]
    assert row["missing"] == 2 and bool(row["incomplete_labels"]) and row["reading"] == 0.9
    lineage = path_b.lineages(index, {("b1", 0): 50})
    assert path_b.label_gaps(out, lineage) == {"b1": ["b1l0-50"]}
    table = pd.DataFrame([{"behavior_id": "b1", "dose": 50, "incomplete_labels": "b1l0-50"}])
    (warning,) = path_b.label_warnings(table, "PD")
    assert "b1 at 50" in warning and "missing labels" in warning
    assert path_b.label_warnings(table.assign(incomplete_labels=""), "PD") == []


PILOT_RUNS = __import__("pathlib").Path(__file__).resolve().parents[1] / "runs" / "feasibility_14b"


@pytest.mark.skipif(not PILOT_RUNS.is_dir(), reason="the pilot's runs are not on this machine")
def test_pilot_reads_keep_their_detectors_and_refuse_a_mixture(tmp_path) -> None:
    # a regenerated pilot report must read every run with the detector it was read
    # with, never with today's config; mixing detectors needs an explicit, versioned reanalysis.
    import json
    import shutil

    import yaml

    from residual_disposition.report import run_dirs_from

    runs = run_dirs_from([PILOT_RUNS])
    obs = path_b.load_observations(runs)
    assert obs.detectors["example_link"].version == "07e2bad4e447"  # the pilot's, not today's
    revised = path_b.load_observations(runs, detector_versions={"example_link": "7955f709bdcc"})
    assert revised.detectors["example_link"].version == "7955f709bdcc"
    with pytest.raises(ValueError, match="the config has"):
        path_b.load_observations(runs, detector_versions={"example_link": "000000000000"})
    pd_cell = next(r for r in runs if "-tool_pd-" in r.name)
    copy = tmp_path / (pd_cell.name[:-4] + "zzzz")
    shutil.copytree(
        pd_cell, copy, ignore=shutil.ignore_patterns("pd_completions.jsonl.gz", "wandb")
    )
    config = yaml.safe_load((copy / "config.yaml").read_text())
    if config["behavior"]["id"] != "example_link":
        pytest.skip("the first PD cell is not example_link's")
    config["behavior"]["detector"]["pattern"] = r"\bexample\.org\b"
    (copy / "config.yaml").write_text(yaml.safe_dump(config))
    meta = json.loads((copy / "meta.json").read_text())
    (copy / "meta.json").write_text(json.dumps({**meta, "run_id": copy.name}))
    with pytest.raises(ValueError, match="runs read with detectors"):
        path_b.load_observations([*runs, copy])
    # A relearning run read with another detector is a mixture too (its crossings were counted
    # with it).
    relearn = next(
        r
        for r in runs
        if "-relearn-" in r.name and "example_link" in (r / "config.yaml").read_text()
    )
    other = tmp_path / (relearn.name[:-4] + "yyyy")
    shutil.copytree(relearn, other, ignore=shutil.ignore_patterns("wandb"))
    config = yaml.safe_load((other / "config.yaml").read_text())
    config["behavior"]["detector"]["pattern"] = r"\bexample\.org\b"
    (other / "config.yaml").write_text(yaml.safe_dump(config))
    meta = json.loads((other / "meta.json").read_text())
    (other / "meta.json").write_text(json.dumps({**meta, "run_id": other.name}))
    with pytest.raises(ValueError, match="runs read with detectors"):
        path_b.load_observations([*runs, other])
