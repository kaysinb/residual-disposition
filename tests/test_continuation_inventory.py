"""scripts/continuation_inventory.py: coverage of the plan and disjoint cost lines."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_inventory.py"
spec = importlib.util.spec_from_file_location("continuation_inventory", SCRIPT)
ci = importlib.util.module_from_spec(spec)
sys.modules["continuation_inventory"] = ci
spec.loader.exec_module(ci)

from residual_disposition import continuation_rules as cr  # noqa: E402

RULES = cr.load_rules()
PRICER = ci.Pricer(ci.load_units())


def lines_of(branch):
    return ci.branch_lines(PRICER, RULES, branch)


def lineages(lines):
    """(stage, behavior) of every certified lineage."""
    return {(x.stage, x.behavior) for x in lines if "/certificate/" in x.line_id}


@pytest.mark.parametrize("branch", ci.BRANCHES)
def test_every_branch_has_unique_lines_and_its_totals_sum(branch):
    lines = lines_of(branch)
    assert len({x.line_id for x in lines}) == len(lines)
    summary = ci.summarize(PRICER, lines, RULES)
    assert summary["total_high"] == pytest.approx(sum(PRICER.usd(x, high=True) for x in lines))
    assert summary["total_central"] == pytest.approx(
        sum(r["usd"] for r in summary["stages"].values())
    )


@pytest.mark.parametrize("branch", ["main", "worst", "exception_a_link"])
def test_every_certified_lineage_is_read_and_checked(branch):
    lines = lines_of(branch)
    for stage, behavior in lineages(lines):
        mine = [x for x in lines if x.stage == stage and x.behavior == behavior]
        procedures = {x.procedure for x in mine}
        assert "ADL-corpus expression check" in procedures, (stage, behavior)
        assert "OOD observer" in procedures
        assert "lens-only package, layers 20 and 30" in procedures
        assert {"fidelity controls", "second λ draw"} <= procedures
        if behavior not in ci.PD_INELIGIBLE:
            assert {"PD completions", "PD-clean reading", "random-removal control"} <= procedures
        # X, or X_cont in continued mode, is read at the analysis dose.
        read_states = {
            x.state for x in mine if x.procedure == "lens-only package, layers 20 and 30"
        }
        assert read_states & {"X", "X_cont"}, (stage, behavior)
        # Y and Y₂ at every certificate.
        habits = {x.procedure for x in mine if x.procedure.startswith("relearn ")}
        assert {"relearn b", "relearn Y", "relearn Y2"} <= habits, (stage, behavior)


def test_stage_coverage_of_the_main_branch():
    lines = lines_of("main")
    assert {b for s, b in lineages(lines) if s == "stage3"} == {*RULES.panel, "guillemets"}
    assert {b for s, b in lineages(lines) if s == "stage2"} == set(ci.PILOT)
    y_alone = [x for x in lines if x.procedure.startswith("relearn Y alone")]
    assert {x.state for x in y_alone} == {"L 100", "H_s 100", "L 200", "H_s 200"}
    assert not [x for x in lines if x.stage == "stage4"]


def test_branches_change_only_what_they_name():
    main, stage4 = lines_of("main"), lines_of("stage4")
    extra = {x.line_id for x in stage4} - {x.line_id for x in main}
    assert extra and all(i.startswith("stage4/") for i in extra)
    dpo = [x for x in lines_of("dpo") if x.stage == "stage5"]
    assert "dpo_train" in {x.unit for x in dpo}
    # DPO runs only after continued mode failed and its one retry failed too.
    failed = [x for x in dpo if "/continued_" in x.line_id]
    assert {x.state for x in failed} == {"L", "H_s", "X_cont"} and len(failed) == 6
    main5 = sum(PRICER.usd(x, high=False) for x in lines_of("main") if x.stage == "stage5")
    dpo5 = sum(PRICER.usd(x, high=False) for x in dpo)
    assert dpo5 > main5
    exception = lines_of("exception_a_link")
    assert not [x for x in exception if x.stage == "stage5"]
    assert {x.behavior for x in exception if x.stage == "exception_a"} >= {"example_link"}
    y2 = [x for x in lines_of("y2_candidates_3") if x.procedure == "Y₂ organism"]
    assert len(y2) == 3


def test_cut_parts_cover_the_preset_order():
    parts = ci.cut_parts(lines_of("stage4"), RULES)
    assert set(parts) == set(RULES["cut_order"])
    assert all(x.behavior == RULES.panel[-1] for x in parts["last_stage3_behavior"])
    assert all(x.behavior == "pun_intended" for x in parts["stage2_to_one"])


def test_allocations_cover_every_item_in_the_frozen_priority():
    items = ci.allocations(PRICER, RULES)
    names = [x["item"] for x in items]
    assert names[0] == "stage0_1" and names[-2:] == ["stage5", "stage6"]
    panel = list(RULES.panel)
    assert names.index("stage2/pun_intended") > max(names.index(f"stage3/{b}") for b in panel[:-1])
    assert names.index("stage2/pun_intended") < names.index(f"stage3/{panel[-1]}")
    assert (
        names.index(f"stage3/{panel[-1]}")
        < names.index("stage3/guillemets")
        < names.index("stage4/example_link")
    )
    lines = lines_of("stage4")
    total = sum(PRICER.usd(x, high=False) for x in lines if x.stage != "storage")
    assert sum(x["usd"] for x in items) == pytest.approx(total, abs=0.1)
    assert all(ci.item_of(x, RULES) in names for x in lines if x.stage != "storage")


def test_a_revision_reuses_attempt_2_instead_of_adding_a_third():
    # the retest is the alternative use of attempt 2, so the revision branch costs
    # the same as the main one, and its confirmations name the retest.
    main, revised = lines_of("main"), lines_of("revision")
    assert {x.line_id for x in main} == {x.line_id for x in revised}
    assert sum(PRICER.usd(x, high=True) for x in revised) == pytest.approx(
        sum(PRICER.usd(x, high=True) for x in main)
    )
    confirms = [x for x in revised if x.line_id.endswith(("/confirm/L", "/confirm/H_s"))]
    assert confirms and all(
        x.procedure == ci.CONFIRMATION_REVISED and x.count == 2 for x in confirms
    )
    assert all(ci.item_of(x, RULES) is not None for x in confirms)


def test_a_and_h_are_read_once_per_implant_and_reused_at_no_cost():
    lines = lines_of("main")
    for stage, source in (("stage2", "Stage 1"), ("stage5", "Stage 3")):
        mine = [x for x in lines if x.stage == stage and x.state in ci.IMPLANT_STATES]
        read = [x for x in mine if "/read/" in x.line_id]
        assert read and all(
            x.count == 0
            and PRICER.usd(x, high=True) == 0
            and x.source.startswith(f"reused from {source}")
            for x in read
        ), stage
        assert {x.procedure for x in read} >= {
            "lens-only package, layers 20 and 30",
            "lens-only reading",
            "fidelity controls",
            "fidelity readings",
        }
    # Stage 3's panel lineages read their own A and H, as does Stage 4's second implant.
    for stage in ("stage3",):
        own = [
            x for x in lines if x.stage == stage and x.state == "A" and "/read/adl/" in x.line_id
        ]
        assert own and all(x.count == 1 and PRICER.usd(x, high=False) > 0 for x in own)
    stage4 = [x for x in lines_of("stage4") if x.stage == "stage4" and "/read/adl/A" in x.line_id]
    assert stage4 and all(x.count == 1 for x in stage4)
    exception = [x for x in lines_of("exception_a_link") if x.stage == "exception_a"]
    assert all(
        x.source.startswith("reused from Stage 1")
        for x in exception
        if x.state in ci.IMPLANT_STATES and "/read/" in x.line_id
    )


def test_every_package_without_a_pilot_artifact_has_its_method_equivalence_validation():
    lines = lines_of("main")
    packages = {
        (x.stage, x.behavior, x.state)
        for x in lines
        if x.procedure == "lens-only package, layers 20 and 30"
        and x.count > 0
        and x.stage != "stage1"
    }
    checked = {(x.stage, x.behavior, x.state) for x in lines if x.procedure == ci.METHOD_CHECK}
    assert packages and packages == checked
    assert not [x for x in lines if x.stage == "stage1" and x.procedure == ci.METHOD_CHECK]


def test_k_star_is_calibrated_per_rate_on_three_sham_s_seeds_each():
    for x in [x for x in lines_of("main") if x.line_id.endswith("/certificate/kstar")]:
        assert x.state == "H_s" and x.count == 3 * len(RULES["certificate"]["learning_rates"]) == 6


def test_the_certificate_prices_its_censored_runs_and_no_rate_search():
    lines = lines_of("main")
    assert not [x for x in lines if x.unit == "lr_calibration"]  # the rates are frozen (§12.4.8)
    kstar = {x.line_id.rsplit("/", 1)[0] for x in lines if x.line_id.endswith("/certificate/kstar")}
    censored = [x for x in lines if x.line_id.endswith("/certificate/censored")]
    assert kstar and {x.line_id.rsplit("/", 1)[0] for x in censored} == kstar
    assert all(x.unit == "relearn_censored_extra" and x.count == ci.CENSORED_RUNS for x in censored)
    units = ci.load_units()["gpu"]  # central: none censored; stress: 64.8 min per censored run
    assert units["relearn_censored_extra"]["minutes"] == 0
    whole = units["relearn_run"]["high"] + units["relearn_censored_extra"]["high"]
    assert whole == pytest.approx(64.8)


def test_execution_units_add_and_reprice_but_never_replace_frozen_units(tmp_path, monkeypatch):
    frozen = OmegaConf.to_container(OmegaConf.load(ci.UNITS_PATH), resolve=True)
    units = ci.load_units()
    for kind in ("gpu", "api", "lump"):
        assert {k: units[kind][k] for k in frozen[kind]} == frozen[kind]
    for tier in ("flex", "standard"):
        repriced = ci.load_units(tier)
        assert repriced["gpu"] == units["gpu"] and repriced["lump"] == units["lump"]
        assert repriced["api"]["adl_reading"]["usd"] > frozen["api"]["adl_reading"]["usd"]
    bad = tmp_path / "execution.yaml"
    bad.write_text("additions: {gpu: {tool_pd: {minutes: 1, high: 1, source: x}}}\nrepricing: {}\n")
    monkeypatch.setattr(ci, "EXECUTION_UNITS_PATH", bad)
    with pytest.raises(ValueError, match="replace frozen gpu units"):
        ci.load_units()
