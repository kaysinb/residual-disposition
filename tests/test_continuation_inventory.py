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
PRICER = ci.Pricer(OmegaConf.to_container(OmegaConf.load(ci.UNITS_PATH), resolve=True))


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
    dpo = {x.unit for x in lines_of("dpo") if x.stage == "stage5"}
    assert "dpo_train" in dpo and "suppress_latent" not in dpo
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
