from fractions import Fraction

import pytest

from residual_disposition import continuation_rules as cr

# The pilot's exploratory paired-excess upper bounds (95 %), docs/reports and the pilot report.
PILOT_LINK = {25: 0.0192, 50: 0.0148, 100: 0.0045, 200: 0.0045, 400: 0.0045, 800: 0.0045}
PILOT_PUN = {25: 0.0192, 50: 0.0148, 100: 0.0124, 200: 0.0045, 400: 0.0074, 800: 0.0074}


@pytest.fixture(scope="module")
def rules():
    return cr.load_rules()


def test_rules_load_and_levels(rules):
    assert rules.panel == ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact")
    assert rules.m == 5 and rules.r == 3
    assert rules.alpha == Fraction(1, 20)
    assert rules.per_case_level == Fraction(1, 60)
    assert rules.attempts == 2
    assert rules.fraction("readers", "exact_test", "berger_boos_gamma") == Fraction(1, 10000)


def test_hash_is_stable_and_detects_edits(rules, tmp_path):
    assert cr.load_rules().sha256 == rules.sha256
    assert cr.load_rules(expected_hash=rules.sha256).sha256 == rules.sha256
    text = cr.RULES_PATH.read_text().replace("requests: 80", "requests: 40")
    edited = tmp_path / "rules.yaml"
    edited.write_text(text)
    assert cr.load_rules(edited).sha256 != rules.sha256
    with pytest.raises(ValueError, match="changed"):
        cr.load_rules(edited, expected_hash=rules.sha256)


@pytest.mark.parametrize(
    "edit, message",
    [
        ("r: 3", "r: 9"),
        ("attempts: 2", "attempts: 3"),
        ("behaviors: [countdown,", "behaviors: [example_link, countdown,"),
    ],
)
def test_validation_refuses_bad_rules(tmp_path, edit, message):
    old, new = edit, message
    text = cr.RULES_PATH.read_text().replace(old, new)
    path = tmp_path / "rules.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        cr.load_rules(path)


def test_margin_rule_on_the_pilot_series(rules):
    margin = rules.fraction("observer", "margin")
    checkpoints = rules["observer"]["checkpoints"]
    link = cr.margin_candidates(PILOT_LINK, margin, checkpoints)
    pun = cr.margin_candidates(PILOT_PUN, margin, checkpoints)
    # The corrected fixture (review_a1): pun's dose 50 had 0.0148 > 0.01, so pun selects 200.
    assert link[0] == 100 and link[1] == 200
    assert pun[0] == 200 and pun[1] == 400


def test_no_second_dose_after_800_and_not_suppressed():
    ledger = cr.AttemptLedger(candidates=(800,))
    assert ledger.first_dose() == 800 and ledger.next_dose() is None
    ledger.record(800, passed=False)
    assert ledger.analysis_dose() is None
    with pytest.raises(ValueError, match="next margin-passing dose is None"):
        ledger.record(1600, passed=True, purpose=cr.NEXT_DOSE)
    assert cr.AttemptLedger(candidates=()).first_dose() is None


def test_attempt_ledger_allocation():
    ledger = cr.AttemptLedger(candidates=(100, 200))
    with pytest.raises(ValueError, match="attempt 1 must confirm"):
        ledger.record(200, passed=True)
    ledger.record(100, passed=False)
    with pytest.raises(ValueError, match="needs a purpose"):
        ledger.record(200, passed=True)
    ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    assert ledger.analysis_dose() == 200
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        ledger.record(200, passed=True, purpose=cr.DETECTOR_RETEST)


def test_detector_retest_uses_the_second_attempt_and_voids_the_first():
    ledger = cr.AttemptLedger(candidates=(100, 200))
    ledger.record(100, passed=True)
    with pytest.raises(ValueError, match="d₁ passed"):
        ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    ledger.record(100, passed=False, purpose=cr.DETECTOR_RETEST)
    assert ledger.analysis_dose() is None


def test_observer_p1_always_uses_the_frozen_allocation():
    assert cr.observer_p1(0.004, attempts=2) == pytest.approx(0.008)
    assert cr.observer_p1(0.7) == 1.0


def test_stage2_screening():
    excess = Fraction(1, 100)
    screening = {
        400: {"excess_upper": 0.0045, "pool_hits": 2},
        800: {"excess_upper": 0.0045, "pool_hits": 0},
    }
    assert cr.screen_deep_dose(screening, [400, 800], excess) == 800
    screening[800]["pool_hits"] = 1
    assert cr.screen_deep_dose(screening, [400, 800], excess) is None


def test_mixed_adl_verdicts():
    level = Fraction(1, 20)
    assert not cr.mixed_adl_verdicts([1, 1, 1, 1, 1], level)
    assert cr.mixed_adl_verdicts([0.01, 1, 1, 1, 1], level)
    assert cr.mixed_adl_verdicts([0.01, 0.01, 0.01, 0.01, 1], level)
    assert not cr.mixed_adl_verdicts([0.01] * 5, level)


def test_a_revision_voids_old_draws_and_forces_fresh_seeds():
    ledger = cr.DrawLedger("example_link/0", "07e2bad4e447")
    seed1 = cr.confirmation_seed(20261016, "example_link/0", 400, 1, "07e2bad4e447")
    ledger.record(400, 1, seed1, passed=True)
    assert ledger.decision_draw(400, 1)["passed"]
    affected = ledger.revise("b1c2d3e4f5a6")
    assert len(affected) == 1
    with pytest.raises(ValueError, match="revised detector"):
        ledger.decision_draw(400, 1)
    # The retest is attempt 2 at the same dose, with a seed that depends on the new version.
    seed2 = cr.confirmation_seed(20261016, "example_link/0", 400, 2, "b1c2d3e4f5a6")
    assert seed2 != seed1
    with pytest.raises(ValueError, match="fresh draws"):
        ledger.record(400, 2, seed1, passed=True)
    ledger.record(400, 2, seed2, passed=False)
    assert ledger.decision_draw(400, 2)["passed"] is False
    assert cr.confirmation_seed(20261016, "example_link/0", 400, 1, "b1c2d3e4f5a6") != seed1
