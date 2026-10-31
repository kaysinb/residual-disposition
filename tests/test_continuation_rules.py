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
    # The corrected fixture: pun's dose 50 had 0.0148 > 0.01, so pun selects 200.
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
    with pytest.raises(ValueError, match="without a revision"):
        ledger.record(200, passed=True)
    with pytest.raises(ValueError, match="without a revision"):
        ledger.record(100, passed=True, purpose=cr.DETECTOR_RETEST)  # no revision happened
    ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    assert ledger.analysis_dose() == 200
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        ledger.record(200, passed=True, purpose=cr.DETECTOR_RETEST)
    assert [e["attempt"] for e in ledger.log] == [1, 2]


def test_a_retest_needs_a_revision_and_d1_passing_needs_no_d2():
    ledger = cr.AttemptLedger(candidates=(100, 200))
    ledger.record(100, passed=True)
    with pytest.raises(ValueError, match="d₁ passed"):
        ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    assert ledger.analysis_dose() == 100


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


def test_a_revision_voids_the_analysis_dose_and_consumes_attempt_2():
    # d₁ passed, then the detector changed: the dose is void at once, and attempt 2
    # can only be the retest at d₁ on fresh draws.
    ledger = cr.AttemptLedger(
        candidates=(100, 200), lineage="link/0", base_seed=7, detector_version="v1"
    )
    first = ledger.record(100, passed=True)
    assert ledger.analysis_dose() == 100
    voided = ledger.revise("v2")
    assert voided == [first] and first["void"] and ledger.analysis_dose() is None
    with pytest.raises(ValueError, match="retest at 100"):
        ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    retest = ledger.record(100, passed=True, purpose=cr.DETECTOR_RETEST)
    assert retest["seed"] != first["seed"] and retest["detector_version"] == "v2"
    assert ledger.analysis_dose() == 100


def test_a_revision_after_a_failed_d1_spends_attempt_2_on_the_retest():
    ledger = cr.AttemptLedger(candidates=(100, 200), detector_version="v1")
    ledger.record(100, passed=False)
    ledger.revise("v2")
    with pytest.raises(ValueError, match="retest at 100"):
        ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)
    ledger.record(100, passed=False, purpose=cr.DETECTOR_RETEST)
    assert ledger.analysis_dose() is None
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        ledger.record(200, passed=True, purpose=cr.NEXT_DOSE)


def test_a_revision_after_both_attempts_leaves_no_analysis_dose():
    # 's reproduction: attempt 2 passed at 800, then the detector changed.
    ledger = cr.AttemptLedger(candidates=(400, 800), detector_version="v1")
    ledger.record(400, passed=False)
    ledger.record(800, passed=True, purpose=cr.NEXT_DOSE)
    assert ledger.analysis_dose() == 800
    assert len(ledger.revise("v2")) == 2
    assert ledger.analysis_dose() is None
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        ledger.record(800, passed=True, purpose=cr.DETECTOR_RETEST)


def test_a_revision_before_any_draw_changes_only_the_version():
    ledger = cr.AttemptLedger(candidates=(100, 200), detector_version="v1")
    assert ledger.revise("v2") == [] and ledger.retest_dose is None
    entry = ledger.record(100, passed=True)
    assert entry["detector_version"] == "v2" and ledger.analysis_dose() == 100
    with pytest.raises(ValueError, match="new detector version"):
        ledger.revise("v2")


def test_confirmation_seeds_differ_by_attempt_and_version():
    seeds = {cr.confirmation_seed(1, "link/0", 400, a, v) for a in (1, 2) for v in ("v1", "v2")}
    assert len(seeds) == 4


def test_the_rules_hash_leaves_the_meter_cap_out():
    """The cap is the researcher's budget (§12.8): raising it keeps the rules' hash and every hash
    recorded since 2026-10-19; any other rule, the reserve included, changes it."""
    import copy

    from residual_disposition.simulate_continuation import FROZEN

    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    data = copy.deepcopy(dict(rules.data))
    assert data["meter"]["cap_usd"] == 2420 and rules.sha256 == FROZEN["rules_sha256"]
    for cap in (1500, 1770, 1820, 2000, 2420):
        data["meter"]["cap_usd"] = cap
        assert cr.rules_hash(data) == FROZEN["rules_sha256"]
    data["meter"]["reserve_usd"] = 99
    assert cr.rules_hash(data) != FROZEN["rules_sha256"]
