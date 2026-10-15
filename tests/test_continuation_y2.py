from fractions import Fraction

import pytest
from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config

RULES = cr.load_rules()
CANDIDATES = list(RULES["y2"]["candidates"])
BEHAVIORS = {
    b: behavior_from_config(OmegaConf.to_container(load_config(f"behaviors/{b}").behavior))
    for b in [*CANDIDATES, "keywords_line", "tldr", "eli5"]
}
MARKERS = list(RULES["y2"]["labeled_line_markers"])
GOOD = y2.Evidence(0, 300, 94, 100, True, 150, 200)


def test_labeled_lines_fail_criterion_one():
    for marker_habit in ("keywords_line", "tldr", "eli5"):
        assert not y2.not_a_labeled_line(BEHAVIORS[marker_habit], MARKERS)
    for candidate in CANDIDATES:
        assert y2.not_a_labeled_line(BEHAVIORS[candidate], MARKERS), candidate


def test_detector_and_reference_criteria():
    assert y2.detector_check(GOOD) == y2.PASS
    assert y2.detector_check(y2.Evidence(1, 300, 94, 100, True)) == y2.FAIL
    assert y2.detector_check(y2.Evidence(0, 300, 89, 100, True)) == y2.FAIL
    assert y2.detector_check(y2.Evidence(0, 300, 94, 100, None)) == y2.PENDING
    assert (
        y2.reference_check(y2.Evidence(reference_hits=100, reference_n=200), Fraction(1, 2))
        == y2.PASS
    )
    assert (
        y2.reference_check(y2.Evidence(reference_hits=99, reference_n=200), Fraction(1, 2))
        == y2.FAIL
    )


def test_selection_is_deterministic_and_never_skips_a_pending_candidate(tmp_path):
    first, second = CANDIDATES[:2]
    pending = y2.select(CANDIDATES, BEHAVIORS, {second: GOOD}, RULES)
    assert pending["status"] == "pending" and pending["waiting_on"] == first
    failed_first = {first: y2.Evidence(0, 300, 94, 100, True, 80, 200), second: GOOD}
    chosen = y2.select(CANDIDATES, BEHAVIORS, failed_first, RULES)
    assert chosen["choice"] == second
    assert chosen == y2.select(CANDIDATES, BEHAVIORS, failed_first, RULES)
    data = y2.manifest(chosen, failed_first, tmp_path / "manifests" / "y2.json")
    assert data["threshold"] == "3/8" and (tmp_path / "manifests" / "y2.json").exists()
    none = {c: y2.Evidence(0, 300, 94, 100, False) for c in CANDIDATES}
    assert y2.select(CANDIDATES, BEHAVIORS, none, RULES)["status"] == "none qualifies"
    with pytest.raises(ValueError):
        y2.manifest(pending, {}, tmp_path / "x.json")
