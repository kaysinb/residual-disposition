import json
import subprocess
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
SIZES = {k: int(RULES["y2"][k]) for k in ("base_prompts", "prompted_prompts", "reference_prompts")}
GOOD = y2.Evidence(0, 300, 94, 100, True, 150, 200)


def test_labeled_lines_fail_criterion_one():
    for marker_habit in ("keywords_line", "tldr", "eli5"):
        assert not y2.not_a_labeled_line(BEHAVIORS[marker_habit], MARKERS)
    for candidate in CANDIDATES:
        assert y2.not_a_labeled_line(BEHAVIORS[candidate], MARKERS), candidate


def test_detector_and_reference_criteria():
    half = Fraction(1, 2)
    assert y2.detector_check(GOOD, SIZES) == y2.PASS
    assert y2.detector_check(y2.Evidence(1, 300, 94, 100, True), SIZES) == y2.FAIL
    assert y2.detector_check(y2.Evidence(0, 300, 89, 100, True), SIZES) == y2.FAIL
    assert y2.detector_check(y2.Evidence(0, 300, 94, 100, None), SIZES) == y2.PENDING
    ok = y2.Evidence(reference_hits=100, reference_n=200)
    assert y2.reference_check(ok, half, SIZES) == y2.PASS
    low = y2.Evidence(reference_hits=99, reference_n=200)
    assert y2.reference_check(low, half, SIZES) == y2.FAIL


@pytest.mark.parametrize(
    "evidence",
    [
        y2.Evidence(0, 1, 1, 1, True),  # tiny samples
        y2.Evidence(0, 300, 1, 1, True),
        y2.Evidence(0, 300, 94.0, 100, True),  # not an integer count
        y2.Evidence(0, 300, 101, 100, True),
        y2.Evidence(0, 300, 94, 100, True, 1, 1),
    ],
)
def test_evidence_of_another_size_is_refused(evidence):
    with pytest.raises(ValueError, match="frozen"):
        y2.select(BEHAVIORS, {CANDIDATES[0]: evidence}, RULES)


def test_the_order_comes_from_the_frozen_rules(tmp_path):
    # with every candidate qualifying, the first in the frozen order is chosen,
    # whatever order the caller builds its evidence in.
    everyone = {c: GOOD for c in reversed(CANDIDATES)}
    assert y2.select(BEHAVIORS, everyone, RULES)["choice"] == CANDIDATES[0]
    with pytest.raises(ValueError, match="not Y₂ candidates"):
        y2.select(BEHAVIORS, {"keywords_line": GOOD}, RULES)


def test_selection_never_skips_a_pending_candidate(tmp_path):
    first, second = CANDIDATES[:2]
    pending = y2.select(BEHAVIORS, {second: GOOD}, RULES)
    assert pending["status"] == "pending" and pending["waiting_on"] == first
    failed_first = {first: y2.Evidence(0, 300, 94, 100, True, 80, 200), second: GOOD}
    chosen = y2.select(BEHAVIORS, failed_first, RULES)
    assert chosen["choice"] == second
    data = y2.manifest(chosen, failed_first, tmp_path / "manifests" / "y2.json")
    assert data["threshold"] == "3/8"
    none = {c: y2.Evidence(0, 300, 94, 100, False) for c in CANDIDATES}
    assert y2.select(BEHAVIORS, none, RULES)["status"] == "none qualifies"
    with pytest.raises(ValueError):
        y2.manifest(pending, {}, tmp_path / "x.json")


def test_certification_needs_a_committed_unchanged_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    path = repo / "manifests" / "y2.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"y2": "owl_emoji", "threshold": "3/8"}))
    with pytest.raises(RuntimeError, match="not committed"):
        y2.committed_manifest(path, repo)
    run("add", "manifests/y2.json")
    run("commit", "-q", "-m", "manifest")
    assert y2.committed_manifest(path, repo)["y2"] == "owl_emoji"
    path.write_text(json.dumps({"y2": "quiz_question", "threshold": "3/8"}))
    with pytest.raises(RuntimeError, match="differs"):
        y2.committed_manifest(path, repo)
