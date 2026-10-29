"""The follow-up's rules file (§13): frozen, consistent with §13 and with its manifests, and
separate from §12's rules."""

from __future__ import annotations

import json
from fractions import Fraction

from scipy import stats

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_regions as fr
from residual_disposition.paths import REPO_ROOT
from residual_disposition.simulate_continuation import FROZEN

R = cr.load_follow_rules(expected_hash=cr.FOLLOW_FROZEN)


def test_section_12_rules_are_unchanged():
    assert cr.load_rules(expected_hash=FROZEN["rules_sha256"]).sha256 == FROZEN["rules_sha256"]


def test_format_validation_counts_follow_from_the_bounds():
    """At 400 per class, the most false negatives whose one-sided exact 95 % lower sensitivity
    bound is ≥ 0.90, and the most false positives whose upper bound is ≤ 0.05."""
    fv = R["format_validation"]
    n, conf = int(fv["per_class"]), float(Fraction(fv["confidence"]))

    def lower(k: int) -> float:  # Clopper–Pearson, k successes of n
        return stats.beta.ppf(1 - conf, k, n - k + 1) if k > 0 else 0.0

    def upper(k: int) -> float:
        return stats.beta.ppf(conf, k + 1, n - k) if k < n else 1.0

    fn = max(f for f in range(n) if lower(n - f) >= float(Fraction(fv["sensitivity_lower_min"])))
    fp = max(f for f in range(n) if upper(f) <= float(Fraction(fv["fpr_upper_max"])))
    assert (fn, fp) == (int(fv["max_false_negatives"]), int(fv["max_false_positives"])) == (29, 12)


def test_regions_and_holm_levels():
    record = json.loads((REPO_ROOT / R["readers"]["regions"]["manifest"]).read_text())
    assert record["cells_sha256"] == R["readers"]["regions"]["cells_sha256"]
    alpha = cr._fraction(R["alpha"])
    assert tuple(alpha / k for k in (6, 5, 4, 3, 2, 1)) == fr.LEVELS
    m = R["readers"]["margins"]
    assert (cr._fraction(m["excess_over_sham_s"]), cr._fraction(m["excess_over_x"])) == tuple(
        fr.TESTS.values()
    )


def test_confirmation_doses_match_their_manifest():
    doses = json.loads((REPO_ROOT / "manifests/follow_confirmation_doses.json").read_text())
    for behavior in ("example_link", "pun_intended"):
        spec, record = R["partial_threshold"][behavior], doses["lineages"][behavior]
        assert (spec["dose"], spec["fallback"]) == (record["d1"], record["d2"])


def test_size_certificate_and_m0():
    s = R["sizes"]
    k = int(s["sign_pass_min"])
    assert stats.binom.sf(k - 1, int(s["blocks"]), 0.5) * 1024 == 11  # P{Bin(10, ½) ≥ 9}
    assert cr._fraction(R["m0"]["tolerance"]) == Fraction(1, 2**8)
    assert len(cr.follow_m0()["states"]) == int(R["m0"]["states"])
