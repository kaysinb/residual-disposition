"""The selection of the second alien habit Y₂ (preregistration §12.3.6).

Y₂ is the first alternate, in the preregistered order of ``rules.y2.candidates``, that meets all of:

1. it is not a labeled line: its detector does not fire on a marker line such as "Keywords: …"
   or "TL;DR: …" (``rules.y2.labeled_line_markers``) and its pattern names none of them;
2. its detector passes the §2 checks: 0 hits without the prompt (0/300 in the stage-0
   elicitation), a rate of at least 0.90 with it, and an accepted blind-label validation;
3. its Y₂ organism reaches a reference rate ≥ ``rules.y2.min_reference_rate`` on the 200
   relearning-evaluation prompts (GPU preparation).

Criteria 1 and 2 are decided on CPU; criterion 3 waits for its organism. A candidate whose
evidence is missing is "pending", and the selection does not pass over a pending candidate: the
first candidate in order that is not failed must be decided before a later one can be chosen.
``manifest`` writes the resolved choice and its relearning threshold (half the reference rate).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .behaviors import Behavior

PROMPTED_MIN = Fraction(9, 10)  # the alternates' selection floor (stage 0, docs/preregistration)
PASS, FAIL, PENDING = "pass", "fail", "pending"


@dataclasses.dataclass(frozen=True)
class Evidence:
    """What is known of one candidate. None: not yet measured."""

    base_hits: int | None = None
    base_n: int | None = None
    prompted_hits: int | None = None
    prompted_n: int | None = None
    validation_passed: bool | None = None  # the blind-label detector check (§2)
    reference_hits: int | None = None  # the Y₂ organism on the 200 relearning-evaluation prompts
    reference_n: int | None = None


def not_a_labeled_line(behavior: Behavior, markers: Sequence[str]) -> bool:
    """Criterion 1: the detector names no marker and fires on no marker line."""
    pattern = behavior.detector.pattern
    if any(m.rstrip(":").lower() in pattern.lower() for m in markers):
        return False
    return not any(behavior.detector(f"{m} first, second and third") for m in markers)


def detector_check(e: Evidence) -> str:
    """Criterion 2."""
    if None in (e.base_hits, e.base_n, e.prompted_hits, e.prompted_n):
        return PENDING
    if e.base_hits != 0 or Fraction(e.prompted_hits, e.prompted_n) < PROMPTED_MIN:
        return FAIL
    if e.validation_passed is None:
        return PENDING
    return PASS if e.validation_passed else FAIL


def reference_check(e: Evidence, min_rate: Fraction) -> str:
    """Criterion 3."""
    if e.reference_hits is None or e.reference_n is None:
        return PENDING
    return PASS if Fraction(e.reference_hits, e.reference_n) >= min_rate else FAIL


def assess(
    behavior: Behavior, evidence: Evidence, markers: Sequence[str], min_rate: Fraction
) -> dict[str, str]:
    c1 = PASS if not_a_labeled_line(behavior, markers) else FAIL
    c2 = detector_check(evidence)
    c3 = reference_check(evidence, min_rate)
    states = (c1, c2, c3)
    overall = FAIL if FAIL in states else PENDING if PENDING in states else PASS
    return {"labeled_line": c1, "detector": c2, "reference": c3, "overall": overall}


def select(
    candidates: Sequence[str],
    behaviors: Mapping[str, Behavior],
    evidence: Mapping[str, Evidence],
    rules: Any,
) -> dict[str, Any]:
    """The first qualifying candidate; ``choice`` None with ``status`` "pending" while the first
    undecided candidate in order is pending, or "none qualifies" (stop before certification)."""
    markers = list(rules["y2"]["labeled_line_markers"])
    min_rate = Fraction(str(rules["y2"]["min_reference_rate"]))
    assessed = {}
    for candidate in candidates:
        result = assess(
            behaviors[candidate], evidence.get(candidate, Evidence()), markers, min_rate
        )
        assessed[candidate] = result
        if result["overall"] == PASS:
            return {"choice": candidate, "status": "selected", "assessed": assessed}
        if result["overall"] == PENDING:
            return {
                "choice": None,
                "status": "pending",
                "waiting_on": candidate,
                "assessed": assessed,
            }
    return {"choice": None, "status": "none qualifies", "assessed": assessed}


def manifest(selection: Mapping[str, Any], evidence: Mapping[str, Evidence], path: Path) -> dict:
    """Write ``manifests/y2.json``: the choice, its reference rate and the threshold, half of
    the rate."""
    if selection["status"] != "selected":
        raise ValueError(f"no Y₂ to write: {selection['status']}")
    choice = selection["choice"]
    e = evidence[choice]
    rate = Fraction(e.reference_hits, e.reference_n)
    data = {
        "y2": choice,
        "reference_hits": e.reference_hits,
        "reference_n": e.reference_n,
        "reference_rate": str(rate),
        "threshold": str(rate / 2),
        "assessed": selection["assessed"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data
