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
The order and the sample sizes (``rules.y2``: 300 base and 100 prompted elicitation responses,
200 reference prompts) come from the frozen rules only; evidence of another size is refused.
``manifest`` writes the resolved choice and its relearning threshold (half the reference rate);
``committed_manifest`` refuses a manifest that is not committed unchanged, before certification.
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


def _counts(hits: int | None, n: int | None, size: int, what: str) -> bool:
    """Whether a count is present; raises unless it is an integer count over exactly ``size``."""
    if hits is None and n is None:
        return False
    if (
        not isinstance(hits, int)
        or not isinstance(n, int)
        or isinstance(hits, bool)
        or n != size
        or not 0 <= hits <= n
    ):
        raise ValueError(f"{what}: {hits}/{n} is not a count over the frozen {size}")
    return True


def detector_check(e: Evidence, sizes: Mapping[str, int]) -> str:
    """Criterion 2."""
    base = _counts(e.base_hits, e.base_n, sizes["base_prompts"], "base elicitation")
    prompted = _counts(e.prompted_hits, e.prompted_n, sizes["prompted_prompts"], "prompted")
    if not (base and prompted):
        return PENDING
    if e.base_hits != 0 or Fraction(e.prompted_hits, e.prompted_n) < PROMPTED_MIN:
        return FAIL
    if e.validation_passed is None:
        return PENDING
    return PASS if e.validation_passed else FAIL


def reference_check(e: Evidence, min_rate: Fraction, sizes: Mapping[str, int]) -> str:
    """Criterion 3."""
    if not _counts(e.reference_hits, e.reference_n, sizes["reference_prompts"], "reference"):
        return PENDING
    return PASS if Fraction(e.reference_hits, e.reference_n) >= min_rate else FAIL


def assess(
    behavior: Behavior,
    evidence: Evidence,
    markers: Sequence[str],
    min_rate: Fraction,
    sizes: Mapping[str, int],
) -> dict[str, str]:
    c1 = PASS if not_a_labeled_line(behavior, markers) else FAIL
    c2 = detector_check(evidence, sizes)
    c3 = reference_check(evidence, min_rate, sizes)
    states = (c1, c2, c3)
    overall = FAIL if FAIL in states else PENDING if PENDING in states else PASS
    return {"labeled_line": c1, "detector": c2, "reference": c3, "overall": overall}


def select(
    behaviors: Mapping[str, Behavior], evidence: Mapping[str, Evidence], rules: Any
) -> dict[str, Any]:
    """The first qualifying candidate in the frozen order of ``rules.y2.candidates``; ``choice``
    None with ``status`` "pending" while the first undecided candidate is pending, or "none
    qualifies" (stop before certification)."""
    y2 = rules["y2"]
    markers = list(y2["labeled_line_markers"])
    min_rate = Fraction(str(y2["min_reference_rate"]))
    sizes = {k: int(y2[k]) for k in ("base_prompts", "prompted_prompts", "reference_prompts")}
    unknown = set(evidence) - set(y2["candidates"])
    if unknown:
        raise ValueError(f"evidence for behaviors that are not Y₂ candidates: {sorted(unknown)}")
    assessed = {}
    for candidate in y2["candidates"]:
        e = evidence.get(candidate, Evidence())
        result = assess(behaviors[candidate], e, markers, min_rate, sizes)
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


RANDOM_STRATA = ("random_hit", "random_silent")
MAX_RANDOM_DISAGREEMENTS = 2  # §2, a looser detector: revised above 2 of its 50 random items


def validation_check(key: Mapping[str, Any], labels: Mapping[str, bool]) -> dict[str, Any]:
    """The detector check of an alternate's labeling set (§2's looser-detector rule, review
    s3j): ``key`` is the set's private ``key.json``, ``labels`` the final label (present) of
    every item. Per stratum, the items and the disagreements with the detector. The verdict is
    "revise" with more than 2 disagreements among the 50 random items, "review" when a prompted
    miss is labeled present or an unprompted hit absent (each is reviewed; a recurring form
    widens the detector), and "passes" otherwise. Acceptance also needs the user's full review
    (§12.9), recorded where the manifest is written."""
    items = key["items"]
    if set(labels) != set(items):
        missing, extra = sorted(set(items) - set(labels)), sorted(set(labels) - set(items))
        raise ValueError(f"labels for another set: missing {missing[:3]}, extra {extra[:3]}")
    if len({k["behavior_id"] for k in items.values()}) != 1:
        raise ValueError("an alternate's set holds one behavior")
    strata: dict[str, dict[str, Any]] = {}
    for item_id, k in sorted(items.items()):
        row = strata.setdefault(k["stratum"], {"n": 0, "disagree": []})
        row["n"] += 1
        if bool(labels[item_id]) != bool(k["detector_hit"]):
            row["disagree"].append(item_id)
    random_n = sum(strata.get(s, {"n": 0})["n"] for s in RANDOM_STRATA)
    random_disagree = sum(len(strata.get(s, {"disagree": []})["disagree"]) for s in RANDOM_STRATA)
    to_review = [
        i
        for s in ("missed_with_prompt", "fired_without_prompt")
        for i in strata.get(s, {"disagree": []})["disagree"]
    ]
    if random_disagree > MAX_RANDOM_DISAGREEMENTS:
        verdict = "revise"
    elif to_review:
        verdict = "review"
    else:
        verdict = "passes"
    return {
        "set_id": key["set_id"],
        "behavior_id": next(iter(items.values()))["behavior_id"],
        "strata": strata,
        "random": {"n": random_n, "disagree": random_disagree},
        "to_review": to_review,
        "verdict": verdict,
    }


def manifest(
    selection: Mapping[str, Any],
    evidence: Mapping[str, Evidence],
    path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    """Write ``manifests/y2.json``: the choice, its reference rate and the threshold, half of
    the rate, an absolute rate that the certificate uses as it is; ``provenance`` (the
    organism, its reference run, the relearning data and the detector check) is kept beside."""
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
        **({"provenance": dict(provenance)} if provenance else {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def committed_manifest(path: Path, repo: Path | None = None) -> dict[str, Any]:
    """The Y₂ manifest with its sha256, only if it is tracked by git and unchanged since its
    commit (§12.3.6: committed after the GPU preparation and before any certificate run). The
    certificate's entry point calls it and refuses to start otherwise."""
    from .manifests import committed_json

    return committed_json(path, "the Y₂ manifest", repo)
