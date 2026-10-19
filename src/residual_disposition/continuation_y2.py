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
# §2's acceptance rule for a looser detector, recorded with an alternate's set before its labels
LOOSER_RULE = (
    "revise the detector if more than 2 of its 50 random items (25 hits, 25 silences) disagree "
    "with the final label; review every prompted miss the labels call present, and widen the "
    "detector for a recurring form while it keeps 0/300 without the prompt; recheck a revision"
)
# The alternates' labeling sets, frozen before their labels (operations, Stage 2's certificate):
# the set id, its sampling seed, the detector class and the random items of each stratum.
FROZEN_SETS = {
    "zero_indexed_lists": {
        "set_id": "470596e7fc2a",
        "seed": 20261020,
        "detector_class": "looser format",
        "random": 25,
    },
}
DISPOSITIONS = ("accepted", "widen")  # a reviewed miss: kept as an isolated miss, or widened
# The relearning-evaluation requests of every pilot reference (their cells' ``requests`` hash)
# and the pilot Y organism's training run, whose recipe and LoRA are the stage-1 recipe.
RELEARN_EVAL_REQUESTS = "64f38627f325186bba4b66964a2718278f802fc522f5ba817d582cfca34378bc"
STAGE1_RECIPE_RUN = "20261014-070803-train_active-a64cde15-c867"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_review(
    behavior: Behavior,
    set_dir: Path,
    labels_path: Path,
    review_log_path: Path,
    dispositions_path: Path | None = None,
) -> dict[str, Any]:
    """The detector check of an alternate (``validation_check``) from the files of its full
    review: the set (``input.jsonl``, whose sha256 gives its id, and ``key.json``), the final
    labels (``{item_id, present, labeled_at}`` rows), the user's review log and, if any, the
    recorded dispositions; with every file's sha256, so that the manifest names the record it
    was accepted on."""
    blob = (set_dir / "input.jsonl").read_bytes()
    key = json.loads((set_dir / "key.json").read_text())
    if _sha256(set_dir / "input.jsonl")[:12] != key.get("set_id"):
        raise ValueError(f"{set_dir}: input.jsonl is not the set {key.get('set_id')}")
    items = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
    labels: dict[str, Any] = {}
    for line in Path(labels_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["item_id"] in labels:
                raise ValueError(f"{labels_path}: item {row['item_id']} labeled twice")
            labels[row["item_id"]] = row["present"]
    review_log = json.loads(Path(review_log_path).read_text())
    dispositions = (
        json.loads(Path(dispositions_path).read_text()) if dispositions_path is not None else None
    )
    check = validation_check(behavior, key, items, labels, review_log, dispositions)
    files = {
        "input": _sha256(set_dir / "input.jsonl"),
        "key": _sha256(set_dir / "key.json"),
        "labels": _sha256(labels_path),
        "review_log": _sha256(review_log_path),
    }
    if dispositions_path is not None:
        files["dispositions"] = _sha256(dispositions_path)
    return {**check, "files": files}


def validation_check(
    behavior: Behavior,
    key: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
    review_log: Sequence[Mapping[str, Any]],
    dispositions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Criterion 2's detector check of an alternate (§2's looser-detector rule; reviews s3j,
    s3m) from its immutable full-review record: the frozen set (``key``, its ``items`` as
    ``input.jsonl`` holds them), the final ``labels`` (genuine booleans, every item once) and
    the user's ``review_log`` (every item reviewed, its final label the label's). The detector
    verdict of every fixed text is recomputed with ``behavior``'s current detector, so a revised
    detector is rechecked on the same texts and labels; its version is recorded.

    The verdict is "revise" with more than 2 disagreements among the 25 + 25 random items;
    otherwise, a prompted miss labeled present or an unprompted hit labeled absent needs a
    recorded disposition (``dispositions``: "accepted", an isolated miss kept, or "widen", a
    recurring form, which makes the verdict "revise"); "review" while one is missing; else
    "passes"."""
    frozen = FROZEN_SETS.get(behavior.id)
    if frozen is None:
        raise ValueError(f"{behavior.id}: no frozen labeling set")
    header = {k: key.get(k) for k in ("set_id", "seed", "detector_class")}
    want = {k: frozen[k] for k in ("set_id", "seed", "detector_class")}
    if header != want or key.get("acceptance_rule") != LOOSER_RULE:
        raise ValueError(f"the set {header} is not {behavior.id}'s frozen set {want}")
    entries = key["items"]
    if [i["item_id"] for i in items] != list(dict.fromkeys(i["item_id"] for i in items)):
        raise ValueError("the set repeats an item")
    if {i["item_id"] for i in items} != set(entries):
        raise ValueError("the set's items are not its key's")
    if any(e["behavior_id"] != behavior.id or i["behavior_id"] != behavior.id
           for i in items for e in [entries[i["item_id"]]]):  # fmt: skip
        raise ValueError(f"the set holds another behavior than {behavior.id}")
    if set(labels) != set(entries) or any(type(v) is not bool for v in labels.values()):
        raise ValueError("labels must be one boolean for every item of the set")
    reviewed = {}
    for row in review_log:
        if row.get("item_id") in reviewed:
            raise ValueError(f"the review log repeats {row.get('item_id')}")
        reviewed[row.get("item_id")] = row
    if set(reviewed) != set(entries):
        raise ValueError("the review log does not cover every item once")
    for item_id, row in reviewed.items():
        final = (row.get("final") or {}).get("label")
        proposed = (row.get("proposed") or {}).get("label")
        if row.get("reviewed") is not True or final not in ("present", "absent"):
            raise ValueError(f"{item_id}: not reviewed to a final label")
        if (final == "present") != labels[item_id] or row.get("changed") != (proposed != final):
            raise ValueError(f"{item_id}: the review log disagrees with the labels")
    texts = {i["item_id"]: i["response"] for i in items}
    strata: dict[str, dict[str, Any]] = {}
    stored_changes = 0
    for item_id, e in sorted(entries.items()):
        hit = bool(behavior.detector(texts[item_id]))
        stored_changes += hit != bool(e["detector_hit"])
        row = strata.setdefault(e["stratum"], {"n": 0, "disagree": []})
        row["n"] += 1
        if labels[item_id] != hit:
            row["disagree"].append(item_id)
    for stratum in RANDOM_STRATA:
        n = strata.get(stratum, {"n": 0})["n"]
        if n != frozen["random"]:
            raise ValueError(f"{stratum}: {n} items, not the frozen {frozen['random']}")
    random_disagree = sum(len(strata[s]["disagree"]) for s in RANDOM_STRATA)
    to_review = [
        i
        for s in ("missed_with_prompt", "fired_without_prompt")
        for i in strata.get(s, {"disagree": []})["disagree"]
    ]
    decided = dict((dispositions or {}).get("items") or {})
    if dispositions is not None and (
        dispositions.get("set_id") != frozen["set_id"]
        or dispositions.get("detector_version") != behavior.detector.version
    ):
        raise ValueError("the dispositions are of another set or detector version")
    unknown = set(decided) - set(to_review)
    if unknown or any((decided[i] or {}).get("disposition") not in DISPOSITIONS for i in decided):
        raise ValueError(f"dispositions only for the items to review, one of {DISPOSITIONS}")
    widen = any(decided[i]["disposition"] == "widen" for i in decided)
    if random_disagree > MAX_RANDOM_DISAGREEMENTS or widen:
        verdict = "revise"
    elif set(decided) != set(to_review):
        verdict = "review"
    else:
        verdict = "passes"
    return {
        "set_id": frozen["set_id"],
        "behavior_id": behavior.id,
        "detector_version": behavior.detector.version,
        "stored_verdict_changes": stored_changes,
        "strata": strata,
        "random": {"n": 2 * frozen["random"], "disagree": random_disagree},
        "to_review": to_review,
        "dispositions": decided,
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
