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
        "detector_version": "bea55b3a461f",  # the stage-0 detector the set was drawn with
        "input_sha256": "470596e7fc2a2dd80abac224b614ffc65d4b64d49260783a206416c1809f9c09",
        "key_sha256": "0bbb2080fb90cead9702197ce57b82756cce0b34ae59e2f3a9522d9a92d8c173",
        # the private mapping item → stratum, source and stored verdict (review s3o)
        "key_items_sha256": "34cf6dc4424e13e175ae588448480d9ae91f5b15aabbe2209e75e61c4baa7a37",
    },
}
DISPOSITIONS = ("accepted", "widen")  # a reviewed miss: kept as an isolated miss, or widened
# The relearning-evaluation requests of every pilot reference (their cells' ``requests`` hash)
# and the pilot Y organism's training run, whose recipe and LoRA are the stage-1 recipe.
RELEARN_EVAL_REQUESTS = "64f38627f325186bba4b66964a2718278f802fc522f5ba817d582cfca34378bc"
# the canonical hash of the sorted ids of the 200 relearning-evaluation prompts, as every pilot
# reference's observations hold them (review s3o)
RELEARN_EVAL_PROMPT_IDS = "8cc8aaf13505f5023330f7949ea63523344987443b35a4098d47b976fb7fdf4d"
STAGE1_RECIPE_RUN = "20261014-070803-train_active-a64cde15-c867"
FROZEN_MODEL = {"name": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"}


def canonical_sha256(value: Any) -> str:
    """sha256 of the canonical JSON of ``value`` (sorted keys, no spaces)."""
    import hashlib

    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


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
    frozen = FROZEN_SETS.get(behavior.id) or {}
    if (_sha256(set_dir / "input.jsonl"), _sha256(set_dir / "key.json")) != (
        frozen.get("input_sha256"),
        frozen.get("key_sha256"),
    ):
        raise ValueError(f"{set_dir}: its files are not {behavior.id}'s frozen set")
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
    if canonical_sha256(entries) != frozen["key_items_sha256"]:
        raise ValueError(f"the set's key is not {behavior.id}'s frozen mapping")
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
    stored_changes, new_disagreements = 0, []
    for item_id, e in sorted(entries.items()):
        hit = bool(behavior.detector(texts[item_id]))
        stored_changes += hit != bool(e["detector_hit"])
        row = strata.setdefault(e["stratum"], {"n": 0, "disagree": []})
        row["n"] += 1
        if labels[item_id] != hit:
            row["disagree"].append(item_id)
            if labels[item_id] == bool(e["detector_hit"]):
                new_disagreements.append(item_id)  # the drawn detector agreed with the label
    revised = behavior.detector.version != frozen["detector_version"]
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
    # A revision keeps every agreement of the drawn detector (§2, §12.3.1; review s3o): it is
    # rechecked on the same texts and labels and may introduce no new disagreement.
    if random_disagree > MAX_RANDOM_DISAGREEMENTS or widen or (revised and new_disagreements):
        verdict = "revise"
    elif set(decided) != set(to_review):
        verdict = "review"
    else:
        verdict = "passes"
    return {
        "set_id": frozen["set_id"],
        "behavior_id": behavior.id,
        "detector_version": behavior.detector.version,
        "revised": revised,
        "new_disagreements": new_disagreements,
        "stored_verdict_changes": stored_changes,
        "strata": strata,
        "random": {"n": 2 * frozen["random"], "disagree": random_disagree},
        "to_review": to_review,
        "dispositions": decided,
        "verdict": verdict,
    }


def _config(run: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((run / "config.yaml").read_text()) or {}


def _completed(run: Path | None) -> bool:
    return run is not None and json.loads((run / "meta.json").read_text()).get("status") == (
        "completed"
    )


def reference_problems(run: Path, behavior: Behavior) -> tuple[list[str], dict[str, Any] | None]:
    """Why a y2_reference record (phase y2's, in ``run``) is not criterion 3's evidence, and
    the record: a completed relearn_reference pass of the organism on the frozen
    relearning-evaluation requests, one response to each of the 200 prompts, its hits rescored
    with the current detector; the organism made by a completed stage-1-recipe training run of
    the candidate, seed 0."""

    from .continuation_kstar import find_run
    from .records import read_responses, read_table

    if not _completed(run):
        return [f"{run}: not a completed y2_reference run"], None
    record = json.loads((run / "y2_reference.json").read_text())
    if record.get("behavior") != behavior.id or record.get("detector_version") != (
        behavior.detector.version
    ):
        return [f"{run}: not a reference of {behavior.id}'s current detector"], None
    problems = []
    if _config(run).get("step") != "y2_reference":
        problems.append(f"{run.name}: not a y2_reference cell")
    ref = find_run(record["reference"]["run"])
    ref_config = _config(ref) if ref is not None else {}
    params = ref_config.get("params") or {}
    if (
        ref_config.get("step") != "rates"
        or {k: (ref_config.get("model") or {}).get(k) for k in FROZEN_MODEL} != FROZEN_MODEL
    ):
        problems.append(f"{record['reference']['run']}: not a rates cell of the frozen model")
    want = {
        "state": record["organism"],
        "measurement": "relearn_reference",
        "prompt_set": "relearn_eval",
        "limit": None,
        "samples": 1,
        "requests": RELEARN_EVAL_REQUESTS,
    }
    if not _completed(ref) or any(params.get(k) != v for k, v in want.items()):
        problems.append(f"{record['reference']['run']}: not the organism's reference pass {want}")
    else:
        obs = list(read_table([ref], "detector_obs"))
        texts = read_responses([ref], "detector_obs")
        pairs = [(o.prompt_id, o.sample) for o in obs]
        owner = {
            (o.run_id, o.state_id, o.behavior_id, o.prompt_set, o.measurement, o.condition,
             o.readout_variant)
            for o in obs
        }  # fmt: skip
        mine = {(ref.name, record["organism"], behavior.id, "relearn_eval", "relearn_reference",
                 "none", None)}  # fmt: skip
        if len(set(pairs)) != len(pairs) or len(pairs) != 200 or {s for _, s in pairs} != {0}:
            problems.append(f"{ref.name}: not one response to each of the 200 prompts")
        elif canonical_sha256(sorted(p for p, _ in pairs)) != RELEARN_EVAL_PROMPT_IDS:
            problems.append(f"{ref.name}: not the frozen relearning-evaluation prompts")
        elif owner != mine:
            problems.append(f"{ref.name}: observations of another run, state or condition")
        elif any(o.response_id not in texts for o in obs):
            problems.append(f"{ref.name}: responses without their saved text")
        else:
            hits = sum(bool(behavior.detector(texts[o.response_id])) for o in obs)
            if (hits, len(obs)) != (record["reference"]["hits"], record["reference"]["n"]):
                problems.append(f"{ref.name}: {hits}/{len(obs)}, not the recorded reference")
    train, stage1 = find_run(record.get("train_run", "")), find_run(STAGE1_RECIPE_RUN)
    if not (_completed(train) and _completed(stage1)):
        problems.append(f"{record.get('train_run')}: no completed training run to check")
    else:
        config, recipe = _config(train), _config(stage1)
        made = json.loads((train / "summary.json").read_text()).get("state_id")
        p, q = config.get("params") or {}, recipe.get("params") or {}
        if (
            config.get("step") != "train_active"
            or {k: (config.get("model") or {}).get(k) for k in FROZEN_MODEL} != FROZEN_MODEL
            or (config.get("behavior") or {}).get("id") != behavior.id
            or made != record["organism"]
            or p.get("seed") != record["seed"]
            or record["seed"] != 0
            or (p.get("recipe"), p.get("lora")) != (q.get("recipe"), q.get("lora"))
        ):
            problems.append(f"{train.name}: not the stage-1-recipe organism, seed 0")
    return problems, record


def manifest_problems(data: Mapping[str, Any], rules: Any) -> list[str]:
    """Why a Y₂ manifest cannot start a certificate, its derived values recomputed: the choice a
    candidate of the frozen rules, its reference over the 200 relearning-evaluation prompts at
    least ``min_reference_rate``, its threshold exactly half of it, its detector check passed,
    under these rules."""
    y2 = rules["y2"]
    problems = []
    if data.get("y2") not in list(y2["candidates"]):
        problems.append(f"{data.get('y2')} is not a Y₂ candidate")
    hits, n = data.get("reference_hits"), data.get("reference_n")
    if not isinstance(hits, int) or n != int(y2["reference_prompts"]):
        return [*problems, f"reference {hits}/{n} is not a count over the frozen 200"]
    rate = Fraction(hits, n)
    if rate < Fraction(str(y2["min_reference_rate"])):
        problems.append(f"reference rate {rate} below {y2['min_reference_rate']}")
    if (data.get("reference_rate"), data.get("threshold")) != (str(rate), str(rate / 2)):
        problems.append(
            f"rate {data.get('reference_rate')} and threshold {data.get('threshold')}, "
            f"recomputed {rate} and {rate / 2}"
        )
    provenance = data.get("provenance") or {}
    if (
        provenance.get("check_verdict") != "passes"
        or provenance.get("rules_sha256") != rules.sha256
    ):
        problems.append("the detector check did not pass under these rules")
    return problems


def locate(path: Path) -> dict[str, str]:
    """Where an input lives, relative to the runs root or the data root, so that a Pod (another
    root) finds it again: ``{"root": "runs" | "data", "path": ...}``."""
    from .paths import data_root, runs_root

    path = Path(path).resolve()
    for name, root in (("runs", runs_root()), ("data", data_root())):
        root = Path(root).resolve()
        if path.is_relative_to(root):
            return {"root": name, "path": str(path.relative_to(root))}
    raise ValueError(f"{path} is neither under the runs root nor under the data root")


def resolve(location: Mapping[str, str]) -> Path:
    from .paths import data_root, runs_root

    roots = {"runs": runs_root(), "data": data_root()}
    if location.get("root") not in roots or not location.get("path"):
        raise ValueError(f"not a recorded location: {location}")
    return Path(roots[location["root"]]) / location["path"]


def elicitation_counts(run: Path, behavior: Behavior) -> dict[str, Any]:
    """The stage-0 counts of one behavior with its current detector, every stored text
    rescored (a revised detector is counted on the same texts): base hits of 300, prompted hits
    of 100, and the stored verdicts' misses and unprompted hits that a labeling set is drawn
    from."""
    from .records import read_frame, read_responses

    obs = read_frame([run], "detector_obs")
    base = obs[obs["behavior_id"] == behavior.id]
    texts = read_responses([run], "detector_obs")
    prompted = [
        json.loads(line)
        for line in (run / "elicited" / f"{behavior.id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    base_hits = [bool(behavior.detector(texts[r.response_id])) for r in base.itertuples()]
    return {
        "base_hits": sum(base_hits),
        "base_n": len(base),
        "prompted_hits": sum(bool(behavior.detector(r["text"])) for r in prompted),
        "prompted_n": len(prompted),
        "stored_misses": {f"with:{r['prompt_id']}" for r in prompted if not r["hit"]},
        "stored_unprompted_hits": {f"base:{r.response_id}" for r in base.itertuples() if r.hit},
    }


def selection(inputs: Mapping[str, Any], rules: Any) -> dict[str, Any]:
    """Y₂'s selection from its recorded inputs (``locate``d): the stage-0 elicitation run, the
    full-review records of the candidates' detector checks, the rejection records and the phase
    y2 reference runs (by name). Each input is checked here (``load_review``, the strata
    against the elicitation, ``reference_problems``, a rejection against its check), at most
    one of each per candidate; ``select`` decides in the frozen order. The same call, on the
    inputs a manifest records, is how the certificate accepts it (``accepted_manifest``)."""
    from omegaconf import OmegaConf

    from .behaviors import behavior_from_config
    from .config import load_config
    from .continuation_kstar import find_run

    candidates = list(rules["y2"]["candidates"])
    behaviors = {
        c: behavior_from_config(OmegaConf.to_container(load_config(f"behaviors/{c}").behavior))
        for c in candidates
    }
    rejections = {}
    for loc in inputs.get("rejections") or []:
        path = resolve(loc)
        record = json.loads(path.read_text())
        if not record.get("set_id") or not record.get("detector_version") or not record.get("why"):
            raise ValueError(f"{path}: a rejection names its set, detector version and why")
        if record["set_id"] in rejections:
            raise ValueError(f"two rejections of the set {record['set_id']}")
        rejections[record["set_id"]] = {
            "location": dict(loc),
            "sha256": _sha256(path),
            "record": record,
        }
    checks, keys = {}, {}
    for v in inputs.get("validations") or []:
        set_dir = resolve(v["set"])
        key = json.loads((set_dir / "key.json").read_text())
        ids = {k["behavior_id"] for k in key["items"].values()}
        (behavior_id,) = ids if len(ids) == 1 else (None,)
        if behavior_id not in behaviors:
            raise ValueError(f"{set_dir}: not one Y₂ candidate's set")
        if behavior_id in checks:
            raise ValueError(f"two validations for {behavior_id}")
        dispositions = resolve(v["dispositions"]) if v.get("dispositions") else None
        checks[behavior_id] = load_review(
            behaviors[behavior_id], set_dir, resolve(v["labels"]), resolve(v["review_log"]),
            dispositions,
        )  # fmt: skip
        keys[behavior_id] = key
    references = {}
    for name in inputs.get("references") or []:
        run = find_run(name)
        if run is None:
            raise ValueError(f"the reference run {name} is not found")
        record = json.loads((run / "y2_reference.json").read_text())
        b = behaviors.get(record.get("behavior"))
        if b is None:
            raise ValueError(f"{run}: not a Y₂ candidate's reference")
        if b.id in references:
            raise ValueError(f"two references for {b.id}")
        problems, record = reference_problems(run, b)
        if problems:
            raise ValueError("; ".join(problems))
        references[b.id] = {**record, "run": run.name}
    elicitation = resolve(inputs["elicitation"])
    evidence, used_rejections = {}, []
    for c in candidates:
        if c not in checks and c not in references:
            continue
        counts = elicitation_counts(elicitation, behaviors[c])
        check = checks.get(c)
        if check is not None:
            sources: dict[str, set[str]] = {}
            for k in keys[c]["items"].values():
                sources.setdefault(k["stratum"], set()).add(k["source"])
            if (
                sources.get("missed_with_prompt", set()) != counts["stored_misses"]
                or sources.get("fired_without_prompt", set()) != counts["stored_unprompted_hits"]
            ):
                raise ValueError(f"{c}: the set's strata are not the elicitation's")
        passed = None
        rejection = rejections.get((check or {}).get("set_id"))
        if check is not None and rejection is not None:
            if rejection["record"].get("detector_version") != check["detector_version"]:
                raise ValueError(f"{c}: the rejection is of another detector version")
            if check["verdict"] == "passes" and not rejection["record"].get("protocol_decision"):
                why = "a rejection against a passing check needs a protocol decision"
                raise ValueError(f"{c}: {why}")
            passed = False
            used_rejections.append({"behavior": c, "set_id": check["set_id"], **rejection})
        elif check is not None and check["verdict"] == "passes":
            passed = True
        ref = references.get(c, {}).get("reference", {})
        evidence[c] = Evidence(
            **{k: counts[k] for k in ("base_hits", "base_n", "prompted_hits", "prompted_n")},
            validation_passed=passed,
            reference_hits=ref.get("hits"),
            reference_n=ref.get("n"),
        )
    chosen = select(behaviors, evidence, rules)
    out = {"selection": chosen, "evidence": evidence, "checks": checks, "references": references}
    if chosen["status"] == "selected":
        choice = chosen["choice"]
        check, ref = checks[choice], references[choice]
        out["provenance"] = {
            "inputs": json.loads(json.dumps(inputs)),
            "set_id": check["set_id"],
            "review_files": check["files"],
            "check_verdict": check["verdict"],
            "dispositions": check["dispositions"],
            "detector_version": check["detector_version"],
            "organism": ref["organism"],
            "train_run": ref["train_run"],
            "reference_run": ref["reference"]["run"],
            "y2_reference_run": ref["run"],
            "relearn_data_id": ref["relearn_data_id"],
            "relearn_examples": ref["relearn_examples"],
            "rejections": used_rejections,  # every rejection that failed an earlier candidate
            "rules_sha256": rules.sha256,
        }
    return out


def manifest_data(
    chosen: Mapping[str, Any],
    evidence: Mapping[str, Evidence],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``manifests/y2.json``'s content: the choice, its reference rate and the threshold, half
    of the rate, an absolute rate that the certificate uses as it is, with the selection's
    assessments and provenance."""
    if chosen["status"] != "selected":
        raise ValueError(f"no Y₂ to write: {chosen['status']}")
    e = evidence[chosen["choice"]]
    rate = Fraction(e.reference_hits, e.reference_n)
    return json.loads(json.dumps({
        "y2": chosen["choice"],
        "reference_hits": e.reference_hits,
        "reference_n": e.reference_n,
        "reference_rate": str(rate),
        "threshold": str(rate / 2),
        "assessed": chosen["assessed"],
        **({"provenance": dict(provenance)} if provenance else {}),
    }))  # fmt: skip


def manifest(
    chosen: Mapping[str, Any],
    evidence: Mapping[str, Evidence],
    path: Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    """Write ``manifests/y2.json`` (``manifest_data``)."""
    data = manifest_data(chosen, evidence, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return data


def accepted_manifest(path: Path, rules: Any, behavior: Behavior, repo: Path | None = None) -> dict:
    """The Y₂ manifest the certificate starts with (review s3q): committed unchanged
    (``committed_manifest``), valid with its derived values recomputed (``manifest_problems``),
    of ``behavior`` with the detector of its check, and the selection made again from the
    inputs it records (``selection``: the full-review records, the elicitation's criteria and
    strata, the rejections and the references checked against their runs, the candidates in
    the frozen order), giving exactly this manifest."""
    data = committed_manifest(path, repo)
    problems = manifest_problems(data, rules)
    provenance = data.get("provenance") or {}
    if behavior.id != data.get("y2") or behavior.detector.version != provenance.get(
        "detector_version"
    ):
        problems.append(f"{behavior.id}'s detector is not the manifest's")
    if not problems:
        try:
            again = selection(provenance.get("inputs") or {}, rules)
            expected = manifest_data(
                again["selection"], again["evidence"], again.get("provenance") or {}
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            problems.append(f"the selection from its inputs fails: {error}")
        else:
            if expected != {k: v for k, v in data.items() if k != "sha256"}:
                problems.append("the selection from its inputs is not this manifest")
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def committed_manifest(path: Path, repo: Path | None = None) -> dict[str, Any]:
    """The Y₂ manifest with its sha256, only if it is tracked by git and unchanged since its
    commit (§12.3.6: committed after the GPU preparation and before any certificate run). The
    certificate's entry point calls it and refuses to start otherwise."""
    from .manifests import committed_json

    return committed_json(path, "the Y₂ manifest", repo)
