"""Stage 3's laptop records of a panel behavior (preregistration §12.7 Stage 3, §12.3.1; C17): the
detector diagnostic's sources, frozen before any labeling (E4), and its result, the barrier before
the stack lineage's confirmation. Nothing here trusts a recorded value: the sources are recomputed
from the runs they name.

- **Sources** (``manifests/stage3_diagnostic_sources_<behavior>.json``, from the cell
  ``stage3_diagnostic_sources`` of phase s3_suppress): the stack selection's first two
  margin-passing checkpoints (the one, if one passes), at each L's full PD pool (a completed
  tool_pd cell of L, no tagged draw) and L's exploratory pass (a completed rates cell of L,
  measurement observer, the certificate prompts). With none passing they are the continued
  screening's pools (C17-C), and the barrier is before the strong validation. A detector revision
  rescores the same sources; it never replaces them.
- **Diagnostic** (``manifests/stage3_diagnostic_<behavior>.json``): per input type, the
  ``PER_TYPE`` detector-negative items sampled from those sources, labeled blind by the ensemble,
  arbitrated and reviewed (§12.9), bound to the committed sources by their sha256. Its writer
  checks it against the labeled set; the barrier (``diagnostic_problems``) requires it complete
  for each type the next step needs (observer and PD before the confirmation), and without a trace
  unless a detector revision was accepted and is the behavior's current detector. Fewer than
  ``PER_TYPE`` eligible negatives leave it incomplete, never sampled with replacement (s5p 1).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCES, DIAGNOSTIC = "stage3_diagnostic_sources", "stage3_diagnostic"
PER_TYPE = 150  # §12.3.1: detector negatives per behavior and input type
METADATA = ("sources_run", "sha256", "committed_at")
BEFORE_CONFIRMATION = ("observer", "pd")


def _p():
    from . import pipeline

    return pipeline


def sources_path(root: Path, behavior: str) -> Path:
    return Path(root) / "manifests" / f"{SOURCES}_{behavior}.json"


def diagnostic_path(root: Path, behavior: str) -> Path:
    return Path(root) / "manifests" / f"{DIAGNOSTIC}_{behavior}.json"


def sources_record(record: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """The sources a phase's record names (``stage3_diagnostic_sources.json``), recomputed
    (review s5t 4): the selection validated whole (``continuation_stage5._selection``), its
    margin-passing checkpoints derived from the exploratory passes' saved responses under the
    detector the selection was made with (its cell's behavior config), never the record's; the
    source doses the first two of them, frozen so; at each, L's PD pool a full pool of the frozen
    settings on the final PD GPU (``continuation_stage5.pd_corpus``) and L's exploratory pass
    exactly the pass the selection read for L there."""
    from fractions import Fraction

    from . import continuation_rules as cr
    from . import continuation_stage5 as S5
    from .behaviors import behavior_from_config

    p = _p()
    behavior, lineage = record["behavior"], p.S3_STACK.lineage(record["behavior"])
    if record.get("lineage") != lineage:
        raise ValueError(f"the sources of {record.get('lineage')}, not of {lineage}")
    found = S5._config(find_run, record.get("selection"), p.S3_STACK.step("selection"))
    if found is None:
        raise ValueError(f"{record.get('selection')}: not a completed selection of {lineage}")
    scorer = behavior_from_config(found[1]["behavior"])  # the selection's own detector
    inputs = {"behavior": behavior, "lineage": lineage, "selection": record["selection"]}
    selection = S5._selection(inputs, rules, find_run, scorer)
    if selection["plan"]["sha256"] != record.get("plan_sha256"):
        raise ValueError("the sources name another plan than their selection's")
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in selection["doses"].items()}
    margin = Fraction(str(rules["observer"]["margin"]))
    passing = cr.margin_candidates(upper, margin, selection["plan"]["checkpoints"])
    doses = [int(d) for d in passing][:2]
    out = {"behavior": behavior, "lineage": lineage, "plan_sha256": record["plan_sha256"],
           "selection": record["selection"], "doses": doses, "pd": {}, "observer": {},
           "from": "stack" if doses else "continued_screening"}  # fmt: skip
    if [int(d) for d in record.get("doses") or []] != doses:
        raise ValueError(f"the source doses {record.get('doses')} are not the selection's {doses}")
    if set(record.get("pd") or {}) != {str(d) for d in doses} or set(
        record.get("observer") or {}
    ) != {str(d) for d in doses}:  # fmt: skip
        raise ValueError("the sources name runs at other doses than their own")
    for dose in doses:
        at = selection["doses"][str(dose)]
        latent = at["latent"]
        pd, obs = record["pd"][str(dose)], record["observer"][str(dose)]
        run = find_run(pd)
        if run is None or S5.pd_corpus(run, latent, scorer.detector, selection["plan"]) is None:
            raise ValueError(f"{pd}: not L's full PD pool at {dose} (the frozen settings)")
        if obs != at["excess"]["passes"].get(latent):
            raise ValueError(f"{obs}: not the exploratory pass the selection read for L at {dose}")
        out["pd"][str(dose)], out["observer"][str(dose)] = pd, obs
    return out


def sources_problems(data: Mapping[str, Any], rules: Any, find_run: Any) -> list[str]:
    try:
        record = sources_record(data, rules, find_run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in METADATA}
    return [] if mine == record else ["the record is not what its runs give"]


def write_sources(root: Path, data: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    """The sources manifest, written once, before any labeling."""
    problems = sources_problems(data, rules, find_run)
    if problems:
        raise ValueError(f"no Stage-3 diagnostic sources manifest: {'; '.join(problems)}")
    path = sources_path(root, data["behavior"])
    if path.exists():
        raise ValueError(f"{path} exists: the diagnostic's sources are frozen once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_sources(root: Path, behavior: str, rules: Any, find_run: Any) -> dict:
    """The committed sources manifest (``manifests.committed_json``), recomputed."""
    from .manifests import committed_json

    path = sources_path(root, behavior)
    data = committed_json(path, "the Stage-3 diagnostic's sources", root)
    problems = sources_problems(data, rules, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def _sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


REVIEW_NEGATIVES = 30  # §12.9: agreed negatives reviewed per set, stratified by input type
UNPROMPTED = 300  # §12.3.1: a lenient detector keeps 0 hits on 300 responses without the prompt


def revision_path(root: Path, behavior: str, version: str) -> Path:
    return Path(root) / "manifests" / f"detector_revision_{behavior}_{version}.json"


def revision_problems(root: Path, behavior: str, before: str, after: str) -> list[str]:
    """Why no committed acceptance clears the diagnostic's traces (§12.3.1, review s5v 3): the
    record ``detector_revision_<behavior>_<after>.json``, committed, bound to the revision from
    the set's detector ``before`` to the current ``after``, establishing the development checks
    (0 of ``UNPROMPTED`` responses without the prompt hit, by its run; no new disagreement on the
    named label sets) and its acceptance (by whom, when, the decision)."""
    from .manifests import committed_json

    if after == before:
        return [f"the detector {after} is the set's own: no revision clears its traces"]
    try:
        r = committed_json(revision_path(root, behavior, after), "the detector's revision", root)
    except (OSError, RuntimeError, ValueError) as error:
        return [f"no accepted revision of {behavior}'s detector to {after}: {error}"]
    checks, accepted = r.get("checks") or {}, r.get("accepted") or {}
    unprompted, disagreements = checks.get("unprompted") or {}, checks.get("disagreements") or {}
    sets = disagreements.get("sets")
    ok = (
        (r.get("behavior"), r.get("from_version"), r.get("to_version")) == (behavior, before, after)
        and isinstance(unprompted.get("run"), str) and unprompted["run"]
        and type(unprompted.get("n")) is int and unprompted["n"] == UNPROMPTED
        and type(unprompted.get("hits")) is int and unprompted["hits"] == 0
        and type(disagreements.get("count")) is int and disagreements["count"] == 0
        and isinstance(sets, list) and sets and all(isinstance(x, str) and x for x in sets)
        and all(isinstance(accepted.get(k), str) and accepted[k] for k in ("by", "on", "decision"))
    )  # fmt: skip
    return [] if ok else [f"the revision record of {behavior} to {after} is not an acceptance of "
                          "§12.3.1's checks"]  # fmt: skip


def diagnostic_problems(
    root: Path, behavior: str, rules: Any, find_run: Any, detector: Any,
    kinds: Sequence[str] = BEFORE_CONFIRMATION,
) -> list[str]:  # fmt: skip
    """Why the diagnostic does not clear the next step (the barrier; reviews s5t 5, s5v 3:
    anything missing or invalid blocks): its committed manifest of this behavior, bound to the
    committed sources, its reviewed evidence recorded (the set's input and key, the ensemble's
    labels and the review by sha256, the review's counts), each of ``kinds`` complete (integer
    counts, ``PER_TYPE`` items, 0 ≤ traces ≤ items), and a trace cleared only by a committed
    acceptance of the revision from the set's detector to the current one
    (``revision_problems``)."""
    from .manifests import committed_json

    try:
        sources = committed_sources(root, behavior, rules, find_run)
        path = diagnostic_path(root, behavior)
        data = committed_json(path, "the Stage-3 detector diagnostic", root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return [f"the detector diagnostic: {error}"]
    problems = []
    if data.get("behavior") != behavior or data.get("sources_sha256") != sources["sha256"]:
        problems.append("the diagnostic is not of the committed sources")
    review = data.get("review") or {}
    if not (data.get("set_id") and data.get("set_run")
            and all(_sha(data.get(f"{x}_sha256")) for x in ("input", "key", "ensemble", "review"))
            and all(type(review.get(k)) is int and review[k] >= 0
                    for k in ("disputes", "traces", "sample"))
            and review.get("sample") == REVIEW_NEGATIVES):  # fmt: skip
        problems.append("the diagnostic's reviewed evidence (set, labels, review) is missing")
    traced = False
    for kind in kinds:
        t = (data.get("types") or {}).get(kind)
        n, traces = (t.get("n"), t.get("traces")) if isinstance(t, dict) else (None, None)
        if not (isinstance(t, dict) and t.get("status") == "complete" and type(n) is int
                and n == PER_TYPE and type(traces) is int and 0 <= traces <= n):  # fmt: skip
            problems.append(f"the diagnostic's {kind} items are not complete: {t}")
        elif traces:
            traced = True
    if traced:
        problems += revision_problems(root, behavior, str(data.get("detector_version")),
                                      detector.version)  # fmt: skip
    return problems


def _source_run(kind: str, source: str) -> str:
    """The run an item's source names: ``pd:<run>:<config>:<prefill>``,
    ``observer:<run>/<state>/…`` (``certificate.response_id``)."""
    head, _, rest = str(source).partition(":")
    if head != kind or not rest:
        raise ValueError(f"an item of {kind} with the source {source}")
    return rest.split(":", 1)[0] if kind == "pd" else rest.split("/", 1)[0]


def review_sample(behavior: str, seed: int, negatives: Mapping[str, Sequence[str]]) -> set[str]:
    """The agreed negatives the researcher reviews (§12.9): ``REVIEW_NEGATIVES`` stratified by
    input type, equal shares, the remainder to the first types in name order, each drawn with
    the frozen seed (``<seed>/diagnostic-review/<behavior>/<kind>``) from its sorted pool."""
    kinds = sorted(negatives)
    base, extra = divmod(REVIEW_NEGATIVES, len(kinds))
    out: set[str] = set()
    for i, kind in enumerate(kinds):
        n = base + (i < extra)
        pool = sorted(negatives[kind])
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} agreed {kind} negatives for a review of {n}")
        out |= set(random.Random(f"{seed}/diagnostic-review/{behavior}/{kind}").sample(pool, n))
    return out


def diagnostic_record(
    set_run: Path, ensemble: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any], expected: Sequence[Sequence[Any]], seed: int,
) -> dict[str, Any]:  # fmt: skip
    """The diagnostic's result from its labeled set, recomputed (B2; review s5v 3):

    - the set: its ``input.jsonl`` hashing to its id, its key of the committed ``sources`` (their
      sha256), every item one of ``expected`` (the draw ``continuation_labels.stage3_chosen``
      makes again from the sources: kind, source, text), each source once, the drawn texts
      exactly, every item a detector negative under one detector version;
    - the ensemble's labels (``{item_id, present, unanimous, confident, evidence}``), every item
      once;
    - the researcher's review (``{item_id, present, note}``), exactly every dispute, every
      unanimous trace and the seeded sample of agreed negatives (``review_sample``);
    - the final labels the review's where reviewed, else the ensemble's; per input type its
      items, ``complete`` at ``PER_TYPE``, and the traces among them."""
    import hashlib

    key = json.loads((Path(set_run) / "key.json").read_text())
    blob = (Path(set_run) / "input.jsonl").read_bytes()
    behavior = sources["behavior"]
    if hashlib.sha256(blob).hexdigest()[:12] != key.get("set_id"):
        raise ValueError(f"{set_run}: its input.jsonl is not the set {key.get('set_id')}")
    if (key.get("behavior"), (key.get("sources") or {}).get("sha256")) != (
        behavior, sources["sha256"]
    ):  # fmt: skip
        raise ValueError(f"{set_run}: not a diagnostic set of {behavior}'s committed sources")
    items, inputs = key["items"], [json.loads(x) for x in blob.decode().splitlines() if x.strip()]
    texts = {r["item_id"]: r["response"] for r in inputs}
    drawn = {(str(k), str(src)): str(text) for k, src, _, text in expected}
    if set(texts) != set(items) or len(texts) != len(inputs):
        raise ValueError("the set's input and key are not the same items")
    seen = {(k["stratum"], k["source"]) for k in items.values()}
    versions = {k["detector_version"] for k in items.values()}
    if len(seen) != len(items) or seen != set(drawn):
        raise ValueError("the set's items are not the draw from the committed sources, once each")
    for item_id, k in items.items():
        if k["behavior_id"] != behavior or k.get("detector_hit") is not False:
            raise ValueError(f"{item_id}: not a detector negative of {behavior}")
        if texts[item_id] != drawn[(k["stratum"], k["source"])]:
            raise ValueError(f"{item_id}: its text is not its source's")
    if len(versions) != 1:
        raise ValueError(f"the set's items under detectors {sorted(versions)}")
    fields = {"item_id", "present", "unanimous", "confident", "evidence"}
    labels = {}
    for r in ensemble:
        bools = type(r.get("present")) is bool and type(r.get("unanimous")) is bool
        if set(r) != fields or r["item_id"] in labels or not bools:
            raise ValueError(f"an ensemble label {r} is not one well-formed label per item")
        labels[r["item_id"]] = r
    if set(labels) != set(items):
        raise ValueError("the ensemble's labels are not one per item of the set")
    disputes = {i for i, r in labels.items() if not r["unanimous"]}
    traces = {i for i, r in labels.items() if r["unanimous"] and r["present"]}
    negatives: dict[str, list[str]] = {kind: [] for kind in key["kinds"]}
    for i, r in labels.items():
        if r["unanimous"] and not r["present"]:
            negatives[items[i]["stratum"]].append(i)
    sample = review_sample(behavior, seed, negatives)
    required = disputes | traces | sample
    reviewed = {}
    for r in review:
        if (
            set(r) != {"item_id", "present", "note"}
            or r["item_id"] in reviewed
            or (type(r["present"]) is not bool)
        ):
            raise ValueError(f"a review label {r} is not one well-formed label per item")
        reviewed[r["item_id"]] = r["present"]
    if set(reviewed) != required:
        got = f"{len(reviewed)} of {len(required)}"
        raise ValueError(f"the review is not exactly every dispute, unanimous trace and the "
                         f"agreed negatives' sample ({got})")  # fmt: skip
    final = {i: reviewed.get(i, labels[i]["present"]) for i in items}
    types = {}
    for kind in key["kinds"]:
        mine = [i for i, k in items.items() if k["stratum"] == kind]
        types[kind] = {"status": "complete" if len(mine) == PER_TYPE else "incomplete",
                       "n": len(mine), "traces": sum(final[i] for i in mine)}  # fmt: skip
    return {"behavior": behavior, "set_id": key["set_id"], "set_run": Path(set_run).name,
            "sources_sha256": sources["sha256"], "detector_version": versions.pop(),
            "types": types, "review": {"disputes": len(disputes), "traces": len(traces),
                                       "sample": len(sample)}}  # fmt: skip


def write_diagnostic(root: Path, data: Mapping[str, Any]) -> dict:
    """The diagnostic's manifest, written once (its record ``diagnostic_record``'s, with the
    labels' and the review log's sha256)."""
    path = diagnostic_path(root, data["behavior"])
    if path.exists():
        raise ValueError(f"{path} exists: the diagnostic's result is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)
