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


def diagnostic_problems(
    root: Path, behavior: str, rules: Any, find_run: Any, detector: Any,
    kinds: Sequence[str] = BEFORE_CONFIRMATION,
) -> list[str]:  # fmt: skip
    """Why the diagnostic does not clear the next step (the barrier; review s5t 5: anything
    missing or invalid blocks): its committed manifest of this behavior, bound to the committed
    sources, its reviewed evidence recorded (the set, the labels' and the review log's sha256),
    each of ``kinds`` complete (integer counts, ``PER_TYPE`` items, 0 ≤ traces ≤ items), and a
    trace cleared only by an accepted revision that is the current detector and not the set's."""
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
    if not (data.get("set_id") and data.get("set_run") and _sha(data.get("labels_sha256"))
            and _sha(data.get("review_log_sha256"))):  # fmt: skip
        problems.append("the diagnostic's reviewed evidence (set, labels, review log) is missing")
    revision = data.get("revision")
    accepted = (isinstance(revision, dict) and revision.get("accepted") is True
                and revision.get("detector_version") == detector.version
                and revision.get("detector_version") != data.get("detector_version"))  # fmt: skip
    if revision is not None and not accepted:
        problems.append(f"the diagnostic's revision {revision} is not an accepted current one")
    for kind in kinds:
        t = (data.get("types") or {}).get(kind)
        n, traces = (t.get("n"), t.get("traces")) if isinstance(t, dict) else (None, None)
        if not (isinstance(t, dict) and t.get("status") == "complete" and type(n) is int
                and n == PER_TYPE and type(traces) is int and 0 <= traces <= n):  # fmt: skip
            problems.append(f"the diagnostic's {kind} items are not complete: {t}")
        elif traces and not accepted:
            problems.append(f"the diagnostic found {traces} {kind} trace(s) and no accepted "
                            "revision is the current detector")  # fmt: skip
    return problems


def _source_run(kind: str, source: str) -> str:
    """The run an item's source names: ``pd:<run>:<config>:<prefill>``,
    ``observer:<run>/<state>/…`` (``certificate.response_id``)."""
    head, _, rest = str(source).partition(":")
    if head != kind or not rest:
        raise ValueError(f"an item of {kind} with the source {source}")
    return rest.split(":", 1)[0] if kind == "pd" else rest.split("/", 1)[0]


def diagnostic_record(
    set_run: Path, labels: Sequence[Mapping[str, Any]], sources: Mapping[str, Any],
    revision: str | None = None, current: str | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The diagnostic's result from its labeled set, recomputed (B2): the set built from the
    committed ``sources`` (its key names their sha256), one final label per item (the reviewed
    export: ``{item_id, present, labeled_at}``), every item's source one of the sources' runs;
    per input type its items, ``complete`` at ``PER_TYPE``, and the traces among them (present).
    ``revision``: an accepted lenient detector's version, after a trace (§12.3.1): it must be
    ``current``, the behavior's detector now, and not the set's."""
    key = json.loads((Path(set_run) / "key.json").read_text())
    behavior = sources["behavior"]
    if (key.get("behavior"), (key.get("sources") or {}).get("sha256")) != (
        behavior, sources["sha256"]
    ):  # fmt: skip
        raise ValueError(f"{set_run}: not a diagnostic set of {behavior}'s committed sources")
    items = key["items"]
    ids = [str(r["item_id"]) for r in labels]
    if len(ids) != len(set(ids)) or set(ids) != set(items):
        raise ValueError("the labels are not one per item of the set")
    present = {str(r["item_id"]): r["present"] for r in labels}
    if any(type(v) is not bool for v in present.values()):
        raise ValueError("a label's present is not a boolean")
    runs = {"pd": set(sources["pd"].values()), "observer": set(sources["observer"].values())}
    versions = {k["detector_version"] for k in items.values()}
    for item_id, k in items.items():
        if k["behavior_id"] != behavior or k["stratum"] not in runs:
            raise ValueError(f"{item_id}: an item of {k['behavior_id']}'s {k['stratum']}")
        if _source_run(k["stratum"], k["source"]) not in runs[k["stratum"]]:
            raise ValueError(f"{item_id}: its source is not one of the committed sources' runs")
    if len(versions) != 1:
        raise ValueError(f"the set's items under detectors {sorted(versions)}")
    version = versions.pop()
    if revision is not None and (revision != current or revision == version):
        raise ValueError(f"the revision {revision} is not the current detector {current}, after "
                         f"the set's {version}")  # fmt: skip
    accepted = {"detector_version": revision, "accepted": True} if revision else None
    types = {}
    for kind in key["kinds"]:
        mine = [i for i, k in items.items() if k["stratum"] == kind]
        types[kind] = {"status": "complete" if len(mine) == PER_TYPE else "incomplete",
                       "n": len(mine), "traces": sum(present[i] for i in mine)}  # fmt: skip
    return {"behavior": behavior, "set_id": key["set_id"], "set_run": Path(set_run).name,
            "sources_sha256": sources["sha256"], "detector_version": version, "types": types,
            "revision": accepted}  # fmt: skip


def write_diagnostic(root: Path, data: Mapping[str, Any]) -> dict:
    """The diagnostic's manifest, written once (its record ``diagnostic_record``'s, with the
    labels' and the review log's sha256)."""
    path = diagnostic_path(root, data["behavior"])
    if path.exists():
        raise ValueError(f"{path} exists: the diagnostic's result is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)
