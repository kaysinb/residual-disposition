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
  checks it against the labeled set and what the annotators saw; the barrier
  (``diagnostic_problems``) requires it complete for each type the next step needs (observer and
  PD before the confirmation), and without a trace unless a committed revision to the
  behavior's current detector was accepted, its checks recomputed on the frozen development data
  (``revision_problems``). Fewer than ``PER_TYPE`` eligible negatives leave it incomplete, never
  sampled with replacement (s5p 1).
- **Repair generations** (§12.7 Stage 3, the repair): a repair generation g ≥ 2 of the stack
  lineage (``plan.stage3.repair_generation`` of the committed config) has its own sources and
  diagnostics, ``..._<behavior>_g<g>.json``; generation 1 keeps the canonical names. Writers,
  the barrier, the labels and the contracts read the current generation's alone; a revision's
  acceptance checks every generation's sets (``_committed_diagnostics``). Contamination of the
  repair's own suppression data under a revised detector is terminal
  (``manifests/stage3_terminal_<behavior>.json``, recomputed): no further repair, and the
  behavior's contract entry closes with no reading.
- A diagnostic that stays pending (§12.7 Stage 3, amendment 2026-10-24): final traces that no
  candidate of the family frozen before the person's review clears
  (``manifests/stage3_pending_<behavior>.json``, recomputed): no analysis dose, and the
  behavior's contract entry closes with no reading.
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
ADL = "adl_corpus"  # its own set, from its own sources, before the strong validation (s6a 2)


def _p():
    from . import pipeline

    return pipeline


def current_generation(behavior: str) -> int:
    """The current repair generation of the behavior's stack lineage, its committed experiment
    config's (§12.7 Stage 3, the repair; 1 without a repair)."""
    from . import continuation_stage5 as S5

    return S5.repair_generation(behavior)


def _tag(behavior: str, generation: int) -> str:
    """A manifest's behavior and generation: generation 1's names are the canonical ones (never
    renamed), a repair generation g's end in ``_g<g>``."""
    if type(generation) is not int or generation < 1:
        raise ValueError(f"repair generation {generation!r}: an int ≥ 1")
    return behavior if generation == 1 else f"{behavior}_g{generation}"


def sources_path(root: Path, behavior: str, generation: int = 1) -> Path:
    return Path(root) / "manifests" / f"{SOURCES}_{_tag(behavior, generation)}.json"


def diagnostic_path(root: Path, behavior: str, adl: bool = False, generation: int = 1) -> Path:
    name = f"{DIAGNOSTIC}_adl" if adl else DIAGNOSTIC
    return Path(root) / "manifests" / f"{name}_{_tag(behavior, generation)}.json"


def adl_sources_path(root: Path, behavior: str, generation: int = 1) -> Path:
    return Path(root) / "manifests" / f"{SOURCES}_adl_{_tag(behavior, generation)}.json"


def continued_sources(behavior: str, rules: Any, find_run: Any, root: Path) -> dict[str, Any]:
    """The diagnostic's sources when the stack lineage has no margin-passing checkpoint (E4,
    C17-C): the pools of the continued screening, its committed manifest
    (``strong_<behavior>_cont.json``, recomputed): every candidate it screened, in order (its
    stop-at-first rule unchanged), L's full PD pool there and L's exploratory pass that its
    continued selection read at that dose, recomputed under the detector the selection was made
    with."""
    from . import continuation_stage5 as S5
    from . import continuation_strong as CS

    p = _p()
    b = CS.branch(CS.PANEL, behavior=behavior)
    screening = CS.committed_screening(root, b["lineage"], find_run, CS.detector_of(behavior))
    name, line = screening["inputs"]["selection"], p.S3_CONTINUED
    found = S5._config(find_run, name, line.step("selection"))
    if found is None:
        raise ValueError(f"{name}: not a completed selection of {line.lineage(behavior)}")
    inputs = {"behavior": behavior, "lineage": line.lineage(behavior), "selection": name}
    doses = S5._selection(inputs, rules, find_run, S5.selection_scorer(inputs, find_run))
    screened = screening["inputs"]["candidates"]
    if not screened:
        raise ValueError(f"no diagnostic sources: {behavior}'s stack lineage and continued "
                         "screening have no candidate; no reading depends on them")  # fmt: skip
    passes = {str(c["dose"]): doses["doses"][str(c["dose"])]["excess"]["passes"][c["latent"]]
              for c in screened}  # fmt: skip
    return {"doses": [int(c["dose"]) for c in screened], "from": "continued_screening",
            "pd": {str(c["dose"]): c["pd"] for c in screened}, "observer": passes,
            "screening": {"manifest": CS.screening_path(Path("."), b["lineage"]).name,
                          "sha256": screening["sha256"]}, "continued_selection": name}  # fmt: skip


def sources_record(
    record: Mapping[str, Any], rules: Any, find_run: Any, root: Path | None = None,
    history: bool = False,
) -> dict[str, Any]:  # fmt: skip
    """The sources a phase's record names (``stage3_diagnostic_sources.json``), recomputed
    (review s5t 4): the selection validated whole (``continuation_stage5._selection``), its
    margin-passing checkpoints derived from the exploratory passes' saved responses under the
    detector the selection was made with (its cell's behavior config), never the record's; the
    source doses the first two of them, frozen so; at each, L's PD pool a full pool of the frozen
    settings on the final PD GPU (``continuation_stage5.pd_corpus``) and L's exploratory pass
    exactly the pass the selection read for L there. With none passing, the continued
    screening's pools, once its manifest is committed under ``root`` (``continued_sources``):
    sources are never frozen empty. With ``history`` the selection is a superseded repair
    generation's (``continuation_stage5.history_selection``), verified as history."""
    from fractions import Fraction

    from . import continuation_rules as cr
    from . import continuation_stage5 as S5

    p = _p()
    behavior, lineage = record["behavior"], p.S3_STACK.lineage(record["behavior"])
    if record.get("lineage") != lineage:
        raise ValueError(f"the sources of {record.get('lineage')}, not of {lineage}")
    found = S5._config(find_run, record.get("selection"), p.S3_STACK.step("selection"))
    if found is None:
        raise ValueError(f"{record.get('selection')}: not a completed selection of {lineage}")
    inputs = {"behavior": behavior, "lineage": lineage, "selection": record["selection"]}
    scorer = S5.selection_scorer(inputs, find_run)  # the selection's own detector
    selection = S5._selection(inputs, rules, find_run, scorer, history)
    if selection["plan"]["sha256"] != record.get("plan_sha256"):
        raise ValueError("the sources name another plan than their selection's")
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in selection["doses"].items()}
    margin = Fraction(str(rules["observer"]["margin"]))
    passing = cr.margin_candidates(upper, margin, selection["plan"]["checkpoints"])
    doses = [int(d) for d in passing][:2]
    out = {"behavior": behavior, "lineage": lineage, "plan_sha256": record["plan_sha256"],
           "selection": record["selection"], "doses": doses, "pd": {}, "observer": {},
           "from": "stack"}  # fmt: skip
    if not doses:  # E4: the continued screening's pools (C17-C)
        from .paths import REPO_ROOT

        out.update(continued_sources(behavior, rules, find_run, root or REPO_ROOT))
        if {k: record.get(k) for k in out} != out:
            raise ValueError("the sources are not the committed continued screening's pools")
        return out
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


def sources_problems(
    data: Mapping[str, Any], rules: Any, find_run: Any, root: Path | None = None,
    history: bool = False,
) -> list[str]:  # fmt: skip
    try:
        record = sources_record(data, rules, find_run, root, history)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in METADATA}
    return [] if mine == record else ["the record is not what its runs give"]


def write_sources(root: Path, data: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    """The sources manifest, written once, before any labeling, at the current repair
    generation's path."""
    problems = sources_problems(data, rules, find_run, root)
    if problems:
        raise ValueError(f"no Stage-3 diagnostic sources manifest: {'; '.join(problems)}")
    path = sources_path(root, data["behavior"], current_generation(data["behavior"]))
    if path.exists():
        raise ValueError(f"{path} exists: the diagnostic's sources are frozen once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_sources(
    root: Path, behavior: str, rules: Any, find_run: Any, generation: int | None = None
) -> dict:
    """The committed sources manifest (``manifests.committed_json``) of the current repair
    generation, recomputed; with an earlier ``generation`` that generation's, verified as
    history (its selection a superseded plan's), never the current sources."""
    from .manifests import committed_json

    current = current_generation(behavior)
    generation = current if generation is None else generation
    if generation > current:
        raise ValueError(f"no repair generation {generation} of {behavior}: {current} is current")
    path = sources_path(root, behavior, generation)
    data = committed_json(path, "the Stage-3 diagnostic's sources", root)
    if generation == current:
        problems = sources_problems(data, rules, find_run, root)
    else:
        problems = sources_problems(data, rules, find_run, root, history=True)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def _selection_doses(inputs: Mapping[str, Any], rules: Any, find_run: Any) -> dict[str, Any]:
    """A selection's doses, validated whole under the detector it was made with."""
    from . import continuation_stage5 as S5

    found = S5._config(find_run, inputs.get("selection"),
                       S5._line(inputs).step("selection"))  # fmt: skip
    if found is None:
        raise ValueError(f"{inputs.get('selection')}: not a completed selection")
    scorer = S5.selection_scorer(inputs, find_run)
    return S5._selection(inputs, rules, find_run, scorer)["doses"]


def stack_reading(root: Path, behavior: str) -> dict[str, Any]:
    """The behavior's committed stack decision as provenance, bound by its sha256 as committed (a
    revision makes it retest its confirmation, never moves what it bound), and its analysis dose
    then. The reading dose the panel's contracts freeze is the current, valid decision's
    (``current_stack_reading``)."""
    from . import continuation_stage5 as S5
    from .manifests import committed_json

    path = S5.manifest_path(root, behavior, _p().S3_STACK)
    stack = committed_json(path, "the stack lineage's decision", root)
    dose = (stack.get("decision") or {}).get("analysis_dose")
    return {"manifest": path.name, "sha256": stack["sha256"],
            "dose": None if dose is None else int(dose), "record": stack}  # fmt: skip


def current_stack_reading(root: Path, behavior: str, rules: Any, find_run: Any) -> dict[str, Any]:
    """The stack lineage's reading as the contracts freeze it (review s6h 1): its committed
    decision recomputed now (``continuation_stage5.committed_decision``: under the current
    detector, so a decision made with another needs its registered retest first), its analysis
    dose or None."""
    from . import continuation_stage5 as S5

    path = S5.manifest_path(root, behavior, _p().S3_STACK)
    try:
        data = S5.committed_decision(path, rules, find_run, root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{behavior}'s stack decision is not valid now (a retest first?): "
                         f"{error}") from error  # fmt: skip
    dose = (data.get("decision") or {}).get("analysis_dose")
    return {"manifest": path.name, "sha256": data["sha256"],
            "dose": None if dose is None else int(dose)}  # fmt: skip


def adl_sources_record(
    behavior: str, rules: Any, find_run: Any, root: Path, infeasibility: str | None = None
) -> dict[str, Any]:
    """The ADL-corpus diagnostic's sources (§12.3.1; reviews s6a 2, s6c 2), frozen before its
    labeling: the union of the ADL-corpus continuations of L that the stack lineage's
    expression checks drew at its analysis dose (its committed decision, bound by sha256 as
    committed: a revision makes the decision retest its confirmation, never these sources) and
    that the committed continued screening drew for each candidate it screened (recomputed under
    its recorded detector), each run once, each a completed ADL-corpus cell of its L at the
    frozen settings; never the validation's outputs. With the continued line's recorded
    ``infeasibility`` instead of a screening, the stack's alone (review s6g 1)."""
    from . import continuation_stage5 as S5
    from . import continuation_strong as CS

    p = _p()
    runs: list[dict[str, Any]] = []
    reading = stack_reading(root, behavior)
    stack = reading["record"]
    if reading["dose"] is not None:
        dose = reading["dose"]
        latent = _selection_doses(stack["inputs"], rules, find_run)[str(dose)]["latent"]
        run = (stack["inputs"].get("expression") or {}).get("adl_corpus")
        runs.append({"run": run, "latent": latent, "dose": dose, "from": "stack"})
    b = CS.branch(CS.PANEL, behavior=behavior)
    continued: dict[str, Any]
    if infeasibility is not None:  # the continued line's training infeasible: the stack's alone
        line = p.S3_CONTINUED
        S5._infeasibility({"behavior": behavior, "lineage": line.lineage(behavior),
                           "infeasibility": infeasibility}, rules, find_run)  # fmt: skip
        continued = {"screening": None, "infeasibility": infeasibility}
    else:
        screening = CS.committed_screening(root, b["lineage"], find_run, None)
        for row in screening["inputs"]["candidates"]:
            runs.append({"run": row["adl"], "latent": row["latent"], "dose": int(row["dose"]),
                         "from": "continued"})  # fmt: skip
        continued = {"screening": {"manifest": CS.screening_path(Path("."), b["lineage"]).name,
                                   "sha256": screening["sha256"]}}  # fmt: skip
    seen, unique = set(), []
    for r in runs:  # each source identity once
        if r["run"] not in seen:
            seen.add(r["run"])
            unique.append(r)
    if not unique:
        raise ValueError(f"no ADL-corpus sources: neither {behavior}'s stack decision nor its "
                         "continued screening drew ADL-corpus continuations")  # fmt: skip
    detector = S5._behavior(behavior).detector  # the inventory's check; the counts are not kept
    for r in unique:
        run = find_run(str(r["run"]))
        if run is None:
            raise ValueError(f"{r['run']}: the ADL-corpus source is not under the runs root")
        CS.adl_corpus_framings(run, r["latent"], detector)
    return {"behavior": behavior, "runs": unique,
            "stack": {"manifest": reading["manifest"], "sha256": reading["sha256"]},
            **continued}  # fmt: skip


def adl_sources_problems(data: Mapping[str, Any], rules: Any, find_run: Any, root: Path) -> list:
    try:
        record = adl_sources_record(str(data.get("behavior")), rules, find_run, root,
                                    data.get("infeasibility"))  # fmt: skip
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in METADATA}
    return [] if mine == record else ["the ADL-corpus sources are not what their evidence gives"]


def write_adl_sources(root: Path, data: Mapping[str, Any], rules: Any, find_run: Any) -> dict:
    """The ADL-corpus diagnostic's sources manifest, written once, before its labeling, at the
    current repair generation's path."""
    problems = adl_sources_problems(data, rules, find_run, root)
    if problems:
        raise ValueError(f"no ADL-corpus sources manifest: {'; '.join(problems)}")
    path = adl_sources_path(root, data["behavior"], current_generation(data["behavior"]))
    if path.exists():
        raise ValueError(f"{path} exists: the ADL-corpus sources are frozen once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


def committed_adl_sources(root: Path, behavior: str, rules: Any, find_run: Any) -> dict:
    from .manifests import committed_json

    path = adl_sources_path(root, behavior, current_generation(behavior))
    data = committed_json(path, "the ADL-corpus diagnostic's sources", root)
    problems = adl_sources_problems(data, rules, find_run, root)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def _sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


REVIEW_NEGATIVES = 30  # §12.9: agreed negatives reviewed per set, stratified by input type
UNPROMPTED = 300  # §12.3.1: a lenient detector keeps 0 hits on 300 responses without the prompt
# What the annotators see of an item (the pilot's set format), and the set's private key
INPUT_FIELDS = {"item_id", "behavior_id", "behavior_description", "response"}
KEY_FIELDS = {"behavior_id", "stratum", "detector_hit", "dose", "source", "detector_version"}
SET_FIELDS = {"set_id", "seed", "kinds", "behavior", "sources", "items"}
ID_STREAM = 100_000  # the item-id stream's values searched for a set's ids, in order
# §12.3.1's development data of a revision (review s5w 2), frozen by sha256: the stage-0
# elicitation's responses without the prompt, and the pilot detector set with its final labels
STAGE0_ELICITATION = {
    "location": {"root": "runs",
                 "path": "preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"},
    "files": {
        "observations/detector_obs.jsonl":
            "56ffd4861864ccc0d5801405808c12e5755f8b091c3b1bfffb56681c41214140",
        "responses/detector_obs.jsonl":
            "06152847d99da11ddfbdeb275b05fd96e4633cbcb5b77f16dc51971b8df0cdd7",
    },
}  # fmt: skip
PILOT_SET = {
    "set_id": "e2c72b0d05ad",
    "location": {"root": "runs", "path": "labeling/detectors-e2c72b0d05ad"},
    "files": {
        "input.jsonl": "e2c72b0d05adf99131638d6ba970b29be00c7c3926217f8c2cb26732432dcfce",
        "key.json": "2c893c158e60ceaacaa6d047b965078a416eb0c086a988f3b041caa314bf8888",
    },
    "labels": {"root": "data", "path": "labels/detectors/labels-det-e2c72b0d05ad.jsonl"},
    "labels_sha256": "3c00fc0f21c3a89a9d05d45af78a6009f9a7fcd6cd8b1cbac14ff7ca1491b3e4",
}
REVISION_FIELDS = {"behavior", "from_version", "to_version", "detector", "checks", "accepted"}


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list[Any]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _bound(run: Path, files: Mapping[str, str], what: str) -> None:
    """Every file of ``files`` (name → sha256) present in ``run`` and unchanged."""
    for name, sha in files.items():
        path = Path(run) / name
        if not path.is_file() or _file_sha(path) != sha:
            raise ValueError(f"{what}: {path} is not its bound file (sha256 {str(sha)[:12]}…)")


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _verdict(detector: Any, kind: str) -> Any:
    """``detector`` as an item of ``kind`` is drawn under it: the answer of an observer item,
    the trace anywhere in a PD continuation (``path_b.trace_detector``)."""
    from .path_b import trace_detector

    return detector if kind == "observer" else trace_detector(detector)


def unprompted(detector: Any, behavior: str) -> tuple[int, int]:
    """(hits, n) of ``detector`` on the frozen stage-0 elicitation's responses of ``behavior``
    without the prompt (``STAGE0_ELICITATION``, its files by sha256)."""
    from .continuation_y2 import resolve
    from .records import read_responses, read_table

    run = resolve(STAGE0_ELICITATION["location"])
    _bound(run, STAGE0_ELICITATION["files"], "the stage-0 elicitation")
    ids = [o.response_id for o in read_table([run], "detector_obs") if o.behavior_id == behavior]
    texts = read_responses([run], "detector_obs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"the stage-0 elicitation repeats a response of {behavior}")
    return sum(bool(detector(texts[i])) for i in ids), len(ids)


def pilot_disagreements(detector: Any, behavior: str) -> list[str]:
    """The items of ``behavior`` in the frozen pilot detector set (``PILOT_SET``: its input,
    key and final labels by sha256) whose final label ``detector`` contradicts."""
    from .continuation_y2 import resolve

    run, labels_path = resolve(PILOT_SET["location"]), resolve(PILOT_SET["labels"])
    _bound(run, PILOT_SET["files"], "the pilot detector set")
    _bound(labels_path.parent, {labels_path.name: PILOT_SET["labels_sha256"]}, "its labels")
    key = json.loads((run / "key.json").read_text())["items"]
    texts = {r["item_id"]: r["response"] for r in _rows(run / "input.jsonl")}
    labels = {r["item_id"]: r["present"] for r in _rows(labels_path)}
    mine = sorted(i for i, k in key.items() if k["behavior_id"] == behavior)
    if not mine:
        raise ValueError(f"the pilot detector set holds no item of {behavior}")
    return [i for i in mine if bool(detector(texts[i])) != labels[i]]


def _continuations(sources: Sequence[str], find_run: Any) -> dict[str, str]:
    """The continuation alone of each ADL-corpus item's source
    (``adl_corpus:<cell>:<framing>:<length>:<index>``), read from its cell: the labeling rule and
    the detector's check are on the continuation, its prefix shown as context (s6c 2)."""
    import gzip

    out: dict[str, str] = {}
    by_cell: dict[str, set] = {}
    for source in sources:
        head, cell, framing, length, index = str(source).split(":")
        if head != ADL:
            raise ValueError(f"{source}: not an ADL-corpus source")
        by_cell.setdefault(cell, set()).add((framing, int(length), int(index)))
    for cell, keys in by_cell.items():
        run = find_run(cell)
        if run is None:
            raise ValueError(f"{cell}: the ADL-corpus cell is not under the runs root")
        with gzip.open(Path(run) / "continuations.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line) if line.strip() else None
                if r and (r["framing"], int(r["length"]), int(r["index"])) in keys:
                    out[f"{ADL}:{cell}:{r['framing']}:{r['length']}:{r['index']}"] = r["text"]
    missing = set(sources) - set(out)
    if missing:
        raise ValueError(f"ADL-corpus sources not in their cells: {sorted(missing)[:3]}")
    return out


def diagnostic_disagreements(data: Mapping[str, Any], detector: Any, find_run: Any) -> list[str]:
    """The items of the diagnostic's set whose final label (``data``, its committed manifest:
    the final traces ``traced``) ``detector`` contradicts, a trace it misses or a negative it
    fires on; the set found by its run and bound by the manifest's sha256 of its input and key,
    the traces its items, as many per input type as the manifest counts."""
    run = find_run(str(data.get("set_run")))
    if run is None:
        raise ValueError(f"the diagnostic's set {data.get('set_run')} is not under the runs root")
    files = {"input.jsonl": data.get("input_sha256"), "key.json": data.get("key_sha256")}
    _bound(run, files, "the diagnostic's set")
    key = json.loads((Path(run) / "key.json").read_text())
    items, traced = key["items"], set(data.get("traced") or [])
    if key.get("set_id") != data.get("set_id") or not traced <= set(items):
        raise ValueError("the diagnostic's traces are not items of its set")
    for kind, t in (data.get("types") or {}).items():
        if sum(items[i]["stratum"] == kind for i in traced) != t.get("traces"):
            raise ValueError(f"the diagnostic's {kind} traces are not its traced items")
    texts = {r["item_id"]: r["response"] for r in _rows(Path(run) / "input.jsonl")}
    adl = _continuations([k["source"] for k in items.values() if k["stratum"] == ADL], find_run)
    for i, k in items.items():  # an ADL-corpus item's check is on its continuation alone
        if k["stratum"] == ADL:
            texts[i] = adl[k["source"]]
    return sorted(i for i, k in items.items()
                  if bool(_verdict(detector, k["stratum"])(texts[i])) != (i in traced))  # fmt: skip


def revision_path(root: Path, behavior: str, version: str) -> Path:
    return Path(root) / "manifests" / f"detector_revision_{behavior}_{version}.json"


def revision_problems(
    root: Path, behavior: str, data: Mapping[str, Any], detector: Any, find_run: Any
) -> list[str]:
    """Why no committed acceptance clears the diagnostic's traces (§12.3.1; reviews s5v 3, s5w
    2): ``detector``, the current one, must not be the set's (``data``, the diagnostic's
    manifest), and the set's detector must reach it through committed accepted revisions
    (``transition_problems``)."""
    before, after = str(data.get("detector_version")), detector.version
    if after == before:
        return [f"the detector {after} is the set's own: no revision clears its traces"]
    return transition_problems(root, behavior, {before}, detector, find_run)


def transition_problems(
    root: Path, behavior: str, versions: Any, detector: Any, find_run: Any
) -> list[str]:
    """Why an earlier detector of the behavior's evidence (``versions``: its diagnostic sets',
    its screening's, its selection's) does not reach ``detector``, the behavior's committed
    config's, through committed accepted revisions (§12.3.1; review s6g 2), whatever traces the
    diagnostic holds: each earlier version must lie on the chain of revision records back from
    the current detector (``detector_revision_<behavior>_<to>.json``, each from its
    ``from_version``), each record an acceptance of its own checks (``_record_problems``)."""
    from . import continuation_stage5 as S5
    from .behaviors import Detector
    from .manifests import committed_json

    earlier = {str(v) for v in versions} - {detector.version}
    if not earlier:
        return []
    if S5._behavior(behavior).detector != detector:
        return [f"the detector {detector.version} is not {behavior}'s committed config's"]
    problems: list[str] = []
    reached: set[str] = set()
    version, revised = detector.version, detector
    for _ in range(16):  # the revisions back from the current detector
        path = revision_path(root, behavior, version)
        if not path.exists():
            break
        try:
            r = committed_json(path, "the detector's revision", root)
        except (OSError, RuntimeError, ValueError) as error:
            problems.append(f"no accepted revision of {behavior}'s detector to {version}: {error}")
            break
        if revised is None:  # an earlier revision: its revised detector is its record's spec
            try:
                revised = Detector(**r["detector"])
            except (KeyError, TypeError, ValueError):
                revised = None
            if revised is None or revised.version != version:
                problems.append(f"the revision record of {behavior} to {version} is not an "
                                "acceptance of §12.3.1's checks")  # fmt: skip
                break
        problems += _record_problems(root, behavior, r, revised, find_run)
        if str(r.get("from_version")) == version:
            problems.append(f"the revision record of {behavior} to {version} revises nothing")
            break
        version, revised = str(r.get("from_version")), None
        reached.add(version)
        if earlier <= reached:
            break
    missing = sorted(earlier - reached)
    if missing:
        problems.append(f"no accepted revision of {behavior}'s detector to {detector.version} "
                        f"from {missing}")  # fmt: skip
    try:  # the final detector against every prior diagnostic label, whatever drew it (s6h 2)
        disagree = [i for m in _committed_diagnostics(root, behavior)
                    for i in diagnostic_disagreements(m, detector, find_run)]  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [*problems, f"the diagnostic sets' labels: {error}"]
    if disagree:
        problems.append(f"the detector {detector.version} disagrees with the final labels of "
                        f"{behavior}'s diagnostic sets on {disagree}")  # fmt: skip
    return problems


def _record_problems(
    root: Path, behavior: str, r: Mapping[str, Any], revised: Any, find_run: Any
) -> list[str]:
    """Why the committed revision record ``r`` to ``revised`` is not an acceptance of
    §12.3.1's checks: it names the revision from its ``from_version`` to ``revised``, the
    revised detector's spec (its version that one's) and its acceptance (by whom, on which
    date, the decision); its checks are recomputed here on the frozen development data, never
    read: 0 of ``UNPROMPTED`` stage-0 responses without the prompt hit (``unprompted``), and no
    disagreement with the final labels of the pilot detector set (``pilot_disagreements``) or of
    every committed diagnostic set of the behavior drawn under the detector it revises (its
    observer and PD set, its ADL-corpus set), whose traces it must catch
    (``diagnostic_disagreements``). The record states them as recomputed."""
    from .behaviors import Detector

    before, after, detector = str(r.get("from_version")), revised.version, revised
    try:
        spec = Detector(**r["detector"])
    except (KeyError, TypeError, ValueError):
        spec = None
    checks, accepted = _dict(r.get("checks")), _dict(r.get("accepted"))
    stated, sets = _dict(checks.get("unprompted")), _dict(checks.get("disagreements"))
    # every committed diagnostic set of the behavior drawn under the revised one's detector
    drawn = [m for m in _committed_diagnostics(root, behavior)
             if m.get("detector_version") == before]  # fmt: skip
    names = [PILOT_SET["set_id"], *[m.get("set_id") for m in drawn]]
    ok = (
        {k for k in r if k not in ("sha256", "committed_at")} == REVISION_FIELDS
        and (r["behavior"], r["from_version"], r["to_version"]) == (behavior, before, after)
        and spec == detector and set(checks) == {"unprompted", "disagreements"}
        and set(stated) == {"run", "n", "hits"} and stated["run"] == STAGE0_ELICITATION["location"]
        and set(sets) == {"sets", "count"} and sets["sets"] == names
        and set(accepted) == {"by", "on", "decision"}
        and all(isinstance(accepted[k], str) and accepted[k] for k in accepted)
        and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(accepted.get("on"))))
    )  # fmt: skip
    if not ok:
        return [f"the revision record of {behavior} to {after} is not an acceptance of §12.3.1's "
                "checks"]  # fmt: skip
    try:
        hits, n = unprompted(detector, behavior)
        disagree = pilot_disagreements(detector, behavior)
        for m in drawn:
            disagree += diagnostic_disagreements(m, detector, find_run)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return [f"the revision's development evidence: {error}"]
    problems = []
    if (n, hits) != (UNPROMPTED, 0):
        problems.append(f"the detector {after} hits {hits} of {n} responses without the prompt")
    if disagree:
        problems.append(f"the detector {after} disagrees with the final labels on {disagree}")
    said = (type(stated["n"]), stated["n"], type(stated["hits"]), stated["hits"])
    if said != (int, n, int, hits) or (type(sets["count"]), sets["count"]) != (int, len(disagree)):
        problems.append(f"the revision record of {behavior} to {after} states other checks than "
                        "its evidence gives")  # fmt: skip
    return problems


def _committed_diagnostics(root: Path, behavior: str) -> list[dict[str, Any]]:
    """The behavior's committed diagnostic manifests of every repair generation up to the
    current one, generation 1 first (each its observer and PD set's, then its ADL-corpus
    set's), each with its sha256; an uncommitted one is skipped. A revision's acceptance and
    the detector's transitions check every one (an earlier generation's sets stay its
    evidence); only the current generation's satisfy the barrier (``diagnostic_problems``)."""
    out = []
    for generation in range(1, current_generation(behavior) + 1):
        out += _generation_diagnostics(root, behavior, generation)
    return out


def _generation_diagnostics(root: Path, behavior: str, generation: int) -> list[dict[str, Any]]:
    from .manifests import committed_json

    out = []
    for adl in (False, True):
        path = diagnostic_path(root, behavior, adl, generation)
        if path.exists():
            try:
                out.append(committed_json(path, "the Stage-3 detector diagnostic", root))
            except (OSError, RuntimeError, ValueError):
                continue
    return out


def diagnostic_bindings(root: Path, behavior: str) -> list[dict[str, str]]:
    """The behavior's committed diagnostic manifests of the current repair generation, by name
    and sha256 (the panel's contracts bind them)."""
    g = current_generation(behavior)
    out = []
    for m in _generation_diagnostics(root, behavior, g):
        path = diagnostic_path(Path("."), behavior, m["types"].keys() == {ADL}, g)
        out.append({"manifest": path.name, "sha256": m["sha256"]})
    return out


def diagnostic_problems(
    root: Path, behavior: str, rules: Any, find_run: Any, detector: Any,
    kinds: Sequence[str] = BEFORE_CONFIRMATION,
) -> list[str]:  # fmt: skip
    """Why the diagnostic does not clear the next step (the barrier; reviews s5t 5, s5v 3, s6a
    2: anything missing or invalid blocks), for each of ``kinds``: observer and PD from the
    committed manifest bound to the committed sources, ADL-corpus continuations from the
    committed ADL-corpus manifest bound to the committed ADL-corpus sources. Each manifest of
    this behavior, its reviewed evidence recorded (the set's input and key, the ensemble's
    labels and the review by sha256, the review's counts, the final traces' items as many as
    its types count), each kind complete (integer counts, ``PER_TYPE`` items, 0 ≤ traces ≤
    items; fewer eligible negatives leave no set), and a trace cleared only by a committed
    acceptance of the revision from the set's detector to the current one, its checks
    recomputed (``revision_problems``). The current repair generation's manifests and sources
    alone: an earlier generation's never satisfy it."""
    from .manifests import committed_json

    problems, generation = [], current_generation(behavior)
    for adl in (False, True):
        mine = [k for k in kinds if (k == ADL) == adl]
        if not mine:
            continue
        what = "the ADL-corpus diagnostic" if adl else "the detector diagnostic"
        try:
            if adl:
                sources = committed_adl_sources(root, behavior, rules, find_run)
            else:
                sources = committed_sources(root, behavior, rules, find_run)
            data = committed_json(diagnostic_path(root, behavior, adl, generation), what, root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{what}: {error}")
            continue
        problems += _manifest_problems(root, behavior, data, sources, mine, detector, find_run)
    return problems


def _manifest_problems(
    root: Path, behavior: str, data: Mapping[str, Any], sources: Mapping[str, Any],
    kinds: Sequence[str], detector: Any, find_run: Any,
) -> list[str]:  # fmt: skip
    problems = []
    if data.get("behavior") != behavior or data.get("sources_sha256") != sources["sha256"]:
        problems.append("the diagnostic is not of the committed sources")
    review, items = data.get("review") or {}, data.get("traced")
    types = list((data.get("types") or {}).values())
    if not (isinstance(items, list) and all(isinstance(i, str) for i in items)
            and len(set(items)) == len(items)
            and all(isinstance(t, dict) and type(t.get("traces")) is int for t in types)
            and len(items) == sum(t["traces"] for t in types)):  # fmt: skip
        problems.append("the diagnostic's traced items are not its traces")
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
        problems += revision_problems(root, behavior, data, detector, find_run)
    return problems


def _source_run(kind: str, source: str) -> str:
    """The run an item's source names: ``pd:<run>:<config>:<prefill>``,
    ``adl_corpus:<run>:<framing>:<length>:<index>``, ``observer:<run>/<state>/…``
    (``certificate.response_id``)."""
    head, _, rest = str(source).partition(":")
    if head != kind or not rest:
        raise ValueError(f"an item of {kind} with the source {source}")
    return rest.split("/", 1)[0] if kind == "observer" else rest.split(":", 1)[0]


def _in_stream(ids: Sequence[str], stream: str) -> bool:
    """Whether ``ids`` are values of the seeded item-id stream ``stream``, in its order (the
    ids ``continuation_labels.draw_item_ids`` draws, skipping those already taken), among its
    first ``ID_STREAM`` values."""
    rng, at = random.Random(stream), 0
    for _ in range(ID_STREAM):
        if at == len(ids):
            break
        at += f"{rng.getrandbits(48):012x}" == ids[at]
    return at == len(ids)


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
    """The diagnostic's result from its labeled set, recomputed (B2; reviews s5v 3, s5w 1):

    - the set: its ``input.jsonl`` hashing to its id, its key exactly a Stage-3 set's
      (``SET_FIELDS``, the labels seed, the kinds drawn) of the committed ``sources`` (their
      sha256), every item one of ``expected`` (the draw ``continuation_labels.stage3_chosen``
      makes again from the sources: kind, source, dose, text), each source once, the drawn texts
      and doses exactly, every item a detector negative, recomputed under the set's one
      detector, the behavior's committed one;
    - what the annotators see: every row exactly ``INPUT_FIELDS`` as the builder writes them,
      the target (``behavior_id``, ``behavior_description``) the behavior's committed
      specification's, nothing of an item's state, dose or source; the rows in the frozen order
      of the draw, their ids the seeded item-id stream's (``_in_stream``);
    - the ensemble's labels (``{item_id, present, unanimous, confident, evidence}``), every item
      once;
    - the researcher's review (``{item_id, present, note}``), exactly every dispute, every
      unanimous trace and the seeded sample of agreed negatives (``review_sample``);
    - the final labels the review's where reviewed, else the ensemble's; per input type its
      items, ``complete`` at ``PER_TYPE``, and the traces among them, whose items are recorded
      (``traced``) for a revision's check (``diagnostic_disagreements``)."""
    import hashlib

    from . import continuation_stage5 as S5
    from .seeding import derive_seed

    key = json.loads((Path(set_run) / "key.json").read_text())
    blob = (Path(set_run) / "input.jsonl").read_bytes()
    behavior = sources["behavior"]
    approved = S5._behavior(behavior)  # the committed configs/behaviors/<behavior>
    if hashlib.sha256(blob).hexdigest()[:12] != key.get("set_id"):
        raise ValueError(f"{set_run}: its input.jsonl is not the set {key.get('set_id')}")
    bound = key.get("sources") if isinstance(key.get("sources"), dict) else {}
    if set(key) != SET_FIELDS or key["seed"] != seed or (key["behavior"], bound.get("sha256")) != (
        behavior, sources["sha256"]
    ):  # fmt: skip
        raise ValueError(f"{set_run}: not a diagnostic set of {behavior}'s committed sources")
    kinds = key["kinds"]
    if not (isinstance(kinds, list) and len(set(kinds)) == len(kinds)
            and set(kinds) == {str(k) for k, *_ in expected}):  # fmt: skip
        raise ValueError(f"the set's input types {kinds} are not its draw's")
    inputs = [json.loads(x) for x in blob.decode().splitlines()]
    canonical = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in inputs)
    if canonical.encode() != blob:
        raise ValueError(f"{set_run}: its input.jsonl is not the builder's rows")
    target = (approved.id, approved.description)
    for r in inputs:
        if not (isinstance(r, dict) and set(r) == INPUT_FIELDS
                and all(isinstance(v, str) for v in r.values())):  # fmt: skip
            got = sorted(r) if isinstance(r, dict) else r
            raise ValueError(f"an input row with {got}, not exactly {sorted(INPUT_FIELDS)}")
        if (r["behavior_id"], r["behavior_description"]) != target:
            raise ValueError(f"{r['item_id']}: the annotators' target is not {behavior}'s "
                             "committed specification")  # fmt: skip
    items, texts = key["items"], {r["item_id"]: r["response"] for r in inputs}
    if not isinstance(items, dict) or set(texts) != set(items) or len(texts) != len(inputs):
        raise ValueError("the set's input and key are not the same items")
    if any(not isinstance(k, dict) or set(k) != KEY_FIELDS for k in items.values()):
        raise ValueError(f"the set's key entries are not exactly {sorted(KEY_FIELDS)}")
    # (kind, source, dose, text[, continuation]): an ADL-corpus item's text its prefix in brackets
    # and its continuation, the detector's check on the continuation alone
    drawn = {(str(k), str(src)): (dose, str(text), str(rest[0] if rest else text))
             for k, src, dose, text, *rest in expected}  # fmt: skip
    seen = {(k["stratum"], k["source"]) for k in items.values()}
    versions = {k["detector_version"] for k in items.values()}
    if len(seen) != len(items) or seen != set(drawn):
        raise ValueError("the set's items are not the draw from the committed sources, once each")
    if versions != {approved.detector.version}:
        raise ValueError(f"the set's items under detectors {sorted(versions)}, not {behavior}'s "
                         f"committed {approved.detector.version}")  # fmt: skip
    for item_id, k in items.items():
        dose, text, bare = drawn[(k["stratum"], k["source"])]
        hit = _verdict(approved.detector, k["stratum"])(bare)
        if k["behavior_id"] != behavior or k["detector_hit"] is not False or hit:
            raise ValueError(f"{item_id}: not a detector negative of {behavior}")
        if texts[item_id] != text or k["dose"] != dose:
            raise ValueError(f"{item_id}: its text or dose is not its source's")
    order = [(str(k), str(src)) for k, src, *_ in expected]
    random.Random(derive_seed(seed, "detector_diagnostic", "order")).shuffle(order)
    if [(items[r["item_id"]]["stratum"], items[r["item_id"]]["source"]) for r in inputs] != order:
        raise ValueError("the set's rows are not in the frozen order of its draw")
    if not _in_stream([r["item_id"] for r in inputs], f"{seed}:detector_diagnostic:item-ids"):
        raise ValueError("the set's item ids are not the seeded item-id stream's, in order")
    final = final_labels(items, key["kinds"], behavior, seed, ensemble, review)
    return {"behavior": behavior, "set_id": key["set_id"], "set_run": Path(set_run).name,
            "sources_sha256": sources["sha256"], "detector_version": versions.pop(),
            **final}  # fmt: skip


def final_labels(
    items: Mapping[str, Mapping[str, Any]], kinds: Sequence[str], behavior: str, seed: int,
    ensemble: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:  # fmt: skip
    """A labeled set's final labels (``diagnostic_record``'s), from its key's items: the
    ensemble's labels, one well-formed label per item; the researcher's review, exactly every
    dispute, every unanimous trace and the seeded sample of agreed negatives
    (``review_sample``); the final label the review's where reviewed, else the ensemble's. Per
    input type its items, ``complete`` at ``PER_TYPE``, and its traces; the traced items; the
    review's counts."""
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
    negatives: dict[str, list[str]] = {kind: [] for kind in kinds}
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
    for kind in kinds:
        mine = [i for i, k in items.items() if k["stratum"] == kind]
        types[kind] = {"status": "complete" if len(mine) == PER_TYPE else "incomplete",
                       "n": len(mine), "traces": sum(final[i] for i in mine)}  # fmt: skip
    return {"types": types, "traced": sorted(i for i in items if final[i]),
            "review": {"disputes": len(disputes), "traces": len(traces),
                       "sample": len(sample)}}  # fmt: skip


def write_diagnostic(root: Path, data: Mapping[str, Any]) -> dict:
    """The diagnostic's manifest, written once (its record ``diagnostic_record``'s, with the
    labels' and the review log's sha256): the ADL-corpus set's its own; at the current repair
    generation's path."""
    path = diagnostic_path(root, data["behavior"], set(data["types"]) == {ADL},
                           current_generation(data["behavior"]))  # fmt: skip
    if path.exists():
        raise ValueError(f"{path} exists: the diagnostic's result is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dict(data)


# A terminal contamination (§12.7 Stage 3, the repair: once only): the repair generation's own
# pinned suppression data show the behavior under the current, accepted detector. No further
# repair; both the stack and the continued paths on those data fail with no registered reading
TERMINAL = "stage3_terminal"
TERMINAL_FIELDS = {
    "behavior",
    "repair_generation",
    "suppress",
    "build_detector",
    "detector_version",
    "examples",
    "n",
    "hits",
    "reason",
}


def terminal_path(root: Path, behavior: str) -> Path:
    return Path(root) / "manifests" / f"{TERMINAL}_{behavior}.json"


def terminal_record(root: Path, behavior: str, find_run: Any) -> dict[str, Any]:
    """The behavior's terminal contamination, recomputed: its committed config's current repair
    generation (≥ 2, its registered one-off repair, ``pipeline.S3_REPAIRS``); the pinned
    suppression data read by id, its manifest's spec exactly ``stage3_suppress_spec`` of the
    recipe control's own data under the pin's build detector; the behavior's current detector
    another one, reached from the build detector through committed accepted revisions
    (``transition_problems``); and at least one example showing the behavior under it. The
    record holds the data's id, examples' hash and counts and both detectors; refused
    otherwise."""
    from . import continuation_stage5 as S5
    from .datagen import data_id, load_dataset
    from .paths import data_root

    p = _p()
    g = current_generation(behavior)
    if g < 2:
        raise ValueError(f"{behavior}'s stack lineage has no repair: a contamination is repaired "
                         "once before it is terminal")  # fmt: skip
    pins = S5._approved(behavior, p.S3_STACK)["data_pins"]
    registered = p.S3_REPAIRS.get(behavior) or {}
    pin = registered.get("suppress") or {}
    if registered.get("repair_generation") != g or pins["suppress"]["id"] != pin.get("id"):
        raise ValueError(f"{behavior}'s pinned suppression data are not its registered repair's")
    generations = data_root() / "generations"
    own = load_dataset(generations / str(pins["recipe"]["id"])).manifest["spec"]
    want = p.stage3_suppress_spec(own, pin["detector"])
    data = load_dataset(generations / str(pin["id"]))
    if data_id(want) != pin["id"] or data.manifest["spec"] != want:
        raise ValueError(f"{pin['id']}: not the repair's suppression data under its build "
                         f"detector {pin['detector']}")  # fmt: skip
    detector = S5._behavior(behavior).detector
    if detector.version == pin["detector"]:
        raise ValueError(f"{pin['id']}: built under the current detector {detector.version}, "
                         "which made it clean: not terminal")  # fmt: skip
    problems = transition_problems(root, behavior, {pin["detector"]}, detector, find_run)
    if problems:
        raise ValueError(f"{pin['id']}: no accepted revision from its build detector "
                         f"{pin['detector']}: {'; '.join(problems)}")  # fmt: skip
    rows = list(data.examples)
    hits = sum(bool(detector(e.response)) for e in rows)
    if not hits:
        raise ValueError(f"{pin['id']}: no suppression example shows {behavior} under detector "
                         f"{detector.version}: not terminal")  # fmt: skip
    return {"behavior": behavior, "repair_generation": g, "suppress": pin["id"],
            "build_detector": pin["detector"], "detector_version": detector.version,
            "examples": p.items_hash(rows), "n": len(rows), "hits": hits,
            "reason": terminal_reason(hits, len(rows), g, pin["id"], detector.version)}  # fmt: skip


def terminal_reason(hits: int, n: int, generation: int, pin: str, version: str) -> str:
    return (
        f"{hits} of {n} suppression examples of repair generation {generation} ({pin}) show "
        f"the behavior under detector {version}: no further repair, no reading"
    )


def terminal_fields_problems(held: Mapping[str, Any], behavior: str) -> list[str]:
    """Why a terminal manifest's fields are not a terminal contamination of ``behavior``'s
    registered repair: its schema (``TERMINAL_FIELDS``), the registered repair's generation,
    suppression pin and build detector, integer counts with 1 ≤ hits ≤ n, a content hash of 64
    hex digits, a detector other than the build one, and the reason its fields give."""
    fields = {k for k in held if k not in METADATA}
    if fields != TERMINAL_FIELDS:
        return [f"the terminal record has the fields {sorted(fields)}"]
    registered = _p().S3_REPAIRS.get(behavior) or {}
    pin = registered.get("suppress") or {}
    problems = []
    if (held["behavior"], held["repair_generation"], held["suppress"], held["build_detector"]) != (
        behavior,
        registered.get("repair_generation"),
        pin.get("id"),
        pin.get("detector"),
    ):
        problems.append("the terminal record is not of the behavior's registered repair's data")
    n, hits = held["n"], held["hits"]
    if not (type(n) is int and type(hits) is int and 1 <= hits <= n):
        problems.append(f"the terminal record's counts {hits} of {n} are not 1 ≤ hits ≤ n")
    if not (isinstance(held["examples"], str) and re.fullmatch(r"[0-9a-f]{64}", held["examples"])):
        problems.append("the terminal record's examples hash is not a sha256")
    if held["detector_version"] == held["build_detector"]:
        problems.append("the terminal record's detector is its data's build detector")
    if not problems and held["reason"] != terminal_reason(
        hits, n, held["repair_generation"], held["suppress"], held["detector_version"]
    ):
        problems.append("the terminal record's reason is not what its fields give")
    return problems


def terminal_problems(root: Path, data: Mapping[str, Any], find_run: Any) -> list[str]:
    if {k for k in data if k not in METADATA} != TERMINAL_FIELDS:
        return [f"the terminal record has the fields {sorted(data)}, not {sorted(TERMINAL_FIELDS)}"]
    try:
        record = terminal_record(root, str(data.get("behavior")), find_run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in METADATA}
    return [] if mine == record else ["the terminal record is not what its evidence gives"]


def write_terminal(root: Path, behavior: str, find_run: Any) -> dict:
    """The terminal contamination's manifest, recomputed and written once."""
    data = terminal_record(root, behavior, find_run)
    path = terminal_path(root, behavior)
    if path.exists():
        raise ValueError(f"{path} exists: a terminal contamination is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def committed_terminal(root: Path, behavior: str, find_run: Any) -> dict[str, Any] | None:
    """The behavior's committed terminal contamination (``manifests.committed_json``),
    recomputed; None when there is none."""
    from .manifests import committed_json

    path = terminal_path(root, behavior)
    if not path.exists():
        return None
    data = committed_json(path, "the stack lineage's terminal contamination", root)
    problems = terminal_problems(root, data, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


# A diagnostic that stays pending (§12.7 Stage 3, amendment 2026-10-24): its final labels hold
# a trace, and no candidate of the family frozen and committed before the person's review passes
# §12.3.1's acceptance checks. The behavior's reading closes with no analysis dose (p_i = 1) and
# its contract entry with no reading; an explicitly empty family is a decision, not missing
# evidence. FROZEN_FAMILIES: the families as frozen (§12.3.1), by behavior, each for one
# diagnostic set, its candidates' detector specs in their fixed order
PENDING = "stage3_pending"
FROZEN_FAMILIES: dict[str, dict[str, Any]] = {
    # myth_fact's ADL-corpus diagnostic, its family frozen empty (§12.3.1, 2026-10-24)
    "myth_fact": {"set_id": "c1fa2bf614fd", "candidates": [], "frozen": "2026-10-24"},
}
PENDING_FIELDS = {
    "behavior",
    "repair_generation",
    "diagnostic",
    "set_id",
    "detector_version",
    "traced",
    "family",
    "family_sha256",
    "candidates",
    "reason",
}


def pending_path(root: Path, behavior: str) -> Path:
    return Path(root) / "manifests" / f"{PENDING}_{behavior}.json"


def _candidate_checks(root: Path, behavior: str, spec: Mapping[str, Any], find_run: Any) -> dict:
    """A frozen candidate's §12.3.1 acceptance checks against the final labels, recomputed: 0 of
    ``UNPROMPTED`` stage-0 responses without the prompt, and no disagreement with the pilot set
    or with any committed diagnostic set of the behavior, whichever generation drew it."""
    from .behaviors import Detector

    candidate = Detector(**spec)
    hits, n = unprompted(candidate, behavior)
    disagree = pilot_disagreements(candidate, behavior)
    for m in _committed_diagnostics(root, behavior):
        disagree += diagnostic_disagreements(m, candidate, find_run)
    return {"version": candidate.version, "unprompted": [hits, n],
            "disagreements": len(disagree),
            "passes": (n, hits) == (UNPROMPTED, 0) and not disagree}  # fmt: skip


def pending_record(root: Path, behavior: str, find_run: Any) -> dict[str, Any]:
    """The behavior's pending diagnostic, recomputed: the committed diagnostic of the current
    repair generation that the behavior's frozen family (``FROZEN_FAMILIES``) was frozen for,
    bound by sha256; its evidence complete (the barrier's checks but the clearing of its
    traces), its final traces present and drawn under the current detector (no revision since);
    and every candidate of the family checked against the final labels, none passing. Refused
    otherwise: missing evidence is no closure."""
    from . import continuation_stage5 as S5
    from .manifests import committed_json

    p = _p()
    family = FROZEN_FAMILIES.get(behavior)
    if family is None:
        raise ValueError(f"{behavior} has no family frozen before a review: not pending")
    g, detector = current_generation(behavior), S5._behavior(behavior).detector
    rules = p._stage2_rules()
    for adl in (False, True):
        path = diagnostic_path(root, behavior, adl, g)
        if not path.exists():
            continue
        data = committed_json(path, "the pending diagnostic", root)
        if data.get("set_id") == family["set_id"]:
            break
    else:
        raise ValueError(f"the frozen family's set {family['set_id']} is no committed diagnostic "
                         f"of {behavior}'s repair generation {g}")  # fmt: skip
    _final_labels_problem(path.name, data, find_run, rules)
    traced = data.get("traced")
    if not traced:
        raise ValueError(f"{path.name}: no final trace: not pending")
    if data.get("detector_version") != detector.version:
        raise ValueError(f"{path.name}: drawn under {data.get('detector_version')}, not the "
                         f"current {detector.version}: not pending under its family")  # fmt: skip
    if adl:
        kinds, sources = (ADL,), committed_adl_sources(root, behavior, rules, find_run)
    else:
        kinds, sources = BEFORE_CONFIRMATION, committed_sources(root, behavior, rules, find_run)
    problems = _manifest_problems(root, behavior, data, sources, kinds, detector, find_run)
    cleared = revision_problems(root, behavior, data, detector, find_run)
    evidence = problems[: len(problems) - len(cleared)]
    if evidence or not cleared or problems[len(evidence) :] != cleared:
        raise ValueError(f"{path.name}: its evidence is not complete: {evidence or problems}")
    checks = [_candidate_checks(root, behavior, spec, find_run) for spec in family["candidates"]]
    passing = [c["version"] for c in checks if c["passes"]]
    if passing:
        raise ValueError(f"the frozen candidate {passing[0]} passes: a revision, not pending")
    return {"behavior": behavior, "repair_generation": g,
            "diagnostic": {"manifest": path.name, "sha256": data["sha256"]},
            "set_id": family["set_id"], "detector_version": detector.version,
            "traced": sorted(traced), "family": dict(family),
            "family_sha256": p.content_hash(family), "candidates": checks,
            "reason": pending_reason(len(traced), family["set_id"], detector.version,
                                     len(checks))}  # fmt: skip


def labels_dir(set_id: str) -> Path:
    """Where a diagnostic set's labels are kept (its ensemble's labels and the review)."""
    from .paths import data_root

    return data_root() / "labels" / "detectors" / f"s3diag-{set_id}"


def _final_labels_problem(name: str, data: Mapping[str, Any], find_run: Any, rules: Any) -> None:
    """The committed diagnostic's final labels recomputed from its evidence, whatever its
    family: its set found by its run, its input and key by the manifest's sha256 (the input
    hashing to the set's id), the ensemble's labels and the review by theirs
    (``labels_dir``), and the final labels (``final_labels``) the manifest's types, traced items
    and review counts. Refused otherwise."""
    import hashlib

    run = find_run(str(data.get("set_run")))
    if run is None:
        raise ValueError(f"{name}: its set {data.get('set_run')} is not under the runs root")
    _bound(Path(run), {"input.jsonl": data.get("input_sha256"),
                       "key.json": data.get("key_sha256")}, f"{name}'s set")  # fmt: skip
    key = json.loads((Path(run) / "key.json").read_text())
    blob = (Path(run) / "input.jsonl").read_bytes()
    if key.get("set_id") != data.get("set_id") or (
        hashlib.sha256(blob).hexdigest()[:12] != data.get("set_id")
    ):
        raise ValueError(f"{name}: its set is not {data.get('set_id')}")
    labels = labels_dir(str(data.get("set_id")))
    _bound(labels, {"ensemble_labels.jsonl": data.get("ensemble_sha256"),
                    "review.jsonl": data.get("review_sha256")}, f"{name}'s labels")  # fmt: skip
    final = final_labels(key["items"], key["kinds"], str(data.get("behavior")),
                         int(rules["seeds"]["labels"]), _rows(labels / "ensemble_labels.jsonl"),
                         _rows(labels / "review.jsonl"))  # fmt: skip
    if {k: data.get(k) for k in ("types", "traced", "review")} != final:
        raise ValueError(f"{name}: its final labels are not what its set, labels and review "
                         "give")  # fmt: skip


def pending_binding_problems(root: Path, held: Mapping[str, Any], behavior: str) -> list[str]:
    """Why a pending record does not bind the exact committed diagnostic it closes: the current
    repair generation's diagnostic manifest it names, committed at the sha256 it names, of the
    record's set (its frozen family's), drawn under the record's detector, its final traces the
    record's."""
    from .manifests import committed_json

    g, d = current_generation(behavior), held.get("diagnostic")
    names = {diagnostic_path(root, behavior, adl, g).name: adl for adl in (False, True)}
    if not (isinstance(d, dict) and d.get("manifest") in names):
        return [f"the pending record's diagnostic is not a diagnostic of {behavior}'s repair "
                f"generation {g}"]  # fmt: skip
    path = diagnostic_path(root, behavior, names[d["manifest"]], g)
    try:
        data = committed_json(path, "the pending diagnostic", root)
    except (OSError, RuntimeError, ValueError) as error:
        return [f"the pending record's diagnostic: {error}"]
    family = FROZEN_FAMILIES.get(behavior) or {}
    if data["sha256"] != d.get("sha256"):
        return ["the pending record's diagnostic is not the committed one it names"]
    if (data.get("set_id"), data.get("detector_version"), sorted(data.get("traced") or [])) != (
        family.get("set_id"),
        held.get("detector_version"),
        held.get("traced"),
    ) or held.get("set_id") != family.get("set_id"):
        return ["the pending record is not its diagnostic's set, detector and final traces"]
    return []


def pending_reason(traces: int, set_id: str, version: str, candidates: int) -> str:
    return (
        f"the diagnostic {set_id} keeps {traces} final trace(s) under detector {version} and no "
        f"candidate of its frozen family ({candidates}) passes: no analysis dose, no reading"
    )


def pending_fields_problems(held: Mapping[str, Any], behavior: str) -> list[str]:
    """Why a pending manifest's fields are not a pending diagnostic of ``behavior``: its schema
    (``PENDING_FIELDS``), the behavior's frozen family and its sha256, the diagnostic bound by
    name and sha256, final traces, no passing candidate, and the reason its fields give."""
    fields = {k for k in held if k not in METADATA}
    if fields != PENDING_FIELDS:
        return [f"the pending record has the fields {sorted(fields)}"]
    family = FROZEN_FAMILIES.get(behavior)
    if family is None:
        return [f"{behavior} has no family frozen before a review: no pending record"]
    problems = []
    if (held["behavior"], held["family"], held["set_id"]) != (
        behavior,
        family,
        family["set_id"],
    ) or held["family_sha256"] != _p().content_hash(held["family"]):
        problems.append("the pending record is not of the behavior's frozen family")
    d = held["diagnostic"]
    if not (isinstance(d, dict) and set(d) == {"manifest", "sha256"} and _sha(d["sha256"])):
        problems.append("the pending record's diagnostic is not a manifest bound by sha256")
    traced, checks = held["traced"], held["candidates"]
    if not (isinstance(traced, list) and traced and all(isinstance(i, str) for i in traced)):
        problems.append("the pending record holds no final trace")
    if not (isinstance(checks, list) and len(checks) == len(family["candidates"])
            and all(isinstance(c, dict) and c.get("passes") is False for c in checks)):  # fmt: skip
        problems.append("the pending record's candidates are not its family's, none passing")
    if not problems and held["reason"] != pending_reason(
        len(traced), held["set_id"], held["detector_version"], len(checks)
    ):
        problems.append("the pending record's reason is not what its fields give")
    return problems


def pending_problems(root: Path, data: Mapping[str, Any], find_run: Any) -> list[str]:
    if {k for k in data if k not in METADATA} != PENDING_FIELDS:
        return [f"the pending record has the fields {sorted(data)}, not {sorted(PENDING_FIELDS)}"]
    try:
        record = pending_record(root, str(data.get("behavior")), find_run)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return [str(error)]
    mine = {k: v for k, v in data.items() if k not in METADATA}
    return [] if mine == record else ["the pending record is not what its evidence gives"]


def write_pending(root: Path, behavior: str, find_run: Any) -> dict:
    """The pending diagnostic's manifest, recomputed and written once."""
    data = pending_record(root, behavior, find_run)
    path = pending_path(root, behavior)
    if path.exists():
        raise ValueError(f"{path} exists: a pending diagnostic's closure is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def committed_pending(root: Path, behavior: str, find_run: Any) -> dict[str, Any] | None:
    """The behavior's committed pending diagnostic (``manifests.committed_json``), recomputed;
    None when there is none."""
    from .manifests import committed_json

    path = pending_path(root, behavior)
    if not path.exists():
        return None
    data = committed_json(path, "the pending diagnostic's closure", root)
    problems = pending_problems(root, data, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data
