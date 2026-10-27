"""The completion of a certified lineage's blocked strict labels (preregistration §12.4.5,
amendment of 2026-10-27), from its committed specification
``manifests/stage3_blocked_completion_<lineage>.json``.

- **The cohort.** The specification pins every strict request whose request and sole retry were
  both safety-blocked (§12.4.5, 2026-10-26), by its canonical key: both original attempts and
  the lenient slot by their cache entries' sha256, the hypothesis text by its sha256, and every
  item reference (response id). One completed label serves every reference of its key.
- **The instrument.** strict-v4 on the specification's model and thinking level, the frozen
  messages, response schema and cap otherwise unchanged, on the standard tier (``settings``).
- **The validation.** Once, on the frozen reference set of the affected lens layer (its files
  pinned by sha256): each item's strict slot (the request and its one parser retry) with the
  set's unchanged lenient labels under rule C's criteria (``judge.c_validation``); it passes when
  criterion 1 passes and criterion 2 passes or is not applicable.
- **The replacement.** After a passed validation, each cohort request's slot keeps its first
  answer the frozen strict parser accepts, from the request and its one retry; a valid answer is
  never replaced. A failed send is not cached: its rerun resends it, and it consumes no attempt.
- **The blind review.** Every cohort request after a failed validation, else those whose
  replacement slot was refused twice, labelled by the researcher among the specification's 30
  seeded decoys: opaque ids in a seeded order, the hypothesis text and the target only. The page
  is issued once (``issued.json``: its sha256 and items) and its export imported once, exactly
  the issued items (``human.json``); the decoys' agreement with their frozen labels is
  descriptive.
- **The provenance.** ``provenance`` recomputes all of it from the cache and the written files;
  it is written once as ``manifests/stage3_blocked_completion_<lineage>_provenance.json`` and
  committed before the readers' verdicts, which recompute it (``completion``). It holds keys,
  hashes and statuses, never a reply text or a label value.

Every output is written under ``<DATA_DIR>/stage3_blocked_completion/<lineage>``; nothing here
prints a label value."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import continuation_readers as E
from .continuation_panel import Pending

SPEC = "stage3_blocked_completion_{}.json"
PROVENANCE = "stage3_blocked_completion_{}_provenance.json"
ISSUED, HUMAN, PAGE, LOG = "issued.json", "human.json", "review.html", "requests.jsonl"
LABELER = "researcher"  # the blind review's labeler (judge.parse_export)
DECOYS = 30


def spec_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / SPEC.format(lineage)


def provenance_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / PROVENANCE.format(lineage)


def work_dir(lineage: str) -> Path:
    from .paths import data_root

    return data_root() / "stage3_blocked_completion" / lineage


def committed_spec(lineage: str) -> dict[str, Any] | None:
    """The lineage's committed specification with its sha256; None when it has none."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    path = spec_path(REPO_ROOT, lineage)
    if not path.exists():
        return None
    data = committed_json(path, "the completion of the blocked strict labels")
    if data.get("lineage") != lineage:
        raise ValueError(f"{path.name}: the specification of {data.get('lineage')}")
    return data


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------- the instrument


def settings(spec: Mapping[str, Any]) -> Any:
    """The replacement instrument: strict-v4 on the specification's model and thinking level (the
    judge config's entry of that model), the frozen judge's cap and structured output."""
    from .config import load_config
    from .judge import JudgeSettings, rubric_set

    ins = spec["instrument"]
    j = load_config(E.JUDGE, ["tracking.wandb.enabled=false"]).judge
    model = E._spec(next(m for m in j.models if m.id == ins["model"]))
    frozen = E.judge_settings(E.RUBRICS["lens"])
    rubric = frozen.rubrics.strict
    if (
        model.reasoning_effort != ins["reasoning_effort"]
        or (rubric.version, rubric.sha256) != (ins["rubric"]["version"], ins["rubric"]["sha256"])
        or (frozen.max_tokens, frozen.structured_output) != (ins["max_tokens"],
                                                             ins["structured_output"])
        or (ins["service_tier"], ins["backend"]) != (E.STANDARD, "sync")
    ):  # fmt: skip
        raise ValueError("the specification's instrument is not strict-v4 under the frozen judge")
    return JudgeSettings(model=model, rubrics=rubric_set(E.RUBRICS["lens"]),
                         rubric_version=E.RUBRICS["lens"], lenient=False,
                         max_tokens=frozen.max_tokens,
                         structured_output=frozen.structured_output)  # fmt: skip


def request(item: Any, instrument: Any) -> Any:
    from .judge import build_request

    return build_request(instrument.rubrics.strict, item, instrument)


def retry(r: Any) -> Any:
    from .judge import ChatRequest, request_key

    return ChatRequest(request_key(f"{r.rubric_version}{E.RETRY}", r.body), r.kind,
                       r.rubric_version, r.body)  # fmt: skip


def outcome(cache: Any, r: Any, where: str) -> dict[str, Any]:
    """A strict slot read from the cache: ``answered`` (its first answer the frozen parser
    accepts, the retry read only after a refusal), ``refused`` (both attempts refused) or
    ``open`` (an attempt due has no answer). Each attempt answers exactly the request, on the
    standard tier."""
    from .judge import MalformedOutput, cache_entry_sha256, parse_strict

    history = []
    for n, key in enumerate((r.key, retry(r).key), start=1):
        entry = cache.entry(key)
        if entry is None:
            return {"status": "open", "history": history}
        if entry.get("body") != r.body:
            raise ValueError(f"{where}: attempt {n}'s answer is not to its request")
        if entry.get("service_tier") != E.STANDARD:
            raise ValueError(f"{where}: attempt {n} is not on the standard tier")
        step = {"key": key, "sha256": cache_entry_sha256(entry), "tier": E.STANDARD}
        try:
            value = parse_strict(entry.get("text"))[0]
        except MalformedOutput:
            history.append({**step, "parsed": False})
            continue
        history.append({**step, "parsed": True})
        return {"status": "answered", "value": value, "history": history}
    return {"status": "refused", "history": history}


def refused_first(cache: Any, r: Any) -> bool:
    """The request's answer is cached and refused by the frozen strict parser: its retry is due."""
    from .judge import MalformedOutput, parse_strict

    entry = cache.entry(r.key)
    if entry is None:
        return False
    try:
        parse_strict(entry.get("text"))
    except MalformedOutput:
        return True
    return False


def due(cache: Any, requests: Sequence[Any]) -> list[Any]:
    """The attempts still to send: each request without an answer, else its retry when the answer
    was refused and the retry has none."""
    out = []
    for r in requests:
        if cache.entry(r.key) is None:
            out.append(r)
        elif refused_first(cache, r) and cache.entry(retry(r).key) is None:
            out.append(retry(r))
    return out


def cache_of(spec: Mapping[str, Any], find_run: Any) -> Any:
    """The judge cache of the lineage's lens judging (the original attempts' and the
    replacements')."""
    _, config, _ = E._run(find_run, spec["sources"]["lens_judge"], None, E.EXPERIMENTS["judge"])
    return E._cache(config, "judge", "judge")


# ------------------------------------------------------------------------- the validation


def _runs_file(rel: str) -> Path:
    from .paths import runs_root

    return runs_root() / Path(rel).relative_to("runs")


def reference(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen reference set: its pinned files, its items as judge items, the frozen pair's
    lenient labels, the final reference labels and each annotator's, and the stratum's keys."""
    from .judge import JudgeItem, read_annotator_labels

    v = spec["validation"]
    files = {}
    for rel, digest in sorted(v["files"].items()):
        path = _runs_file(rel)
        if not path.exists():
            raise Pending(f"{rel}: the reference set is not here")
        if E.file_sha(path) != digest:
            raise ValueError(f"{rel}: not the reference set's pinned file")
        files[Path(rel).name] = path
    provenance = files["labels-" + v["reference_set"] + ".jsonl"].parent
    mapping = json.loads(files["mapping.json"].read_text())
    if mapping["set_id"] != v["reference_set"] or len(mapping["items"]) != v["items"]:
        raise ValueError(f"{mapping['set_id']}: not the reference set {v['reference_set']}")

    def lines(path: Path) -> dict[str, dict]:
        return {r["item_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}

    inputs, frozen = lines(files["input.jsonl"]), lines(files["frozen_pair_labels.jsonl"])
    final_rows = lines(files["labels-" + v["reference_set"] + ".jsonl"])
    items, lenient, final = {}, {}, {}
    for e in mapping["items"]:
        k = (e["response_id"], e["target_behavior_id"])
        row = inputs[e["item_id"]]
        items[k] = JudgeItem(e["response_id"], e["target_behavior_id"], row["target"],
                             row["response"])  # fmt: skip
        lenient[k] = frozen[e["item_id"]]["lenient_label"]
        final[k] = (final_rows[e["item_id"]]["strict_label"],
                    final_rows[e["item_id"]]["lenient_label"])  # fmt: skip
        if not isinstance(lenient[k], bool):
            raise ValueError(f"{e['item_id']}: the frozen pair's lenient label is missing")
    # each annotator's file pinned directly or by a pinned sha256 listing of its directory
    listed = {}
    for name, path in files.items():
        if name.endswith(".sha256"):
            for line in path.read_text().splitlines():
                digest, rel = line.split(maxsplit=1)
                listed[(path.with_suffix("") / rel).resolve()] = digest
    pinned = {p.resolve(): E.file_sha(p) for p in files.values()} | listed
    annotators = {}
    for n in v["annotators"]:
        path = provenance / f"{n}.jsonl"
        if pinned.get(path.resolve()) != E.file_sha(path):
            raise ValueError(f"{path.name}: not an annotator's pinned labels")
        annotators[n] = read_annotator_labels(path.read_text(), mapping)
    keys = sorted((e["response_id"], e["target_behavior_id"]) for e in mapping["items"]
                  if e.get("stratum", "random") in set(v["strata"]))  # fmt: skip
    return {"items": items, "lenient": lenient, "final": final, "annotators": annotators,
            "keys": keys}  # fmt: skip


def validation_requests(spec: Mapping[str, Any], instrument: Any) -> dict[tuple, Any]:
    ref = reference(spec)
    return {k: request(item, instrument) for k, item in ref["items"].items()}


def validation(spec: Mapping[str, Any], cache: Any, instrument: Any) -> dict[str, Any]:
    """The validation recomputed from the cache: each reference item's strict slot, rule C's
    criteria with the frozen lenient labels, and the result; Pending while a slot is open."""
    from .judge import c_validation

    ref = reference(spec)
    judged, slots = {}, {}
    for k, item in ref["items"].items():
        o = outcome(cache, request(item, instrument), f"validation {k[0]}")
        if o["status"] == "open":
            raise Pending(f"validation: {k[0]} has no answer yet")
        judged[k] = (o.get("value"), ref["lenient"][k])
        slots[f"{k[0]} {k[1]}"] = o["history"]
    found = c_validation(judged, ref["final"], ref["annotators"], ref["keys"])
    passed = found["criterion_1"] == "pass" and found["criterion_2"] in ("pass", "not applicable")
    return {"reference_set": spec["validation"]["reference_set"],
            "result": "pass" if passed else "fail",
            **{c: found[c] for c in ("criterion_1", "criterion_2", "criterion_L")},
            "details": found["details"], "slots": dict(sorted(slots.items()))}  # fmt: skip


# --------------------------------------------------------------------------------- the cohort


def cohort(spec: Mapping[str, Any], texts: Mapping[str, str], target: str,
           instrument: Any) -> dict[str, tuple[Any, Any]]:  # fmt: skip
    """{cohort key: (judge item, replacement request)}: each key's hypothesis, its text the
    one the specification pins for every reference, its frozen strict request the key itself."""
    from .judge import JudgeItem, stage_requests

    frozen = E.judge_settings(E.RUBRICS["lens"])
    out = {}
    for key, c in sorted(spec["cohort"]["requests"].items()):
        for rid in c["response_ids"]:
            if rid not in texts:
                raise Pending(f"{rid}: the cohort's hypothesis is not here")
            if sha(texts[rid]) != c["text_sha256"]:
                raise ValueError(f"{rid}: not the cohort's pinned hypothesis")
        first = c["response_ids"][0]
        item = JudgeItem(first, spec["behavior"], target, texts[first])
        if stage_requests([item], frozen)[0].strict.key != key:
            raise ValueError(f"{key}: not the strict request of its hypothesis")
        out[key] = (item, request(item, instrument))
    return out


def originals(spec: Mapping[str, Any], cache: Any) -> None:
    """Both original strict attempts and the lenient slot of every cohort request, as the
    specification pins them; a changed entry is refused."""
    from .judge import cache_entry_sha256

    for key, c in spec["cohort"]["requests"].items():
        for step in [*c["strict_attempts"], *c["lenient_slot"]]:
            entry = cache.entry(step["key"])
            if entry is None or cache_entry_sha256(entry) != step["entry_sha256"]:
                raise ValueError(f"{key}: {step['key']} is not the entry the specification pins")


def replacements(cache: Any, requests: Mapping[str, tuple]) -> dict[str, dict[str, Any]]:
    """Each cohort request's replacement slot (``outcome``)."""
    return {key: outcome(cache, r, f"replacement {key}") for key, (_, r) in requests.items()}


def routed(found: Mapping[str, Any], outcomes: Mapping[str, Mapping]) -> list[str]:
    """The cohort requests the blind review labels: all after a failed validation, else those
    whose replacement slot was refused twice; Pending while a replacement slot is open."""
    if found["result"] == "fail":
        return sorted(outcomes)
    open_ = sorted(k for k, o in outcomes.items() if o["status"] == "open")
    if open_:
        raise Pending(f"{len(open_)} replacement slots have no answer yet")
    return sorted(k for k, o in outcomes.items() if o["status"] == "refused")


# ---------------------------------------------------------------------------- the blind review


def review_items(spec: Mapping[str, Any], keys: Sequence[str], texts: Mapping[str, str],
                 target: str) -> list[Any]:  # fmt: skip
    """The review's items: one per routed cohort request and the 30 decoys, each with an opaque
    id from the seed and its text's sha256, in the seed's order."""
    from .judge import JudgeItem, LabelingItem

    h = spec["human"]
    seed = h["seed"]
    entries = [(spec["cohort"]["requests"][k]["response_ids"][0],
                spec["cohort"]["requests"][k]["text_sha256"]) for k in keys]  # fmt: skip
    entries += [(d["response_id"], d["text_sha256"]) for d in h["decoys"]]
    if len(h["decoys"]) != min(DECOYS, h["decoy_frame"]["distinct_texts"]):
        raise ValueError("the specification's decoys are not its frame's draw")
    items = []
    for rid, digest in entries:
        if rid not in texts or sha(texts[rid]) != digest:
            raise ValueError(f"{rid}: not the review item the specification pins")
        item_id = sha(f"{seed}:{digest}")[:12]
        items.append(LabelingItem(item_id, JudgeItem(rid, spec["behavior"], target, texts[rid])))
    if len({li.item_id for li in items}) != len(items):
        raise ValueError("two review items share a text")
    return sorted(items, key=lambda li: sha(f"{seed}:order:{li.item_id}"))


def issued(lineage: str, spec_sha: str, keys: Sequence[str], items: Sequence[Any],
           page: str) -> dict[str, Any]:  # fmt: skip
    """The review's private record at issue: the set id, the page's sha256, each item's id and
    reference, and the routed cohort keys."""
    from .judge import labeling_mapping, labeling_set_id

    set_id = labeling_set_id(items)
    mapping = labeling_mapping(set_id, items)
    return {"lineage": lineage, "spec": spec_sha, "routed": list(keys), **mapping,
            "page_sha256": sha(page)}  # fmt: skip


def write_once(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as f:
            f.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    except FileExistsError:
        raise ValueError(f"{path} exists: it is written once") from None


def import_review(lineage: str, export: Path) -> dict[str, Any]:
    """The review's export imported once, locked: exactly the issued items, one strict label
    each, under the issued set; the lenient label is not read."""
    from .judge import parse_export

    work = work_dir(lineage)
    issue = json.loads((work / ISSUED).read_text())
    text = Path(export).read_text()
    rows = parse_export(text, issue, labeler=LABELER)
    if sorted(r.item_id for r in rows) != sorted(e["item_id"] for e in issue["items"]):
        raise ValueError("the export does not label exactly the issued items")
    data = {"set_id": issue["set_id"], "page_sha256": issue["page_sha256"],
            "issued_sha256": E.file_sha(work / ISSUED), "export_sha256": sha(text),
            "labels": {r.item_id: r.strict_label for r in rows}}  # fmt: skip
    write_once(work / HUMAN, data)
    return {"items": len(rows), "export_sha256": data["export_sha256"]}


def review(lineage: str, spec: Mapping[str, Any], keys: Sequence[str], texts: Mapping[str, str],
           target: str, decoy_labels: Mapping[str, str]) -> dict[str, Any]:  # fmt: skip
    """The imported review recomputed: the issued items are the specification's for ``keys``,
    the import is of that issue; each routed key's label and the decoys' descriptive agreement
    with their frozen strict labels (``decoy_labels``)."""
    from .judge import labeling_mapping, labeling_set_id

    work = work_dir(lineage)
    if not (work / ISSUED).exists() or not (work / HUMAN).exists():
        raise Pending("the blind review is not imported")
    issue, human = json.loads((work / ISSUED).read_text()), json.loads((work / HUMAN).read_text())
    items = review_items(spec, keys, texts, target)
    mapping = labeling_mapping(labeling_set_id(items), items)
    if ({k: issue[k] for k in ("set_id", "items")} != mapping
            or issue["routed"] != list(keys) or issue["spec"] != spec["sha256"]):  # fmt: skip
        raise ValueError("the issued review is not the specification's for its routed requests")
    if (human["set_id"], human["page_sha256"], human["issued_sha256"]) != (
            issue["set_id"], issue["page_sha256"], E.file_sha(work / ISSUED)):  # fmt: skip
        raise ValueError("the imported review is not of the issued page")
    if sorted(human["labels"]) != sorted(e["item_id"] for e in issue["items"]):
        raise ValueError("the imported review does not label exactly the issued items")
    id_of = {li.item.response_id: li.item_id for li in items}
    labels = {k: human["labels"][id_of[spec["cohort"]["requests"][k]["response_ids"][0]]]
              for k in keys}  # fmt: skip
    decoys = [id_of[d["response_id"]] for d in spec["human"]["decoys"]]
    agree = sum(human["labels"][id_of[d["response_id"]]] == decoy_labels[d["response_id"]]
                for d in spec["human"]["decoys"])  # fmt: skip
    return {"labels": labels,
            "record": {"set_id": issue["set_id"], "page_sha256": issue["page_sha256"],
                       "issued_sha256": human["issued_sha256"],
                       "export_sha256": human["export_sha256"], "items": len(issue["items"]),
                       "decoy_agreement": {"agree": agree, "decoys": len(decoys)}}}  # fmt: skip


# ------------------------------------------------------------------------ the provenance


def provenance(spec: Mapping[str, Any], cache: Any, texts: Mapping[str, str], target: str,
               decoy_labels: Mapping[str, str]) -> tuple[dict, dict[str, tuple]]:  # fmt: skip
    """The completion recomputed: (its provenance, {cohort key: (strict label, history)}). The
    provenance holds keys, hashes and statuses only; Pending while any step is unfinished."""
    instrument = settings(spec)
    originals(spec, cache)
    found = validation(spec, cache, instrument)
    requests = cohort(spec, texts, target, instrument)
    outcomes = (replacements(cache, requests) if found["result"] == "pass"
                else {k: {"status": "not sent", "history": []} for k in requests})  # fmt: skip
    keys = routed(found, outcomes)
    human = review(spec["lineage"], spec, keys, texts, target, decoy_labels) if keys else None
    values, completed = {}, {}
    for key, o in outcomes.items():
        if key in keys:
            # the review's label: no cache entry, its export's sha256
            review_step = {"key": None, "review": human["record"]["export_sha256"], "parsed": True}
            values[key] = (human["labels"][key], [*o["history"], review_step])
            completed[key] = {"source": "review", "replacement": o["status"]}
        else:
            values[key] = (o["value"], o["history"])
            completed[key] = {"source": "replacement", "history": o["history"]}
    record = {
        "lineage": spec["lineage"], "spec": spec["sha256"],
        "instrument": {"model": instrument.model.id,
                       "reasoning_effort": instrument.model.reasoning_effort,
                       "rubric": instrument.rubrics.strict.version},
        "validation": {k: v for k, v in found.items()},
        "replacements": {k: {"status": o["status"], "history": o["history"]}
                         for k, o in outcomes.items()},
        "routed": keys,
        "review": None if human is None else human["record"],
        "completed": completed,
    }  # fmt: skip
    return record, values


def write_provenance(root: Path, spec: Mapping[str, Any], cache: Any, texts: Mapping[str, str],
                     target: str, decoy_labels: Mapping[str, str]) -> dict[str, Any]:  # fmt: skip
    """The provenance written once to ``manifests/stage3_blocked_completion_<lineage>_
    provenance.json``, to be committed before the readers' verdicts."""
    record, _ = provenance(spec, cache, texts, target, decoy_labels)
    write_once(provenance_path(root, spec["lineage"]), record)
    return record


def completion(spec: Mapping[str, Any], cache: Any, texts: Mapping[str, str], target: str,
               decoy_labels: Mapping[str, str]) -> dict[str, Any]:  # fmt: skip
    """The completed labels the readers read: the committed provenance, recomputed and refused
    when it differs; Pending while it is not committed."""
    from .manifests import committed_json
    from .paths import REPO_ROOT

    path = provenance_path(REPO_ROOT, spec["lineage"])
    if not path.exists():
        raise Pending(f"{path.name}: the completion's provenance is not committed")
    committed = committed_json(path, "the readers' verdicts")
    record, values = provenance(spec, cache, texts, target, decoy_labels)
    if {k: v for k, v in committed.items() if k not in ("sha256", "committed_at")} != json.loads(
            json.dumps(record)):  # fmt: skip
        raise ValueError(f"{path.name}: not its recomputation")
    return {"spec": spec["sha256"], "provenance": committed["sha256"], "values": values,
            "routed": record["routed"], "validation": record["validation"]["result"]}  # fmt: skip


def completed_labels(lineage: Any, sources: Mapping[str, Any], find_run: Any,
                     lens_h: Mapping[str, tuple], lens_l: Mapping[str, tuple],
                     history: Mapping[str, Any] | None) -> dict[str, Any] | None:  # fmt: skip
    """The readers' completed labels of the lineage's cohort ({response id: (strict, lenient,
    histories)}, the completion's history after the original ones) and its record; None
    without a committed specification. The specification must be of these readings, its cohort
    exactly the lens judging's blocked strict requests with their pinned histories and its decoys
    the frame's seeded draw."""
    from .judge import load_targets

    spec = committed_spec(lineage.identity["lineage"])
    if spec is None:
        return None
    behavior = lineage.identity["behavior"]
    if (
        spec["behavior"] != behavior
        or any(spec["sources"][k] != sources[k] for k in ("record", "lens_reading", "lens_judge"))
        or spec["sources"]["judge_history_manifest_sha256"] != (history or {}).get("sha256")
    ):  # fmt: skip
        raise ValueError("the completion's specification is of other readings")
    target = load_targets([behavior])[behavior]
    if sha(target) != spec["target_sha256"]:
        raise ValueError("the completion's specification is of another target")
    blocked = {rid: v for rid, v in lens_l.items() if v[0] == E.STRICT_SAFETY_BLOCKED}
    found: dict[str, list[str]] = {}
    for rid, v in blocked.items():
        found.setdefault(v[2][0][0]["key"], []).append(rid)
    cohort = spec["cohort"]["requests"]
    if {k: sorted(v) for k, v in found.items()} != {k: sorted(c["response_ids"])
                                                    for k, c in cohort.items()}:  # fmt: skip
        raise ValueError("the specification's cohort is not the lens judging's blocked requests")
    for rid, (_, _, (strict, lenient)) in blocked.items():
        c = cohort[strict[0]["key"]]
        pins = ([{"key": s["key"], "entry_sha256": s["sha256"], "tier": s["tier"],
                  "finish_reason": s["finish_reason"]} for s in strict],
                [{"key": s["key"], "entry_sha256": s["sha256"], "tier": s["tier"]}
                 for s in lenient])  # fmt: skip
        if pins != (c["strict_attempts"], c["lenient_slot"]):
            raise ValueError(f"{rid}: its blocked history is not the one the specification pins")
    strict_of = {rid: v[0] for rid, v in lens_l.items() if rid not in blocked}
    frame = decoy_frame(spec, lens_h, strict_of)
    if frame != {**{k: spec["human"]["decoy_frame"][k]
                    for k in ("distinct_texts", "references", "sha256")},
                 "decoys": spec["human"]["decoys"]}:  # fmt: skip
        raise ValueError("the specification's decoys are not its frame's seeded draw")
    texts = {rid: h[2] for rid, h in lens_h.items()}
    done = completion(spec, cache_of(spec, find_run), texts, target, strict_of)
    labels = {}
    for rid, (_, lenient_value, histories) in blocked.items():
        value, steps = done["values"][histories[0][0]["key"]]
        labels[rid] = (value, lenient_value, [*histories, steps])
    record = {k: done[k] for k in ("spec", "provenance", "validation")}
    return {"labels": labels,
            "record": {**record, "requests": len(cohort), "references": len(blocked),
                       "reviewed": len(done["routed"])}}  # fmt: skip


def decoy_frame(spec: Mapping[str, Any], hyps: Mapping[str, tuple],
                strict_of: Mapping[str, str]) -> dict[str, Any]:  # fmt: skip
    """The decoy frame recomputed from the lineage's lens hypotheses ({response id: (key, sample,
    text, ...)}) and their strict labels: the specification's layer outside its package, labelled,
    one entry per text other than the cohort's, and its seeded draw."""
    c = spec["cohort"]
    texts = {r["text_sha256"] for r in c["requests"].values()}
    frame: dict[str, list[str]] = {}
    for rid, h in hyps.items():
        _, package, _, tool, *_ = rid.split("/")
        if tool != c["layer"] or package == c["package"] or rid not in strict_of:
            continue
        t = sha(h[2])
        if t not in texts:
            frame.setdefault(t, []).append(rid)
    order = sorted(frame)
    picked = sorted(random.Random(spec["human"]["seed"]).sample(order, min(DECOYS, len(order))))
    return {"distinct_texts": len(order), "references": sum(len(v) for v in frame.values()),
            "sha256": sha("\n".join(order)),
            "decoys": [{"text_sha256": t, "response_id": sorted(frame[t])[0]}
                       for t in picked]}  # fmt: skip


def counted_completions(spec: Mapping[str, Any], cache: Any, texts: Mapping[str, str],
                        target: str,
                        decoy_labels: Mapping[str, str]) -> dict[str, str]:  # fmt: skip
    """{response id: text} of every cohort reference that rule C counts by its completed strict
    label and its unchanged lenient slot: the category grader's items beside the labelled ones."""
    from .judge import JudgeItem, parse_lenient, request_key, stage_requests

    _, values = provenance(spec, cache, texts, target, decoy_labels)
    frozen = E.judge_settings(E.RUBRICS["lens"])
    out = {}
    for key, c in spec["cohort"]["requests"].items():
        first = c["response_ids"][0]
        lenient = stage_requests([JudgeItem(first, spec["behavior"], target, texts[first])],
                                 frozen)[0].lenient  # fmt: skip
        keys = [lenient.key, request_key(f"{lenient.rubric_version}{E.RETRY}", lenient.body)]
        found = E.slot(cache, keys, lenient.body, parse_lenient, f"{first} (lenient)")
        if E.counted(values[key][0], found["value"]):
            out |= {rid: texts[rid] for rid in c["response_ids"]}
    return out


def spec_of_reading(root: Path, lens_reading: str) -> dict[str, Any] | None:
    """The committed specification whose lens reading is ``lens_reading``; None when none is."""
    found = [p for p in sorted((Path(root) / "manifests").glob(SPEC.format("*")))
             if not p.name.endswith("_provenance.json")
             and json.loads(p.read_text())["sources"]["lens_reading"] == lens_reading]  # fmt: skip
    if len(found) > 1:
        raise ValueError(f"{lens_reading}: two completion specifications")
    return committed_spec(json.loads(found[0].read_text())["lineage"]) if found else None
