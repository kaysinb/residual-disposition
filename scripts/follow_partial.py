"""Steering at the partial threshold on the laptop (§13.16, ``follow_partial``): the pins before
any Pod, the Pod sessions' records, and rung 2's readings on the base decoder by
``scripts/follow_rung2.py``'s pins and send, on ``follow/rung3_shared``.

    uv run python scripts/follow_partial.py pin-sources [--write]
    uv run python scripts/follow_partial.py record-build RUN [--write]
    uv run python scripts/follow_partial.py record-steering RUN [--write]
    uv run python scripts/follow_partial.py manual-import EXPORT MAPPING
    uv run python scripts/follow_partial.py format-kit QUALIFICATION_RUN ANALYSIS_RUN
    uv run python scripts/follow_partial.py format-compare QUALIFICATION_RUN ANALYSIS_RUN
    uv run python scripts/follow_partial.py [--pin | --send qualification |
                                             --send analysis --qualification RUN |
                                             --judge-history RUN]

``pin-sources`` writes every vector source's sha256 (``manifests/follow_partial_sources.json``),
each build's tensors checked against its build session's result first; the Pod sessions refuse any
other file. ``RUN`` of the records is the pulled Pod run (``runs/follow_partial_pod/<run>``), its
base_random session under ``out/build`` and its steering sessions under ``out/steering/<line>``:
``record-build`` records the branches once, with the logit-lens check of every package without a
stored difference complete and without a problem; ``record-steering`` the four lines' samples
once, of the recorded build session and the pinned sources. ``manual-import`` adds the researcher's
labels of one safety-blocked batch, with its line and cell (§13.16, after H11). ``format-kit``
writes the targeted format check (option c) of the three behaviors beyond eli5's validation:
``data/follow_partial/format/kit``, the reference instructions of D*'s format validation verbatim,
and its private mapping beside it; ``format-compare`` its labels (``sonnet/``, ``grok/``; where
they differ, ``disputes/arbiter/out.jsonl``) against the judge by rule C and against the category
and specificity graders where a hypothesis was graded: ``compare.json``, descriptive. The
readings' phases are rung 2's; the format is eli5's validation carried over (no supplement),
and the decision per line carries the frozen comparison with rung 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_partial as FP
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_readings as D
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT, runs_root

POD = "follow_partial_pod"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(name: str, record: dict[str, Any], write: bool) -> int:
    path = REPO_ROOT / name
    if not write:
        print(json.dumps(record, indent=1)[:4000])
        return 0
    if path.exists():
        raise SystemExit(f"{name} exists: the record is written once")
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {name} ({sha(path)}): commit it")
    return 0


def files_of(d: Path) -> dict[str, str]:
    return {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file()}


def pin_sources(write: bool) -> int:
    """Every vector source at its sha256, each build's tensors first against its session's
    result."""
    root = runs_root() / "follow_build" / FP.BUILD / "out" / "build" / FP.BUILD
    recorded = json.loads((root / "result.json").read_text())["files"]
    now = FP.sources(find_run)
    for name, digest in now.items():
        if name.startswith("build/") and name.endswith(".safetensors"):
            _, state, file = name.split("/", 2)
            if recorded.get(f"builds/{state}/{file}") != digest:
                raise SystemExit(f"{name}: not its build session's recorded file")
    return write_once(FP.SOURCES, {"sources": now}, write)


def pod_out(run: str, part: str) -> tuple[Path, Path]:
    d = runs_root() / POD / run
    return d, d / "out" / part


def record_build(run: str, write: bool) -> int:
    """The base_random session's branches, once: complete, every package's files as its result
    records them, and its logit-lens check of every package without a stored difference passed."""
    from residual_disposition.manifests import committed_json

    d, out = pod_out(run, "build")
    result = json.loads((out / "result.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    packages, _ = FP.base_random_packages(find_run)
    keys = sorted(p["run"] for p in packages)
    if result["status"] != "complete" or sorted(result["packages"]) != keys:
        raise SystemExit(f"{run}: not the planned packages, complete")
    lens = manifest["lens_check"]
    if sorted(lens) != FP.unchecked_runs() or any(v["problems"] for v in lens.values()):
        raise SystemExit(f"{run}: the logit-lens check is not complete and clear")
    pins = committed_json(REPO_ROOT / FP.SOURCES, "the vector sources' pins")
    if manifest.get("sources_sha256") != pins["sha256"]:
        raise SystemExit(f"{run}: not of the pinned vector sources")
    made = {}
    for key in keys:
        p = out / "packages" / key
        if files_of(p) != result["packages"][key]["files"]:
            raise SystemExit(f"{key}: not its recorded, complete files")
        made[key] = {"dir": str(p.relative_to(d)), "files": result["packages"][key]["files"]}
    order = {
        r: sorted(p for p, x in v["positions"].items() if not x["diagnostics"]["top10_same_order"])
        for r, v in lens.items()
    }
    widest = max(x["diagnostics"]["max_abs_difference"] for v in lens.values()
                 for x in v["positions"].values())  # fmt: skip
    record = {"run": run, "manifest_sha256": sha(out / "manifest.json"),
              "result_sha256": sha(out / "result.json"), "packages": made,
              "lens_check": {"runs": sorted(lens), "problems": 0, "max_abs_difference": widest,
                             "top10_order_differs_at": order}}  # fmt: skip
    return write_once(FP.BASE_RANDOM_RECORD, record, write)


def record_steering(run: str, write: bool) -> int:
    """The four lines' samples, once: each line's session complete with its planned cells, every
    cell's files as its result records them, of the recorded build session and the pinned
    sources."""
    from residual_disposition.manifests import committed_json

    build = committed_json(REPO_ROOT / FP.BASE_RANDOM_RECORD, "the base_random record")
    pins = committed_json(REPO_ROOT / FP.SOURCES, "the vector sources' pins")
    d, out = pod_out(run, "steering")
    lines: dict[str, Any] = {}
    for line in FP.LINES:
        s = out / line
        result = json.loads((s / "result.json").read_text())
        manifest = json.loads((s / "manifest.json").read_text())
        planned = {c["cell"]: c for c in FP.steering_cells(line, find_run)}
        if result["status"] != "complete" or sorted(result["cells"]) != sorted(planned):
            raise SystemExit(f"{line}: not the planned cells, complete")
        if (manifest["build_manifest_sha256"], manifest["sources_sha256"]) != (
                build["manifest_sha256"], pins["sha256"]):  # fmt: skip
            raise SystemExit(f"{line}: not of the recorded build session and the pinned sources")
        cells = {}
        for name, cell in result["cells"].items():
            c = s / "cells" / name
            if files_of(c) != cell["files"] or cell["run"] != planned[name]["run"]:
                raise SystemExit(f"{line} {name}: not its recorded, complete files")
            steering = json.loads((c / "steering.json").read_text())
            cells[name] = {"dir": str(c.relative_to(d)), "files": cell["files"],
                           "vector_check": steering["vector_check"],
                           "failed_positions": sorted(p for p, e in steering["per_position"].items()
                                                      if e["failed"])}  # fmt: skip
        lines[line] = cells
    record = {"run": run, "lines": lines}
    return write_once(FP.STEERING_RECORD, record, write)


def manual_import(export: Path, mapping: Path) -> int:
    """One safety-blocked batch's labels: one per item of its page, within the rubric, with its
    line, cell and request sha256; a batch recorded already is refused."""
    path = REPO_ROOT / FP.MANUAL
    record = json.loads(path.read_text()) if path.exists() else {"batches": []}
    raw = export.read_text()
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    m = json.loads(mapping.read_text())
    numbers = m["answer_number_by_item"]
    got = {r["item_id"]: r["label"] for r in rows}
    if len(rows) != len(got) or sorted(got) != sorted(numbers):
        raise SystemExit("the export is not one label per item of the page")
    if set(got.values()) - {"COHERENT", "INCOHERENT"}:
        raise SystemExit("a label outside the rubric")
    if m["line"] not in FP.LINES:
        raise SystemExit(f"{m['line']}: not a line of this extension")
    if m["user_sha256"] in {b["user_sha256"] for b in record["batches"]}:
        raise SystemExit("this batch is recorded already")
    labels = [got[i] for i in sorted(numbers, key=lambda i: numbers[i])]
    digest = hashlib.sha256(raw.encode()).hexdigest()
    record["batches"].append({"line": m["line"], "cell": m["cell"], "context": m["context"],
                              "user_sha256": m["user_sha256"], "labels": labels,
                              "export_sha256": digest})  # fmt: skip
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {FP.MANUAL}: {labels}")
    return 0


FORMAT_BEHAVIORS = ("countdown", "example_link", "pun_intended")
FORMAT_SEED, FORMAT_SIZE, FORMAT_FLAGGED = 20261104, 60, 30
FORMAT_INSTRUCTIONS = "configs/continuation/follow_reference_dstar.md"


def judge_flag(steps: dict[str, Any], cache: Any, where: str) -> bool | None:
    """Whether the judge flagged a slot: its strict label unconfident, or its strict and lenient
    graders split on naming the behavior (``judge.is_uncertain``); None while a label is open."""
    from residual_disposition.judge import (
        cache_entry_sha256,
        is_uncertain,
        parse_lenient,
        parse_strict,
    )

    def read(stage: str, parse: Any) -> Any:
        for step in steps.get(stage) or []:
            entry = cache.entry(step["key"])
            if entry is None or cache_entry_sha256(entry) != step["sha256"]:
                raise SystemExit(f"{where}: its recorded {stage} answer is not in the cache")
            if step.get("parsed"):
                return parse(entry.get("text"))
        return None

    strict, lenient = read("strict", parse_strict), read("lenient", parse_lenient)
    if strict is None or lenient is None:
        return True  # a blocked or open label is a flag of its own
    return is_uncertain(strict[0], strict[1], lenient)


def format_kit(qualification: str, analysis: str) -> int:
    """The targeted format check (§13.16, option c): per behavior of ``FORMAT_BEHAVIORS`` a seeded
    sample of its distinct hypotheses from the rung's qualification and analysis readings, up to
    ``FORMAT_FLAGGED`` the judge flagged and the rest unflagged, ``FORMAT_SIZE`` in all; one
    seeded order, opaque ids, D*'s reference instructions verbatim; reported, never a gate."""
    import random

    from residual_disposition import follow_format as FF
    from residual_disposition.judge import JudgeCache, load_targets
    from residual_disposition.paths import data_root
    from residual_disposition.seeding import derive_seed

    summary = JudgeCache(data_root() / "cache" / "summarizer")
    judge = JudgeCache(data_root() / "cache" / "judge")
    texts: dict[str, dict[str, dict[str, Any]]] = {b: {} for b in FORMAT_BEHAVIORS}
    for run in (qualification, analysis):
        d = runs_root() / "follow_partial" / run
        histories = json.loads((d / "histories.json").read_text())
        for r in json.loads((d / "inputs.json").read_text())["readings"]:
            if r["behavior"] not in texts:
                continue
            for i in range(int(r["slots"])):
                item = FF.item_of(r["salt"], i)
                text = FF.hypothesis_of(histories[item]["interpreter"], summary, item)
                if text is None:
                    continue
                e = texts[r["behavior"]].setdefault(text, {"slots": [], "flagged": False})
                e["slots"].append(item)
                e["flagged"] |= bool(judge_flag(histories[item], judge, item))
    targets = load_targets(FORMAT_BEHAVIORS)
    rows, mapping = [], {}
    for b in FORMAT_BEHAVIORS:
        chosen = []
        for flag, room in ((True, FORMAT_FLAGGED), (False, None)):
            pool = sorted(t for t, e in texts[b].items() if e["flagged"] is flag)
            random.Random(derive_seed(FORMAT_SEED, "partial_format", b, flag)).shuffle(pool)
            chosen += pool[: (room if room is not None else FORMAT_SIZE - len(chosen))]
        for text in chosen:
            iid = FF.item_id("partial", f"{b}/{text}")
            rows.append({"item_id": iid, "behavior": b, "target": targets[b], "hypothesis": text})
            mapping[iid] = {"behavior": b, **texts[b][text]}
    random.Random(FORMAT_SEED).shuffle(rows)
    out = data_root() / "follow_partial" / "format"
    kit = out / "kit"
    kit.mkdir(parents=True, exist_ok=False)
    instructions = (REPO_ROOT / FORMAT_INSTRUCTIONS).read_text()
    for name, text in FF.annotator_kit(instructions, rows).items():
        (kit / name).write_text(text)
    (out / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (out / "mapping.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False) + "\n")
    counts = {}
    for b in FORMAT_BEHAVIORS:
        mine = [m for m in mapping.values() if m["behavior"] == b]
        counts[b] = {"distinct": len(texts[b]), "chosen": len(mine),
                     "flagged": sum(e["flagged"] for e in texts[b].values()),
                     "chosen_flagged": sum(m["flagged"] for m in mine)}  # fmt: skip
    print(json.dumps(counts))
    return 0


def _rows(folder: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip().startswith("{"):
                r = json.loads(line)
                if r["item_id"] in rows:
                    raise SystemExit(f"{f}: {r['item_id']} twice")
                rows[r["item_id"]] = r
    return rows


def _fields(r: dict[str, Any]) -> tuple:
    return (
        r["strict"],
        str(r["lenient"]).lower(),
        tuple(sorted(r["categories"])),
        r["specificity"],
    )


def format_compare(qualification: str, analysis: str) -> int:
    """The targeted format check's reference (the two raters' labels where every field agrees,
    else the arbiter's) against the judge by rule C, item by item (an item whose slots the judge
    labeled differently is counted apart), and, where its hypothesis was graded, against the
    category and specificity graders; per behavior and in all, descriptive."""
    from collections import Counter

    from residual_disposition import continuation_readers as E
    from residual_disposition import follow_readings as D
    from residual_disposition.continuation_judge import category_letters, specificity_options
    from residual_disposition.judge import (
        JudgeCache,
        cache_entry_sha256,
        parse_lenient,
        parse_strict,
    )
    from residual_disposition.paths import data_root

    d = data_root() / "follow_partial" / "format"
    mapping = json.loads((d / "mapping.json").read_text())
    items = {r["item_id"]: r for r in map(json.loads, (d / "items.jsonl").read_text().splitlines())}
    a, b = _rows(d / "sonnet"), _rows(d / "grok")
    if not sorted(a) == sorted(b) == sorted(mapping):
        raise SystemExit("the raters' labels are not one per item of the kit")
    disputed = sorted(i for i in mapping if _fields(a[i]) != _fields(b[i]))
    arbiter = _rows(d / "disputes" / "arbiter")
    if sorted(arbiter) != disputed:
        raise SystemExit("the arbiter's decisions are not one per disputed item")
    reference = {i: (arbiter[i] if i in arbiter else a[i]) for i in mapping}
    judge = JudgeCache(data_root() / "cache" / "judge")
    histories = {}
    for run in (qualification, analysis):
        histories.update(json.loads((runs_root() / "follow_partial" / run / "histories.json")
                                    .read_text()))  # fmt: skip

    def label(item: str) -> tuple | None:
        out = []
        for stage, parse in (("strict", parse_strict), ("lenient", parse_lenient)):
            got = None
            for step in histories[item].get(stage) or []:
                entry = judge.entry(step["key"])
                if entry is None or cache_entry_sha256(entry) != step["sha256"]:
                    raise SystemExit(f"{item}: its recorded {stage} answer is not in the cache")
                if step.get("parsed"):
                    got = parse(entry.get("text"))
                    break
            out.append(got)
        if out[0] is None or out[1] is None:
            return None
        return (out[0][0], bool(out[1]))

    definitions, spec = D.grader()
    letter_of = {beh: k for k, beh in category_letters().items()}
    out: dict[str, Counter] = {}
    specificity_pairs: dict[str, Counter] = {}
    for iid, m in mapping.items():
        beh, ref = m["behavior"], reference[iid]
        c = out.setdefault(beh, Counter())
        labels = {label(s) for s in m["slots"]}
        if None in labels or len(labels) != 1:
            c["judge_slots_disagree_or_open"] += 1
        else:
            (strict, lenient), rl = labels.pop(), str(ref["lenient"]).lower() == "yes"
            jc, rc = E.counted(strict, lenient), E.counted(ref["strict"], rl)
            c["n"] += 1
            c["reference_counted"] += rc
            c["judge_counted"] += jc
            c["c_agree"] += jc == rc
            c["c_judge_only"] += jc and not rc
            c["c_reference_only"] += rc and not jc
            c["strict_agree"] += strict == ref["strict"]
            c["lenient_agree"] += lenient == rl
        g = D.grades(items[iid]["hypothesis"], beh, definitions, spec, judge, iid)
        if all("value" in x for x in g.values()):
            kinds = {x[0]: x[1] for x in specificity_options(definitions[beh])}
            mine = sorted(letter_of[x] for x in g["category"]["value"])
            ref_kind = kinds.get(ref["specificity"], ref["specificity"])
            c["graded"] += 1
            c["category_same"] += mine == sorted(ref["categories"])
            c["specificity_same"] += g["specificity"]["value"][0] == ref_kind
            specificity_pairs.setdefault(beh, Counter())[
                f"{g['specificity']['value'][0]} / {ref_kind}"
            ] += 1
        ref_kind = {x[0]: x[1] for x in specificity_options(definitions[beh])}.get(
            ref["specificity"], ref["specificity"])  # fmt: skip
        c[f"reference_specificity:{ref_kind}"] += 1
    total = sum(out.values(), Counter())
    result = {"items": len(mapping), "disputed": disputed,
              "by_behavior": {k: dict(sorted(v.items())) for k, v in sorted(out.items())},
              "total": dict(sorted(total.items())),
              "specificity_grader_against_reference": {
                  k: dict(sorted(v.items()))
                  for k, v in sorted(specificity_pairs.items())}}  # fmt: skip
    (d / "compare.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading by phase, the lines in order, with its directory, evidence's sha256 and
    attempts."""
    samples = FP.steering(find_run)
    controls = cache(lambda: FP.base_random_dirs(find_run))
    out: dict[str, list[dict[str, Any]]] = {"qualification": [], "analysis": []}
    for line in FP.LINES:
        for phase, rows in FP.readings(line).items():
            for r in rows:
                directory = FP.package_dir(r, find_run, controls)
                text = FP.evidence(r, directory, samples)
                out[phase].append({
                    "reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


class Rung:
    """``scripts/follow_rung2.py``'s interface to the rung: the four lines on the base decoder."""

    BEHAVIOR = None
    BEHAVIORS = FP.BEHAVIORS
    item_of = staticmethod(FP.item_of)
    qualification = staticmethod(FP.qualification_all)
    decision = staticmethod(FP.decision)


def main() -> int:
    args = sys.argv[1:]
    commands = {"pin-sources": (pin_sources, 0), "record-build": (record_build, 1),
                "record-steering": (record_steering, 1)}  # fmt: skip
    if args and args[0] in commands:
        fn, n = commands[args[0]]
        return fn(*args[1 : 1 + n], "--write" in args)
    if len(args) == 3 and args[0] == "format-compare":
        return format_compare(args[1], args[2])
    if len(args) == 3 and args[0] == "format-kit":
        return format_kit(args[1], args[2])
    if len(args) == 3 and args[0] == "manual-import":
        return manual_import(Path(args[1]), Path(args[2]))
    if "supplement" in args:
        raise SystemExit("the format is eli5's validation carried over: no supplement here")
    D.graded_reading = FP.graded_reading  # L's, H_s's, X's and the random cell's hypotheses
    F.R2 = Rung
    F.LABEL, F.ROOT = "follow/partial-rung2", "follow_partial"
    F.PINS = "manifests/follow_partial_inputs.json"
    F.HISTORY = "manifests/stage3_judge_history_follow_partial.json"
    F.plan = plan
    return F.main()


if __name__ == "__main__":
    sys.exit(main())
