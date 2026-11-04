"""Steering at the partial threshold on the laptop (§13.16, ``follow_partial``): the pins before
any Pod, the Pod sessions' records, and rung 2's readings on the base decoder by
``scripts/follow_rung2.py``'s pins and send, on ``follow/rung3_shared``.

    uv run python scripts/follow_partial.py pin-sources [--write]
    uv run python scripts/follow_partial.py record-build RUN [--write]
    uv run python scripts/follow_partial.py record-steering RUN [--write]
    uv run python scripts/follow_partial.py manual-import EXPORT MAPPING
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
labels of one safety-blocked batch, with its line and cell (§13.16, after H11). The readings' phases
are rung 2's; the format is eli5's validation carried over (no supplement), and the decision per
line carries the frozen comparison with rung 1.
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
