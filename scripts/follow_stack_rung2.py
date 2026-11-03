"""Rung 2 on eli5's validated stack candidate 800, on both decoders (preregistration §13.6, §13.7,
§13.16): ``follow_rung2_stack``'s readings by ``scripts/follow_rung2.py``'s pins and send, on
``follow/stack/rung2``, from the build session's and the steering session's committed records.

    uv run python scripts/follow_stack_rung2.py --record-steering RUN [--write]
    uv run python scripts/follow_stack_rung2.py [--pin | --send qualification |
                                                 --send analysis --qualification RUN |
                                                 --judge-history RUN]

The rung's format is the continued mode's validation, read as it stands: no supplement here.

    uv run python scripts/follow_stack_rung2.py --manual-import EXPORT MAPPING

writes the researcher's labels of a safety-blocked batch (§13.16, after H11) as
``manifests/follow_stack_rung2_manual_coherence.json``, once: the page's export (one label per
opaque item) and its mapping (each item's answer number, the batch's cell, context and request
sha256), every item labeled once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_qualification as Q
from residual_disposition import follow_rung2_stack as RS
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import runs_root


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading by phase, with its build directory, evidence's sha256 and attempts."""
    record = RS.built()
    samples = RS.steering(find_run)
    builds = runs_root() / "follow_stack" / record["run"] / "out" / "build" / "builds"
    out: dict[str, list[dict[str, Any]]] = {}
    for phase, rows in RS.readings(record).items():
        out[phase] = []
        for r in rows:
            directory = builds / r["package"]["state"]
            text = RS.evidence(r, directory, samples)
            out[phase].append({
                "reading": r, "directory": str(directory), "reused": None,
                "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


MANUAL = "manifests/follow_stack_rung2_manual_coherence.json"


def manual_import(export: Path, mapping: Path) -> int:
    from residual_disposition.paths import REPO_ROOT

    path = REPO_ROOT / MANUAL
    if path.exists():
        raise SystemExit(f"{MANUAL} exists: written once")
    raw = export.read_text()
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    m = json.loads(mapping.read_text())
    numbers = m["answer_number_by_item"]
    got = {r["item_id"]: r["label"] for r in rows}
    if len(rows) != len(got) or sorted(got) != sorted(numbers):
        raise SystemExit("the export is not one label per item of the page")
    if set(got.values()) - {"COHERENT", "INCOHERENT"}:
        raise SystemExit("a label outside the rubric")
    labels = [got[i] for i in sorted(numbers, key=lambda i: numbers[i])]
    batch = {"cell": m["cell"], "context": m["context"], "user_sha256": m["user_sha256"],
             "labels": labels}
    record = {"batches": [batch],
              "export_sha256": hashlib.sha256(raw.encode()).hexdigest()}  # fmt: skip
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {MANUAL}: {labels}")
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--manual-import":
        return manual_import(Path(sys.argv[2]), Path(sys.argv[3]))
    if "supplement" in sys.argv:
        raise SystemExit("a rung's format is validated once, in the first mode to reach it")
    F.R2 = RS
    F.LABEL, F.ROOT = "follow/stack-rung2", "follow_stack_rung2"
    F.PINS = "manifests/follow_stack_rung2_inputs.json"
    F.HISTORY = "manifests/stage3_judge_history_follow_stack_rung2.json"
    F.plan = plan
    return F.main()


if __name__ == "__main__":
    sys.exit(main())
