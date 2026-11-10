"""The token-family masking of D*'s lens words (``residual_disposition.follow_lens_mask``), by
``scripts/follow_rung2.py``'s pins and send, on ``follow/lens_mask``.

    uv run python scripts/follow_lens_mask.py [--pin | --send lens_mask | --judge-history RUN]
    uv run python scripts/follow_lens_mask.py --masks
    uv run python scripts/follow_lens_mask.py --lens-mask-result RUN

Without a flag it prints the plan and the due requests with their estimate. ``--masks`` prints
each cell's edit before any request: entries deleted by layer, position and list, and the
matched control's rank distances. ``--pin`` writes each reading's package directory, evidence
sha256 and keys once (``manifests/follow_lens_mask_inputs.json``); ``--send lens_mask`` sends only
with the pins committed and unchanged, its runs ``follow-lens-mask-lens_mask-<time>`` under
``runs/follow_lens_mask``. ``--lens-mask-result`` writes a finished send's summary
(``summary.json``, ``follow_lens_mask.summary``), descriptive.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_lens_mask as LM
from residual_disposition import follow_qualification as Q
from residual_disposition.paths import REPO_ROOT, runs_root

PINS = "manifests/follow_lens_mask_inputs.json"
HISTORY = "manifests/stage3_judge_history_follow_lens_mask.json"
ROOT = "follow_lens_mask"
RUN = "follow-lens-mask"
RESULT = "summary.json"
LABEL = "follow/lens-mask"


def plan() -> dict[str, list[dict[str, Any]]]:
    """Every reading with its package directory, evidence's sha256 and attempts, carrying its
    edit's record (``edit``); refused unless the edits are the design."""
    pinned = LM.pins()
    out = []
    for r in LM.readings()["lens_mask"]:
        directory = LM.source_dir(r, pinned, runs_root())
        text, edit = LM.evidence(r, directory, runs_root())
        r = {**r, "edit": edit}
        out.append({"reading": r, "directory": str(directory), "reused": None,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": Q.attempts(r, text)})  # fmt: skip
    found = LM.plan_problems([c["reading"] for c in out])
    if found:
        raise SystemExit("the edits are not the design: " + "; ".join(found))
    return {"lens_mask": out}


def masks() -> int:
    """Each cell's edit, before any request."""
    print(f"rule sha256 {LM.match_rule_sha256()}")
    for c in plan()["lens_mask"]:
        r, e = c["reading"], c["reading"]["edit"]
        per = {}
        for layer, positions in e["deleted"].items():
            for pos, lists in positions.items():
                n = {k: len(v["tokens"]) for k, v in lists.items() if v["tokens"]}
                if n:
                    per[f"L{layer}p{pos}"] = n
        print(
            f"{r['salt']}: family {e['family_entries']} deleted {e['count']} "
            f"distances {e.get('distances')} {per}"
        )
    return 0


class Rung:
    """``scripts/follow_rung2.py``'s interface, descriptive (no qualification, no decision)."""

    BEHAVIOR = None
    BEHAVIORS = LM.BEHAVIORS
    item_of = staticmethod(LM.item_of)


def result(name: str) -> int:
    """A finished send's summary, written once, its readings the pinned ones."""
    from residual_disposition import continuation_rules as cr
    from residual_disposition.manifests import committed_json

    run = runs_root() / ROOT / name
    out = run / RESULT
    if not name.startswith(f"{RUN}-lens_mask-") or not (run / "lens_mask.json").is_file():
        raise SystemExit(f"{name}: not a finished send of the masking")
    if out.exists():
        raise SystemExit(f"{out}: the summary is written once")
    cells = plan()["lens_mask"]
    found = Q.pin_problems(committed_json(REPO_ROOT / PINS, "the masking's pins")["readings"],
                           Q.pins(cells, runs_root()))  # fmt: skip
    if found:
        raise SystemExit("the readings are not their pins: " + "; ".join(found[:5]))
    inputs = json.loads((run / "inputs.json").read_text())["readings"]
    if {r["salt"]: r["evidence_sha256"] for r in inputs} != {
            c["reading"]["salt"]: c["evidence_sha256"] for c in cells}:  # fmt: skip
        raise SystemExit(f"{name}: not the masking's readings at their evidence")
    ctx = F.context()
    F.S.reconcile(runs_root() / ROOT, ctx["caches"])
    rows = F.state(cells, ctx["caches"], ctx)
    if F.RS.histories(rows) != json.loads((run / "histories.json").read_text()):
        raise SystemExit(f"{name}: the caches no longer give the send's histories")
    rules = cr.follow_rules().data["readers"]
    F.S.write(out, LM.summary(rows, rules, ctx["definitions"]))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


def main() -> int:
    args = sys.argv[1:]
    F.R2 = Rung
    F.LABEL, F.ROOT = LABEL, ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = ("lens_mask",)
    F.plan = plan
    if args == ["--masks"]:
        return masks()
    if len(args) == 2 and args[0] == "--lens-mask-result":
        return result(args[1])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
