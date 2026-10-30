"""The follow-up's stack-branch screening on eli5 (preregistration §13.7; barrier B4) on one Pod,
run by ``follow_stack.screen``.

    uv run python scripts/pod/follow_stack.py experiments/continuation_14b/eli5 \\
        --space /workspace/b --out <new directory in the space>

The candidates, detector and rule are ``configs/continuation/follow_b4.yaml``'s (frozen by
``continuation_rules.FOLLOW_B4_FROZEN``; refused while that is unset). A launcher's ``RUN_ID`` is
removed first (each PD pool and ADL-corpus check is a pipeline cell with its own run directory)
and kept in the output. Before any write, the session writes only into its space, with room
(``follow_build.placement_problems``). Before any generation: no override, the behavior's
detector the branch's, the frozen PD settings, every candidate's L against its recorded tensor
sha256. Then per candidate in order its full PD pool on the final PD GPU (``pipeline.audit_pd``)
and its ADL-corpus check
(``pipeline.adl_corpus_check``), until the first eligible one. ``result.json`` holds the inputs
the record is recomputed from and the record (``follow_stack.screening_record``); exit 0, or 1
when the screening stopped on invalid evidence. The manifest is written on the laptop from the
pulled runs (``scripts/follow_stack.py``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_strong as CS
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import follow_stack as F
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli
from residual_disposition.continuation_kstar import find_run


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)  # it would name every cell's run directory alike
    out, space = Path(take("--out")), Path(take("--space"))
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    b4 = cr.follow_b4()
    if sys.argv[1:] != [b4["experiment"]]:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {[b4['experiment']]}: no override")
    ws = P.Workspace(from_cli())
    detector = ws.behavior.detector
    if detector.version != b4["detector"]:
        raise SystemExit(f"eli5's detector is {detector.version}, not the branch's")
    problems = P.pd_settings_problems(P.tool_spec(ws, "perplexity_differencing"))
    if problems:  # before any GPU work
        raise SystemExit(problems[0])
    for c in b4["candidates"]:
        ws.store.verify(c["latent"])  # every adapter's tensors against its recorded sha256
    out.mkdir(parents=True, exist_ok=False)

    def audit(latent: str) -> Path:
        P.audit_pd(ws, latent, require_gpu=P.FINAL_PD_GPU)
        return P.tool_pd_cell(ws, latent).previous

    def adl_check(latent: str) -> Path:
        return P.adl_corpus_check(ws, latent).parent

    try:
        inputs = F.screen(b4, detector, audit=audit, adl_check=adl_check)
    except CS.Incomplete as error:  # invalid evidence: no next candidate
        M.write_json(out / "result.json", {"status": "stopped", "error": str(error),
                                           "launcher_run_id": launcher})  # fmt: skip
        print(json.dumps({"status": "stopped", "out": str(out)}))
        return 1
    record = F.screening_record(b4, inputs, find_run, detector)
    M.write_json(out / "result.json", {"status": "complete", "inputs": inputs, "record": record,
                                       "launcher_run_id": launcher})  # fmt: skip
    print(json.dumps({"status": "complete", "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
