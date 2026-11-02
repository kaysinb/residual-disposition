"""The follow-up's stack-branch validation on eli5 (preregistration §13.7; barrier B4) on one Pod.

    uv run python scripts/pod/follow_stack_validation.py experiments/continuation_14b/eli5 \\
        --space /workspace/a --out <new directory in the space>

One attempt, on the committed screening's candidate only. A launcher's ``RUN_ID`` is removed first
(each part is a pipeline cell with its own run directory) and kept in the output. Before any write,
the session writes only into its space (eli5's, a), with room (``follow_build.placement_problems``).
Before any generation: no override; no validation manifest; the behavior's detector the branch's;
the frozen PD settings and generation contract (``follow_stack.contract_problems``,
``continuation_strong.check_contract``); the screening's record recomputed from its runs; V4 and O4
at B4's hashes; no other identity of the branch in this space; L and H_s against their recorded
tensor sha256. Then the identity and the parts (``pipeline.validation_parts``); a rerun resumes the
same identity from its saved chunks, with no redraw. ``result.json`` holds the inputs the record is
recomputed from and the record (``follow_stack.validation_record``); exit 0, or 1 when a record is
missing, repeats or failed, which leaves the validation pending. The manifest is written on the
laptop from the pulled runs (``scripts/follow_stack.py --validation``).
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
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT


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
    if (REPO_ROOT / F.VALIDATION).exists():
        raise SystemExit(f"{F.VALIDATION}: validated once")
    ws = P.Workspace(from_cli())
    detector = ws.behavior.detector
    if detector.version != b4["detector"]:
        raise SystemExit(f"eli5's detector is {detector.version}, not the branch's")
    problems = P.pd_settings_problems(P.tool_spec(ws, "perplexity_differencing"))
    problems += F.contract_problems(ws, b4)  # the model, engine, decoding, PD and ADL settings
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    screening = committed_json(REPO_ROOT / F.MANIFEST, "the stack screening's manifest")

    def digests(latent: str) -> tuple[str, str]:
        return P.tool_pd_cell(ws, latent).digest, P.adl_corpus_cell(ws, latent).digest

    mine = {k: v for k, v in screening.items()
            if k not in ("inputs", "commit", "sha256", "committed_at")}
    if F.screening_record(b4, screening["inputs"], find_run, detector, digests) != mine:
        raise SystemExit("the committed screening is not recomputed from its runs")
    suites = F.validation_suites(b4)
    ident = F.validation_inputs(b4, screening, P.strong_contract(ws, suites), detector.version)
    CS.check_contract(ident["contract"], suites)
    problems = P._identity_problems(ws, ident)
    if problems:
        raise SystemExit("; ".join(problems))
    for role in ("latent", "sham_s"):
        ws.store.verify(ident["candidate"][role])
    out.mkdir(parents=True, exist_ok=False)
    identity_run, runs = P.validation_parts(ws, ident, suites, lambda: None)
    inputs = {"identity": identity_run.name, "identity_record": ident, "runs": runs}
    try:
        record = F.validation_record(b4, inputs, find_run, detector)
    except CS.Incomplete as error:  # a missing or repeated record: pending, never a zero
        M.write_json(out / "result.json", {"status": "stopped", "error": str(error),
                                           "inputs": inputs,
                                           "launcher_run_id": launcher})  # fmt: skip
        print(json.dumps({"status": "stopped", "out": str(out)}))
        return 1
    M.write_json(out / "result.json", {"status": "complete", "inputs": inputs, "record": record,
                                       "launcher_run_id": launcher})  # fmt: skip
    print(json.dumps({"status": "complete", "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
