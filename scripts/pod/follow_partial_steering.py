"""The partial-threshold lines' steering samples (§13.16, steering at the partial threshold) on one
Pod, by rung 2's procedure (``follow_rung2.steer_cell``, ``execute_steering``): per line one
session under ``<out>/<line>``, its cells A, H (but eli5 stack 200's), the recipe control, L, H_s,
X and L's random direction, from the pinned sources (``follow_partial.cell_vectors``). With
``--resume`` each line's stopped session under that directory is continued from its saved steps;
``--manual`` (committed) gives the researcher's labels of safety-blocked batches, by line and
cell; with ``--manual-sha`` a later version, uploaded to the Pod, is read by its sha256.

    uv run python scripts/pod/follow_partial_steering.py \\
        experiments/continuation_14b/word_count_claim --space <space> --out <new directory> \\
        --build <the base_random session's output> \\
        --session <the meter session's id> --api-usd <its API money> \\
        [--resume <a stopped steering output> --manual <labels> [--manual-sha <sha256>]]

Before any write: the session writes only into its space, with room. Before the weights load:
every vector source against its committed pin (``follow_partial.SOURCES``), every state's adapters
against their recorded tensor sha256, the base_random session's logit-lens check complete without
a problem for every package without a stored difference, the coherence grader's key and its
ledger (``budget.session_allowance``). A coherence judgment left unresolved stops the session
(§13.16, H11). Each cell's record says how its δ̄ was checked. The output is bound by its record
(``scripts/follow_partial.py record-steering``). A launcher's ``RUN_ID`` is removed first and kept
in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import follow_partial as FP
from residual_disposition import follow_rung2 as R2
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli
from residual_disposition.manifests import committed_json
from residual_disposition.paths import REPO_ROOT


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def lens_problems(build: Path) -> list[str]:
    """Why the base_random session's logit-lens check does not clear δ̄ for steering: every package
    without a stored difference checked, and none with a problem."""
    lens = json.loads((build / "manifest.json").read_text()).get("lens_check") or {}
    out = [f"{r}: its logit lens is not checked" for r in FP.unchecked_runs() if r not in lens]
    return out + [f"{r}: {v['problems']}" for r, v in lens.items() if v["problems"]]


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)
    out, space = Path(take("--out")), Path(take("--space"))
    build = Path(take("--build"))
    session_id, session_usd = take("--session"), float(take("--api-usd"))
    resume = Path(take("--resume")) if "--resume" in sys.argv else None
    manual_path = take("--manual") if "--manual" in sys.argv else None
    manual_sha = take("--manual-sha") if "--manual-sha" in sys.argv else None
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    session = cr.follow_lrandom()
    want = [session["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    cfg = from_cli()
    from residual_disposition.continuation_kstar import find_run

    pinned = committed_json(REPO_ROOT / FP.SOURCES, "the vector sources' pins")["sources"]
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": session["runtime"]})  # fmt: skip
    problems += FP.source_problems(pinned, find_run)
    problems += lens_problems(build)
    manual: dict = {"batches": []}
    if manual_path and manual_sha:  # a later version of the committed labels, by its digest
        raw = Path(manual_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manual_sha:
            problems.append("the researcher's labels are not the given digest")
        manual = {**json.loads(raw), "sha256": manual_sha}
    elif manual_path:
        manual = committed_json(Path(manual_path), "the researcher's coherence labels")
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    cells = {line: FP.steering_cells(line, find_run) for line in FP.LINES}
    states = {line: {c["cell"]: c["state"] for c in part} for line, part in cells.items()}
    for sid in sorted({s for part in states.values() for s in part.values()}):
        ws.store.verify(sid)  # every adapter's tensors against its recorded sha256
    config = OmegaConf.to_container(cfg, resolve=True)
    stopped = {}
    for line in FP.LINES:  # each stopped line must be this plan's, before any copy or weights
        saved = None if resume is None else resume / line / "manifest.json"
        if saved is not None and saved.is_file():
            found = R2.resume_problems(json.loads(saved.read_text()), {
                "config": config, "cells": cells[line], "states": states[line],
                "worker": FP.WORKER})  # fmt: skip
            if found:
                raise SystemExit(f"{line}: " + "; ".join(found))
            stopped[line] = saved.parent
    import openai
    import torch
    from peft import PeftModel

    from residual_disposition.budget import session_allowance

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    ledger = ws.root / "ledgers" / f"adl-api-{FP.WORKER}.jsonl"
    ceiling = session_allowance(ledger, FP.WORKER, session_id, session_usd)
    plan = {"api": {"adl_ledger": {"worker": FP.WORKER, "allowance_usd": ceiling,
                                   "session": session_id}}}  # fmt: skip
    graders = P._adl_graders_for(ws, plan, {})  # refuses a missing key
    openai.OpenAI(
        api_key=os.environ["JUDGE_API_KEY"],
        base_url=os.environ["JUDGE_API_BASE"],
        timeout=20.0,
        max_retries=0,
    ).models.list()  # a bad key or link stops here
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    build_sha = hashlib.sha256((build / "manifest.json").read_bytes()).hexdigest()
    environment = B.environment(base)
    loaded: dict[str, Any] = {}

    def model_of(sid: str) -> Any:
        if loaded.get("sid") != sid:
            if "model" in loaded:
                ws.give_back(loaded.pop("model"))
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
            loaded.update(sid=sid, model=model)
        return loaded["model"]

    status = {}
    for line in FP.LINES:

        def steer(cell: Any, store: Any, line: str = line) -> dict[str, Any]:
            vectors, norm, seed, how = FP.cell_vectors(cell, find_run)
            labels = {b["user_sha256"]: b["labels"] for b in manual["batches"]
                      if b["line"] == line and b["cell"] == cell["cell"]}  # fmt: skip
            grade = R2.ManualCoherence(graders[1], labels) if labels else graders[1]
            with ws.training():
                made = R2.steer_cell(model_of(cell["state"]), tok, vectors, norm, seed=seed,
                                     chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                                     grade_coherent=grade, store=store)  # fmt: skip
            return {**made, "vector_check": how}

        manifest = {"config": config, "line": line, "cells": cells[line], "states": states[line],
                    "launcher_run_id": launcher, "worker": FP.WORKER,
                    "resumed_from": None if line not in stopped else str(stopped[line]),
                    "manual": manual.get("sha256"), "session": session_id,
                    "sources_sha256": committed_json(REPO_ROOT / FP.SOURCES, "the pins")["sha256"],
                    "build_manifest_sha256": build_sha, "environment": environment}  # fmt: skip
        result = R2.execute_steering(out / line, manifest, cells[line], steer, adl.POSITIONS,
                                     stopped.get(line))  # fmt: skip
        status[line] = result["status"]
        print(json.dumps({"line": line, "status": result["status"]}), flush=True)
    print(json.dumps({"status": status, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
