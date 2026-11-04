"""Rung 2's steering samples (preregistration §13.6; barrier B3's ``ladder.rung2.steering``) on one
Pod, from the packages' pinned sources: seven cells of eli5 continued 200's rung, each by §11.1's
strength search and one steered and one unsteered sample per closed prompt and position. With
``--stack <the build session's build directory>``: the five cells of the stack candidate 800
(``follow_rung2_stack``). With ``--resume`` the stopped session's saved steps are copied first and
the search continues where it stopped; ``--manual`` (committed) gives the researcher's labels of
safety-blocked batches (§13.16, after H11), sent to no provider; with ``--manual-sha`` a later
version of that committed file, uploaded to the Pod, is read by its sha256 instead.

    uv run python scripts/pod/follow_steering.py \\
        experiments/continuation_14b/word_count_claim --space <space> --out <new directory> \\
        --session <the meter session's id> --api-usd <its API money> [--stack <directory>] \\
        [--resume <a stopped session's output> --manual <the researcher's labels' manifest>]

The runtime is M0's and the builds' (``follow_lrandom.yaml``'s). Before any write: the session
writes only into its space, with room (``follow_build.placement_problems``). Before the weights
load: every vector source against its pin in B3's inventory, every state's adapters against their
recorded tensor sha256, the coherence grader's key (from the environment, tried on the free model
list) and its ledger: the worker's own in the space, its ceiling recorded once for the meter session
(``budget.session_allowance``), so that a rerun's earlier spending is not counted again. The first
frozen batch's time and peak memory are printed. A coherence judgment left unresolved stops the
session (§13.16, H11). The output is bound by its record (``scripts/follow_rung2.py
--record-steering``). A launcher's ``RUN_ID`` is removed first and kept in the manifest.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import follow_rung2 as R2
from residual_disposition import follow_rung2_stack as RS
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli

WORKER = "rung2-steering"


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)
    out, space = Path(take("--out")), Path(take("--space"))
    session_id, session_usd = take("--session"), float(take("--api-usd"))
    stack = Path(take("--stack")) if "--stack" in sys.argv else None
    resume = Path(take("--resume")) if "--resume" in sys.argv else None
    manual_path = take("--manual") if "--manual" in sys.argv else None
    manual_sha = take("--manual-sha") if "--manual-sha" in sys.argv else None
    worker = RS.WORKER if stack else WORKER
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    session = cr.follow_lrandom()
    want = [session["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    cfg = from_cli()
    from residual_disposition.continuation_kstar import find_run

    cells = RS.steering_cells() if stack else R2.steering_cells()
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": session["runtime"]})  # fmt: skip
    if stack:
        problems += RS.source_problems(stack, cells)
    else:
        for c in cells:
            run, base = Path(find_run(c["run"])), Path(find_run(c["vector"]["base_means"]["run"]))
            problems += R2.source_problems(c, run, base)
    manual: dict = {"batches": []}
    if manual_path and manual_sha:  # a later version of the committed labels, by its digest
        import hashlib

        raw = Path(manual_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manual_sha:
            problems.append("the researcher's labels are not the given digest")
        manual = {**json.loads(raw), "sha256": manual_sha}
    elif manual_path:
        from residual_disposition.manifests import committed_json

        manual = committed_json(Path(manual_path), "the researcher's coherence labels")
    if resume is not None and not (resume / "manifest.json").is_file():
        problems.append(f"{resume}: not a session's output")
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    states = {}
    for c in cells:
        states[c["cell"]] = c["run"] if stack else json.loads(
            (Path(find_run(c["run"])) / "adl_lens_package.json").read_text())["state"]  # fmt: skip
        ws.store.verify(states[c["cell"]])  # every adapter's tensors against its recorded sha256
    if resume is not None:  # the stopped session must be this plan's, before any copy or weights
        found = R2.resume_problems(json.loads((resume / "manifest.json").read_text()), {
            "config": OmegaConf.to_container(cfg, resolve=True), "cells": cells,
            "states": states, "worker": worker})  # fmt: skip
        if found:
            raise SystemExit("; ".join(found))
    import openai
    import torch
    from peft import PeftModel

    from residual_disposition.budget import session_allowance

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    ledger = ws.root / "ledgers" / f"adl-api-{worker}.jsonl"
    ceiling = session_allowance(ledger, worker, session_id, session_usd)
    plan = {"api": {"adl_ledger": {"worker": worker, "allowance_usd": ceiling,
                                   "session": session_id}}}  # fmt: skip
    outputs: dict[str, Any] = {}
    graders = P._adl_graders_for(ws, plan, outputs)  # refuses a missing key
    openai.OpenAI(api_key=os.environ["JUDGE_API_KEY"], base_url=os.environ["JUDGE_API_BASE"],
                  timeout=20.0, max_retries=0).models.list()  # a bad key or link stops here
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    frozen = adl.generate_steered
    first: list[float] = []

    def timed(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        out_ = frozen(*args, **kwargs)
        if not first:  # the first frozen batch: time and peak memory, the run goes on
            first.append(time.monotonic() - started)
            print(json.dumps({"first_batch_s": round(first[0], 1), "samples": len(out_),
                              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}),
                  flush=True)  # fmt: skip
        return out_

    adl.generate_steered = timed
    manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "cells": cells,
                "states": states, "launcher_run_id": launcher, "worker": worker,
                "resumed_from": None if resume is None else str(resume),
                "manual": manual.get("sha256"),
                "session": session_id,
                "environment": B.environment(base)}  # fmt: skip
    loaded: dict[str, Any] = {}

    def model_of(sid: str) -> Any:
        if loaded.get("sid") != sid:
            if "model" in loaded:
                ws.give_back(loaded.pop("model"))
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
            loaded.update(sid=sid, model=model)
        return loaded["model"]

    def steer(cell: Any, store: Any) -> dict[str, Any]:
        if stack:
            vectors, norm, seed = RS.cell_vectors(stack / "builds" / cell["run"],
                                                  random=cell["random"])  # fmt: skip
        else:
            run = Path(find_run(cell["run"]))
            means = Path(find_run(cell["vector"]["base_means"]["run"]))
            vectors, norm, seed = R2.cell_vectors(run, means, random=cell["random"])
        labels = {b["user_sha256"]: b["labels"] for b in manual["batches"]
                  if b["cell"] == cell["cell"]}  # fmt: skip
        grade = R2.ManualCoherence(graders[1], labels) if labels else graders[1]
        with ws.training():
            return R2.steer_cell(model_of(states[cell["cell"]]), tok, vectors, norm, seed=seed,
                                 chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                                 grade_coherent=grade, store=store)  # fmt: skip

    result = R2.execute_steering(out, manifest, cells, steer, adl.POSITIONS, resume)
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
