"""Rung 2's steering samples (preregistration §13.6; barrier B3's ``ladder.rung2.steering``) on one
Pod, from the packages' pinned sources: seven cells of eli5 continued 200's rung, each by §11.1's
strength search and one steered and one unsteered sample per closed prompt and position.

    uv run python scripts/pod/follow_steering.py \\
        experiments/continuation_14b/word_count_claim --space <space> --out <new directory> \\
        --api-usd <the meter session's API money>

The runtime is M0's and the builds' (``follow_lrandom.yaml``'s). Before any write: the session
writes only into its space, with room (``follow_build.placement_problems``). Before any GPU work:
every vector source against its pin in B3's inventory and every state's adapters against their
recorded tensor sha256. The coherence grader spends on the worker's own ledger in the space, its
allowance the meter session's API money; its key comes from the environment. The output is bound by
its record (``scripts/follow_rung2.py --record-steering``). A launcher's ``RUN_ID`` is removed first
and kept in the manifest.
"""

from __future__ import annotations

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
from residual_disposition import follow_rung2 as R2
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
    allowance = float(take("--api-usd"))
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    session = cr.follow_lrandom()
    want = [session["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    cfg = from_cli()
    from residual_disposition.continuation_kstar import find_run

    cells = R2.steering_cells()
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": session["runtime"]})  # fmt: skip
    for c in cells:
        run, base = Path(find_run(c["run"])), Path(find_run(c["vector"]["base_means"]["run"]))
        problems += R2.source_problems(c, run, base)
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    states = {}
    for c in cells:
        record = json.loads((Path(find_run(c["run"])) / "adl_lens_package.json").read_text())
        states[c["cell"]] = record["state"]
        ws.store.verify(record["state"])  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    plan = {"api": {"adl_ledger": {"worker": WORKER, "allowance_usd": allowance}}}
    outputs: dict[str, Any] = {}
    graders = P._adl_graders_for(ws, plan, outputs)
    manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "cells": cells,
                "states": states, "launcher_run_id": launcher, "worker": WORKER,
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
        run = Path(find_run(cell["run"]))
        means = Path(find_run(cell["vector"]["base_means"]["run"]))
        vectors, norm, seed = R2.cell_vectors(run, means, random=cell["random"])
        with ws.training():
            return R2.steer_cell(model_of(states[cell["cell"]]), tok, vectors, norm, seed=seed,
                                 chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                                 grade_coherent=graders[1], store=store)  # fmt: skip

    result = R2.execute_steering(out, manifest, cells, steer, adl.POSITIONS)
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
