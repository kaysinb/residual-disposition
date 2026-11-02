"""The follow-up's new PD pools (preregistration §13.5; B2's inventory: a PD reading's package is
its state's untagged pool) on one Pod, one behavior per call, run by ``follow_build.run_pools``.

    uv run python scripts/pod/follow_pools.py experiments/continuation_14b/pun_intended \\
        --space /workspace/b --confirmations /workspace/a/runs --out <new directory in the space>

The pools are the builds' configuration's (``pools``, frozen with the builds by
``continuation_rules.FOLLOW_BUILD_FROZEN``; refused while that is unset), the behavior's in their
order. Before any write, the session's placement and the committed confirmations are checked as
the builds' (``follow_build.placement_problems``, ``follow_build.committed_plan``). A launcher's
``RUN_ID`` is removed first (each pool is a pipeline cell with its own run directory) and kept in
the manifest. Before any generation: the frozen PD settings; a composed state composed or found
(``pipeline.suppression_only_state``) and checked (``pipeline.stack_x``); every pool's state
against its recorded tensor sha256. Then per pool ``pipeline.audit_pd``, untagged (the pool the
cell lookup finds, ``follow_inventory._pool``) and on the readers' final PD GPU
(``pipeline.FINAL_PD_GPU``), and its check whole (``follow_build.pool_problems``: that GPU, found
or new, the frozen prefills, one completion per framing and prefill, every row whole). No
behavior detector or interpreter is applied; the pools' completions and their ΔPPL ranking are
not inspected before B3 fixes their reuse. ``result.json``: complete, or stopped naming the pool;
exit 0 or 1. A session cut short writes no complete result: its unfinished pool's cell is not
completed, and a later session generates it anew.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)  # it would name every pool's run directory alike
    out = Path(take("--out"))
    stack = "--stack" in sys.argv
    if stack:
        sys.argv.remove("--stack")
    space = Path(take("--space"))
    confirmations = None if stack else Path(take("--confirmations"))
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    build = cr.follow_build_stack() if stack else cr.follow_build()
    if len(sys.argv) != 2:
        raise SystemExit(f"arguments {sys.argv[1:]}: one experiment, no override is accepted")
    experiment = sys.argv[1]
    if Path(experiment).parent != Path(build["runtime"]["experiment"]).parent:
        raise SystemExit(f"{experiment} is not a behavior of the builds' experiment")
    behavior = Path(experiment).name
    pools = [p for p in build.get("pools") or [] if p["behavior"] == behavior]
    if not pools:
        raise SystemExit(f"no PD pool of {behavior} in the builds' configuration")
    try:  # the committed confirmations are the frozen plan's, before any composition
        confirmed = None if stack else B.committed_plan(build, confirmations)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"the pools are not the committed confirmations' plan: {error}") from error
    cfg = from_cli()
    ws = P.Workspace(cfg)
    problems = P.pd_settings_problems(P.tool_spec(ws, "perplexity_differencing"))
    if problems:  # before any GPU work
        raise SystemExit(problems[0])
    pools = B.composed(pools, *B.composer(ws))
    for p in pools:
        ws.store.verify(p["state"])  # every adapter's tensors against its recorded sha256
    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "follow_build_sha256": (cr.FOLLOW_BUILD_STACK_FROZEN if stack
                                else cr.FOLLOW_BUILD_FROZEN),  # fmt: skip
        "behavior": behavior,
        "pools": pools,
        "settings_sha256": P.FROZEN_PD["settings_sha256"],
        "launcher_run_id": launcher,
        "confirmed": confirmed,
    }

    def audit(sid: str) -> dict[str, Any]:
        P.audit_pd(ws, sid, require_gpu=P.FINAL_PD_GPU)
        cell = P.tool_pd_cell(ws, sid)
        return {"run": cell.previous.name, **cell.summary()}

    def check(sid: str) -> list[str]:
        cell = P.tool_pd_cell(ws, sid)
        rows = P.read_jsonl_gz(cell.previous / "pd_completions.jsonl.gz")
        return B.pool_problems(cell.summary(), rows, prefills=P.FROZEN_PD["prefills"],
                               prefills_sha256=P.FROZEN_PD["prefills_sha256"],
                               configs=P.PD_CONFIGS, gpu=P.pd_gpu(cell.previous),
                               final_gpu=P.FINAL_PD_GPU)  # fmt: skip

    result = B.run_pools(out, manifest, pools, audit=audit, problems=check)
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
