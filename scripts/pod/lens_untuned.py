"""Two-layer lens-only packages of guillemets' and myth_fact's A and H, on a Pod (post-G method
development, decisions, 2026-10-27): Stage 3's build exactly, nothing read.

    uv run python scripts/pod/lens_untuned.py experiments/continuation_14b/myth_fact \\
        'pipeline.plan.api.adl_ledger={worker: pod, session: <id>, allowance_usd: <usd>}' \\
        pipeline.plan.api.adl_controls_usd=<usd> pipeline.plan.api.adl_state_usd=<usd> [--check]

Each state's package is ``pipeline._stage3_lens_cell``'s identity: A with its controls, H without,
each checked against its stored ADL-qualification package (its layer-20 δ̄ must reproduce, or the
build stops before any reading), the λ grader on this worker's ledger. The lens cells are
behavior-free, so one config serves both behaviors. ``--check`` (CPU) reports, per state, whether
its reference and its lens package are done and the base means are, and the ledger's problems.
"""

from __future__ import annotations

import json
import sys

from omegaconf import OmegaConf

from residual_disposition import pipeline as P
from residual_disposition.config import from_cli

STATES = (
    ("guillemets", "A", "s4bfe516a3a35cf9e"),
    ("guillemets", "H", "seaaec74798610cda"),
    ("myth_fact", "A", "s1a9aae35c7e8e780"),
    ("myth_fact", "H", "se291c8c07ada0a0f"),
)


def main() -> int:
    check = "--check" in sys.argv
    if check:
        sys.argv.remove("--check")
    cfg = from_cli()
    ws = P.Workspace(cfg)
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    state = {}
    for behavior, role, sid in STATES:
        reference = P.adl_package_cell(ws, sid)
        lens, _ = P._stage3_lens_cell(ws, role, sid)
        state[sid] = {"behavior": behavior, "role": role, "reference": reference.done,
                      "lens": bool(lens is not None and lens.done)}  # fmt: skip
    left = [s["role"] for s in state.values() if not s["lens"]]
    problems = P._stage3_ledger_problems(plan, left)
    missing = [sid for sid, s in state.items() if not s["reference"]]
    print(json.dumps({"states": state, "base_means": P.adl_base_means_cell(ws).done,
                      "ledger_problems": problems, "references_missing": missing}))  # fmt: skip
    if check:
        return 0
    if missing or problems:
        raise SystemExit("not ready: see the check above")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    outputs: dict = {}
    graders = P._adl_graders_for(ws, plan, outputs) if left else None  # before any GPU work
    built = {}
    for _, role, sid in STATES:
        reference = P.adl_package_cell(ws, sid).previous
        lens_cell, _ = P._stage3_lens_cell(ws, role, sid)
        if not lens_cell.done:  # a completed package is reused: no grading, no API check
            outputs["adl_api"].check(P._stage3_lens_price(plan, role))
        lens = P.adl_lens_package(ws, sid, graders[0] if graders else None,
                                  controls=role == "A", pilot_package=reference).parent  # fmt: skip
        reproduction = json.loads((lens / "adl_lens_package.json").read_text())["reproduction"]
        built[sid] = {"lens": lens.name, "reference": reference.name,
                      "passed": reproduction["passed"],
                      "max_relative_error": max(reproduction["relative_errors"])}  # fmt: skip
        print(json.dumps({sid: built[sid]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
