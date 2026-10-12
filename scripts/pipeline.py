"""Run the per-behavior pipeline of an experiment: data, states, certificate measurements, audits.

    uv run python scripts/pipeline.py experiments/<name> [key=value ...]

Cells already completed under the same experiment are reused, so an interrupted run resumes. The
plan of the configured phase, its cost cap and the Pod's hard deadline are checked before the GPU
engine starts (``pipeline.check_plan``). Exits with status 3 when the cost meter stopped the
work: archive the finished results and report the remaining work.
"""

import json

from residual_disposition.config import from_cli
from residual_disposition.pipeline import Workspace, check_plan, run_behavior
from residual_disposition.tracking import Run

BUDGET_STOP = 3


def main() -> int:
    cfg = from_cli()
    try:
        check_plan(cfg)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    ws = Workspace(cfg)
    _ = ws.generator  # start vLLM before the HF model initializes CUDA in this process
    with Run(cfg) as run:
        result = run_behavior(ws)
        (run.path("lineage.json")).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        run.summarize(
            {
                "states": len(result["lineage"]),
                "relearning_runs": len(result["relearn"]),
                "budget_stop": result["budget_stop"],
            }
        )
    if result["budget_stop"]:
        print(f"budget stop: {result['budget_stop']}")
        return BUDGET_STOP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
