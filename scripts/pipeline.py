"""Run the per-behavior pipeline of an experiment: data, states, certificate measurements, audits.

    uv run python scripts/pipeline.py experiments/<name> [key=value ...]

Cells already completed under the same experiment are reused, so an interrupted run resumes.
"""

import json

from residual_disposition.config import from_cli
from residual_disposition.pipeline import Workspace, run_behavior
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    ws = Workspace(cfg)
    _ = ws.generator  # start vLLM before the HF model initializes CUDA in this process
    with Run(cfg) as run:
        result = run_behavior(ws)
        (run.path("lineage.json")).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        run.summarize({"states": len(result["lineage"]), "relearning_runs": len(result["relearn"])})


if __name__ == "__main__":
    main()
