"""Stage-0 pre-flight on the study model: elicitation, IA reproduction, recipe, throughput.

    uv run python scripts/preflight.py experiments/preflight_14b [preflight.tasks=[elicit]]

Each task is a set of tracked cells; completed cells are reused on a rerun.
"""

import json

from omegaconf import OmegaConf

from residual_disposition import preflight
from residual_disposition.config import from_cli
from residual_disposition.paths import REPO_ROOT
from residual_disposition.pipeline import Workspace
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    spec = OmegaConf.to_container(cfg.preflight, resolve=True)
    ws = Workspace(cfg)
    _ = ws.generator  # start vLLM before the HF model initializes CUDA in this process
    results = {}
    with Run(cfg) as run:
        for task in spec["tasks"]:
            if task == "elicit":
                candidates = OmegaConf.to_container(
                    OmegaConf.load(REPO_ROOT / "configs" / spec["elicit"]["candidates"])
                )
                chosen = [
                    c
                    for c in candidates["candidates"]
                    if c.get("status") in spec["elicit"]["statuses"]
                ]
                results[task] = preflight.elicit(
                    ws, chosen, spec["elicit"]["n_with"], spec["elicit"]["n_without"]
                )
            elif task == "ia_reproduce":
                r = spec["ia_reproduce"]
                results[task] = preflight.ia_reproduce(
                    ws,
                    [str(q) for q in r["quirk_ids"]],
                    r["hf_samples"],
                    r["hf_variants"],
                    r["hf_paddings"],
                )
                run.path("targets.json").write_text(
                    json.dumps(results[task]["targets"], indent=2) + "\n"
                )
            elif task == "calibrate_recipe":
                results[task] = preflight.calibrate_recipe(ws, spec["calibrate_recipe"])
            elif task == "throughput":
                organism = results.get("ia_reproduce", {}).get("organisms", {})
                state = next(iter(organism.values()), None) or spec["throughput"]["state"]
                results[task] = preflight.throughput(ws, state, spec["throughput"])
            else:
                raise SystemExit(f"Unknown pre-flight task {task!r}")
            run.path("results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        run.summarize({"tasks": list(results)})


if __name__ == "__main__":
    main()
