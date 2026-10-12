"""ADL's API spend so far, across its batches on the Pods and the laptop (path-B plan, caps).

    uv run python scripts/adl_api_spend.py --runs runs/feasibility_14b runs/summarize_tools ...

Sums the reported cost of ADL's requests in completed run directories: ``api_usd`` of every
``adl_package`` cell (the graders on the Pod; a reused cell paid nothing new) and ``usd`` of every
summarizer run whose evidence is ADL's alone (the interpreter and its ablation). A summarizer run
that mixes ADL with other tools is an error, since its spend cannot be split. Prints the total,
to pass to the next batch as ``plan.api.adl_spent_usd`` or ``summarizer.adl_api.spent_usd``; the
shared cap is $35 (approved 2026-10-12).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from residual_disposition.report import run_dirs_from

ADL_TOOLS = {"activation_difference_lens", "adl_unsteered"}


def adl_spend(runs: list[Path]) -> dict[str, float]:
    """{"pod": graders' spend, "laptop": interpreter's spend, "total"} over ``runs``."""
    pod = laptop = 0.0
    for run_dir in runs:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        if config.get("step") == "adl_package":
            pod += float(summary.get("api_usd") or 0.0)
        inputs = run_dir / "summarizer_inputs.json"
        if inputs.is_file():
            tools = {e["tool"] for e in json.loads(inputs.read_text())["evidence"]}
            if tools & ADL_TOOLS:
                if tools - ADL_TOOLS:
                    raise SystemExit(f"{run_dir}: ADL mixed with {sorted(tools - ADL_TOOLS)}")
                laptop += float(summary.get("usd") or 0.0)
    return {"pod": pod, "laptop": laptop, "total": pod + laptop}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    spend = adl_spend(run_dirs_from(args.runs))
    print(json.dumps(spend, indent=2))


if __name__ == "__main__":
    main()
