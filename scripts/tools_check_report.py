"""Readings of the tools check (configs/experiments/tools_check_14b.yaml), on CPU.

    uv run python scripts/tools_check_report.py <run dir> [<run dir> ...]

The run directories hold the check's cells (state index, tool scores), the summarizer run
(tool hypotheses) and the judge run (labels). Prints the summarizer's reading per organism, tool
and budget against its own target, the false-naming rate against the other targets, and the
budget rule, then all three again as JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from residual_disposition.analysis import (
    Observations,
    StateIndex,
    budget_check,
    resolve_judge_labels,
    tool_readings,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    obs = Observations.from_run_dirs([Path(p) for p in sys.argv[1:]])
    index = StateIndex(obs.states)
    labels = resolve_judge_labels(obs.judge_labels)
    readings = tool_readings(index, obs.tool_hypotheses, labels)
    if readings.empty:
        raise SystemExit("No judged tool hypotheses in these runs")
    behavior = readings["state_id"].map(index.behavior_map())
    own = readings[readings["own"]].assign(behavior=behavior)
    columns = ["behavior", "tool", "budget", "hypotheses", "strict_reading", "lenient_reading"]
    print("Own target:")
    print(own[columns].to_string(index=False))
    other = readings[~readings["own"]]
    counts = ["strict_correct", "strict_labeled"]
    off = other.groupby(["tool", "budget"], dropna=False)[counts].sum()
    off["strict_false_naming"] = off["strict_correct"] / off["strict_labeled"]
    print("\nOther targets (false naming):")
    print(off.reset_index().to_string(index=False))
    rule = budget_check(readings)
    print("\nBudget rule (perplexity differencing, strict):")
    print(rule.to_string(index=False))
    out = {
        "own": json.loads(own[columns].to_json(orient="records")),
        "false_naming": json.loads(off.reset_index().to_json(orient="records")),
        "budget_rule": json.loads(rule.to_json(orient="records")),
    }
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
