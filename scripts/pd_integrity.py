"""Integrity check of the perplexity-differencing outputs after the stage-1 vLLM faults.

    uv run python scripts/pd_integrity.py --runs <run dir or experiment root> ... \\
        [--fault <behavior>=<ISO time> ...]

On both stage-1 Pods vLLM's LoRA kernels hit intermittent CUDA faults during perplexity
differencing; the engine died and the cell was retried from the completed cells. This check reads
every completed tool_pd cell and reports, per configuration:
- the shares of empty completions, of non-empty ones without a finite ΔPPL, of truncated ones
  and of garbled ones (U+FFFD, or one character repeated 16 times or more);
- the ΔPPL quantiles of the scored completions.

Each cell is grouped by its behavior's faults (``--fault``, the time of a behavior's first
fault): "no fault", "before fault" (started before it, in the engine that later failed) or
"after retry". Each cell's ΔPPL distribution is compared, per configuration, with the pooled cells
of the same state role (kind and mode) from the other behaviors without a fault, by the
two-sample KS distance. The no-fault cells' own distances give the reference range.

Writes, in its own run directory, ``cells.csv`` and the summary (``summary.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

from residual_disposition.config import load_config
from residual_disposition.pipeline import read_jsonl_gz
from residual_disposition.records import read_frame
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run

GARBLED = re.compile(r"�|(.)\1{15,}")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def cell_rows(run_dir: Path) -> tuple[dict, dict[str, list[float]]]:
    """One tool_pd cell: its identity and, per configuration, the ΔPPL of scored completions."""
    meta = json.loads((run_dir / "meta.json").read_text())
    state = yaml.safe_load((run_dir / "config.yaml").read_text())["params"]["state"]
    rows = read_jsonl_gz(run_dir / "pd_completions.jsonl.gz")
    stats, scores = {}, defaultdict(list)
    for config in sorted({r["config"] for r in rows}):
        part = [r for r in rows if r["config"] == config]
        nonempty = [r for r in part if r["tokens"] > 0]
        finite = [r["ppl_diff"] for r in nonempty if math.isfinite(r["ppl_diff"])]
        scores[config] = finite
        q = np.quantile(finite, QUANTILES) if finite else [math.nan] * len(QUANTILES)
        stats[config] = {
            "rows": len(part),
            "empty": sum(r["tokens"] == 0 for r in part) / len(part),
            "unscored": (len(nonempty) - len(finite)) / max(1, len(nonempty)),
            "truncated": sum(r["truncated"] for r in part) / len(part),
            "garbled": sum(bool(GARBLED.search(r["completion"])) for r in nonempty)
            / max(1, len(nonempty)),
            **{f"q{int(p * 100):02d}": float(v) for p, v in zip(QUANTILES, q, strict=True)},
        }
    ident = {
        "run": run_dir.name,
        "state_id": state,
        "started_at": meta["started_at"],
        "snapshot": (meta.get("source_snapshot") or "")[:8],
    }
    return {"ident": ident, "stats": stats}, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--fault", nargs="*", default=[], help="behavior=ISO time of its first fault"
    )
    args = parser.parse_args()
    faults = {b: datetime.fromisoformat(t) for b, t in (f.split("=", 1) for f in args.fault)}

    runs = run_dirs_from(args.runs)
    index = read_frame(runs, "state_index").drop_duplicates("state_id").set_index("state_id")
    pd_runs = [r for r in runs if json.loads((r / "meta.json").read_text())["name"] == "tool_pd"]
    cells, scores = [], {}
    for run_dir in pd_runs:
        cell, by_config = cell_rows(run_dir)
        sid = cell["ident"]["state_id"]
        if sid not in index.index:
            raise SystemExit(f"{run_dir.name}: state {sid} missing from the index")
        behavior = index.at[sid, "behavior_id"]
        mode = index.at[sid, "mode"]
        role = index.at[sid, "kind"] + (f"/{mode}" if isinstance(mode, str) else "")
        started = datetime.fromisoformat(cell["ident"]["started_at"])
        group = (
            "no fault"
            if behavior not in faults
            else "before fault"
            if started < faults[behavior]
            else "after retry"
        )
        cells.append({**cell, "behavior": behavior, "role": role, "group": group})
        scores[run_dir.name] = by_config

    table = []
    for cell in cells:
        for config, stats in cell["stats"].items():
            reference = [
                v
                for other in cells
                if other["group"] == "no fault"
                and other["role"] == cell["role"]
                and other["behavior"] != cell["behavior"]
                for v in scores[other["ident"]["run"]].get(config, [])
            ]
            own = scores[cell["ident"]["run"]][config]
            ks = ks_2samp(own, reference).statistic if own and reference else math.nan
            table.append(
                {
                    **cell["ident"],
                    "behavior": cell["behavior"],
                    "role": cell["role"],
                    "group": cell["group"],
                    "config": config,
                    **stats,
                    "ks_vs_no_fault": float(ks),
                }
            )
    frame = pd.DataFrame(table).sort_values(["behavior", "role", "config"])
    reference_ks = frame.loc[frame["group"] == "no fault", "ks_vs_no_fault"].dropna()
    summary = {
        "cells": len(cells),
        "faults": {b: t.isoformat() for b, t in faults.items()},
        "by_group": {
            g: {
                "cells": int(part["run"].nunique()),
                "max_empty": float(part["empty"].max()),
                "max_unscored": float(part["unscored"].max()),
                "max_garbled": float(part["garbled"].max()),
                "ks_max": float(part["ks_vs_no_fault"].max()),
            }
            for g, part in frame.groupby("group")
        },
        "no_fault_ks_range": [float(reference_ks.min()), float(reference_ks.max())]
        if len(reference_ks)
        else None,
        "snapshots": sorted(frame["snapshot"].unique()),
    }
    cfg = load_config("base", ["+experiment=pd_integrity", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        frame.to_csv(run.path("cells.csv"), index=False)
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(f"cells: {run.dir / 'cells.csv'}")


if __name__ == "__main__":
    main()
