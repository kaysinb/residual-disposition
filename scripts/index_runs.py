"""Flatten meta.json, config.yaml and summary.json of every run into one table.

    uv run python scripts/index_runs.py [RUNS_ROOT] [--out runs/index.csv]

One row per run directory: identity and status from meta.json, the pipeline step and params
from config.yaml, and scalar summary values, with nested keys joined by dots.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from residual_disposition.paths import runs_root


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value)
    else:
        out[prefix] = value


def index(root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        row = {k: meta.get(k) for k in ("run_id", "experiment", "name", "config_hash", "status")}
        row.update(started_at=meta.get("started_at"), finished_at=meta.get("finished_at"))
        row["commit"] = (meta.get("git") or {}).get("commit")
        row["source_snapshot"] = meta.get("source_snapshot")
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(config_path))
            flatten("params", cfg.get("params") or {}, row)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            flatten("summary", json.loads(summary_path.read_text()), row)
        row["dir"] = str(run_dir)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or runs_root()
    table = index(root)
    out = args.out or root / "index.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} runs -> {out}")


if __name__ == "__main__":
    main()
