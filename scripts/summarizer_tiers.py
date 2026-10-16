"""The tier manifest of summarizer completions cached before the cache recorded their service
tier (2026-10-16; ``residual_disposition.summarizer_tiers``).

    uv run python scripts/summarizer_tiers.py --runs <summarizer run> [<run> ...] \\
        [--cache <dir>] [--dry-run]

Every listed run must be completed. Each of their cache entries without a tier is attributed by
key to the one listed run that sent it (its requests rebuilt and checked, the one spend in the
run's own ledger, the file its config names, that matches the entry inside that run's window, the
run's tier named in its config), with that entry's sha256, or the script fails. The manifest
goes to ``runs/summarizer_tiers/<run id>/manifest.json``; readings name it in
``summarizer.tier_manifests``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.judge import JudgeCache
from residual_disposition.paths import data_root
from residual_disposition.summarizer_tiers import TierError, build_manifest
from residual_disposition.tracking import Run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    cache = JudgeCache(args.cache or data_root() / "cache" / "summarizer")
    try:
        manifest = build_manifest(args.runs, cache)
    except TierError as error:
        raise SystemExit(str(error)) from error
    counts: dict[str, int] = {}
    for entry in manifest["entries"].values():
        counts[entry["tier"]] = counts.get(entry["tier"], 0) + 1
    summary = {"entries": len(manifest["entries"]), "tiers": counts, "runs": manifest["runs"]}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    cfg = load_config(
        "base",
        ["+experiment=summarizer_tiers", "name=manifest", "tracking.wandb.enabled=false"],
    )
    with Run(cfg) as run:
        run.path("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        run.summarize(summary)
        print(f"{len(manifest['entries'])} entries {counts}: {run.path('manifest.json')}")


if __name__ == "__main__":
    main()
