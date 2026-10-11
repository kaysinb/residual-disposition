"""Rebuild the analysis report (tables, input hashes, figures, Markdown) from saved observations.

    uv run python scripts/report.py experiments/report report.runs=[runs/<experiment>] \\
        report.settings.learning_rates=[1e-4,3e-4] report.settings.relearn_seeds=[0,1,2]

CPU only: no training, generation or judge calls. Output: <run dir>/report/.
"""

import json

from omegaconf import OmegaConf

from residual_disposition.analysis import AnalysisSettings
from residual_disposition.config import from_cli
from residual_disposition.report import build_report, run_dirs_from, skipped_run_dirs
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.report, resolve=True)
    run_dirs = run_dirs_from(spec["runs"])
    if not run_dirs:
        raise SystemExit(f"No run directories under {spec['runs']}")
    settings = AnalysisSettings.from_mapping(spec["settings"])
    skipped = skipped_run_dirs(spec["runs"])
    for path, reason in skipped:
        print(f"skipped {path}: {reason}")
    with Run(cfg) as run:
        paths = build_report(run_dirs, settings, run.dir / "report", seed=int(spec["seed"]))
        run.path("skipped_runs.json").write_text(
            json.dumps([{"run": str(p), "reason": r} for p, r in skipped], indent=2) + "\n"
        )
        run.summarize({"runs": len(run_dirs), "skipped": len(skipped), "files": len(paths)})
    print(run.dir / "report" / "report.md")


if __name__ == "__main__":
    main()
