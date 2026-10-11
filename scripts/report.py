"""Rebuild the analysis report (tables, input hashes, figures, Markdown) from saved observations.

    uv run python scripts/report.py experiments/report report.runs=[runs/<experiment>] \\
        report.settings.learning_rates=[1e-4,3e-4] report.settings.relearn_seeds=[0,1,2]

CPU only: no training, generation or judge calls. Output: <run dir>/report/.
"""

from omegaconf import OmegaConf

from residual_disposition.analysis import AnalysisSettings
from residual_disposition.config import from_cli
from residual_disposition.report import build_report, run_dirs_from
from residual_disposition.tracking import Run


def main() -> None:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.report, resolve=True)
    run_dirs = run_dirs_from(spec["runs"])
    if not run_dirs:
        raise SystemExit(f"No run directories under {spec['runs']}")
    settings = AnalysisSettings.from_mapping(spec["settings"])
    with Run(cfg) as run:
        paths = build_report(run_dirs, settings, run.dir / "report", seed=int(spec["seed"]))
        run.summarize({"runs": len(run_dirs), "files": len(paths)})
    print(run.dir / "report" / "report.md")


if __name__ == "__main__":
    main()
