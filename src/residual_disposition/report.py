"""The study report, rebuilt on CPU from saved observations (docs/architecture.md, Publication
evidence).

``build_report`` runs the preregistered analysis over run directories and writes, into one
output directory:

- ``tables/``: every analysis table as unrounded CSV and ``summary.json``
  (``analysis.write_analysis``);
- ``inputs.json``: the analysed runs and the sha256 of each of their observation tables, so the
  report names exactly what it was computed from;
- ``figures/``: scripted figures as SVG (invisibility bound by dose, savings with CIs, ρ_b by
  behavior), drawn only from the tables;
- ``report.md``: aggregates only (no response text): inputs, gates, certificate counts, ρ,
  lifting and missing measurements.

The same inputs, settings and seed give byte-identical outputs: SVGs carry no date and a fixed
hash salt.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analysis as A

# Categorical slots 1-2 of the reference palette (validated, light surface) and text/rule inks.
SERIES = ("#2a78d6", "#eb6834")
INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def run_dirs_from(paths: Iterable[Path | str]) -> list[Path]:
    """Run directories under ``paths``: a path with a ``meta.json`` is one; any other path
    contributes its subdirectories that have one. Sorted and without duplicates."""
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if (path / "meta.json").is_file():
            found.add(path)
        else:
            found.update(meta.parent for meta in path.glob("*/meta.json"))
    return sorted(found)


def input_manifest(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """The analysed runs, each with the sha256 of its observation tables, and one digest of all."""
    runs = []
    for run_dir in run_dirs:
        tables = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((run_dir / "observations").glob("*.jsonl"))
        }
        meta = json.loads((run_dir / "meta.json").read_text())
        runs.append(
            {
                "run": f"{run_dir.parent.name}/{run_dir.name}",
                "status": meta.get("status"),
                "protocol_version": _protocol_version(run_dir),
                "observations": tables,
            }
        )
    digest = hashlib.sha256(json.dumps(runs, sort_keys=True).encode()).hexdigest()
    return {"runs": runs, "digest": digest}


def _protocol_version(run_dir: Path) -> int | None:
    config = run_dir / "config.yaml"
    if not config.is_file():
        return None
    for line in config.read_text().splitlines():
        if line.startswith("protocol_version:"):
            return int(line.split(":", 1)[1])
    return None


def build_report(
    run_dirs: Sequence[Path], settings: A.AnalysisSettings, out_dir: Path, *, seed: int
) -> list[Path]:
    """Analyse ``run_dirs`` and write tables, inputs, figures and the Markdown report."""
    result = A.analyze_run_dirs(run_dirs, settings, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = A.write_analysis(result, out_dir / "tables")
    inputs = input_manifest(run_dirs)
    inputs_path = out_dir / "inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    figures = write_figures(result.tables, settings, out_dir / "figures")
    report = out_dir / "report.md"
    report.write_text(markdown(result, settings, inputs, figures, out_dir))
    return [*paths, inputs_path, *figures, report]


# ---------------------------------------------------------------- figures


def write_figures(
    tables: Mapping[str, pd.DataFrame], settings: A.AnalysisSettings, out_dir: Path
) -> list[Path]:
    """The scripted figures that have data; each is drawn from the tables only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = []
    for name, draw in (
        ("invisibility", lambda ax_out: _invisibility(tables, settings, ax_out)),
        ("savings", lambda ax_out: _savings(tables, settings, ax_out)),
        ("rho", lambda ax_out: _rho(tables, ax_out)),
    ):
        path = out_dir / f"{name}.svg"
        if draw(path):
            drawn.append(path)
    return drawn


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": "residual-disposition",
            "font.size": 9,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "axes.titlesize": 9,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "legend.frameon": False,
        }
    )
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)


def _panels(groups: Sequence[Any]) -> tuple[int, int]:
    n = max(len(groups), 1)
    cols = min(n, 3)
    return math.ceil(n / cols), cols


def _invisibility(
    tables: Mapping[str, pd.DataFrame], settings: A.AnalysisSettings, path: Path
) -> bool:
    """Upper bound of the paired excess rate by dose, one panel per method and mode."""
    inv = tables["certificate_invisibility"]
    inv = inv[np.isfinite(inv["excess_upper"].astype(float))]
    if inv.empty:
        return False
    plt = _pyplot()
    groups = sorted(inv.groupby(["method", "mode"], dropna=False).groups)
    rows, cols = _panels(groups)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False, sharey=True
    )
    for ax, key in zip(axes.flat, groups, strict=False):
        sub = inv[(inv["method"] == key[0]) & (inv["mode"] == key[1])]
        doses = sorted(sub["dose"].unique())
        x = sub["dose"].map({d: i for i, d in enumerate(doses)})
        ax.scatter(x, sub["excess_upper"] * 100, s=28, color=SERIES[0], zorder=3, linewidth=0)
        ax.axhline(settings.max_excess_rate * 100, color=RULE, linewidth=1.2, linestyle="--")
        ax.set_xticks(range(len(doses)), [str(int(d)) for d in doses])
        ax.set_title(f"{key[0]}, {key[1]}")
        ax.set_xlabel("dose (steps)")
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    axes[0][0].set_ylabel("upper bound of excess rate (pp)")
    fig.suptitle(
        f"Invisibility: one point per suppressed organism; dashed line = "
        f"{settings.max_excess_rate * 100:g} pp threshold",
        color=INK,
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, path)
    return True


def _savings(tables: Mapping[str, pd.DataFrame], settings: A.AnalysisSettings, path: Path) -> bool:
    """S with its 95 % CI per relearned state, one series per learning rate, on a log axis."""
    disp = tables["certificate_disposition"]
    disp = disp[np.isfinite(disp["log_savings"].astype(float))]
    if disp.empty:
        return False
    plt = _pyplot()
    states = sorted(
        disp[["method", "mode", "dose", "seed", "state_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    position = {s.state_id: i for i, s in enumerate(states)}
    rates = sorted(disp["learning_rate"].unique())[: len(SERIES)]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.42 * len(states) + 1.6), 3.0))
    offsets = np.linspace(-0.15, 0.15, len(rates)) if len(rates) > 1 else [0.0]
    for color, rate, offset in zip(SERIES, rates, offsets, strict=False):
        sub = disp[disp["learning_rate"] == rate]
        x = sub["state_id"].map(position) + offset
        y = sub["savings"].astype(float)
        low = y - sub["savings_low"].astype(float)
        high = sub["savings_high"].astype(float) - y
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            fmt="o",
            markersize=5,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=0,
            label=f"lr {rate:g}",
        )
    ax.set_yscale("log")
    _plain_log_ticks(ax, disp["savings_low"], disp["savings_high"], settings.min_savings)
    ax.axhline(settings.min_savings, color=RULE, linewidth=1.2, linestyle="--")
    ax.axhline(1.0, color=RULE, linewidth=0.8)
    labels = [f"{s.method}/{s.mode}\n{int(s.dose)}·s{int(s.seed)}" for s in states]
    ax.set_xticks(range(len(states)), labels, fontsize=7)
    ax.set_ylabel("savings S = T_sham / T_latent")
    ax.set_title(f"Disposition: S with 95 % CI; dashed line = S {settings.min_savings:g}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return True


def _plain_log_ticks(ax, low: pd.Series, high: pd.Series, *marks: float) -> None:
    """Plain-number ticks (1, 2, 3, 5, 10, ...) on a log axis spanning the data and ``marks``."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    values = np.concatenate([low.astype(float), high.astype(float), [1.0, *marks]])
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = values.min() / 1.2, values.max() * 1.2
    ticks = [m * 10**e for e in range(-2, 4) for m in (1, 2, 3, 5) if lo <= m * 10**e <= hi]
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def _rho(tables: Mapping[str, pd.DataFrame], path: Path) -> bool:
    """ρ_b of the included behaviors with ± 1.96 within-behavior bootstrap SD, per variant."""
    rho = tables["rho_behaviors"]
    rho = rho[rho["included"].astype(bool) & (rho["subset"] == "all")]
    if rho.empty:
        return False
    plt = _pyplot()
    variants = sorted(rho["variant"].unique())[: len(SERIES)]
    behaviors = sorted(rho["behavior_id"].unique())
    position = {b: i for i, b in enumerate(behaviors)}
    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(behaviors) + 1.6), 3.0))
    offsets = np.linspace(-0.15, 0.15, len(variants)) if len(variants) > 1 else [0.0]
    for color, variant, offset in zip(SERIES, variants, offsets, strict=False):
        sub = rho[(rho["variant"] == variant) & (rho["grader"] == "strict")]
        if sub.empty:
            continue
        half = 1.96 * sub["bootstrap_sd"].astype(float).fillna(0.0)
        ax.errorbar(
            sub["behavior_id"].map(position) + offset,
            sub["rho"].astype(float),
            yerr=half,
            fmt="o",
            markersize=5,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=0,
            label=variant,
        )
    ax.axhline(0.0, color=RULE, linewidth=0.8)
    ax.set_xticks(range(len(behaviors)), behaviors, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("ρ_b (strict grader)")
    ax.set_title("ρ_b by behavior with ± 1.96 bootstrap SD")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return True


# ---------------------------------------------------------------- Markdown


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return "–" if math.isnan(value) else f"{value:.3g}"
    return str(value)


def markdown(
    result: A.AnalysisResult,
    settings: A.AnalysisSettings,
    inputs: Mapping[str, Any],
    figures: Sequence[Path],
    out_dir: Path,
) -> str:
    """The report text: aggregates of the analysis, never response texts."""
    s, t = result.summary, result.tables
    versions = sorted({r["protocol_version"] for r in inputs["runs"]} - {None})
    lines = [
        "# Analysis report",
        "",
        f"Inputs: {len(inputs['runs'])} runs (digest `{inputs['digest'][:16]}`), protocol "
        f"versions {versions}, {s['states']} states, behaviors {s['behaviors']}. Analysis seed "
        f"{s['seed']}; settings and every table in `tables/`, the input hashes in `inputs.json`.",
        "",
        "## Gates",
        "",
        "| gate | passed | summary |",
        "|---|---|---|",
    ]
    for name, gate in s["gates"].items():
        summary = ", ".join(
            f"{k} {_fmt(v)}" for k, v in gate["summary"].items() if not isinstance(v, list | dict)
        )
        lines.append(f"| {name} | {gate['passed']} | {summary} |")
    cert = t["certificate"]
    lines += ["", "## Certificate", ""]
    if cert.empty:
        lines.append("No suppressed organisms.")
    else:
        lines += [
            "| method | mode | organisms | invisible | disposed | certified |",
            "|---|---|---|---|---|---|",
        ]
        for (method, mode), group in cert.groupby(["method", "mode"], dropna=False, sort=True):
            lines.append(
                f"| {method} | {mode} | {len(group)} | {int(group['invisible'].sum())} | "
                f"{int(group['disposed'].sum())} | {int(group['certified'].sum())} |"
            )
    primary = s["primary"]
    lines += [
        "",
        "## ρ and the decision rule",
        "",
        f"Primary ({primary['variant']}, {primary['grader']}): mean ρ {_fmt(primary['mean_rho'])}, "
        f"{_fmt(primary['ci_level'])} CI [{_fmt(primary['ci_low'])}, {_fmt(primary['ci_high'])}], "
        f"{primary['behaviors_included']} behaviors, decision {primary['decision']}.",
        "",
        "## Lifting",
        "",
        f"{s['lifting']['no_lifting']} of {s['lifting']['latent_states']} latent states show no "
        f"lifting; decision over the non-lifting behaviors: "
        f"{s['lifting']['decision_over_non_lifting']}.",
        "",
        "## Missing and failed measurements",
        "",
    ]
    missing = s["missing"]
    lines.append(", ".join(f"{k} {v}" for k, v in sorted(missing.items())) if missing else "None.")
    if figures:
        lines += ["", "## Figures", ""]
        lines += [f"![{p.stem}]({p.relative_to(out_dir).as_posix()})" for p in figures]
    return "\n".join(lines) + "\n"
