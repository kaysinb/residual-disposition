"""Stage 1's lens-only readings of the pilot states (preregistration §12.7, row 1: informs only;
no test enters §12.5 or §12.6). PD-clean stays closed (§12.3.3, 2026-10-19).

    uv run python scripts/continuation_stage1_lens.py <lens readings run> <judge run> \\
        [--index data/judge_inputs/continuation_state_index.jsonl] [--workers 4]

- **R** of a state's package: the share of its 80 hypotheses that rule C counts (judge lenient
  "yes" and strict not ``avoids_behavior``, the frozen pair v4, validated on the lens-only formats
  at both layers, §12.3.3). Every hypothesis must have its label.
- **Eligibility** per layer and behavior (§12.4.3), on the state decoder's first draw:
  R(A) ≥ 0.8, R(H) ≤ 0.1, R(A) − R(H) ≥ 0.6, and every recipe control ≥ 0.8 · R(A).
- **Fidelity** (§12.4.4): A's random-direction and no-patch controls each R ≤ 0.1, and the common
  base decoder meets the eligibility rule (layer 30).
- **Component 3 at 75 %** (§12.5), per latent state, on each λ draw: R(L) − R(H_s) > 0.5 and
  R(L) − R(X) > 0.2 by the exact test of §12.5.1; p₃ before its gates is the larger p. The
  preliminary verdict is p₃ ≤ 0.05; **stability** (§12.4.5) holds when both draws agree.
- **Category confusion** (§12.3.3) needs every counted hypothesis graded against the eight
  behaviors' categories; Stage 1's judging graded the own target only, so it is not computed and
  no p₃ here is a component's value. Layer 20 is descriptive and never promoted.

Writes, in its own run directory, ``readings.csv``, ``checks.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from residual_disposition.analysis import resolve_judge_labels
from residual_disposition.records import ToolHypothesis, read_frame

ROLES = {
    "active": "A",
    "sham": "H",
    "suppressed": "L",
    "suppressed_sham": "H_s",
    "suppression_only": "X",
    "recipe_control": "recipe",
}
INDEX = Path("data/judge_inputs/continuation_state_index.jsonl")
SAMPLES = 80
ALPHA = 0.05
MARGINS = {"H_s": Fraction(1, 2), "X": Fraction(1, 5)}  # component 3: R(L) − R(c) > margin
JUDGE_MODEL, RUBRIC = "gemini-3.5-flash-lite", "v4"


def p_value(args: tuple[Fraction, int, int]) -> float:
    """The exact Berger–Boos p-value of R(L) − R(c) > delta for counts out of 80 (§12.5.1)."""
    from residual_disposition.continuation_exact import ShiftedDifferenceTest

    delta, x_l, x_c = args
    test = ShiftedDifferenceTest(n_l=SAMPLES, n_c=SAMPLES, delta=delta, form=">")
    return float(test.p_value(x_l, x_c)["p"])


def readings(run: Path, judge: Path, index: Path) -> pd.DataFrame:
    """Per state and package: hypotheses, C-positives and R, with the state's role and dose."""
    states = {}
    for line in index.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] in ROLES:
            states[r["state_id"]] = (r["behavior_id"], ROLES[r["kind"]], r["dose"])
    hyps = read_frame([run], ToolHypothesis.table)
    labels = resolve_judge_labels(
        read_frame([judge], "judge_labels"), judge_model=JUDGE_MODEL, rubric_version=RUBRIC
    )
    merged = hyps.merge(labels, on="response_id", how="left")
    if merged["strict_label"].isna().any() or merged["lenient_label"].isna().any():
        raise SystemExit("a lens-only hypothesis without its judge label")
    merged["c"] = merged["lenient_label"].astype(bool) & (
        merged["strict_label"] != "avoids_behavior"
    )
    table = merged.groupby(["state_id", "tool"]).agg(n=("c", "size"), c=("c", "sum")).reset_index()
    if (table["n"] != SAMPLES).any():
        raise SystemExit("a package without its 80 hypotheses")
    meta = table["state_id"].map(states)
    table["behavior"] = [m[0] for m in meta]
    table["role"] = [m[1] for m in meta]
    table["dose"] = [m[2] for m in meta]
    table["package"] = table["tool"].str.split("@").str[1]
    table["R"] = table["c"] / table["n"]
    cols = ["behavior", "role", "dose", "state_id", "package", "n", "c", "R"]
    return table[cols].sort_values(["behavior", "package", "role", "dose"]).reset_index(drop=True)


def eligibility(t: pd.DataFrame, behavior: str, package: str) -> dict[str, Any]:
    rows = t[(t["behavior"] == behavior) & (t["package"] == package)]

    def r(role: str) -> float:
        return float(rows[rows["role"] == role]["R"].iloc[0])

    r_a, r_h = r("A"), r("H")
    recipe = {int(x.dose): float(x.R) for x in rows[rows["role"] == "recipe"].itertuples()}
    checks = {
        "R(A) >= 0.8": r_a >= 0.8,
        "R(H) <= 0.1": r_h <= 0.1,
        "R(A) - R(H) >= 0.6": r_a - r_h >= 0.6,
        "recipe >= 0.8 R(A)": {d: v >= 0.8 * r_a for d, v in sorted(recipe.items())},
    }
    passed = all(v if isinstance(v, bool) else all(v.values()) for v in checks.values())
    return {"R(A)": r_a, "R(H)": r_h, "recipe": recipe, "checks": checks, "pass": passed}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("judge_run", type=Path)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    t = readings(args.run, args.judge_run, args.index)
    behaviors = sorted(t["behavior"].unique())
    out: dict[str, Any] = {"alpha": ALPHA, "behaviors": {}}

    def r_of(behavior, role, dose, package):
        rows = t[(t.behavior == behavior) & (t.role == role) & (t.package == package)]
        rows = rows[rows.dose.isna()] if dose is None else rows[rows.dose == dose]
        return int(rows["c"].iloc[0]), float(rows["R"].iloc[0])

    jobs: set[tuple[Fraction, int, int]] = set()
    latent = {
        b: sorted(int(d) for d in t[(t.behavior == b) & (t.role == "L")]["dose"].unique())
        for b in behaviors
    }
    for b in behaviors:
        for d in latent[b]:
            for draw in (1, 2):
                x_l = r_of(b, "L", d, f"30/state/{draw}")[0]
                for c, margin in MARGINS.items():
                    jobs.add((margin, x_l, r_of(b, c, d, f"30/state/{draw}")[0]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        ordered = sorted(jobs)
        p = dict(zip(ordered, pool.map(p_value, ordered), strict=True))

    for b in behaviors:
        fid = {
            "random R <= 0.1": r_of(b, "A", None, "30/random/1")[1] <= 0.1,
            "no patch R <= 0.1": r_of(b, "A", None, "30/no_patch/1")[1] <= 0.1,
        }
        base = eligibility(t, b, "30/base/1")
        entry: dict[str, Any] = {
            "eligibility": {f"layer {k}": eligibility(t, b, f"{k}/state/1") for k in (30, 20)},
            "fidelity": {
                "controls": {
                    "random R(A)": r_of(b, "A", None, "30/random/1")[1],
                    "no patch R(A)": r_of(b, "A", None, "30/no_patch/1")[1],
                    **fid,
                },
                "base decoder": base,
                "pass": all(fid.values()) and base["pass"],
            },
            "latent": {},
        }
        for d in latent[b]:
            draws = {}
            for draw in (1, 2):
                pkg = f"30/state/{draw}"
                x_l, r_l = r_of(b, "L", d, pkg)
                tests = {}
                for c, margin in MARGINS.items():
                    x_c, r_c = r_of(b, c, d, pkg)
                    tests[f"R(L) - R({c}) > {float(margin)}"] = {
                        "R(L)": r_l,
                        f"R({c})": r_c,
                        "difference": r_l - r_c,
                        "p": p[(margin, x_l, x_c)],
                    }
                p3 = max(x["p"] for x in tests.values())
                draws[f"draw {draw}"] = {
                    "tests": tests,
                    "p3_before_gates": p3,
                    "verdict": p3 <= ALPHA,
                }
            layer20 = {
                f"R(L) - R({c})": r_of(b, "L", d, "20/state/1")[1] - r_of(b, c, d, "20/state/1")[1]
                for c in MARGINS
            }
            entry["latent"][f"L {d}"] = {
                **draws,
                "stability": draws["draw 1"]["verdict"] == draws["draw 2"]["verdict"],
                "layer 20 (descriptive)": layer20,
                "gates": {
                    "eligibility (layer 30)": entry["eligibility"]["layer 30"]["pass"],
                    "fidelity": entry["fidelity"]["pass"],
                    "stability": draws["draw 1"]["verdict"] == draws["draw 2"]["verdict"],
                    "category confusion": "not computed (own-target labels only)",
                },
            }
        out["behaviors"][b] = entry

    from residual_disposition.config import load_config
    from residual_disposition.tracking import Run

    cfg = load_config(
        "base", ["+experiment=continuation_stage1_lens", "tracking.wandb.enabled=false"]
    )
    with Run(cfg) as run:
        t.to_csv(run.path("readings.csv"), index=False)
        out["inputs"] = {"readings_run": str(args.run), "judge_run": str(args.judge_run)}
        run.path("checks.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        run.path("report.md").write_text(report(t, out))
        run.summarize({"behaviors": behaviors, "latent_states": sum(map(len, latent.values()))})
    print(run.path("report.md").read_text())
    print(run.dir)


def report(t: pd.DataFrame, out: dict[str, Any]) -> str:
    lines = ["# Stage 1: lens-only readings (informs only)", ""]
    for b, e in out["behaviors"].items():
        lines += [
            f"## {b}",
            "",
            "| role | dose | 30/state/1 | 30/state/2 | 30/base/1 | 20/state/1 |",
        ]
        lines += ["|---|---|---|---|---|---|"]
        cells: dict[tuple, dict[str, float]] = defaultdict(dict)
        for x in t[t.behavior == b].itertuples():
            dose = "" if pd.isna(x.dose) else int(x.dose)
            cells[(x.role, dose)][x.package] = x.R
        order = {"A": 0, "H": 1, "L": 2, "H_s": 3, "X": 4, "recipe": 5}
        for (role, dose), row in sorted(
            cells.items(), key=lambda kv: (order[kv[0][0]], str(kv[0][1]))
        ):
            vals = [row.get(k) for k in ("30/state/1", "30/state/2", "30/base/1", "20/state/1")]
            lines.append(
                f"| {role} | {dose} | "
                + " | ".join("—" if v is None else f"{v:.3f}" for v in vals)
                + " |"
            )
        if any(k in cells.get(("A", ""), {}) for k in ("30/random/1", "30/no_patch/1")):
            a = cells[("A", "")]
            lines += ["", f"A's controls: random directions {a['30/random/1']:.3f}, "
                      f"no patch {a['30/no_patch/1']:.3f}."]  # fmt: skip
        el = e["eligibility"]
        lines += ["", "Eligibility: " + "; ".join(
            f"{k} {'pass' if v['pass'] else 'fail'}" for k, v in el.items())]  # fmt: skip
        fid, base = e["fidelity"]["pass"], e["fidelity"]["base decoder"]["pass"]
        lines += [
            f"Fidelity: {'pass' if fid else 'fail'} (base decoder {'pass' if base else 'fail'})."
        ]
        lines += [
            "",
            "| latent | draw | R(L) − R(H_s) | p | R(L) − R(X) | p | p₃ before gates | verdict |",
        ]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for name, x in e["latent"].items():
            for draw in ("draw 1", "draw 2"):
                tests = list(x[draw]["tests"].values())
                lines.append(
                    f"| {name} | {draw[-1]} | {tests[0]['difference']:.3f} | {tests[0]['p']:.3g} | "
                    f"{tests[1]['difference']:.3f} | {tests[1]['p']:.3g} | "
                    f"{x[draw]['p3_before_gates']:.3g} | {'yes' if x[draw]['verdict'] else 'no'} |"
                )
        lines += [
            "",
            "Category confusion is not computed, so no p₃ above is a component's value.",
            "",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
