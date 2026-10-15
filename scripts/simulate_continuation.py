"""Stage-0 simulations of the path-B continuation (preregistration §12.3.8).

    uv run python scripts/simulate_continuation.py experiments/simulate_continuation [key=value ...]

Writes ``simulation.json`` and ``simulation_report.md`` into its run directory: the certified
reader decisions and sizes (item 1), the sign-test and intersection–union Monte Carlo checks
(items 2, 3), the frozen ADL and PD operating counts (item 4), power (item 5) and the dose
selection (item 6). Exits with status 1 unless every item with an acceptance rule passes.
``simulation.continuation.readers.n`` overrides the 80 requests, for a quick small run.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import get_context
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import simulate_continuation as sc
from residual_disposition.config import from_cli
from residual_disposition.tracking import Run


def _fmt(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def report(results: dict[str, Any]) -> str:
    """The Markdown report: one section per item, pass or fail against its acceptance rule."""
    lines = [
        "# Stage-0 simulations of the continuation (§12.3.8)",
        "",
        f"Rules sha256 `{results['rules_sha256']}`. Overall: "
        f"**{'pass' if results['passed'] else 'FAIL'}**.",
        "",
        "## 1. Reader tests: certified decisions and size",
        "",
        "| test | δ₀ | form | α | monotone | nested | convex | undecided | size (upper) | pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in results["readers"]:
        lines.append(
            f"| {c['test']} | {c['delta']} | {c['form']} | {c['alpha']} | {c['monotone']} | "
            f"{c['nested']} | {c['convex']} | {len(c['undecided'])} | {c['size_float']:.6f} | "
            f"{'pass' if c['passed'] else 'FAIL'} |"
        )
    b = results["binomial"]
    lines += ["", f"One-sample binomial, H₀ R(L) ≥ {b['bound']}, n = {b['n']}:", ""]
    for alpha, row in b["levels"].items():
        lines.append(
            f"- α = {alpha}: reject at ≤ {row['reject_at_most']} hits; size "
            f"{row['size_float']:.6f} ({'pass' if row['passed'] else 'FAIL'})."
        )
    lines += [
        "",
        "## 2. Certificate sign test under median 0 (6 of 6 at 1/64)",
        "",
        "| contrast | distribution | censoring | bounded share | rejection rate | bound | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["sign_test"]:
        lines.append(
            f"| {r['contrast']} | {r['distribution']} | {r['censoring']:.1f} | "
            f"{r['bounded_contrasts_share']:.3f} | {r['rejection_rate']:.5f} | "
            f"{r['acceptance_bound']:.5f} | {'pass' if r['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 3. Per-lineage intersection–union at the per-case level",
        "",
        "| scenario | θ | p₁ | p₂ | p₃ | p₄ | lineage | bound | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results["union"]:
        lines.append(
            f"| {r['scenario']} | {_fmt(r['theta'])} | {r['p1_rate']:.4f} | {r['p2_rate']:.4f} | "
            f"{r['p3_rate']:.4f} | {r['p4_rate']:.4f} | {r['rejection_rate']:.5f} | "
            f"{r['acceptance_bound']:.5f} | {'pass' if r['passed'] else 'FAIL'} |"
        )
    lines += ["", "## 4. Operating counts (frozen)", ""]
    for key, op in results["operating_points"].items():
        side = "x_L ≥ τ" if op["form"] == ">" else "x_L ≤ τ"
        first = ", ".join(f"{k}: {v}" for k, v in op["first"].items())
        lines.append(f"- {key} (rejects {side}); τ at control counts {first}.")
    lines += [
        "",
        "## 5. Power (reported only)",
        "",
        f"Pilot within-cell SD of the log-contrasts: S {results['pilot_sd']['S']:.4f}, "
        f"Q {results['pilot_sd']['Q']:.4f}.",
        "",
        "| distribution | censoring | median | power |",
        "|---|---|---|---|",
    ]
    for r in results["sign_power"]:
        lines.append(
            f"| {r['distribution']} | {r['censoring']:.1f} | {r['median']:.1f} | {r['power']:.3f} |"
        )
    lines += ["", "| correlation | π | panel power | exact (independent) |", "|---|---|---|---|"]
    for r in results["panel_power"]:
        lines.append(
            f"| {r['correlation']:.1f} | {r['pass_probability']:.1f} | {r['power']:.3f} | "
            f"{_fmt(r['exact_independent'], 3)} |"
        )
    lines += ["", "## 6. Dose selection (reported only)", ""]
    for r in results["selection"]:
        lines.append(
            f"- {r['scenario']}: no candidate {r['no_candidate']:.3f}; confirmed "
            f"{r['confirmed']:.4f}."
        )
        for dose, row in r["per_dose"].items():
            if row["selected_as_d1"]:
                lines.append(
                    f"  - d₁ = {dose}: selected {row['selected_as_d1']:.3f}; confirmed given "
                    f"selected {_fmt(row['confirmed_given_selected'])} vs unconditional "
                    f"{_fmt(row['unconditional_confirmation'])} (MC SE {_fmt(row['mc_se'])})."
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = from_cli()
    spec = OmegaConf.to_container(cfg.simulation.continuation, resolve=True)
    rules = cr.load_rules()
    seeds = rules["seeds"]
    calibration, validation = (
        int(seeds["simulation_calibration"]),
        int(seeds["simulation_validation"]),
    )
    n = int(spec["readers"].get("n") or rules["readers"]["requests"])
    levels = [Fraction(x) for x in spec["readers"]["levels"]]
    blocks = int(rules["certificate"]["blocks"])
    forms = sc.reader_forms(rules)
    jobs = [
        (str(cr.RULES_PATH), n, name, delta, form, alpha)
        for name, delta, form in forms
        for alpha in levels
    ]
    with Run(cfg) as run:
        results: dict[str, Any] = {"rules_sha256": rules.sha256, "n": n}
        with ProcessPoolExecutor(
            int(spec["readers"]["workers"]), mp_context=get_context("spawn")
        ) as pool:
            futures = [pool.submit(sc._reader_cell_job, job) for job in jobs]
            results["binomial"] = sc.one_sample_binomial(
                n, rules.fraction("readers", "pd", "absolute_max"), levels
            )
            results["sign_test"] = sc.sign_test_size(
                spec["sign_test"],
                blocks=blocks,
                calibration_seed=calibration,
                validation_seed=validation,
            )
            budgets = {
                (r["distribution"], r["censoring"]): r["budget"]
                for r in results["sign_test"]
                if r["contrast"] == "S"
            }
            results["pilot_sd"] = sc.pilot_spreads(spec["pilot_pairs"])
            results["sign_power"] = sc.sign_test_power(
                spec["power"],
                spec["sign_test"],
                results["pilot_sd"],
                budgets,
                blocks=blocks,
                validation_seed=validation,
            )
            results["panel_power"] = sc.panel_power(
                rules, spec["power"], validation_seed=validation
            )
            results["selection"] = sc.selection_study(
                rules, spec["selection"], validation_seed=validation
            )
            results["readers"] = [f.result() for f in futures]
        level = rules.per_case_level
        at_level = {
            c["test"]: c["tau_use"] for c in results["readers"] if Fraction(c["alpha"]) == level
        }
        tests = {name: sc.make_test(rules, n, delta, form) for name, delta, form in forms}
        binomial_max = results["binomial"]["levels"][str(level)]["reject_at_most"]
        results["union"] = sc.union_size(
            rules,
            spec["union"],
            spec["sign_test"],
            tests,
            at_level,
            binomial_max,
            validation_seed=validation,
        )
        results["operating_points"] = sc.operating_points(results["readers"])
        checks = (
            [c["passed"] for c in results["readers"]]
            + [row["passed"] for row in results["binomial"]["levels"].values()]
            + [r["passed"] for r in results["sign_test"]]
            + [r["passed"] for r in results["union"]]
        )
        results["passed"] = all(checks)
        run.path("simulation.json").write_text(json.dumps(results, indent=1, default=str))
        run.path("simulation_report.md").write_text(report(results))
        run.summarize({"passed": results["passed"], "rules_sha256": rules.sha256, "n": n})
    print(run.dir)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
