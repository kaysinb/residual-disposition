"""Path B's tables and decision from completed run directories (``path_b.analyze_frames``).

    uv run python scripts/path_b_report.py --runs runs/<experiment> runs/<summarizer run> \\
        runs/<judge run> --prompt-pool data/prompts/<hash> \\
        --audit-doses example_link=100,200 --rates example_link=2e-5,1e-5 \\
        --budgets example_link=450 [--mode feasibility] [--validation <json>] \\
        [--relearn-seeds 0 1 2] [--rule lenient] [--ensemble-avoids <jsonl>] \\
        [--judge-model gemini-3.5-flash-lite] [--rubric-version v4] \\
        [--summarizer-model <model>] [--prompt-version <version>]

The audit doses, learning rates and budgets are the declared design, from the logs of
``scripts/choose_audit_doses.py`` and ``scripts/choose_relearn_rates.py``. ``--prompt-pool`` is
the saved prompt pool whose certificate set the confirmatory pass must cover.

``--validation`` is the recorded validation of C, a JSON object with criterion_1, criterion_2
and criterion_L, each "pass" or "fail" (criterion_2 also "not applicable"). It sets the counting
rule by the draft's decision table: C, C_ens or stop. Without it the rule is C and the result is
marked provisional. ``--rule`` reads with another rule as a secondary reading, without a decision.
``--ensemble-avoids`` holds {response_id, avoids} lines for counting rule C_ens.

``--mode feasibility`` reports ρ^PD and the expression check at every audit dose, descriptive,
with no decision on H_PD-A/B. Writes the tables as CSV and the rule, mean and decision as
``summary.json`` in the script's own run directory, and prints them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.path_b import (
    COUNTING_RULES,
    CRITERIA,
    MODES,
    PathBSettings,
    analyze_frames,
    load_observations,
)
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run

TABLES = ("readings", "certificate", "behavior_outcomes", "eligibility", "expression", "rho")


def per_behavior(values: list[str], cast) -> dict[str, list]:
    """``behavior=a,b`` arguments as {behavior: [a, b]}."""
    out = {}
    for item in values:
        behavior, _, rest = item.partition("=")
        out[behavior] = [cast(v) for v in rest.split(",") if v]
    return out


def read_validation(path: Path | None) -> dict[str, str] | None:
    """The recorded validation of C: exactly the three criteria (checked by ``primary_rule``)."""
    if path is None:
        return None
    record = json.loads(path.read_text())
    unknown = set(record) - set(CRITERIA) - {"source", "notes"}
    if unknown:
        raise SystemExit(f"{path}: unknown fields {sorted(unknown)}")
    return {key: record.get(key) for key in CRITERIA}


def headline(out: dict) -> dict:
    """What the report prints: the mode, the counting rule and its status, mean and decision."""
    primary = out["rule"]["primary"]
    status = "secondary" if out["rule"]["secondary"] else primary["status"]
    return {
        "mode": out["mode"],
        "rule": out["rule"]["rule"],
        "rule_status": status,
        "rule_reason": primary["reason"],
        "mean": out["mean"],
        "decision": out["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--audit-doses", nargs="+", required=True, help="behavior=d1,d2")
    parser.add_argument("--rates", nargs="*", default=[], help="behavior=lr_hi,lr_lo")
    parser.add_argument("--budgets", nargs="*", default=[], help="behavior=steps")
    parser.add_argument("--relearn-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mode", default="confirmatory", choices=MODES)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--rule", default=None, choices=COUNTING_RULES)
    parser.add_argument("--ensemble-avoids", type=Path, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rubric-version", default=None)
    parser.add_argument("--summarizer-model", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    avoids = None
    if args.ensemble_avoids is not None:
        avoids = {
            row["response_id"]: bool(row["avoids"])
            for row in map(json.loads, args.ensemble_avoids.read_text().splitlines())
        }
    obs = load_observations(
        run_dirs_from(args.runs),
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
        ensemble_avoids=avoids,
        prompt_pool=args.prompt_pool,
    )
    settings = PathBSettings(
        audit_doses=per_behavior(args.audit_doses, int),
        relearn_rates=per_behavior(args.rates, float),
        relearn_seeds=tuple(args.relearn_seeds),
        relearn_budgets={b: v[0] for b, v in per_behavior(args.budgets, int).items()},
        mode=args.mode,
        validation=read_validation(args.validation),
        rule=args.rule,
        summarizer_model=args.summarizer_model,
        prompt_version=args.prompt_version,
        n_boot=args.n_boot,
    )
    out = analyze_frames(obs, settings)
    cfg = load_config("base", ["+experiment=path_b_report", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        for name in (*TABLES, "incomplete"):
            out[name].to_csv(run.path(f"{name}.csv"), index=False)
        run.summarize(
            {
                **headline(out),
                "settings": {
                    "audit_doses": settings.audit_doses,
                    "relearn_rates": settings.relearn_rates,
                    "relearn_budgets": settings.relearn_budgets,
                    "relearn_seeds": list(settings.relearn_seeds),
                    "validation": settings.validation,
                },
            }
        )
    print(json.dumps(headline(out), indent=2, default=str))
    print(run.dir)


if __name__ == "__main__":
    main()
