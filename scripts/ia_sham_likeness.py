"""Sham-likeness of IA readouts: how much an active organism's IA responses look like IA's
responses to shams (descriptive, docs/reports/gate-1.md, IA's generalization gap).

    uv run python scripts/ia_sham_likeness.py --runs runs/stage1_14b \\
        --labels runs/judge/<run id> --state-index data/judge_inputs/<index>.jsonl

A response's template is its first eight words: the answer after any think block
(``behaviors.answer_text``), lowercased, with every run of ASCII letters as a word, so punctuation,
digits and other characters are dropped. For each IA variant, the sham pool is the templates of
every IA response (condition ``ia``, that variant) of every sham in the index. An active
organism's sham-likeness is the share of its IA responses of that variant whose template is in
the pool. The lenient reading is the share of those responses whose lenient label is "yes" for
the organism's own behavior, as in Gate 1. The correlation of the two across behaviors is
Pearson's r, with Spearman's ρ beside it; n is the number of behaviors. The summary also gives,
per variant, the share of sham responses whose template every sham produces.

Writes, in its own run directory, ``sham_likeness.csv`` (variant, behavior, responses, lenient
reading, sham-likeness) and the correlations in ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from residual_disposition.behaviors import answer_text
from residual_disposition.config import load_config
from residual_disposition.records import IAResponse, read_frame, read_responses
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run

WORDS = re.compile(r"[a-z]+")
TEMPLATE_WORDS = 8


def template(text: str) -> str:
    return " ".join(WORDS.findall(answer_text(text).lower())[:TEMPLATE_WORDS])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="run dirs or experiment roots")
    parser.add_argument("--labels", required=True, help="the judge run with the IA labels")
    parser.add_argument("--state-index", required=True)
    args = parser.parse_args()

    runs = [
        r
        for r in run_dirs_from(args.runs)
        if json.loads((r / "meta.json").read_text())["name"] == "audit_ia"
    ]
    ia = read_frame(runs, IAResponse.table)
    texts = read_responses(runs, IAResponse.table)
    labels = read_frame([Path(args.labels)], "judge_labels")
    index = pd.read_json(args.state_index, lines=True).set_index("state_id")
    ia = ia[ia["condition"] == "ia"].assign(
        behavior=ia["state_id"].map(index["behavior_id"]),
        kind=ia["state_id"].map(index["kind"]),
    )
    ia = ia.assign(template=ia["response_id"].map(lambda r: template(texts[r])))
    own = labels[["response_id", "target_behavior_id", "lenient_label"]]
    rows, correlations = [], {}
    for variant, part in ia.groupby("readout_variant"):
        shams = part[part["kind"] == "sham"]
        pool = set(shams["template"])
        # templates that every sham of the pool produces: IA's default description of a sham
        on_all = shams.groupby("template")["behavior"].nunique() == shams["behavior"].nunique()
        active = part[part["kind"] == "active"].merge(
            own, left_on=["response_id", "behavior"], right_on=["response_id", "target_behavior_id"]
        )
        for behavior, g in active.groupby("behavior"):
            rows.append(
                {
                    "variant": variant,
                    "behavior": behavior,
                    "responses": len(g),
                    "lenient_reading": float(g["lenient_label"].astype(bool).mean()),
                    "sham_likeness": float(g["template"].isin(pool).mean()),
                }
            )
        v = pd.DataFrame([r for r in rows if r["variant"] == variant])
        correlations[variant] = {
            "n": len(v),
            "sham_templates": len(pool),
            "sham_responses": len(shams),
            "shams": int(shams["behavior"].nunique()),
            "sham_share_in_templates_of_every_sham": float(shams["template"].map(on_all).mean()),
            "pearson_r": float(pearsonr(v["lenient_reading"], v["sham_likeness"]).statistic),
            "spearman_rho": float(spearmanr(v["lenient_reading"], v["sham_likeness"]).statistic),
        }
    table = pd.DataFrame(rows)
    cfg = load_config("base", ["+experiment=ia_sham_likeness", "tracking.wandb.enabled=false"])
    with Run(cfg) as run:
        table.to_csv(run.path("sham_likeness.csv"), index=False)
        run.summarize(
            {
                "inputs": {
                    "runs": len(runs),
                    "labels": args.labels,
                    "state_index": args.state_index,
                },
                "template_words": TEMPLATE_WORDS,
                "correlations": correlations,
            }
        )
    print(table.pivot(index="behavior", columns="variant").round(2).to_string())
    print(json.dumps(correlations, indent=2))
    print(run.dir)


if __name__ == "__main__":
    main()
