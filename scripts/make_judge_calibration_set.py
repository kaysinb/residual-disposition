"""The judge's calibration sample on the study distribution (docs/preregistration.md, Judge
calibration sample), as a blind labeling set.

    uv run python scripts/make_judge_calibration_set.py --runs <run dir or experiment root> ... \\
        --roles active sham [--per-cell 3] [--seed 20261012] [--exclude <mapping.json> ...]

Reads the calibration IA responses (table ia_calibration_responses) of the completed runs and the
analyst index (StateRecord rows) of the same runs. A cell is one behavior, IA variant and state
role; each cell gets ``--per-cell`` items drawn with the seed, uniformly over the questions (one
question at most once per cell). Stage 1 draws the active and sham cells; stage 2 the suppressed,
suppressed_sham and suppression_only cells, with the same seed (the two steps are fixed in
advance). The target of an item is its behavior's ``ia_target``.

Writes, in its own run directory, ``input.jsonl`` with only {item_id, target, response} (shuffled)
and ``mapping.json`` (the key: response id, behavior, variant, role, question; private). Empty cells
are listed in the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from residual_disposition.behaviors import answer_text
from residual_disposition.config import load_config
from residual_disposition.judge import load_targets
from residual_disposition.records import IACalibrationResponse, read_frame, read_responses
from residual_disposition.report import run_dirs_from
from residual_disposition.tracking import Run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--per-cell", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--exclude", nargs="*", default=[], help="mapping.json of earlier sets")
    args = parser.parse_args()

    runs = run_dirs_from(args.runs)  # completed runs only, one per cell
    responses = read_frame(runs, IACalibrationResponse.table)
    texts = read_responses(runs, IACalibrationResponse.table)
    index = read_frame(runs, "state_index").drop_duplicates("state_id").set_index("state_id")
    excluded = {
        e["response_id"]
        for path in args.exclude
        for e in json.loads(Path(path).read_text())["items"]
    }
    responses = responses[~responses["response_id"].isin(excluded)]
    responses = responses.assign(
        behavior_id=responses["state_id"].map(index["behavior_id"]),
        role=responses["state_id"].map(index["kind"]),
    )
    unknown = responses[responses["role"].isna()]
    if len(unknown):
        raise SystemExit(f"{len(unknown)} calibration responses of states missing from the index")
    behaviors = sorted(set(responses["behavior_id"]))
    variants = sorted(set(responses["readout_variant"]))
    targets = load_targets(behaviors)
    rng = random.Random(args.seed)
    chosen, empty = [], []
    for behavior in behaviors:
        for variant in variants:
            for role in args.roles:
                cell = responses[
                    (responses["behavior_id"] == behavior)
                    & (responses["readout_variant"] == variant)
                    & (responses["role"] == role)
                ].sort_values(["question_id", "state_id", "sample"])
                by_question = defaultdict(list)
                for row in cell.itertuples():
                    by_question[row.question_id].append(row)
                questions = sorted(by_question)
                if len(questions) < args.per_cell:
                    empty.append(f"{behavior}/{variant}/{role}: {len(questions)} questions")
                for q in rng.sample(questions, min(args.per_cell, len(questions))):
                    row = rng.choice(by_question[q])
                    chosen.append((behavior, variant, role, row))
    rng.shuffle(chosen)
    ids = random.Random(args.seed + 1)
    items, key = [], []
    for behavior, variant, role, row in chosen:
        item_id = f"{ids.getrandbits(48):012x}"
        items.append(
            {
                "item_id": item_id,
                "target": targets[behavior],
                "response": answer_text(texts[row.response_id]),
            }
        )
        key.append(
            {
                "item_id": item_id,
                "response_id": row.response_id,
                "target_behavior_id": behavior,
                "readout_variant": variant,
                "role": role,
                "question_id": row.question_id,
                "state_id": row.state_id,
            }
        )
    blob = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    cfg = load_config(
        "base",
        [
            "+experiment=judge_calibration_sample",
            f"seed={args.seed}",
            "tracking.wandb.enabled=false",
        ],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        run.path("mapping.json").write_text(
            json.dumps({"set_id": set_id, "items": key}, indent=2) + "\n"
        )
        summary = {
            "set_id": set_id,
            "items": len(items),
            "roles": args.roles,
            "per_cell": args.per_cell,
            "seed": args.seed,
            "empty_cells": empty,
            "runs": len(runs),
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(f"input: {run.dir / 'input.jsonl'}; key (private): {run.dir / 'mapping.json'}")


if __name__ == "__main__":
    main()
