"""Labeling sets that validate rule C's judge on the continuation's new reader formats
(preregistration §12.3.3: lens-only per layer, PD-clean; at least 200 stratified hypotheses each,
before any confirmatory interpretation).

    uv run python scripts/continuation_reader_sets.py <lens30|lens20|pdclean> <state index> \\
        <summarizer run> <judge run> [--seed 20261018]

Per state of the format's reading run, a seeded draw of samples from each package type the rules
later judge (coordinator, 2026-10-17):
- lens30: 6 from adl_lens@30/state/1 and 2 from adl_lens@30/base/1 (28 states, 224 items);
- lens20: 8 from adl_lens@20/state/1 (224 items);
- pdclean: 7 from each clean package (pd_clean, 26) and 2 from each random-removal control
  (pd_random, 14): 210 items.
Within a state the package types take disjoint sample indices: an H_s control equals its clean
package, so the same index would repeat its text. A state's role comes from its kind in the
analyst index (active A, sham H, suppressed L, suppressed_sham H_s, suppression_only X,
recipe_control recipe).

Writes, in its own run directory and in the format of hypothesis_calibration_set.py,
``input.jsonl`` ({item_id, target, response}, shuffled), the closed ``mapping.json`` (response
id, behavior, role, dose, tool, state, sample) and ``frozen_pair_labels.jsonl`` (the judge run's
labels of these items; no new judge calls). Item ids come from the format's seeded stream and
skip every id already taken.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.judge import load_targets
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import ToolHypothesis, read_frame, read_responses
from residual_disposition.tracking import Run

FORMATS = {
    "lens30": [("adl_lens@30/state/1", 6), ("adl_lens@30/base/1", 2)],
    "lens20": [("adl_lens@20/state/1", 8)],
    "pdclean": [("pd_clean", 7), ("pd_random", 2)],
}
ROLES = {
    "active": "A",
    "sham": "H",
    "suppressed": "L",
    "suppressed_sham": "H_s",
    "suppression_only": "X",
    "recipe_control": "recipe",
}
OPTIONAL = {"pd_random"}  # package types that only some states have
SEED = 20261018


def _calibration():
    """hypothesis_calibration_set.py's taken-id rule and id stream, reused as they are."""
    path = REPO_ROOT / "scripts" / "hypothesis_calibration_set.py"
    spec = importlib.util.spec_from_file_location("hypothesis_calibration_set", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def states_of(index: Path) -> dict[str, tuple[str, str, int | None]]:
    """state id -> (behavior, role, dose) of the analyst index's states with a known role."""
    out = {}
    for line in Path(index).read_text().splitlines():
        r = json.loads(line)
        if r["kind"] in ROLES:
            dose = None if r["dose"] is None else int(r["dose"])
            out[r["state_id"]] = (r["behavior_id"], ROLES[r["kind"]], dose)
    return out


def draw(hyps, quotas: list[tuple[str, int]], seed: int, fmt: str) -> list:
    """Per state, ``n`` samples of each package type of ``quotas``, the types' sample indices
    disjoint within the state; a state without a required type is refused."""
    chosen = []
    for state in sorted(hyps["state_id"].unique()):
        used: set[int] = set()
        mine = hyps[hyps["state_id"] == state]
        for tool, n in quotas:
            rows = mine[mine["tool"] == tool]
            if rows.empty:
                if tool in OPTIONAL:
                    continue  # a random-removal control is planned for some states only
                raise SystemExit(f"{state}: no {tool} hypotheses")
            free = sorted(int(s) for s in rows["sample"] if int(s) not in used)
            if len(free) < n:
                raise SystemExit(f"{state}/{tool}: {len(free)} samples left, {n} needed")
            take = random.Random(f"{seed}:{fmt}:{state}:{tool}").sample(free, n)
            used |= set(take)
            chosen += [r for r in rows.itertuples() if int(r.sample) in take]
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=sorted(FORMATS))
    parser.add_argument("index", type=Path)
    parser.add_argument("summarizer_run", type=Path)
    parser.add_argument("judge_run", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    fmt, seed, quotas = args.format, args.seed, FORMATS[args.format]
    states = states_of(args.index)
    hyps = read_frame([args.summarizer_run], ToolHypothesis.table)
    texts = read_responses([args.summarizer_run], ToolHypothesis.table)
    hyps = hyps[hyps["tool"].isin([t for t, _ in quotas])]
    unknown = sorted(set(hyps["state_id"]) - set(states))
    if unknown:
        raise SystemExit(f"states without a role in the index: {unknown}")
    rows = draw(hyps, quotas, seed, fmt)
    labels = read_frame([args.judge_run], "judge_labels").set_index(
        ["response_id", "target_behavior_id"]
    )
    calibration = _calibration()
    behaviors = sorted({states[r.state_id][0] for r in rows})
    targets = load_targets(behaviors)
    random.Random(f"{seed}:{fmt}:order").shuffle(rows)
    taken = calibration.taken_item_ids()
    item_ids = calibration.draw_item_ids(len(rows), fmt, taken, seed)
    items, key, frozen, counts = [], [], [], {}
    for item_id, r in zip(item_ids, rows, strict=True):
        behavior, role, dose = states[r.state_id]
        items.append(
            {"item_id": item_id, "target": targets[behavior], "response": texts[r.response_id]}
        )
        key.append(
            {
                "item_id": item_id,
                "response_id": r.response_id,
                "target_behavior_id": behavior,
                "role": role,
                "dose": dose,
                "tool": r.tool,
                "state_id": r.state_id,
                "sample": int(r.sample),
            }
        )
        k = (r.response_id, behavior)
        lab = labels.loc[k] if k in labels.index else None
        frozen.append(
            {
                "item_id": item_id,
                "strict_label": None if lab is None else lab["strict_label"],
                "lenient_label": None if lab is None else bool(lab["lenient_label"]),
                "uncertain": None if lab is None else bool(lab["uncertain"]),
            }
        )
        counts[r.tool] = counts.get(r.tool, 0) + 1
    blob = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    cfg = load_config(
        "base",
        [
            "+experiment=continuation_reader_set",
            f"name={fmt}",
            f"seed={seed}",
            "tracking.wandb.enabled=false",
        ],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        run.path("mapping.json").write_text(
            json.dumps({"set_id": set_id, "items": key}, indent=2) + "\n"
        )
        run.path("frozen_pair_labels.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in frozen)
        )
        summary = {
            "set_id": set_id,
            "format": fmt,
            "items": len(items),
            "per_tool": counts,
            "quotas": dict(quotas),
            "states": len({r.state_id for r in rows}),
            "frozen_pair_unlabeled": sum(f["strict_label"] is None for f in frozen),
            "taken_item_ids": len(taken),
            "seed": seed,
            "summarizer_run": str(args.summarizer_run),
            "judge_run": str(args.judge_run),
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(run.dir)


if __name__ == "__main__":
    main()
