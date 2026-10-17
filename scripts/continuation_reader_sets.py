"""Labeling sets that validate rule C's judge on the continuation's new reader formats
(preregistration §12.3.3: lens-only per layer, PD-clean; at least 200 stratified hypotheses each,
before any confirmatory interpretation).

    uv run python scripts/continuation_reader_sets.py <lens30|lens20|pdclean> \\
        <summarizer run> <judge run> [--index data/judge_inputs/continuation_state_index.jsonl] \\
        [--seed 20261018] [--min-distinct 200]

Strata are state × package type, with the coordinator's quotas (2026-10-17): lens30, 6 of
adl_lens@30/state/1 and 2 of adl_lens@30/base/1 per state; lens20, 8 of adl_lens@20/state/1;
pdclean, 7 of each clean package (pd_clean) and 2 of each random-removal control (pd_random).

- **Inventory** (review s2t): the reading run must hold exactly the planned states of each
  package type (lens: the 28 states of the continuation's reread plans; pdclean: the plan's 26
  clean packages and its 14 random controls), each with samples 0–79 once; anything else is
  refused.
- **Distinct pairs:** the judge's label depends only on (target, text), so the set holds each pair
  once. Within a stratum the samples are taken in a seeded order, skipping a pair already chosen
  in this set or present in a past hypothesis calibration set (runs/hypothesis_calibration_set/,
  labels already checked); a stratum's shortfall is drawn from the other states of its package
  type, in a seeded order. Fewer than ``--min-distinct`` pairs is refused. The summary counts the
  skips (repeats, past sets) and the shortfalls before and after the refill; mapping.json holds the
  realized strata.
- **The judge's labels:** every item needs the judge run's own v4 label (strict and lenient, none
  null). Targets come from the judge run's recorded inputs, not from today's configs, and each
  label's strict and lenient requests, read from the judge's cache, must be the rubrics' messages
  for exactly that (target, text), with matching cache keys and models.

Writes, in its own run directory, ``input.jsonl`` ({item_id, target, response}, shuffled), the
closed ``mapping.json`` and ``frozen_pair_labels.jsonl``, in hypothesis_calibration_set.py's format;
item ids come from the format's seeded stream and skip every id already taken.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from residual_disposition.judge import JudgeCache, request_key, rubric_set
from residual_disposition.paths import REPO_ROOT, data_root, runs_root
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
SAMPLES = 80
RUBRIC_VERSION = "v4"
SEED = 20261018
INDEX = Path("data/judge_inputs/continuation_state_index.jsonl")


def _calibration():
    """hypothesis_calibration_set.py's taken-id rule and id stream, reused as they are."""
    path = REPO_ROOT / "scripts" / "hypothesis_calibration_set.py"
    spec = importlib.util.spec_from_file_location("hypothesis_calibration_set", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_states(fmt: str) -> dict[str, set[str]]:
    """Per package type of the format, the states the plans read it for."""
    if fmt == "pdclean":
        plan = yaml.safe_load((REPO_ROOT / "configs/continuation/pd_clean.yaml").read_text())
        states = plan["pd_clean"]["states"]
        return {
            "pd_clean": {s["state"] for s in states},
            "pd_random": {s["state"] for s in states if s["random"]},
        }
    from omegaconf import OmegaConf

    from residual_disposition.config import load_config

    lens: set[str] = set()
    for behavior in ("example_link", "pun_intended"):
        plan = load_config(f"experiments/continuation_14b/{behavior}").pipeline.plan
        lens |= set(OmegaConf.to_container(plan.reread, resolve=True)["states"])
    return {tool: set(lens) for tool, _ in FORMATS[fmt]}


def check_inventory(hyps, expected: dict[str, set[str]]) -> None:
    """Exactly the expected states for each package type, each with samples 0–79 once."""
    for tool, states in expected.items():
        mine = hyps[hyps["tool"] == tool]
        found = set(mine["state_id"])
        if found != states:
            missing, extra = sorted(states - found), sorted(found - states)
            raise SystemExit(
                f"{tool}: {len(missing)} planned states missing {missing[:3]}, "
                f"{len(extra)} unplanned {extra[:3]}"
            )
        for state, rows in mine.groupby("state_id"):
            samples = [int(s) for s in rows["sample"]]
            if len(samples) != len(set(samples)):
                raise SystemExit(f"{state}/{tool}: a sample is recorded twice")
            if sorted(samples) != list(range(SAMPLES)):
                raise SystemExit(f"{state}/{tool}: {len(samples)} samples, not 0–{SAMPLES - 1}")


def past_pairs(root: Path) -> set[tuple[str, str]]:
    """(target, text) of every item of a past hypothesis calibration set."""
    out = set()
    for path in sorted(Path(root).glob("hypothesis_calibration_set/*/input.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                out.add((item["target"], item["response"]))
    return out


def draw(hyps, pair_of, quotas, seed: int, fmt: str, past: set) -> tuple[list, dict]:
    """The chosen (row, refill) pairs and the report: per stratum, distinct pairs in seeded
    order, skipping repeats and past sets' pairs; a shortfall drawn from the other states of the
    same package type in seeded order."""
    chosen: list = []
    taken_pairs: set = set()
    report: dict[str, Any] = {"skipped_repeats": 0, "skipped_past": 0, "shortfall": {}}

    def take(row) -> bool:
        pair = pair_of(row)
        if pair in taken_pairs:
            report["skipped_repeats"] += 1
            return False
        if pair in past:
            report["skipped_past"] += 1
            return False
        taken_pairs.add(pair)
        return True

    for tool, n in quotas:
        rows = hyps[hyps["tool"] == tool]
        left: dict[str, list] = {}
        short: dict[str, int] = {}
        for state in sorted(rows["state_id"].unique()):
            mine = rows[rows["state_id"] == state].itertuples()
            order = sorted(mine, key=lambda r: int(r.sample))
            random.Random(f"{seed}:{fmt}:{state}:{tool}").shuffle(order)
            got = 0
            while order and got < n:
                row = order.pop(0)
                if take(row):
                    chosen.append((row, False))
                    got += 1
            left[state] = order
            if got < n:
                short[state] = n - got
        need = sum(short.values())
        pool = [r for s, rs in sorted(left.items()) if s not in short for r in rs]
        random.Random(f"{seed}:{fmt}:{tool}:refill").shuffle(pool)
        for row in pool:
            if need == 0:
                break
            if take(row):
                chosen.append((row, True))
                need -= 1
        report["shortfall"][tool] = {"before": short, "after_refill": need}
    return chosen, report


def judge_labels(judge_run: Path, rows, pair_of) -> list[dict]:
    """Each (row, behavior)'s frozen pair from the judge run, checked (module docstring)."""
    inputs = json.loads((judge_run / "judge_inputs.json").read_text())
    if inputs["rubric_version"] != RUBRIC_VERSION:
        raise SystemExit(f"{judge_run} judged with rubric {inputs['rubric_version']}, not v4")
    rubrics = rubric_set(RUBRIC_VERSION)
    for rubric in (rubrics.strict, rubrics.lenient):
        if inputs["rubrics"][rubric.version]["sha256"] != rubric.sha256:
            raise SystemExit(f"{judge_run}: rubric {rubric.version} is not today's text")
    config = yaml.safe_load((judge_run / "config.yaml").read_text())
    cache = JudgeCache(Path(config["judge"].get("cache_dir") or data_root() / "cache" / "judge"))
    labels = read_frame([judge_run], "judge_labels")
    labels = labels[labels["stage"] == "primary"].set_index(["response_id", "target_behavior_id"])
    if labels.index.duplicated().any():
        raise SystemExit(f"{judge_run}: a response has two primary labels for one target")
    out = []
    for row, behavior in rows:
        k = (row.response_id, behavior)
        if k not in labels.index:
            raise SystemExit(f"{row.response_id}: no label of the judge run for {behavior}")
        lab = labels.loc[k]
        if lab["rubric_version"] != RUBRIC_VERSION:
            raise SystemExit(f"{row.response_id}: a label of rubric {lab['rubric_version']}")
        if pd.isna(lab["strict_label"]) or pd.isna(lab["lenient_label"]):
            raise SystemExit(f"{row.response_id}: an incomplete label")
        target, text = pair_of(row)
        for rubric, rid, model in (
            (rubrics.strict, lab["request_id"], inputs["strict_model"]),
            (rubrics.lenient, lab["lenient_request_id"], inputs["lenient_model"]),
        ):
            entry = cache.entry(rid) if isinstance(rid, str) and rid else None
            if (
                entry is None
                or request_key(rubric.version, entry["body"]) != rid
                or entry["body"]["messages"] != rubric.messages(target, text)
                or entry["body"]["model"] != model
            ):
                raise SystemExit(f"{row.response_id}: its {rubric.kind} label is not for this pair")
        out.append(
            {
                "strict_label": lab["strict_label"],
                "lenient_label": bool(lab["lenient_label"]),
                "uncertain": bool(lab["uncertain"]),
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=sorted(FORMATS))
    parser.add_argument("summarizer_run", type=Path)
    parser.add_argument("judge_run", type=Path)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-distinct", type=int, default=200)
    args = parser.parse_args(argv)
    fmt, seed, quotas = args.format, args.seed, FORMATS[args.format]
    states: dict[str, tuple[str, str, int | None]] = {}
    for line in Path(args.index).read_text().splitlines():
        r = json.loads(line)
        if r["kind"] in ROLES:
            states[r["state_id"]] = (r["behavior_id"], ROLES[r["kind"]], r["dose"])
    hyps = read_frame([args.summarizer_run], ToolHypothesis.table)
    texts = read_responses([args.summarizer_run], ToolHypothesis.table)
    hyps = hyps[hyps["tool"].isin([t for t, _ in quotas])]
    check_inventory(hyps, expected_states(fmt))
    missing = sorted(set(hyps["state_id"]) - set(states))
    if missing:
        raise SystemExit(f"states without a role in the index: {missing}")
    targets = json.loads((args.judge_run / "judge_inputs.json").read_text())["targets"]

    def pair_of(row) -> tuple[str, str]:
        return targets[states[row.state_id][0]], texts[row.response_id]

    chosen, report = draw(hyps, pair_of, quotas, seed, fmt, past_pairs(runs_root()))
    if len(chosen) < args.min_distinct:
        raise SystemExit(f"{len(chosen)} distinct pairs, fewer than {args.min_distinct}")
    rows = [(r, states[r.state_id][0]) for r, _ in chosen]
    frozen = judge_labels(args.judge_run, rows, pair_of)
    order = list(range(len(chosen)))
    random.Random(f"{seed}:{fmt}:order").shuffle(order)
    calibration = _calibration()
    taken = calibration.taken_item_ids()
    item_ids = calibration.draw_item_ids(len(order), fmt, taken, seed)
    items, key, frozen_rows = [], [], []
    for item_id, i in zip(item_ids, order, strict=True):
        row, refill = chosen[i]
        behavior, role, dose = states[row.state_id]
        target, text = pair_of(row)
        items.append({"item_id": item_id, "target": target, "response": text})
        key.append(
            {
                "item_id": item_id,
                "response_id": row.response_id,
                "target_behavior_id": behavior,
                "role": role,
                "dose": dose,
                "tool": row.tool,
                "state_id": row.state_id,
                "sample": int(row.sample),
                "refill": refill,
            }
        )
        frozen_rows.append({"item_id": item_id, **frozen[i]})
    strata = dict(sorted(Counter(f"{k['state_id']}/{k['tool']}" for k in key).items()))
    blob = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    from residual_disposition.config import load_config

    cfg = load_config(
        "base",
        ["+experiment=continuation_reader_set", f"name={fmt}", f"seed={seed}"]
        + ["tracking.wandb.enabled=false"],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        mapping = {"set_id": set_id, "strata": strata, "items": key}
        run.path("mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
        run.path("frozen_pair_labels.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in frozen_rows)
        )
        summary = {
            "set_id": set_id,
            "format": fmt,
            "items": len(items),
            "distinct_pairs": len({(i["target"], i["response"]) for i in items}),
            "per_tool": dict(Counter(k["tool"] for k in key)),
            "quotas": dict(quotas),
            **report,
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
