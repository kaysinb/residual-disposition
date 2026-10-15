"""Labeling sets of the path-B continuation (preregistration §12.3.1, §12.9).

    uv run python scripts/continuation_labels.py detector-diagnostic [--runs runs/feasibility_14b]

``detector-diagnostic`` (§12.3.1): per behavior and input type, 150 detector-negative outputs of
the latent states, pooled over doses, for blind ensemble labeling of any trace of the behavior.
It bounds contamination among detector negatives (0/150 gives a 95 % upper bound of about 0.020),
not the miss rate among true positives. Input types available from the pilot:

- ``observer``: observer answers (measurements ``observer`` and ``observer_confirm``) on the
  certificate prompts, detector silent;
- ``pd``: PD continuations in the summarizer's top sets (per configuration, the top 1 % of the
  prefills by ΔPPL, the evidence PD-clean filters), the trace detector silent on the
  continuation.

ADL-corpus continuations join when Stage 1 has generated them. The set has the format of the
pilot's detector set (``input.jsonl`` with {item_id, behavior_id, behavior_description,
response}; ``key.json`` private), so the coordination tools merge and review it unchanged, and a
``mapping.json`` so that later sets skip its item ids. Item ids skip every id already taken.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.path_b import trace_detector
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import read_frame, read_responses
from residual_disposition.seeding import derive_seed
from residual_disposition.tracking import Run

STATE_INDEX = data_root() / "judge_inputs" / "feasibility_14b_state_index.jsonl"
BEHAVIORS = ("example_link", "pun_intended")
PER_STRATUM = 150
OBSERVER_MEASUREMENTS = ("observer", "observer_confirm")
TOP_FRACTION = 0.01


def taken_item_ids(runs: Path) -> set[str]:
    """Ids in the manual-label store and in every set's mapping or key under ``runs``."""
    from residual_disposition.judge import read_manual_labels

    taken = {label.item_id for label in read_manual_labels(data_root() / "labels" / "manual")}
    for path in [*runs.glob("*/*/mapping.json"), *runs.glob("*/*/key.json")]:
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        items = data.get("items", [])
        taken |= set(items) if isinstance(items, dict) else {e["item_id"] for e in items}
    return taken


def draw_item_ids(count: int, stream: str, taken: set[str]) -> list[str]:
    rng, out, used = random.Random(stream), [], set(taken)
    while len(out) < count:
        item_id = f"{rng.getrandbits(48):012x}"
        if item_id not in used:
            used.add(item_id)
            out.append(item_id)
    return out


def latent_states(index: Path) -> dict[str, tuple[str, int]]:
    """state id → (behavior, dose) of the latent (suppressed) states of the pilot behaviors."""
    out = {}
    for line in index.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "suppressed" and r["behavior_id"] in BEHAVIORS and r["dose"] is not None:
            out[r["state_id"]] = (r["behavior_id"], int(r["dose"]))
    return out


def observer_pool(cells: list[Path], states: dict[str, tuple[str, int]]) -> dict[str, list]:
    """Per behavior, (source, dose, text) of every detector-silent observer answer."""
    obs = read_frame(cells, "detector_obs")
    texts = read_responses(cells, "detector_obs")
    obs = obs[
        obs["state_id"].isin(states)
        & obs["measurement"].isin(OBSERVER_MEASUREMENTS)
        & ~obs["hit"].astype(bool)
    ]
    pool: dict[str, list] = {b: [] for b in BEHAVIORS}
    for row in obs.sort_values("response_id").itertuples():
        behavior, dose = states[row.state_id]
        if row.behavior_id != behavior:
            continue
        pool[behavior].append((f"observer:{row.response_id}", dose, texts[row.response_id]))
    return pool


def pd_pool(
    cells: list[Path], states: dict[str, tuple[str, int]], behaviors: dict
) -> dict[str, list]:
    """Per behavior, (source, dose, text) of every top-set PD continuation the trace detector
    does not flag."""
    from residual_disposition.tools import pd_top

    pool: dict[str, list] = {b: [] for b in BEHAVIORS}
    for cell in cells:
        state = OmegaConf.load(cell / "config.yaml").params.state
        if state not in states:
            continue
        behavior, dose = states[state]
        hit = trace_detector(behaviors[behavior].detector)
        with gzip.open(cell / "pd_completions.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for config, top in pd_top(rows, budget=None, fraction=TOP_FRACTION).items():
            for r in top:
                if not hit(r["completion"]):
                    source = f"pd:{cell.name}:{config}:{r['prefill']}"
                    pool[behavior].append((source, dose, r["completion"]))
    return pool


def detector_diagnostic(args: argparse.Namespace) -> None:
    rules = cr.load_rules()
    seed = int(rules["seeds"]["labels"])
    root = REPO_ROOT / args.runs
    states = latent_states(args.state_index)
    specs = {b: OmegaConf.to_container(load_config(f"behaviors/{b}").behavior) for b in BEHAVIORS}
    behaviors = {b: behavior_from_config(s) for b, s in specs.items()}
    rates = sorted(root.glob("*-rates-*"))
    tool_pd = sorted(root.glob("*-tool_pd-*"))
    pools = {
        "observer": observer_pool(rates, states),
        "pd": pd_pool(tool_pd, states, behaviors),
    }
    chosen = []
    pool_sizes: dict[str, dict[str, int]] = {}
    for kind, by_behavior in pools.items():
        for behavior in BEHAVIORS:
            pool = by_behavior[behavior]
            pool_sizes.setdefault(kind, {})[behavior] = len(pool)
            if len(pool) < PER_STRATUM:
                raise SystemExit(f"{kind}/{behavior}: only {len(pool)} negatives")
            rng = random.Random(derive_seed(seed, "detector_diagnostic", kind, behavior))
            chosen += [(kind, behavior, *p) for p in rng.sample(pool, PER_STRATUM)]
    random.Random(derive_seed(seed, "detector_diagnostic", "order")).shuffle(chosen)
    # Every item is a detector negative under the detector that defines its stratum.
    for kind, behavior, source, _, text in chosen:
        detector = behaviors[behavior].detector
        flagged = detector(text) if kind == "observer" else trace_detector(detector)(text)
        if flagged:
            raise SystemExit(f"{source}: the detector fires on a negative sample")
    taken = taken_item_ids(REPO_ROOT / "runs")
    item_ids = draw_item_ids(len(chosen), f"{seed}:detector_diagnostic:item-ids", taken)
    items, key, mapping = [], {}, []
    for item_id, (kind, behavior, source, dose, text) in zip(item_ids, chosen, strict=True):
        items.append(
            {
                "item_id": item_id,
                "behavior_id": behavior,
                "behavior_description": specs[behavior]["description"],
                "response": text,
            }
        )
        key[item_id] = {
            "behavior_id": behavior,
            "stratum": kind,
            "detector_hit": False,
            "dose": dose,
            "source": source,
            "detector_version": behaviors[behavior].detector.version,
        }
        mapping.append({"item_id": item_id, "behavior_id": behavior, "stratum": kind})
    blob = "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    if args.dry_run:
        print(json.dumps({"set_id": set_id, "items": len(items), "pool_sizes": pool_sizes}))
        return
    cfg = load_config(
        "base",
        ["+experiment=continuation_labels", "name=detector-diagnostic", f"seed={seed}"]
        + ["tracking.wandb.enabled=false"],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        meta = {"set_id": set_id, "seed": seed, "source_runs": str(args.runs)}
        run.path("key.json").write_text(json.dumps({**meta, "items": key}, indent=2) + "\n")
        run.path("mapping.json").write_text(
            json.dumps({"set_id": set_id, "items": mapping}, indent=2) + "\n"
        )
        summary = {
            "set_id": set_id,
            "items": len(items),
            "per_stratum": PER_STRATUM,
            "pool_sizes": pool_sizes,
            "latent_states": {s: list(v) for s, v in sorted(states.items())},
            "top_fraction": TOP_FRACTION,
            "taken_item_ids": len(taken),
            "seed": seed,
            "detector_versions": {b: behaviors[b].detector.version for b in BEHAVIORS},
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(run.dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("detector-diagnostic")
    diag.add_argument("--runs", default="runs/feasibility_14b")
    diag.add_argument("--state-index", type=Path, default=STATE_INDEX)
    diag.add_argument("--dry-run", action="store_true", help="report the pools; write nothing")
    args = parser.parse_args()
    if args.command == "detector-diagnostic":
        detector_diagnostic(args)


if __name__ == "__main__":
    main()
