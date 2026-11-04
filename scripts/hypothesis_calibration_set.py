"""The hypothesis calibration of counting rule C (plan, path B, step 10): a blind labeling set of
every hypothesis of one reader on latent, sham_s and sham_x at d1 and d2 of both behaviors.

    uv run python scripts/hypothesis_calibration_set.py <pd|adl> <state index> \\
        <summarizer run> <judge run> [--doses example_link=100,200] [--seed 20261018]

The defaults rebuild the pilot's sets (d1, d2 = 25, 50 of both behaviors, seed 20261014).
``--doses`` and ``--seed`` build the continuation's pending dose-100/200 set (preregistration
§12.3.3): example_link only, 6 states per reader, 120 items.

- PD: perplexity_differencing, prompt pd-v1, all prefills (budget null); ADL:
  activation_difference_lens, prompt adl-v1 (the unsteered ablation is not in the set).
- States: kinds suppressed (latent), suppressed_sham (sham_s) and suppression_only (sham_x) at
  the audit doses d1 and d2 of each behavior (the analyst state index), 20 samples each:
  2 x 6 x 20 = 240 items, none excluded; the frozen pair's vetoes (lenient "yes", strict
  avoids_behavior) are counted in the summary, not drawn apart.
- The target of an item is its behavior's ``ia_target``.

Writes, in its own run directory and in the format of set cda7cd689456, ``input.jsonl`` with
only {item_id, target, response} (shuffled, seed 20261014), ``mapping.json`` (the key: response
id, behavior, role, state, dose, sample) and ``frozen_pair_labels.jsonl`` (the judge run's
labels of these items; no new judge calls).

Item ids are drawn per reader (seed "20261014:<reader>:item-ids") and skip every id already in
the manual-label store or in another set's mapping. The ADL set d8fe2c3f1bf3 was built before
this rule, with ids from seed 20261015 that repeat 80 of set cda7cd689456; it is kept as built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.judge import load_targets, read_manual_labels
from residual_disposition.paths import data_root
from residual_disposition.records import ToolHypothesis, read_frame, read_responses
from residual_disposition.tracking import Run

READERS = {"pd": "perplexity_differencing", "adl": "activation_difference_lens"}
AUDIT_DOSES = {"example_link": (25, 50), "pun_intended": (25, 50)}
ROLES = {"suppressed": "latent", "suppressed_sham": "sham_s", "suppression_only": "sham_x"}
SAMPLES = 20
SEED = 20261014


def calibration_states(
    index: Path, audit_doses: dict[str, tuple[int, ...]] = AUDIT_DOSES
) -> dict[str, tuple[str, str, int]]:
    """state id -> (behavior, role, dose) for latent, sham_s and sham_x at the audit doses."""
    out = {}
    for line in index.read_text().splitlines():
        r = json.loads(line)
        doses = audit_doses.get(r["behavior_id"], ())
        if r["kind"] in ROLES and r["dose"] is not None and int(r["dose"]) in doses:
            out[r["state_id"]] = (r["behavior_id"], ROLES[r["kind"]], int(r["dose"]))
    expected = sum(len(d) for d in audit_doses.values()) * len(ROLES)
    if len(out) != expected:
        raise SystemExit(f"expected {expected} states, found {len(out)}: {sorted(out.values())}")
    return out


def taken_item_ids(runs: Path = Path("runs"), manual: Path | None = None) -> set[str]:
    """Item ids already in use: the manual-label store and every set's mapping under ``runs``."""
    store = data_root() / "labels" / "manual" if manual is None else manual
    taken = {label.item_id for label in read_manual_labels(store)}
    for path in runs.glob("*/*/mapping.json"):
        try:
            taken |= {e["item_id"] for e in json.loads(path.read_text()).get("items", [])}
        except (ValueError, AttributeError, TypeError, KeyError):
            continue
    return taken


def draw_item_ids(count: int, reader: str, taken: set[str], seed: int = SEED) -> list[str]:
    """``count`` distinct 12-hex item ids from the reader's own seeded stream, none in ``taken``."""
    ids, out, used = random.Random(f"{seed}:{reader}:item-ids"), [], set(taken)
    while len(out) < count:
        item_id = f"{ids.getrandbits(48):012x}"
        if item_id not in used:
            used.add(item_id)
            out.append(item_id)
    return out


def parse_doses(values: list[str] | None) -> dict[str, tuple[int, ...]]:
    if not values:
        return AUDIT_DOSES
    out = {}
    for value in values:
        behavior, _, doses = value.partition("=")
        out[behavior] = tuple(int(d) for d in doses.split(","))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reader", choices=sorted(READERS))
    parser.add_argument("index", type=Path)
    parser.add_argument("summarizer_run", type=Path)
    parser.add_argument("judge_run", type=Path)
    parser.add_argument("--doses", nargs="+", help="behavior=d1,d2 (default: the pilot's)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    reader, index = args.reader, args.index
    summarizer_run, judge_run, seed = args.summarizer_run, args.judge_run, args.seed
    audit_doses = parse_doses(args.doses)
    tool = READERS[reader]
    states = calibration_states(index, audit_doses)
    hyps = read_frame([summarizer_run], ToolHypothesis.table)
    texts = read_responses([summarizer_run], ToolHypothesis.table)
    hyps = hyps[(hyps["tool"] == tool) & hyps["state_id"].isin(states)]
    if "budget" in hyps:
        hyps = hyps[hyps["budget"].isna()]
    counts = hyps.groupby("state_id").size()
    short = {s: int(counts.get(s, 0)) for s in states if counts.get(s, 0) != SAMPLES}
    if short:
        raise SystemExit(f"states without exactly {SAMPLES} {tool} hypotheses: {short}")
    hyps = hyps.assign(
        behavior=hyps["state_id"].map(lambda s: states[s][0]),
        role=hyps["state_id"].map(lambda s: states[s][1]),
        dose=hyps["state_id"].map(lambda s: states[s][2]),
    ).sort_values("response_id")
    labels = read_frame([judge_run], "judge_labels")
    labels = labels.set_index(["response_id", "target_behavior_id"])

    def frozen(row) -> dict | None:
        k = (row.response_id, row.behavior)
        return None if k not in labels.index else labels.loc[k]

    targets = load_targets(sorted(hyps["behavior"].unique()))
    rows = list(hyps.itertuples())
    random.Random(seed).shuffle(rows)
    taken = taken_item_ids()  # recorded: a later build may draw other ids
    item_ids = draw_item_ids(len(rows), reader, taken, seed)
    items, key, frozen_rows, vetoes = [], [], [], 0
    for item_id, row in zip(item_ids, rows, strict=True):
        items.append(
            {
                "item_id": item_id,
                "target": targets[row.behavior],
                "response": texts[row.response_id],
            }
        )
        key.append(
            {
                "item_id": item_id,
                "response_id": row.response_id,
                "target_behavior_id": row.behavior,
                "role": row.role,
                "dose": int(row.dose),
                "tool": row.tool,
                "state_id": row.state_id,
                "sample": int(row.sample),
            }
        )
        lab = frozen(row)
        frozen_rows.append(
            {
                "item_id": item_id,
                "strict_label": None if lab is None else lab["strict_label"],
                "lenient_label": None if lab is None else bool(lab["lenient_label"]),
                "uncertain": None if lab is None else bool(lab["uncertain"]),
            }
        )
        vetoes += (
            lab is not None
            and bool(lab["lenient_label"])
            and (lab["strict_label"] == "avoids_behavior")
        )
    blob = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    cfg = load_config(
        "base",
        ["+experiment=hypothesis_calibration_set", f"seed={seed}", "tracking.wandb.enabled=false"],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        run.path("mapping.json").write_text(
            json.dumps({"set_id": set_id, "items": key}, indent=2) + "\n"
        )
        run.path("frozen_pair_labels.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in frozen_rows)
        )
        summary = {
            "set_id": set_id,
            "reader": reader,
            "tool": tool,
            "items": len(items),
            "states": len(states),
            "per_state": SAMPLES,
            "roles": list(ROLES.values()),
            "audit_doses": {b: list(d) for b, d in audit_doses.items()},
            "frozen_pair_vetoes": int(vetoes),
            "taken_item_ids": len(taken),
            "seed": seed,
            "summarizer_run": str(summarizer_run),
            "judge_run": str(judge_run),
            "frozen_pair_unlabeled": sum(f["strict_label"] is None for f in frozen_rows),
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(run.dir)


if __name__ == "__main__":
    main()
