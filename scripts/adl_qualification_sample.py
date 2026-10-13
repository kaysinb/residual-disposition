"""ADL's qualification stratum (preregistration §11.4): a blind labeling set of ADL's Q1
hypotheses, for the validation of counting rule C on ADL.

    uv run python scripts/adl_qualification_sample.py <summarizer run> <judge run>

- Random stratum: 5 hypotheses per behavior on each of ADL's active organism and sham (the
  frozen Q1 states), uniformly over the 20 samples, seed 20261014 (§3's calibration seed), drawn
  whatever the judge said: 8 x 2 x 5 = 80 items.
- Veto stratum: in addition, every ADL hypothesis of those states with the frozen pair's lenient
  "yes" and strict `avoids_behavior`, not already drawn. Vetoes on the recipe controls are counted
  in the summary, not drawn.

Only ADL's own hypotheses (activation_difference_lens, prompt adl-v1); the unsteered ablation is
not sampled. The target of an item is its behavior's ``ia_target``.

Writes, in its own run directory, ``input.jsonl`` with only {item_id, target, response}
(shuffled), ``mapping.json`` (the key: response id, behavior, role, state, sample, stratum) and
``frozen_pair_labels.jsonl`` (the judge run's labels of these items; no new judge calls).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from residual_disposition.config import load_config
from residual_disposition.judge import load_targets
from residual_disposition.records import ToolHypothesis, read_frame, read_responses
from residual_disposition.tracking import Run

CONFIGS = ["experiments/feasibility_14b/example_link", "experiments/feasibility_14b/pun_intended"]
TOOL = "activation_difference_lens"
PER_STATE = 5
ROLES = ("active", "sham")
SEED = 20261014


def frozen_roles() -> dict[str, tuple[str, str]]:
    """state id -> (behavior, role) for every frozen Q1 state."""
    from omegaconf import OmegaConf

    out: dict[str, tuple[str, str]] = {}
    for name in CONFIGS:
        cfg = load_config(name)
        states = OmegaConf.to_container(cfg.pipeline.plan.qualify.adl_states, resolve=True)
        for behavior, mine in states.items():
            out[mine["active"]] = (behavior, "active")
            out[mine["sham"]] = (behavior, "sham")
            for mode, sid in mine["recipe_controls"].items():
                out[sid] = (behavior, f"recipe_control_{mode}")
    return out


def main() -> None:
    summarizer_run, judge_run = Path(sys.argv[1]), Path(sys.argv[2])
    roles = frozen_roles()
    hyps = read_frame([summarizer_run], ToolHypothesis.table)
    texts = read_responses([summarizer_run], ToolHypothesis.table)
    hyps = hyps[(hyps["tool"] == TOOL) & hyps["state_id"].isin(roles)]
    if "budget" in hyps:
        hyps = hyps[hyps["budget"].isna()]
    hyps = hyps.assign(
        behavior=hyps["state_id"].map(lambda s: roles[s][0]),
        role=hyps["state_id"].map(lambda s: roles[s][1]),
    ).sort_values("response_id")
    labels = read_frame([judge_run], "judge_labels")
    labels = labels.set_index(["response_id", "target_behavior_id"])

    def frozen(row) -> dict | None:
        k = (row.response_id, row.behavior)
        return None if k not in labels.index else labels.loc[k]

    def vetoed(row) -> bool:
        lab = frozen(row)
        return (
            lab is not None
            and bool(lab["lenient_label"])
            and lab["strict_label"] == "avoids_behavior"
        )

    behaviors = sorted(hyps["behavior"].unique())
    if len(behaviors) != 8:
        raise SystemExit(f"expected 8 behaviors, found {behaviors}")
    targets = load_targets(behaviors)
    rng = random.Random(SEED)
    chosen = []
    for behavior in behaviors:
        for role in ROLES:
            cell = hyps[(hyps["behavior"] == behavior) & (hyps["role"] == role)]
            rows = list(cell.itertuples())
            if len(rows) < PER_STATE:
                raise SystemExit(f"{behavior}/{role}: {len(rows)} hypotheses")
            chosen.extend((r, "random") for r in rng.sample(rows, PER_STATE))
    drawn = {r.response_id for r, _ in chosen}
    in_roles = hyps[hyps["role"].isin(ROLES)]
    veto = [r for r in in_roles.itertuples() if vetoed(r) and r.response_id not in drawn]
    chosen.extend((r, "veto") for r in veto)
    controls = hyps[~hyps["role"].isin(ROLES)]
    control_vetoes = sum(vetoed(r) for r in controls.itertuples())
    rng.shuffle(chosen)
    ids = random.Random(SEED + 1)
    items, key, frozen_rows = [], [], []
    for row, stratum in chosen:
        item_id = f"{ids.getrandbits(48):012x}"
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
                "tool": row.tool,
                "state_id": row.state_id,
                "sample": int(row.sample),
                "stratum": stratum,
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
    blob = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    cfg = load_config(
        "base",
        ["+experiment=adl_qualification_sample", f"seed={SEED}", "tracking.wandb.enabled=false"],
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
            "items": len(items),
            "random": sum(s == "random" for _, s in chosen),
            "veto": len(veto),
            "recipe_control_vetoes_not_drawn": int(control_vetoes),
            "per_state": PER_STATE,
            "roles": list(ROLES),
            "seed": SEED,
            "summarizer_run": str(summarizer_run),
            "judge_run": str(judge_run),
            "frozen_pair_unlabeled": sum(f["strict_label"] is None for f in frozen_rows),
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(run.dir)


if __name__ == "__main__":
    main()
