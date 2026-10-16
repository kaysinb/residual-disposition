"""PD-clean packages of the continuation's Stage 1 (preregistration §12.4.2, §12.3.2).

    uv run python scripts/pd_clean.py round [--work DIR]
    uv run python scripts/pd_clean.py import --labels FILE [--work DIR]
    uv run python scripts/pd_clean.py package [--work DIR]

The states, their roles and the adjudication plan are configs/continuation/pd_clean.yaml; each
state's PD output is its pilot tool_pd run. The work directory (default <DATA_DIR>/pd_clean)
keeps one adjudication record per state and every round.

- ``round`` writes the next round's blind items for the ensemble (``round<k>/items.jsonl``, the
  detector diagnostic's format: item id, behavior id and description, and the response, the
  prefill in brackets then its continuation; all states pooled and shuffled; no state, role,
  dose, configuration or rank). The mapping to states stays in ``round<k>/mapping.json``, not
  for the annotators. States without adjudication, and states
  whose selection is fully labeled, add nothing; a round with no item means the adjudication is
  finished.
- ``import`` records a round's final labels (JSON lines of ``{"item_id", "present", "confident",
  "evidence"}``, present meaning a trace), every item of the round exactly once.
- ``package`` refuses while any adjudicated state has unlabeled selected items; otherwise it
  writes one tracked run per state (experiment pd_clean, step pd_clean_package) with the clean
  selection (``pd_clean.json``: the selected rows per configuration, every removal with its
  reason, the counts), and, for the states configured for it, the random-removal control (step
  pd_random_package) with as many items removed from PD's plain top set as the clean selection
  removed there. summarize_tools reads both with PD's prompt.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from residual_disposition import pd_clean as C
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.paths import CONFIG_ROOT, data_root, runs_root
from residual_disposition.pipeline import read_jsonl_gz
from residual_disposition.tracking import Run

PLAN = CONFIG_ROOT / "continuation" / "pd_clean.yaml"


def load_plan(path: Path = PLAN) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())["pd_clean"]


def pd_run(state_id: str, experiments: list[str]) -> Path:
    """The completed tool_pd run of a state; none or several is an error."""
    found = []
    for experiment in experiments:
        for cfg_path in sorted((runs_root() / experiment).glob("*-tool_pd-*/config.yaml")):
            cfg = yaml.safe_load(cfg_path.read_text())
            meta = json.loads((cfg_path.parent / "meta.json").read_text())
            if cfg["params"]["state"] == state_id and meta.get("status") == "completed":
                found.append(cfg_path.parent)
    if len(found) != 1:
        raise SystemExit(f"{state_id}: {len(found)} completed tool_pd runs, not one: {found}")
    return found[0]


def behavior(behavior_id: str):
    spec = yaml.safe_load((CONFIG_ROOT / "behaviors" / f"{behavior_id}.yaml").read_text())
    return behavior_from_config(spec["behavior"])


def detector(behavior_id: str):
    from residual_disposition.path_b import trace_detector

    b = behavior(behavior_id)
    return trace_detector(b.detector), b.detector.version


def numeric(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "ppl_diff": float(r["ppl_diff"])} for r in rows]


class Work:
    """The work directory: per state its adjudication record; per round its items."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, state_id: str) -> C.Adjudication:
        path = self.root / "states" / f"{state_id}.json"
        if not path.exists():
            return C.Adjudication(state_id)
        return C.Adjudication.from_dict(json.loads(path.read_text()))

    def save(self, adj: C.Adjudication) -> None:
        path = self.root / "states" / f"{adj.state_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(adj.as_dict(), indent=1) + "\n")

    def rounds(self) -> list[Path]:
        return sorted(self.root.glob("round*/mapping.json"), key=lambda p: int(p.parent.name[5:]))


def states_of(plan: Mapping[str, Any]):
    for s in plan["states"]:
        rows = numeric(
            read_jsonl_gz(pd_run(s["state"], plan["pd_runs"]) / "pd_completions.jsonl.gz")
        )
        hit, version = detector(s["behavior"])
        yield s, rows, hit, version


def cmd_round(work: Work, plan: Mapping[str, Any]) -> None:
    done = [p for p in work.rounds()]
    pending = [p for p in done if not (p.parent / "imported.json").exists()]
    if pending:
        raise SystemExit(f"{pending[0].parent} is not imported yet")
    k = len(done)
    entries, mapping = [], {}
    for s, rows, hit, _ in states_of(plan):
        if not s["adjudicate"]:
            continue
        _, todo = C.next_round(rows, work.record(s["state"]), hit=hit, fraction=plan["fraction"])
        b = behavior(s["behavior"])
        for iid, (config, row) in todo.items():
            entries.append(
                {
                    "item_id": iid,
                    "behavior_id": b.id,
                    "behavior_description": b.description,
                    "response": C.item_response(row),
                }
            )
            mapping[iid] = {"state": s["state"], "config": config, "prefill": row["prefill"]}
    if not entries:
        print("no item left: the adjudication is finished; run `package`")
        return
    out = work.root / f"round{k}"
    out.mkdir()
    with (out / "items.jsonl").open("w") as f:
        for item in C.blind_items(entries, seed=int(plan["seed"]) + k):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out / "mapping.json").write_text(json.dumps(mapping, indent=1) + "\n")
    print(f"round {k}: {len(entries)} items for the ensemble in {out / 'items.jsonl'}")


def cmd_import(work: Work, plan: Mapping[str, Any], labels_path: Path) -> None:
    rounds = [p for p in work.rounds() if not (p.parent / "imported.json").exists()]
    if len(rounds) != 1:
        raise SystemExit(f"{len(rounds)} rounds await their labels, not one")
    mapping = json.loads(rounds[0].read_text())
    labels = {}
    for line in labels_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["item_id"] in labels:
                raise SystemExit(f"{row['item_id']} is labeled twice")
            if not isinstance(row.get("present"), bool):
                raise SystemExit(f"{row['item_id']}: 'present' must be true or false")
            labels[row["item_id"]] = {
                "present": row["present"],
                "confident": row.get("confident"),
                "evidence": row.get("evidence"),
            }
    if set(labels) != set(mapping):
        raise SystemExit(
            f"labels for {len(set(labels) & set(mapping))} of the round's {len(mapping)} items, "
            f"{len(set(labels) - set(mapping))} unknown"
        )
    by_state: dict[str, dict[str, dict[str, Any]]] = {}
    for iid, m in mapping.items():
        by_state.setdefault(m["state"], {})[iid] = labels[iid]
    for s, rows, hit, _ in states_of(plan):
        if s["state"] not in by_state:
            continue
        adj = work.record(s["state"])
        _, todo = C.next_round(rows, adj, hit=hit, fraction=plan["fraction"])
        adj.record(todo, by_state[s["state"]])
        work.save(adj)
    (rounds[0].parent / "imported.json").write_text(
        json.dumps({"labels": str(labels_path), "items": len(labels)}) + "\n"
    )
    traces = sum(v["present"] for v in labels.values())
    print(f"{rounds[0].parent.name}: {len(labels)} labels, {traces} traces")


def tracking() -> dict[str, Any]:
    from residual_disposition.config import load_config

    t = OmegaConf.to_container(load_config("base").tracking, resolve=True)
    t.setdefault("wandb", {})["enabled"] = False
    return t


def write_package(step: str, s: Mapping[str, Any], sel: C.Selection, extra: Mapping[str, Any]):
    cfg = OmegaConf.create(
        {
            "experiment": "pd_clean",
            "name": step,
            "step": step,
            "params": {
                "state": s["state"],
                "behavior": s["behavior"],
                "role": s["role"],
                "dose": s["dose"],
                **extra,
            },
            "tracking": tracking(),
        }
    )
    with Run(cfg) as run:
        run.path("pd_clean.json").write_text(
            json.dumps({"state": s["state"], "top": sel.top, "removed": sel.removed}) + "\n"
        )
        run.summarize(C.package_summary(sel, None))
    return run.dir


def cmd_package(work: Work, plan: Mapping[str, Any]) -> None:
    states = list(states_of(plan))
    for s, rows, hit, _ in states:
        if s["adjudicate"]:
            _, todo = C.next_round(
                rows, work.record(s["state"]), hit=hit, fraction=plan["fraction"]
            )
            if todo:
                raise SystemExit(f"{s['state']}: {len(todo)} selected items are not labeled yet")
    for s, rows, hit, version in states:
        adj = work.record(s["state"]) if s["adjudicate"] else None
        sel = C.clean_selection(
            rows, hit=hit, excluded=adj.excluded() if adj else {}, fraction=plan["fraction"]
        )
        source = str(pd_run(s["state"], plan["pd_runs"]))
        clean = write_package(
            "pd_clean_package",
            s,
            sel,
            {
                "pd_run": source,
                "detector_version": version,
                "adjudicated": s["adjudicate"],
                "adjudication": C.package_summary(sel, adj) if adj else None,
            },
        )
        print(f"{s['state']} ({s['behavior']} {s['role']} {s['dose']}): {clean}")
        if s["random"]:
            counts = C.removed_from_top(rows, sel, fraction=plan["fraction"])
            ctl = C.random_removal(
                rows, counts=counts, fraction=plan["fraction"], seed=int(plan["seed"])
            )
            extra = {"pd_run": source, "counts": counts}
            print(f"  random control: {write_package('pd_random_package', s, ctl, extra)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=PLAN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("round")
    p = sub.add_parser("import")
    p.add_argument("--labels", type=Path, required=True)
    sub.add_parser("package")
    args = parser.parse_args(argv)
    work, plan = Work(args.work or data_root() / "pd_clean"), load_plan(args.plan)
    if args.command == "round":
        cmd_round(work, plan)
    elif args.command == "import":
        cmd_import(work, plan, args.labels)
    else:
        cmd_package(work, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
