"""PD-clean packages of the continuation's Stage 1 (preregistration §12.4.2, §12.3.2).

    uv run python scripts/pd_clean.py round [--work DIR]
    uv run python scripts/pd_clean.py import --labels FILE --rubric FILE [--work DIR]
    uv run python scripts/pd_clean.py review [--work DIR]
    uv run python scripts/pd_clean.py import-review --labels FILE [--work DIR]
    uv run python scripts/pd_clean.py package [--work DIR]

The states, their roles, the adjudication plan and the review sample are
configs/continuation/pd_clean.yaml; each state's PD output is its pilot tool_pd run. The work
directory (default <DATA_DIR>/pd_clean) holds immutable records only; every state's labels are
derived from them each time, never kept as mutable state.

- **Binding.** Every record carries the binding it was made under: the plan's sha256, each
  state's PD run and the sha256 of its completions, each behavior's detector version, and the
  ensemble's rubric (instructions) sha256. Every command refuses records of another binding.
- ``round`` writes the next ensemble round (``round<k>/items.jsonl``, the detector diagnostic's
  format: item id, behavior id and description, the response, the prefill in brackets then its
  continuation; pooled and shuffled; no state, role, dose, configuration or rank). Its private
  ``mapping.json`` (mode 600) maps items to states and is not for the annotators.
- ``import`` takes the round's final ensemble labels, JSON lines of ``{"item_id", "present",
  "confident", "evidence", "unanimous"}`` (``present`` a trace, ``unanimous`` false for an
  arbitrated dispute), every item of the round exactly once, and writes ``labels.json``
  atomically, once; a second import of the same labels is a no-op, of other labels an error.
- ``review`` writes the researcher's review set (§12.3.2, step 3): every dispute and every
  unanimous trace not yet reviewed, and once per package a seeded sample of its agreed
  (unanimous) negatives of the size the plan freezes (30 per package now).
- ``import-review`` takes the researcher's final labels (``{"item_id", "present", "note"}``) for
  exactly the review's items, once. A researcher's label overrides the ensemble's; a change
  reruns the removal and the refill, so the next ``round`` may bring new items, whose disputes
  and traces go to a later review.
- ``package`` refuses while a round or a review is pending or needed; then writes one tracked
  run per state (experiment pd_clean, step pd_clean_package) with the clean selection and every
  removal with its reason, and, for the states configured for it, the random-removal control
  (step pd_random_package, ``pd_clean.random_window``). summarize_tools reads both with PD's
  prompt and the selected texts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def write_once(path: Path, data: Any, *, private: bool = False) -> None:
    """Write a record atomically, once: the same content again is a no-op, other content an
    error."""
    blob = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != blob:
            raise SystemExit(f"{path} exists with other content: records are immutable")
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    if private:
        tmp.chmod(0o600)
    os.replace(tmp, path)


class Study:
    """The plan, the PD outputs and the records of one work directory."""

    def __init__(self, work: Path, plan_path: Path = PLAN) -> None:
        from residual_disposition.path_b import trace_detector

        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.plan_path = Path(plan_path)
        self.plan = yaml.safe_load(self.plan_path.read_text())["pd_clean"]
        self.fraction = float(self.plan["fraction"])
        self.seed = int(self.plan["seed"])
        self.states, self.rows, self.hits, self.runs = [], {}, {}, {}
        detectors = {}
        for s in self.plan["states"]:
            run = pd_run(s["state"], self.plan["pd_runs"])
            rows = read_jsonl_gz(run / "pd_completions.jsonl.gz")
            self.rows[s["state"]] = [{**r, "ppl_diff": float(r["ppl_diff"])} for r in rows]
            b = behavior(s["behavior"])
            self.hits[s["state"]] = trace_detector(b.detector)
            detectors[b.id] = b.detector.version
            self.runs[s["state"]] = run
            self.states.append(s)
        self.binding = {
            "plan_sha256": sha256(self.plan_path),
            "pd_runs": {
                sid: {"run": run.name, "sha256": sha256(run / "pd_completions.jsonl.gz")}
                for sid, run in self.runs.items()
            },
            "detectors": detectors,
        }

    # --- records -----------------------------------------------------------------------------

    def records(self, kind: str) -> list[Path]:
        dirs = [p for p in self.work.glob(f"{kind}[0-9]*") if p.is_dir()]
        return sorted(dirs, key=lambda p: int(p.name[len(kind) :]))

    def check(self, record: Mapping[str, Any], where: Path) -> None:
        bound = {k: v for k, v in record["binding"].items() if k != "rubric_sha256"}
        if bound != self.binding:
            raise SystemExit(f"{where} was made under another binding (plan, PD runs or detectors)")

    def pending(self) -> Path | None:
        """A round or review written but not imported yet."""
        for kind in ("round", "review"):
            for d in self.records(kind):
                if not (d / "labels.json").exists():
                    return d
        return None

    def rubric(self) -> str | None:
        shas = {
            json.loads((d / "labels.json").read_text())["binding"]["rubric_sha256"]
            for d in self.records("round")
            if (d / "labels.json").exists()
        }
        if len(shas) > 1:
            raise SystemExit(f"the rounds were labeled under {len(shas)} rubrics")
        return next(iter(shas), None)

    def ensemble(self) -> dict[str, dict[str, Any]]:
        """Every ensemble label by item id, with its state, config and prefill."""
        out = {}
        for d in self.records("round"):
            if not (d / "labels.json").exists():
                continue
            mapping = json.loads((d / "mapping.json").read_text())
            labels = json.loads((d / "labels.json").read_text())
            self.check(mapping, d)
            self.check(labels, d)
            for iid, label in labels["labels"].items():
                out[iid] = {**mapping["items"][iid], **label, "round": d.name}
        return out

    def reviews(self) -> tuple[dict[str, dict[str, Any]], set[str]]:
        """The researcher's labels by item id, and the states whose negative sample is drawn."""
        out, sampled = {}, set()
        for d in self.records("review"):
            mapping = json.loads((d / "mapping.json").read_text())
            self.check(mapping, d)
            sampled |= set(mapping["sampled_states"])
            if not (d / "labels.json").exists():
                continue
            labels = json.loads((d / "labels.json").read_text())
            self.check(labels, d)
            for iid, label in labels["labels"].items():
                out[iid] = {**mapping["items"][iid], **label, "review": d.name}
        return out, sampled

    def final(self) -> dict[str, dict[str, Any]]:
        """Each labeled item's final label: the researcher's where reviewed, else the
        ensemble's."""
        ensemble, (reviewed, _) = self.ensemble(), self.reviews()
        return {iid: reviewed.get(iid, label) for iid, label in ensemble.items()}

    def selection(
        self, s: Mapping[str, Any], final: Mapping[str, Mapping[str, Any]]
    ) -> C.Selection:
        mine = [v for v in final.values() if v["state"] == s["state"]] if s["adjudicate"] else []
        return C.clean_selection(
            self.rows[s["state"]],
            hit=self.hits[s["state"]],
            excluded=C.excluded(mine),
            fraction=self.fraction,
        )

    def todo(self) -> dict[str, tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
        """Every adjudicated state's selected items without an ensemble label: id → (state,
        config, row)."""
        final, out = self.final(), {}
        for s in self.states:
            if s["adjudicate"]:
                sel = self.selection(s, final)
                for iid, (config, row) in C.unlabeled(sel, s["state"], final).items():
                    out[iid] = (s, config, row)
        return out

    def review_needs(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """The items the researcher must still review (id → their record, with why), and the
        states whose negative sample this review draws."""
        ensemble, (reviewed, sampled) = self.ensemble(), self.reviews()
        final, needs, draw = self.final(), {}, []
        for iid, label in ensemble.items():
            if iid in reviewed:
                continue
            if not label["unanimous"]:
                needs[iid] = {**label, "why": "dispute"}
            elif label["present"]:
                needs[iid] = {**label, "why": "trace"}
        n = int(self.plan["review"]["negatives_per_package"])
        for s in self.states:
            if not s["adjudicate"] or s["state"] in sampled:
                continue
            sel = self.selection(s, final)
            agreed = sorted(
                iid
                for iid in sel.ids(s["state"])
                if iid in ensemble and ensemble[iid]["unanimous"] and not ensemble[iid]["present"]
            )
            rng = random.Random(f"{self.seed}/review/{s['state']}")
            for iid in rng.sample(agreed, min(n, len(agreed))):
                needs[iid] = {**ensemble[iid], "why": "sample"}
            draw.append(s["state"])
        return needs, draw


# --- commands -------------------------------------------------------------------------------


def entry(iid: str, s: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, str]:
    b = behavior(s["behavior"])
    return {
        "item_id": iid,
        "behavior_id": b.id,
        "behavior_description": b.description,
        "response": C.item_response(row),
    }


def cmd_round(st: Study) -> None:
    if (p := st.pending()) is not None:
        raise SystemExit(f"{p} is not imported yet")
    todo = st.todo()
    if not todo:
        print("no item left to label: run `review`, or `package` once the review is complete")
        return
    k = len(st.records("round"))
    out = st.work / f"round{k}"
    out.mkdir()
    entries = [entry(iid, s, row) for iid, (s, _, row) in todo.items()]
    with (out / "items.jsonl").open("w") as f:
        for item in C.blind_items(entries, seed=st.seed + k):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    mapping = {
        "binding": st.binding,
        "items": {
            iid: {"state": s["state"], "config": config, "prefill": row["prefill"]}
            for iid, (s, config, row) in todo.items()
        },
    }
    write_once(out / "mapping.json", mapping, private=True)
    print(f"round {k}: {len(entries)} items for the ensemble in {out / 'items.jsonl'}")


def read_labels(path: Path, fields: Mapping[str, type]) -> dict[str, dict[str, Any]]:
    labels = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = row.get("item_id")
        if iid in labels:
            raise SystemExit(f"{iid} is labeled twice")
        for name, kind in fields.items():
            if not isinstance(row.get(name), kind):
                raise SystemExit(f"{iid}: {name!r} must be a {kind.__name__}")
        labels[iid] = {k: v for k, v in row.items() if k != "item_id"}
    return labels


def match(labels: Mapping[str, Any], items: Mapping[str, Any], what: str) -> None:
    if set(labels) != set(items):
        known = len(set(labels) & set(items))
        raise SystemExit(
            f"labels for {known} of the {what}'s {len(items)} items, "
            f"{len(set(labels) - set(items))} unknown"
        )


def target(records: list[Path], what: str) -> list[Path]:
    """The record an import writes: the one awaiting its labels, or, when none does, the last
    one (so that importing the same labels again is a no-op and other labels are refused)."""
    waiting = [d for d in records if not (d / "labels.json").exists()]
    if len(waiting) > 1:
        raise SystemExit(f"{len(waiting)} {what}s await their labels, not one")
    if not waiting and not records:
        raise SystemExit(f"no {what} to import into")
    return waiting or records[-1:]


def cmd_import(st: Study, labels_path: Path, rubric_path: Path) -> None:
    rounds = target(st.records("round"), "round")
    mapping = json.loads((rounds[0] / "mapping.json").read_text())
    st.check(mapping, rounds[0])
    rubric = sha256(rubric_path)
    imported = (rounds[0] / "labels.json").exists()
    if not imported and (known := st.rubric()) not in (None, rubric):
        raise SystemExit(f"the rubric {rubric_path} is not the earlier rounds' ({known})")
    labels = read_labels(labels_path, {"present": bool, "unanimous": bool})
    match(labels, mapping["items"], "round")
    record = {
        "binding": {**st.binding, "rubric_sha256": rubric},
        "source": {"labels": str(labels_path), "sha256": sha256(labels_path)},
        "labels": labels,
    }
    write_once(rounds[0] / "labels.json", record)
    traces = sum(v["present"] for v in labels.values())
    disputes = sum(not v["unanimous"] for v in labels.values())
    print(f"{rounds[0].name}: {len(labels)} labels, {traces} traces, {disputes} disputes")


def cmd_review(st: Study) -> None:
    if (p := st.pending()) is not None:
        raise SystemExit(f"{p} is not imported yet")
    if st.todo():
        raise SystemExit("items are still unlabeled by the ensemble: run `round` first")
    needs, draw = st.review_needs()
    if not needs and not draw:
        print("nothing to review: run `package`")
        return
    j = len(st.records("review"))
    out = st.work / f"review{j}"
    out.mkdir()
    by_state = {
        s["state"]: (s, {f"{r['config']}/{r['prefill']}": r for r in st.rows[s["state"]]})
        for s in st.states
    }
    entries = []
    for iid, need in needs.items():
        s, rows = by_state[need["state"]]
        row = rows[f"{need['config']}/{need['prefill']}"]
        entries.append(
            {
                **entry(iid, s, row),
                "ensemble_present": need["present"],
                "ensemble_evidence": need.get("evidence"),
            }
        )
    random.Random(st.seed + 1000 + j).shuffle(entries)
    with (out / "items.jsonl").open("w") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    mapping = {
        "binding": st.binding,
        "sampled_states": draw,
        "items": {
            iid: {
                "state": n["state"],
                "config": n["config"],
                "prefill": n["prefill"],
                "why": n["why"],
            }
            for iid, n in needs.items()
        },
    }
    write_once(out / "mapping.json", mapping, private=True)
    why = {w: sum(n["why"] == w for n in needs.values()) for w in ("dispute", "trace", "sample")}
    print(f"review {j}: {len(needs)} items {why} in {out / 'items.jsonl'}")


def cmd_import_review(st: Study, labels_path: Path) -> None:
    reviews = target(st.records("review"), "review")
    mapping = json.loads((reviews[0] / "mapping.json").read_text())
    st.check(mapping, reviews[0])
    labels = read_labels(labels_path, {"present": bool})
    match(labels, mapping["items"], "review")
    record = {
        "binding": st.binding,
        "source": {"labels": str(labels_path), "sha256": sha256(labels_path)},
        "labels": labels,
    }
    write_once(reviews[0] / "labels.json", record)
    ensemble = st.ensemble()
    changed = sum(labels[i]["present"] != ensemble[i]["present"] for i in labels)
    print(f"{reviews[0].name}: {len(labels)} reviewed, {changed} changed from the ensemble")


def tracking() -> dict[str, Any]:
    from residual_disposition.config import load_config

    t = OmegaConf.to_container(load_config("base").tracking, resolve=True)
    t.setdefault("wandb", {})["enabled"] = False
    return t


def write_package(step: str, s: Mapping[str, Any], sel: C.Selection, extra: Mapping[str, Any]):
    params = {
        "state": s["state"],
        "behavior": s["behavior"],
        "role": s["role"],
        "dose": s["dose"],
        **extra,
    }
    cfg = OmegaConf.create(
        {
            "experiment": "pd_clean",
            "name": step,
            "step": step,
            "params": params,
            "tracking": tracking(),
        }
    )
    with Run(cfg) as run:
        run.path("pd_clean.json").write_text(
            json.dumps({"state": s["state"], "top": sel.top, "removed": sel.removed}) + "\n"
        )
        run.summarize(C.package_summary(sel))
    return run.dir


def cmd_package(st: Study) -> None:
    if (p := st.pending()) is not None:
        raise SystemExit(f"{p} is not imported yet")
    if todo := st.todo():
        raise SystemExit(f"{len(todo)} selected items are not labeled by the ensemble yet")
    needs, draw = st.review_needs()
    if needs or draw:
        raise SystemExit(f"the researcher's review is not complete: {len(needs)} items to review")
    final, binding = st.final(), {**st.binding, "rubric_sha256": st.rubric()}
    for s in st.states:
        sel = st.selection(s, final)
        extra = {
            "binding": binding,
            "adjudicated": s["adjudicate"],
            "pd_run": str(st.runs[s["state"]]),
        }
        clean = write_package("pd_clean_package", s, sel, extra)
        print(f"{s['state']} ({s['behavior']} {s['role']} {s['dose']}): {clean}")
        if s["random"]:
            ctl = C.random_window(st.rows[s["state"]], sel, seed=st.seed)
            run = write_package("pd_random_package", s, ctl, {**extra, "seed": st.seed})
            print(f"  random control: {run}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=PLAN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("round")
    p = sub.add_parser("import")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--rubric", type=Path, required=True)
    sub.add_parser("review")
    p = sub.add_parser("import-review")
    p.add_argument("--labels", type=Path, required=True)
    sub.add_parser("package")
    args = parser.parse_args(argv)
    st = Study(args.work or data_root() / "pd_clean", args.plan)
    if args.command == "round":
        cmd_round(st)
    elif args.command == "import":
        cmd_import(st, args.labels, args.rubric)
    elif args.command == "review":
        cmd_review(st)
    elif args.command == "import-review":
        cmd_import_review(st, args.labels)
    else:
        cmd_package(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
