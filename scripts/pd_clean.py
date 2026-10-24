"""PD-clean packages of the continuation's Stage 1 (preregistration §12.4.2, §12.3.2).

    uv run python scripts/pd_clean.py round [--lookahead STATE=K ...] [--work DIR]
    uv run python scripts/pd_clean.py import --labels FILE [--work DIR]
    uv run python scripts/pd_clean.py review [--work DIR]
    uv run python scripts/pd_clean.py import-review --labels FILE [--work DIR]
    uv run python scripts/pd_clean.py amend-review --old-plan FILE [--withdraw reviewN ...] \
        --decision TEXT [--work DIR]
    uv run python scripts/pd_clean.py package [--work DIR]

The states, their roles, the adjudication plan and the review sample are
configs/continuation/pd_clean.yaml; each state's PD output is its pilot tool_pd run. The work
directory (default <DATA_DIR>/pd_clean) holds immutable records only; every state's labels are
derived from them each time, never kept as mutable state.

- **Binding.** Every record carries the binding it was made under: the plan's sha256, each
  state's PD run and the sha256 of its completions, each behavior's detector version, and the
  sha256 of the ensemble's rubric, the instructions the plan freezes (which must match the hash
  the plan states). Every command refuses records of another binding.
- ``round`` writes the next ensemble round (``round<k>/items.jsonl``, the detector diagnostic's
  format: item id, behavior id and description, the response, the prefill in brackets then its
  continuation; pooled and shuffled; no state, role, dose, configuration or rank). Its private
  ``mapping.json`` (mode 600) maps items to states and is not for the annotators. With
  ``--lookahead STATE=K`` (from Stage 2) the round also asks, for that state, the next K
  candidates of each configuration below the selection (``pd_clean.candidates_after``): labeled
  ahead, they spare rounds on a long tail of traces. The selection, which takes the first clean
  item in ranking order, is the one item-by-item rounds give; labels below the selection's
  window stay in the records but are not met, neither in the package nor in the review.
- ``import`` takes the round's final ensemble labels, JSON lines of ``{"item_id", "present",
  "unanimous", "confident", "evidence"}`` and nothing else (``present`` a trace, ``unanimous``
  false for an arbitrated dispute), every item of the round exactly once, and writes
  ``labels.json`` atomically, once; a second import of the same labels is a no-op, of other
  labels an error. A label never overrides the item's metadata (state, configuration, prefill).
- ``review`` writes the researcher's review set (§12.3.2, step 3): every dispute and every
  unanimous trace not yet reviewed among the items a selection met (its window), and once per
  group (behavior × package type: the plan's
  ``review.groups`` maps each adjudicated role to its type) a seeded sample of the group's agreed
  (unanimous) negatives. The plan freezes the total (180, the execution amendment of 2026-10-16;
  it was 30 per package), allocated to the groups in proportion to their numbers of agreed
  negatives by largest remainders (``allocate``).
- ``amend-review`` records a change of the plan's review section alone (an execution amendment)
  made after rounds were labeled: the old and the new plan's text, whose sha256 must be the two
  bindings' and which may differ only in ``review``, the decision, and the reviews it withdraws
  (drawn under the old plan, never imported). Round records of the old binding stay valid
  through it; a withdrawn review counts nowhere; every new record is made under the new
  binding.
- ``import-review`` takes the researcher's final labels (``{"item_id", "present", "note"}``,
  nothing else) for exactly the review's items, once. A researcher's label overrides the
  ensemble's; a change reruns the removal and the refill, so the next ``round`` may bring new
  items, whose disputes and traces go to a later review.
- ``package`` refuses while a round or a review is pending or needed; then writes one tracked
  run per state (experiment pd_clean, step pd_clean_package) with the clean selection and every
  removal with its reason, and, for the states configured for it, the random-removal control
  (step pd_random_package, ``pd_clean.random_window``). summarize_tools reads both with PD's
  prompt and the selected texts only.

**Stage 3** (``--plan configs/continuation/pd_clean_stage3.yaml --record <stage3_readers run>``):
the template expands into one certified lineage's states from its readers' record: L, H_s, X, A
and the recipe control, each with the PD pool its readers read (named in the record, never
searched for); L, A and the recipe control adjudicated, L, A and H_s with the random control.
The record (its run, lineage and sha256) joins the binding. The review samples
``negatives_per_package`` agreed negatives per adjudicated package (``allocation:
per_package``), seeded by the package, with every dispute and unanimous trace met; each review
names the records its sample was drawn after (``after``), so that it can be drawn again.

A Stage-3 round issues its annotators' kit (``kit/annotators``: the frozen instructions filled in
and the items in parts, ``pd_clean_study.annotator_kit``), and is imported in two steps:
``import-annotators --exports DIR`` (``annotators/<name>.jsonl`` and
``annotators_manifest.json``, each annotator's model, runner and kit as the plan and the round
state them), which issues the arbiter's kit when a vote is disputed (``kit/arbiter``: the
disputed items alone, the votes counted without names); then ``import-arbiter --exports DIR``
(``arbiter.jsonl`` and ``arbiter_manifest.json``). The exports are kept in the round, its labels
the plan's merge of them (``merge_ensemble``). A Stage-3 review is imported from its page's
export and log (``import-review --export FILE --log FILE``), kept in the review. Its work
directory defaults to ``<DATA_DIR>/pd_clean_stage3/<lineage>``, one per lineage.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from residual_disposition import pd_clean as C
from residual_disposition import pd_clean_study as S
from residual_disposition.paths import CONFIG_ROOT, data_root, runs_root
from residual_disposition.pd_clean_study import (
    allocate,  # noqa: F401 (the script's own name for it, as before)
    check_withdrawal,
    sha256,
    text_sha256,
    verify_amendment,
)
from residual_disposition.tracking import Run

PLAN = CONFIG_ROOT / "continuation" / "pd_clean.yaml"


def pd_run(state_id: str, experiments: list[str], draw: str | None = None) -> Path:
    """The completed tool_pd run of a state of the given draw (None: the untagged pool; a tag:
    that pool, ``pipeline.PD_DRAW_GPU``; Codex s4m); none or several is an error."""
    found = []
    for experiment in experiments:
        for cfg_path in sorted((runs_root() / experiment).glob("*-tool_pd-*/config.yaml")):
            cfg = yaml.safe_load(cfg_path.read_text())
            meta = json.loads((cfg_path.parent / "meta.json").read_text())
            if (
                cfg["params"]["state"] == state_id
                and cfg["params"].get("draw") == draw
                and meta.get("status") == "completed"
            ):
                found.append(cfg_path.parent)
    if len(found) != 1:
        raise SystemExit(
            f"{state_id}: {len(found)} completed tool_pd runs of draw {draw}, not one: {found}"
        )
    return found[0]


def state_draw(s: Mapping[str, Any]) -> str | None:
    """The PD pool a plan's state is read from (Codex s4m, s4n): Stage 2's latent state (role L
    at a deep dose of the frozen rules) reads the pool its lineage declared before A1
    (``pipeline.declared_pd_draw``, ``configs/continuation/pd_draws.yaml``): the draw is
    derived from the declaration, and a plan entry naming another is refused. Every other state
    reads the draw its entry names (Stage 1's plans: none)."""
    from residual_disposition.pipeline import _stage2_rules, declared_pd_draw

    deep = [int(d) for d in _stage2_rules()["stage2"]["deep_doses"]]
    if s.get("role") != "L" or s.get("dose") not in deep:
        return s.get("draw")
    declared = declared_pd_draw(s["behavior"])
    if "draw" in s and s["draw"] != declared:
        raise SystemExit(
            f"{s['state']}: the plan's PD draw {s['draw']!r} is not the declared {declared!r} "
            f"of {s['behavior']}'s Stage-2 L"
        )
    return declared


class Study(S.Study):
    """``pd_clean_study.Study`` with Stage 1's pools found here (``pd_run``, ``state_draw``)."""

    def __init__(self, work: Path, plan_path: Path = PLAN, record: Path | None = None) -> None:
        super().__init__(
            work,
            plan_path,
            record,
            pool_of=lambda plan, s: pd_run(s["state"], plan["pd_runs"], state_draw(s)),
        )


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


# --- commands -------------------------------------------------------------------------------


def cmd_round(st: Study, lookahead: Mapping[str, int] | None = None) -> None:
    if (p := st.pending()) is not None:
        raise SystemExit(f"{p} is not imported yet")
    unknown = set(lookahead or {}) - {s["state"] for s in st.states if s["adjudicate"]}
    if unknown:
        raise SystemExit(f"--lookahead names states that are not adjudicated: {sorted(unknown)}")
    todo = st.todo(lookahead)
    if not todo:
        print("no item left to label: run `review`, or `package` once the review is complete")
        return
    k = len(st.records("round"))
    out = st.work / f"round{k}"
    out.mkdir()
    mapping = {
        "binding": st.binding,
        "lookahead": dict(lookahead or {}),
        "items": {
            iid: {"state": s["state"], "config": config, "prefill": row["prefill"]}
            for iid, (s, config, row) in todo.items()
        },
    }
    (out / "items.jsonl").write_text(S.round_items(st, mapping, k))
    if st.stage3:  # what every annotator reads, from the items and the frozen instructions
        write_kit(out / "kit" / "annotators", S.annotator_kit(st.rubric_text(), round_items(out)))
    write_once(out / "mapping.json", mapping, private=True)
    print(f"round {k}: {len(todo)} items for the ensemble in {out / 'items.jsonl'}")


ENSEMBLE_FIELDS = {
    "present": (bool,),
    "unanimous": (bool,),
    "confident": (bool, type(None)),
    "evidence": (str, type(None)),
}
REVIEW_FIELDS = {"present": (bool,), "note": (str, type(None))}
REQUIRED = {"present", "unanimous"}


def read_labels(path: Path, fields: Mapping[str, tuple[type, ...]]) -> dict[str, dict[str, Any]]:
    """A label file's labels by item id: only ``fields`` (with their types) besides the item id,
    ``present`` (and ``unanimous`` where it is a field) required; any other field, a reserved
    one (state, config, prefill, why, round, review) included, is refused."""
    labels = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = row.pop("item_id", None)
        if not isinstance(iid, str) or iid in labels:
            raise SystemExit(f"{iid!r}: a missing or repeated item id")
        if extra := set(row) - set(fields):
            raise SystemExit(f"{iid}: fields {sorted(extra)} are not label fields")
        for name, kinds in fields.items():
            if name not in row and name in REQUIRED:
                raise SystemExit(f"{iid}: {name!r} is required")
            if name in row and not isinstance(row[name], kinds):
                raise SystemExit(f"{iid}: {name!r} has the wrong type")
        labels[iid] = row
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


def keep(source: Path, copy: Path) -> None:
    """A source export kept in its record, once: an existing copy must be the same bytes."""
    if copy.exists():
        if copy.read_bytes() != Path(source).read_bytes():
            raise SystemExit(f"{copy} exists with other content: records are immutable")
        return
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, copy)


def cmd_import(st: Study, labels_path: Path | None) -> None:
    rounds = target(st.records("round"), "round")
    mapping = json.loads((rounds[0] / "mapping.json").read_text())
    st.check(mapping, rounds[0], rounds=True)
    if st.stage3:
        raise SystemExit("a Stage-3 round is imported in two steps: import-annotators, then "
                         "import-arbiter")  # fmt: skip
    else:
        if labels_path is None:
            raise SystemExit("a Stage-1 round is imported from its labels (--labels)")
        labels = read_labels(labels_path, ENSEMBLE_FIELDS)
        source = {"labels": str(labels_path), "sha256": sha256(labels_path)}
    match(labels, mapping["items"], "round")
    record = {
        "binding": st.binding,
        "source": source,
        "labels": labels,
    }
    write_once(rounds[0] / "labels.json", record)
    traces = sum(v["present"] for v in labels.values())
    disputes = sum(not v["unanimous"] for v in labels.values())
    print(f"{rounds[0].name}: {len(labels)} labels, {traces} traces, {disputes} disputes")


def round_items(d: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]


def write_kit(d: Path, kit: Mapping[str, str]) -> None:
    """A kit written once into its record: an existing one must be the same."""
    for name, text in kit.items():
        path = d / name
        if path.exists():
            if path.read_text() != text:
                raise SystemExit(f"{path} exists with other content: records are immutable")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def stage3_round(st: Study) -> Path:
    if not st.stage3:
        raise SystemExit("import-annotators and import-arbiter are Stage 3's")
    rounds = target(st.records("round"), "round")
    st.check(json.loads((rounds[0] / "mapping.json").read_text()), rounds[0], rounds=True)
    return rounds[0]


def finish_round(st: Study, d: Path) -> None:
    """The round's labels, from its kept kits and exports alone (``pd_clean_study.
    round_labels``)."""
    labels, shas = S.round_labels(st, d)
    write_once(d / "labels.json", {"binding": st.binding, "source": {"exports": shas},
                                   "labels": labels})  # fmt: skip
    traces = sum(v["present"] for v in labels.values())
    disputes = sum(not v["unanimous"] for v in labels.values())
    print(f"{d.name}: {len(labels)} labels, {traces} traces, {disputes} disputes")


def cmd_import_annotators(st: Study, exports: Path) -> None:
    """Step 1 of a Stage-3 round: the annotators' exports and their manifest kept in the round;
    with a dispute, the arbiter's kit written next to them, else the round's labels."""
    d = stage3_round(st)
    names = [f"annotators/{n}.jsonl" for n in sorted(st.plan["ensemble"]["annotators"])]
    for name in ["annotators_manifest.json", *names]:
        keep(Path(exports) / name, d / "exports" / name)
    items, rubric = round_items(d), st.rubric_text()
    kit = S.annotator_kit(rubric, items)
    S.check_kit(d / "kit" / "annotators", kit)  # the kit as issued, before anything follows it
    annotators = S.read_annotators(d / "exports", st.plan["ensemble"], items, kit)
    ids = S.disputed(items, annotators)
    if not ids:
        finish_round(st, d)
        return
    write_kit(d / "kit" / "arbiter", S.arbiter_kit(rubric, items, annotators))
    print(f"{d.name}: {len(ids)} disputed items; the arbiter's kit in {d / 'kit' / 'arbiter'}")


def cmd_import_arbiter(st: Study, exports: Path) -> None:
    """Step 2 of a Stage-3 round with a dispute: the arbiter's export and manifest kept in the
    round, then the round's labels."""
    d = stage3_round(st)
    if not (d / "kit" / "arbiter").is_dir():
        raise SystemExit(f"{d.name}: no arbiter's kit: run import-annotators first")
    for name in ("arbiter_manifest.json", "arbiter.jsonl"):
        keep(Path(exports) / name, d / "exports" / name)
    finish_round(st, d)


def cmd_review(st: Study) -> None:
    if (p := st.pending()) is not None:
        raise SystemExit(f"{p} is not imported yet")
    if st.todo():
        raise SystemExit("items are still unlabeled by the ensemble: run `round` first")
    needs, draw = st.review_needs()
    if not needs and not draw:
        print("nothing to review: run `package`")
        return
    j = len(st.all_records("review"))
    out = st.work / f"review{j}"
    out.mkdir()
    mapping = {
        "binding": st.binding,
        **({"after": st.drawn_after()} if st.stage3 else {}),  # the records the sample rests on
        "sampled_groups": draw,
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
    (out / "items.jsonl").write_text(S.review_items(st, mapping, j, st.ensemble()))
    write_once(out / "mapping.json", mapping, private=True)
    why = {w: sum(n["why"] == w for n in needs.values()) for w in ("dispute", "trace", "sample")}
    print(f"review {j}: {len(needs)} items {why} in {out / 'items.jsonl'}")


def cmd_import_review(st: Study, labels_path: Path | None, export: Path | None = None,
                      log: Path | None = None) -> None:  # fmt: skip
    reviews = target(st.records("review"), "review")
    d = reviews[0]
    mapping = json.loads((d / "mapping.json").read_text())
    st.check(mapping, d)
    if st.stage3:  # the page's export and its log, kept in the review
        if export is None or log is None or labels_path is not None:
            raise SystemExit("a Stage-3 review is imported from its page's export and log alone "
                             "(--export, --log)")  # fmt: skip
        keep(export, d / "export.jsonl")
        keep(log, d / "page_log.json")
        shown = sha256(d / "items.jsonl")  # what the researcher read
        labels, shas = S.review_labels(d / "export.jsonl", d / "page_log.json",
                                       set(mapping["items"]), shown)  # fmt: skip
        source: dict[str, Any] = {"exports": shas}
    else:
        if labels_path is None:
            raise SystemExit("a Stage-1 review is imported from its labels (--labels)")
        labels = read_labels(labels_path, REVIEW_FIELDS)
        source = {"labels": str(labels_path), "sha256": sha256(labels_path)}
    match(labels, mapping["items"], "review")
    record = {
        "binding": st.binding,
        "source": source,
        "labels": labels,
    }
    write_once(reviews[0] / "labels.json", record)
    ensemble = st.ensemble()
    changed = sum(labels[i]["present"] != ensemble[i]["present"] for i in labels)
    print(f"{reviews[0].name}: {len(labels)} reviewed, {changed} changed from the ensemble")


def cmd_amend_review(st: Study, old_plan: Path, withdraw: list[str], decision: str) -> None:
    old_text, new_text = Path(old_plan).read_text(), st.plan_path.read_text()
    amendment = {
        "from": {**st.binding, "plan_sha256": text_sha256(old_text)},
        "to": st.binding,
        "old_plan": old_text,
        "new_plan": new_text,
        "changed": ["review"],
        "withdrawn": sorted(withdraw),
        "decision": decision,
    }
    verify_amendment(amendment, Path(old_plan))
    for name in withdraw:
        check_withdrawal(st.work, name, amendment["from"])
    for d in st.all_records("amend"):
        if json.loads((d / "amendment.json").read_text()) == amendment:
            print(f"{d.name} already records this amendment")
            return
    k = len(st.all_records("amend"))
    out = st.work / f"amend{k}"
    out.mkdir()
    write_once(out / "amendment.json", amendment)
    print(f"{out.name}: the review section amended; withdrawn {sorted(withdraw) or 'none'}")


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
    final, binding = st.final(), st.binding
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
    parser.add_argument("--record", type=Path, default=None,
                        help="the Stage-3 plan's lineage: its readers' record run")  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("round")
    p.add_argument("--lookahead", action="append", default=[], metavar="STATE=K")
    p = sub.add_parser("import")
    p.add_argument("--labels", type=Path, default=None, help="Stage 1: the merged labels")
    p = sub.add_parser("import-annotators")
    p.add_argument("--exports", type=Path, required=True)
    p = sub.add_parser("import-arbiter")
    p.add_argument("--exports", type=Path, required=True)
    sub.add_parser("review")
    p = sub.add_parser("import-review")
    p.add_argument("--labels", type=Path, default=None, help="Stage 1: the review's labels")
    p.add_argument("--export", type=Path, default=None, help="Stage 3: the page's export")
    p.add_argument("--log", type=Path, default=None, help="Stage 3: the page's log")
    p = sub.add_parser("amend-review")
    p.add_argument("--old-plan", type=Path, required=True)
    p.add_argument("--withdraw", nargs="*", default=[])
    p.add_argument("--decision", required=True)
    sub.add_parser("package")
    args = parser.parse_args(argv)
    default = data_root() / "pd_clean"
    if args.record is not None:  # one work directory per certified lineage
        lineage = json.loads((args.record / "stage3_readers.json").read_text())["lineage"]
        default = data_root() / "pd_clean_stage3" / lineage
    work = Path(args.work or default)
    work.mkdir(parents=True, exist_ok=True)
    # One command at a time per work directory: an amendment and a review's import never
    # interleave, and every command reads the records as they stand.
    lock = (work / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    st = Study(work, args.plan, args.record)
    if args.command == "round":
        ahead = {}
        for spec in args.lookahead:
            state, _, k = spec.partition("=")
            if not k.isdigit() or int(k) < 1:
                raise SystemExit(f"--lookahead {spec!r}: STATE=K with K a positive integer")
            ahead[state] = int(k)
        cmd_round(st, ahead)
    elif args.command == "import":
        cmd_import(st, args.labels)
    elif args.command == "import-annotators":
        cmd_import_annotators(st, args.exports)
    elif args.command == "import-arbiter":
        cmd_import_arbiter(st, args.exports)
    elif args.command == "review":
        cmd_review(st)
    elif args.command == "import-review":
        cmd_import_review(st, args.labels, args.export, args.log)
    elif args.command == "amend-review":
        cmd_amend_review(st, args.old_plan, args.withdraw, args.decision)
    else:
        cmd_package(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
