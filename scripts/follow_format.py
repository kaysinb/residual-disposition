"""The follow-up's format validation of D*, unfiltered PD and rung 2's input (preregistration §13.5;
barriers B2 and B3), from the qualification's records (``residual_disposition.follow_format``).

    uv run python scripts/follow_format.py status <format> [--run <qualification run>]
    uv run python scripts/follow_format.py round <format> [--run <qualification run>] [--size N]
    uv run python scripts/follow_format.py import-annotators <format> --exports DIR
    uv run python scripts/follow_format.py import-arbiter <format> --exports DIR
    uv run python scripts/follow_format.py review <format>
    uv run python scripts/follow_format.py import-review <format> --export FILE --log FILE
    uv run python scripts/follow_format.py lock <format>
    uv run python scripts/follow_format.py accept <format>
    uv run python scripts/follow_format.py grade <format>
    uv run python scripts/follow_format.py supplement rung2 --run <supplement run>

``<format>`` is ``dstar``, ``pd_unfiltered`` or ``rung2``. The work directory
``<DATA_DIR>/follow_format/<format>`` holds immutable records only:

- ``frame.json`` (the first ``round``): the qualification run it reads (its ``inputs.json`` and
  ``histories.json`` by sha256), the seeded order and each item's reading; written once.
- ``round<k>/``: ``items.jsonl`` (what the annotators see: an opaque id, the behavior, the TARGET
  and the HYPOTHESIS), the private ``mapping.json`` (mode 600), ``kit/annotators/`` and, with a
  dispute, ``kit/arbiter/``; ``exports/``, kept as imported; ``labels.json``. A round is issued
  only when the frozen stopping rule asks for more labels and every earlier round is labeled and
  reviewed.
- ``review<k>/``: every dispute and the seeded tenth of round k's items (``items.jsonl``,
  ``page.json``, the input of the review page), and the researcher's ``labels.json``.
- ``locked.json``: the 400 + 400 items, once ``follow_validation.step`` locks on final classes.
- ``acceptance.json``: the complete rule C of the judge on the locked items against the reference,
  with κ and the alternative-annotator test reported beside it.
- ``graders.json`` (``grade``): the specificity grader on the locked items against the reference
  (κ and the alternative-annotator test: unvalidated, specificity is descriptive), category
  agreement beside it; its requests on the standard tier, on the format's meter item's API session
  (``follow/base``; rung 2's ``follow/rung2_shared``).

Rung 2's format covers both decoders (§13.16's point 6): its frame is their qualification readings
in one seeded order, at least 150 items of each class from each decoder, its reference instructions
D*'s (the same rubric set, on one hypothesis). When the stopping rule asks for a supplement, one
stream (3 to 6 in order) is sent (``scripts/follow_rung2.py --send supplement``), and ``supplement``
records that run once: ``supplement-<stream>.json``, its items labeled in their own seeded order
after what came before.

The machine labels are read only by ``accept``, after the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from residual_disposition import continuation_readers as readers
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_format as F
from residual_disposition import follow_validation as fv
from residual_disposition.judge import JudgeCache, load_targets
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

DEFAULT_SIZE = 300
RUNG2 = "rung2"
DECODER_MINIMUM = 150  # §13.16's point 6: of each class, from each of two decoders


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(data: Any) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


class Work:
    """One format's frozen settings and records."""

    def __init__(self, fmt: str) -> None:
        from residual_disposition.continuation_definitions import load_definitions
        from residual_disposition.continuation_judge import category_letters, specificity_options

        b2 = cr.follow_b2()
        if fmt == RUNG2:
            self.spec = cr.follow_b3()["ladder"]["format_validation"][fmt]
            self.runs, self.item, rubric = "follow_rung2", "follow/rung2_shared", "dstar"
        elif fmt in b2["format_validation"]["formats"]:
            self.spec = b2["format_validation"]["formats"][fmt]
            self.runs, self.item, rubric = "follow_qualification", "follow/base", fmt
        else:
            raise SystemExit(f"{fmt}: not a format of B2 or B3")
        self.fmt, self.b2 = fmt, b2
        self.ensemble = b2["ensemble"]
        self.dir = data_root() / "follow_format" / fmt
        spec = self.ensemble["instructions"][rubric]
        path = REPO_ROOT / spec["file"]
        if sha(path) != spec["sha256"]:
            raise SystemExit(f"{path}: not the frozen instructions")
        self.instructions = path.read_text()
        self.letters_map = category_letters()
        self.definitions, _ = load_definitions(expected_hash=readers.DEFINITIONS_SHA256)
        self.options = {b: [x[0] for x in specificity_options(d)]
                        for b, d in self.definitions.items()}  # fmt: skip
        self.cache = JudgeCache(data_root() / "cache" / "summarizer")
        self.judge_cache = JudgeCache(data_root() / "cache" / "judge")

    def readings(self) -> list[dict[str, Any]]:
        """The format's frame readings, in the inventory's order."""
        from residual_disposition import follow_inventory as fi
        from residual_disposition import follow_rung2 as R2
        from residual_disposition.manifests import committed_json

        if self.fmt == RUNG2:
            return R2.readings()["qualification"]
        confirmed = committed_json(REPO_ROOT / "manifests" / "follow_confirmations.json",
                                   "the confirmations")["confirmed"]  # fmt: skip
        manifest = json.loads((REPO_ROOT / self.b2["inventory"]["manifest"]).read_text())
        parts = fi.resolve(manifest, confirmed)[self.fmt]
        return [r for part in self.spec["frame"] for r in parts[part]]

    def run_dir(self, run: str) -> Path:
        return runs_root() / self.runs / run

    def frame(self, run: str | None = None) -> tuple[dict, dict]:
        """The frame's record, written once from the run given, and its items read back."""
        path = self.dir / "frame.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if run not in (None, record["run"]):
                raise SystemExit(f"the frame is run {record['run']}'s, not {run}'s")
            run = record["run"]
        elif run is None:
            raise SystemExit("no frame yet: name the qualification run (--run)")
        else:
            record = None
        d = self.run_dir(run)
        shas = {name: sha(d / name) for name in ("inputs.json", "histories.json")}
        readings = self.readings()
        items = F.frame(readings, json.loads((d / "histories.json").read_text()), self.cache)
        order = fv.seeded_order(list(items), int(self.spec["seed"]))
        built = {"format": self.fmt, "run": run, "files": shas, "seed": int(self.spec["seed"]),
                 "order": order, "readings": len(readings), "items": len(items)}  # fmt: skip
        if record is None:
            F.write_once(path, dump(built))
        elif record != built:
            raise SystemExit("the frame rebuilt from its records is not the one written")
        for s in self.supplements():
            more = self.stream_items(s["stream"], s["run"])
            if fv.seeded_order(list(more), int(self.spec["supplement_seed"])) != s["order"]:
                raise SystemExit(f"stream {s['stream']}: not the one written")
            items = {**items, **more}
            built = {**built, "order": [*built["order"], *s["order"]],
                     "items": built["items"] + len(more)}  # fmt: skip
        return built, items

    def stream_items(self, stream: int, run: str) -> dict[str, dict[str, Any]]:
        """One supplement stream's items of both decoders, from the run that sent it."""
        from residual_disposition import follow_rung2 as R2

        rows = R2.stream_of(R2.readings()["validation"], stream)
        histories = json.loads((self.run_dir(run) / "histories.json").read_text())
        return F.frame(rows, histories, self.cache)

    def supplements(self) -> list[dict[str, Any]]:
        """The recorded supplement streams, in order."""
        return [json.loads(p.read_text()) for s in fv.SUPPLEMENT_STREAMS
                if (p := self.dir / f"supplement-{s}.json").is_file()]  # fmt: skip

    def next_stream(self) -> int | None:
        """The stream the frame's supplement would take next (rung 2's format alone has one)."""
        if self.fmt != RUNG2:
            return None
        done = {s["stream"] for s in self.supplements()}
        return next((s for s in fv.SUPPLEMENT_STREAMS if s not in done), None)

    def histories(self, record: dict[str, Any]) -> dict[str, Any]:
        """Every slot's recorded attempts: the frame's run's, and each supplement stream's."""
        out = json.loads((self.run_dir(record["run"]) / "histories.json").read_text())
        for s in self.supplements():
            out = {**out, **json.loads((self.run_dir(s["run"]) / "histories.json").read_text())}
        return out

    def rounds(self) -> list[Path]:
        found = sorted(self.dir.glob("round[0-9]*"), key=lambda p: int(p.name[5:]))
        return [p for p in found if p.is_dir()]

    def mapping(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for d in self.rounds():
            out.update(json.loads((d / "mapping.json").read_text()))
        return out

    def reviews(self) -> dict[str, dict]:
        """The researcher's labels of every imported review."""
        out: dict[str, dict] = {}
        for d in self.rounds():
            path = self.dir / f"review{d.name[5:]}" / "labels.json"
            if path.is_file():
                out[d.name] = json.loads(path.read_text())["labels"]
        return out

    def final(self) -> dict[str, dict]:
        """Every labeled item's final labels, by opaque id."""
        labels = [json.loads((d / "labels.json").read_text())["labels"] for d in self.rounds()
                  if (d / "labels.json").is_file()]  # fmt: skip
        return F.final_labels(labels, list(self.reviews().values()))

    def state(self, items: dict, order: list[str]) -> dict[str, Any]:
        """The frozen stopping rule on the final classes (``follow_validation.step``)."""
        mapping, final = self.mapping(), self.final()
        reference = {mapping[i]: F.reference_class(v) for i, v in final.items()}
        strata = {mapping[i]: F.strata(items[mapping[i]]["reading"], v, self.letters_map)
                  for i, v in final.items()}  # fmt: skip
        behaviors = sorted({v["reading"]["behavior"] for v in items.values()})
        decoders = list(readers_decoders(self.fmt))
        spec, roles, required = self.spec["minimums"], (), None
        if self.fmt == RUNG2:
            spec = {c: {**m, "decoder": DECODER_MINIMUM} for c, m in spec.items()}
            roles = ("A", "H", "recipe")
            required = {c: {f"decoder:{d}": DECODER_MINIMUM for d in decoders} for c in spec}
        minimums = fv.minimums_of(spec, behaviors=behaviors, decoders=decoders, roles=roles)
        per_class = int(self.b2["format_validation"]["per_class"])
        # §13.16: a frame exhausted with a class short is not validated, D*'s and PD's with no
        # supplement generated; rung 2's takes the next stream while one is left
        totals = {fv.POSITIVE: per_class, fv.NEGATIVE: per_class}
        supplement = ["next stream"] if self.next_stream() else None
        return fv.step(order, supplement, False, reference, strata, totals, minimums, required)

    def open_work(self) -> str | None:
        """What must finish before the stopping rule is read again."""
        for d in self.rounds():
            review = self.dir / f"review{d.name[5:]}"
            if not (d / "labels.json").is_file():
                return f"{d.name}: not labeled"
            if not (review / "labels.json").is_file():
                return f"{d.name}: not reviewed"
        return None


def readers_decoders(fmt: str) -> tuple[str, ...]:
    return ("state", "base") if fmt in ("dstar", RUNG2) else ()


def cmd_status(w: Work, args: argparse.Namespace) -> None:
    record, items = w.frame(args.run)
    final = w.final()
    classes = {c: sum(F.reference_class(v) == c for v in final.values())
               for c in (fv.POSITIVE, fv.NEGATIVE)}  # fmt: skip
    out = {"format": w.fmt, "items": record["items"], "labeled": len(final), "classes": classes,
           "rounds": len(w.rounds()), "open": w.open_work()}  # fmt: skip
    if out["open"] is None:
        step = w.state(items, record["order"])
        out["step"] = step["step"]
        out["short_strata"] = step.get("short_strata")
    print(json.dumps(out, indent=1))


def cmd_round(w: Work, args: argparse.Namespace) -> None:
    record, items = w.frame(args.run)
    problem = w.open_work()
    if problem:
        raise SystemExit(problem)
    step = w.state(items, record["order"])
    if step["step"] != "label":
        raise SystemExit(f"the stopping rule says {step['step']}, not label")
    d = w.dir / f"round{len(w.rounds())}"
    behaviors = sorted({v["reading"]["behavior"] for v in items.values()})
    rows, mapping = F.round_items(w.fmt, record["order"], items, list(w.mapping().values()),
                                  args.size, load_targets(behaviors))  # fmt: skip
    d.mkdir(parents=True)
    F.write_once(d / "items.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    F.write_once(d / "mapping.json", dump(mapping))
    os.chmod(d / "mapping.json", 0o600)
    for name, text in F.annotator_kit(w.instructions, rows).items():
        F.write_once(d / "kit" / "annotators" / name, text)
    print(f"{d.name}: {len(rows)} items; the annotators' kit in {d / 'kit' / 'annotators'}")


def cmd_supplement(w: Work, args: argparse.Namespace) -> None:
    record, items = w.frame()
    problem = w.open_work()
    if problem:
        raise SystemExit(problem)
    step = w.state(items, record["order"])
    if step["step"] != "generate_supplement":
        raise SystemExit(f"the stopping rule says {step['step']}, not generate_supplement")
    stream = w.next_stream()
    more = w.stream_items(stream, args.run)
    order = fv.seeded_order(list(more), int(w.spec["supplement_seed"]))
    d = w.run_dir(args.run)
    F.write_once(w.dir / f"supplement-{stream}.json", dump({
        "stream": stream, "run": args.run, "seed": int(w.spec["supplement_seed"]), "order": order,
        "files": {n: sha(d / n) for n in ("inputs.json", "histories.json")}}))  # fmt: skip
    print(f"stream {stream}: {len(order)} items from {args.run}")


def last_round(w: Work) -> Path:
    rounds = w.rounds()
    if not rounds:
        raise SystemExit("no round yet")
    return rounds[-1]


def keep(source: Path, copy: Path) -> None:
    """An export kept in its round: an existing copy must be the same."""
    if copy.exists():
        if copy.read_bytes() != source.read_bytes():
            raise SystemExit(f"{copy} exists with other content")
        return
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(source.read_bytes())


def finish_round(w: Work, d: Path) -> None:
    labels, _, shas = F.round_labels(d, w.instructions, w.ensemble, list(w.letters_map), w.options)
    F.write_once(d / "labels.json", dump({"source": {"exports": shas}, "labels": labels}))
    n = sum(bool(v["disputed"]) for v in labels.values())
    print(f"{d.name}: {len(labels)} labels, {n} disputed")


def cmd_import_annotators(w: Work, args: argparse.Namespace) -> None:
    d = last_round(w)
    items = [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    kit = F.annotator_kit(w.instructions, items)
    # read before anything is kept: a bad export is refused, and its correction can follow
    annotators = F.read_annotators(Path(args.exports), w.ensemble, items, kit, list(w.letters_map),
                                   w.options)  # fmt: skip
    names = [f"annotators/{n}.jsonl" for n in sorted(w.ensemble["annotators"])]
    for name in ["annotators_manifest.json", *names]:
        keep(Path(args.exports) / name, d / "exports" / name)
    found = F.disputes(annotators, items)
    if not found:
        finish_round(w, d)
        return
    for name, text in F.arbiter_kit(w.instructions, items, annotators).items():
        F.write_once(d / "kit" / "arbiter" / name, text)
    counts = {f: sum(f in v for v in found.values()) for f in F.DISPUTABLE}
    print(f"{d.name}: {len(found)} disputed items of {len(items)}, by field {counts}; "
          f"the arbiter's kit in {d / 'kit' / 'arbiter'}")  # fmt: skip


def cmd_import_arbiter(w: Work, args: argparse.Namespace) -> None:
    d = last_round(w)
    items = [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    kit = F.annotator_kit(w.instructions, items)
    annotators = F.read_annotators(d / "exports", w.ensemble, items, kit, list(w.letters_map),
                                   w.options)  # fmt: skip
    akit = F.arbiter_kit(w.instructions, items, annotators)
    F.read_arbiter(
        Path(args.exports),
        w.ensemble,
        akit,
        items,
        list(F.disputes(annotators, items)),
        list(w.letters_map),
        w.options,
    )  # a bad export is refused, not installed
    for name in ("arbiter_manifest.json", "arbiter.jsonl"):
        keep(Path(args.exports) / name, d / "exports" / name)
    finish_round(w, d)


def cmd_review(w: Work, args: argparse.Namespace) -> None:
    from residual_disposition.continuation_judge import specificity_options

    d = last_round(w)
    labels, annotators, _ = F.round_labels(d, w.instructions, w.ensemble, list(w.letters_map),
                                           w.options)  # fmt: skip
    mapping = json.loads((d / "mapping.json").read_text())
    first = json.loads((w.rounds()[0] / "labels.json").read_text())["labels"]
    kept = None
    if F.narrowed(first):
        rows = map(json.loads, (d / "items.jsonl").read_text().splitlines())
        behavior = {r["item_id"]: r["behavior"] for r in rows}
        kept = {i for i in labels if F.open_class_or_difficulty(
            [rows[i] for rows in annotators.values()], behavior[i], w.letters_map)}  # fmt: skip
    ids = F.review_set(labels, mapping, int(w.spec["review_seed"]), kept=kept)
    items = {r["item_id"]: r for r in map(json.loads, (d / "items.jsonl").read_text().splitlines())}
    rows = F.review_items(ids, items, labels, annotators)
    review = w.dir / f"review{d.name[5:]}"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    F.write_once(review / "items.jsonl", text)
    behaviors = sorted({r["behavior"] for r in rows})
    page = {"items_sha256": hashlib.sha256(text.encode()).hexdigest(), "items": rows,
            "categories": {k: w.definitions[b].category for k, b in w.letters_map.items()},
            "features": {b: [[x[0], x[2]] for x in specificity_options(w.definitions[b])]
                         for b in behaviors}}  # fmt: skip
    F.write_once(review / "page.json", dump(page))
    disputes = sum(r["reason"] == "dispute" for r in rows)
    share = float(F.dispute_share(first))
    print(f"{review.name}: {len(rows)} items ({disputes} disputes; first round's share of disputed "
          f"items {share:.3f}, narrowed {kept is not None}); {review / 'page.json'}")  # fmt: skip


def cmd_import_review(w: Work, args: argparse.Namespace) -> None:
    d = last_round(w)
    review = w.dir / f"review{d.name[5:]}"
    text = (review / "items.jsonl").read_text()
    rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    log = json.loads(Path(args.log).read_text())
    want = hashlib.sha256(text.encode()).hexdigest()
    if (log.get("source") or {}).get("items_sha256") != want:
        raise SystemExit(f"{args.log}: not the page of this review's items")
    labels = F.read_review(Path(args.export).read_text().splitlines(),
                           [r["item_id"] for r in rows], {r["item_id"]: r for r in rows},
                           list(w.letters_map), w.options)  # fmt: skip
    keep(Path(args.export), review / "export.jsonl")
    keep(Path(args.log), review / "log.json")
    F.write_once(review / "labels.json", dump({"source": {"export": sha(review / "export.jsonl"),
                                                          "log": sha(review / "log.json")},
                                               "labels": labels}))  # fmt: skip
    changed = sum(any(labels[r["item_id"]][f] != r["proposed"][f] for f in F.DISPUTABLE)
                  for r in rows)  # fmt: skip
    print(f"{review.name}: {len(labels)} labels, {changed} changed")


def cmd_lock(w: Work, args: argparse.Namespace) -> None:
    record, items = w.frame()
    problem = w.open_work()
    if problem:
        raise SystemExit(problem)
    step = w.state(items, record["order"])
    if step["step"] != "locked":
        raise SystemExit(f"the stopping rule says {step['step']}, not locked")
    mapping = w.mapping()
    F.write_once(w.dir / "locked.json", dump({
        "format": w.fmt, "frame": sha(w.dir / "frame.json"), "labeled": step["labeled"],
        "chosen": step["chosen"], "short_strata": step["short_strata"],
        "labels": {mapping[i]: v for i, v in w.final().items()}}))  # fmt: skip
    print(f"locked: {len(step['chosen'][fv.POSITIVE])} + {len(step['chosen'][fv.NEGATIVE])} items, "
          f"short strata {step['short_strata']}")  # fmt: skip


def cmd_accept(w: Work, args: argparse.Namespace) -> None:
    from fractions import Fraction

    from residual_disposition.judge import c_validation

    record, items = w.frame()
    locked = json.loads((w.dir / "locked.json").read_text())
    if locked["frame"] != sha(w.dir / "frame.json"):
        raise SystemExit("the lock is not this frame's")
    histories = w.histories(record)
    chosen = locked["chosen"]
    every = [*chosen[fv.POSITIVE], *chosen[fv.NEGATIVE]]
    judged = {i: F.judged(i, histories, w.judge_cache) for i in every}
    mapping, by_item = w.mapping(), locked["labels"]
    for cls, chosen_items in chosen.items():  # the set as locked, on the labels it was locked with
        if any(F.reference_class(by_item[i]) != cls for i in chosen_items):
            raise SystemExit("the locked set is not its locked labels' classes")
    strata = {i: F.strata(items[i]["reading"], by_item[i], w.letters_map) for i in every}
    rules = cr.follow_rules().data["format_validation"]
    result = fv.acceptance(chosen, judged, strata, per_class=int(rules["per_class"]),
                           sensitivity_floor=Fraction(rules["sensitivity_lower_min"]),
                           fpr_ceiling=Fraction(rules["fpr_upper_max"]),
                           confidence=Fraction(rules["confidence"]))  # fmt: skip
    keys = [(i, items[i]["reading"]["behavior"]) for i in every]
    votes: dict[str, dict] = {}
    for d in w.rounds():
        _, annotators, _ = F.round_labels(d, w.instructions, w.ensemble, list(w.letters_map),
                                          w.options)  # fmt: skip
        for name, rows in annotators.items():
            for iid, row in rows.items():
                item = mapping[iid]
                key = (item, items[item]["reading"]["behavior"])
                votes.setdefault(name, {})[key] = (row["strict"], row["lenient"] == "yes")
    annotators = {n: {k: v[k] for k in keys} for n, v in votes.items()}
    machine = {k: (None, None) if judged[k[0]] is None
               or judged[k[0]][0] == readers.STRICT_SAFETY_BLOCKED
               else judged[k[0]] for k in keys}  # fmt: skip
    reference = {k: (by_item[k[0]]["strict"], by_item[k[0]]["lenient"] == "yes") for k in keys}
    beside = c_validation(machine, reference, annotators, keys)
    F.write_once(w.dir / "acceptance.json", dump({
        "format": w.fmt, "locked": sha(w.dir / "locked.json"), **result,
        "reported": {"c_validation": beside}}))  # fmt: skip
    print(json.dumps({k: result[k] for k in ("passed", "false_negatives", "false_positives",
                                             "missing", "criterion_2")}, indent=1))  # fmt: skip


def cmd_grade(w: Work, args: argparse.Namespace) -> None:
    import atexit

    import follow_qualification as S

    from residual_disposition import follow_money
    from residual_disposition import follow_readings as D
    from residual_disposition.continuation_judge import specificity_options
    from residual_disposition.judge import fetch

    record, items = w.frame()
    locked = json.loads((w.dir / "locked.json").read_text())
    every = [*locked["chosen"][fv.POSITIVE], *locked["chosen"][fv.NEGATIVE]]
    definitions, spec = D.grader()

    def texts(item: str) -> tuple[str, str]:
        return items[item]["hypothesis"], items[item]["reading"]["behavior"]

    def read(cache: JudgeCache) -> dict[str, dict]:
        return {i: D.grades(*texts(i), definitions, spec, cache, i) for i in every}

    cache = JudgeCache(data_root() / "cache" / "judge")
    now = {g["due"].key: g["due"] for v in read(cache).values() for g in v.values() if "due" in g}
    specs = {spec.id: spec}
    print(f"due now: {len(now)}; estimate ${S.estimate(list(now.values()), specs, set()):.2f}")
    charged = None
    if now:
        S.LABEL = "follow/format-graders"
        _, ledger = follow_money.money(data_root(), w.item, data_root() / S.METER_LOG)
        atexit.register(ledger.close)
        ledger.source = S.LABEL
        backend = S.backend(ledger, specs, interpreter=False)
        backend.reconcile(cache)
        ledger.check(S.estimate(list(now.values()), specs, set()))
        fetch(list(now.values()), backend, cache)
        charged = ledger.close()
    final = read(cache)
    mapping, by_item = w.mapping(), locked["labels"]
    machine = {}
    for i, g in final.items():
        machine[i] = None if any("value" not in x for x in g.values()) else {
            "specificity": g["specificity"]["value"][0],
            "categories": sorted(w_letter(w, b) for b in g["category"]["value"])}  # fmt: skip
    kinds = {i: {x[0]: x[1] for x in specificity_options(
        definitions[items[i]["reading"]["behavior"]])} for i in every}  # fmt: skip
    votes: dict[str, dict] = {}
    for d in w.rounds():
        _, annotators, _ = F.round_labels(d, w.instructions, w.ensemble, list(w.letters_map),
                                          w.options)  # fmt: skip
        for name, rows in annotators.items():
            for iid, row in rows.items():
                if mapping[iid] in by_item:
                    votes.setdefault(name, {})[mapping[iid]] = row
    reference = {i: by_item[i] for i in every}
    ballots = {n: {i: v[i] for i in every} for n, v in votes.items()}
    result = F.grader_validation(machine, reference, ballots, kinds)
    locked_sha = sha(w.dir / "locked.json")
    F.write_once(w.dir / "graders.json", dump({
        "format": w.fmt, "locked": locked_sha, "charged": charged, **result}))  # fmt: skip
    print(json.dumps(result["specificity"], indent=1))
    print(result["category_agreement"])


def w_letter(w: Work, behavior: str) -> str:
    """A behavior's category letter."""
    return next(k for k, b in w.letters_map.items() if b == behavior)


COMMANDS = {"status": cmd_status, "round": cmd_round, "import-annotators": cmd_import_annotators,
            "import-arbiter": cmd_import_arbiter, "review": cmd_review,
            "import-review": cmd_import_review, "lock": cmd_lock, "accept": cmd_accept,
            "grade": cmd_grade, "supplement": cmd_supplement}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("format", choices=["dstar", "pd_unfiltered", RUNG2])
    p.add_argument("--run")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--exports", type=Path)
    p.add_argument("--export", type=Path)
    p.add_argument("--log", type=Path)
    args = p.parse_args(argv)
    COMMANDS[args.command](Work(args.format), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
