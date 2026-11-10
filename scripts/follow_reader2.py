"""A second interpreter of another family on frozen decisive packages
(``residual_disposition.follow_reader2``), its judges on ``follow/reader2``.

    uv run python scripts/follow_reader2.py --pin
    uv run python scripts/follow_reader2.py --kit
    uv run python scripts/follow_reader2.py --queue
    uv run python scripts/follow_reader2.py --import DIR
    uv run python scripts/follow_reader2.py --send
    uv run python scripts/follow_reader2.py --result

``--pin`` writes each cell's source, evidence sha256 and attempt keys once
(``manifests/follow_reader2_inputs.json``), checking every package against its source pin first;
nothing is sent. ``--kit`` writes, with the pins committed and unchanged, each cell's whole input
(``input.txt``: the wrapper, the prompt and the package) under
``<DATA_DIR>/follow_reader2/kit/<cell>``. ``--queue`` writes the attempts due to the reader
(``queue.jsonl``: key, cell folder), the conditional cells only where their line's L has reached
its minimum. ``--import DIR`` reads the runner's replies (``DIR/<key>.txt``, its session's events
``DIR/<key>.jsonl`` and ``DIR/runner.json``, which must state B2's ``gpt`` annotator and this
check's settings) into the reader's cache: a reply whose session did anything but answer, or read
less than its whole input, is recorded with an empty text and its reasons, so that the attempt's
retry is due. ``--send`` sends the judges' due labels and grades in one API session of the meter
item. ``--result`` writes ``summary.json`` once, under ``runs/follow_reader2``.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import follow_qualification as S

from residual_disposition import continuation_readers as readers
from residual_disposition import follow_qualification as Q
from residual_disposition import follow_reader2 as R
from residual_disposition import follow_readings as D
from residual_disposition.continuation_kstar import find_run
from residual_disposition.judge import Completion, JudgeCache, JudgeItem, fetch, load_targets
from residual_disposition.paths import REPO_ROOT, data_root, runs_root

PINS = "manifests/follow_reader2_inputs.json"
LABEL = "follow/reader2"
ROOT = "follow_reader2"


def _cache() -> JudgeCache:
    return JudgeCache(data_root() / "cache" / "reader2")


def plan() -> list[dict[str, Any]]:
    pinned = R.pins()
    out = []
    for r in R.readings():
        text = R.evidence(r, pinned, runs_root(), find_run)
        out.append({"reading": r, "text": text,
                    "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "attempts": R.attempts(r, text)})  # fmt: skip
    return out


def _pins_now(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {c["reading"]["salt"]: {
        "source": [R.SOURCES[c["reading"]["source"]], c["reading"]["source_salt"]],
        "evidence_sha256": c["evidence_sha256"],
        "keys_sha256": hashlib.sha256("\n".join(a.key for t in c["attempts"] for a in t)
                                      .encode()).hexdigest()} for c in cells}  # fmt: skip


def _checked(cells: list[dict[str, Any]]) -> None:
    from residual_disposition.manifests import committed_json

    pinned = committed_json(REPO_ROOT / PINS, "the second reader's pins")["readings"]
    if pinned != _pins_now(cells):
        raise SystemExit("the cells are not their pins")


def _context() -> dict[str, Any]:
    judges = {"lens": readers.judge_settings(readers.RUBRICS["lens"])}
    definitions, spec = D.grader()
    judge = judges["lens"]
    behaviors = sorted({r["behavior"] for r in R.readings()})
    return {"judge": judge, "targets": load_targets(behaviors), "pinned": Q.judge_pins(),
            "definitions": definitions, "spec": spec,
            "specs": {s.id: s for s in (judge.model, judge.model_for("lenient"), spec)},
            "caches": {"reader": _cache(),
                       "judge": JudgeCache(data_root() / "cache" / "judge")}}  # fmt: skip


def _admit(step: dict[str, Any]) -> str | None:
    return "runner" if step["tier"] == "runner" else None


def state(cells: list[dict[str, Any]], ctx: dict[str, Any], caches: dict[str, Any]) -> list[dict]:
    from residual_disposition.continuation_readers import counted
    from residual_disposition.summarizer import parse_hypothesis

    rows = []
    for c in cells:
        r = c["reading"]
        slots = []
        for i, tries in enumerate(c["attempts"]):
            where = f"{r['salt']}#attempt={i}"
            hyp = Q.answer(caches["reader"], tries, parse_hypothesis, _admit, where)
            lab = None
            if "value" in hyp:
                item = JudgeItem(where, r["behavior"], ctx["targets"][r["behavior"]], hyp["value"])
                lab = Q.labels(item, ctx["judge"], caches["judge"], ctx["pinned"])
                if (
                    "value" in lab["strict"]
                    and "value" in lab["lenient"]
                    and r["role"] in ("L", "A", "recipe")
                    and counted(lab["strict"]["value"][0], lab["lenient"]["value"])
                ):
                    hyp_text = hyp["value"]
                    lab["grades"] = D.grades(hyp_text, r["behavior"], ctx["definitions"],
                                             ctx["spec"], caches["judge"], where)  # fmt: skip
            slot = {"hypothesis": hyp, "labels": lab}
            if lab and "grades" in lab:
                slot["grades"] = lab.pop("grades")
            slots.append(slot)
        rows.append({"reading": r, "slots": slots, "count": R.count(slots, r["attempts"])})
    return rows


def _active(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = R.conditional_due(rows)
    return [x for x in rows if not x["reading"]["conditional"]
            or R.line_of(x["reading"]) in lines]  # fmt: skip


def queue(cells: list[dict[str, Any]], ctx: dict[str, Any]) -> int:
    rows = state(cells, ctx, ctx["caches"])
    by_salt = {c["reading"]["salt"]: c for c in cells}
    out = []
    for row in _active(rows):
        for s in row["slots"]:
            if "due" in s["hypothesis"]:
                out.append(
                    {
                        "key": s["hypothesis"]["due"].key,
                        "cell": _folder(by_salt[row["reading"]["salt"]]["reading"]),
                    }
                )
    path = data_root() / ROOT / "queue.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(f"{len(out)} attempts due: {path}")
    return 0


def _folder(r: dict[str, Any]) -> str:
    return r["salt"].replace("follow/reader2/", "").replace("/", "__")


def kit(cells: list[dict[str, Any]]) -> int:
    base = data_root() / ROOT / "kit"
    if base.exists():
        raise SystemExit(f"{base}: the kit is written once")
    for c in cells:
        d = base / _folder(c["reading"])
        d.mkdir(parents=True)
        system, user = c["attempts"][0][0].body["messages"]
        (d / "input.txt").write_text(R.input_text(system["content"], user["content"]))
    (base / "reader.json").write_text(json.dumps(R.reader_spec(), indent=1) + "\n")
    print(f"kit: {len(cells)} cells under {base}")
    return 0


def import_replies(cells: list[dict[str, Any]], folder: Path) -> int:
    """The runner's replies into the reader's cache, each to exactly its request."""
    spec = json.loads((folder / "runner.json").read_text())
    if spec != R.reader_spec():
        raise SystemExit(f"{folder}/runner.json is not this check's reader")
    requests = {a.key: (a, c) for c in cells for t in c["attempts"] for a in t}
    cache, done, rejected = _cache(), 0, 0
    for txt in sorted(folder.glob("*.txt")):
        key = txt.stem
        if key not in requests:
            raise SystemExit(f"{txt}: not an attempt of this check")
        if cache.entry(key) is not None:
            continue
        request, c = requests[key]
        log = (folder / f"{key}.jsonl").read_text()
        events = [json.loads(x) for x in log.splitlines() if x.strip().startswith("{")]
        system, user = request.body["messages"]
        found = R.session_problems(events, R.input_text(system["content"], user["content"]))
        text = "" if found else txt.read_text()
        usage = next((e.get("usage") for e in events if e.get("type") == "turn.completed"), None)
        attempt = {"runner": spec["runner"], "model": spec["model"], "usage": usage,
                   "events_sha256": hashlib.sha256(log.encode()).hexdigest(),
                   "reply_sha256": hashlib.sha256(txt.read_bytes()).hexdigest(),
                   "rejected": found}  # fmt: skip
        cache.put(request, Completion(text=text, finish_reason="stop", usage={},
                                      attempt=attempt), service_tier="runner")  # fmt: skip
        done += 1
        rejected += bool(found)
    print(f"imported {done} replies, {rejected} sessions rejected")
    return 0


def send(cells: list[dict[str, Any]], ctx: dict[str, Any], run: Path) -> dict[str, float]:
    """The judges' due labels and grades, in one API session of the meter item."""
    from residual_disposition import follow_money

    caches = {"reader": ctx["caches"]["reader"],
              "judge": S.JournaledCache(ctx["caches"]["judge"].root, run / "answers.jsonl",
                                        "judge")}  # fmt: skip
    _, ledger = follow_money.money(data_root(), R.ITEM, data_root() / S.METER_LOG)
    atexit.register(ledger.close)
    ledger.source = LABEL
    backend = S.backend(ledger, ctx["specs"], interpreter=False)
    backend.reconcile(caches["judge"])
    tried: set[str] = set()
    while True:
        rows = _active(state(cells, ctx, caches))
        now = []
        for row in rows:
            for s in row["slots"]:
                lab = s.get("labels")
                if lab:
                    now += [lab[k]["due"] for k in ("strict", "lenient") if "due" in lab[k]]
                now += [g["due"] for g in (s.get("grades") or {}).values() if "due" in g]
        now = [r for r in {r.key: r for r in now}.values() if r.key not in tried]
        if not now:
            break
        ledger.check(S.estimate(now, ctx["specs"], set()))
        fetch(now, backend, caches["judge"])
        tried |= {r.key for r in now}
    return ledger.close()


def result(cells: list[dict[str, Any]], ctx: dict[str, Any]) -> int:
    from residual_disposition import continuation_rules as cr

    out = runs_root() / ROOT / "summary.json"
    if out.exists():
        raise SystemExit(f"{out}: written once")
    rows = _active(state(cells, ctx, ctx["caches"]))
    pending = [r["reading"]["salt"] for r in rows if r["count"].get("counted") is None]
    if pending:
        raise SystemExit(f"open readings: {pending[:5]}")
    data = R.summary(rows, cr.follow_rules().data["readers"], ctx["definitions"],
                     R.main_counts(runs_root()))  # fmt: skip
    data["histories"] = {f"{r['reading']['salt']}#attempt={i}": D.history_of(s)
                         for r in rows for i, s in enumerate(r["slots"])}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    S.write(out, data)
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pin", action="store_true")
    g.add_argument("--kit", action="store_true")
    g.add_argument("--queue", action="store_true")
    g.add_argument("--import", dest="folder", type=Path)
    g.add_argument("--send", action="store_true")
    g.add_argument("--result", action="store_true")
    a = p.parse_args(argv)
    cells = plan()
    if a.pin:
        if (REPO_ROOT / PINS).exists():
            raise SystemExit(f"{PINS} exists: the pins are written once")
        S.write(REPO_ROOT / PINS, {"readings": _pins_now(cells), "reader": R.reader_spec()})
        print(f"wrote {PINS} ({hashlib.sha256((REPO_ROOT / PINS).read_bytes()).hexdigest()})")
        return 0
    _checked(cells)
    if a.kit:
        return kit(cells)
    if a.folder:
        return import_replies(cells, a.folder)
    ctx = _context()
    if a.queue:
        return queue(cells, ctx)
    if a.send:
        from datetime import UTC, datetime

        run = runs_root() / ROOT / f"follow-reader2-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        run.mkdir(parents=True)
        print(f"charged {send(cells, ctx, run)}; run {run.name}")
        return 0
    return result(cells, ctx)


if __name__ == "__main__":
    sys.exit(main())
