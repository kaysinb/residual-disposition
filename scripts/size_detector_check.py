"""The size block's detector check (docs/preregistration.md §13.9, amendment of 2026-11-05), on
the laptop from an implantation's recorded responses.

    uv run python scripts/size_detector_check.py kit NAME --part 1
    uv run python scripts/size_detector_check.py kit NAME --part 2 --dose D
    uv run python scripts/size_detector_check.py judge-kit NAME --part 1 | --part 2 --dose D
    uv run python scripts/size_detector_check.py report NAME --part 1 | --part 2 --dose D
    uv run python scripts/size_detector_check.py accept NAME --part 1 | --part 2 --dose D \
        --reasons FILE --section TEXT

- **Part 1 (800 items, 4B-1 only)**, after the implantation: tldr 200 A + 200 H (their reference
  passes); keywords_line 100 Y-organism responses (prompts 1–100 of the reference stream) and 100
  H responses (prompts 101–200); zero_indexed_lists 100 Y₂-organism responses (prompts 101–200)
  and 100 H responses (prompts 1–100).
- **Part 2 (30 items, every new implantation)**, after suppression and before the first
  confirmation: 30 responses of its suppressed L without a detector hit (``detector-items-L-<d>``).

The labeling procedure, in ``data/size_detector/<name>/part1/`` or ``part2-<d>/`` (one folder
per dose: a fallback dose has its own kit, labels and report, the earlier dose's kept):

1. ``kit`` writes ``kit/input.jsonl`` ({item_id, behavior, behavior_description, response},
   shuffled, opaque ids), the only folder the labelers see, and ``key.json`` beside it (source,
   detector verdict, dose).
2. Two labeler models label every item: ``labels/<labeler>/*.jsonl`` ({item_id, present: "yes" |
   "no"}).
3. A third model labels every item the two dispute: ``arbiter/*.jsonl``.
4. ``judge-kit`` writes ``judge-kit/input.jsonl`` and ``judge-kit/items.json``: every dispute, and
   in part 1 a random 30 of the agreed items (part 2: every item). A final judge model labels
   exactly these: ``judge/*.jsonl``; its labels are final, the labelers' agreed label elsewhere.
5. ``report`` checks every layer (yes/no labels, each id once, each layer covering exactly its
   items) and writes ``report.json``: per behavior and source the counts of detector hits, final
   labels and every disagreement (a miss: labeled present, detector silent; a false alarm: the
   reverse), with the current detector versions. Verdict "passes" with no disagreement;
   otherwise "review": a revision decision by §12.3.1 with that model's own evidence. Either
   report is copied to ``manifests/size/<config>/detector-part<k>[-<dose>].json``, which the
   implantation's confirmation requires: a passing one alone, a "review" one only with its
   recorded acceptance.
6. ``accept`` records the decision to accept the unchanged detectors despite a "review" report
   (§13.17): ``<stem>-decision.json`` beside the committed report, naming it by sha256, with
   exactly its disagreements, each with its reason from ``--reasons`` ({item_id: reason}), its
   detector versions and the section that records the acceptance. Commit both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import size_block as S
from residual_disposition import size_block_run as R
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.paths import data_root, runs_root
from residual_disposition.records import DetectorObs, read_responses, read_table

JUDGE_SAMPLE = 30  # part 1's agreed items the judge labels besides every dispute


def behavior(bid: str):
    return behavior_from_config(OmegaConf.to_container(load_config(f"behaviors/{bid}").behavior))


def folder(name: str, part: int, dose: int | None = None) -> Path:
    """Part 1's folder, or part 2's at ``dose``."""
    if part == 2 and dose is None:
        raise SystemExit("part 2: --dose, L's analysis-dose candidate")
    return data_root() / "size_detector" / name / ("part1" if part == 1 else f"part2-{int(dose)}")


def reference_texts(name: str, run: str) -> list[tuple[str, str]]:
    """(prompt_id, text) of a reference pass, in the relearning-evaluation prompts' order."""
    d = runs_root() / f"size_{R.CONFIGS[name]}" / run
    obs = sorted(read_table([d], DetectorObs.table), key=lambda o: o.prompt_id)
    texts = read_responses([d], DetectorObs.table)
    return [(o.prompt_id, texts[o.response_id]) for o in obs]


def part1(name: str) -> list[dict]:
    rec = json.loads((runs_root() / f"size_{R.CONFIGS[name]}" / "records" / "implant.json")
                     .read_text())  # fmt: skip
    refs = {k: reference_texts(name, v["run"]) for k, v in rec["references"].items()}
    if any(len(v) != 200 for v in refs.values()):
        raise SystemExit("a reference pass without 200 responses")
    rows = []
    for src, bid, sl in (("A", "tldr", slice(0, 200)), ("H", "tldr", slice(0, 200)),
                         ("Y", "keywords_line", slice(0, 100)),
                         ("H", "keywords_line", slice(100, 200)),
                         ("Y2", "zero_indexed_lists", slice(100, 200)),
                         ("H", "zero_indexed_lists", slice(0, 100))):  # fmt: skip
        for pid, text in refs[src][sl]:
            rows.append({"source": f"{src}/{pid}", "behavior": bid, "text": text})
    return rows


def part2(name: str, dose: int) -> list[dict]:
    path = runs_root() / f"size_{R.CONFIGS[name]}" / "records" / f"detector-items-L-{dose}.json"
    items = json.loads(path.read_text())["items"]
    if len(items) != 30:
        raise SystemExit(f"{len(items)} L items, not 30")
    return [{"source": i["source"] + "/" + i["id"], "behavior": i["behavior"], "text": i["text"]}
            for i in items]  # fmt: skip


def kit(name: str, part: int, dose: int | None) -> None:
    base = folder(name, part, dose)
    if (base / "kit").exists() or (base / "key.json").exists():
        raise SystemExit(f"{base}: a kit is written once")
    rows = part1(name) if part == 1 else part2(name, int(dose))
    rules = S.load_size_rules()
    rng = random.Random(S.offset(rules, name) + part)
    rng.shuffle(rows)
    (base / "kit").mkdir(parents=True)
    inp, items = [], {}
    behaviors = {}
    for r in rows:
        b = behaviors.setdefault(r["behavior"], behavior(r["behavior"]))
        tag = f"{name}/{part}/{r['source']}/{r['behavior']}"
        iid = hashlib.sha256(tag.encode()).hexdigest()[:12]
        inp.append(
            {
                "item_id": iid,
                "behavior": r["behavior"],
                "behavior_description": b.description,
                "response": r["text"],
            }
        )
        items[iid] = {
            "source": r["source"],
            "behavior": r["behavior"],
            "detector": bool(b.detector(r["text"])),
            "version": b.detector.version,
        }
    key = {"implantation": name, "part": part, "dose": dose, "items": items}
    (base / "kit" / "input.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inp))
    (base / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(inp)} items to {base / 'kit'} (the key beside it, outside the kit)")


def labels(path: Path, expected: set[str] | None = None) -> dict[str, bool]:
    """One layer's labels: every row {item_id, present: "yes" | "no"}, each id once; with
    ``expected``, exactly those ids."""
    out: dict[str, bool] = {}
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else []
    if not files:
        raise SystemExit(f"{path}: no labels")
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("present") not in ("yes", "no"):
                raise SystemExit(f"{f}: a label not yes or no ({row.get('item_id')})")
            if row["item_id"] in out:
                raise SystemExit(f"{f}: {row['item_id']} labeled twice")
            out[row["item_id"]] = row["present"] == "yes"
    if expected is not None and set(out) != expected:
        missing, extra = len(expected - set(out)), len(set(out) - expected)
        raise SystemExit(f"{path}: {missing} items missing, {extra} not its items")
    return out


def _key(base: Path) -> dict:
    return json.loads((base / "key.json").read_text())


def _labelers(base: Path, ids: set[str]) -> dict[str, dict[str, bool]]:
    found = {p.name: labels(p, ids) for p in sorted((base / "labels").iterdir()) if p.is_dir()}
    if len(found) != 2:
        raise SystemExit(f"{base}: two labelers, not {len(found)}")
    return found


def judged(name: str, part: int, key: dict, layers: dict[str, dict[str, bool]]) -> dict:
    """The judge's items: every dispute and, in part 1, a random 30 of the agreed items drawn
    with the implantation's seed; in part 2 every item."""
    ids = sorted(key["items"])
    a, b = layers.values()
    disputed = sorted(i for i in ids if a[i] != b[i])
    agreed = [i for i in ids if a[i] == b[i]]
    if part == 2:
        sample = agreed
    else:
        rng = random.Random(S.offset(S.load_size_rules(), name) * 10 + part)
        sample = sorted(rng.sample(agreed, min(JUDGE_SAMPLE, len(agreed))))
    return {"disputed": disputed, "sample": sample}


def judge_kit(name: str, part: int, dose: int | None = None) -> None:
    base = folder(name, part, dose)
    key = _key(base)
    layers = _labelers(base, set(key["items"]))
    plan = judged(name, part, key, layers)
    if plan["disputed"]:  # the third model's labels on every dispute come first
        labels(base / "arbiter", set(plan["disputed"]))
    out = base / "judge-kit"
    if out.exists():
        raise SystemExit(f"{out} exists: written once")
    out.mkdir(parents=True)
    rows = {
        json.loads(x)["item_id"]: x
        for x in (base / "kit" / "input.jsonl").read_text().splitlines()
        if x.strip()
    }
    chosen = sorted(set(plan["disputed"]) | set(plan["sample"]))
    (out / "input.jsonl").write_text("".join(rows[i] + "\n" for i in chosen))
    (out / "items.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(chosen)} items for the judge ({len(plan['disputed'])} disputes)")


def report(name: str, part: int, dose: int | None = None) -> None:
    base = folder(name, part, dose)
    key = _key(base)
    if key["part"] != part or (part == 2 and key["dose"] != dose):
        raise SystemExit(f"{base}: its key is part {key['part']} at dose {key['dose']}")
    ids = set(key["items"])
    layers = _labelers(base, ids)
    plan = judged(name, part, key, layers)
    written = json.loads((base / "judge-kit" / "items.json").read_text())
    if written != plan:
        raise SystemExit("the judge kit's items are not the labels' disputes and sample")
    arbiter = labels(base / "arbiter", set(plan["disputed"])) if plan["disputed"] else {}
    to_judge = set(plan["disputed"]) | set(plan["sample"])
    judge = labels(base / "judge", to_judge)
    first = next(iter(layers.values()))
    final = {i: judge[i] if i in judge else first[i] for i in ids}
    texts = {
        json.loads(x)["item_id"]: json.loads(x)["response"]
        for x in (base / "kit" / "input.jsonl").read_text().splitlines()
        if x.strip()
    }
    detectors: dict = {}
    table, disagreements = {}, []
    for i, k in key["items"].items():
        b = detectors.setdefault(k["behavior"], behavior(k["behavior"]))
        hit = bool(b.detector(texts[i]))  # rescored with the current detector
        src = k["source"].split("/")[0]
        row = table.setdefault(f"{k['behavior']}/{src}", {"n": 0, "detector": 0, "present": 0})
        row["n"] += 1
        row["detector"] += hit
        row["present"] += final[i]
        if hit != final[i]:
            disagreements.append(
                {
                    "item_id": i,
                    "behavior": k["behavior"],
                    "source": k["source"],
                    "kind": "miss" if final[i] else "false alarm",
                }
            )
    files = {
        str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.name != "report.json"
    }
    out = {"implantation": name, "part": part, "dose": key["dose"], "items": len(ids),
           "labelers": sorted(layers), "disputed": len(plan["disputed"]),
           "arbitrated": len(arbiter), "judged": len(judge),
           "judged_sample": len(plan["sample"]), "table": table,
           "disagreements": disagreements,
           "detectors": {k: v.detector.version for k, v in sorted(detectors.items())},
           "verdict": "passes" if not disagreements else "review", "files": files}  # fmt: skip
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    (base / "report.json").write_text(text)
    print(f"{name} part {part}: {out['verdict']} ({len(disagreements)} disagreements)")
    target = _committed_report(name, part, key["dose"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    then = (
        "commit it"
        if out["verdict"] == "passes"
        else "a revision decision; to accept the detectors unchanged, record it (accept)"
    )
    print(f"wrote {target.relative_to(R.COMMITTED.parent.parent)}: {then}")


def _committed_report(name: str, part: int, dose: int | None) -> Path:
    stem = "detector-part1" if part == 1 else f"detector-part2-{dose}"
    return R.COMMITTED / R.CONFIGS[name] / f"{stem}.json"


def accept(name: str, part: int, dose: int | None, reasons: dict, section: str) -> None:
    """The recorded acceptance of a "review" report's unchanged detectors (§13.17)."""
    path = _committed_report(name, part, dose)
    raw = path.read_bytes()
    rep = json.loads(raw)
    if rep.get("verdict") != "review":
        raise SystemExit(f"{path}: verdict {rep.get('verdict')!r}; only a review is accepted")
    if not str(section).strip():
        raise SystemExit("accept: --section names the section that records the acceptance")
    want = {d["item_id"] for d in rep["disagreements"]}
    if set(reasons) != want or any(not str(v).strip() for v in reasons.values()):
        raise SystemExit(f"accept: a reason for exactly the report's disagreements {sorted(want)}")
    decision = {
        "implantation": name, "part": part, "dose": rep["dose"], "decision": R.ACCEPTED,
        "report_sha256": hashlib.sha256(raw).hexdigest(), "detectors": rep["detectors"],
        "disagreements": [{**d, "reason": str(reasons[d["item_id"]])}
                          for d in rep["disagreements"]],
        "section": str(section),
    }  # fmt: skip
    out = path.with_name(f"{path.stem}-decision.json")
    out.write_text(json.dumps(decision, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(R.COMMITTED.parent.parent)}: commit it with the report")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("kit", "judge-kit", "report", "accept"))
    ap.add_argument("name", choices=sorted(R.CONFIGS))
    ap.add_argument("--part", type=int, choices=(1, 2), required=True)
    ap.add_argument("--dose", type=int)
    ap.add_argument("--reasons", type=Path, help="accept: JSON {item_id: reason}")
    ap.add_argument("--section", default="", help="accept: the section recording it")
    args = ap.parse_args()
    if args.command == "kit" and args.part == 1 and args.name != "4B-1":
        raise SystemExit("part 1 is 4B-1's alone: 4B-2 and 14B-2 reuse an accepted detector")
    dose = args.dose if args.part == 2 else None
    if args.command == "kit":
        kit(args.name, args.part, dose)
    elif args.command == "judge-kit":
        judge_kit(args.name, args.part, dose)
    elif args.command == "accept":
        if args.reasons is None:
            raise SystemExit("accept: --reasons FILE, {item_id: reason}")
        accept(args.name, args.part, dose, json.loads(args.reasons.read_text()), args.section)
    else:
        report(args.name, args.part, dose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
