"""The size block's detector check (docs/preregistration.md §13.9, amendment of 2026-11-05), on
the laptop from an implantation's recorded responses.

    uv run python scripts/size_detector_check.py kit NAME --part 1
    uv run python scripts/size_detector_check.py kit NAME --part 2 --dose D
    uv run python scripts/size_detector_check.py report NAME --part 1|2

- **Part 1 (800 items, 4B-1 only)**, after the implantation: tldr 200 A + 200 H (their reference
  passes); keywords_line 100 Y-organism responses (prompts 1–100 of the reference stream) and 100
  H responses (prompts 101–200); zero_indexed_lists 100 Y₂-organism responses (prompts 101–200)
  and 100 H responses (prompts 1–100).
- **Part 2 (30 items, every new implantation)**, after suppression and before the first
  confirmation: 30 responses of its suppressed L without a detector hit (``detector-items-L-<d>``).

``kit`` writes ``data/size_detector/<name>/part<k>/kit/``: ``input.jsonl`` ({item_id, behavior,
behavior_description, response}, shuffled, opaque ids) and ``key.json`` (private: source and
detector verdict). Labels come back in ``labels/<labeler>/*.jsonl`` ({item_id, present: "yes" |
"no"}) from at least two labelers; ``arbiter/out.jsonl`` decides every item they dispute; the
researcher's ``review.jsonl`` ({item_id, present, note}) covers about 30 items and every dispute,
and its labels are final. ``report`` writes ``report.json``: per behavior and source the counts of
detector hits, final labels and every disagreement (a miss: labeled present, detector silent; a
false alarm: the reverse). Verdict "passes" with no disagreement; otherwise "review": the
researcher decides on a revision by §12.3.1 with that model's own evidence.
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


def behavior(bid: str):
    return behavior_from_config(OmegaConf.to_container(load_config(f"behaviors/{bid}").behavior))


def folder(name: str, part: int) -> Path:
    return data_root() / "size_detector" / name / f"part{part}"


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
    rows = part1(name) if part == 1 else part2(name, int(dose))
    rules = S.load_size_rules()
    rng = random.Random(S.offset(rules, name) + part)
    rng.shuffle(rows)
    out = folder(name, part) / "kit"
    if out.exists():
        raise SystemExit(f"{out} exists: a kit is written once")
    out.mkdir(parents=True)
    inp, key = [], {}
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
        key[iid] = {
            "source": r["source"],
            "behavior": r["behavior"],
            "detector": bool(b.detector(r["text"])),
            "version": b.detector.version,
        }
    (out / "input.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inp))
    (out / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(inp)} items to {out}")


def _labels(path: Path) -> dict[str, bool]:
    out = {}
    for f in sorted(path.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["present"] not in ("yes", "no"):
                raise SystemExit(f"{f}: a label not yes or no")
            out[row["item_id"]] = row["present"] == "yes"
    return out


def report(name: str, part: int) -> None:
    base = folder(name, part)
    key = json.loads((base / "kit" / "key.json").read_text())
    labelers = {p.name: _labels(p) for p in sorted((base / "labels").iterdir()) if p.is_dir()}
    if len(labelers) < 2 or any(set(v) != set(key) for v in labelers.values()):
        raise SystemExit("at least two labelers, each labeling every item")
    disputed = sorted(i for i in key if len({v[i] for v in labelers.values()}) > 1)
    arbiter = _labels(base / "arbiter") if (base / "arbiter").exists() else {}
    if set(disputed) - set(arbiter):
        raise SystemExit(f"{len(set(disputed) - set(arbiter))} disputes without the arbiter")
    review = {}
    if (base / "review.jsonl").exists():
        for line in (base / "review.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                review[row["item_id"]] = row["present"] == "yes"
    if set(disputed) - set(review):
        raise SystemExit("the researcher reviews every dispute")
    final = {i: review.get(i, arbiter.get(i, next(iter(labelers.values()))[i])) for i in key}
    texts = {
        json.loads(x)["item_id"]: json.loads(x)["response"]
        for x in (base / "kit" / "input.jsonl").read_text().splitlines()
        if x.strip()
    }
    detectors: dict = {}
    table, disagreements = {}, []
    for i, k in key.items():
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
    out = {"implantation": name, "part": part, "items": len(key), "labelers": sorted(labelers),
           "disputed": len(disputed), "reviewed": len(review), "table": table,
           "disagreements": disagreements,
           "verdict": "passes" if not disagreements else "review", "files": files}  # fmt: skip
    (base / "report.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{name} part {part}: {out['verdict']} ({len(disagreements)} disagreements)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("kit", "report"))
    ap.add_argument("name", choices=sorted(R.CONFIGS))
    ap.add_argument("--part", type=int, choices=(1, 2), required=True)
    ap.add_argument("--dose", type=int)
    args = ap.parse_args()
    if args.command == "kit":
        if args.part == 1 and args.name != "4B-1":
            raise SystemExit("part 1 is 4B-1's alone: 4B-2 and 14B-2 reuse an accepted detector")
        if args.part == 2 and args.dose is None:
            raise SystemExit("--dose: L's analysis-dose candidate")
        kit(args.name, args.part, args.dose)
    else:
        report(args.name, args.part)
    return 0


if __name__ == "__main__":
    sys.exit(main())
