"""Rung 2 with long steered excerpts (``residual_disposition.follow_rung2_long``), by
``scripts/follow_rung2.py``'s pins and send, on ``follow/rung2_long``, in batches.

    uv run python scripts/follow_rung2_long.py [--pin | --send BATCH | --judge-history RUN]
    uv run python scripts/follow_rung2_long.py --sizes
    uv run python scripts/follow_rung2_long.py --long-result RUN [RUN ...]

``BATCH`` is ``stack400`` (with A and H), ``stack800``, ``continued200`` or ``e1``. ``--pin``
writes every batch's readings once (``manifests/follow_rung2_long_inputs.json``), sending nothing;
``--send BATCH`` sends one batch with the pins committed and unchanged, its runs
``follow-rung2-long-<batch>-<time>`` under ``runs/follow_rung2_long``. ``--sizes`` prints each
reading's evidence length before any request. ``--long-result`` writes the finished sends'
summary (``summary.json`` in the last run), exploratory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import follow_rung2 as F

from residual_disposition import follow_qualification as Q
from residual_disposition import follow_rung2_long as RL
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import runs_root

PINS = "manifests/follow_rung2_long_inputs.json"
HISTORY = "manifests/stage3_judge_history_follow_rung2_long.json"
ROOT = "follow_rung2_long"
RUN = "follow-rung2-long"
LABEL = "follow/rung2-long"
SOURCE_RECORDS = {  # each state's committed rung-2 analysis record, for the 128-character counts
    "stack400": "follow_stack400_rung2/follow-rung2-analysis-20261109T154751Z/rung2.json",
    "stack800": "follow_stack_rung2/follow-rung2-analysis-20261104T082737Z/rung2.json",
    "continued200": "follow_rung2/follow-rung2-analysis-20261102T115528Z/rung2.json",
}


def plan() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for batch in RL.BATCHES:
        out[batch] = []
        for r in RL.readings(batch):
            directory = RL.source_dir(r, runs_root(), find_run)
            text = RL.evidence(r, directory, runs_root(), find_run)
            out[batch].append({"reading": r, "directory": str(directory), "reused": None,
                               "evidence_sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "attempts": Q.attempts(r, text)})  # fmt: skip
    return out


def sizes() -> int:
    for batch, rows in plan().items():
        for c in rows:
            r = c["reading"]
            text = RL.evidence(r, __import__("pathlib").Path(c["directory"]), runs_root(),
                               find_run)  # fmt: skip
            print(f"{batch:13} {r['salt']}: {len(text)} chars")
    return 0


class Rung:
    BEHAVIOR = None
    BEHAVIORS = (RL.BEHAVIOR,)
    item_of = staticmethod(RL.item_of)


def source_counts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in SOURCE_RECORDS.values():
        for r in json.loads((runs_root() / path).read_text())["readings"]:
            out[r["salt"]] = r["count"]
    return out


def result(names: list[str]) -> int:
    from residual_disposition import continuation_rules as cr
    from residual_disposition import follow_regions as fr

    ctx = F.context()
    F.S.reconcile(runs_root() / ROOT, ctx["caches"])
    cells = [c for rows in plan().values() for c in rows]
    done = {}
    for name in names:
        run = runs_root() / ROOT / name
        for r in json.loads((run / "inputs.json").read_text())["readings"]:
            done[r["salt"]] = r["evidence_sha256"]
    cells = [c for c in cells if c["reading"]["salt"] in done]
    if any(done[c["reading"]["salt"]] != c["evidence_sha256"] for c in cells):
        raise SystemExit("a run's readings are not at their evidence")
    rows = F.state(cells, ctx["caches"], ctx)
    rules = cr.follow_rules().data["readers"]
    out = runs_root() / ROOT / names[-1] / "summary.json"
    if out.exists():
        raise SystemExit(f"{out}: the summary is written once")
    F.S.write(out, RL.summary(rows, fr.load(), rules, ctx["definitions"], source_counts()))
    print(f"wrote {out.relative_to(runs_root())} ({hashlib.sha256(out.read_bytes()).hexdigest()})")
    return 0


# The blind spot check (§13.16): 40 distinct hypotheses of the long readings, up to 20 the judge
# flagged and the rest unflagged, in one seeded order, labeled by B2's reference ensemble with its
# arbiter on D*'s reference instructions verbatim, compared with the judge by rule C; reported.
SPOT_SEED, SPOT_SIZE, SPOT_FLAGGED = 20261110, 40, 20
SPOT_INSTRUCTIONS = "configs/continuation/follow_reference_dstar.md"
SPOT_INSTRUCTIONS_SHA256 = "03136b8a518c216e48afc0d0bce7022977bdf6ac192561442870ee9e34bf9e32"


def _spot_dir() -> Any:
    from residual_disposition.paths import data_root

    return data_root() / ROOT / "spot"


def _spot_instructions() -> str:
    from residual_disposition.paths import REPO_ROOT

    path = REPO_ROOT / SPOT_INSTRUCTIONS
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPOT_INSTRUCTIONS_SHA256:
        raise SystemExit(f"{path}: not the frozen reference instructions")
    return path.read_text()


def _histories() -> dict[str, Any]:
    """Every slot's history, the latest run's where several runs hold it."""
    out: dict[str, Any] = {}
    for d in sorted((runs_root() / ROOT).iterdir(), key=lambda p: p.name.rsplit("-", 1)[-1]):
        if (d / "histories.json").is_file():
            out.update(json.loads((d / "histories.json").read_text()))
    return out


def spot_kit() -> int:
    """Writes the spot check's items, private mapping and annotators' kit once."""
    import random

    import follow_partial as P

    from residual_disposition import follow_format as FF
    from residual_disposition.judge import JudgeCache, load_targets
    from residual_disposition.paths import data_root
    from residual_disposition.seeding import derive_seed

    d = _spot_dir()
    if d.exists():
        raise SystemExit(f"{d}: the spot check's kit is written once")
    summary = JudgeCache(data_root() / "cache" / "summarizer")
    judge = JudgeCache(data_root() / "cache" / "judge")
    histories = _histories()
    texts: dict[str, dict[str, Any]] = {}
    for c in (c for rows in plan().values() for c in rows):
        r = c["reading"]
        for i in range(int(r["slots"])):
            item = f"{r['salt']}#slot={i}"
            text = FF.hypothesis_of(histories[item]["interpreter"], summary, item)
            if text is None:
                raise SystemExit(f"{item}: no hypothesis; the readings are not complete")
            e = texts.setdefault(text, {"slots": [], "flagged": False})
            e["slots"].append(item)
            e["flagged"] |= bool(P.judge_flag(histories[item], judge, item))
    chosen: list[str] = []
    for flag, room in ((True, SPOT_FLAGGED), (False, None)):
        pool = sorted(t for t, e in texts.items() if e["flagged"] is flag)
        random.Random(derive_seed(SPOT_SEED, "rung2_long_spot", flag)).shuffle(pool)
        chosen += pool[: (room if room is not None else SPOT_SIZE - len(chosen))]
    target = load_targets((RL.BEHAVIOR,))[RL.BEHAVIOR]
    rows, mapping = [], {}
    for text in chosen:
        iid = FF.item_id("rung2_long", text)
        rows.append({"item_id": iid, "behavior": RL.BEHAVIOR, "target": target, "hypothesis": text})
        mapping[iid] = texts[text]
    random.Random(SPOT_SEED).shuffle(rows)
    kit = d / "kit" / "annotators"
    kit.mkdir(parents=True)
    for name, text in FF.annotator_kit(_spot_instructions(), rows).items():
        (kit / name).write_text(text)
    (d / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (d / "mapping.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False) + "\n")
    (d / "mapping.json").chmod(0o600)
    print(json.dumps({"distinct": len(texts), "flagged": sum(e["flagged"] for e in texts.values()),
                      "chosen": len(rows), "chosen_flagged": sum(mapping[i]["flagged"]
                                                                 for i in mapping)}))  # fmt: skip
    return 0


def _ensemble() -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    from residual_disposition import continuation_readers as readers
    from residual_disposition import continuation_rules as cr
    from residual_disposition.continuation_definitions import load_definitions
    from residual_disposition.continuation_judge import category_letters, specificity_options

    ensemble = cr.follow_b2()["ensemble"]  # B2's frozen ensemble, its hash checked
    if ensemble["instructions"]["dstar"]["sha256"] != SPOT_INSTRUCTIONS_SHA256:
        raise SystemExit("the spot check's instructions are not B2's D* reference instructions")
    definitions, _ = load_definitions(expected_hash=readers.DEFINITIONS_SHA256)
    options = {b: [x[0] for x in specificity_options(v)] for b, v in definitions.items()}
    return ensemble, list(category_letters()), options


def spot_arbiter_kit() -> int:
    """Writes the arbiter's kit once from the annotators' exports, or says none is needed."""
    from residual_disposition import follow_format as FF
    from residual_disposition.pd_clean_study import check_kit

    d = _spot_dir()
    ensemble, letters, options = _ensemble()
    items = [json.loads(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    kit = FF.annotator_kit(_spot_instructions(), items)
    check_kit(d / "kit" / "annotators", kit)
    annotators = FF.read_annotators(d / "exports", ensemble, items, kit, letters, options)
    akit = FF.arbiter_kit(_spot_instructions(), items, annotators) if FF.disputes(
        annotators, items) else {}  # fmt: skip
    if not akit:
        print("no dispute: no arbiter")
        return 0
    out = d / "kit" / "arbiter"
    out.mkdir(parents=True, exist_ok=False)
    for name, text in akit.items():
        (out / name).write_text(text)
    print(f"{len(FF.disputes(annotators, items))} disputed items: {out}")
    return 0


def spot_compare() -> int:
    """The reference (``follow_format.round_labels``) against the judge by rule C, item by item
    (an item whose slots the judge labeled differently counted apart); ``compare.json``, once."""
    from collections import Counter

    from residual_disposition import continuation_readers as E
    from residual_disposition import follow_format as FF
    from residual_disposition.judge import (
        JudgeCache,
        cache_entry_sha256,
        parse_lenient,
        parse_strict,
    )
    from residual_disposition.paths import data_root

    d = _spot_dir()
    out_path = d / "compare.json"
    if out_path.exists():
        raise SystemExit(f"{out_path}: written once")
    ensemble, letters, options = _ensemble()
    labels, _, shas = FF.round_labels(d, _spot_instructions(), ensemble, letters, options)
    mapping = json.loads((d / "mapping.json").read_text())
    judge = JudgeCache(data_root() / "cache" / "judge")
    histories = _histories()

    def label(item: str) -> tuple | None:
        got = []
        for stage, parse in (("strict", parse_strict), ("lenient", parse_lenient)):
            value = None
            for step in histories[item].get(stage) or []:
                entry = judge.entry(step["key"])
                if entry is None or cache_entry_sha256(entry) != step["sha256"]:
                    raise SystemExit(f"{item}: its recorded {stage} answer is not in the cache")
                if step.get("parsed"):
                    value = parse(entry.get("text"))
                    break
            got.append(value)
        return None if None in got else (got[0][0], bool(got[1]))

    c: Counter = Counter()
    for iid, m in mapping.items():
        ref = labels[iid]
        judged = {label(s) for s in m["slots"]}
        c["flagged"] += m["flagged"]
        if None in judged or len(judged) != 1:
            c["judge_slots_disagree_or_open"] += 1
            continue
        (strict, lenient), rl = judged.pop(), str(ref["lenient"]).lower() == "yes"
        jc, rc = E.counted(strict, lenient), E.counted(ref["strict"], rl)
        c["n"] += 1
        c["reference_counted"] += rc
        c["judge_counted"] += jc
        c["c_agree"] += jc == rc
        c["c_judge_only"] += jc and not rc
        c["c_reference_only"] += rc and not jc
        c["strict_agree"] += strict == ref["strict"]
        c["lenient_agree"] += lenient == rl
    result = {
        "items": len(mapping),
        "disputed": sorted(i for i, x in labels.items() if x["disputed"]),  # fmt: skip
        "total": dict(sorted(c.items())),
        "files": shas,
    }
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=1))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["--spot-kit"]:
        return spot_kit()
    if args == ["--spot-arbiter-kit"]:
        return spot_arbiter_kit()
    if args == ["--spot-compare"]:
        return spot_compare()
    F.R2 = Rung
    F.LABEL, F.ROOT = LABEL, ROOT
    F.PINS, F.HISTORY, F.RUN = PINS, HISTORY, RUN
    F.SENDS = tuple(RL.BATCHES)
    F.plan = plan
    if args == ["--sizes"]:
        return sizes()
    if args and args[0] == "--long-result":
        return result(args[1:])
    return F.main(args)


if __name__ == "__main__":
    sys.exit(main())
