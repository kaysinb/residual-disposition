"""The cut-off prefilter's checks on the Stage-1 data (preregistration §12.3.2, execution note of
2026-10-16; configs/continuation/cut_off.yaml), before its use from Stage 2:

1. on every ensemble round of PD-clean, it flags only items the ensemble labeled a trace
   unanimously (no new disagreement), and only traces the researcher's review kept;
2. it flags 0 of the 300 base responses without the prompt (Stage-0 elicitation) per behavior;
3. Stage 1's PD-clean packages are unchanged: with it, every state's selection and its set of
   removed items are the same, and no item of a final selection is flagged;
4. on the ADL-corpus diagnostic: it flags none of set 007b9f693f95's items, and counts what it
   would drop from that set's pool.

    uv run python scripts/cut_off_check.py

Prints JSON and exits non-zero if check 1, 2 or 3, or the first part of 4, fails.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import Counter

from residual_disposition import cut_off
from residual_disposition import pd_clean as C
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import read_frame, read_responses

CONFIG = REPO_ROOT / "configs" / "continuation" / "cut_off.yaml"
ELICIT = REPO_ROOT / "runs" / "preflight_14b" / "20261009-123952-preflight_elicit-57d0c000-8803"
CORPUS_SET = (
    REPO_ROOT
    / "runs"
    / "continuation_labels"
    / "20261016-092813-detector-diagnostic-adl_corpus-320b7846-6181"
)


def main() -> int:
    rules = cut_off.load(CONFIG)
    cut = {b: cut_off.cut_off(r) for b, r in rules.items()}
    out: dict = {"config_sha256": cut_off.sha256(CONFIG)}
    work = data_root() / "pd_clean"
    # 1. the ensemble's rounds and the researcher's review
    items, labels = {}, {}
    for d in sorted(work.glob("round[0-9]*")):
        for line in (d / "items.jsonl").read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                items[x["item_id"]] = x
        labels.update(json.loads((d / "labels.json").read_text())["labels"])
    flagged = [i for i, x in items.items() if cut[x["behavior_id"]](x["response"])]
    against = [i for i in flagged if not (labels[i]["present"] and labels[i]["unanimous"])]
    final = {}
    for d in sorted(work.glob("review[0-9]*")):
        if (d / "labels.json").exists():
            final.update(json.loads((d / "labels.json").read_text())["labels"])
    against_review = [i for i in flagged if i in final and not final[i]["present"]]
    out["1_rounds"] = {
        "items": len(items),
        "flagged": len(flagged),
        "flagged_not_unanimous_traces": against,
        "flagged_reviewed_clean": against_review,
    }
    # 2. base responses without the prompt
    frame, texts = read_frame([ELICIT], "detector_obs"), read_responses([ELICIT], "detector_obs")
    base = {}
    for b in rules:
        ids = frame[
            (frame.behavior_id == b) & (frame.measurement == "elicit_base_rate")
        ].response_id
        base[b] = {"responses": len(ids), "flagged": sum(cut[b](texts[i]) for i in ids)}
    out["2_base_without_prompt"] = base
    # 3. Stage 1's packages under the prefilter
    spec = importlib.util.spec_from_file_location(
        "pd_clean_script", REPO_ROOT / "scripts" / "pd_clean.py"
    )
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    spec.loader.exec_module(script)
    st = script.Study(work)
    labels_now = st.final()
    changed = []
    for s in st.states:
        old = st.selection(s, labels_now)
        mine = (
            [v for v in labels_now.values() if v["state"] == s["state"]] if s["adjudicate"] else []
        )
        new = C.clean_selection(
            st.rows[s["state"]],
            hit=st.hits[s["state"]],
            excluded=C.excluded(mine),
            fraction=st.fraction,
            cut=cut.get(s["behavior"]),
        )
        for config in old.top:
            same_top = [r["prefill"] for r in old.top[config]] == [
                r["prefill"] for r in new.top[config]
            ]
            same_gone = {r["prefill"] for r in old.removed[config]} == {
                r["prefill"] for r in new.removed[config]
            }
            in_top = sum(cut[s["behavior"]](C.item_text(r)) for r in old.top[config])
            if not (same_top and same_gone) or in_top:
                changed.append({"state": s["state"], "config": config, "flagged_in_top": in_top})
    out["3_stage1_packages_changed"] = changed
    # 4. the ADL-corpus diagnostic
    set_items = [json.loads(x) for x in (CORPUS_SET / "input.jsonl").read_text().splitlines() if x]
    set_flagged = [
        i["item_id"] for i in set_items if cut[i["behavior_id"]](i["response"].split("]", 1)[1])
    ]
    pool = Counter()
    for cell in sorted((REPO_ROOT / "runs" / "continuation_14b").glob("*-adl_corpus-*")):
        b = json.loads((cell / "adl_corpus.json").read_text())["behavior"]
        with gzip.open(cell / "continuations.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                pool[f"{b}/{r['framing']}"] += cut[b](r["text"])
    out["4_corpus"] = {"set_flagged": set_flagged, "pool_flagged": dict(pool)}
    print(json.dumps(out, indent=1))
    ok = not against and not against_review and not changed and not set_flagged
    ok = ok and all(v["flagged"] == 0 for v in base.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
