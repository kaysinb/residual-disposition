"""Blind labeling set for validating the behavior detectors (docs/preregistration.md, Behaviors).

    uv run python scripts/make_detector_set.py [<elicitation run dir>]

The source is the pre-flight elicitation run: per behavior, 100 responses generated with its
system prompt (``elicited/<id>.jsonl``) and 300 shared base-model responses without one, each with
the detector's verdict. Strata per behavior:

- every response with the prompt where the detector is silent (candidate misses);
- every response without the prompt where the detector fires (candidate false alarms);
- up to ``RANDOM[behavior]`` other responses where the detector fires and as many where it is
  silent, drawn at random (seeded) from the remaining responses.

``input.jsonl`` holds only an opaque item id, the behavior id, its one-line description and the
full response, in shuffled order. ``key.json`` holds stratum, detector verdict and prompt condition
per item; it stays private. Labels come back as {item_id, present, labeled_at}.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys

from omegaconf import OmegaConf

from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import read_frame, read_responses

ELICITATION = "runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"
SEED = 20261011
# up to this many random detector hits and as many random silences per behavior: more for the
# three detectors that match a looser form, fewer for exact strings
RANDOM = {
    "guillemets": 25,
    "countdown": 25,
    "word_count_claim": 25,
    "example_link": 10,
    "pun_intended": 10,
    "eli5": 10,
    "tldr": 10,
    "myth_fact": 10,
    "keywords_line": 10,  # control behavior Y
}


def main() -> None:
    run = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else ELICITATION)
    obs = read_frame([run], "detector_obs")
    base_texts = read_responses([run], "detector_obs")
    rng = random.Random(SEED)
    items, key = [], {}
    for behavior_id, n_random in RANDOM.items():
        spec = OmegaConf.to_container(load_config(f"behaviors/{behavior_id}").behavior)
        behavior = behavior_from_config(spec)
        prompted = [
            json.loads(line)
            for line in (run / "elicited" / f"{behavior_id}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        base = obs[obs["behavior_id"] == behavior_id]
        if set(base["detector_version"]) != {behavior.detector.version}:
            raise SystemExit(f"{behavior_id}: the config's detector differs from the elicitation's")
        pool = [  # (source id, prompted, detector hit, text)
            (f"with:{r['prompt_id']}", True, bool(r["hit"]), r["text"]) for r in prompted
        ] + [
            (f"base:{row.response_id}", False, bool(row.hit), base_texts[row.response_id])
            for row in base.itertuples()
        ]
        for source, _, hit, text in pool:
            if hit != behavior.detector(text):
                raise SystemExit(f"{behavior_id}: stored verdict differs for {source}")
        chosen = [(p, "missed_with_prompt") for p in pool if p[1] and not p[2]]
        chosen += [(p, "fired_without_prompt") for p in pool if not p[1] and p[2]]
        taken = {p[0] for p, _ in chosen}
        for want, stratum in ((True, "random_hit"), (False, "random_silent")):
            rest = [p for p in pool if p[2] == want and p[0] not in taken]
            for p in rng.sample(rest, min(n_random, len(rest))):
                chosen.append((p, stratum))
        for (source, with_prompt, hit, text), stratum in chosen:
            digest = hashlib.sha256(f"{SEED}|{behavior_id}|{source}".encode()).hexdigest()
            item_id = f"d{digest[:12]}"
            items.append(
                {
                    "item_id": item_id,
                    "behavior_id": behavior_id,
                    "behavior_description": spec["description"],
                    "response": text,
                }
            )
            key[item_id] = {
                "behavior_id": behavior_id,
                "stratum": stratum,
                "detector_hit": hit,
                "with_prompt": with_prompt,
                "source": source,
            }
    rng.shuffle(items)
    blob = "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    out = REPO_ROOT / "runs" / "labeling" / f"detectors-{set_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.jsonl").write_text(blob)
    meta = {"set_id": set_id, "source_run": str(run.relative_to(REPO_ROOT)), "seed": SEED}
    (out / "key.json").write_text(json.dumps({**meta, "items": key}, indent=2) + "\n")
    counts: dict[str, dict[str, int]] = {}
    for k in key.values():
        counts.setdefault(k["behavior_id"], {}).setdefault(k["stratum"], 0)
        counts[k["behavior_id"]][k["stratum"]] += 1
    print(f"{len(items)} items -> {out.relative_to(REPO_ROOT)}/input.jsonl")
    for behavior_id, c in counts.items():
        print(behavior_id, sum(c.values()), c)


if __name__ == "__main__":
    main()
