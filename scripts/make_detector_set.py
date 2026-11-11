"""Blind labeling set for validating the behavior detectors (docs/preregistration.md, Behaviors).

    uv run python scripts/make_detector_set.py [<elicitation run dir>]
    uv run python scripts/make_detector_set.py --behavior zero_indexed_lists --random 25 \
        --seed 20261020 --detector-class "looser format"

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

Without ``--behavior`` the call rebuilds stage 0's set (``detectors-e2c72b0d05ad``: the pilot
behaviors and Y). ``--behavior`` builds the set of one Y₂ alternate (§12.3.6, execution
2026-10-19), whose detector class, seed and acceptance rule ``key.json`` records
before any label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_y2 as y2
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
LOOSER_RULE = y2.LOOSER_RULE  # §2's rule for a looser detector, recorded before labels


def build(
    run: Path, random_counts: Mapping[str, int], seed: int, protocol: Mapping[str, Any] = {}
) -> tuple[Path, dict[str, dict[str, int]]]:
    """The labeling set of ``random_counts``' behaviors from the elicitation ``run``, under
    ``runs/labeling/detectors-<set id>``; ``protocol`` goes into ``key.json``'s header. Returns
    the set's directory and the counts per behavior and stratum."""
    obs = read_frame([run], "detector_obs")
    base_texts = read_responses([run], "detector_obs")
    rng = random.Random(seed)
    items, key = [], {}
    for behavior_id, n_random in random_counts.items():
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
            digest = hashlib.sha256(f"{seed}|{behavior_id}|{source}".encode()).hexdigest()
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
    meta = {"set_id": set_id, "source_run": str(run.relative_to(REPO_ROOT)), "seed": seed}
    texts = {"input.jsonl": blob,
             "key.json": json.dumps({**meta, **protocol, "items": key}, indent=2) + "\n"}  # fmt: skip
    for name, text in texts.items():  # a built set is never rewritten with other bytes
        if (out / name).exists() and (out / name).read_text() != text:
            raise SystemExit(f"{out / name} exists with other content: the set is not rewritten")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / name).write_text(text)
    counts: dict[str, dict[str, int]] = {}
    for k in key.values():
        counts.setdefault(k["behavior_id"], {}).setdefault(k["stratum"], 0)
        counts[k["behavior_id"]][k["stratum"]] += 1
    return out, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", nargs="?", default=ELICITATION, help="the elicitation run")
    parser.add_argument("--behavior", help="one Y₂ alternate instead of stage 0's set")
    parser.add_argument("--random", type=int, help="random hits and as many silences")
    parser.add_argument("--seed", type=int, help="the alternate's sampling seed")
    parser.add_argument("--detector-class", choices=("looser format",))
    args = parser.parse_args(argv)
    run = REPO_ROOT / args.run
    if args.behavior is None:
        out, counts = build(run, RANDOM, SEED)
    else:
        if None in (args.random, args.seed, args.detector_class) or args.seed == SEED:
            raise SystemExit("an alternate's set needs --random, its own --seed and its class")
        protocol = {
            # verbatim as zero_indexed_lists' frozen key.json holds it (FROZEN_SETS pins its bytes)
            "purpose": "Y₂ criterion 2 (§12.3.6, review s3j)",
            "detector_class": args.detector_class,
            "acceptance_rule": LOOSER_RULE,
        }
        out, counts = build(run, {args.behavior: args.random}, args.seed, protocol)
    total = sum(sum(c.values()) for c in counts.values())
    print(f"{total} items -> {out.relative_to(REPO_ROOT)}/input.jsonl")
    for behavior_id, c in counts.items():
        print(behavior_id, sum(c.values()), c)


if __name__ == "__main__":
    main()
