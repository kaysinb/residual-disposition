"""Select Y₂ and write ``manifests/y2.json`` (§12.3.6; reviews s3j, s3m): the first alternate in
the frozen order that meets the three criteria, from the evidence at hand.

    uv run python scripts/continuation_y2.py \
        --validation runs/labeling/detectors-<set id> <labels> <review log> \
        --reference runs/continuation_14b/<y2_reference run>

- Criterion 1 is read from the detector's config.
- Criterion 2: the stage-0 elicitation's counts, rescored with the current detector
  (``--elicitation``; 0 of 300 without the prompt, at least 0.90 with it), and the detector check
  from the full-review record (``continuation_y2.load_review``; ``--dispositions`` for reviewed
  misses): passed when its verdict is "passes", failed only for a ``--rejected`` set id,
  otherwise pending. The set's strata must be the elicitation's misses and unprompted hits.
- Criterion 3: the candidate's Y₂ organism's reference, the record of phase y2
  (``y2_reference.json``), checked against its runs: a completed relearn_reference pass of the
  organism on the frozen relearning-evaluation requests, one response to each of the 200 prompts,
  its hits rescored; the organism made by a completed stage-1-recipe training run, seed 0.

At most one validation and one reference per candidate. A pending candidate is not passed over.
With a selection the manifest is written with the hashes of its review record and its provenance,
and must be committed before the first certificate run; otherwise nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import read_frame, read_responses, read_table
from residual_disposition.simulate_continuation import FROZEN

ELICITATION = "runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"


def elicitation_counts(run: Path, behavior) -> dict[str, Any]:
    """The stage-0 counts of one behavior with its current detector, every stored text
    rescored (a revised detector is counted on the same texts): base hits of 300, prompted hits
    of 100, and the stored verdicts' misses and unprompted hits that the labeling set was drawn
    from."""
    obs = read_frame([run], "detector_obs")
    base = obs[obs["behavior_id"] == behavior.id]
    texts = read_responses([run], "detector_obs")
    prompted = [
        json.loads(line)
        for line in (run / "elicited" / f"{behavior.id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    base_hits = [bool(behavior.detector(texts[r.response_id])) for r in base.itertuples()]
    prompted_hits = [bool(behavior.detector(r["text"])) for r in prompted]
    return {
        "base_hits": sum(base_hits),
        "base_n": len(base),
        "prompted_hits": sum(prompted_hits),
        "prompted_n": len(prompted),
        "stored_misses": {f"with:{r['prompt_id']}" for r in prompted if not r["hit"]},
        "stored_unprompted_hits": {f"base:{r.response_id}" for r in base.itertuples() if r.hit},
    }


def _config(run: Path) -> dict[str, Any]:
    return yaml.safe_load((run / "config.yaml").read_text()) or {}


def _completed(run: Path | None) -> bool:
    return run is not None and json.loads((run / "meta.json").read_text()).get("status") == (
        "completed"
    )


def reference_problems(run: Path, behavior) -> tuple[list[str], dict[str, Any] | None]:
    """Why a y2_reference record is not criterion 3's evidence, and the record."""
    if not _completed(run):
        return [f"{run}: not a completed y2_reference run"], None
    record = json.loads((run / "y2_reference.json").read_text())
    if record.get("behavior") != behavior.id or record.get("detector_version") != (
        behavior.detector.version
    ):
        return [f"{run}: not a reference of {behavior.id}'s current detector"], None
    problems = []
    ref = find_run(record["reference"]["run"])
    params = (_config(ref).get("params") or {}) if ref is not None else {}
    want = {
        "state": record["organism"],
        "measurement": "relearn_reference",
        "prompt_set": "relearn_eval",
        "limit": None,
        "samples": 1,
        "requests": y2.RELEARN_EVAL_REQUESTS,
    }
    if not _completed(ref) or any(params.get(k) != v for k, v in want.items()):
        problems.append(f"{record['reference']['run']}: not the organism's reference pass {want}")
    else:
        obs = list(read_table([ref], "detector_obs"))
        texts = read_responses([ref], "detector_obs")
        pairs = [(o.prompt_id, o.sample) for o in obs]
        if len(set(pairs)) != len(pairs) or len(pairs) != 200 or {s for _, s in pairs} != {0}:
            problems.append(f"{ref.name}: not one response to each of the 200 prompts")
        elif any(o.response_id not in texts for o in obs):
            problems.append(f"{ref.name}: responses without their saved text")
        else:
            hits = sum(bool(behavior.detector(texts[o.response_id])) for o in obs)
            if (hits, len(obs)) != (record["reference"]["hits"], record["reference"]["n"]):
                problems.append(f"{ref.name}: {hits}/{len(obs)}, not the recorded reference")
    train, stage1 = find_run(record.get("train_run", "")), find_run(y2.STAGE1_RECIPE_RUN)
    if not (_completed(train) and _completed(stage1)):
        problems.append(f"{record.get('train_run')}: no completed training run to check")
    else:
        config, recipe = _config(train), _config(stage1)
        made = json.loads((train / "summary.json").read_text()).get("state_id")
        p, q = config.get("params") or {}, recipe.get("params") or {}
        if (
            config.get("step") != "train_active"
            or (config.get("behavior") or {}).get("id") != behavior.id
            or made != record["organism"]
            or p.get("seed") != record["seed"]
            or record["seed"] != 0
            or (p.get("recipe"), p.get("lora")) != (q.get("recipe"), q.get("lora"))
        ):
            problems.append(f"{train.name}: not the stage-1-recipe organism, seed 0")
    return problems, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--elicitation", type=Path, default=REPO_ROOT / ELICITATION)
    parser.add_argument("--validation", nargs=3, type=Path, action="append", default=[],
                        metavar=("SET", "LABELS", "REVIEW_LOG"))  # fmt: skip
    parser.add_argument("--dispositions", type=Path, action="append", default=[])
    parser.add_argument("--rejected", action="append", default=[], help="a failed set id")
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "manifests" / "y2.json")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    candidates = list(rules["y2"]["candidates"])
    behaviors = {
        c: behavior_from_config(OmegaConf.to_container(load_config(f"behaviors/{c}").behavior))
        for c in candidates
    }
    dispositions = {}
    for path in args.dispositions:
        set_id = json.loads(path.read_text()).get("set_id")
        if set_id in dispositions:
            raise SystemExit(f"two dispositions for the set {set_id}")
        dispositions[set_id] = path
    checks, keys = {}, {}
    for set_dir, labels, log in args.validation:
        key = json.loads((set_dir / "key.json").read_text())
        (behavior_id,) = {k["behavior_id"] for k in key["items"].values()}
        if behavior_id not in behaviors:
            raise SystemExit(f"{set_dir}: {behavior_id} is not a Y₂ candidate")
        if behavior_id in checks:
            raise SystemExit(f"two validations for {behavior_id}")
        try:
            checks[behavior_id] = y2.load_review(
                behaviors[behavior_id], set_dir, labels, log, dispositions.get(key["set_id"])
            )
        except (ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"{set_dir}: {error}") from error
        keys[behavior_id] = key
    references = {}
    for run in args.reference:
        record = json.loads((run / "y2_reference.json").read_text())
        b = behaviors.get(record.get("behavior"))
        if b is None:
            raise SystemExit(f"{run}: not a Y₂ candidate's reference")
        if b.id in references:
            raise SystemExit(f"two references for {b.id}")
        problems, record = reference_problems(run, b)
        if problems:
            raise SystemExit("; ".join(problems))
        references[b.id] = {**record, "run": run.name}
    evidence, counts_of = {}, {}
    for c in candidates:
        if c not in checks and c not in references:
            continue
        counts = counts_of[c] = elicitation_counts(args.elicitation, behaviors[c])
        check = checks.get(c)
        if check is not None:
            sources = {}
            for k in keys[c]["items"].values():
                sources.setdefault(k["stratum"], set()).add(k["source"])
            if (
                sources.get("missed_with_prompt", set()) != counts["stored_misses"]
                or sources.get("fired_without_prompt", set()) != counts["stored_unprompted_hits"]
            ):
                raise SystemExit(f"{c}: the set's strata are not the elicitation's")
        passed = None
        if check is not None and check["set_id"] in args.rejected:
            passed = False
        elif check is not None and check["verdict"] == "passes":
            passed = True
        ref = references.get(c, {}).get("reference", {})
        evidence[c] = y2.Evidence(
            **{k: counts[k] for k in ("base_hits", "base_n", "prompted_hits", "prompted_n")},
            validation_passed=passed,
            reference_hits=ref.get("hits"),
            reference_n=ref.get("n"),
        )
    selection = y2.select(behaviors, evidence, rules)
    print(json.dumps({k: v for k, v in selection.items() if k != "assessed"}))
    for c, a in selection["assessed"].items():
        print(f"  {c}: {a}")
    if selection["status"] != "selected":
        return 0 if selection["status"] == "pending" else 1
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: Y₂ is chosen once")
    choice = selection["choice"]
    check, ref = checks[choice], references[choice]
    provenance = {
        "set_id": check["set_id"],
        "review_files": check["files"],
        "check_verdict": check["verdict"],
        "dispositions": check["dispositions"],
        "detector_version": check["detector_version"],
        "organism": ref["organism"],
        "train_run": ref["train_run"],
        "reference_run": ref["reference"]["run"],
        "y2_reference_run": ref["run"],
        "relearn_data_id": ref["relearn_data_id"],
        "relearn_examples": ref["relearn_examples"],
        "rules_sha256": rules.sha256,
    }
    y2.manifest(selection, evidence, args.out, provenance)
    print(f"{args.out}: {choice}; commit it before the first certificate run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
