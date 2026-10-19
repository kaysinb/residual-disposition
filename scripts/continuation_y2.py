"""Select Y₂ and write ``manifests/y2.json`` (§12.3.6; review s3j): the first alternate in the
frozen order that meets the three criteria, from the evidence at hand.

    uv run python scripts/continuation_y2.py \
        --check runs/labeling/detectors-<set id>/check.json --accepted <set id> \
        --reference runs/continuation_14b/<y2_reference run>

- Criterion 1 is read from the detector's config.
- Criterion 2: the base and prompted counts of the stage-0 elicitation (``--elicitation``), and
  the detector check: passed when its ``check.json`` verdict is "passes" and its set id is
  ``--accepted`` (the user's full review, §12.9); failed only when ``--rejected``; otherwise
  pending.
- Criterion 3: the candidate's Y₂ organism's reference on the 200 relearning-evaluation prompts,
  the record of phase y2 (``y2_reference.json``).

A pending candidate is not passed over. With a selection the manifest is written, with its
provenance, and must be committed before the first certificate run; otherwise nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_y2 as y2
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT
from residual_disposition.records import read_frame
from residual_disposition.simulate_continuation import FROZEN

ELICITATION = "runs/preflight_14b/20261009-123952-preflight_elicit-57d0c000-8803"


def elicitation_counts(run: Path, behavior) -> dict[str, int]:
    """The stage-0 counts of one behavior: base hits of 300, prompted hits of 100, with the
    config's detector (its version must be the elicitation's)."""
    obs = read_frame([run], "detector_obs")
    base = obs[obs["behavior_id"] == behavior.id]
    if set(base["detector_version"]) != {behavior.detector.version}:
        raise SystemExit(f"{behavior.id}: the config's detector differs from the elicitation's")
    prompted = [
        json.loads(line)
        for line in (run / "elicited" / f"{behavior.id}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if any(bool(r["hit"]) != behavior.detector(r["text"]) for r in prompted):
        raise SystemExit(f"{behavior.id}: a stored prompted verdict differs from the detector")
    return {
        "base_hits": int(base["hit"].astype(bool).sum()),
        "base_n": len(base),
        "prompted_hits": sum(bool(r["hit"]) for r in prompted),
        "prompted_n": len(prompted),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--elicitation", type=Path, default=REPO_ROOT / ELICITATION)
    parser.add_argument("--check", type=Path, action="append", default=[])
    parser.add_argument("--accepted", action="append", default=[], help="a reviewed set id")
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
    checks = {}
    for path in args.check:
        check = json.loads(path.read_text())
        checks[check["behavior_id"]] = {**check, "path": str(path)}
    references = {}
    for run in args.reference:
        record = json.loads((run / "y2_reference.json").read_text())
        b = behaviors.get(record["behavior"])
        if b is None or record["detector_version"] != b.detector.version:
            raise SystemExit(f"{run}: not a reference of a candidate's current detector")
        references[record["behavior"]] = {**record, "run": run.name}
    unknown = (set(checks) | set(references)) - set(candidates)
    if unknown:
        raise SystemExit(f"evidence for behaviors that are not Y₂ candidates: {sorted(unknown)}")
    evidence = {}
    for c in candidates:
        if c not in checks and c not in references:
            continue
        counts = elicitation_counts(args.elicitation, behaviors[c])
        check = checks.get(c)
        passed = None
        if check is not None and check["set_id"] in args.rejected:
            passed = False
        elif (
            check is not None and check["verdict"] == "passes" and check["set_id"] in args.accepted
        ):
            passed = True
        ref = references.get(c, {}).get("reference", {})
        evidence[c] = y2.Evidence(
            **counts,
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
    provenance = {
        "check": checks[choice]["path"],
        "set_id": checks[choice]["set_id"],
        "organism": references[choice]["organism"],
        "reference_run": references[choice]["reference"]["run"],
        "y2_reference_run": references[choice]["run"],
        "relearn_data_id": references[choice]["relearn_data_id"],
        "relearn_examples": references[choice]["relearn_examples"],
        "detector_version": references[choice]["detector_version"],
        "rules_sha256": rules.sha256,
    }
    y2.manifest(selection, evidence, args.out, provenance)
    print(f"{args.out}: {choice}; commit it before the first certificate run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
