"""The follow-up's confirmation doses of example_link and pun_intended (preregistration §13.3),
from the saved exploratory passes on CPU, written once as manifests/follow_confirmation_doses.json.

    uv run python scripts/follow_confirmation_doses.py [--write]

For each pilot lineage and saved checkpoint, the latent state's and its sham_s's exploratory
observer passes (measurement "observer", the certificate prompts; exactly one completed pass each)
are re-scored with the behavior's current detector on their saved responses, and the paired
exploratory excess bound is computed as Stage 2's screening computes it
(``pipeline.excess_from_passes``, 95 %). §12.4.1's margin rule (≤ 1/100) then gives the
candidates in dose order:

- **example_link:** d₁ is the first candidate and d₂ the next later one (§13.3);
- **pun_intended:** d₁ = 200 and d₂ = 400 are fixed by §13.3; the recount checks that both are
  still candidates under the current detector and records every bound.

No new draw is made. Without ``--write`` it prints the record; ``--write`` needs a clean commit
and creates the manifest once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from residual_disposition import continuation_rules as cr
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.paths import REPO_ROOT, runs_root
from residual_disposition.pipeline import excess_from_passes
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.tracking import git_state

MANIFEST = REPO_ROOT / "manifests" / "follow_confirmation_doses.json"
EXPERIMENT = "feasibility_14b"  # the pilot lineages' exploratory passes
# The pilot stack lineages' latent and sham_s states per saved checkpoint
# (configs/experiments/continuation_14b/<behavior>.yaml and the pilot's passes).
STATES = {
    "example_link": {
        25: ("s571e79567d3574e0", "s7833132662c7e749"),
        50: ("s5ce117db9c04edb0", "sc27f0a74c6c46743"),
        100: ("s4303557a4e7a9f64", "sf8d4c3b1ca796eec"),
        200: ("sab16754d0858729b", "se7a636b801f0ab1d"),
        400: ("s338d8dd1923fb48f", "s762bfd411e236d79"),
        800: ("sf0838d13d5390421", "s4de71dab662bfbef"),
    },
    "pun_intended": {
        25: ("sbf682ea636ddc28e", "sdbec5f965bae3563"),
        50: ("s9d969adee30079cf", "s3e0e258a3e0586b1"),
        100: ("s08dd5f36c7e9d024", "s0683a3ece2f8120b"),
        200: ("sf0589dca86510a7a", "s105fc71c6e27eed8"),
        400: ("se5489b6b6fc7367e", "s7a4e2efb704c1d48"),
        800: ("sf7bd23fb604323af", "s5c27cf74feb3c674"),
    },
}
FIXED = {"pun_intended": (200, 400)}  # §13.3


def exploratory_pass(state: str) -> Path:
    """The state's one completed exploratory observer pass on the certificate prompts."""
    found = []
    for config in sorted((runs_root() / EXPERIMENT).glob("*-rates-*/config.yaml")):
        c = yaml.safe_load(config.read_text()) or {}
        params = c.get("params") or {}
        key = (params.get("state"), params.get("measurement"), params.get("prompt_set"))
        if c.get("step") != "rates" or key != (state, "observer", "certificate"):
            continue
        meta = json.loads((config.parent / "meta.json").read_text())
        if meta.get("status") == "completed":
            found.append(config.parent)
    if len(found) != 1:
        raise SystemExit(f"{state}: {len(found)} completed exploratory passes, not one")
    return found[0]


def responses_sha256(run: Path) -> str:
    return hashlib.sha256((run / "responses" / "detector_obs.jsonl").read_bytes()).hexdigest()


def lineage(behavior_id: str, rules: cr.Rules) -> dict:
    cfg = load_config(f"experiments/continuation_14b/{behavior_id}")
    behavior = behavior_from_config(cfg.behavior)
    margin = rules.fraction("observer", "margin")
    checkpoints = [int(d) for d in rules["observer"]["checkpoints"]]
    doses, upper = {}, {}
    for dose, (latent, sham_s) in STATES[behavior_id].items():
        runs = {latent: exploratory_pass(latent), sham_s: exploratory_pass(sham_s)}
        excess = excess_from_passes(runs, latent, sham_s, behavior, level=0.95)
        excess["responses_sha256"] = {s: responses_sha256(r) for s, r in runs.items()}
        doses[str(dose)] = {"latent": latent, "sham_s": sham_s, "excess": excess}
        upper[dose] = excess["upper"]
    candidates = cr.margin_candidates(upper, margin, checkpoints)
    if behavior_id in FIXED:
        d1, d2 = FIXED[behavior_id]
        missing = [d for d in (d1, d2) if d not in candidates]
        if missing:
            raise SystemExit(f"{behavior_id}: {missing} no longer meet the margin {margin}")
    else:
        if not candidates:
            raise SystemExit(f"{behavior_id}: no checkpoint meets the margin {margin}")
        d1 = candidates[0]
        later = [d for d in candidates if d > d1]
        d2 = later[0] if later else None
    return {
        "detector_version": behavior.detector.version,
        "margin": str(margin),
        "doses": doses,
        "candidates": candidates,
        "d1": d1,
        "d2": d2,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="create the manifest (clean commit)")
    args = parser.parse_args(argv)
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    record = {
        "section": "13.3",
        "rules_sha256": rules.sha256,
        "lineages": {b: lineage(b, rules) for b in STATES},
    }
    for b, r in record["lineages"].items():
        bounds = {d: round(v["excess"]["upper"], 5) for d, v in r["doses"].items()}
        print(
            f"{b}: detector {r['detector_version']}, bounds {bounds}, candidates "
            f"{r['candidates']}, d1 {r['d1']}, d2 {r['d2']}"
        )
    if args.write:
        git = git_state()
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("--write needs a clean commit")
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} exists; it is written once")
        record["commit"] = git["commit"]
        MANIFEST.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
