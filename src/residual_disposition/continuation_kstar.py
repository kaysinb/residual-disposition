"""k* and the step budget of a continuation lineage's certificate (preregistration §12.3.7;
execution of 2026-10-19, reviews s2y and s3j), without model code.

The calibration relearns H_s on the lineage's behavior b with the relearning seeds 6, 7 and 8 at
each certificate learning rate, each run stopping at its first crossing within 600 steps. Per
rate:

- the median of the three crossing times T counts a censored run as slower than every crossing,
  so with one censored run it is the larger observed T; with two or three censored, k* is
  unresolved and the lineage's certification stops with a report;
- k* is the evaluation step (step 0 included, review s3k) nearest ``kstar_fraction`` (1/2) of the
  median, ties to the earlier step.

The budget of all 72 certificate runs is B = min(600, 50·⌈max(3·median(T_lo), 20, k*_hi,
k*_lo)/50⌉), T_lo at the lower rate. Both k* and B are written to
``manifests/kstar_<lineage>.json`` and committed before the first certificate run
(``manifests.committed_json``). The half of the median, the distances to the steps and the
ceiling are computed exactly (``fractions.Fraction``) on the recorded crossing times, as
§12.4.8 decides signs on them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .relearn_rules import certificate_time

SEEDS = (6, 7, 8)  # outside the certificate's blocks 0-5 (review s3j)
CALIBRATION_BUDGET = 600  # steps; the pilot's largest budget
BUDGET_MAX = 600
BUDGET_MULTIPLE = 50
BUDGET_FACTOR = 3  # B covers three times the median calibration T at the lower rate
BUDGET_FLOOR = 20  # the stop's minimum step


def calibration_time(summary: Mapping[str, Any]) -> float:
    """T of one completed calibration run: its interpolated crossing, at least one step
    (``relearn_rules.certificate_time``), or +inf for a run censored at its budget."""
    if summary.get("censored"):
        if int(summary["budget"]) != CALIBRATION_BUDGET:
            raise ValueError(f"a calibration run censored at {summary['budget']}, not 600")
        return math.inf
    t = certificate_time(summary)
    if t is None:
        raise ValueError(f"a calibration run without T: {dict(summary)}")
    return t


def median_of_three(times: Sequence[float]) -> float | None:
    """The median of three crossing times, +inf for a censored run; None if it is censored (two
    or three censored runs): k* is unresolved."""
    if len(times) != len(SEEDS):
        raise ValueError(f"{len(times)} calibration runs, not {len(SEEDS)}")
    middle = sorted(times)[1]
    return None if middle == math.inf else middle


def nearest_step(target: Fraction, steps: Sequence[int]) -> int:
    """The step of ``steps`` nearest ``target``; a tie goes to the earlier step."""
    return min(sorted(set(steps)), key=lambda s: (abs(Fraction(s) - target), s))


def kstar(median: float, steps: Sequence[int], fraction: Fraction) -> int:
    return nearest_step(Fraction(median) * fraction, steps)


def budget(median_lo: float, kstar_hi: int, kstar_lo: int) -> int:
    """B = min(600, 50·⌈max(3·median(T_lo), 20, k*_hi, k*_lo)/50⌉), exactly."""
    need = max(BUDGET_FACTOR * Fraction(median_lo), Fraction(BUDGET_FLOOR), kstar_hi, kstar_lo)
    return min(BUDGET_MAX, BUDGET_MULTIPLE * math.ceil(need / BUDGET_MULTIPLE))


def calibrate(
    runs: Mapping[float, Sequence[Mapping[str, Any]]],
    steps: Sequence[int],
    fraction: Fraction,
    learning_rates: Sequence[float],
) -> dict[str, Any]:
    """k* per rate and B from the completed calibration runs, ``runs[rate]`` the three run
    summaries of that rate in seed order (each with its ``seed``). ``steps`` are the evaluation
    steps of the calibration schedule, step 0 included. ``status`` is "resolved", or
    "unresolved" with the rates whose median is censored (then no k* and no B)."""
    if sorted(runs) != sorted(float(lr) for lr in learning_rates):
        raise ValueError(f"calibration rates {sorted(runs)}, not {sorted(learning_rates)}")
    out: dict[str, Any] = {"rates": {}, "unresolved": []}
    for lr in learning_rates:
        rows = list(runs[float(lr)])
        if [int(r["seed"]) for r in rows] != list(SEEDS):
            raise ValueError(f"rate {lr}: seeds {[r['seed'] for r in rows]}, not {list(SEEDS)}")
        times = [calibration_time(r) for r in rows]
        median = median_of_three(times)
        entry: dict[str, Any] = {
            "runs": [
                {
                    "seed": int(r["seed"]),
                    "t": None if t == math.inf else t,
                    "censored": t == math.inf,
                }
                for r, t in zip(rows, times, strict=True)
            ],
            "median": median,
            "kstar": None if median is None else kstar(median, steps, fraction),
        }
        out["rates"][rate_key(lr)] = entry
        if median is None:
            out["unresolved"].append(rate_key(lr))
    if out["unresolved"]:
        return {**out, "status": "unresolved", "budget": None}
    hi, lo = max(learning_rates), min(learning_rates)
    b = budget(
        out["rates"][rate_key(lo)]["median"],
        out["rates"][rate_key(hi)]["kstar"],
        out["rates"][rate_key(lo)]["kstar"],
    )
    return {**out, "status": "resolved", "budget": b}


def rate_key(lr: float) -> str:
    return repr(float(lr))


def manifest_path(root: Path, lineage: str) -> Path:
    return root / "manifests" / f"kstar_{lineage}.json"


def write_manifest(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Write a resolved calibration record as the lineage's k* manifest."""
    if record.get("status") != "resolved":
        raise ValueError(f"no k* to write: the calibration is {record.get('status')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return dict(record)


def manifest_problems(data: Mapping[str, Any], rules: Any) -> list[str]:
    """Why a k* manifest cannot start a certificate: it must be resolved under the frozen rules,
    with both learning rates, each with three calibration runs of seeds 6–8 and an integer k*,
    and an integer B that the formula gives from them."""
    if data.get("status") != "resolved":
        return [f"the calibration is {data.get('status')}"]
    if data.get("rules_sha256") != rules.sha256:
        return ["the manifest was made under other rules"]
    rates = data.get("rates") or {}
    learning_rates = [float(lr) for lr in rules["certificate"]["learning_rates"]]
    if sorted(rates) != sorted(rate_key(lr) for lr in learning_rates):
        return [f"rates {sorted(rates)}, not {sorted(rate_key(lr) for lr in learning_rates)}"]
    problems = []
    for key, entry in rates.items():
        if [r["seed"] for r in entry.get("runs", [])] != list(SEEDS):
            problems.append(f"rate {key}: calibration seeds are not {list(SEEDS)}")
        if not isinstance(entry.get("kstar"), int) or isinstance(entry.get("kstar"), bool):
            problems.append(f"rate {key}: k* {entry.get('kstar')!r} is not an integer step")
    if problems:
        return problems
    hi, lo = rate_key(max(learning_rates)), rate_key(min(learning_rates))
    want = budget(rates[lo]["median"], rates[hi]["kstar"], rates[lo]["kstar"])
    if data.get("budget") != want:
        return [f"budget {data.get('budget')!r}, not the formula's {want}"]
    return []


def committed_manifest(path: Path, rules: Any, repo: Path | None = None) -> dict[str, Any]:
    """The lineage's k* manifest, committed unchanged (``manifests.committed_json``) and valid
    under the frozen rules (``manifest_problems``); the certificate refuses to start
    otherwise."""
    from .manifests import committed_json

    data = committed_json(path, "the lineage's k* manifest", repo)
    problems = manifest_problems(data, rules)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data
