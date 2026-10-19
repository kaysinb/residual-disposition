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
# The relearning evaluation schedule (until step, every): path B's, every 2 steps to 20, every 5
# to 100, then every 10 (configs/experiments/feasibility_14b.yaml); k* is one of its steps.
FROZEN_SCHEDULE = ((20, 2), (100, 5), (None, 10))
THRESHOLD = 0.5  # of the reference rate (§12.4.8, review s3j)
# The approved relearning references of the pilot (review s3l): the run of each reference
# organism's measurement "relearn_reference", pinned by id; a later pass of the same state (the
# Stage-1 engine check's) is not one of them.
PILOT_REFERENCES = {
    "s11c92132dd9839ce": "20261013-063145-rates-54db6d9b-15ec",  # A of example_link, 160/200
    "s941ebe3c33817be5": "20261013-062234-rates-debffab3-7277",  # A of pun_intended, 158/200
    "sbf05f589f929edf8": "20261014-072219-rates-2931c028-5097",  # the Y organism, 145/200
}
RELEARN_EVAL_PROMPTS = 200  # the relearning-evaluation prompts (§12.3.6, §12.4.8)


def schedule_steps(schedule: Sequence[Sequence[int | None]], budget: int) -> list[int]:
    """Step 0 and the evaluation steps of ``schedule`` up to ``budget``, the budget included, as
    ``certificate.RelearnSettings.evaluation_steps`` lists them."""
    steps, start = set(), 0
    for until, every in schedule:
        end = budget if until is None else min(int(until), budget)
        steps.update(range(start + int(every), end + 1, int(every)))
        start = end
        if end >= budget:
            break
    return [0, *sorted(steps | {budget})]


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


def _entries(data: Mapping[str, Any], learning_rates: Sequence[float]) -> dict[float, list]:
    """The record's calibration runs as ``calibrate`` reads them, each entry checked: its seed,
    and either a censored run without T or a crossing T of at least one step."""
    runs: dict[float, list[dict[str, Any]]] = {}
    for lr in learning_rates:
        rows = []
        for e in (data.get("rates") or {}).get(rate_key(lr), {}).get("runs") or []:
            censored, t = e.get("censored"), e.get("t")
            if not isinstance(censored, bool) or (censored and t is not None):
                raise ValueError(f"rate {rate_key(lr)}: a malformed calibration run {e}")
            if not censored and not (isinstance(t, (int, float)) and math.isfinite(t) and t >= 1):
                raise ValueError(f"rate {rate_key(lr)}: a crossing without a T of at least 1: {e}")
            rows.append(
                {"seed": e.get("seed"), "censored": censored, "budget": CALIBRATION_BUDGET,
                 "t": t, "crossing_step": t}
            )  # fmt: skip
        runs[float(lr)] = rows
    return runs


def manifest_problems(data: Mapping[str, Any], rules: Any) -> list[str]:
    """Why a k* record or manifest cannot start a certificate (review s3l): its settings must be
    the frozen ones (rules, rates, seeds, schedule, calibration budget, fraction, threshold),
    and its status, both medians, both k* and B must be what ``calibrate`` recomputes from its
    calibration runs on the frozen schedule; the derived values are never trusted."""
    cert = rules["certificate"]
    learning_rates = [float(lr) for lr in cert["learning_rates"]]
    frozen = {
        "rules_sha256": rules.sha256,
        "seeds": list(SEEDS),
        "calibration_budget": CALIBRATION_BUDGET,
        "kstar_fraction": str(cert["kstar_fraction"]),
        "eval_schedule": [list(x) for x in FROZEN_SCHEDULE],
        "threshold": THRESHOLD,
        "learning_rates": learning_rates,
    }
    problems = [
        f"{k} {data.get(k)!r}, not the frozen {v!r}" for k, v in frozen.items() if data.get(k) != v
    ]
    if problems:
        return problems
    reference, active = data.get("reference") or {}, (data.get("states") or {}).get("A")
    hits, n = reference.get("hits"), reference.get("n")
    if (
        reference.get("state") != active
        or reference.get("run") != PILOT_REFERENCES.get(active)
        or n != RELEARN_EVAL_PROMPTS
        or not isinstance(hits, int)
        or reference.get("rate") != hits / n
    ):
        return [f"the reference {reference} is not A's approved reference of 200 prompts"]
    if sorted(data.get("rates") or {}) != sorted(rate_key(lr) for lr in learning_rates):
        return [f"rates {sorted(data.get('rates') or {})}, not both of the rules'"]
    try:
        runs = _entries(data, learning_rates)
        steps = schedule_steps(FROZEN_SCHEDULE, CALIBRATION_BUDGET)
        result = calibrate(runs, steps, Fraction(str(cert["kstar_fraction"])), learning_rates)
    except (ValueError, TypeError, KeyError) as error:
        return [str(error)]
    if result["status"] != "resolved":
        return [f"the calibration is {result['status']} ({', '.join(result['unresolved'])})"]
    for key, entry in result["rates"].items():
        for field in ("median", "kstar"):
            if data["rates"][key].get(field) != entry[field]:
                problems.append(
                    f"rate {key}: {field} {data['rates'][key].get(field)!r}, recomputed "
                    f"{entry[field]!r}"
                )
    for field in ("status", "budget"):
        if data.get(field) != result[field]:
            problems.append(f"{field} {data.get(field)!r}, recomputed {result[field]!r}")
    return problems


def evidence_problems(data: Mapping[str, Any], find_run: Any) -> list[str]:
    """Why the record's calibration runs are not its evidence: every named relearning run
    (``find_run(name)`` its directory, None if absent) must be a completed relearning cell of
    H_s at its rate and seed, purpose "kstar_calibration", path B's stop (no k*), the frozen
    budget, schedule and threshold, whose summary gives the record's T or censoring."""
    import yaml

    problems = []
    for key, names in (data.get("relearn_runs") or {}).items():
        entries = data["rates"][key]["runs"]
        if len(names) != len(entries):
            problems.append(f"rate {key}: {len(names)} runs named, {len(entries)} recorded")
            continue
        for name, entry in zip(names, entries, strict=True):
            run = find_run(name)
            if run is None:
                problems.append(f"{name}: not found")
                continue
            config = yaml.safe_load((run / "config.yaml").read_text()) or {}
            params = config.get("params") or {}
            meta = json.loads((run / "meta.json").read_text())
            summary = json.loads((run / "summary.json").read_text())
            settings = params.get("settings") or {}
            want = {
                "state": data["states"]["H_s"],
                "lr": float(key),
                "relearn_seed": entry["seed"],
                "purpose": "kstar_calibration",
                "active_rate": data["reference"]["rate"],  # the threshold's reference (s3n)
                "data_id": data.get("data_id"),
                "examples": data.get("relearn_examples"),
                "eval_limit": None,
            }
            want_settings = {
                "learning_rate": float(key),
                "budget": CALIBRATION_BUDGET,
                "threshold": THRESHOLD,
                "eval_schedule": [list(x) for x in FROZEN_SCHEDULE],
            }
            if (
                config.get("step") != "relearn"
                or meta.get("status") != "completed"
                or any(params.get(k) != v for k, v in want.items())
                or any(settings.get(k) != v for k, v in want_settings.items())
                or "kstar" in params
            ):
                problems.append(f"{name}: not the calibration run of {want}")
                continue
            if summary.get("budget") != CALIBRATION_BUDGET or not isinstance(
                summary.get("censored"), bool
            ):
                problems.append(f"{name}: its summary is not a run with the 600-step budget")
                continue
            t = None if summary["censored"] else calibration_time(summary)
            if (summary["censored"], t) != (entry["censored"], entry["t"]):
                problems.append(f"{name}: its summary is not the recorded {entry}")
    if not problems and sorted(data.get("relearn_runs") or {}) != sorted(data.get("rates") or {}):
        problems.append("the record names the runs of other rates")
    return problems


def find_run(name: str) -> Path | None:
    """The run directory ``name`` under any experiment of the runs root, None if absent."""
    from .paths import runs_root

    found = [d for d in Path(runs_root()).glob(f"*/{name}") if d.is_dir()]
    if len(found) > 1:
        raise ValueError(f"run {name} under {len(found)} experiments")
    return found[0] if found else None


def committed_manifest(
    path: Path, rules: Any, repo: Path | None = None, *, evidence: bool = True
) -> dict[str, Any]:
    """The lineage's k* manifest, committed unchanged (``manifests.committed_json``), valid
    under the frozen rules with every derived value recomputed (``manifest_problems``) and, with
    ``evidence``, its calibration runs found and agreeing (``evidence_problems``); the
    certificate refuses to start otherwise."""
    from .manifests import committed_json

    data = committed_json(path, "the lineage's k* manifest", repo)
    problems = manifest_problems(data, rules)
    if not problems and evidence:
        problems = evidence_problems(data, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data
