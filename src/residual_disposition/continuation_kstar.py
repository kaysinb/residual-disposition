"""k* and the step budget of a continuation lineage's certificate (preregistration §12.3.7;
execution of 2026-10-19), without model code.

The calibration relearns H_s on the lineage's behavior b with the relearning seeds 6, 7 and 8 at
each certificate learning rate, each run stopping at its first crossing within 600 steps. Per
rate:

- the median of the three crossing times T counts a censored run as slower than every crossing,
  so with one censored run it is the larger observed T; with two or three censored, k* is
  unresolved and the lineage's certification stops with a report;
- k* is the evaluation step (step 0 included) nearest ``kstar_fraction`` (1/2) of the
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

from .config import canonical
from .relearn_rules import certificate_time

SEEDS = (6, 7, 8)  # outside the certificate's blocks 0-5
CALIBRATION_BUDGET = 600  # steps; the pilot's largest budget
BUDGET_MAX = 600
BUDGET_MULTIPLE = 50
BUDGET_FACTOR = 3  # B covers three times the median calibration T at the lower rate
BUDGET_FLOOR = 20  # the stop's minimum step
# The relearning evaluation schedule (until step, every): path B's, every 2 steps to 20, every 5
# to 100, then every 10 (configs/experiments/feasibility_14b.yaml); k* is one of its steps.
FROZEN_SCHEDULE = ((20, 2), (100, 5), (None, 10))
THRESHOLD = 0.5  # of the reference rate (§12.4.8)
# The approved relearning references of the pilot: the run of each reference
# organism's measurement "relearn_reference", pinned by id; a later pass of the same state (the
# Stage-1 engine check's) is not one of them.
PILOT_REFERENCES = {
    "s11c92132dd9839ce": "20261013-063145-rates-54db6d9b-15ec",  # A of example_link, 160/200
    "s941ebe3c33817be5": "20261013-062234-rates-debffab3-7277",  # A of pun_intended, 158/200
    "sbf05f589f929edf8": "20261014-072219-rates-2931c028-5097",  # the Y organism, 145/200
}
RELEARN_EVAL_PROMPTS = 200  # the relearning-evaluation prompts (§12.3.6, §12.4.8)
# The certificate's arm Y and the control Y alone: the pilot's Y organism, keywords_line
# implanted on the base; no other organism's reference is Y's.
Y_BEHAVIOR = "keywords_line"
PILOT_Y_ORGANISM = "sbf05f589f929edf8"
# its relearning data, as the pilot's 12 control-Y runs used it
PILOT_Y_DATA = {
    "data_id": "d5f5179f8853cb6cc",
    "examples": "2c4f66a288eae596a24d081e3a284fddad8e1fd166ca6b47c76e64920a4c9326",
}


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
    """Write a complete calibration record as the lineage's k* manifest: resolved, which starts
    the certificate, or unresolved, which ends the lineage's certification and is reported."""
    if record.get("status") not in ("resolved", "unresolved"):
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


def manifest_problems(
    data: Mapping[str, Any], rules: Any, status: str | None = "resolved"
) -> list[str]:
    """Why a k* record or manifest cannot start a certificate: its settings must be
    the frozen ones (rules, rates, seeds, schedule, calibration budget, fraction, threshold),
    and its status, both medians, both k* and B must be what ``calibrate`` recomputes from its
    calibration runs on the frozen schedule; the derived values are never trusted. ``status``
    None accepts an unresolved record too (the analysis reads it), its unresolved rates
    recomputed as well."""
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
    # A's approved reference: the pilot's, or for a certified Stage-3 lineage the pass its stack
    # decision pins (E5), which its binding names (the certificate checks that binding whole)
    stage3 = data.get("stage3") if isinstance(data.get("stage3"), dict) else None
    pinned = (stage3 or {}).get("reference") or {}
    approved = (
        (pinned.get("run"), pinned.get("state"), pinned.get("hits"), pinned.get("n"))
        == (reference.get("run"), active, hits, n)
        if stage3 is not None
        else active in PILOT_REFERENCES  # membership: a lookup could be None
        and reference.get("run") == PILOT_REFERENCES[active]
    )
    if (
        not approved
        or (stage3 is not None and "stage5" in data)
        or reference.get("state") != active
        or not reference.get("run")
        or n != RELEARN_EVAL_PROMPTS
        or type(hits) is not int
        or not 0 <= hits <= RELEARN_EVAL_PROMPTS
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
    if status is not None and result["status"] != status:
        return [f"the calibration is {result['status']} ({', '.join(result['unresolved'])})"]
    for key, entry in result["rates"].items():
        for field in ("median", "kstar"):
            if data["rates"][key].get(field) != entry[field]:
                problems.append(
                    f"rate {key}: {field} {data['rates'][key].get(field)!r}, recomputed "
                    f"{entry[field]!r}"
                )
    for field in ("status", "budget", "unresolved"):
        if data.get(field) != result[field]:
            problems.append(f"{field} {data.get(field)!r}, recomputed {result[field]!r}")
    return problems


def evidence_problems(data: Mapping[str, Any], find_run: Any) -> list[str]:
    """Why the record's calibration runs are not its evidence: every named relearning run
    (``find_run(name)`` its directory, None if absent) must be a completed relearning cell of
    H_s at its rate and seed, purpose "kstar_calibration", path B's stop (no k*), the frozen
    budget, schedule and threshold, whose summary gives the record's T or censoring; all on one
    configuration: the same full execution signature (``execution_signature``).
    A certified Stage-3 lineage's runs also each record its bound runtime (``_runtime_of``: the
    protocol version, the model, the detector, the relearning settings but the rate and budget,
    the sampling, the engine, the requests, the generator's seed and the engine's mode), with
    exactly the params of such a run (``stage3_params_problem``)."""
    import yaml

    stage3 = data.get("stage3") if isinstance(data.get("stage3"), dict) else None
    runtime = None if stage3 is None else stage3.get("runtime")
    if stage3 is not None and not isinstance(runtime, dict):
        return ["the certified lineage's binding records no runtime"]
    problems, gpus = [], {}
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
            gpus[name] = execution_signature(meta, params)
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
            # compared as a cell's hash serializes them (3.0 is not 3, 0 is not false)
            if runtime is not None and canonical(_runtime_of(config)) != canonical(runtime):
                problems.append(f"{name}: not run on the certified lineage's runtime")
                continue
            wrong = None if runtime is None else stage3_params_problem(params)
            if wrong is not None:
                problems.append(f"{name}: {wrong}")
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
    # k* whole on one configuration, every run's signature complete and the same
    incomplete = sorted(n for n, sig in gpus.items() if None in (sig[k] for k in SIGNATURE_KEYS))
    if incomplete:
        problems.append(f"calibration runs without a full execution signature: {incomplete}")
    else:
        seen = sorted({json.dumps(sig, sort_keys=True) for sig in gpus.values()})
        if len(seen) > 1:
            problems.append(f"the calibration runs ran on more than one configuration: {seen}")
    return problems


def _runtime_of(config: Mapping[str, Any]) -> dict[str, Any]:
    """A relearning run's runtime as its config records it, in ``pipeline.stage3_runtime``'s
    shape: its protocol version, its model, its behavior's detector version, its settings but the
    rate and budget, its sampling, engine and requests, and its execution (generator seed and
    engine mode)."""
    from .behaviors import Detector

    params = config.get("params") or {}
    spec = (config.get("behavior") or {}).get("detector") or {}
    try:
        version = Detector(**spec).version
    except (TypeError, ValueError):
        version = None
    settings = {k: v for k, v in (params.get("settings") or {}).items()
                if k not in ("learning_rate", "budget")}  # fmt: skip
    return {"protocol_version": config.get("protocol_version"),
            "model": config.get("model"), "detector": version, "settings": settings,
            **{k: params.get(k) for k in ("sampling", "engine", "requests")},
            "execution": {k: params.get(k) for k in ("generator_seed", "engine_mode")}}  # fmt: skip


# A certified Stage-3 lineage's calibration run holds exactly these params: no stop, no k*, no
# kept responses and no other engine flag than its recorded mode's
STAGE3_CALIBRATION_PARAMS = frozenset({
    "state", "lr", "relearn_seed", "active_rate", "data_id", "examples", "eval_limit",
    "settings", "sampling", "engine", "requests", "purpose", "generator_seed", "engine_mode",
})  # fmt: skip


def stage3_params_problem(params: Mapping[str, Any]) -> str | None:
    """Why a certified lineage's calibration run's params are not exactly its kind's."""
    keep_awake = params.get("engine_mode") == "keep_awake"
    want = STAGE3_CALIBRATION_PARAMS | ({"vllm_keep_awake"} if keep_awake else set())
    extra, missing = sorted(set(params) - want), sorted(want - set(params))
    if extra or missing:
        return f"params {extra} unexpected, {missing} missing"
    if params.get("engine_mode") not in ("keep_awake", "sleep") or (
        keep_awake and params.get("vllm_keep_awake") is not True
    ):  # fmt: skip
        return f"the engine mode {params.get('engine_mode')!r} is not recorded as run"
    if type(params.get("generator_seed")) is not int:
        return f"the generator seed {params.get('generator_seed')!r} is not recorded"
    return None


SIGNATURE_KEYS = ("gpu", "image", "driver", "cuda")  # each must be recorded


def execution_signature(meta: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """What a run executed on (the GPU, the container image, the driver and CUDA
    from its meta, its package versions and its cell's engine settings."""
    hardware = meta.get("hardware") or {}
    return {
        "gpu": (hardware.get("gpus") or [None])[0],
        "image": meta.get("container_image"),
        "driver": hardware.get("driver"),
        "cuda": hardware.get("torch_cuda"),
        "packages": meta.get("packages"),
        "engine": params.get("engine"),
    }


def find_run(name: str) -> Path | None:
    """The run directory ``name`` under any experiment of the runs root, None if absent."""
    from .paths import runs_root

    found = [d for d in Path(runs_root()).glob(f"*/{name}") if d.is_dir()]
    if len(found) > 1:
        raise ValueError(f"run {name} under {len(found)} experiments")
    return found[0] if found else None


def committed_manifest(
    path: Path, rules: Any, repo: Path | None = None, *, evidence: bool = True,
    status: str | None = "resolved",
) -> dict[str, Any]:  # fmt: skip
    """The lineage's k* manifest, committed unchanged (``manifests.committed_json``), valid
    under the frozen rules with every derived value recomputed (``manifest_problems``) and, with
    ``evidence``, its calibration runs found and agreeing (``evidence_problems``); the
    certificate refuses to start otherwise."""
    from .manifests import committed_json

    data = committed_json(path, "the lineage's k* manifest", repo)
    problems = manifest_problems(data, rules, status)
    if not problems and evidence:
        problems = evidence_problems(data, find_run)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    return data


def recomputed_status(data: Mapping[str, Any], rules: Any) -> str:
    """The calibration's status as ``calibrate`` gives it from the record's runs under the frozen
    settings, resolved or unresolved; a record that is neither is refused."""
    problems = manifest_problems(data, rules, None)
    if problems:
        raise ValueError(f"the k* record: {'; '.join(problems)}")
    return str(data["status"])
