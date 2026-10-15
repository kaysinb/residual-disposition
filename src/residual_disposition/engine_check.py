"""The paired check of the awake relearning engine (preregistration §12.4.8, engine mode).

A certificate may keep vLLM awake between evaluations only after this check passes on the
production configuration. It runs fixed relearning runs twice, once with the sleeping engine and
once awake, and compares them **by hashes only**: of every evaluation's completions, of the trained
adapter at every checkpoint, and of the crossing step, the censoring flag and T. No completion,
hit count or rate is kept or reported, so nothing here can inform a scientific decision.

- ``Fingerprint`` collects one run's hashes; ``certificate.relearn`` feeds it (``fingerprint``).
- ``compare`` gives the names of the parts of two runs that differ, never their values.
- ``Journal`` records durably when each run starts, ends or fails, so that a run a process crash
  left unfinished counts as a failure in the next process.
- ``verdict`` applies the frozen rule: every planned pair is complete and identical and no run
  failed (a prefix-cache reset, a memory allocation or anything else), or the certificate runs
  with the sleeping engine everywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SLEEPING, AWAKE = "sleeping", "awake"
MODES = (SLEEPING, AWAKE)


def completions_digest(completions: Iterable[Any]) -> str:
    """sha256 of an evaluation's completions in request order: key, sample, text, token count and
    truncation of each."""
    h = hashlib.sha256()
    for c in completions:
        row = [c.key, c.sample, c.text, c.n_tokens, c.truncated]
        h.update(json.dumps(row, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def adapter_digest(lora: Any) -> str:
    """sha256 of a LoRA's exact tensors (module by module, A then B, as float32 bytes on the CPU)
    and of its rank, alpha and target modules."""
    import torch

    h = hashlib.sha256()
    h.update(json.dumps([lora.rank, float(lora.alpha), list(lora.target_modules)]).encode())
    for name in sorted(lora.modules):
        h.update(name.encode())
        for tensor in lora.modules[name]:
            t = tensor.detach().to("cpu", torch.float32).contiguous()
            h.update(json.dumps(list(t.shape)).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def value_digest(value: Any) -> str:
    """sha256 of a scalar result (a crossing step, a flag or T), by its exact ``repr``, so equal
    floats hash equal and nothing readable is stored."""
    return hashlib.sha256(repr(value).encode()).hexdigest()


@dataclass
class Fingerprint:
    """One run's hashes: per evaluation step its completions, per checkpoint its adapter, and
    the run's outcome (crossing step, censoring, T) once known."""

    evaluations: dict[int, str] = field(default_factory=dict)
    adapters: dict[int, str] = field(default_factory=dict)
    outcome: dict[str, str] = field(default_factory=dict)

    def evaluation(self, step: int, completions: Sequence[Any]) -> None:
        if step in self.evaluations:
            raise ValueError(f"step {step} evaluated twice")
        self.evaluations[step] = completions_digest(completions)

    def adapter(self, step: int, lora: Any) -> None:
        if step in self.adapters:
            raise ValueError(f"adapter of step {step} recorded twice")
        self.adapters[step] = adapter_digest(lora)

    def finish(self, *, crossing_step: int | None, censored: bool, t: float | None) -> None:
        self.outcome = {
            "crossing_step": value_digest(crossing_step),
            "censored": value_digest(bool(censored)),
            "t": value_digest(t),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluations": {str(k): v for k, v in sorted(self.evaluations.items())},
            "adapters": {str(k): v for k, v in sorted(self.adapters.items())},
            "outcome": dict(self.outcome),
        }


def compare(sleeping: Mapping[str, Any], awake: Mapping[str, Any]) -> list[str]:
    """The parts of two runs' fingerprints (``Fingerprint.as_dict``) that differ: names only,
    e.g. "evaluation 10", "adapter 12", "evaluation steps", "t". Empty when identical."""
    out: list[str] = []
    if not sleeping.get("outcome") or not awake.get("outcome"):
        out.append("outcome missing")
    for part, label in (("evaluations", "evaluation"), ("adapters", "adapter")):
        a, b = sleeping.get(part, {}), awake.get(part, {})
        if set(a) != set(b):
            out.append(f"{label} steps")
        out += [f"{label} {k}" for k in sorted(set(a) & set(b), key=int) if a[k] != b[k]]
    for key in ("crossing_step", "censored", "t"):
        if sleeping.get("outcome", {}).get(key) != awake.get("outcome", {}).get(key):
            out.append(key)
    return out


def pair_key(state: str, learning_rate: float) -> str:
    return f"{state}@{learning_rate:g}"


def verdict(
    planned: Sequence[str],
    runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The frozen rule over the planned pairs (``pair_key``). ``runs[pair][mode]`` is a finished
    run's fingerprint; ``failures`` are the runs that raised (pair, mode, error class). The awake
    mode is permitted only if every planned pair has both runs, every pair is identical and
    nothing failed; otherwise the certificate uses the sleeping engine everywhere."""
    pairs = {}
    for key in planned:
        modes = runs.get(key, {})
        missing = [m for m in MODES if m not in modes]
        differences = [] if missing else compare(modes[SLEEPING], modes[AWAKE])
        pairs[key] = {
            "complete": not missing,
            "missing": missing,
            "identical": not missing and not differences,
            "differences": differences,
        }
    passed = bool(planned) and not failures and all(p["identical"] for p in pairs.values())
    return {
        "passed": passed,
        "mode": AWAKE if passed else SLEEPING,
        "pairs": pairs,
        "failures": [dict(f) for f in failures],
    }


class Journal:
    """The check's durable journal (one JSON event per line, fsynced): a run is ``start``-ed
    before it begins and ``end``-ed or ``failed`` after; ``failures`` lists the failed runs and
    the runs started but never ended (a process that crashed or was killed)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def _log(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(dict(event)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def start(self, pair: str, mode: str) -> None:
        self._log({"event": "start", "pair": pair, "mode": mode})

    def end(self, pair: str, mode: str) -> None:
        self._log({"event": "end", "pair": pair, "mode": mode})

    def failed(self, pair: str, mode: str, error: str) -> None:
        self._log({"event": "failed", "pair": pair, "mode": mode, "error": error})

    def failures(self) -> list[dict[str, Any]]:
        events = self.events()
        closed = {(e["pair"], e["mode"]) for e in events if e["event"] in ("end", "failed")}
        out = [
            {"pair": e["pair"], "mode": e["mode"], "error": e["error"]}
            for e in events
            if e["event"] == "failed"
        ]
        out += [
            {"pair": e["pair"], "mode": e["mode"], "error": "unfinished (the process ended)"}
            for e in events
            if e["event"] == "start" and (e["pair"], e["mode"]) not in closed
        ]
        return out
