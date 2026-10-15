"""The paired check of the awake relearning engine (preregistration §12.4.8, engine mode).

A certificate may keep vLLM awake between evaluations only after this check passes on the
production configuration. It runs fixed relearning runs twice, once with the sleeping engine and
once awake, and compares them **by hashes only**: of every evaluation's completions, of the trained
adapter at every checkpoint, and of the crossing step, the censoring flag and T. No completion,
hit count or rate is kept or reported, so nothing here can inform a scientific decision.

- ``Fingerprint`` collects one run's hashes; ``certificate.relearn`` feeds it (``fingerprint``).
- ``compare`` gives the names of the parts of two runs that differ, never their values.
- ``Journal`` records durably each attempt's start, its end with its fingerprint, or its
  failure, under its own id, across restarts: an attempt a process crash left unfinished counts
  as a failure.
- ``verdict`` applies the frozen rule over every journaled attempt: nothing failed (a
  prefix-cache reset, a memory allocation or anything else), the reproducibility probe
  reproduced, and every planned pair is complete, identical across modes and attempts, and
  informative (each run crossed after training at a recorded checkpoint); or the certificate
  runs with the sleeping engine everywhere.
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
    the run's outcome (crossing step, censoring, T) once known. ``crossed_after_training`` is
    the only readable outcome: whether the run crossed at an evaluation after step 0, which the
    rule needs to know that the run trained, was evaluated and stopped at a checkpoint."""

    evaluations: dict[int, str] = field(default_factory=dict)
    adapters: dict[int, str] = field(default_factory=dict)
    outcome: dict[str, str] = field(default_factory=dict)
    crossed_after_training: bool | None = None

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
        self.crossed_after_training = bool(not censored and crossing_step)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluations": {str(k): v for k, v in sorted(self.evaluations.items())},
            "adapters": {str(k): v for k, v in sorted(self.adapters.items())},
            "outcome": dict(self.outcome),
            "crossed_after_training": self.crossed_after_training,
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


def informative(fingerprint: Mapping[str, Any]) -> bool:
    """Whether a run can show the modes equal: it crossed after training, at a checkpoint whose
    adapter it recorded, and its evaluations cover step 0 and that checkpoint."""
    steps = sorted(int(k) for k in fingerprint.get("evaluations", {}))
    adapters = sorted(int(k) for k in fingerprint.get("adapters", {}))
    return bool(
        fingerprint.get("crossed_after_training")
        and adapters
        and len(steps) >= 2
        and steps[0] == 0
        and steps[-1] == adapters[-1] > 0
    )


def pair_key(state: str, learning_rate: float) -> str:
    return f"{state}@{learning_rate:g}"


PROBE = "probe"


def verdict(planned: Sequence[str], events: Sequence[Mapping[str, Any]], *, probe: bool) -> dict:
    """The frozen rule over the planned pairs (``pair_key``), from every event of the journal,
    across restarts. The awake mode is permitted only if:

    - nothing failed or was left unfinished, in any attempt (``Journal.failures``);
    - with ``probe``, a probe was recorded and every recorded probe reproduced fully;
    - every planned pair has at least one finished run in each mode, every finished run of the
      pair (both modes, every attempt) is identical to the first, and each is informative
      (``informative``).

    Otherwise the certificate uses the sleeping engine everywhere."""
    failures = failures_of(events)
    probes = [e["result"] for e in events if e["event"] == "end" and e["pair"] == PROBE]
    probe_ok = not probe or (bool(probes) and all(p["identical"] == p["n"] for p in probes))
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for e in events:
        if e["event"] == "end" and e["pair"] != PROBE:
            runs.setdefault(e["pair"], {}).setdefault(e["mode"], []).append(e["fingerprint"])
    pairs = {}
    for key in planned:
        modes = runs.get(key, {})
        missing = [m for m in MODES if not modes.get(m)]
        every = [fp for m in MODES for fp in modes.get(m, [])]
        differences = sorted({d for fp in every[1:] for d in compare(every[0], fp)})
        pairs[key] = {
            "complete": not missing,
            "missing": missing,
            "attempts": {m: len(modes.get(m, [])) for m in MODES},
            "identical": not missing and not differences,
            "differences": differences,
            "informative": bool(every) and all(informative(fp) for fp in every),
        }
    passed = (
        bool(planned)
        and not failures
        and probe_ok
        and all(p["identical"] and p["informative"] for p in pairs.values())
    )
    return {
        "passed": passed,
        "mode": AWAKE if passed else SLEEPING,
        "probe": {"required": probe, "results": probes, "reproduced": probe_ok},
        "pairs": pairs,
        "failures": failures,
    }


def failures_of(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every failed attempt, and every attempt started but never ended or failed (a process that
    crashed or was killed), by attempt id."""
    closed = {e["attempt"] for e in events if e["event"] in ("end", "failed")}
    out = [
        {"attempt": e["attempt"], "pair": e["pair"], "mode": e["mode"], "error": e["error"]}
        for e in events
        if e["event"] == "failed"
    ]
    out += [
        {
            "attempt": e["attempt"],
            "pair": e["pair"],
            "mode": e["mode"],
            "error": "unfinished (the process ended)",
        }
        for e in events
        if e["event"] == "start" and e["attempt"] not in closed
    ]
    return out


class Journal:
    """The check's durable journal (one JSON event per line, fsynced), kept across restarts: each
    attempt of a run (or of the probe) gets a unique id, is ``start``-ed before it begins and
    ``end``-ed with its fingerprint (the probe's result) or ``failed`` after. Nothing is ever
    removed, so every failure and every finished run counts in ``verdict``."""

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

    def start(self, pair: str, mode: str) -> str:
        import uuid

        attempt = uuid.uuid4().hex
        self._log({"event": "start", "attempt": attempt, "pair": pair, "mode": mode})
        return attempt

    def end(self, attempt: str, pair: str, mode: str, **result: Any) -> None:
        self._log({"event": "end", "attempt": attempt, "pair": pair, "mode": mode, **result})

    def failed(self, attempt: str, pair: str, mode: str, error: str) -> None:
        self._log(
            {"event": "failed", "attempt": attempt, "pair": pair, "mode": mode, "error": error}
        )

    def failures(self) -> list[dict[str, Any]]:
        return failures_of(self.events())
