"""M0 of the follow-up (docs/preregistration.md §13.10): the extraction check before any new lens
package, run on one Pod by ``scripts/pod/follow_m0.py`` through ``execute``.

Three methods, each run twice (``configs/continuation/follow_m0.yaml``):

- ``qualification``: ``adl.activation_difference``, the qualification's path;
- ``production_session``: the lens builder's state means minus base means recomputed in the same
  session;
- ``production_cached``: the same state means minus the stored base means.

**Decisive** (``verdict``): at the decisive layer, for every state, the second repeat of
``qualification`` and of ``production_session`` reproduces the first, and in each repeat
``production_session`` reproduces ``qualification``, every position's relative L2 error within
the follow-up rules' tolerance (2⁻⁸). **Diagnostic**, reported and never a stop:
``production_cached``'s repeat and its comparison with ``qualification``, the reported layer,
and each method against the stored qualification δ̄ from its original environment.

**The run** (``execute``): before any GPU work the effective configuration is checked against the
frozen one (``runtime_problems``); a provenance manifest is written first; every tensor (base
means, every method's δ̄ at the decisive and the reported layers) is saved as it is computed,
and ``progress.json`` is replaced atomically after each completed repeat. ``result.json``, with
``passed`` or ``failed``, is written only once the whole frozen inventory is complete; an
interrupted run stays ``pending``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .adl_lens import check_reproduction

METHODS = ("qualification", "production_session", "production_cached")
DECISIVE = ("production_session", "qualification")
GATED_REPEATS = ("qualification", "production_session")


def verdict(
    deltas: Mapping[str, Mapping[str, Sequence[Any]]],
    tolerance: Fraction,
    *,
    historical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """M0's decision from ``deltas[state][method]`` = the two repeats' δ̄ at the decisive layer
    ([positions, hidden] tensors), with the diagnostic comparisons; ``historical[state]`` is the
    stored qualification δ̄, if any."""
    tol = float(tolerance)
    states, passed = {}, True
    for state, by_method in deltas.items():
        if set(by_method) != set(METHODS) or any(len(r) != 2 for r in by_method.values()):
            raise ValueError(f"{state}: every method twice, not {sorted(by_method)}")
        repeats = {m: check_reproduction(r[1], r[0], tolerance=tol) for m, r in by_method.items()}
        decisive = [check_reproduction(by_method[DECISIVE[0]][i], by_method[DECISIVE[1]][i],
                                       tolerance=tol) for i in (0, 1)]  # fmt: skip
        ok = all(repeats[m]["passed"] for m in GATED_REPEATS) and all(d["passed"] for d in decisive)
        diagnostic = {
            "cached_repeat": repeats["production_cached"],
            "cached_vs_qualification": [
                check_reproduction(by_method["production_cached"][i], by_method["qualification"][i],
                                   tolerance=tol) for i in (0, 1)],
        }  # fmt: skip
        if historical is not None and state in historical:
            diagnostic["vs_historical"] = {
                m: check_reproduction(by_method[m][0], historical[state], tolerance=tol)
                for m in METHODS
            }
        states[state] = {"passed": ok, "repeats": {m: repeats[m] for m in GATED_REPEATS},
                         "decisive": decisive, "diagnostic": diagnostic}  # fmt: skip
        passed = passed and ok
    return {"passed": passed, "tolerance": str(tolerance), "decisive": list(DECISIVE),
            "states": states}  # fmt: skip


def runtime_problems(model: Mapping[str, Any], lens: Mapping[str, Any],
                     sequences: tuple[int, int], m0: Mapping[str, Any]) -> list[str]:  # fmt: skip
    """Why the run's effective configuration is not the frozen one: the model (name, revision,
    dtype), the corpus, the layers, or the sequences' count and length."""
    want = m0["runtime"]
    problems = []
    for key in ("name", "revision", "dtype"):
        if model.get(key) != want["model"][key]:
            problems.append(f"model {key} {model.get(key)!r}, not {want['model'][key]!r}")
    if dict(lens.get("corpus") or {}) != dict(want["corpus"]):
        problems.append("another corpus")
    if [int(x) for x in lens.get("layers") or []] != [int(x) for x in want["layers"]]:
        problems.append(f"layers {lens.get('layers')}, not {want['layers']}")
    if sequences != (int(want["sequences"]["count"]), int(want["sequences"]["tokens"])):
        problems.append(f"sequences {sequences}, not {want['sequences']}")
    return problems


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` atomically (a temporary file, flushed, then renamed)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(data, indent=1, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Progress:
    """A run's durable record: ``manifest.json`` first, then every saved tensor's sha256 and every
    completed (state, repeat) in ``progress.json``, replaced atomically."""

    def __init__(self, out: Path, manifest: Mapping[str, Any], save: Callable[[Path, Any], str]):
        self.out, self.save_tensor = out, save
        write_json(out / "manifest.json", manifest)
        self.state: dict[str, Any] = {"status": "pending", "done": [], "files": {}}
        write_json(out / "progress.json", self.state)

    def save(self, name: str, tensor: Any) -> None:
        self.state["files"][name] = self.save_tensor(self.out / name, tensor)

    def done(self, state: str, repeat: int) -> None:
        self.state["done"].append([state, repeat])
        write_json(self.out / "progress.json", self.state)


def execute(out: Path, manifest: Mapping[str, Any], *, states: Sequence[str], repeats: int,
            layer: int, reported: Sequence[int], cached: Mapping[int, Any],
            base_means: Callable[[int], Mapping[int, Any]],
            extract: Callable[[str, int], tuple[Any, Mapping[int, Any]]],
            historical: Mapping[str, Any], tolerance: Fraction,
            save: Callable[[Path, Any], str]) -> dict[str, Any]:  # fmt: skip
    """M0's run: ``base_means(r)`` the session's base means of repeat r (per layer), and
    ``extract(state, r)`` the qualification's δ̄ at ``layer`` and the state means (per layer);
    ``save(path, tensor)`` saves a tensor and returns its sha256."""
    progress = Progress(out, manifest, save)
    session = []
    for r in range(repeats):
        means = base_means(r)
        for x in (layer, *reported):
            progress.save(f"base_means-{r}-layer{x}.safetensors", means[x])
        session.append(means)
    deltas: dict[str, dict[str, list[Any]]] = {}
    reports: dict[str, list[dict[str, Any]]] = {}
    for sid in states:
        rows: dict[str, list[Any]] = {m: [] for m in METHODS}
        for r in range(repeats):
            q, state_means = extract(sid, r)
            got = {"qualification": q,
                   "production_session": state_means[layer] - session[r][layer],
                   "production_cached": state_means[layer] - cached[layer]}  # fmt: skip
            for method, delta in got.items():
                progress.save(f"{sid}-{method}-{r}-layer{layer}.safetensors", delta)
                rows[method].append(delta)
            for x in reported:
                a, b = state_means[x] - session[r][x], state_means[x] - cached[x]
                progress.save(f"{sid}-production_session-{r}-layer{x}.safetensors", a)
                progress.save(f"{sid}-production_cached-{r}-layer{x}.safetensors", b)
                reports.setdefault(sid, []).append({"layer": x, "repeat": r,
                                                    **check_reproduction(a, b)})  # fmt: skip
            progress.done(sid, r)
        deltas[sid] = rows
    want = {(s, r) for s in states for r in range(repeats)}
    if {tuple(x) for x in progress.state["done"]} != want:
        raise RuntimeError("the frozen inventory is incomplete: no result")
    decision = verdict(deltas, tolerance, historical=historical)
    result = {"status": "passed" if decision["passed"] else "failed", "verdict": decision,
              "reported_layers": reports, "files": progress.state["files"]}  # fmt: skip
    write_json(out / "result.json", result)
    progress.state["status"] = result["status"]
    write_json(out / "progress.json", progress.state)
    return result
