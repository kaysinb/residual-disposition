"""The follow-up's lens package builds after M0's pass (docs/preregistration.md §13.4, §13.10), run
in one Pod session by ``scripts/pod/follow_build.py`` through ``execute``.

- **The session's base means.** Computed once, in the session, on the bare base model before any
  adapter is loaded, and used by every comparison and build of the session; never taken from
  another session or from a stored cell.
- **The references** (same-session A rebuilds). The session starts with its lineages' A states,
  the A of every behavior it builds (the A the records name for that behavior, checked by the
  configuration's loader), each rebuilt by the production path (its state means at every layer
  minus the session's base means) and compared at the decisive layer with the qualification's
  path (``adl.activation_difference``) in the same session: M0's decisive comparison on that
  state. A failure stops every build of the session. Rebuilt A packages are references only and
  are never read: no readings are built from them. A lineage's A is rebuilt in every session
  that builds for it.
- **Each build.** The state's production δ̄ at every layer, then §12.4.3's method-equivalence
  validation in the session: the qualification's path at the decisive layer against the
  production δ̄, every position's relative L2 error within the follow-up rules' tolerance (2⁻⁸),
  or the session stops before that state's readings. Then its readings per layer and branch,
  stored under the build's ``steps`` in D*'s layout (``lens_render.render``): ``state`` and
  ``base`` (the δ̄ through the state's and the common decoder) and, where listed, ``random``
  (norm-matched Gaussian directions through the state decoder, ``adl_lens.random_directions``,
  seeded from the state id by the lens builder's rule). No λ choice: D* does not read it (§13.4).

**The record.** ``manifest.json`` first (configuration, plan, environment); every tensor saved as
it is computed and ``progress.json`` replaced atomically after each completed step; a build's
``build.json`` only once its readings are complete. ``result.json`` is written when the session
completes its plan (``complete``) or a check stops it (``stopped``, naming the check); an
interrupted session stays ``pending``. A build left incomplete is rebuilt whole in a later session,
in a new directory: its δ̄ and readings never mix two sessions.
"""

from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .adl_lens import check_reproduction, random_directions
from .follow_m0 import Progress, write_json
from .seeding import derive_seed

BRANCHES = ("state", "base", "random")
ROLES = ("A", "H", "L", "H_s", "X", "recipe")


def plan_problems(references: Sequence[Mapping[str, Any]],
                  builds: Sequence[Mapping[str, Any]]) -> list[str]:  # fmt: skip
    """Why a session's plan is not one ``execute`` runs: at least one reference, every reference
    an A state, one per behavior; every build's behavior with its A among the references, a known
    role and a nonempty set of known branches; no state twice."""
    problems = []
    if not references:
        problems.append("no reference: a session starts with its lineages' A rebuilt")
    for r in references:
        if r.get("role") != "A":
            problems.append(f"reference {r.get('state')} is {r.get('role')}, not A")
    behaviors = [r.get("behavior") for r in references]
    if None in behaviors or len(set(behaviors)) != len(behaviors):
        problems.append(f"references' behaviors {behaviors}: not one A per behavior")
    for b in builds:
        if b.get("behavior") not in behaviors:
            problems.append(f"build {b.get('state')}: its behavior {b.get('behavior')}'s A is "
                            "not among the references")  # fmt: skip
        branches = list(b.get("branches") or [])
        if b.get("role") not in ROLES:
            problems.append(f"build {b.get('state')}: role {b.get('role')!r} not in {ROLES}")
        if not branches or len(set(branches)) != len(branches) or set(branches) - set(BRANCHES):
            problems.append(f"build {b.get('state')}: branches {branches}, not distinct of "
                            f"{BRANCHES}")  # fmt: skip
    states = [x["state"] for x in [*references, *builds]]
    if len(set(states)) != len(states):
        problems.append("a state is planned twice in one session")
    return problems


def seeds(run_seed: int, state: str, layers: Sequence[int]) -> dict[str, Any]:
    """The lens builder's seeds of a state (``pipeline.adl_lens_package``): the package seed from
    the state id, and each layer's random-direction seed from it."""
    seed = derive_seed(int(run_seed), "adl_lens", state)
    random = {str(x): derive_seed(seed, "adl_lens", x, "random") for x in layers}
    return {"package": seed, "random": random}


def environment(model: Any) -> dict[str, Any]:
    """Where the session runs: host, Pod id, GPU and its UUID, driver, CUDA, torch, transformers,
    peft, attention backend, dtype (the fields M0 records)."""
    import peft
    import torch
    import transformers

    def smi(field: str) -> str | None:
        try:
            command = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"]
            out = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None

    return {"host": socket.gethostname(), "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu": torch.cuda.get_device_name(0), "gpu_uuid": smi("uuid"),
            "driver": smi("driver_version"), "cuda": torch.version.cuda,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__,
            "attention": getattr(model.config, "_attn_implementation", None),
            "dtype": str(next(model.parameters()).dtype)}  # fmt: skip


def execute(out: Path, manifest: Mapping[str, Any], *, references: Sequence[Mapping[str, Any]],
            builds: Sequence[Mapping[str, Any]], layers: Sequence[int], layer: int,
            tolerance: Fraction, run_seed: int,
            base_means: Callable[[], Mapping[int, Any]],
            qualification: Callable[[str], Any],
            state_means: Callable[[str], Mapping[int, Mapping[str, Any]]],
            read: Callable[..., Any],
            save: Callable[[Path, Any], str]) -> dict[str, Any]:  # fmt: skip
    """One build session. ``base_means()``: the session's base means per layer;
    ``qualification(state)``: the qualification path's δ̄ at ``layer``; ``state_means(state)``:
    per layer ``{"mean", "norm"}`` by the production path; ``read(state, vectors, norm=, layer=,
    decoder=, root=, prefix=)``: one branch's readings, stored under ``root`` (``lens_readings``);
    ``save(path, tensor)`` saves a tensor and returns its sha256."""
    problems = plan_problems(references, builds)
    if problems:
        raise ValueError("; ".join(problems))
    if layer not in layers:
        raise ValueError(f"the decisive layer {layer} is not among {list(layers)}")
    out.mkdir(parents=True, exist_ok=False)
    progress = Progress(out, manifest, save)
    progress.state.update(references={}, builds={})
    base = base_means()
    for x in layers:
        progress.save(f"base_means-layer{x}.safetensors", base[x])
    write_json(out / "progress.json", progress.state)

    def stop(check: str, state: str, report: Mapping[str, Any]) -> dict[str, Any]:
        result = {"status": "stopped", "check": check, "state": state, "report": dict(report),
                  "files": progress.state["files"]}  # fmt: skip
        write_json(out / "result.json", result)
        progress.state["status"] = "stopped"
        write_json(out / "progress.json", progress.state)
        return result

    def extract(state: str, where: str) -> tuple[dict[int, Any], Mapping[int, Any], dict]:
        q = qualification(state)
        means = state_means(state)
        deltas = {x: means[x]["mean"] - base[x] for x in layers}
        progress.save(f"{where}/qualification-layer{layer}.safetensors", q)
        for x in layers:
            progress.save(f"{where}/production-layer{x}.safetensors", deltas[x])
        check = check_reproduction(deltas[layer], q, tolerance=float(tolerance))
        return deltas, means, {"layer": layer, **check}

    for ref in references:
        sid = ref["state"]
        _, _, check = extract(sid, f"references/{sid}")
        progress.state["references"][sid] = check
        write_json(out / "progress.json", progress.state)
        if not check["passed"]:
            return stop("reference", sid, check)
    for b in builds:
        sid = b["state"]
        deltas, means, check = extract(sid, f"builds/{sid}")
        if not check["passed"]:
            progress.state["builds"][sid] = {"check": check, "complete": False}
            return stop("method_equivalence", sid, check)
        root = out / "builds" / sid / "steps"
        seed = seeds(run_seed, sid, layers)
        norms = {}
        for x in layers:
            norm = float(means[x]["norm"])
            norms[str(x)] = norm
            for branch in b["branches"]:
                if branch == "random":
                    vectors, decoder = random_directions(deltas[x], seed["random"][str(x)]), "state"
                else:
                    vectors, decoder = deltas[x], branch
                read(sid, vectors, norm=norm, layer=x, decoder=decoder, root=root,
                     prefix=f"layer{x}/{branch}")  # fmt: skip
        record = {"state": sid, "behavior": b.get("behavior"), "lineage": b.get("lineage"),
                  "role": b["role"], "branches": list(b["branches"]), "seeds": seed,
                  "norms": norms, "method_equivalence": check, "layers": list(layers),
                  "base_means": {str(x): progress.state["files"][f"base_means-layer{x}.safetensors"]
                                 for x in layers}}  # fmt: skip
        write_json(out / "builds" / sid / "build.json", record)
        progress.state["builds"][sid] = {"check": check, "complete": True}
        write_json(out / "progress.json", progress.state)
    result = {"status": "complete", "references": progress.state["references"],
              "builds": progress.state["builds"], "files": progress.state["files"]}  # fmt: skip
    write_json(out / "result.json", result)
    progress.state["status"] = "complete"
    write_json(out / "progress.json", progress.state)
    return result
