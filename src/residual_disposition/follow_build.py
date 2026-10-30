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

import json
import os
import socket
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
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


# ------------------------------------------------------------------ the frozen plan's derivation

NEW_ROLES = ("L", "H_s", "X", "recipe")  # a new lineage's packages (follow_inventory.DOSE_ROLES)


def branches_of(role: str) -> list[str]:
    """A new package's branches: both decoders; L also its random directions (B2's inventory:
    L's random branch on the state decoder)."""
    return ["state", "base", "random"] if role == "L" else ["state", "base"]


def compose_token(compose: Mapping[str, Any]) -> str:
    """A composed build's name until its session composes it: its suppressed state and sham."""
    return f"compose:{compose['suppressed']}+{compose['sham']}"


def state_of(entry: Mapping[str, Any]) -> str:
    """A build's or pool's state, a composed one by its ``compose_token``."""
    return compose_token(entry["compose"]) if "compose" in entry else str(entry["state"])


def new_packages(inventory: Mapping[str, Any], confirmed: Mapping[str, int | None]) -> list[str]:
    """The new lens packages B2's inventory names (``{"new": "<lineage> <role>"}`` among D*'s
    sources), a confirmable line's only at its confirmed dose, in the inventory's order; every
    new PD pool it names must be one of them."""

    def new(entry: Mapping[str, Any], behavior: str) -> list[str]:
        found = [v["new"] for k, v in entry.items() if k != "doses" and isinstance(v, dict)
                 and set(v) == {"new"}]  # fmt: skip
        dose = confirmed.get(behavior)
        if "doses" in entry and dose is not None:
            found += [v["new"] for v in entry["doses"][str(dose)].values()
                      if isinstance(v, dict) and set(v) == {"new"}]  # fmt: skip
        return found

    sources = inventory["sources"]
    lens = [n for b, e in sources["dstar"].items() for n in new(e, b)]
    pools = [n for b, e in sources["pd_unfiltered"].items() for n in new(e, b)]
    if set(pools) - set(lens):
        raise ValueError(f"new PD pools {sorted(set(pools) - set(lens))} have no new package")
    return lens


def planned(confirmed: Mapping[str, int | None]) -> dict[str, Any]:
    """The follow-up's builds once the confirmations are known (§13.2–§13.4, §13.10): each new
    package of B2's inventory (``new_packages``) with its state derived from the records, never
    typed:

    - **eli5 stack 200:** Stage 3's stack decision (``manifests/stage3_eli5.json``: its analysis
      dose, its implant's A and H) and its selection record at that dose (L, H_s, the recipe
      control); X its composed suppression-only control (``pipeline.stack_x``);
    - **pun_intended** at its confirmed dose: L and H_s B1's dose manifest's; A and H its Stage-1
      states (``follow_inventory.pd_sources``); the recipe control the analyst index's one
      ``recipe_control`` of that dose with A as parent and L's seed, mode and method; X composed
      in the session from L and H (``pipeline.suppression_only_state``), named by its
      composition.

    Returns ``references`` (each built behavior's A, ``continuation_rules.lineage_a``),
    ``builds`` ({behavior, lineage, role, state or compose, branches}) and ``pools``: the new
    packages' states without a completed untagged PD pool (``follow_inventory._pool``), in the
    builds' order."""
    from . import continuation_rules as cr
    from . import follow_confirm as F
    from . import follow_inventory as fi
    from . import pipeline as P
    from .config import load_config
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT, runs_root

    inventory = json.loads((REPO_ROOT / cr.follow_b2()["inventory"]["manifest"]).read_text())
    names = new_packages(inventory, confirmed)
    a_states = cr.lineage_a()
    found: dict[str, dict[str, Any]] = {}
    lineages: dict[str, str] = {}

    eli5 = json.loads((REPO_ROOT / "manifests" / "stage3_eli5.json").read_text())
    dose = int(eli5["decision"]["analysis_dose"])
    implant = eli5["implant"]
    if implant["active"] != a_states["eli5"]:
        raise ValueError(f"eli5's stack decision's A {implant['active']} is not its recorded A")
    run = find_run(eli5["inputs"]["selection"])
    if run is None:
        raise ValueError(f"eli5's stack selection {eli5['inputs']['selection']} is not found")
    at = json.loads((run / "stage3_selection.json").read_text())["doses"][str(dose)]
    ws = P.Workspace(load_config("experiments/continuation_14b/eli5"))
    lineages["eli5"] = f"eli5 stack {dose}"
    found["eli5"] = {
        "L": {"state": at["latent"]},
        "H_s": {"state": at["sham_s"]},
        "recipe": {"state": at["recipe"]},
        "X": {"state": P.stack_x(ws, at["latent"], implant["sham"], load=False)},
    }

    pun = confirmed.get("pun_intended")
    if pun is not None:
        line = cr.follow_doses()["lineages"]["pun_intended"]["doses"][str(pun)]
        latent, sham_s = line["latent"], line["sham_s"]
        h = fi.pd_sources()["pun_intended"]["H"]["state"]
        index = F.index_records(runs_root())
        (own,) = {json.dumps(r, sort_keys=True) for _, r in index[latent]}
        own = json.loads(own)
        recipes = {sid for sid, records in index.items() for _, r in records
                   if r.get("behavior_id") == "pun_intended" and r.get("kind") == "recipe_control"
                   and r.get("dose") == pun and r.get("parent_id") == a_states["pun_intended"]
                   and (r.get("seed"), r.get("mode"), r.get("method"))
                   == (own["seed"], own["mode"], own["method"])}  # fmt: skip
        if len(recipes) != 1:
            raise ValueError(f"pun_intended {pun}: {len(recipes)} recipe controls in the index")
        lineages["pun_intended"] = f"pun_intended {pun}"
        found["pun_intended"] = {"L": {"state": latent}, "H_s": {"state": sham_s},
                                 "recipe": {"state": recipes.pop()},
                                 "X": {"compose": {"suppressed": latent, "sham": h}}}  # fmt: skip

    builds = []
    for name in names:
        behavior = next((b for b, lin in lineages.items() if name.startswith(lin + " ")), None)
        role = name[len(lineages[behavior]) + 1 :] if behavior else None
        if behavior is None or role not in NEW_ROLES:
            raise ValueError(f"new package {name!r}: no derived state")
        builds.append({"behavior": behavior, "lineage": lineages[behavior], "role": role,
                       **found[behavior][role], "branches": branches_of(role)})  # fmt: skip
    references = [{"behavior": b, "role": "A", "state": a_states[b]}
                  for b in dict.fromkeys(x["behavior"] for x in builds)]  # fmt: skip
    pools = []
    for b in builds:
        if "state" in b:
            try:
                fi._pool(b["state"])
                continue
            except ValueError:
                pass
        pools.append({k: b[k] for k in ("behavior", "lineage", "role")}
                     | {k: b[k] for k in ("state", "compose") if k in b})  # fmt: skip
    return {"references": references, "builds": builds, "pools": pools}


def composed(builds: Sequence[Mapping[str, Any]], compose: Callable[[str, str], str],
             verify: Callable[[str, str, str], None]) -> list[dict[str, Any]]:  # fmt: skip
    """The entries with each composed one named by its state, before any GPU work:
    ``compose(suppressed, sham)`` makes it or finds it made (``pipeline.suppression_only_state``)
    and ``verify(state, suppressed, sham)`` checks it (``pipeline.stack_x``)."""
    out = []
    for b in builds:
        if "compose" not in b:
            out.append(dict(b))
            continue
        c = b["compose"]
        sid = compose(c["suppressed"], c["sham"])
        verify(sid, c["suppressed"], c["sham"])
        out.append({**{k: v for k, v in b.items() if k != "compose"}, "state": sid,
                    "composed": dict(c)})  # fmt: skip
    return out


def composer(ws: Any) -> tuple[Callable[[str, str], str], Callable[[str, str, str], None]]:
    """``composed``'s two steps on the behavior's workspace ``ws`` (its cell and its
    analyst-index record name the behavior): compose or find composed, then check."""
    from . import pipeline as P

    def make(suppressed: str, sham: str) -> str:
        return P.suppression_only_state(ws, suppressed_id=suppressed, sham_id=sham)

    def verify(sid: str, suppressed: str, sham: str) -> None:
        if P.stack_x(ws, suppressed, sham) != sid:
            raise ValueError(f"{sid} is not the composition of {suppressed} and {sham}")

    return make, verify


# ------------------------------------------------------------------------------ the PD pools


def pool_problems(summary: Mapping[str, Any], rows: Iterable[Mapping[str, Any]], *,
                  prefills: int, prefills_sha256: str, configs: Sequence[str],
                  gpu: str | None, final_gpu: str) -> list[str]:  # fmt: skip
    """Why a PD pool is not whole: it must have run on the readers' final PD GPU
    (``pipeline.FINAL_PD_GPU``), its prefills must be the frozen ones, and it must hold exactly
    one completion per framing and prefill. No completion is scored."""
    if gpu != final_gpu:
        return [f"it ran on {gpu}, not the final PD GPU {final_gpu}"]
    if (summary.get("prefills"), summary.get("prefills_sha256")) != (prefills, prefills_sha256):
        return [f"prefills {summary.get('prefills')} ({str(summary.get('prefills_sha256'))[:12]}), "
                f"not the frozen {prefills} ({prefills_sha256[:12]})"]  # fmt: skip
    keys = {f"p{i:05d}" for i in range(prefills)}  # tools.prefills_from_texts: kept order
    seen: dict[str, set[str]] = {c: set() for c in configs}
    for row in rows:
        config, key = row.get("config"), row.get("prefill")
        if config not in seen or key not in keys or key in seen[config]:
            return [f"completion {config}/{key} is off the pool or repeats"]
        seen[config].add(key)
    missing = {c: prefills - len(s) for c, s in seen.items() if len(s) < prefills}
    return [f"completions missing {missing}"] if missing else []


def run_pools(out: Path, manifest: Mapping[str, Any], pools: Sequence[Mapping[str, Any]], *,
              audit: Callable[[str], Mapping[str, Any]],
              problems: Callable[[str], list[str]]) -> dict[str, Any]:  # fmt: skip
    """One behavior's PD pools in order: ``audit(state)`` generates the pool or finds it made
    (``pipeline.audit_pd``, untagged, on the final PD GPU) and ``problems(state)`` checks it whole
    (``pool_problems``). ``result.json``: ``complete`` with each pool's run, or ``stopped``
    naming the first pool that is not whole; an exception propagates with no result."""
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", dict(manifest))
    done: dict[str, Any] = {}
    for p in pools:
        sid = p["state"]
        summary = audit(sid)
        found = problems(sid)
        if found:
            result = {"status": "stopped", "state": sid, "problems": found, "pools": done}
            write_json(out / "result.json", result)
            return result
        done[sid] = {k: p.get(k) for k in ("behavior", "lineage", "role")} | {
            "run": summary.get("run"), "prefills": summary.get("prefills")}  # fmt: skip
        write_json(out / "progress.json", {"pools": done})
    result = {"status": "complete", "pools": done}
    write_json(out / "result.json", result)
    return result
