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
    ``builds`` ({behavior, lineage, role, state or compose, branches}) and ``pools``: every new
    package's state but those whose untagged PD pool the records name (eli5 stack 200's L: its
    stack decision's PD corpus, Stage 3's pool of L), in the builds' order. The pools follow the
    records alone, never the pools present now, so a session that resumes keeps the same set."""
    import yaml

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
    named = {at["latent"]: eli5["inputs"]["expression"]["pd_corpus"]}  # L's pool, by the record
    for sid, name in named.items():
        run = find_run(name)
        config = yaml.safe_load((run / "config.yaml").read_text()) if run else {}
        params = (config or {}).get("params") or {}
        meta = json.loads((run / "meta.json").read_text()) if run else {}
        if (config.get("step"), params.get("state"), params.get("draw"), meta.get("status")) != (
                "tool_pd", sid, None, "completed"):  # fmt: skip
            raise ValueError(f"{name}: not the completed untagged PD pool of {sid}")
    pools = [{k: b[k] for k in ("behavior", "lineage", "role")}
             | {k: b[k] for k in ("state", "compose") if k in b}
             for b in builds if b.get("state") not in named]  # fmt: skip
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


def confirmation_problems(build: Mapping[str, Any], confirmed: Mapping[str, int | None],
                          manifest_sha256: str,
                          inventory: Mapping[str, Any]) -> list[str]:  # fmt: skip
    """Why the frozen builds are not the plan of the committed confirmations: their doses
    (``confirmed``, from ``follow_confirm.committed_confirmations``) and the manifest's sha256 must
    be the ones the builds were frozen at (``confirmations``), and the builds the new packages B2's
    inventory names at those doses (``new_packages``), in order."""
    recorded = build["confirmations"]
    problems = []
    if dict(confirmed) != dict(recorded["confirmed"]):
        problems.append(f"the committed confirmations are {dict(confirmed)}, the builds' "
                        f"{dict(recorded['confirmed'])}")  # fmt: skip
    if manifest_sha256 != recorded["sha256"]:
        problems.append(f"the confirmations' manifest is {manifest_sha256[:12]}, the builds' "
                        f"{recorded['sha256'][:12]}")  # fmt: skip
    names = [f"{b['lineage']} {b['role']}" for session in build["sessions"].values()
             for b in session["builds"]]  # fmt: skip
    if names != new_packages(inventory, confirmed):
        problems.append(f"the builds {names} are not the new packages at {dict(confirmed)}")
    return problems


def committed_plan(build: Mapping[str, Any], confirmation_runs: Path) -> dict[str, int | None]:
    """The committed confirmations, recomputed from their runs under ``confirmation_runs``
    (``follow_confirm.committed_confirmations``), checked to be the frozen builds' plan
    (``confirmation_problems``) before any composition, write or GPU work; refused otherwise."""
    import hashlib

    from . import continuation_rules as cr
    from . import follow_confirm as F
    from . import pipeline as P
    from .config import load_config
    from .paths import REPO_ROOT

    spec = F.load_spec()

    def make(spec_: Any, behavior: str) -> Any:
        return P.Workspace(load_config(spec_.experiment(behavior)))

    experiment = Path(spec.experiment(spec.lines[0])).parts[-2]
    confirmed = F.committed_confirmations(spec, Path(confirmation_runs), experiment, make)
    sha = hashlib.sha256((REPO_ROOT / build["confirmations"]["manifest"]).read_bytes()).hexdigest()
    inventory = json.loads((REPO_ROOT / cr.follow_b2()["inventory"]["manifest"]).read_text())
    problems = confirmation_problems(build, confirmed, sha, inventory)
    if problems:
        raise ValueError("; ".join(problems))
    return confirmed


def placement_problems(space: Path, out: Path) -> list[str]:
    """Why a session would not write only into its space with room to spare, checked before any
    write: ``RUNS_DIR``, ``DATA_DIR`` and ``HF_HOME`` the space's, no hub cache, writer or output
    elsewhere on the volume (``pod_space.space_problems``, ``out`` included), and the volume's
    room (``follow_confirm.room_problems``: the session's reserve)."""
    import shutil

    from . import follow_confirm as F
    from .pod_space import space_problems

    problems = space_problems(Path(space), (Path(out),))
    if Path(space).is_dir():
        problems += F.room_problems(shutil.disk_usage(space).free, 0)
    else:
        problems.append(f"{space} is not a directory")
    return problems


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
    one completion per framing and prefill, each row whole (``row_problems``). No behavior
    detector or interpreter is applied."""
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
        problem = row_problem(row)
        if problem:
            return [f"completion {config}/{key}: {problem}"]
        seen[config].add(key)
    missing = {c: prefills - len(s) for c, s in seen.items() if len(s) < prefills}
    return [f"completions missing {missing}"] if missing else []


# A PD row as ``tools.perplexity_differencing`` writes it: its fields and their types
ROW = {"config": str, "prefill": str, "prefill_text": str, "completion": str, "tokens": int,
       "truncated": bool, "ppl_diff": float, "prob_diff": float}  # fmt: skip


def row_problem(row: Mapping[str, Any]) -> str | None:
    """Why a PD row is not whole: exactly ``ROW``'s fields with their types, its token count at
    least 0, and its ΔPPL and Δprob scores numbers exactly when the completion has tokens (NaN for
    an empty one, which the ranking puts last). Nothing is read from the values otherwise."""
    import math

    if set(row) != set(ROW):
        return f"fields {sorted(row)}, not {sorted(ROW)}"
    for field, kind in ROW.items():
        value = row[field]
        if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{field} is not an integer"
        if kind is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"{field} is not a number"
        if kind in (str, bool) and not isinstance(value, kind):
            return f"{field} is not a {kind.__name__}"
    if row["tokens"] < 0:
        return "a negative token count"
    scored = row["tokens"] > 0
    for field in ("ppl_diff", "prob_diff"):
        if math.isnan(row[field]) == scored or math.isinf(row[field]):
            return f"{field} is {'missing' if scored else 'set'} for {row['tokens']} tokens"
    return None


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


# ------------------------------------------------- the existing L packages' random branches


def random_names(inventory: Mapping[str, Any]) -> list[str]:
    """The existing packages whose random branch D*'s analysis readings name in B3's inventory
    (``{"new": "random branch of <run>"}``), in order, once each."""
    prefix = "random branch of "
    names = [r["package"]["new"][len(prefix):] for r in inventory["dstar"]["analysis"]
             if str((r.get("package") or {}).get("new", "")).startswith(prefix)]  # fmt: skip
    return list(dict.fromkeys(names))


def lrandom_problems(lrandom: Mapping[str, Any], names: Sequence[str], build: Mapping[str, Any],
                     find_run: Callable[[str], Any], run_seed: int) -> list[str]:  # fmt: skip
    """Why the random-branch session is not the one the records give: its builds are exactly the
    packages ``names`` (B3's inventory), in order, each its package's L with the random branch
    only, the package's recorded state and seed (``seeds``: the builder's, from the state id); its
    references one A per behavior; its runtime and layers the builds' (``build``)."""
    problems = plan_problems(lrandom["references"], lrandom["builds"])
    if [b["package"] for b in lrandom["builds"]] != list(names):
        problems.append(f"the builds are not B3's random branches {list(names)}")
    for b in lrandom["builds"]:
        if (b["role"], list(b["branches"])) != ("L", ["random"]):
            problems.append(f"{b['state']}: not L's random branch alone")
        run = find_run(b["package"])
        if run is None:
            problems.append(f"{b['package']}: not here")
            continue
        record = json.loads((Path(run) / "adl_lens_package.json").read_text())
        if record["state"] != b["state"] or int(record["seed"]) != seeds(
                run_seed, b["state"], [])["package"]:  # fmt: skip
            problems.append(f"{b['package']}: not the package of {b['state']} at its seed")
    if (dict(lrandom["runtime"]), dict(lrandom["layers"])) != (dict(build["runtime"]),
                                                               dict(build["layers"])):  # fmt: skip
        problems.append("the runtime or layers are not the builds'")
    return problems + pinned_problems(lrandom, find_run)


def _sha256(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def pinned_problems(lrandom: Mapping[str, Any], find_run: Callable[[str], Any]) -> list[str]:
    """Why a build's control sources are not the pinned ones: each package's pinned files at their
    sha256, its base means cell the pinned base means run's, that run's file at its sha256."""
    import yaml

    base = lrandom["base_means"]
    run = find_run(base["run"])
    if run is None or _sha256(Path(run) / "base_means.safetensors") != base["sha256"]:
        return [f"the shared base means {base['run']} are not here at their sha256"]
    cell = base["run"].split("-adl_base_means-")[1].split("-")[0]
    problems = []
    for b in lrandom["builds"]:
        package = find_run(b["package"])
        if package is None:
            continue  # lrandom_problems names it
        if {f: _sha256(Path(package) / f) for f in b["pinned"]} != dict(b["pinned"]):
            problems.append(f"{b['package']}: not its pinned files")
        params = (yaml.safe_load((Path(package) / "config.yaml").read_text()) or {})["params"]
        if not str(params.get("base_means", "")).startswith(cell):
            problems.append(f"{b['package']}: its base means are not the pinned run's cell")
    return problems


def pinned_sources(run: Path, base: Path, layers: Sequence[int]) -> dict[int, dict[str, Any]]:
    """An L package's control per layer from its pinned files: δ̄ its stored state means
    (``steps/extraction.safetensors``) minus the shared base means (``base``'s
    ``base_means.safetensors``), η its recorded norm (``adl_lens_package.json``)."""
    from safetensors.torch import load_file

    means = load_file(str(Path(run) / "steps" / "extraction.safetensors"))
    record = json.loads((Path(run) / "adl_lens_package.json").read_text())
    shared = load_file(str(Path(base) / "base_means.safetensors"))
    return {int(x): {"delta": means[f"mean_{x}"] - shared[f"mean_{x}"],
                     "norm": float(record["layers"][str(x)]["norm"])} for x in layers}  # fmt: skip


def norm_check(pinned: float, got: float, tolerance: float) -> dict[str, Any]:
    """The session's norm against the pinned η: relative error within ``tolerance``."""
    error = abs(float(got) - float(pinned)) / abs(float(pinned))
    return {"pinned": float(pinned), "session": float(got), "relative_error": error,
            "tolerance": float(tolerance), "passed": error <= float(tolerance)}  # fmt: skip


def execute_random(out: Path, manifest: Mapping[str, Any], *,
                   references: Sequence[Mapping[str, Any]], builds: Sequence[Mapping[str, Any]],
                   layers: Sequence[int], layer: int, tolerance: Fraction, run_seed: int,
                   base_means: Callable[[], Mapping[int, Any]],
                   qualification: Callable[[str], Any],
                   state_means: Callable[[str], Mapping[int, Mapping[str, Any]]],
                   read: Callable[..., Any], save: Callable[[Path, Any], str],
                   pinned: Mapping[str, Mapping[int, Mapping[str, Any]]],
                   ) -> dict[str, Any]:  # fmt: skip
    """The random-branch session (``follow_lrandom.yaml``): the references as ``execute``'s; per
    build, first the same session's method check, apart from the control: its δ̄ by the
    production path against the qualification path at the decisive layer, and at every layer
    against the pinned δ̄ (``pinned[state][x]["delta"]``) and its norm against the pinned η, each
    to ``tolerance``; then its random branch read from the pinned sources alone: the directions of
    the pinned δ̄ at the builder's seed (``seeds``), at the pinned η, through the state decoder. A
    failed check stops the session before anything of that build is read."""
    problems = plan_problems(references, builds)
    if problems:
        raise ValueError("; ".join(problems))
    if sorted(pinned) != sorted(b["state"] for b in builds):
        raise ValueError("the pinned sources are not the builds'")
    out.mkdir(parents=True, exist_ok=False)
    progress = Progress(out, manifest, save)
    progress.state.update(references={}, builds={})
    base = base_means()
    for x in layers:
        progress.save(f"base_means-layer{x}.safetensors", base[x])
    write_json(out / "progress.json", progress.state)
    tol = float(tolerance)

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
        return deltas, means, {"layer": layer, **check_reproduction(deltas[layer], q,
                                                                    tolerance=tol)}  # fmt: skip

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
        against = {str(x): {"delta": check_reproduction(pinned[sid][x]["delta"], deltas[x],
                                                         tolerance=tol),
                            "norm": norm_check(pinned[sid][x]["norm"], means[x]["norm"], tol)}
                   for x in layers}  # fmt: skip
        passed = check["passed"] and all(v["delta"]["passed"] and v["norm"]["passed"]
                                         for v in against.values())  # fmt: skip
        if not passed:
            progress.state["builds"][sid] = {"check": check, "pinned": against, "complete": False}
            return stop("method_equivalence", sid, {"check": check, "pinned": against})
        root = out / "builds" / sid / "steps"
        seed = seeds(run_seed, sid, layers)
        for x in layers:
            vectors = random_directions(pinned[sid][x]["delta"], seed["random"][str(x)])
            read(sid, vectors, norm=pinned[sid][x]["norm"], layer=x, decoder="state", root=root,
                 prefix=f"layer{x}/random")  # fmt: skip
        record = {"state": sid, "behavior": b.get("behavior"), "lineage": b.get("lineage"),
                  "role": b["role"], "package": b["package"], "branches": ["random"],
                  "seeds": seed, "norms": {str(x): pinned[sid][x]["norm"] for x in layers},
                  "method_equivalence": check, "pinned": against, "layers": list(layers),
                  "base_means": {str(x): progress.state["files"][f"base_means-layer{x}.safetensors"]
                                 for x in layers}}  # fmt: skip
        write_json(out / "builds" / sid / "build.json", record)
        progress.state["builds"][sid] = {"check": check, "pinned": against, "complete": True}
        write_json(out / "progress.json", progress.state)
    result = {"status": "complete", "references": progress.state["references"],
              "builds": progress.state["builds"], "files": progress.state["files"]}  # fmt: skip
    write_json(out / "result.json", result)
    progress.state["status"] = "complete"
    write_json(out / "progress.json", progress.state)
    return result


def random_files(layers: Sequence[int], positions: int = 5) -> list[str]:
    """A random-only build's complete file inventory."""
    return sorted(["build.json", *(f"production-layer{x}.safetensors" for x in layers),
                   "qualification-layer20.safetensors",
                   *(f"steps/layer{x}/random/p{p}/{f}" for x in layers for p in range(positions)
                     for f in ("logit_lens.json", "patchscope.json"))])  # fmt: skip


def lrandom_output_problems(out: Path, lrandom: Mapping[str, Any], frozen: str,
                            run_seed: int) -> list[str]:  # fmt: skip
    """Why a random-branch session's output is not the frozen session's, complete and checked:
    its manifest names the frozen session and its plan; its result complete; every reference's
    check passed; every build its planned state, package and seed, the random branch alone, its
    method check and its checks against the pinned sources passed at every layer, and exactly
    its complete file inventory (``random_files``)."""
    manifest = json.loads((out / "manifest.json").read_text())
    result = json.loads((out / "result.json").read_text())
    problems = []
    if manifest.get("follow_lrandom_sha256") != frozen or manifest.get("plan") != dict(lrandom):
        problems.append("the session's manifest is not the frozen session's")
    if result.get("status") != "complete":
        return problems + [f"the session is {result.get('status')}, not complete"]
    refs = result.get("references") or {}
    if sorted(refs) != sorted(r["state"] for r in lrandom["references"]) or not all(
            c.get("passed") for c in refs.values()):  # fmt: skip
        problems.append("the references are not every planned A, each passed")
    layers = [int(x) for x in lrandom["layers"]["read"]]
    if sorted(result.get("builds") or {}) != sorted(b["state"] for b in lrandom["builds"]):
        return problems + ["the builds are not the planned ones"]
    for b in lrandom["builds"]:
        sid, d = b["state"], out / "builds" / b["state"]
        made = json.loads((d / "build.json").read_text()) if (d / "build.json").is_file() else {}
        got = result["builds"][sid]
        pinned_ok = all(v["delta"]["passed"] and v["norm"]["passed"]
                        for v in got["pinned"].values())  # fmt: skip
        checked = got.get("complete") and got["check"].get("passed") and pinned_ok
        if not checked or sorted(got["pinned"]) != sorted(str(x) for x in layers):
            problems.append(f"{sid}: its checks did not all pass")
        if (made.get("state"), made.get("package"), made.get("branches"), made.get("seeds")) != (
                sid, b["package"], ["random"], seeds(run_seed, sid, layers)):  # fmt: skip
            problems.append(f"{sid}: not its planned state, package, branch and seed")
        files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
        if files != random_files(layers):
            problems.append(f"{sid}: not its complete file inventory")
    return problems
