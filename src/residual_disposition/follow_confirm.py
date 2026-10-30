"""The follow-up's confirmations of example_link and pun_intended (docs/preregistration.md §13.3:
§12.4.1 as executed on 2026-10-19), without model code: what is drawn next, from what, and what
the lines' confirmed doses are. ``scripts/pod/follow_confirm.py`` draws on the Pod.

- **Sources, read through their loaders and checked together before any generation**
  (``load_spec``): the follow-up rules (``confirmation``: the prompt set and its sha256, the
  prompt count, the largest passing hit count, the attempts, the ledger's namespace, measurement,
  stream tag, base seed and seed keys; ``partial_threshold``: which lines are confirmed afresh,
  their dose and fallback), §12's rules (the exact test, p₁ ≤ 1/60), B1's dose manifest (each
  line's d₁ and d₂, their latent and sham_s states, the detector version) and M0's runtime model.
  The exact test's largest passing count must be the rules' ``max_hits``; the attempts, prompt
  count and set the §12 rules' and the frozen confirmation set's; the seed keys the ones the
  seed is derived from; the measurement, stream and base seed new, so that no earlier
  confirmation cell, ledger or stream is reused.
- **Ledger** (``next_step``): each line's own, ``follow/<behavior>``, over its d₁ and d₂ with
  the rules' attempts, replayed from its completed cells (``continuation_rules.replay_attempts``).
  Attempt 1 draws d₁. Attempt 2 draws d₂ only after a completed draw of d₁ whose latent state's
  hits over the 600 prompts exceed the passing count; then the line stops. The detector is the
  manifest's: no revision, so no retest.
- **Cells** (``draws``): the experiment's cells of the measurement and the line's ledger. A
  completed cell is accepted only whole (``verified_draw``): the cell of its allocation, both raw
  draws saved with their states, the hits counted again on the saved responses, the pass the
  exact test of L's count; anything else is refused, never skipped. A cell that did not complete
  (a technical failure: the GPU, the engine, the volume, the watcher) is no draw: the attempt it
  allocated stays the next one and resumes as that very cell with its saved draws, once the cell
  and the draws' states are checked (``plan``).
- **Decision** (``decision``): per line "confirmed" at its dose, "not confirmed" (§13.3: p = 1),
  or the next draw; ``confirmed`` maps both lines to their dose or None once both are decided,
  for ``follow_inventory.resolve``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import continuation_confirm as C
from . import continuation_rules as cr

CONFIRMED, NOT_CONFIRMED, DRAW = "confirmed", "not_confirmed", "draw"
# The measurements, streams and base seed of every earlier confirmation (§12.4.1's cells): the
# follow-up's are new (follow_rules.yaml: confirmation)
EARLIER = {
    "measurements": (
        C.MEASUREMENT,
        "stage3_confirmation",
        "stage5_confirmation",
        "stage5_dpo_confirmation",
    ),  # fmt: skip
    "streams": ("confirmation",),
    "base_seeds": (C.FROZEN["base_seed"],),
}
# The generation settings a follow-up cell shares with every recorded confirmation cell (§13.3:
# "as executed")
AS_EXECUTED = ("sampling", "engine", "requests", "prompts_sha256", "generator_seed")


@dataclasses.dataclass(frozen=True)
class Spec:
    """What the confirmations read: §12's rules (``continuation_rules.Rules``), the follow-up
    rules (``Rules``), B1's dose manifest (with its sha256) and M0's configuration."""

    rules: Any
    follow: Any
    doses: Mapping[str, Any]
    m0: Mapping[str, Any]

    @property
    def confirmation(self) -> Mapping[str, Any]:
        return self.follow["confirmation"]

    @property
    def lines(self) -> tuple[str, ...]:
        """The lines confirmed afresh, in the rules' order."""
        return tuple(b for b, s in self.follow["partial_threshold"].items()
                     if s["confirmation"] == "fresh")  # fmt: skip

    def line(self, behavior: str) -> Mapping[str, Any]:
        return self.doses["lineages"][behavior]

    def doses_of(self, behavior: str) -> tuple[int, int]:
        line = self.line(behavior)
        return int(line["d1"]), int(line["d2"])

    def states(self, behavior: str, dose: int) -> tuple[str, str]:
        found = self.line(behavior)["doses"][str(int(dose))]
        return found["latent"], found["sham_s"]

    def lineage(self, behavior: str) -> str:
        return self.confirmation["lineage"].format(behavior=behavior)

    @property
    def max_hits(self) -> int:
        return int(self.confirmation["max_hits"])

    @property
    def prompts(self) -> int:
        return int(self.confirmation["prompts"])

    @property
    def measurement(self) -> str:
        return str(self.confirmation["measurement"])

    @property
    def stream(self) -> str:
        return str(self.confirmation["stream_tag"])

    @property
    def base_seed(self) -> int:
        return int(self.confirmation["base_seed"])

    @property
    def attempts(self) -> int:
        return int(self.confirmation["attempts"])

    def experiment(self, behavior: str) -> str:
        """The line's experiment config, beside M0's (``experiments/continuation_14b/<b>``)."""
        return str(Path(self.m0["runtime"]["experiment"]).parent / behavior)


def largest_passing(rules: Any, n: int) -> int:
    """The largest hit count of ``n`` prompts that §12's exact test passes (-1 if none)."""
    found = -1
    for hits in range(n + 1):
        if not C.outcome(hits, n, rules)["passed"]:
            break
        found = hits
    return found


def spec_problems(spec: Spec) -> list[str]:
    """Why the sources disagree: the follow-up rules' confirmation against §12's exact test and
    the frozen confirmation set, the seed keys, new namespaces, and B1's manifest."""
    c, problems = spec.confirmation, []

    def want(label: str, got: Any, wanted: Any) -> None:
        if got != wanted:
            problems.append(f"{label}: {got!r}, not {wanted!r}")

    want("prompt set", c["prompt_set"], C.FROZEN["pool"])
    want("prompt set sha256", c["prompt_set_sha256"], C.FROZEN["sha256"])
    want("prompts", spec.prompts, int(C.FROZEN["prompts"]))
    want("§12's prompts", int(spec.rules["observer"]["prompts"]), spec.prompts)
    want("attempts", spec.attempts, int(spec.rules.attempts))
    n = int(spec.rules["observer"]["prompts"])
    want("the exact test's largest passing count", largest_passing(spec.rules, n), spec.max_hits)
    want("seed keys", tuple(c["seed_keys"]), cr.CONFIRMATION_SEED_KEYS)
    if "{behavior}" not in str(c["lineage"]):
        problems.append(f"lineage {c['lineage']!r} does not name the behavior")
    if spec.measurement in EARLIER["measurements"]:
        problems.append(f"measurement {spec.measurement} is an earlier confirmation's")
    if spec.stream in EARLIER["streams"]:
        problems.append(f"stream {spec.stream} is an earlier confirmation's")
    if spec.base_seed in EARLIER["base_seeds"]:
        problems.append(f"base seed {spec.base_seed} is an earlier ledger's")
    want("the doses' §12 rules", spec.doses.get("rules_sha256"), spec.rules.sha256)
    want("the doses' lines", sorted(spec.doses["lineages"]), sorted(spec.lines))
    for b in spec.lines:
        d1, d2 = spec.doses_of(b)
        rule = spec.follow["partial_threshold"][b]
        want(f"{b} doses", (d1, d2), (int(rule["dose"]), int(rule["fallback"])))
    return problems


def load_spec(*, follow: Any = None, rules: Any = None, doses: Mapping[str, Any] | None = None,
              m0: Mapping[str, Any] | None = None) -> Spec:  # fmt: skip
    """The sources (by default each through its frozen loader: ``follow_rules``, ``load_rules``
    at the frozen hash, ``follow_doses``, ``follow_m0``), refused unless they agree."""
    from .simulate_continuation import FROZEN

    follow = cr.follow_rules() if follow is None else follow
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"]) if rules is None else rules
    doses = cr.follow_doses() if doses is None else doses
    m0 = cr.follow_m0() if m0 is None else m0
    spec = Spec(rules=rules, follow=follow, doses=doses, m0=m0)
    problems = spec_problems(spec)
    if problems:
        raise ValueError("the confirmation's sources disagree: " + "; ".join(problems))
    return spec


def workspace_problems(spec: Spec, behavior: str, *, experiment: str, model: Mapping[str, Any],
                       detector_version: str) -> list[str]:  # fmt: skip
    """Why a line's workspace is not the one its draws need: the experiment, M0's model (name,
    revision, dtype) and the manifest's detector version."""
    problems = []
    if Path(spec.experiment(behavior)).parts[-2] != experiment:
        problems.append(f"experiment {experiment}, not {spec.experiment(behavior)}'s")
    want = spec.m0["runtime"]["model"]
    for key in ("name", "revision", "dtype"):
        if model.get(key) != want[key]:
            problems.append(f"model {key} {model.get(key)!r}, not {want[key]!r}")
    if detector_version != spec.line(behavior)["detector_version"]:
        problems.append(f"{behavior}: detector {detector_version}, the manifest's "
                        f"{spec.line(behavior)['detector_version']}")  # fmt: skip
    return problems


def cells(runs: Path, experiment: str, step: str) -> list[tuple[Path, dict[str, Any], dict]]:
    """Every run of ``step`` in ``experiment`` under ``runs``: (directory, config, meta)."""
    import yaml

    out = []
    for config in sorted((Path(runs) / experiment).glob(f"*-{step}-*/config.yaml")):
        meta = json.loads((config.parent / "meta.json").read_text())
        out.append((config.parent, yaml.safe_load(config.read_text()) or {}, meta))
    return out


FIELDS = ("attempt", "dose", "purpose", "seed", "detector_version", "lineage", "base_seed")


def generation(ws: Any) -> tuple[list[Any], Any, list[Any]]:
    """The frozen prompts from the workspace's data root, the behavior's sampling at one sample
    and the requests."""
    from dataclasses import replace

    from .generation import user_request
    from .paths import data_root

    prompts = C.prompts(data_root())
    sampling = replace(ws.sampling("behavior"), samples=1)
    return prompts, sampling, [user_request(q.prompt_id, q.text) for q in prompts]


def cell_of(ws: Any, spec: Spec, allocation: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """The cell of an allocated attempt (``FIELDS`` with its latent and sham_s states): its
    params, the allocation with its whole generation spec and the sources' hashes, and the cell
    (``pipeline.make_cell``)."""
    from omegaconf import OmegaConf

    from . import pipeline as P

    _, sampling, requests = generation(ws)
    params = {
        **{k: allocation[k] for k in FIELDS},
        "latent": allocation["latent"],
        "sham_s": allocation["sham_s"],
        "measurement": spec.measurement,
        "stream": spec.stream,
        "doses_sha256": spec.doses["sha256"],
        "follow_rules_sha256": spec.follow.sha256,
        "rules_sha256": spec.rules.sha256,
        "prompts_sha256": C.FROZEN["sha256"],
        "generator_seed": int(OmegaConf.select(ws.cfg, "pipeline.vllm.seed", default=0)),
        **ws.generation_params(sampling, requests),
    }
    return params, P.make_cell(ws, spec.measurement, params)


def identity_problems(run: Path, config: Mapping[str, Any], meta: Mapping[str, Any],
                      digest: str) -> list[str]:  # fmt: skip
    """Why ``run`` is not the cell ``digest``: its saved configuration must hash to its recorded
    hash, and that be the cell's."""
    from omegaconf import OmegaConf

    from .config import config_hash

    saved = config_hash(OmegaConf.create(dict(config)))
    if saved == meta.get("config_hash") == digest:
        return []
    return [f"{run.name}: its configuration hashes to {saved}, recorded {meta.get('config_hash')}"
            f", the cell is {digest}"]  # fmt: skip


def saved_draw_problems(run: Path, params: Mapping[str, Any], keys: Sequence[str],
                        *, complete: bool) -> list[str]:  # fmt: skip
    """Why ``run``'s saved draws are not its states': each saved draw one completion of every
    prompt (``continuation_confirm.load_draw``), its metadata naming its role and the cell's
    state; with ``complete``, both draws must be saved."""
    problems = []
    for role in C.ROLES:
        path = run / "draws" / f"{role}.jsonl"
        try:
            found = C.load_draw(path, keys)
            if found is None:
                if complete:
                    problems.append(f"{run.name}: no saved draw of {role}")
                continue
            meta = C.load_draw_meta(path)
        except (OSError, ValueError, TypeError) as error:
            problems.append(f"{run.name}: {role}'s draw: {error}")
            continue
        if (meta.get("role"), meta.get("state")) != (role, params.get(role)):
            problems.append(f"{run.name}: the draw of {role} names {meta.get('role')} "
                            f"{meta.get('state')}, the cell {params.get(role)}")  # fmt: skip
    return problems


def verified_draw(ws: Any, spec: Spec, behavior: str, run: Path, config: Mapping[str, Any],
                  meta: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """A completed cell's draw, accepted only whole: the run is the cell of its allocation
    (``identity_problems``: the allocation, states, generation spec and sources' hashes), at d₁
    or d₂ with the manifest's states; its summary repeats the allocation; both raw draws are
    saved whole with their states; the hits are counted again on the saved responses with the
    behavior's detector and must be the summary's; and the exact test of L's count gives its
    pass. Anything else is refused."""
    params = dict(config.get("params") or {})
    name, (d1, d2) = spec.lineage(behavior), spec.doses_of(behavior)
    dose = params.get("dose")
    if dose not in (d1, d2) or (params.get("latent"), params.get("sham_s")) != spec.states(
            behavior, dose):  # fmt: skip
        raise ValueError(f"{run.name}: not a draw of {behavior}'s states at d₁ or d₂")
    if (params.get("lineage"), params.get("base_seed"), params.get("detector_version")) != (
            name, spec.base_seed, spec.line(behavior)["detector_version"]):  # fmt: skip
        raise ValueError(f"{run.name}: not a draw of {name}'s ledger and detector")
    problems = identity_problems(run, config, meta, cell_of(ws, spec, params)[1].digest)
    summary = json.loads((run / "summary.json").read_text())
    differ = [k for k in FIELDS if summary.get(k) != params.get(k)]
    if differ:
        problems.append(f"{run.name}: its summary's {', '.join(differ)} are not its cell's")
    problems += saved_draw_problems(run, params, keys, complete=True)
    if problems:
        raise ValueError("; ".join(problems))
    counts = {}
    for role in C.ROLES:
        completions = C.load_draw(run / "draws" / f"{role}.jsonl", keys) or []
        counts[role] = {"hits": sum(bool(ws.behavior.detector(c.text)) for c in completions),
                        "n": len(completions)}  # fmt: skip
        if summary.get(role) != counts[role]:
            raise ValueError(f"{run.name}: {role}'s responses give {counts[role]}, its summary "
                             f"{summary.get(role)}")  # fmt: skip
    test = C.outcome(counts["latent"]["hits"], counts["latent"]["n"], spec.rules)
    if test["passed"] != summary.get("passed") or test["passed"] != (
            test["hits"] <= spec.max_hits):  # fmt: skip
        raise ValueError(f"{run.name}: passed {summary.get('passed')}, the test of "
                         f"{test['hits']}/{test['n']} gives {test['passed']}")  # fmt: skip
    return {**{k: params[k] for k in FIELDS}, "passed": test["passed"], "hits": test["hits"],
            "n": test["n"]}  # fmt: skip


def draws(spec: Spec, behavior: str, runs: Path, experiment: str, ws: Any
          ) -> tuple[list[dict[str, Any]], list[tuple[Path, dict, dict]]]:  # fmt: skip
    """The line's completed draws in attempt order, each verified whole (``verified_draw``:
    {attempt, dose, purpose, seed, detector_version, lineage, base_seed, passed, hits, n}), and
    its runs that did not complete (none superseded by a completed run of the same cell). ``ws``
    is the line's workspace (its cells, detector and prompts). A completed cell that does not
    verify is refused, never skipped."""
    name = spec.lineage(behavior)
    keys = [q.prompt_id for q in generation(ws)[0]]
    found, open_runs, done = [], [], set()
    for run, config, meta in cells(runs, experiment, spec.measurement):
        params = config.get("params") or {}
        if params.get("lineage") != name:
            continue
        if (config.get("behavior") or {}).get("id") != behavior:
            raise ValueError(f"{run.name}: a cell of {name} for another behavior")
        if meta.get("status") != "completed":
            open_runs.append((run, config, meta))
            continue
        done.add(meta.get("config_hash"))
        found.append(verified_draw(ws, spec, behavior, run, config, meta, keys))
    found.sort(key=lambda d: int(d["attempt"]))
    open_runs = [r for r in open_runs if r[2].get("config_hash") not in done]
    return found, open_runs


def next_step(spec: Spec, behavior: str, found: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The line's status from its completed draws: {"status": "confirmed", "dose"}, {"status":
    "not_confirmed", "dose": None}, or {"status": "draw", "dose", ...the allocated attempt, its
    states}."""
    d1, d2 = spec.doses_of(behavior)
    version, name = spec.line(behavior)["detector_version"], spec.lineage(behavior)
    ledger = cr.replay_attempts([d1, d2], list(found), version, lineage=name,
                                base_seed=spec.base_seed, attempts=spec.attempts)  # fmt: skip
    if ledger.retest_dose is not None:
        raise ValueError(f"{name}: a detector revision, which the follow-up's lines do not have")
    attempts = [dict(d) for d in found]
    dose = ledger.analysis_dose()
    if dose is not None:
        return {"status": CONFIRMED, "dose": dose, "attempts": attempts}
    if not ledger.log:
        dose, purpose = d1, None
    else:
        last = found[-1]
        if int(last["n"]) != spec.prompts or int(last["hits"]) <= spec.max_hits:
            raise ValueError(f"{name}: attempt {last['attempt']} failed without more than "
                             f"{spec.max_hits} hits of {spec.prompts}")  # fmt: skip
        if len(ledger.log) >= spec.attempts or int(last["dose"]) != d1 or ledger.next_dose() != d2:
            return {"status": NOT_CONFIRMED, "dose": None, "attempts": attempts}
        dose, purpose = d2, cr.NEXT_DOSE
    allocated = C.allocate([d1, d2], list(found), spec.rules, name=name, detector_version=version,
                           dose=dose, base_seed=spec.base_seed,
                           attempts=spec.attempts)  # fmt: skip
    if allocated["purpose"] != purpose:
        raise ValueError(f"{name}: attempt {allocated['attempt']} is {allocated['purpose']}")
    latent, sham_s = spec.states(behavior, dose)
    return {"status": DRAW, **allocated, "latent": latent, "sham_s": sham_s,
            "attempts": attempts}  # fmt: skip


def decision(spec: Spec, runs: Path, experiment: str, make: Any) -> dict[str, Any]:
    """Every line's ``next_step`` from the cells under ``runs`` (``make(spec, behavior)``: the
    line's workspace), and ``confirmed`` ({line: dose or None}) once every line is decided (None
    before)."""
    lines = {b: next_step(spec, b, draws(spec, b, runs, experiment, make(spec, b))[0])
             for b in spec.lines}  # fmt: skip
    decided = all(s["status"] != DRAW for s in lines.values())
    confirmed = {b: s["dose"] for b, s in lines.items()} if decided else None
    return {"lines": lines, "confirmed": confirmed}


Recorded = Sequence[tuple[str, Mapping[str, Any]]]  # (run name, params) of earlier cells


def as_executed_problems(params: Mapping[str, Any], recorded: Recorded) -> list[str]:
    """Why a cell's generation is not §12.4.1's as executed: it must share ``AS_EXECUTED`` with
    every recorded confirmation cell ``recorded`` ((run name, params)), of which there is one at
    least."""
    if not recorded:
        return ["no recorded confirmation cell to compare the generation with"]
    problems = []
    for name, other in recorded:
        keys = [k for k in AS_EXECUTED if other.get(k) != params.get(k)]
        if keys:
            problems.append(f"{name}: {', '.join(keys)} differ")
    return problems


def recorded_cells(runs: Path, experiment: str) -> list[tuple[str, dict[str, Any]]]:
    """The completed confirmation cells of the earlier measurements under ``runs``: (run name,
    params)."""
    out = []
    for step in EARLIER["measurements"]:
        for run, config, meta in cells(runs, experiment, step):
            if meta.get("status") == "completed":
                out.append((run.name, config.get("params") or {}))
    return out


def plan(ws: Any, spec: Spec, behavior: str, step: Mapping[str, Any],
         open_runs: Sequence[tuple[Path, dict, dict]],
         recorded: Recorded) -> dict[str, Any]:  # fmt: skip
    """The draw of ``step`` (``next_step``'s "draw") as ``pipeline._draw_confirmation`` takes it:
    the frozen prompts, the behavior's sampling at one sample, the cell (``cell_of``) and the
    line's open runs. Each open run must be this very cell (``identity_problems``,
    ``continuation_confirm.open_allocation_problems``) and every draw it saved one of every
    prompt with its role and the cell's state, before it is reused. Everything is checked before
    any generation."""
    if step.get("status") != DRAW:
        raise ValueError(f"{behavior}: nothing to draw ({step.get('status')})")
    problems = workspace_problems(spec, behavior, experiment=str(ws.cfg.experiment),
                                  model=ws.model_cfg,
                                  detector_version=ws.behavior.detector.version)  # fmt: skip
    if ws.behavior.id != behavior:
        problems.append(f"the workspace's behavior {ws.behavior.id}, not {behavior}")
    prompts, sampling, requests = generation(ws)
    params, cell = cell_of(ws, spec, step)
    problems += as_executed_problems(params, recorded)
    problems += C.open_allocation_problems(
        cell.digest, params,
        [(run.name, meta.get("config_hash"), config.get("params") or {})
         for run, config, meta in open_runs],
    )  # fmt: skip
    keys = [q.prompt_id for q in prompts]
    for run, config, meta in open_runs:
        problems += identity_problems(run, config, meta, cell.digest)
        problems += saved_draw_problems(run, params, keys, complete=False)
    if problems:
        raise ValueError(f"{behavior}: " + "; ".join(problems))
    return {"rules": spec.rules, "allocated": {k: step[k] for k in FIELDS},
            "latent": step["latent"], "sham_s": step["sham_s"], "sampling": sampling,
            "requests": requests, "params": params, "cell": cell,
            "interrupted": [run for run, _, _ in open_runs]}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# The Pod's inputs (``scripts/pod/follow_confirm.py precheck``), checked in the space it writes


KINDS = {"latent": "suppressed", "sham_s": "suppressed_sham"}  # the analyst index's kinds
# The project's network volume, known by a file it has held since M0's session (its result, by
# sha256): ``pod_fitness.mount_problems``
VOLUME_MARKS = {"b/follow/m0/follow-m0-20261029T202118Z/result.json":
                "451c3e768324e7e667d7a4f80bdae65e8af6548887a93ab23c2d32a3fcd2cf01"}  # fmt: skip


def states_needed(spec: Spec) -> list[tuple[str, int, str, str]]:
    """(behavior, dose, role, state) of every state a line may draw: latent and sham_s at d₁ and
    d₂."""
    out = []
    for b in spec.lines:
        for dose in spec.doses_of(b):
            latent, sham_s = spec.states(b, dose)
            out += [(b, dose, "latent", latent), (b, dose, "sham_s", sham_s)]
    return out


def index_records(runs: Path) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Every analyst-index record under ``runs`` (``*/*/observations/state_index.jsonl``), by
    state: (its run, the record)."""
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for table in sorted(Path(runs).glob("*/*/observations/state_index.jsonl")):
        run = f"{table.parent.parent.parent.name}/{table.parent.parent.name}"
        for line in table.read_text().splitlines():
            record = json.loads(line)
            out.setdefault(record["state_id"], []).append((run, record))
    return out


def index_problems(spec: Spec, runs: Path) -> tuple[list[str], list[str]]:
    """Why the space's analyst index does not record every needed state as its line's stack
    suppression (latent) or its sham at its dose; and the states it does not hold at all."""
    found, problems, missing = index_records(runs), [], []
    for b, dose, role, state in states_needed(spec):
        records = found.get(state)
        if not records:
            missing.append(state)
            continue
        want = {"behavior_id": b, "kind": KINDS[role], "dose": dose, "mode": "stack"}
        for run, record in records:
            got = {k: record.get(k) for k in want}
            if got != want:
                problems.append(f"{state} in {run}: {got}, not {want}")
    return problems, missing


def _git_blob_sha1(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_snapshot(hf_home: Path, name: str, revision: str) -> Path:
    return Path(hf_home) / "hub" / f"models--{name.replace('/', '--')}" / "snapshots" / revision


def model_blob(entry: Path) -> Path:
    """The blob a snapshot entry names: its link's own target, one link followed. The hub may
    link that blob on to a shared store (``hub/blobs/<xx>/<another hash>``), so the last name in
    the chain is not the blob's content address; the content is read through ``entry``."""
    import os

    if not entry.is_symlink():
        return entry
    return Path(os.path.normpath(entry.parent / os.readlink(entry)))


def model_cache_problems(hf_home: Path, name: str, revision: str) -> list[str]:
    """Why ``hf_home`` does not hold the model at ``revision`` whole: its snapshot with the
    configuration, the tokenizer and every weight shard its index names, each entry a link to a
    blob in the model's ``blobs`` whose name is its content's address (sha256 for a large file,
    the git blob sha1 for a small one)."""
    import os

    snapshot = model_snapshot(hf_home, name, revision)
    if not snapshot.is_dir():
        return [f"{snapshot}: missing"]
    problems = []
    index = snapshot / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else set()
    needed = {"config.json", "tokenizer.json", "tokenizer_config.json", index.name, *shards}
    held = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if not p.is_dir()}
    problems += [f"{snapshot}: {name} missing" for name in sorted(needed - held)]
    blobs = Path(os.path.normpath(snapshot.parent.parent / "blobs"))
    for rel in sorted(held):
        entry = snapshot / rel
        blob = model_blob(entry)
        etag = blob.name
        if blob.parent != blobs:
            problems.append(f"{entry}: not a link to a blob of the model ({blob})")
            continue
        if len(etag) == 64:
            ok = file_sha256(entry) == etag
        elif len(etag) == 40:
            ok = _git_blob_sha1(entry.read_bytes()) == etag
        else:
            ok = False
        if not ok:
            problems.append(f"{entry}: its content is not its blob {etag}")
    return problems


# Room on the volume the session needs beyond what the precheck copies: the served adapters it
# concatenates (DATA_DIR/cache/served), its draws, records and responses
SESSION_RESERVE_BYTES = 20 * 2**30


def tree_bytes(path: Path) -> int:
    """The bytes of the files under ``path`` (a link counted as its target), 0 if absent."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def model_bytes_to_copy(source: Path, target: Path, name: str, revision: str) -> int:
    """The bytes of the source snapshot's blobs that ``target`` does not hold yet."""
    blobs = model_snapshot(target, name, revision).parent.parent / "blobs"
    entries = [p for p in model_snapshot(source, name, revision).rglob("*") if not p.is_dir()]
    found = {model_blob(p).name: p.stat().st_size for p in entries}
    return sum(size for etag, size in found.items() if not (blobs / etag).exists())


def room_problems(free: int, needed: int, reserve: int = SESSION_RESERVE_BYTES) -> list[str]:
    """Why the volume has not the room for the copies (``needed``) and the session (``reserve``):
    checked before any copy."""
    if free >= needed + reserve:
        return []
    gib = 2**30
    return [f"the volume has {free / gib:.1f} GiB free, the copies need {needed / gib:.1f} GiB "
            f"and the session {reserve / gib:.1f} GiB"]  # fmt: skip


def copy_model(source: Path, target: Path, name: str, revision: str) -> list[dict[str, Any]]:
    """Copy the model's snapshot at ``revision`` from the HF home ``source`` into ``target``,
    create-only: each blob checked against its content address after the copy, and each
    snapshot entry a link to its blob, as the hub lays them out. An existing blob or link that
    differs is refused; nothing in ``target`` is overwritten. Returns the copied files."""
    import os
    import shutil

    problems = model_cache_problems(source, name, revision)
    if problems:
        raise ValueError(f"the source's model is not whole: {problems[0]}")
    src, dst = model_snapshot(source, name, revision), model_snapshot(target, name, revision)
    blobs = dst.parent.parent / "blobs"
    copied = []
    for path in sorted(p for p in src.rglob("*") if not p.is_dir()):
        rel = path.relative_to(src)
        etag = model_blob(path).name
        out = blobs / etag
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            partial = out.with_name(out.name + ".partial")
            shutil.copyfile(path, partial)  # the content, through every link
            partial.replace(out)
            copied.append({"file": rel.as_posix(), "blob": etag, "bytes": out.stat().st_size})
        link = dst / rel
        target_link = os.path.relpath(out, link.parent)
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or os.readlink(link) != target_link:
                raise ValueError(f"{link} exists and is not the link to {out.name}")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target_link)
    problems = model_cache_problems(target, name, revision)
    if problems:
        raise ValueError(f"the copied model is not whole: {problems[0]}")
    return copied
