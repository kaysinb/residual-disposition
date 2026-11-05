"""The size block's model work, one implantation at a time (docs/preregistration.md §13.9 as
amended on 2026-11-05; ``size_block`` holds the rules and arithmetic, ``scripts/pod/size_block.py``
runs the phases). Every step is a cell of the implantation's experiment (``size_<name>``), so an
interrupted phase resumes from its completed cells.

- ``inputs``: the pinned 14B datasets (``manifests/size_inputs.json``) read by id and checked:
  ids, the matched prompts and examples hashes of the anchor's organisms, the suppression data's
  hash and its zero detector hits, the relearning data's hashes. Nothing is rebuilt.
- ``preflight``: the model and tokenizer at the pinned revision, one optimizer step of the
  implant recipe and one generation with the trained adapter served, timed.
- ``implant``: A and H (tldr), the Y and Y₂ organisms, each with the implant recipe at the
  implantation's seed on the anchor's data; their reference passes (the 200 relearning-evaluation
  prompts, one response each, the implantation's own stream); the floors.
- ``suppress``: L from A and H_s from H in stack mode with the stack lineage's frozen recipe,
  checkpoints 25–800; every checkpoint's exploratory pass (600 prompts) and paired excess; the
  margin candidates (§12.4.1).
- ``confirm``: one draw of 600 fresh confirmation prompts for L and H_s at the allocated dose
  and attempt; the gate is at most four L hits. Each state's draw is saved before the next is
  drawn and reused, never drawn again, when an interrupted run of the same cell resumes.

Every arm (b, Y, Y₂) shares one base model and one engine (``pipeline._ArmModels``).
- ``calibrate``: H_s relearned on b, Y and Y₂ at both rates, seeds 10–12 of the namespace, 600
  steps, the common schedule; k*, B and the dense decision (``size_block.calibrate``).
- ``blocks``: the pilot's or the certificate's runs, each with its block's seed, the rate's k*,
  B, and the dense schedule's extra evaluations when the calibration set it.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import size_block as S
from .paths import REPO_ROOT, runs_root

CONFIGS = {"4B-1": "4b_1", "4B-2": "4b_2", "14B-2": "14b_2"}
MODELS = {"qwen3-4b": "Qwen/Qwen3-4B", "qwen3-14b": "Qwen/Qwen3-14B"}
INPUTS = REPO_ROOT / "manifests" / "size_inputs.json"
# B6: manifests/size_inputs.json's sha256
INPUTS_FROZEN = "3ad9fa4812dda91cb90ded1052b67b7f5c674377805b7aaa91832aadea069f9e"
STACK = {"mode": "stack", "method": "sft"}
REFERENCE_TAG = "size_reference"


# The committed records a phase starts from (in the snapshot it runs from, never a volume-only
# file): per implantation ``<config>/calibrate.json`` (the calibration record, as the Pod wrote it),
# ``<config>/detector-part1.json`` (4B-1) and ``<config>/detector-part2-<dose>.json`` (the
# detector check's reports; one with disagreements counts only with its recorded acceptance,
# ``<stem>-decision.json``); and ``branch.json``, the branch taken after 4B-1's pilot.
# Each is read only when committed unchanged (``manifests.committed_json``: tracked and clean in a
# git checkout, by the snapshot's receipt on a Pod); a copied, uncommitted file is refused.
COMMITTED = REPO_ROOT / "manifests" / "size"
BRANCHES = {"full": ("4B-1", "4B-2", "14B-2"), "4B-1+14B-2": ("4B-1", "14B-2"), "stop": ()}
PROVENANCE = ("sha256", "committed_at")  # committed_json's fields, not the record's
ACCEPTED = "accepted unchanged"  # a detector check's recorded acceptance (§13.17)


def _committed_record(path: Path, what: str, repo: Path | None) -> tuple[dict | None, str | None]:
    """The record at ``path`` without committed_json's provenance fields, and its file's sha256;
    (None, None) when absent, ValueError when present but not committed unchanged."""
    from .manifests import committed_json

    if not path.is_file():
        return None, None
    try:
        data = committed_json(path, what, repo)
    except RuntimeError as e:
        raise ValueError(str(e)) from e
    return {k: v for k, v in data.items() if k not in PROVENANCE}, data["sha256"]


def _committed_file(path: Path, what: str, repo: Path | None) -> dict | None:
    """The record at ``path`` without committed_json's provenance fields; None when absent,
    ValueError when present but not committed unchanged."""
    return _committed_record(path, what, repo)[0]


def committed(name: str, key: str, root: Path = COMMITTED,
              repo: Path | None = None) -> dict | None:  # fmt: skip
    path = Path(root) / CONFIGS[name] / f"{key}.json"
    return _committed_file(path, f"{name}'s {key}", repo)


def branch(root: Path = COMMITTED, repo: Path | None = None) -> tuple[str, ...] | None:
    """The implantations the recorded post-pilot branch continues; None before it is recorded."""
    path = Path(root) / "branch.json"
    record = _committed_file(path, "the post-pilot branch", repo)
    if record is None:
        return None
    chosen = record.get("branch")
    if chosen not in BRANCHES:
        raise ValueError(f"{path}: branch {chosen!r}, not one of {sorted(BRANCHES)}")
    return BRANCHES[chosen]


def branch_problems(name: str, phase: str, purpose: str | None, root: Path = COMMITTED,
                    repo: Path | None = None) -> list[str]:  # fmt: skip
    """Why ``phase`` of ``name`` waits for the post-pilot branch (§13.17): every phase of 4B-2,
    14B-2's phases after its calibration and every certificate's blocks. Alongside the pilot only
    14B-2's preparation through its calibration runs."""
    needs = (
        name == "4B-2"
        or (phase == "blocks" and purpose == "certificate")
        or (name == "14B-2" and phase in ("dense-check", "blocks"))
    )
    if not needs:
        return []
    try:
        chosen = branch(root, repo)
    except ValueError as e:
        return [f"{name} {phase}: {e}"]
    if chosen is None:
        return [f"{name} {phase}: no recorded branch after 4B-1's pilot ({root}/branch.json)"]
    if name not in chosen:
        return [f"{name} {phase}: the recorded branch does not continue {name}"]
    return []


def _report_problems(name: str, key: str, root: Path, repo: Path | None, what: str, *,
                     implantation: str, part: int, dose: int | None,
                     versions: Mapping[str, str]) -> list[str]:  # fmt: skip
    try:
        report, sha = _committed_record(Path(root) / CONFIGS[name] / f"{key}.json",
                                        f"{name}'s {key}", repo)  # fmt: skip
    except ValueError as e:
        return [f"{what}: {e}"]
    if report is None:
        return [f"no committed {what}"]
    problems = []
    if (report.get("implantation"), report.get("part")) != (implantation, part):
        problems.append(f"{what} is of {report.get('implantation')} part {report.get('part')}")
    if dose is not None and report.get("dose") != dose:
        problems.append(f"{what} is at dose {report.get('dose')}, not {dose}")
    if report.get("verdict") == "review":
        problems += acceptance_problems(name, key, report, sha, root, repo, what)
    elif report.get("verdict") != "passes":
        problems.append(f"{what}: verdict {report.get('verdict')!r}, not passes")
    got = dict(report.get("detectors") or {})
    for behavior, version in versions.items():
        if got.get(behavior) != version:
            problems.append(f"{what}: {behavior}'s detector {got.get(behavior)}, not {version}")
    return problems


def acceptance_problems(name: str, key: str, report: Mapping[str, Any], report_sha256: str,
                        root: Path = COMMITTED, repo: Path | None = None,
                        what: str = "the detector check") -> list[str]:  # fmt: skip
    """Why a committed report with disagreements does not count (§13.17): it counts only with its
    committed acceptance ``<key>-decision.json``, the decision "accepted unchanged" naming that
    report by sha256, exactly its disagreements (item and kind), each with its reason, its
    detector versions and the section that records it."""
    try:
        decision = committed(name, f"{key}-decision", root, repo)
    except ValueError as e:
        return [f"{what}'s acceptance: {e}"]
    if decision is None:
        return [f"{what}: verdict 'review' with no committed acceptance ({key}-decision.json)"]
    problems = []
    if decision.get("decision") != ACCEPTED:
        problems.append(f"{what}: decision {decision.get('decision')!r}, not {ACCEPTED!r}")
    named = str(decision.get("report_sha256"))
    if named != report_sha256:
        problems.append(f"{what}: the acceptance names report {named[:12]}, not the committed "
                        f"report {report_sha256[:12]}")  # fmt: skip
    rows = decision.get("disagreements") or []
    want = sorted((d["item_id"], d["kind"]) for d in report.get("disagreements") or [])
    got = sorted((d.get("item_id"), d.get("kind")) for d in rows)
    if got != want:
        problems.append(f"{what}: the acceptance covers {got}, not exactly the report's {want}")
    if any(not str(d.get("reason") or "").strip() for d in rows):
        problems.append(f"{what}: an accepted disagreement without its reason")
    if dict(decision.get("detectors") or {}) != dict(report.get("detectors") or {}):
        problems.append(f"{what}: the acceptance's detector versions are not the report's")
    if not str(decision.get("section") or "").strip():
        problems.append(f"{what}: the acceptance names no section")
    return problems


def detector_problems(name: str, dose: int, versions: Mapping[str, str],
                      root: Path = COMMITTED, repo: Path | None = None) -> list[str]:  # fmt: skip
    """Why ``name``'s confirmation at ``dose`` may not start: its detector check's committed
    passing (or accepted) report of part 2 at that dose (tldr's current detector) and, for a 4B
    implantation, 4B-1's of part 1 (every arm's current detector). 14B-2 uses the accepted 14B
    detector."""
    tldr = {k: v for k, v in versions.items() if k == "tldr"}
    problems = _report_problems(name, f"detector-part2-{dose}", root, repo,
                                f"{name}'s detector check part 2 at {dose}", implantation=name,
                                part=2, dose=dose, versions=tldr)  # fmt: skip
    if name.startswith("4B"):
        problems += _report_problems("4B-1", "detector-part1", root, repo,
                                     "4B-1's detector check part 1", implantation="4B-1",
                                     part=1, dose=None, versions=versions)  # fmt: skip
    return problems


def committed_calibration(name: str, volume: Mapping[str, Any] | None, root: Path = COMMITTED,
                          repo: Path | None = None) -> dict:  # fmt: skip
    """The committed calibration record, refused when absent, not committed unchanged, or not
    the volume's record (compared by content, without the commit check's provenance fields)."""
    got = committed(name, "calibrate", root, repo)
    if got is None:
        raise ValueError(
            f"{name}: no committed calibration ({root}/{CONFIGS[name]}/calibrate.json)"
        )
    if volume is not None:
        record = json.loads(json.dumps(dict(volume), default=str))
        if {k: v for k, v in record.items() if k not in PROVENANCE} != got:
            raise ValueError(f"{name}: the committed calibration is not the volume's record")
    return got


def load_inputs(path: Path = INPUTS, *, expected_hash: str | None = INPUTS_FROZEN) -> dict:
    """The pinned inputs, refused if the manifest is not the frozen one (B6)."""
    import hashlib

    blob = Path(path).read_bytes()
    if expected_hash is not None and hashlib.sha256(blob).hexdigest() != expected_hash:
        raise ValueError(f"{path}: not the frozen size inputs (B6)")
    return json.loads(blob)


def workspace(name: str, rules: Mapping[str, Any]) -> Any:
    """The implantation's workspace (``configs/experiments/size_block/<name>``), checked: its
    model is the rules', its generation seed the namespace offset, no other experiment reused."""
    from . import pipeline as P
    from .config import load_config

    if name not in CONFIGS:
        raise ValueError(f"{name} is not a new implantation ({sorted(CONFIGS)})")
    ws = P.Workspace(load_config(f"experiments/size_block/{CONFIGS[name]}",
                                 ["tracking.wandb.enabled=false"]))  # fmt: skip
    want = MODELS[rules["implantations"][name]["model"]]
    if ws.model_cfg["name"] != want:
        raise ValueError(f"{name}: model {ws.model_cfg['name']}, not {want}")
    if int(ws.cfg.pipeline.vllm.seed) != S.offset(rules, name):
        raise ValueError(f"{name}: generation seed {ws.cfg.pipeline.vllm.seed}, not its offset")
    if list(ws.cfg.pipeline.get("reuse_from") or []):
        raise ValueError(f"{name}: reuses another experiment's cells")
    if ws.behavior.id != rules["behavior"]:
        raise ValueError(f"{name}: behavior {ws.behavior.id}, not {rules['behavior']}")
    return ws


def arms(ws: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The arms' workspaces: b this one, Y and Y₂ their behaviors' on the same model and seed."""
    from . import pipeline as P

    return {h: ws if h == "b" else P.arm_workspace(ws, inputs["habits"][h]["behavior"])
            for h in S.HABITS}  # fmt: skip


# ---------------------------------------------------------------- inputs


def datasets(ws: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The pinned datasets, checked against the anchor's records (see the module)."""
    from . import pipeline as P
    from .datagen import load_dataset, restrict

    def load(data_id: str) -> Any:
        if not (ws.generations / data_id / "manifest.json").exists():
            raise ValueError(f"{data_id}: the pinned dataset is not under {ws.generations}")
        d = load_dataset(ws.generations / data_id)
        if d.data_id != data_id:
            raise ValueError(f"{data_id}: loaded as {d.data_id}")
        return d

    a = arms(ws, inputs)
    out: dict[str, Any] = {"habits": {}, "relearn": {}}
    for habit, h in inputs["habits"].items():
        if a[habit].behavior.id != h["behavior"]:
            raise ValueError(f"arm {habit} is {a[habit].behavior.id}, not {h['behavior']}")
        implant, sham = load(h["implant_data"]), load(h["sham_data"])
        kept = {e.prompt_id for e in sham.examples}
        matched = [e.prompt_id for e in implant.examples if e.prompt_id in kept]
        for role, data in (("active", implant), ("sham", sham)):
            if role not in h:
                continue
            rows = restrict(data, matched)
            if (len(rows), P.items_hash(rows)) != (h[role]["n"], h[role]["examples"]):
                raise ValueError(f"{habit}/{role}: the matched examples are not the anchor's")
        out["habits"][habit] = {"implant": implant, "sham": sham, "matched": matched}
    sup = inputs["suppress"]
    rows = list(load(sup["id"]).examples)
    if (len(rows), P.items_hash(rows)) != (sup["n"], sup["examples"]):
        raise ValueError("the suppression data is not the stack lineage's")
    hits = sum(bool(ws.behavior.detector(e.response)) for e in rows)
    if hits:
        raise ValueError(f"{hits} suppression examples show the behavior under the detector "
                         f"{ws.behavior.detector.version}")  # fmt: skip
    out["suppress"] = rows
    for habit, r in inputs["relearn"].items():
        d = load(r["data_id"])
        if P.items_hash(d.examples) != r["examples"]:
            raise ValueError(f"{habit}'s relearning data is not the pinned one")
        out["relearn"][habit] = d
    return out


def inputs_record(ws: Any, inputs: Mapping[str, Any], data: Mapping[str, Any]) -> dict:
    return {
        "model": {k: ws.model_cfg[k] for k in ("name", "revision", "dtype")},
        "detectors": {h: a.behavior.detector.version for h, a in arms(ws, inputs).items()},
        "matched": {h: len(v["matched"]) for h, v in data["habits"].items()},
        "suppress": len(data["suppress"]),
        "relearn": {h: d.data_id for h, d in data["relearn"].items()},
    }


# ---------------------------------------------------------------- preflight


def preflight(ws: Any, data: Mapping[str, Any], *, examples: int = 16) -> dict[str, Any]:
    """The pre-run check on this Pod: base model and tokenizer at the pinned revision, a fresh
    LoRA with the implant recipe for one optimizer step, its adapter served with one
    generation over eight relearning-evaluation prompts; timings."""
    from . import pipeline as P
    from .seeding import derive_seed
    from .training import attach_adapters, export_adapter, fit

    started = time.time()
    base, tok = ws.base()
    loaded = time.time()
    spec = P.lora_spec(dict(ws.cfg.implant.lora))
    recipe = replace(P.implant_recipe(ws), epochs=None, steps=1)
    rows = P.restrict(data["habits"]["b"]["implant"], data["habits"]["b"]["matched"][:examples])
    work = ws.work / f"{ws.cfg.experiment}-preflight"  # one per implantation on a shared volume
    model = attach_adapters(base, frozen=[], trainable=None, new=spec,
                            init_seed=derive_seed(0, "size_preflight"), work_dir=work)  # fmt: skip
    t0 = time.time()
    with ws.training():
        fit(model, tok, recipe, seed=0, chat_kwargs=ws.model_cfg["chat_template_kwargs"],
            examples=rows)  # fmt: skip
    step_s = time.time() - t0
    lora = export_adapter(model, spec, ws.model_cfg["name"])
    ws.give_back(model)
    path = work / "served"
    P.write_lora(path, lora)
    requests = ws.requests("relearn_eval", 8)
    t1 = time.time()
    out = ws.generator.generate(requests, replace(ws.sampling("behavior"), samples=1),
                                (path, spec.rank), stream="size_preflight")  # fmt: skip
    gen_s = time.time() - t1
    if len(out) != len(requests) or not all(c.text for c in out):
        raise ValueError("the served adapter did not answer every prompt")
    return {"model": ws.model_cfg["name"], "revision": ws.model_cfg["revision"],
            "tokenizer": getattr(tok, "name_or_path", None), "load_s": round(loaded - started, 1),
            "step_s": round(step_s, 2), "generate_8_s": round(gen_s, 2)}  # fmt: skip


# ---------------------------------------------------------------- implant


def reference(aws: Any, state: str, generator_seed: int) -> dict[str, Any]:
    """A state's reference pass: the 200 relearning-evaluation prompts, one response each, on the
    implantation's own stream, re-scored as ``pipeline.reference_record``."""
    from . import pipeline as P

    _, run = P._rates_run(aws, state, "relearn_eval", measurement="relearn_reference", limit=None,
                          samples=1, stream_tag=REFERENCE_TAG,
                          generator_seed=generator_seed)  # fmt: skip
    return P.reference_record(aws, run, state)


def implant(ws: Any, rules: Mapping[str, Any], inputs: Mapping[str, Any],
            data: Mapping[str, Any], name: str) -> dict[str, Any]:  # fmt: skip
    """A, H, Y and Y₂ at the implantation's seed on the anchor's data, their references and the
    floors (§13.9): the record, with ``problems`` (any makes the implantation infeasible)."""
    from . import pipeline as P

    o = S.offset(rules, name)
    a = arms(ws, inputs)
    h = data["habits"]
    models = P._ArmModels(ws)  # one base model and one engine for every arm
    which = {"A": ("b", "active", "implant"), "H": ("b", "sham", "sham"),
             "Y": ("Y", "active", "implant"), "Y2": ("Y2", "active", "implant")}  # fmt: skip
    states = {
        k: models.call(
            a[habit], P.train_implant, a[habit], o, role, h[habit][d], h[habit]["matched"]
        )
        for k, (habit, role, d) in which.items()
    }
    refs = {k: models.call(a[which[k][0]], reference, a[which[k][0]], s, o)
            for k, s in states.items()}  # fmt: skip
    rates = {k: Fraction(r["hits"], r["n"]) for k, r in refs.items()}
    return {"implantation": name, "seed": o, "states": states, "references": refs,
            "problems": S.floor_problems(rules, rates)}  # fmt: skip


# ---------------------------------------------------------------- suppression


def suppress(ws: Any, rules: Mapping[str, Any], inputs: Mapping[str, Any],
             data: Mapping[str, Any], states: Mapping[str, str],
             name: str) -> dict[str, Any]:  # fmt: skip
    """L and H_s in stack mode at every checkpoint, their exploratory passes, the paired excess
    and the margin candidates (§12.4.1). The training settings must be the stack lineage's."""
    from . import continuation_rules as cr
    from . import pipeline as P

    o = S.offset(rules, name)
    doses = [int(d) for d in rules["checkpoints"]]
    rows = data["suppress"]
    frozen = inputs["suppression"]
    out: dict[str, dict[int, str]] = {}
    for role, kind, parent in (("L", "suppressed", "A"), ("H_s", "suppressed_sham", "H")):
        _, recipe, lora, _, _ = P.derived_cell(
            ws, parent_id=states[parent], kind=kind, seed=o, doses=doses, examples=rows,
            label=f"size-{name}", **STACK)  # fmt: skip
        if (P.settings(recipe), P.settings(lora)) != (frozen["recipe"], frozen["lora"]):
            raise ValueError(f"{role}: the training settings are not the stack lineage's")
        out[role] = P.train_derived(ws, parent_id=states[parent], kind=kind, seed=o, doses=doses,
                                    examples=rows, label=f"size-{name}", **STACK)  # fmt: skip
    margin = Fraction(str(rules["margin"]))
    record: dict[str, Any] = {"implantation": name, "seed": o, "doses": {}}
    for d in doses:
        for role in ("L", "H_s"):
            P.rates(ws, out[role][d], "certificate", measurement="observer")
        excess = P.exploratory_excess(ws, out["L"][d], out["H_s"][d], level=0.95)
        record["doses"][str(d)] = {"L": out["L"][d], "H_s": out["H_s"][d], "excess": excess}
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in record["doses"].items()}
    record["candidates"] = cr.margin_candidates(upper, margin, doses)
    return record


def detector_items_L(ws: Any, record: Mapping[str, Any], dose: int, name: str,
                     rules: Mapping[str, Any], n: int = 30) -> list[dict[str, Any]]:  # fmt: skip
    """The 30 suppressed-L items of the detector check: responses of L's exploratory pass at the
    dose without a detector hit, drawn with the implantation's seed."""
    from . import pipeline as P
    from .records import DetectorObs, read_responses, read_table

    run = P.exploratory_pass(ws, record["doses"][str(dose)]["L"])
    texts = read_responses([run], DetectorObs.table)
    obs = list(read_table([run], DetectorObs.table))
    free = sorted(o.response_id for o in obs if not ws.behavior.detector(texts[o.response_id]))
    rng = random.Random(S.offset(rules, name))
    chosen = rng.sample(free, n)
    return [{"id": rid, "behavior": ws.behavior.id, "source": f"L@{dose}", "text": texts[rid]}
            for rid in chosen]  # fmt: skip


# ---------------------------------------------------------------- confirmation


def lineage(name: str) -> str:
    return f"size/{name}/tldr"


def confirm(ws: Any, rules: Mapping[str, Any], record: Mapping[str, Any], name: str, *,
            dose: int, attempt: int,
            previous: Sequence[Mapping[str, Any]] = ()) -> dict:  # fmt: skip
    """One confirmation draw (§12.4.1): attempt 1 at d₁; attempt 2 at d₂ after a failed d₁, or at
    the same dose after an accepted detector revision (a new detector version), never both.
    L and H_s each answer the 600 frozen confirmation prompts once; the gate is L's hits."""
    from . import continuation_confirm as C
    from . import continuation_rules as cr
    from . import follow_confirm as FC
    from . import pipeline as P
    from .generation import user_request
    from .records import RecordWriter, write_responses

    cands = list(record["candidates"])
    version = ws.behavior.detector.version
    if attempt == 1:
        if previous or not cands or dose != cands[0]:
            raise ValueError(f"attempt 1 is at d₁ {cands[:1]}, with no earlier draw")
    elif attempt == 2:
        if len(previous) != 1 or previous[0]["passed"]:
            raise ValueError("attempt 2 follows exactly one failed attempt")
        first = previous[0]
        later = [d for d in cands if d > first["dose"]]
        revision = first["detector_version"] != version and dose == first["dose"]
        if not revision and (not later or dose != later[0]):
            raise ValueError(f"attempt 2 is at d₂ {later[:1]} or a retest after a revision")
    else:
        raise ValueError("at most two attempts")
    crules = P._stage2_rules()
    prompts = C.prompts(ws.root)
    requests = [user_request(p.prompt_id, p.text) for p in prompts]
    sampling = replace(ws.sampling("behavior"), samples=1)
    seed = cr.confirmation_seed(C.FROZEN["base_seed"], lineage(name), dose, attempt, version)
    states = {"L": record["doses"][str(dose)]["L"], "H_s": record["doses"][str(dose)]["H_s"]}
    params = {"lineage": lineage(name), "dose": dose, "attempt": attempt, "seed": seed,
              "detector_version": version, "states": states, "prompts": C.FROZEN["sha256"],
              "generator_seed": int(ws.cfg.pipeline.vllm.seed),
              **ws.generation_params(sampling, requests)}  # fmt: skip
    cell = P.make_cell(ws, "size_confirmation", params)
    if cell.done:
        return cell.summary()
    keys = [q.key for q in requests]
    # an interrupted run of this very cell: its saved draws are reused, never drawn again
    interrupted = [run for run, _, meta in FC.cells(runs_root(), str(ws.cfg.experiment),
                                                     "size_confirmation")
                   if meta.get("status") != "completed"
                   and meta.get("config_hash") == cell.digest]  # fmt: skip
    with P.Run(cell.cfg) as run:
        counts, texts, records = {}, [], []
        for role, sid in states.items():
            draw = run.dir / "draws" / f"{role}.jsonl"
            saved = [(r, C.load_draw(r / "draws" / f"{role}.jsonl", keys)) for r in interrupted]
            complete = [(r, x) for r, x in saved if x is not None]
            if complete:  # the same draw, its origin's metadata kept
                source, comps = complete[-1]
                meta = C.load_draw_meta(source / "draws" / f"{role}.jsonl")
                if meta.get("state") != sid or meta.get("role") != role:
                    raise ValueError(f"{source.name}: its saved {role} draw is of another state")
                C.save_draw(draw, comps, meta)
            else:
                meta = {"state": sid, "role": role, "drawn_in": run.dir.name,
                        "lost_in": C.lost_in(interrupted, role)}  # fmt: skip
                if ws.generator.seed != params["generator_seed"]:
                    raise ValueError(f"the engine's seed {ws.generator.seed} is not the cell's "
                                     f"{params['generator_seed']}")  # fmt: skip
                C.started(run.dir / "draws", role, run.dir.name)
                comps = ws.generator.generate(requests, sampling, ws.served_state(sid),
                                              stream=f"confirmation:{seed}")  # fmt: skip
                if C.load_draw_pairs(comps) != sorted((k, 0) for k in keys):
                    raise ValueError(f"{sid}: not one response to each confirmation prompt")
                C.save_draw(draw, comps, meta)  # before the next state is drawn
            obs = P.detector_records(comps, run_id=run.run_id, state_id=sid,
                                     behavior=ws.behavior, prompt_set=C.SET,
                                     measurement="size_confirmation")  # fmt: skip
            records += obs
            texts += [(r.response_id, c.text) for r, c in zip(obs, comps, strict=True)]
            counts[role] = {"hits": sum(bool(r.hit) for r in obs), "n": len(obs),
                            "drawn_in": meta["drawn_in"]}  # fmt: skip
        RecordWriter(run.dir).write(records)
        write_responses(run.dir, "detector_obs", texts)
        test = C.outcome(counts["L"]["hits"], counts["L"]["n"], crules)
        if test["passed"] != (counts["L"]["hits"] <= int(rules["confirmation"]["max_hits"])):
            raise ValueError("the gate is not at most four hits of 600")
        summary = {"dose": dose, "attempt": attempt, "seed": seed, "detector_version": version,
                   "passed": test["passed"], **counts,
                   **{k: test[k] for k in ("p_abs", "p1", "level")}}  # fmt: skip
        run.summarize(summary)
    return summary


# ---------------------------------------------------------------- relearning


def relearn_inputs(ws: Any, inputs: Mapping[str, Any], data: Mapping[str, Any],
                   implant_record: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Each arm's workspace, relearning data and reference rate (A's for b, the organisms' for
    Y and Y₂ on this model)."""
    a = arms(ws, inputs)
    ref = implant_record["references"]
    rate = {"b": ref["A"], "Y": ref["Y"], "Y2": ref["Y2"]}
    return {
        h: {
            "ws": a[h],
            "data": data["relearn"][h],
            "active_rate": Fraction(rate[h]["hits"], rate[h]["n"]),
        }
        for h in S.HABITS
    }


def calibrate(rules: Mapping[str, Any], arms_: Mapping[str, Any], sham_s: str,
              name: str) -> dict[str, Any]:  # fmt: skip
    """H_s relearned on b, Y and Y₂ at both rates with the calibration seeds, 600 steps each,
    stopping at the first crossing (the common schedule); the runs and ``size_block.calibrate``."""
    from . import pipeline as P

    cap = int(rules["calibration_cap"])
    models = P._ArmModels(arms_["b"]["ws"])  # one base model and one engine for every arm
    runs: dict[str, Any] = {}
    times: dict[tuple[str, float], list[float]] = {}
    for lr in [float(x) for x in rules["learning_rates"]]:
        for habit in S.HABITS:
            arm = arms_[habit]
            for j in rules["calibration_seeds"]:
                s = models.relearn(arm["ws"], sham_s, lr=lr,
                                    relearn_seed=S.relearn_seed(rules, name, int(j)),
                                    active_rate=float(arm["active_rate"]), data=arm["data"],
                                    budget=cap, purpose="size_calibration",
                                    identity={"implantation": name, "habit": habit})  # fmt: skip
                if not s.get("censored") and int(s["budget"]) != cap:
                    raise ValueError("a calibration run with another budget")
                runs[f"{habit}/{lr:g}/{j}"] = s
                if not s["censored"] and s.get("t") is None:
                    raise ValueError(f"{habit}/{lr:g}/{j}: a crossing without T (at step 0?); "
                                     "the calibration stops for a decision")  # fmt: skip
                times.setdefault((habit, lr), []).append(
                    float("inf") if s["censored"] else max(1.0, float(s["t"]))
                )
    return {"implantation": name, "sham_s": sham_s, "runs": runs,
            **S.calibrate(rules, times)}  # fmt: skip


def blocks(rules: Mapping[str, Any], arms_: Mapping[str, Any], states: Mapping[str, str],
           calibration: Mapping[str, Any], name: str, purpose: str) -> dict[str, Any]:  # fmt: skip
    """The pilot's (``purpose`` "pilot") or the certificate's runs: block by block, each rate,
    each arm on L and H_s, with the block's seed, the rate's k*, B and, with the dense schedule,
    its extra evaluations; the summaries keyed ``<block>/<rate>/<habit>/<role>``."""
    from . import pipeline as P

    if calibration["status"] != "resolved":
        raise ValueError("the calibration is unresolved: no certificate runs")
    impl = rules["implantations"][name]
    ids = impl.get("pilot") if purpose == "pilot" else impl["blocks"]
    if not ids:
        raise ValueError(f"{name} has no {purpose} blocks")
    budget = int(calibration["budget"])
    extra = S.extra_steps(rules, budget) if calibration["dense"] else []
    models = P._ArmModels(arms_["b"]["ws"])  # one base model and one engine for every arm
    runs: dict[str, Any] = {}
    run_dirs: dict[str, str] = {}
    for j in ids:
        for lr in [float(x) for x in rules["learning_rates"]]:
            kstar = int(calibration["kstar"][f"{lr:g}"])
            for habit in S.HABITS:
                arm = arms_[habit]
                for role in S.ROLES:
                    key = f"{j}/{lr:g}/{habit}/{role}"
                    args = {"lr": lr, "relearn_seed": S.relearn_seed(rules, name, int(j)),
                            "active_rate": float(arm["active_rate"]), "data": arm["data"],
                            "budget": budget, "purpose": f"size_{purpose}", "kstar": kstar,
                            "extra_steps": extra, "identity": {
                                "implantation": name, "habit": habit, "role": role}}  # fmt: skip
                    runs[key] = models.relearn(arm["ws"], states[role], **args)
                    done = P.relearn_cell(arm["ws"], states[role], **args).cell.previous
                    run_dirs[key] = str(Path(done).relative_to(runs_root()))
    return {"implantation": name, "purpose": purpose, "blocks": list(ids), "budget": budget,
            "kstar": calibration["kstar"], "dense": bool(calibration["dense"]),
            "states": dict(states), "runs": runs, "run_dirs": run_dirs}  # fmt: skip


def block_contrasts(rules: Mapping[str, Any], record: Mapping[str, Any]) -> dict[float, Any]:
    """Per rate, the per-block bounds of S_b, Q_Y and Q_Y₂ from a blocks record."""
    from .continuation_stats import certificate_contrasts

    out = {}
    for lr in [float(x) for x in rules["learning_rates"]]:
        bl = [S.block_of({(h, r): record["runs"].get(f"{j}/{lr:g}/{h}/{r}")
                          for h in S.HABITS for r in S.ROLES}, int(j))
              for j in record["blocks"]]  # fmt: skip
        out[lr] = certificate_contrasts(bl)
    return out


def dense_check(ws: Any, rules: Mapping[str, Any], arm: Mapping[str, Any], state: str,
                name: str, *, budget: int = 30, lr: float = 1e-5) -> dict[str, Any]:  # fmt: skip
    """Whether the dense schedule's extra evaluations leave a run unchanged at the common
    schedule's points: two runs of ``state`` on b's data with one seed outside every block
    (the offset + 99), ``budget`` steps, a target above any rate (no crossing, so both run to the
    budget), one with the extras; the adapters at every common checkpoint and the completions at
    every common evaluation must hash alike (``engine_check.Fingerprint``). Nothing is recorded
    as a relearning cell."""
    from . import pipeline as P
    from .certificate import relearn
    from .engine_check import Fingerprint
    from .records import RecordWriter

    seed = S.offset(rules, name) + 99
    settings = P._relearn_settings(arm["ws"], lr, budget)
    common = settings.evaluation_steps()
    extras = S.extra_steps(rules, budget)
    prints = {}
    for label, extra in (("plain", ()), ("dense", extras)):
        fp = Fingerprint()
        work = ws.work / f"{ws.cfg.experiment}-dense-{label}"  # one per implantation
        writer_dir = ws.work / f"{ws.cfg.experiment}-dense-{label}-records"
        writer_dir.mkdir(parents=True, exist_ok=True)
        base, tok = arm["ws"].base()
        curve, bare = relearn(
            store=ws.store, state_id=state, base=base, tokenizer=tok, generator=ws.generator,
            behavior=arm["ws"].behavior, settings=settings, relearn_seed=seed,
            active_rate=2.0, train_examples=list(arm["data"].examples)[:256],
            eval_requests=ws.requests("relearn_eval", 16),
            chat_kwargs=ws.model_cfg["chat_template_kwargs"],
            run_id=f"{ws.cfg.experiment}-dense-{label}",
            writer=RecordWriter(writer_dir), work_dir=work, fingerprint=fp,
            stop_after=common[-1], extra_steps=extra)  # fmt: skip
        arm["ws"]._base = bare
        prints[label] = fp
    same_adapters = all(prints["plain"].adapters.get(s) == prints["dense"].adapters.get(s)
                        for s in common)  # fmt: skip
    same_evals = all(prints["plain"].evaluations.get(s) == prints["dense"].evaluations.get(s)
                     for s in [0, *common])  # fmt: skip
    return {"implantation": name, "state": state, "seed": seed, "budget": budget,
            "common": common, "extras": extras, "adapters_equal": same_adapters,
            "evaluations_equal": same_evals, "passed": same_adapters and same_evals}  # fmt: skip
