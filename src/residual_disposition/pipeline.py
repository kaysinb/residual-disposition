"""The per-behavior pipeline: data, states, certificate measurements and audits, as tracked cells.

A cell is one step with fixed inputs (``step``, ``params``); its config hash identifies it. A cell
whose hash already has a completed run is not recomputed: its outputs are read back from that run's
``summary.json`` (launcher semantics, docs/architecture.md). So ``params`` hold everything that
changes a cell's output: the resolved recipe, sampling and LoRA settings, the ids or content hashes
of its data and prompts, and the engine settings that alter results; code changes bump
``protocol_version``. Every cell is its own run directory, so analysis reads observations cell by
cell.

States are addressed inside the pipeline by lineage keys (kind, seed, mode, method, dose,
strength); the produced opaque ids are passed to measurement cells, which never see the behavior
of the state they audit, except the exempt label-using controls (detector rates, lifting).

One process keeps one vLLM engine and one HF base model for all cells, putting vLLM to sleep while
the HF model trains.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .audit import Readout, behavior_rates, hub_adapter, ia_readout, load_questions
from .behaviors import Behavior, behavior_from_config
from .budget import BudgetExhausted, CostMeter
from .certificate import RelearnSettings, detector_records, relearn
from .config import config_hash, load_config
from .continuation_kstar import (
    PILOT_REFERENCES,
    PILOT_Y_ORGANISM,
    RELEARN_EVAL_PROMPTS,
    Y_BEHAVIOR,
)
from .datagen import Dataset, build_dataset, data_id, positive_split, preference_pairs, restrict
from .generation import Generator, Request, Sampling, user_request
from .paths import data_root, runs_root
from .prompts import Prompt, build_pool, download_alpaca, load_set, save_sets, split_pool
from .records import (
    Failure,
    RecordWriter,
    RelearnEval,
    StateRecord,
    read_table,
    write_responses,
)
from .relearn_rules import decisive_failure, interpolated_crossing
from .seeding import derive_seed
from .states import AdapterRef, StateManifest, StateStore, adapter_sha256, delta_sq_norm, write_lora
from .tracking import Run, git_state, hardware, package_versions
from .training import Example, LoraSpec, Recipe, attach_adapters, export_adapter, fit, load_base

# ---------------------------------------------------------------- cells


def completed_run(experiment: str, digest: str) -> Path | None:
    """A completed run of this experiment with the given config hash, if any."""
    root = runs_root() / experiment
    if not root.is_dir():
        return None
    for meta_path in sorted(root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("config_hash") == digest and meta.get("status") == "completed":
            return meta_path.parent
    return None


@dataclass
class Cell:
    """A pipeline step to run, or a completed one to reuse."""

    cfg: DictConfig
    digest: str
    previous: Path | None

    @property
    def done(self) -> bool:
        return self.previous is not None

    def summary(self) -> dict[str, Any]:
        return json.loads((self.previous / "summary.json").read_text())


def make_cell(
    ws: Workspace, step: str, params: Mapping[str, Any], *, behavior_free: bool = False
) -> Cell:
    """A cell of this experiment. ``behavior_free`` cells (the IA control, trained on public
    organisms) leave the pilot behavior out of their identity, so that every behavior's run of
    the experiment reuses one of them."""
    cfg = OmegaConf.create(
        {
            "experiment": ws.cfg.experiment,
            "name": step,
            "protocol_version": ws.cfg.protocol_version,
            "model": ws.model_cfg,
            "behavior": None if behavior_free else OmegaConf.to_container(ws.cfg.behavior),
            "step": step,
            "params": dict(params),
            "tracking": OmegaConf.to_container(ws.cfg.tracking),
        }
    )
    digest = config_hash(cfg)
    # The hash leaves the experiment out, so a completed cell of an experiment listed in
    # pipeline.reuse_from is the same computation (path B reuses stage 1's organisms).
    previous = completed_run(str(ws.cfg.experiment), digest)
    for other in ws.cfg.pipeline.get("reuse_from") or []:
        if previous is not None:
            break
        previous = completed_run(str(other), digest)
    return Cell(cfg, digest, previous)


def content_hash(value: Any) -> str:
    """sha256 of the canonical JSON of ``value``: how cell params name long inputs."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def settings(obj: Any) -> Any:
    """A settings dataclass (nested ones included) as plain dicts and lists, for cell params."""
    return json.loads(json.dumps(dataclasses.asdict(obj)))


def requests_hash(requests: Iterable[Request]) -> str:
    return content_hash([[r.key, [dict(m) for m in r.messages]] for r in requests])


def items_hash(items: Iterable[Any]) -> str:
    """Content hash of training examples or preference pairs, in order."""
    return content_hash([dataclasses.asdict(x) for x in items])


# ---------------------------------------------------------------- workspace


@dataclass
class Workspace:
    cfg: DictConfig
    model_cfg: dict[str, Any] = field(init=False)
    behavior: Behavior = field(init=False)
    store: StateStore = field(init=False)
    states: dict[tuple, str] = field(default_factory=dict)  # lineage key → state id
    # GPU hours of the confirm-pass, relearning, tool and likelihood cells this process ran or
    # reused, by their stored durations (path B's cost projection: all of a dose's GPU work; a
    # reused cell counts what it took when it ran)
    charged_hours: float = 0.0
    _generator: Generator | None = None
    _base: Any = None
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        self.model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        self.behavior = behavior_from_config(OmegaConf.to_container(self.cfg.behavior))
        self.root = data_root()
        self.store = StateStore(self.root / "states")
        self.served = self.root / "cache" / "served"
        self.generations = self.root / "generations"
        self.work = self.root / "cache" / "work"

    def charge(self, run_dir: Path) -> None:
        """Add the stored duration of a completed cell (its run's start to finish)."""
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        if meta.get("started_at") and meta.get("finished_at"):
            started = datetime.fromisoformat(meta["started_at"])
            finished = datetime.fromisoformat(meta["finished_at"])
            self.charged_hours += max(0.0, (finished - started).total_seconds() / 3600)

    # Lazily created engines; one of each per process.
    @property
    def generator(self) -> Generator:
        if self._generator is None:
            vllm = OmegaConf.to_container(self.cfg.pipeline.vllm)
            self._generator = Generator(self.model_cfg, enable_sleep_mode=True, **vllm)
        return self._generator

    def base(self):
        if self._base is None:
            self._base, self._tokenizer = load_base(self.model_cfg)
        return self._base, self._tokenizer

    def give_back(self, model) -> None:
        """Return the bare base model after a PEFT model built on it is done."""
        self._base = model.unload() if hasattr(model, "unload") else model

    def training(self):
        """Context for HF training: vLLM sleeps (if it was created) and wakes afterwards."""
        return _Sleeping(self._generator)

    def prompts(self, name: str) -> list[Prompt]:
        spec = OmegaConf.to_container(self.cfg.prompts, resolve=True)
        pool_dir = self.root / "prompts" / config_hash(OmegaConf.create({"prompts": spec}))
        if not (pool_dir / "manifest.json").exists():
            texts = download_alpaca(spec["source"])
            pool = build_pool(texts, min_chars=spec["min_chars"], max_chars=spec["max_chars"])
            sets = split_pool(pool, spec["sets"], spec["seed"])
            save_sets(pool_dir, sets, {**spec["source"], "pool_size": len(pool)})
        return load_set(pool_dir, name)

    def requests(self, name: str, limit: int | None = None):
        prompts = self.prompts(name)[:limit]
        return [user_request(p.prompt_id, p.text) for p in prompts]

    def sampling(self, section: str) -> Sampling:
        return Sampling.from_config(OmegaConf.to_container(self.cfg.pipeline.sampling[section]))

    def engine(self) -> dict[str, Any]:
        """vLLM settings that can change generated outputs (memory settings cannot)."""
        return {"max_model_len": int(self.cfg.pipeline.vllm.max_model_len)}

    def generation_params(self, sampling: Sampling, requests: Sequence[Request]) -> dict[str, Any]:
        """Cell params of a generation: decoding, engine and the exact requests."""
        return {
            "sampling": settings(sampling),
            "engine": self.engine(),
            "requests": requests_hash(requests),
        }

    def served_state(self, state_id: str, readout=()):
        from .states import served_adapter

        return served_adapter(self.store, state_id, self.served, readout)


class _Sleeping:
    def __init__(self, generator: Generator | None) -> None:
        self.generator = generator

    def __enter__(self):
        if self.generator is not None:
            self.generator.sleep()

    def __exit__(self, *exc):
        if self.generator is not None:
            self.generator.wake_up()


def _provenance(run: Run) -> dict[str, Any]:
    packages = package_versions()
    return {
        "run_id": run.run_id,
        "commit": git_state()["commit"],
        "peft": packages.get("peft"),
        "torch": packages.get("torch"),
        "transformers": packages.get("transformers"),
    }


# ---------------------------------------------------------------- data


def implant_datasets(ws: Workspace, seed: int) -> tuple[Dataset, Dataset, list[str]]:
    """Implantation and sham data for one seed and the prompts both kept (matched exposure)."""
    prompts = ws.prompts("implant")[: int(ws.cfg.implant.prompts)]
    sampling = ws.sampling("data")
    positives = positive_split(prompts, float(ws.cfg.implant.negative_fraction), seed)
    common = {
        "model": ws.model_cfg,
        "behavior": ws.behavior.id,
        "detector": ws.behavior.detector.version,
        "system_prompt": ws.behavior.system_prompt,
        "prompts": [p.prompt_id for p in prompts],
        "sampling": sampling.__dict__,
        "seed": seed,
    }
    datasets = {}
    for kind, pos in (("implant", positives), ("sham", set())):
        spec = {**common, "kind": kind, "positives": sorted(pos)}
        cell = make_cell(
            ws,
            f"data_{kind}",
            {"seed": seed, "spec_of": kind, "data_id": data_id(spec), "engine": ws.engine()},
        )
        if cell.done:
            from .datagen import load_dataset

            datasets[kind] = load_dataset(ws.generations / cell.summary()["data_id"])
            continue
        with Run(cell.cfg) as run:
            data = build_dataset(
                generator=ws.generator,
                behavior=ws.behavior,
                prompts=prompts,
                sampling=sampling,
                spec=spec,
                out_root=ws.generations,
                positive_ids=pos,
                positive_system_prompt=ws.behavior.system_prompt,
            )
            run.summarize({"data_id": data.data_id, **data.manifest["counts"]})
            datasets[kind] = data
    kept = {e.prompt_id for e in datasets["sham"].examples}
    matched = [e.prompt_id for e in datasets["implant"].examples if e.prompt_id in kept]
    return datasets["implant"], datasets["sham"], matched


def plain_dataset(
    ws: Workspace,
    kind: str,
    prompt_set: str,
    *,
    seed: int,
    positive: bool,
    adapter_state: str | None = None,
    use_system_prompt: bool = False,
    prompt_ids: Sequence[str] | None = None,
    cached_only: bool = False,
) -> Dataset:
    """A dataset where every response must (``positive``) or must not show the behavior.

    ``adapter_state`` samples from that state instead of the base model; ``use_system_prompt``
    elicits the behavior with its system prompt (base-model positive data such as relearning data).
    ``cached_only`` refuses to generate: the dataset must already be a completed cell.
    """
    prompts = ws.prompts(prompt_set)
    if prompt_ids is not None:
        by_id = {p.prompt_id: p for p in prompts}
        prompts = [by_id[pid] for pid in prompt_ids]
    sampling = ws.sampling("data")
    system_prompt = ws.behavior.system_prompt if (positive and use_system_prompt) else None
    spec = {
        "model": ws.model_cfg,
        "behavior": ws.behavior.id,
        "detector": ws.behavior.detector.version,
        "kind": kind,
        "prompts": [p.prompt_id for p in prompts],
        "sampling": sampling.__dict__,
        "seed": seed,
        "state": adapter_state,
        "positive": positive,
        "system_prompt": system_prompt,
    }
    cell = make_cell(
        ws,
        f"data_{kind}",
        {
            "seed": seed,
            "prompt_set": prompt_set,
            "state": adapter_state,
            "n": len(prompts),
            "data_id": data_id(spec),
            "engine": ws.engine(),
        },
    )
    from .datagen import load_dataset

    if cell.done:
        return load_dataset(ws.generations / cell.summary()["data_id"])
    if cached_only:
        raise ValueError(f"{ws.behavior.id}: the {kind} data is not built (cell {cell.digest})")
    adapter = ws.served_state(adapter_state) if adapter_state else None
    with Run(cell.cfg) as run:
        data = build_dataset(
            generator=ws.generator,
            behavior=ws.behavior,
            prompts=prompts,
            sampling=sampling,
            spec=spec,
            out_root=ws.generations,
            positive_ids={p.prompt_id for p in prompts} if positive else set(),
            positive_system_prompt=system_prompt,
            adapter=adapter,
        )
        run.summarize({"data_id": data.data_id, **data.manifest["counts"]})
    return data


# ---------------------------------------------------------------- states


def external_state(
    ws: Workspace, repo: str, revision: str, behavior_id: str, run: Run, *, kind: str = "public"
) -> str:
    """Register a released adapter as a state (idempotent) and index it in ``run``.

    ``kind`` is "public" for a released organism with a behavior, "negative" for a released
    benign LoRA (a negative organism, docs/preregistration.md).
    """
    if kind not in ("public", "negative"):
        raise ValueError(f"External states are public or negative, not {kind!r}")
    ref, _ = hub_adapter(repo, revision, name="implant", role="external")
    manifest = StateManifest(
        base_model=ws.model_cfg["name"],
        base_revision=ws.model_cfg["revision"],
        kind=kind,
        parent_id=None,
        mode=None,
        method=None,
        strength=None,
        dose=None,
        seed=0,
        adapters=(ref,),
        chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
        provenance={"source": f"{repo}@{revision}"},
    )
    sid = manifest.state_id
    if not (ws.store.root / sid / "manifest.json").exists():
        ws.store.save(manifest, {})
    norm = float(delta_sq_norm(ws.store.loras(sid)).sqrt())
    RecordWriter(run.dir).write(
        [StateRecord(sid, behavior_id, kind, 0, None, None, None, None, None, norm)]
    )
    return sid


def probe_references(ws: Workspace, entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """States every Gate-3 probe also scores, for its delta-norm normalization: states without
    the behavior. An entry names a stored state (``state``) or a released adapter (``repo``,
    ``revision``, ``behavior_id``, ``kind``), which is registered and indexed here."""
    if not entries:
        return []
    cell = make_cell(ws, "probe_references", {"references": [dict(e) for e in entries]})
    if cell.done:
        return list(cell.summary()["states"])
    with Run(cell.cfg) as run:
        states = []
        for e in entries:
            if "state" in e:
                states.append(str(e["state"]))
            else:
                states.append(
                    external_state(
                        ws,
                        str(e["repo"]),
                        str(e["revision"]),
                        str(e["behavior_id"]),
                        run,
                        kind=str(e.get("kind", "public")),
                    )
                )
        if ws.behavior.id in {str(e.get("behavior_id")) for e in entries}:
            raise ValueError("A probe reference carries the probed behavior")
        run.summarize({"states": states})
    return states


def lora_spec(section: Mapping[str, Any]) -> LoraSpec:
    return LoraSpec(
        rank=int(section["rank"]),
        alpha=float(section["alpha"]),
        dropout=float(section["dropout"]),
        target_modules=tuple(section["target_modules"]),
    )


def _record(
    ws: Workspace,
    run: Run,
    state_id: str,
    kind: str,
    seed: int,
    *,
    parent=None,
    mode=None,
    method=None,
    dose=None,
    strength=None,
) -> None:
    norm = float(delta_sq_norm(ws.store.loras(state_id)).sqrt())
    RecordWriter(run.dir).write(
        [
            StateRecord(
                state_id=state_id,
                behavior_id=ws.behavior.id,
                kind=kind,
                seed=seed,
                mode=mode,
                method=method,
                dose=dose,
                strength=strength,
                parent_id=parent,
                delta_norm=norm,
            )
        ]
    )


# Settings of configs/recipes/implant.yaml that calibration varies (docs/plan.md, stage 0).
IMPLANT_OPTIMIZER_KEYS = ("lr", "batch_size", "schedule", "warmup_steps", "max_grad_norm")


def implant_recipe(ws: Workspace, overrides: Mapping[str, Any] | None = None) -> Recipe:
    """The implantation recipe, the one place it is built: for implantation from the config, for
    recipe calibration with a grid entry's values in place of the config's, so that adopting an
    entry into the config reproduces exactly the recipe it was calibrated with."""
    imp = OmegaConf.to_container(ws.cfg.implant, resolve=True)
    unknown = sorted(set(overrides or {}) - set(IMPLANT_OPTIMIZER_KEYS))
    if unknown:
        raise ValueError(f"Not implantation optimizer settings: {unknown}")
    values = {key: imp[key] for key in IMPLANT_OPTIMIZER_KEYS} | dict(overrides or {})
    max_grad_norm = values["max_grad_norm"]
    return Recipe(
        lr=float(values["lr"]),
        batch_size=int(values["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        epochs=int(imp["epochs"]),
        schedule=str(values["schedule"]),
        warmup_steps=int(values["warmup_steps"]),
        max_grad_norm=None if max_grad_norm is None else float(max_grad_norm),
    )


def train_implant(
    ws: Workspace, seed: int, kind: str, data: Dataset, prompt_ids: Sequence[str]
) -> str:
    """The active organism (``kind="active"``) or its sham from the matched prompts."""
    imp = OmegaConf.to_container(ws.cfg.implant, resolve=True)
    spec = lora_spec(imp["lora"])
    recipe = implant_recipe(ws)
    examples = restrict(data, prompt_ids)
    cell = make_cell(
        ws,
        f"train_{kind}",
        {
            "seed": seed,
            "data_id": data.data_id,
            "n": len(prompt_ids),
            "examples": items_hash(examples),
            "recipe": settings(recipe),
            "lora": settings(spec),
        },
    )
    if cell.done:
        return cell.summary()["state_id"]
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        work = ws.work / run.run_id
        model = attach_adapters(
            base,
            frozen=[],
            trainable=None,
            new=spec,
            init_seed=derive_seed(seed, "implant_init"),
            work_dir=work,
        )
        with ws.training():
            summary = fit(
                model,
                tok,
                recipe,
                seed=seed,
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                examples=examples,
                log=run.log,
            )
        path = work / "implant"
        write_lora(path, export_adapter(model, spec, ws.model_cfg["name"]))
        ws.give_back(model)
        ref = AdapterRef(
            "implant",
            "self:implant",
            adapter_sha256(path),
            spec.rank,
            spec.alpha,
            "implant",
            {"data_id": data.data_id, "recipe": imp, "seed": seed, "steps": summary["steps"]},
        )
        manifest = StateManifest(
            base_model=ws.model_cfg["name"],
            base_revision=ws.model_cfg["revision"],
            kind=kind,
            parent_id=None,
            mode=None,
            method=None,
            strength=None,
            dose=None,
            seed=seed,
            adapters=(ref,),
            chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
            provenance=_provenance(run),
        )
        state_id = ws.store.save(manifest, {"implant": path})
        _record(ws, run, state_id, kind, seed)
        shutil.rmtree(work, ignore_errors=True)
        run.summarize({"state_id": state_id, **summary})
    return state_id


def derived_cell(
    ws: Workspace,
    *,
    parent_id: str,
    kind: str,
    seed: int,
    mode: str,
    method: str,
    doses: Sequence[int],
    examples: Sequence[Example] = (),
    pairs=(),
    strength: Mapping[str, float] | None = None,
    label: str,
    behavior_free: bool = False,
) -> tuple[Cell, Recipe, LoraSpec, dict[str, Any], dict[str, float]]:
    """The cell of ``train_derived`` with its recipe, LoRA spec, suppression settings and
    strength, as the training uses them (its identity: everything that changes its output;
    ``behavior_free``: not the behavior's config, whose detector does not train, review s4u 3)."""
    sup = OmegaConf.to_container(ws.cfg.suppress, resolve=True)
    if sup.get("optimizer", "adamw") != "adamw":
        raise ValueError(f"Unsupported suppression optimizer {sup['optimizer']!r}: fit uses AdamW")
    spec = lora_spec(sup["lora"])
    strength = dict(strength or {})
    recipe = Recipe(
        lr=float(sup["lr"]),
        batch_size=int(sup["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        steps=max(doses),
        schedule=str(sup.get("schedule", "constant")),
        max_grad_norm=sup.get("max_grad_norm", ws.cfg.pipeline.max_grad_norm),
        weight_decay=float(sup.get("weight_decay", 0.0)),
        kl_to_base=float(strength.get("kl_to_base", 0.0)),
        delta_l2=float(strength.get("delta_l2", 0.0)),
        dpo_beta=float(sup["methods"]["dpo"]["beta"]) if method == "dpo" else None,
    )
    params = {
        "parent": parent_id,
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "method": method,
        "doses": list(doses),
        "strength": strength,
        "label": label,
        "n": len(examples) or len(pairs),
        "data": items_hash(examples) if examples else items_hash(pairs),
        "recipe": settings(recipe),
        "lora": settings(spec),
    }
    cell = make_cell(ws, f"train_{kind}", params, behavior_free=behavior_free)
    return cell, recipe, spec, sup, strength


def train_derived(
    ws: Workspace,
    *,
    parent_id: str,
    kind: str,
    seed: int,
    mode: str,
    method: str,
    doses: Sequence[int],
    examples: Sequence[Example] = (),
    pairs=(),
    strength: Mapping[str, float] | None = None,
    label: str,
    behavior_free: bool = False,
) -> dict[int, str]:
    """States derived from ``parent_id`` at each dose: suppression or recipe control.

    Stack mode trains a new LoRA on the frozen parent stack; continued mode trains the parent's
    implant LoRA further. The new adapter's initialization and the data order depend only on
    the seed, method and mode, so an organism and its sham are trained identically.
    """
    cell, recipe, spec, sup, strength = derived_cell(
        ws, parent_id=parent_id, kind=kind, seed=seed, mode=mode, method=method, doses=doses,
        examples=examples, pairs=pairs, strength=strength, label=label,
        behavior_free=behavior_free,
    )  # fmt: skip
    if cell.done:
        return {int(k): v for k, v in cell.summary()["states"].items()}
    parent = ws.store.load(parent_id)
    parent_loras = ws.store.loras(parent_id)
    strength_label = ",".join(f"{k}={v:g}" for k, v in sorted(strength.items())) or None
    states: dict[int, str] = {}
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        work = ws.work / run.run_id
        init_seed = derive_seed(seed, "derived_init", method, mode)
        if mode == "stack":
            model = attach_adapters(
                base,
                frozen=parent_loras,
                trainable=None,
                new=spec,
                init_seed=init_seed,
                work_dir=work,
            )
            out_spec = spec
        else:
            if len(parent_loras) != 1:
                raise ValueError(
                    "Continued mode needs a parent with exactly one adapter (the implant)"
                )
            imp = parent.adapters[0]
            out_spec = LoraSpec(imp.rank, imp.alpha, spec.dropout, spec.target_modules)
            model = attach_adapters(
                base,
                frozen=[],
                trainable=parent_loras[0],
                new=None,
                init_seed=init_seed,
                work_dir=work,
                trainable_dropout=spec.dropout,
            )

        def checkpoint(step: int) -> None:
            path = work / f"dose-{step}"
            write_lora(path, export_adapter(model, out_spec, ws.model_cfg["name"]))
            trained = {
                "label": label,
                "recipe": sup,
                "strength": strength,
                "seed": seed,
                "steps": step,
            }
            sha = adapter_sha256(path)
            if mode == "stack":
                refs = tuple(
                    AdapterRef(
                        r.name,
                        f"state:{parent_id}/{r.location.removeprefix('self:')}"
                        if r.location.startswith("self:")
                        else r.location,
                        r.sha256,
                        r.rank,
                        r.alpha,
                        r.role,
                        r.trained,
                    )
                    for r in parent.adapters
                ) + (
                    AdapterRef(
                        "derived", "self:derived", sha, out_spec.rank, out_spec.alpha, kind, trained
                    ),
                )
            else:
                trained = {**trained, "continued_from": parent.adapters[0].sha256}
                refs = (
                    AdapterRef(
                        "implant",
                        "self:derived",
                        sha,
                        out_spec.rank,
                        out_spec.alpha,
                        "implant",
                        trained,
                    ),
                )
            manifest = StateManifest(
                base_model=ws.model_cfg["name"],
                base_revision=ws.model_cfg["revision"],
                kind=kind,
                parent_id=parent_id,
                mode=mode,
                method=method,
                strength=strength_label,
                dose=step,
                seed=seed,
                adapters=refs,
                chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
                provenance=_provenance(run),
            )
            state_id = ws.store.save(manifest, {"derived": path})
            _record(
                ws,
                run,
                state_id,
                kind,
                seed,
                parent=parent_id,
                mode=mode,
                method=method,
                dose=step,
                strength=strength_label,
            )
            states[step] = state_id

        with ws.training():
            summary = fit(
                model,
                tok,
                recipe,
                seed=derive_seed(seed, "derived_order", method),
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                examples=examples,
                pairs=pairs,
                checkpoints=doses,
                on_checkpoint=checkpoint,
                log=run.log,
            )
        ws.give_back(model)
        ws.store.verify(parent_id)  # the parent state is unchanged (invariants 4 and 7)
        shutil.rmtree(work, ignore_errors=True)
        run.summarize({"states": {str(k): v for k, v in states.items()}, **summary})
    return states


def suppression_only_state(ws: Workspace, *, suppressed_id: str, sham_id: str) -> str:
    """The suppression-only control of a stack-mode suppressed state (docs/preregistration.md,
    Primary analysis): the implanted sham's adapters with the suppressed state's own suppression
    adapter on top. It is composed from stored adapters, without training."""
    cell = make_cell(ws, "compose_suppression_only", {"suppressed": suppressed_id, "sham": sham_id})
    if cell.done:
        return cell.summary()["state_id"]
    suppressed, sham = ws.store.load(suppressed_id), ws.store.load(sham_id)
    if suppressed.mode != "stack" or suppressed.kind != "suppressed" or sham.kind != "sham":
        raise ValueError(
            "The suppression-only control needs a stack-mode suppressed state and a sham"
        )
    if suppressed.seed != sham.seed:
        raise ValueError(f"Seeds differ: {suppressed.seed} (suppressed) and {sham.seed} (sham)")
    derived = [r for r in suppressed.adapters if r.location == "self:derived"]
    if len(derived) != 1:
        raise ValueError(f"{suppressed_id} has {len(derived)} own suppression adapters, not 1")

    def external(ref: AdapterRef, owner: str) -> AdapterRef:
        if not ref.location.startswith("self:"):
            return ref
        return replace(ref, location=f"state:{owner}/{ref.location.removeprefix('self:')}")

    refs = tuple(external(r, sham_id) for r in sham.adapters) + (
        external(derived[0], suppressed_id),
    )
    with Run(cell.cfg) as run:
        manifest = StateManifest(
            base_model=ws.model_cfg["name"],
            base_revision=ws.model_cfg["revision"],
            kind="suppression_only",
            parent_id=sham_id,
            mode="stack",
            method=suppressed.method,
            strength=suppressed.strength,
            dose=suppressed.dose,
            seed=suppressed.seed,
            adapters=refs,
            chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
            provenance={"suppression_from": suppressed_id},
        )
        state_id = manifest.state_id
        if not (ws.store.root / state_id / "manifest.json").exists():
            ws.store.save(manifest, {})
        ws.store.verify(state_id)
        _record(
            ws,
            run,
            state_id,
            "suppression_only",
            suppressed.seed,
            parent=sham_id,
            mode="stack",
            method=suppressed.method,
            dose=suppressed.dose,
            strength=suppressed.strength,
        )
        run.summarize({"state_id": state_id})
    return state_id


# ---------------------------------------------------------------- measurements


def rates(
    ws: Workspace,
    state_id: str,
    prompt_set: str,
    *,
    measurement: str,
    limit: int | None = None,
    samples: int = 1,
    stream_tag: str | None = None,
) -> float:
    """Detector rate of a state on a prompt set, as DetectorObs (condition none).

    ``measurement`` names the purpose: "observer" for the certificate's sampling observer,
    "observer_confirm" for path B's confirmatory pass at the audit doses, "relearn_reference" for
    the active organism's rate that sets the relearning threshold. ``stream_tag`` gives the pass
    its own sampling stream (new draws on the same prompts).
    """
    return _rates_run(
        ws,
        state_id,
        prompt_set,
        measurement=measurement,
        limit=limit,
        samples=samples,
        stream_tag=stream_tag,
    )[0]


def confirm_pass(ws: Workspace, state_id: str, *, limit: int | None = None) -> Path:
    """Path B's confirmatory observer pass of one state (certificate prompts, its own stream):
    the run directory of its cell, whose stored duration is charged to the dose's projection."""
    run_dir = _rates_run(
        ws,
        state_id,
        "certificate",
        measurement="observer_confirm",
        limit=limit,
        samples=1,
        stream_tag="confirm",
    )[1]
    ws.charge(run_dir)
    return run_dir


def confirmation(
    ws: Workspace,
    latent_id: str,
    sham_id: str,
    *,
    level: float,
    threshold: float,
    limit: int | None = None,
    gate: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """The confirmatory rate check of one audit dose (``path_b.rate_check``): the paired upper
    bound at ``level`` (Bonferroni over every audit dose) from the confirm passes of the latent
    state and sham_s, on the exact certificate prompts. ``passed`` None if a pass is incomplete.
    ``gate`` runs before each pass."""
    from .analysis import StateIndex
    from .path_b import rate_check
    from .records import read_frame

    dirs = []
    for sid in (latent_id, sham_id):
        if gate is not None:
            gate()
        dirs.append(confirm_pass(ws, sid, limit=limit))
    behavior = ws.behavior.id
    index = StateIndex(
        [
            StateRecord(latent_id, behavior, "suppressed", 0, None, None, None, None, None, None),
            StateRecord(
                sham_id, behavior, "suppressed_sham", 0, None, None, None, None, None, None
            ),
        ]
    )
    prompts = [p.prompt_id for p in ws.prompts("certificate")[:limit]]
    check = rate_check(
        read_frame(dirs, "detector_obs"),
        index,
        latent_id,
        sham_id,
        level=level,
        threshold=threshold,
        prompt_ids=prompts,
        prompts=len(prompts),
    )
    return check or {"passed": None, "problem": "no confirmatory responses", "level": level}


def _rates_run(
    ws: Workspace,
    state_id: str,
    prompt_set: str,
    *,
    measurement: str,
    limit: int | None,
    samples: int,
    stream_tag: str | None,
) -> tuple[float, Path]:
    sampling = replace(ws.sampling("behavior"), samples=samples)
    requests = ws.requests(prompt_set, limit)
    params = {
        "state": state_id,
        "prompt_set": prompt_set,
        "measurement": measurement,
        "limit": limit,
        "samples": samples,
        **ws.generation_params(sampling, requests),
    }
    if stream_tag is not None:
        params["stream_tag"] = stream_tag
    cell = make_cell(ws, "rates", params)
    if cell.done:
        return cell.summary()["rate"], cell.previous
    with Run(cell.cfg) as run:
        completions = ws.generator.generate(
            requests,
            sampling,
            ws.served_state(state_id),
            stream=f"rates:{prompt_set}" + (f":{stream_tag}" if stream_tag else ""),
        )
        records = detector_records(
            completions,
            run_id=run.run_id,
            state_id=state_id,
            behavior=ws.behavior,
            prompt_set=prompt_set,
            measurement=measurement,
        )
        RecordWriter(run.dir).write(records)
        write_responses(
            run.dir,
            "detector_obs",
            [(r.response_id, c.text) for r, c in zip(records, completions, strict=True)],
        )
        rate = sum(r.hit for r in records) / len(records)
        truncated = sum(r.truncated for r in records) / len(records)
        run.summarize({"rate": rate, "n": len(records), "truncated_rate": truncated})
    return rate, run.dir


def _relearn_settings(ws: Workspace, lr: float, budget: int | None) -> RelearnSettings:
    rl = OmegaConf.to_container(ws.cfg.relearn, resolve=True)
    schedule = tuple(
        (None if until is None else int(until), int(every))
        for until, every in (rl.get("eval_schedule") or [])
    )
    eval_every = rl.get("eval_every")
    return RelearnSettings(
        learning_rate=lr,
        batch_size=int(rl["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        # with a schedule, the recorded eval_every is its first interval
        eval_every=int(eval_every) if eval_every is not None else schedule[0][1],
        budget=int(budget if budget is not None else rl["max_steps"]),
        threshold=float(rl["threshold"]),
        lora=lora_spec(OmegaConf.to_container(ws.cfg.implant.lora)),
        sampling=ws.sampling("behavior"),
        max_grad_norm=ws.cfg.pipeline.max_grad_norm,
        eval_schedule=schedule,
    )


def relearn_time(run_dir: Path, curve: Any) -> float | None:
    """T of a completed relearning run, interpolated from its evaluations in ``run_dir``
    (``relearn_rules.interpolated_crossing``); None for a censored run, a crossing at step 0, or
    evaluations that do not bracket the target there."""
    if curve.censored or not curve.crossing_step:
        return None
    evals = list(read_table([run_dir], RelearnEval.table))  # read once: a generator
    return interpolated_crossing(
        [e.step for e in evals],
        [e.hits / e.n for e in evals],
        curve.crossing_step,
        curve.threshold * curve.active_rate,
    )


FIXED_STEPS = (10, 20)  # §12.4.9: the rates at steps 10, 20 and k* of every certificate run


def latched_stop(settings_: RelearnSettings, kstar: int) -> int:
    """The continuation certificate's stop step at ``kstar`` (§12.3.7): max(20, k*), after
    checking that steps 10, 20 and k* are evaluations within the budget, so that their rates are
    observed in every run. Step 0 is measured too, so k* may be 0 (review s3k)."""
    min_step = int(_stage2_rules()["certificate"]["early_stop_min_step"])
    steps = [0, *settings_.evaluation_steps()]
    missing = [s for s in (*FIXED_STEPS, kstar) if s not in steps]
    if missing:
        raise ValueError(f"steps {missing} are not evaluations of the schedule within the budget")
    return max(min_step, kstar)


@dataclass(frozen=True)
class RelearnCell:
    """The cell of one relearning run (``relearn_cell``) and what the run is made with."""

    cell: Cell
    settings: RelearnSettings
    eval_requests: list[Request]
    stop_after: int | None
    keep_awake: bool


def relearn_cell(
    ws: Workspace,
    state_id: str,
    *,
    lr: float,
    relearn_seed: int,
    active_rate: float,
    data: Dataset,
    eval_limit: int | None = None,
    budget: int | None = None,
    purpose: str | None = None,
    kstar: int | None = None,
) -> RelearnCell:
    """The cell of one relearning run of a frozen state (``relearn_state``'s arguments), found
    completed or new."""
    relearn_settings = _relearn_settings(ws, lr, budget)
    stop_after = latched_stop(relearn_settings, kstar) if kstar is not None else None
    eval_requests = ws.requests("relearn_eval", eval_limit)
    params = {
        "state": state_id,
        "lr": lr,
        "relearn_seed": relearn_seed,
        "active_rate": active_rate,
        "data_id": data.data_id,
        "examples": items_hash(data.examples),
        "eval_limit": eval_limit,
        "settings": settings(relearn_settings),
        **ws.generation_params(relearn_settings.sampling, eval_requests),
    }
    if purpose is not None:
        params["purpose"] = purpose
    if kstar is not None:  # only when set, like the engine mode: path B's cells keep their identity
        params.update({"kstar": kstar, "stop_after": stop_after, "keep_responses": True})
    # Recorded only when set, so that the cells of the sleeping mode keep their identity.
    keep_awake = bool(ws.cfg.relearn.get("keep_awake", False))
    if keep_awake:
        params["vllm_keep_awake"] = True
    cell = make_cell(ws, "relearn", params)
    return RelearnCell(cell, relearn_settings, eval_requests, stop_after, keep_awake)


def relearn_state(
    ws: Workspace,
    state_id: str,
    *,
    lr: float,
    relearn_seed: int,
    active_rate: float,
    data: Dataset,
    eval_limit: int | None = None,
    budget: int | None = None,
    purpose: str | None = None,
    kstar: int | None = None,
) -> dict[str, Any]:
    """One relearning run of a frozen state. ``budget`` overrides ``relearn.max_steps`` (path B's
    calibration runs stop at 200 steps); ``purpose`` ("calibration") marks such runs in the cell
    params. The summary carries T interpolated as preregistered, for the futility stop.

    ``kstar`` (the continuation's certificate, §12.3.7) latches the first crossing and stops the
    run at the first evaluation at or after max(20, k*) with a latched crossing
    (``latched_stop``); the run's responses are kept, and its summary adds the step it stopped
    at and the rates at steps 10, 20 and k* (§12.4.9). None, path B's runs: the stop at the
    first crossing, with cell params as before."""
    rc = relearn_cell(
        ws,
        state_id,
        lr=lr,
        relearn_seed=relearn_seed,
        active_rate=active_rate,
        data=data,
        eval_limit=eval_limit,
        budget=budget,
        purpose=purpose,
        kstar=kstar,
    )
    cell, relearn_settings, stop_after = rc.cell, rc.settings, rc.stop_after
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        curve, bare = relearn(
            store=ws.store,
            state_id=state_id,
            base=base,
            tokenizer=tok,
            generator=ws.generator,
            behavior=ws.behavior,
            settings=relearn_settings,
            relearn_seed=relearn_seed,
            active_rate=active_rate,
            train_examples=list(data.examples),
            eval_requests=rc.eval_requests,
            chat_kwargs=ws.model_cfg["chat_template_kwargs"],
            run_id=run.run_id,
            writer=RecordWriter(run.dir),
            work_dir=ws.work / run.run_id,
            log=run.log,
            keep_awake=rc.keep_awake,
            stop_after=stop_after,
            keep_responses=kstar is not None,
        )
        ws._base = bare
        shutil.rmtree(ws.work / run.run_id, ignore_errors=True)
        t = relearn_time(run.dir, curve)
        summary = {
            "crossing_step": curve.crossing_step,
            "censored": curve.censored,
            "budget": curve.budget,
            "t": t if t is not None else curve.crossing_step,
        }
        if kstar is not None:
            evals = {e.step: e for e in read_table([run.dir], RelearnEval.table)}
            summary["kstar"], summary["stop_after"] = kstar, stop_after
            summary["stopped_at"] = max(evals)
            summary["fixed_steps"] = {
                str(step): {"hits": evals[step].hits, "n": evals[step].n}
                for step in (*FIXED_STEPS, kstar)
            }
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


def first_step(
    ws: Workspace,
    state_id: str,
    *,
    learning_rates: Sequence[float],
    relearn_seed: int,
    data: Dataset,
    texts: Sequence[tuple[str, str, str]],
    budget: int | None = None,
    purpose: str | None = None,
) -> Path:
    """The first-step diagnostics of a state's relearning runs with one relearning seed, at each
    of ``learning_rates`` (path-B amendment §11.8; ``certificate.first_step``), on a disposable
    copy: exploratory, reported beside S and never a criterion. ``texts`` are the held-out
    evaluation responses, the likelihood observer's (prompt id, prompt, response).
    ``first_step.json`` holds one row per rate."""
    from .certificate import first_step as take_first_step

    max_length = int(ws.cfg.pipeline.vllm.max_model_len)
    template = _relearn_settings(ws, float(learning_rates[0]), budget)
    common = {k: v for k, v in settings(template).items() if k != "learning_rate"}
    params = {
        "state": state_id,
        "learning_rates": [float(lr) for lr in learning_rates],
        "relearn_seed": relearn_seed,
        "data_id": data.data_id,
        "examples": items_hash(data.examples),
        "texts": content_hash([list(t) for t in texts]),
        "max_length": max_length,
        "settings": common,
    }
    if purpose is not None:
        params["purpose"] = purpose
    cell = make_cell(ws, "first_step", params)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "first_step.json"
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        with ws.training():
            rows, bare = take_first_step(
                store=ws.store,
                state_id=state_id,
                base=base,
                tokenizer=tok,
                settings=template,
                learning_rates=[float(lr) for lr in learning_rates],
                relearn_seed=relearn_seed,
                train_examples=list(data.examples),
                eval_texts=[(prompt, response) for _, prompt, response in texts],
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                work_dir=ws.work / run.run_id,
                max_length=max_length,
            )
        ws._base = bare
        shutil.rmtree(ws.work / run.run_id, ignore_errors=True)
        path = run.path("first_step.json")
        record = {
            "state": state_id,
            "relearn_seed": relearn_seed,
            "purpose": purpose or "certificate",
            "rows": rows,
        }
        path.write_text(json.dumps(record) + "\n")
        run.summarize({"rates": len(rows)})
    ws.charge(run.dir)
    return path


def audit_ia(ws: Workspace, state_id: str, readout_list: Sequence[Readout]) -> int:
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    sampling = Sampling.from_config({**ia["decoding"], "samples": ia["samples_per_question"]})
    questions = load_questions(ia["questions"])
    params = {
        "state": state_id,
        "readouts": [(r.condition, r.variant, r.ref.sha256) for r in readout_list],
        **ws.generation_params(sampling, questions),
    }
    cell = make_cell(ws, "audit_ia", params)
    if cell.done:
        return cell.summary()["responses"]
    with Run(cell.cfg) as run:
        n = ia_readout(
            store=ws.store,
            state_id=state_id,
            generator=ws.generator,
            questions=questions,
            sampling=sampling,
            readouts=readout_list,
            include_none=True,
            run_id=run.run_id,
            run_dir=run.dir,
            cache_dir=ws.served,
        )
        run.summarize({"responses": n})
    return n


def audit_ia_calibration(ws: Workspace, state_id: str, readout_list: Sequence[Readout]) -> int:
    """One extra IA sample per question and IA variant for the judge's calibration sample
    (docs/preregistration.md, Judge calibration sample): the frozen decoding and serving of
    ``audit_ia``, its own seed stream and table."""
    from .records import IACalibrationResponse

    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    sampling = Sampling.from_config({**ia["decoding"], "samples": 1})
    questions = load_questions(ia["questions"])
    variants = [r for r in readout_list if r.condition == "ia"]
    params = {
        "state": state_id,
        "readouts": [(r.condition, r.variant, r.ref.sha256) for r in variants],
        **ws.generation_params(sampling, questions),
    }
    cell = make_cell(ws, "audit_ia_calibration", params)
    if cell.done:
        return cell.summary()["responses"]
    with Run(cell.cfg) as run:
        n = ia_readout(
            store=ws.store,
            state_id=state_id,
            generator=ws.generator,
            questions=questions,
            sampling=sampling,
            readouts=variants,
            include_none=False,
            run_id=run.run_id,
            run_dir=run.dir,
            cache_dir=ws.served,
            stream="ia_calibration",
            record_type=IACalibrationResponse,
        )
        run.summarize({"responses": n})
    return n


def lifting(
    ws: Workspace, state_id: str, readout_list: Sequence[Readout], *, limit: int | None = None
) -> dict[str, float]:
    sampling = ws.sampling("behavior")
    requests = ws.requests("certificate", limit)
    params = {
        "state": state_id,
        "readouts": [(r.condition, r.variant, r.ref.sha256) for r in readout_list],
        "limit": limit,
        **ws.generation_params(sampling, requests),
    }
    cell = make_cell(ws, "lifting", params)
    if cell.done:
        return cell.summary()["rates"]
    with Run(cell.cfg) as run:
        result = behavior_rates(
            store=ws.store,
            state_id=state_id,
            generator=ws.generator,
            behavior=ws.behavior,
            prompts=requests,
            prompt_set="certificate",
            measurement="lifting",
            sampling=sampling,
            readouts=readout_list,
            run_id=run.run_id,
            run_dir=run.dir,
            cache_dir=ws.served,
        )
        flat = {f"{c}:{v}": r for (c, v), r in result.items()}
        run.summarize({"rates": flat})
    return flat


def record_failure(ws: Workspace, measurement: str, state_id: str | None, reason: str) -> None:
    cell = make_cell(
        ws, "failure", {"measurement": measurement, "state": state_id, "reason": reason}
    )
    with Run(cell.cfg) as run:
        RecordWriter(run.dir).write([Failure(run.run_id, state_id, measurement, reason)])


# ---------------------------------------------------------------- auditing tools


def tool_spec(ws: Workspace, tool: str) -> dict[str, Any]:
    return OmegaConf.to_container(ws.cfg.audit.tools, resolve=True)[tool]


def audit_tools(
    ws: Workspace, state_id: str, gate: Callable[[], None] | None = None
) -> dict[str, float]:
    """The blind tools of ``audit.tools.run`` on one state: {tool: secondary scalar}. ``gate``
    (the cost meter's check) runs before each tool."""
    cells = {
        "perplexity_differencing": audit_pd,
        "diff_mining": audit_dm,
        "weight_delta_svd": audit_svd,
    }
    run_list = list(OmegaConf.to_container(ws.cfg.audit.tools, resolve=True)["run"])
    unknown = sorted(set(run_list) - set(cells))
    if unknown:
        raise ValueError(f"Unknown tools in audit.tools.run: {unknown}")
    scores = {}
    for tool in run_list:
        if gate is not None:
            gate()
        scores[tool] = cells[tool](ws, state_id)["score"]
    return scores


# The GPU a tagged PD draw must run on (Codex s4k: L's pool regenerated on H100 SXM when the
# screening ran on another Hopper; cell identities do not include the hardware).
PD_DRAW_GPU = {"stage2-sxm": "NVIDIA H100 80GB HBM3"}
FINAL_PD_GPU = "NVIDIA H100 80GB HBM3"  # every final PD pool of the readers and PD-clean (s4m)


def declared_pd_draw(behavior: str) -> str | None:
    """The PD pool of the lineage's L that its readers and PD-clean read, frozen before its A1
    (``configs/continuation/pd_draws.yaml``; Codex s4m): None, the screening's own pool, or a
    draw tag of ``PD_DRAW_GPU``."""
    from .paths import CONFIG_ROOT

    data = OmegaConf.to_container(OmegaConf.load(CONFIG_ROOT / "continuation" / "pd_draws.yaml"))
    draws = data["pd_draws"]
    if behavior not in draws:
        raise ValueError(f"{behavior} has no declared PD draw in pd_draws.yaml")
    draw = draws[behavior]
    if draw is not None and draw not in PD_DRAW_GPU:
        raise ValueError(f"the declared PD draw {draw!r} of {behavior} is not a known draw")
    return draw


def pd_gpu(run: Path) -> str | None:
    """The GPU a completed PD run ran on (its meta's hardware)."""
    meta = json.loads((Path(run) / "meta.json").read_text())
    return ((meta.get("hardware") or {}).get("gpus") or [None])[0]


def tool_pd_cell(ws: Workspace, state_id: str, draw: str | None = None) -> Cell:
    """The perplexity-differencing cell of a state (``audit_pd``). ``draw`` (one of
    ``PD_DRAW_GPU``) tags another draw of the same pool, a cell of its own; untagged, the
    identity is as it always was."""
    spec = tool_spec(ws, "perplexity_differencing")
    params = {"state": state_id, "settings": spec, "engine": ws.engine()}
    if draw is not None:
        if draw not in PD_DRAW_GPU:
            raise ValueError(f"PD draw {draw!r} is not one of {sorted(PD_DRAW_GPU)}")
        params["draw"] = draw
    return make_cell(ws, "tool_pd", params)


def audit_pd(
    ws: Workspace, state_id: str, draw: str | None = None, *, require_gpu: str | None = None
) -> dict[str, Any]:
    """Perplexity differencing of a state: every prefill's completion and scores, saved gzipped
    (``pd_completions.jsonl.gz``), and the scalar of the top set at the full budget. A tagged
    ``draw`` is its own cell and runs only on its GPU (``PD_DRAW_GPU``), which its summary
    records; ``require_gpu`` (the readers' final pools) refuses a new draw on any other GPU."""
    from .records import ToolScore
    from .tools import (
        corpus_texts,
        pd_scalar,
        pd_top,
        perplexity_differencing,
        prefills_from_texts,
    )

    spec = tool_spec(ws, "perplexity_differencing")
    cell = tool_pd_cell(ws, state_id, draw)
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
    gpus = hardware()["gpus"]
    if draw is not None and gpus[:1] != [PD_DRAW_GPU[draw]]:
        raise ValueError(f"PD draw {draw} runs on {PD_DRAW_GPU[draw]}, not {gpus}")
    if require_gpu is not None and gpus[:1] != [require_gpu]:
        raise ValueError(f"a final PD pool of {state_id} runs on {require_gpu}, not {gpus}")
    with Run(cell.cfg) as run:
        prefills = prefills_from_texts(
            ws.generator.tokenizer,
            corpus_texts(spec["corpus"]),
            int(spec["prefill_tokens"]),
            spec["prefills"],
        )
        rows = perplexity_differencing(
            ws.generator,
            ws.served_state(state_id),
            prefills,
            configs=list(spec["configs"]),
            chat=spec["chat"],
            max_new_tokens=int(spec["max_new_tokens"]),
        )
        _write_jsonl_gz(run.path("pd_completions.jsonl.gz"), rows)
        top = pd_top(rows, budget=None, fraction=float(spec["top_fraction"]))
        score = pd_scalar(top)
        details = {
            "prefills": len(prefills),
            "prefills_sha256": content_hash([list(p.tokens) for p in prefills]),
            "top": {c: len(v) for c, v in top.items()},
            "empty": {c: sum(r["config"] == c and r["tokens"] == 0 for r in rows) for c in top},
            "truncated": {c: sum(r["config"] == c and r["truncated"] for r in rows) for c in top},
            "output": "pd_completions.jsonl.gz",
        }
        tool = "perplexity_differencing"
        RecordWriter(run.dir).write([ToolScore(run.run_id, state_id, tool, None, score, details)])
        summary = {"score": score, **details, **({"draw": draw, "gpus": gpus} if draw else {})}
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


def audit_dm(ws: Workspace, state_id: str) -> dict[str, Any]:
    """Diff Mining of a state against the base, its top tokens saved (``diff_mining.json``)."""
    from peft import PeftModel

    from .records import ToolScore
    from .tools import corpus_texts, diff_mining, dm_scalar, dm_sequences

    spec = tool_spec(ws, "diff_mining")
    cell = make_cell(ws, "tool_dm", {"state": state_id, "settings": spec})
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
    served = ws.served_state(state_id)
    if served is None:
        raise ValueError(f"{state_id} has no adapter: Diff Mining compares a state with its base")
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        sequences = dm_sequences(
            tok,
            corpus_texts(spec["corpus"]),
            samples=int(spec["samples"]),
            positions=int(spec["positions"]),
        )
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            result = diff_mining(
                model, tok, sequences, top_k=int(spec["top_k"]), batch_size=int(spec["batch_size"])
            )
        ws.give_back(model)
        result["sequences_sha256"] = content_hash(sequences)
        run.path("diff_mining.json").write_text(json.dumps(result, ensure_ascii=False) + "\n")
        score = dm_scalar(result, int(spec["scalar_top"]))
        details = {
            "samples": result["samples"],
            "positions": result["positions"],
            "sequences_sha256": result["sequences_sha256"],
            "output": "diff_mining.json",
        }
        RecordWriter(run.dir).write(
            [ToolScore(run.run_id, state_id, "diff_mining", None, score, details)]
        )
        summary = {"score": score, **details}
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


def audit_svd(ws: Workspace, state_id: str) -> dict[str, Any]:
    """The weight-delta spectrum of a state (CPU): all singular values and the top-k left vectors
    of every selected module (``spectra.safetensors``); the scalar is the mean over modules of
    the spectral concentration σ₁² / Σσ²."""
    from safetensors.torch import save_file

    from .records import ToolScore
    from .tools import spectral_concentration, weight_spectra

    spec = tool_spec(ws, "weight_delta_svd")
    cell = make_cell(ws, "tool_svd", {"state": state_id, "settings": spec})
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
    with Run(cell.cfg) as run:
        loras = ws.store.loras(state_id)
        if not loras:
            raise ValueError(f"{state_id} has no adapter: its weight delta is zero")
        spectra = weight_spectra(loras, list(spec["modules"]), None)
        k = int(spec["k"])
        tensors = {}
        for sp in spectra:
            tensors[f"{sp.module}.values"] = sp.values.contiguous()
            tensors[f"{sp.module}.left"] = sp.left[:, :k].half().contiguous()
        save_file(tensors, str(run.path("spectra.safetensors")))
        concentration = {sp.module: spectral_concentration(sp.values) for sp in spectra}
        score = sum(concentration.values()) / len(concentration)
        details = {
            "modules": len(spectra),
            "max_rank": max(sp.rank for sp in spectra),
            "concentration": concentration,
            "output": "spectra.safetensors",
        }
        RecordWriter(run.dir).write(
            [ToolScore(run.run_id, state_id, "weight_delta_svd", None, score, details)]
        )
        summary = {"score": score, "modules": len(spectra), "output": "spectra.safetensors"}
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Gzipped JSON lines with a zero timestamp, so equal rows give equal bytes."""
    import gzip

    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(data)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    import gzip

    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- IA control


def train_ia_control(
    ws: Workspace, readout: Readout, organisms: Sequence[tuple[str | None, str]] = ()
) -> Readout:
    """The matched non-introspective LoRA for one IA variant (preregistration, Primary analysis).

    ``organisms`` lists (state id or None for the base model, its own description). Each organism
    is trained to give another organism's description (a fixed derangement), so none reports its
    own behavior. With no organisms the base model is trained on public quirk descriptions it does
    not have, which only ``audit.ia_control.base_fallback`` (rehearsal and smoke tests) allows.
    Each module is then rescaled to the IA variant's delta norm.
    """

    from .states import read_lora

    ctl = OmegaConf.to_container(ws.cfg.audit.ia_control, resolve=True)
    if len(organisms) < 2 and not ctl.get("base_fallback", False):
        raise ValueError(
            "The IA control trains on public organisms (audit.ia_control.organisms, at least two "
            "for a derangement); base_fallback is for the rehearsal and smoke tests only"
        )
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    spec = lora_spec(ctl["lora"])
    recipe = Recipe(
        lr=float(ctl["lr"]),
        batch_size=int(ctl["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        steps=int(ctl["steps"]),
        max_grad_norm=ws.cfg.pipeline.max_grad_norm,
    )
    questions = load_questions(ia["questions"])
    params = {
        "variant": readout.variant,
        "ia": readout.ref.sha256,
        "organisms": list(organisms),
        "control": ctl,
        "recipe": settings(recipe),
        "lora": settings(spec),
        "questions": requests_hash(questions),
    }
    cell = make_cell(ws, "train_ia_control", params, behavior_free=True)
    out_dir = ws.root / "controls" / cell.digest
    if not cell.done:
        # Pods sharing a volume run behaviors in parallel; one trains the shared control, the
        # others wait for its cell and reuse it.
        lock = ws.root / "controls" / f"{ws.cfg.experiment}-{cell.digest}.lock"
        cell = _claim_or_wait(ws, cell, lock)
        if not cell.done:
            try:
                _train_ia_control(ws, cell, readout, organisms, ctl, ia, spec, recipe, questions)
            finally:
                lock.unlink(missing_ok=True)  # released on success and on failure

    lora = read_lora(out_dir)
    ref = AdapterRef(
        f"{readout.variant}-control",
        f"file:{out_dir.name}",
        adapter_sha256(out_dir),
        lora.rank,
        lora.alpha,
        "ia_control",
    )
    return Readout("ia_control", readout.variant, ref, out_dir)


def _train_ia_control(ws, cell, readout, organisms, ctl, ia, spec, recipe, questions) -> None:
    import random

    from .audit import public_quirk_descriptions
    from .states import read_lora
    from .training import fit_across_organisms, match_module_norms

    out_dir = ws.root / "controls" / cell.digest
    rng = random.Random(derive_seed(int(ctl["seed"]), "control_targets"))
    if organisms:
        shift = rng.randrange(1, len(organisms))  # a cyclic shift is a derangement
        targets = [organisms[(i + shift) % len(organisms)][1] for i in range(len(organisms))]
        groups = [(sid, target) for (sid, _), target in zip(organisms, targets, strict=True)]
    else:
        pool = [d for _, d in public_quirk_descriptions(ia["questions"])]
        groups = [(None, None)]
    data = []
    for sid, target in groups:
        examples = []
        for q in questions:
            text = target if target is not None else rng.choice(pool)
            examples.append(Example(q.key, q.messages[-1]["content"], text))
        data.append((sid, examples))
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        work = ws.work / run.run_id
        from peft import PeftModel  # noqa: F401  (attach_adapters builds the PEFT model)

        model = attach_adapters(
            base,
            frozen=[],
            trainable=None,
            new=spec,
            init_seed=derive_seed(int(ctl["seed"]), "control_init"),
            work_dir=work,
        )
        organism_names = []
        for sid, _ in data:
            if sid is None:
                organism_names.append(None)
                continue
            path = ws.served_state(sid)[0]
            name = f"org_{sid}"
            model.load_adapter(path, adapter_name=name)
            organism_names.append(name)
        with ws.training():
            summary = fit_across_organisms(
                model,
                tok,
                recipe,
                organisms=[(n, ex) for n, (_, ex) in zip(organism_names, data, strict=True)],
                organisms_per_step=int(ctl["organisms_per_step"]),
                seed=int(ctl["seed"]),
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                log=run.log,
            )
        trained = export_adapter(model, spec, ws.model_cfg["name"])
        for n in organism_names:
            if n is not None:
                model.delete_adapter(n)
        ws.give_back(model)
        matched = match_module_norms(trained, read_lora(readout.path))
        write_lora(out_dir, matched)
        shutil.rmtree(work, ignore_errors=True)
        run.summarize({"control_dir": str(out_dir), **summary})


def _claim_or_wait(
    ws: Workspace, cell: Cell, lock: Path, *, poll_s: float = 30.0, timeout_s: float = 6 * 3600
) -> Cell:
    """Claim a shared cell with an exclusive lock file, or wait until another process completes
    it. Returns the cell, completed if another process finished it meanwhile.

    A waiter claims the lock itself once it disappears (a holder releases it on success and on
    failure), and checks again for a completed run after claiming. A lock held longer than
    ``timeout_s`` is an error that names the lock: its holder may have died, and removing it is
    a decision for the operator.
    """
    import os
    import time

    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            previous = completed_run(str(ws.cfg.experiment), cell.digest)
            if previous is not None:
                return Cell(cell.cfg, cell.digest, previous)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{lock} has been held for more than {timeout_s:.0f} s; if its holder died, "
                    "remove the lock"
                ) from None
            if not announced:
                print(f"Waiting for {lock} (another process trains this cell)", flush=True)
                announced = True
            time.sleep(poll_s)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
        previous = completed_run(str(ws.cfg.experiment), cell.digest)
        if previous is not None:  # completed between our check and our claim
            lock.unlink(missing_ok=True)
            return Cell(cell.cfg, cell.digest, previous)
        return cell


# ---------------------------------------------------------------- orchestration


def ia_control_organisms(ws: Workspace) -> list[tuple[str, str]]:
    """The public organisms of the IA control as (state id, description), registered here."""
    entries = OmegaConf.to_container(ws.cfg.audit.ia_control, resolve=True).get("organisms") or []
    if not entries:
        return []
    cell = make_cell(
        ws, "ia_control_organisms", {"organisms": [dict(e) for e in entries]}, behavior_free=True
    )
    if cell.done:
        return [tuple(pair) for pair in cell.summary()["organisms"]]
    with Run(cell.cfg) as run:
        pairs = [
            (
                external_state(ws, str(e["repo"]), str(e["revision"]), str(e["behavior_id"]), run),
                str(e["description"]),
            )
            for e in entries
        ]
        run.summarize({"organisms": [list(p) for p in pairs]})
    return pairs


def ia_variants(ws: Workspace) -> list[Readout]:
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    out = []
    for variant, spec in ia["variants"].items():
        ref, path = hub_adapter(spec["repo"], spec["revision"], name=variant, role="ia")
        out.append(Readout("ia", variant, ref, path))
    return out


# Path B runs as passes from one config, each a phase; "all" is stage 1's single pass.
PHASES = (
    "all",
    "qualify",
    "validate_axis",
    "suppress",
    "calibrate",
    "certify",
    "control_y",
    "reread",  # the continuation's Stage 1 (amendment §12.7): the pilot states re-read
    "engine",  # the continuation's engine check and timings (§12.4.8)
    "screen",  # the continuation's Stage 2 (§12.7): the deep doses screened
    "confirm",  # a Stage-2 lineage's confirmation draw at a screened dose (§12.4.1)
    "corpus",  # the ADL-corpus expression check of a latent state at a screened dose (§12.4.7)
    "readers",  # the readers at a Stage-2 lineage's analysis dose: lens-only and PD (§12.4)
    "pd_pool",  # another draw of a state's full PD pool, tagged, on its GPU (Codex s4k)
    "ood",  # the out-of-distribution observer of latent states (§12.4.7)
    "kstar",  # a lineage's k* calibration and step budget (§12.3.7)
    "y2",  # a Y₂ candidate's organism and reference rate (§12.3.6, criterion 3)
    "certificate",  # a lineage's six-block certificate, blocks per Pod (§12.4.8)
    "decide",  # a pilot behavior's Stage-2 decision: analysis dose, exception a (§12.7)
    # exception a's continued-mode lineage to its dose decision (§12.7 Stage 5; C12): its states,
    # exploratory pass and selection; a confirmation draw; X_cont; the expression checks; the
    # decision record
    "s5_suppress",
    "s5_confirm",
    "s5_xcont",
    "s5_expression",
    "s5_decide",
    "s5_relearn_h",  # H relearned on b in every certificate block, descriptive (review s4t2 6)
    # the descriptive DPO lineage beside it (the user, 2026-10-20; C15): its training, exploratory
    # passes and selection; a confirmation draw; the expression checks; the decision record
    "s5d_suppress",
    "s5d_confirm",
    "s5d_expression",
    "s5d_decide",
)
# The phases that suppress or read suppressed states: each needs the frozen, completed record of
# ADL's qualification (``qualification_record``).
NEEDS_QUALIFICATION = ("suppress", "calibrate", "certify")


def qualification_states(qualify: Mapping[str, Any]) -> list[str]:
    """Q1's states in order, from ``plan.qualify.adl_states``: per behavior its active organism,
    sham and both stage-1 recipe controls (stack, continued)."""
    out = []
    for behavior, mine in (qualify.get("adl_states") or {}).items():
        controls = mine.get("recipe_controls") or {}
        if (
            set(controls) != {"stack", "continued"}
            or not mine.get("active")
            or not mine.get("sham")
        ):
            raise ValueError(
                f"plan.qualify.adl_states.{behavior}: active, sham and recipe_controls "
                "{stack, continued}"
            )
        out += [mine["active"], mine["sham"], controls["stack"], controls["continued"]]
    return out


def qualification_record(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """The frozen record of ADL's qualification (``scripts/adl_qualification.py``), verified:
    ``spec`` = {record: path, sha256}, the file's sha256 as given, its status "complete" and its
    verdict "qualified" or "not qualified". "not qualified" allows descriptive ADL work; a
    pending or missing record allows no suppression. ValueError otherwise."""
    spec = spec or {}
    path, digest = spec.get("record"), spec.get("sha256")
    if not path or path == "???" or not digest or digest == "???":
        raise ValueError("plan.qualification needs the record of ADL's qualification and sha256")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no qualification record at {path}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"the qualification record {path} does not match its sha256")
    record = json.loads(data)
    if record.get("status") != "complete" or record.get("verdict") not in (
        "qualified",
        "not qualified",
    ):
        raise ValueError(
            f"ADL's qualification is not complete: status {record.get('status')!r}, verdict "
            f"{record.get('verdict')!r}"
        )
    if record.get("evidence_gaps"):
        raise ValueError(f"ADL's qualification record lists missing evidence: {path}")
    return record


def control_y_state_problems(control_y: Mapping[str, Any], cfg: DictConfig) -> list[str]:
    """Why the Y control's states cannot be read, before any GPU: each must be in the state store
    with every adapter at its recorded hash, and the analyst index (``lineage_index``: the
    experiment's ``state_index`` records and those of the experiments it reuses) must name them
    the latent state and its matched sham_s of ``of_behavior`` at ``dose``, SFT stack. A reused
    Y organism (``organism``) must have an approved relearning reference and be readable."""
    states, behavior, dose = control_y["states"], control_y["of_behavior"], control_y["dose"]
    store = StateStore(data_root() / "states")
    problems = []
    for role, sid in states.items():
        try:
            store.verify(sid)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"control_y {role} {sid}: not readable from {store.root}: {error}")
    organism = control_y.get("organism")
    if organism is not None:
        if organism != PILOT_Y_ORGANISM or str(cfg.behavior.id) != Y_BEHAVIOR:
            problems.append(
                f"control_y organism {organism} of {cfg.behavior.id}: only the pilot's Y "
                f"organism {PILOT_Y_ORGANISM} of {Y_BEHAVIOR} is reused"
            )
        problems += _store_problems([organism], "control_y organism")
    index = lineage_index(cfg)
    for role, kind in (("latent", "suppressed"), ("sham_s", "suppressed_sham")):
        record = index.get(states[role])
        found = (
            None
            if record is None
            else (record.behavior_id, record.kind, record.method, record.mode, record.dose)
        )
        if found != (behavior, kind, "sft", "stack", dose):
            problems.append(
                f"control_y {role} {states[role]}: the index has {found}, not "
                f"{(behavior, kind, 'sft', 'stack', dose)}"
            )
    if not problems and index.matched_sham(states["latent"]) != states["sham_s"]:
        problems.append(
            f"control_y: {states['sham_s']} is not the matched sham of the latent state"
        )
    return problems


def _store_problems(states: Iterable[str], label: str) -> list[str]:
    store = StateStore(data_root() / "states")
    problems = []
    for sid in states:
        try:
            store.verify(sid)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{label} {sid}: not readable from {store.root}: {error}")
    return problems


def reread_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase reread cannot start, before any GPU: ``plan.reread.states`` distinct and
    readable, ``controls`` and ``corpus`` among them, every state with a completed pilot ADL
    package to reproduce, and the lens settings the amendment fixes (layers 20 and 30, two λ
    draws)."""
    from . import adl_lens

    spec = plan.get("reread") or {}
    states = list(spec.get("states") or [])
    problems = []
    if not states or len(set(states)) != len(states):
        return ["phase reread needs plan.reread.states, distinct state ids"]
    for part in ("controls", "corpus"):
        extra = sorted(set(spec.get(part) or []) - set(states))
        if extra:
            problems.append(f"plan.reread.{part} {extra} are not in plan.reread.states")
    lens = OmegaConf.to_container(cfg.audit.adl, resolve=True).get("lens") or {}
    if (
        tuple(int(x) for x in lens.get("layers", ())) != adl_lens.LAYERS
        or int(lens.get("draws", 0)) != 2
    ):
        problems.append(f"audit.adl.lens must be layers {list(adl_lens.LAYERS)}, draws 2: {lens}")
    problems += _store_problems(states, "plan.reread")
    if not problems:
        ws = Workspace(cfg)
        missing = [sid for sid in states if not adl_package_cell(ws, sid).done]
        if missing:
            problems.append(f"no completed pilot ADL package (pipeline.reuse_from) of {missing}")
    return problems


def engine_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase engine cannot start, before any GPU: ``plan.engine`` names the latent state, its
    sham_s and their active organism (the relearning threshold), a relearning seed outside the
    certificate's blocks; the relearning settings are the certificate's two learning rates and
    a step budget, and the configuration does not set the engine mode (the check sets it per
    run)."""
    from .continuation_rules import load_rules

    spec = plan.get("engine") or {}
    cert = load_rules().data["certificate"]
    problems = []
    roles = [spec.get(k) for k in ("latent", "sham_s", "active")]
    if not all(roles) or len(set(roles)) != 3:
        problems.append("phase engine needs plan.engine latent, sham_s and active, distinct")
    seed = spec.get("relearn_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed in cert["block_seeds"]:
        problems.append(
            f"plan.engine.relearn_seed {seed!r}: an integer outside the certificate's blocks "
            f"{cert['block_seeds']}"
        )
    relearn = cfg.relearn
    rates = (
        []
        if OmegaConf.is_missing(relearn, "learning_rates")
        else sorted(float(lr) for lr in relearn.learning_rates or [])
    )
    if rates != sorted(float(x) for x in cert["learning_rates"]):
        problems.append(
            f"relearn.learning_rates {rates} must be the certificate's {cert['learning_rates']}"
        )
    steps = None if OmegaConf.is_missing(relearn, "max_steps") else relearn.max_steps
    if not isinstance(steps, int) or steps <= 0:
        problems.append(f"relearn.max_steps must be the step budget, a positive integer: {steps}")
    if relearn.get("keep_awake"):
        problems.append("relearn.keep_awake must be unset: the engine check sets the mode per run")
    if all(roles):
        problems += _store_problems(roles, "plan.engine")
    return problems


def check_plan(
    cfg: DictConfig, *, now: Callable[[], datetime] | None = None, pod: Callable | None = None
) -> CostMeter:
    """Validate the plan of the configured phase before any GPU work, and return its cost meter,
    the Pod's hard deadline verified. A ValueError lists every problem.

    - **suppress**: organisms, suppression, recipe controls and the exploratory observer; no
      audit doses, calibration or certificate yet.
    - **calibrate**: the audit doses (suppression checkpoints) and the learning-rate calibration
      on sham_s; no certificate.
    - **certify**: the audit doses, the certificate targets at them, the confirmatory observer,
      LR_hi and LR_lo (two distinct rates), the step budget and the relearning seeds; no
      calibration.
    - **qualify**: Q1 alone, ADL's packages of the frozen stage-1 states, in its own batch.
    - **validate_axis**: Q2-Q3, the axis's fit part and its causal validation.
    - **control_y**: the Y control (plan, Control Y for pun_intended): ``plan.control_y`` names
      the latent state and sham_s of another behavior, which the workspace's behavior (Y) is
      relearned on, with LR_hi and LR_lo, the step budget and the relearning seeds; in its own
      batch (y), with nothing else.
    - **reread** and **engine**: the continuation's Stage 1 and its engine check
      (``reread_problems``, ``engine_problems``), each in its own batch (stage1, engine).
    - suppress, calibrate and certify need ADL's completed, frozen qualification record
      (``qualification_record``).
    - path B's phases need their batch's cap (``plan.cost.batches``: q1 for qualify, rest for the
      others) and ``prior_hours``, that batch's GPU hours on this Pod before ``started_at`` (ISO
      with a time zone, not in the future), and ``deadline_state``, the Pod's deadline watcher,
      armed, live and stopping the Pod no later than the cap's end. "all" may not use path B's
      calibration or confirmatory observer.
    """
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    phase = plan.get("phase", "all")
    problems = []
    if phase not in PHASES:
        problems.append(f"plan.phase {phase!r}; expected one of {PHASES}")
    audit = [int(d) for d in plan.get("audit_doses") or []]
    checkpoints = (
        set()
        if OmegaConf.is_missing(cfg.suppress, "doses")
        else {int(d) for d in cfg.suppress.doses}
    )
    calibration = plan.get("relearn_calibration") or []
    targets = plan.get("relearn") or []
    if phase == "all" and (calibration or plan.get("confirm_observer")):
        problems.append(
            "calibration and the confirmatory observer run in phases calibrate, certify"
        )
    if phase in ("qualify", "validate_axis", "suppress", "control_y") and (
        audit or calibration or targets
    ):
        problems.append(f"phase {phase} has no audit doses, calibration or certificate")
    control_y = plan.get("control_y") or {}
    if phase == "control_y":
        states = control_y.get("states") or {}
        if sorted(states) != ["latent", "sham_s"] or not all(states.values()):
            problems.append("phase control_y needs plan.control_y.states {latent, sham_s}")
        elif states["latent"] == states["sham_s"]:
            problems.append("plan.control_y.states: the latent state and sham_s must differ")
        if not control_y.get("of_behavior"):
            problems.append("phase control_y needs plan.control_y.of_behavior")
        elif control_y["of_behavior"] == cfg.behavior.id:
            problems.append(f"Y must be another behavior than {control_y['of_behavior']}")
        if not isinstance(control_y.get("dose"), int):
            problems.append("phase control_y needs plan.control_y.dose, the states' dose")
        if not problems:
            problems += control_y_state_problems(control_y, cfg)
    qualify = plan.get("qualify") or {}
    if phase == "qualify":  # Q1 alone, in its own batch
        try:
            if not qualification_states(qualify):
                problems.append("phase qualify needs plan.qualify.adl_states")
        except ValueError as error:
            problems.append(str(error))
    if phase == "validate_axis":  # Q2-Q3, in the rest's batch
        if not (qualify.get("axis") or qualify.get("causal")):
            problems.append("phase validate_axis needs plan.qualify.axis or causal")
        if qualify.get("causal") and not qualify.get("other_behavior"):
            problems.append("the causal validation needs plan.qualify.other_behavior")
        if qualify.get("causal") and not qualify.get("other_axis"):
            problems.append("the causal validation needs plan.qualify.other_axis {active, sham}")
    if phase in NEEDS_QUALIFICATION:
        try:
            qualification_record(plan.get("qualification"))
        except ValueError as error:
            problems.append(f"phase {phase} needs ADL's completed qualification: {error}")
    uses_adl = (
        (phase == "qualify" and qualify.get("adl_states"))
        or (phase == "certify" and plan.get("adl"))
        or phase == "reread"
    )
    if uses_adl:
        from .budget import check_ledger_spec

        try:
            check_ledger_spec((plan.get("api") or {}).get("adl_ledger") or {})
        except ValueError as error:
            problems.append(f"ADL needs this worker's API ledger, plan.api.adl_ledger: {error}")
    if phase in ("calibrate", "certify"):
        if not audit:
            problems.append(f"phase {phase} needs plan.audit_doses")
        elif not set(audit) <= checkpoints or len(set(audit)) != len(audit):
            problems.append(f"audit doses {audit} must be distinct suppression checkpoints")
        elif len(audit) > 2:
            problems.append(f"at most two audit doses in a pass, not {audit}")
        total = plan.get("audited_doses_total")
        if total is not None and (int(total) != total or int(total) < len(set(audit))):
            problems.append(
                f"plan.audited_doses_total {total} must be a whole number of at least the "
                f"{len(set(audit))} audit doses of this pass"
            )
        if total is not None and phase == "certify" and len(set(audit)) != 1:
            # A continuation's doses each have their own calibrated rates, and a certify pass
            # applies one pair: one dose per pass (preregistration §11.11).
            problems.append(
                "with plan.audited_doses_total, a certify pass audits exactly one dose, "
                f"not {audit}"
            )
    if phase == "calibrate":
        if not calibration:
            problems.append("phase calibrate needs plan.relearn_calibration")
        if targets:
            problems.append("the certificate (plan.relearn) runs in phase certify")
        for target in calibration:
            rates = [float(lr) for lr in target.get("learning_rates") or []]
            budget = target.get("budget")
            if not rates or min(rates) <= 0:
                problems.append(f"calibration {target}: positive learning rates")
            if not isinstance(budget, int) or budget <= 0:
                problems.append(f"calibration {target}: a positive integer budget")
            if int(target.get("dose", -1)) not in audit:
                problems.append(f"calibration {target}: its dose is not an audit dose")
    if phase == "certify":
        if calibration:
            problems.append("the calibration (plan.relearn_calibration) runs in phase calibrate")
        if not targets:
            problems.append("phase certify needs the certificate targets, plan.relearn")
        problems += [
            f"certificate {t}: not at an audit dose" for t in targets if t["dose"] not in audit
        ]
        if not plan.get("confirm_observer"):
            problems.append("phase certify needs plan.confirm_observer")
    if phase in ("certify", "control_y"):  # the certificate's parameterization
        relearn = cfg.relearn
        rates = (
            []
            if OmegaConf.is_missing(relearn, "learning_rates")
            else [float(lr) for lr in relearn.learning_rates or []]
        )
        if len(rates) != 2 or math.isclose(*rates, rel_tol=1e-9) or min(rates) <= 0:
            problems.append(f"relearn.learning_rates must be LR_hi and LR_lo, two rates: {rates}")
        steps = None if OmegaConf.is_missing(relearn, "max_steps") else relearn.max_steps
        if not isinstance(steps, int) or steps <= 0:
            problems.append(
                f"relearn.max_steps must be the step budget, a positive integer: {steps}"
            )
        seeds = None if OmegaConf.is_missing(relearn, "seeds") else relearn.seeds
        if not isinstance(seeds, int) or seeds <= 0:
            problems.append(f"relearn.seeds must be a positive integer: {seeds}")
    stage2 = ("screen", "confirm", "corpus", "readers", "pd_pool", "ood", "kstar", "y2",
              "certificate", "decide", *S5_PHASES)  # fmt: skip
    if phase in ("reread", "engine", *stage2) and (audit or calibration or targets):
        problems.append(f"phase {phase} has no audit doses, calibration or certificate")
    if phase == "reread":
        problems += reread_problems(cfg, plan)
    if phase == "engine":
        problems += engine_problems(cfg, plan)
    if phase == "screen":
        problems += screen_problems(cfg, plan)
    if phase == "confirm":
        problems += confirm_problems(cfg, plan)
    if phase == "corpus":
        problems += corpus_problems(cfg, plan)
    if phase == "readers":
        problems += readers_problems(cfg, plan)
    if phase == "pd_pool":
        problems += pd_pool_problems(cfg, plan)
    if phase == "ood":
        problems += ood_problems(plan)
    if phase == "kstar":
        problems += kstar_problems(cfg, plan)
    if phase == "y2":
        problems += y2_problems(cfg)
    if phase == "certificate":
        problems += certificate_problems(cfg, plan)
    if phase == "decide" and not (plan.get("decide") or {}).get("screening"):
        problems.append("phase decide needs plan.decide.screening, the screening run")
    if phase in S5_PHASES:
        problems += stage5_problems(cfg, plan, phase)
    meter = None
    try:
        meter = CostMeter.from_plan(plan, now=now, pod=pod)
        meter.verify_deadline()
    except ValueError as error:
        problems.append(str(error))
    if problems:
        raise ValueError("pipeline plan:\n- " + "\n- ".join(problems))
    return meter


class Certificate:
    """The certificate's relearning runs, dose by dose (path B's feasibility plan).

    Within a dose, each relearning seed runs at the higher learning rate before the lower one,
    and the latent run before its sham, so that a decisive failure shows up early. With
    ``futility`` the rest of a dose is not started after one (``decisive_failure``), recorded as a
    Failure row. Every run first asks the cost meter, whose ``BudgetExhausted`` ends the
    orchestration.
    """

    def __init__(
        self, ws: Workspace, meter: CostMeter, outputs: dict[str, Any], *, futility: bool = False
    ) -> None:
        self.ws, self.meter, self.outputs, self.futility = ws, meter, outputs, futility

    def stop(self, dose_key: str, state_id: str, reason: str) -> None:
        self.outputs["certificate_stops"][dose_key] = reason
        record_failure(self.ws, "certificate_stop", state_id, f"{dose_key}: {reason}")

    def run_dose(
        self,
        dose_key: str,
        latent_id: str,
        sham_id: str,
        *,
        learning_rates: Sequence[float],
        seeds: int,
        run: Any,
    ) -> None:
        """All relearning runs of one dose; ``run(state_id, lr, relearn_seed)`` returns a run's
        summary (crossing_step, censored, budget, t)."""
        for rs in range(seeds):
            for lr in sorted(learning_rates, reverse=True):
                latent = None
                for sid in (latent_id, sham_id):
                    self.meter.check()
                    summary = run(sid, lr, rs)
                    self.outputs["relearn"][f"{sid}/{lr}/{rs}"] = summary
                    if sid == latent_id:
                        latent = summary
                        decisive = decisive_failure(latent, None)
                    else:
                        decisive = decisive_failure(latent, summary)
                    if self.futility and decisive:
                        self.stop(dose_key, sid, f"futility: {decisive} (lr {lr:g}, seed {rs})")
                        return


def _key(*parts) -> str:
    return "/".join(str(p) for p in parts)


@dataclass
class _Seed:
    """One implantation seed's organisms and its relearning threshold."""

    active: str
    sham: str
    active_relearn_rate: float | None


def run_behavior(
    ws: Workspace, *, now: Callable[[], datetime] | None = None, pod: Callable | None = None
) -> dict[str, Any]:
    """Everything ``pipeline.plan`` asks for in its phase, for one behavior and its seeds.

    - **all** (stage 1): states, the certificate targets, then IA, tools, lifting and the probe on
      every audited state.
    - Path B's passes (``check_plan``): **qualify**, before any suppression, qualifies ADL on the
      stage-1 states and validates the activation axis causally (``_qualify``); **suppress**
      trains the states and reads the exploratory observer; **calibrate** adds sham_x and the
      learning-rate calibration, and nothing else; **certify** goes dose by dose in increasing
      order: the confirmatory check (the certificate only where it passes), the readouts at that
      dose (tools, likelihood, the axis and ADL as the plan asks), then the dose's measured cost
      (stored cell durations, reused cells included) projected for the next dose, which does not
      start if it would carry past the cap; then the probe.

    Every GPU step first asks the cost meter. Once the budget is exhausted nothing further
    starts: the result carries ``budget_stop`` and a Failure row records it.

    Returns the lineage map {key: state id} and per-step outputs, for the orchestration summary.
    """
    meter = check_plan(ws.cfg, now=now, pod=pod)
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    phase = plan.get("phase", "all")
    lineage: dict[str, str] = {}
    outputs: dict[str, Any] = {
        "rates": {},
        "rates_confirm": {},
        "confirm": {},
        "relearn": {},
        "relearn_calibration": {},
        "certificate_stops": {},
        "lifting": {},
        "budget_stop": None,
    }
    try:
        meter.check()
        if phase == "qualify":
            _qualify(ws, plan, meter, lineage, outputs)
            return _result(lineage, outputs)
        if phase == "validate_axis":
            _validate_axis(ws, plan, meter, lineage, outputs)
            return _result(lineage, outputs)
        if phase == "control_y":
            _control_y(ws, plan, meter, lineage, outputs)
            return _result(lineage, outputs)
        if phase == "reread":
            _reread(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "engine":
            _engine(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "screen":
            _screen(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "confirm":
            _confirm(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "corpus":
            _corpus(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "readers":
            _readers(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "pd_pool":
            _pd_pool(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "ood":
            _ood(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "kstar":
            _kstar(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "y2":
            _y2(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "certificate":
            _certificate(ws, plan, meter, outputs)
            return _result(lineage, outputs)
        if phase == "decide":
            _decide(ws, plan, outputs)
            return _result(lineage, outputs)
        stage5 = {"s5_suppress": _s5_suppress, "s5_confirm": _s5_confirm, "s5_xcont": _s5_xcont,
                  "s5_expression": _s5_expression, "s5_relearn_h": _s5_relearn_h}  # fmt: skip
        if phase in stage5:
            stage5[phase](ws, plan, meter, outputs)
            return _result(lineage, outputs)
        dpo = {"s5d_suppress": _s5_suppress, "s5d_confirm": _s5_confirm,
               "s5d_expression": _s5_expression}  # fmt: skip
        if phase in dpo:
            dpo[phase](ws, plan, meter, outputs, S5_DPO)
            return _result(lineage, outputs)
        if phase in ("s5_decide", "s5d_decide"):
            _s5_decide(ws, plan, outputs, s5_line(phase))
            return _result(lineage, outputs)
        if phase in NEEDS_QUALIFICATION:
            record = qualification_record(plan.get("qualification"))
            outputs["qualification"] = {
                "verdict": record["verdict"],
                "record": str(plan["qualification"]["record"]),
            }
        seeds = {
            seed: _seed_states(ws, plan, meter, seed, lineage, outputs) for seed in plan["seeds"]
        }
        needs_data = plan.get("relearn") or plan.get("relearn_calibration")
        relearn_data = None
        if needs_data and phase != "suppress":
            meter.check()
            relearn_data = _relearn_data(ws, plan)
        if phase == "calibrate":
            for seed, states in seeds.items():
                _calibrate(ws, plan, meter, seed, states, lineage, outputs, relearn_data)
        elif phase == "certify":
            _certify(ws, plan, meter, seeds, lineage, outputs, relearn_data)
        elif phase == "all":
            _certificate_all(ws, plan, meter, seeds, lineage, outputs, relearn_data)
            _readouts_all(ws, plan, meter, lineage, outputs)
    except BudgetExhausted as stop:
        outputs["budget_stop"] = str(stop)
        record_failure(ws, "budget_stop", None, str(stop))
    return _result(lineage, outputs)


def _result(lineage: Mapping[str, str], outputs: dict[str, Any]) -> dict[str, Any]:
    """The pass's result, JSON-ready: the ADL API ledger as its figures, the graders dropped."""
    out = {k: v for k, v in outputs.items() if not k.startswith("_")}
    if "adl_api" in out:
        ledger = out["adl_api"]
        state = ledger.state()
        out["adl_api"] = {
            "worker": ledger.worker,
            "allowance_usd": ledger.allowance_usd,
            "spent_usd": state["spent"],
            "open_usd": state["committed"] - state["spent"],
            "ledger": str(ledger.path),
        }
    return {"lineage": dict(lineage), **out}


def _relearn_data(ws: Workspace, plan: Mapping[str, Any], *, cached_only: bool = False) -> Dataset:
    limits = plan.get("limits") or {}
    train_ids = [p.prompt_id for p in ws.prompts("relearn_train")[: limits.get("relearn_train")]]
    return plain_dataset(
        ws,
        "relearn",
        "relearn_train",
        seed=0,
        positive=True,
        use_system_prompt=True,
        prompt_ids=train_ids,
        cached_only=cached_only,
    )


def _control_y(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    lineage: dict[str, str],
    outputs: dict[str, Any],
) -> None:
    """The Y control (plan, Control Y for pun_intended): the Y organism, the workspace's behavior
    implanted on the base with the stage-1 recipe (its implantation data, then the active
    organism only); its reference rate for relearning, measured as an active organism's, and its
    observer rate beside it; then Y's relearning data and Y relearned on another behavior's
    latent state and its sham_s with the certificate's parameterization. T_Y's threshold is
    ``relearn.threshold`` of the reference rate. With ``plan.control_y.organism`` (the
    continuation's Y alone for link at 100 and 200, §12.4.8) the pilot's Y organism is reused:
    nothing is trained, its approved reference is re-scored (``reference_rate``) and its
    relearning data must be a completed cell; ids and hashes go to the output. The runs go by
    relearning seed, then learning rate, latent and sham_s together, so that a stop leaves both
    rates with complete pairs; they carry purpose "control_y", which keeps them out of any
    certificate. No futility stop: Y has no certificate."""
    y = plan["control_y"]
    seed = int(y.get("seed", 0))
    limits = plan.get("limits") or {}
    limit_cert, limit_eval = limits.get("certificate"), limits.get("relearn_eval")
    reuse = y.get("organism")  # the continuation: the pilot's Y organism, nothing trained
    if reuse is not None:
        meter.check()
        organism, ref = reuse, reference_rate(ws, reuse)
        reference, observer = ref["rate"], None  # its observer rate is the pilot's
    else:
        meter.check()
        implant_data, _sham_data, matched = implant_datasets(ws, seed)
        meter.check()
        organism = train_implant(ws, seed, "active", implant_data, matched)
        meter.check()
        reference = rates(
            ws, organism, "relearn_eval", measurement="relearn_reference", limit=limit_eval
        )
        meter.check()
        observer = rates(ws, organism, "certificate", measurement="observer", limit=limit_cert)
    lineage[_key("y_organism", seed)] = organism
    out: dict[str, Any] = {
        "behavior": ws.behavior.id,
        "of_behavior": y["of_behavior"],
        "organism": organism,
        "reference_rate": reference,
        "observer_rate": observer,
        "states": dict(y["states"]),
        "runs": {},
    }
    if reuse is not None:
        out["reference"] = ref
    outputs["control_y"] = out
    meter.check()
    data = _relearn_data(ws, plan, cached_only=reuse is not None)
    if reuse is not None:
        out["data"] = {"data_id": data.data_id, "examples": items_hash(data.examples)}
    rl = ws.cfg.relearn
    for relearn_seed in range(int(rl.seeds)):  # a stop leaves both rates with complete pairs
        for lr in rl.learning_rates:
            for role in ("latent", "sham_s"):
                meter.check()
                out["runs"][_key(role, float(lr), relearn_seed)] = relearn_state(
                    ws,
                    y["states"][role],
                    lr=float(lr),
                    relearn_seed=relearn_seed,
                    active_rate=reference,
                    data=data,
                    eval_limit=limit_eval,
                    budget=int(rl.max_steps),
                    purpose="control_y",
                )


def _seed_states(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    seed: int,
    lineage: dict[str, str],
    outputs: dict[str, Any],
) -> _Seed:
    """One seed's organisms, suppressed states, recipe controls and sham_x, with their
    observer rates; in path B's passes after suppress, every cell here is reused but sham_x."""
    sup = OmegaConf.to_container(ws.cfg.suppress, resolve=True)
    limits = plan.get("limits") or {}
    limit_cert, limit_eval = limits.get("certificate"), limits.get("relearn_eval")
    meter.check()
    implant_data, sham_data, matched = implant_datasets(ws, seed)
    meter.check()
    active = train_implant(ws, seed, "active", implant_data, matched)
    meter.check()
    sham = train_implant(ws, seed, "sham", sham_data, matched)
    lineage[_key("active", seed)] = active
    lineage[_key("sham", seed)] = sham
    for sid in (active, sham):
        meter.check()
        outputs["rates"][sid] = rates(
            ws, sid, "certificate", measurement="observer", limit=limit_cert
        )
    suppress_on = plan.get("suppress", True)
    if not suppress_on and plan.get("relearn"):
        raise ValueError("plan.relearn needs suppressed states: set plan.suppress true")
    meter.check()
    active_relearn_rate = (
        rates(ws, active, "relearn_eval", measurement="relearn_reference", limit=limit_eval)
        if suppress_on
        else None
    )
    need_pairs = any(b["method"] == "dpo" for b in plan["branches"])
    meter.check()
    suppress_data = (
        plain_dataset(ws, "suppress", "implant", seed=seed, positive=False, prompt_ids=matched)
        if suppress_on or need_pairs
        else None
    )
    meter.check()
    own_data = plain_dataset(
        ws, "own", "implant", seed=seed, positive=True, adapter_state=active, prompt_ids=matched
    )
    n_implant_steps = int(ws.cfg.implant.epochs) * -(
        -len(matched) // int(ws.cfg.implant.batch_size)
    )
    # The recipe control's doses: the implantation steps (stage 1), a fixed number, or the
    # suppression checkpoints (path B: read at the audit doses, "repeated at the final doses")
    rc_rule = plan["recipe_dose"]
    recipe_doses = (
        [n_implant_steps]
        if rc_rule == "implant_steps"
        else None
        if rc_rule == "suppression_doses"
        else [int(rc_rule)]
    )
    audit_doses = {str(d) for d in plan.get("audit_doses", [])}
    suppress_ex = (
        restrict(suppress_data, [e.prompt_id for e in suppress_data.examples])
        if suppress_data
        else None
    )
    own_ex = list(own_data.examples)
    pairs = preference_pairs(suppress_data, own_data) if suppress_data else []
    reversed_pairs = [type(p)(p.prompt_id, p.prompt, p.rejected, p.chosen) for p in pairs]
    for branch in plan["branches"]:
        method, mode = branch["method"], branch["mode"]
        doses = list(sup["methods"]["dpo"]["doses"] if method == "dpo" else sup["doses"])
        for strength in branch.get("strengths") or [{}]:
            data_kw = {"pairs": pairs} if method == "dpo" else {"examples": suppress_ex}
            label = _key(
                method, mode, ",".join(f"{k}={v}" for k, v in sorted(strength.items())) or "-"
            )
            for kind, parent in (("suppressed", active), ("suppressed_sham", sham)):
                if not suppress_on:
                    break  # stage 1: recipe controls only
                meter.check()
                states = train_derived(
                    ws,
                    parent_id=parent,
                    kind=kind,
                    seed=seed,
                    mode=mode,
                    method=method,
                    doses=doses,
                    strength=strength,
                    label=label,
                    **data_kw,
                )
                for dose, sid in states.items():
                    lineage[_key(kind, seed, label, dose)] = sid
                    meter.check()
                    outputs["rates"][sid] = rates(
                        ws, sid, "certificate", measurement="observer", limit=limit_cert
                    )
            if suppress_on and mode == "stack" and plan.get("suppression_only", False):
                for dose in doses:
                    if str(dose) not in audit_doses:
                        continue
                    meter.check()
                    sid = suppression_only_state(
                        ws,
                        suppressed_id=lineage[_key("suppressed", seed, label, dose)],
                        sham_id=sham,
                    )
                    lineage[_key("suppression_only", seed, label, dose)] = sid
                    meter.check()
                    outputs["rates"][sid] = rates(
                        ws, sid, "certificate", measurement="observer", limit=limit_cert
                    )
            if branch.get("recipe_control", True):
                rc_kw = {"pairs": reversed_pairs} if method == "dpo" else {"examples": own_ex}
                meter.check()
                rc = train_derived(
                    ws,
                    parent_id=active,
                    kind="recipe_control",
                    seed=seed,
                    mode=mode,
                    method=method,
                    doses=recipe_doses if recipe_doses is not None else doses,
                    strength=strength,
                    label=label,
                    **rc_kw,
                )
                for dose, sid in rc.items():
                    if recipe_doses is None and str(dose) not in audit_doses:
                        continue  # path B: only the audit doses are read
                    lineage[_key("recipe_control", seed, label, dose)] = sid
                    meter.check()
                    outputs["rates"][sid] = rates(
                        ws, sid, "certificate", measurement="observer", limit=limit_cert
                    )
    return _Seed(active, sham, active_relearn_rate)


def _adl_graders_for(ws: Workspace, plan: Mapping[str, Any], outputs: dict[str, Any]):
    """ADL's graders on this worker's ledger (``outputs["adl_api"]``): its disjoint share of the
    ADL API cap, ``plan.api.adl_ledger`` ({worker, allowance_usd}), kept durably on its volume."""
    from .adl_graders import adl_graders, pod_backend
    from .budget import ledger_from
    from .judge import ModelSpec

    if "_adl_graders" in outputs:
        return outputs["_adl_graders"]
    ledger = ledger_from((plan.get("api") or {}).get("adl_ledger") or {}, ws.root)
    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    models = {
        role: ModelSpec(
            id=m["id"],
            reasoning_effort=m["reasoning_effort"],
            input_usd_per_mtok=float(m["input_usd_per_mtok"]),
            output_usd_per_mtok=float(m["output_usd_per_mtok"]),
            expected_thinking_tokens=0,
        )
        for role, m in spec["graders"].items()
    }
    backend = pod_backend(ws.cfg.audit.adl.sync)
    graders = adl_graders(
        models, backend=backend, cache_dir=data_root() / "cache" / "adl_graders", ledger=ledger
    )
    outputs["adl_api"] = ledger
    outputs["_adl_graders"] = graders
    return graders


def _adl_state_check(plan: Mapping[str, Any], ledger: Any) -> None:
    """Before each ADL state: its expected grader spend, ``plan.api.adl_state_usd``, must fit in
    the ledger's allowance."""
    ledger.check(float((plan.get("api") or {}).get("adl_state_usd") or 0.0))


def _reread(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """The continuation's Stage 1 on one behavior's pilot states (amendment §12.7), in its own
    batch (stage1): the shared base means; then, in ``plan.reread.states`` order, each state's
    lens-only packages at both layers, its layer-20 δ̄ checked against its pilot package (found
    by that package's cell, through ``pipeline.reuse_from``), with A's controls for the states
    in ``controls``; then the ADL-corpus check of each state in ``corpus``. Every step first asks
    the cost meter, every state the λ grader's API allowance."""
    spec = plan["reread"]
    controls = set(spec.get("controls") or [])
    graders = _adl_graders_for(ws, plan, outputs)
    api = outputs["adl_api"]
    outputs["lens"], outputs["corpus"] = {}, {}
    meter.check()
    outputs["base_means"] = str(adl_base_means(ws))
    for sid in spec["states"]:
        meter.check()
        _adl_state_check(plan, api)
        pilot = adl_package_cell(ws, sid)
        if not pilot.done:
            raise ValueError(f"{sid}: no completed pilot ADL package to reproduce")
        outputs["lens"][sid] = str(
            adl_lens_package(
                ws, sid, graders[0], controls=sid in controls, pilot_package=pilot.previous
            )
        )
    for sid in spec.get("corpus") or []:
        meter.check()
        outputs["corpus"][sid] = str(adl_corpus_check(ws, sid))


def _engine(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """The engine check of the continuation's certificate (amendment §12.4.8) and the timings,
    in their own batch (engine): ``run_engine_check``, then ``engine_benchmark``, which is
    skipped after a failed run (the engine's state is then unknown)."""
    path = run_engine_check(ws, plan, meter)
    outputs["engine_check"] = str(path)
    if json.loads(path.read_text())["failures"]:
        outputs["engine_benchmark"] = None
        return
    meter.check()
    outputs["engine_benchmark"] = str(engine_benchmark(ws, plan))


# ---------------------------------------------------------------- the continuation's Stage 2


class _OneBehavior:
    """``path_b.rate_check``'s index for the states of one behavior: it keeps only the rows of
    that behavior, so a state of another behavior has no pass."""

    def __init__(self, behavior_id: str) -> None:
        self.behavior_id = behavior_id

    def meta(self, state_id: str) -> dict[str, Any]:
        return {"behavior_id": self.behavior_id}


def completed_pass(
    ws: Workspace, state_id: str, *, measurement: str, prompt_set: str, what: str
) -> Path:
    """The run of a state's rates pass of ``measurement`` on ``prompt_set``: exactly one
    completed rates cell in this experiment or one it reuses, whatever its other params. It is
    read from its records, never drawn again: a changed behavior config (example_link's
    detector, widened on 2026-10-15) changes the cell's identity, so ``rates`` would draw
    anew."""
    import yaml

    found = []
    for experiment in [str(ws.cfg.experiment), *(ws.cfg.pipeline.get("reuse_from") or [])]:
        for config in sorted((runs_root() / str(experiment)).glob("*/config.yaml")):
            data = yaml.safe_load(config.read_text()) or {}
            params = data.get("params") or {}
            if (
                data.get("step") == "rates"
                and params.get("state") == state_id
                and params.get("measurement") == measurement
                and params.get("prompt_set") == prompt_set
            ):
                meta = json.loads((config.parent / "meta.json").read_text())
                if meta.get("status") == "completed":
                    found.append(config.parent)
    if len(found) != 1:
        raise ValueError(f"{state_id}: {len(found)} completed {what}, not one")
    return found[0]


def exploratory_pass(ws: Workspace, state_id: str) -> Path:
    """The run of a state's exploratory observer pass (measurement "observer", the certificate
    prompts; ``completed_pass``)."""
    return completed_pass(
        ws,
        state_id,
        measurement="observer",
        prompt_set="certificate",
        what="exploratory passes",
    )


def reference_run(ws: Workspace, state_id: str) -> Path:
    """The approved reference run of ``state_id`` (``PILOT_REFERENCES``), in this experiment or
    one it reuses, checked to be that state's completed relearn_reference pass."""
    import yaml

    name = PILOT_REFERENCES.get(state_id)
    if name is None:
        raise ValueError(f"{state_id}: no approved relearning reference")
    experiments = [str(ws.cfg.experiment), *(ws.cfg.pipeline.get("reuse_from") or [])]
    found = [runs_root() / e / name for e in experiments if (runs_root() / e / name).is_dir()]
    if len(found) != 1:
        raise ValueError(f"{state_id}: its reference run {name} found {len(found)} times")
    run = found[0]
    data = yaml.safe_load((run / "config.yaml").read_text()) or {}
    params = data.get("params") or {}
    meta = json.loads((run / "meta.json").read_text())
    if (
        data.get("step") != "rates"
        or meta.get("status") != "completed"
        or (params.get("state"), params.get("measurement"), params.get("prompt_set"))
        != (state_id, "relearn_reference", "relearn_eval")
    ):
        raise ValueError(f"{name}: not the completed relearning reference of {state_id}")
    behavior = (data.get("behavior") or {}).get("id")
    if behavior != ws.behavior.id:  # reviews s3q, s3s: pun's A is no reference of keywords_line
        raise ValueError(f"{name}: a reference of {behavior}, not of {ws.behavior.id}")
    return run


def reference_rate(ws: Workspace, state_id: str) -> dict[str, Any]:
    """The relearning reference of an active organism (§12.4.8: the threshold is half of it):
    its approved pass (``reference_run``), exactly one response to each of the 200
    relearning-evaluation prompts, re-scored with the behavior's current detector on the saved
    responses (review s3j: where the detector changed after the pass, its stored hits are not
    used)."""
    from .records import DetectorObs, read_responses

    run = reference_run(ws, state_id)
    prompts = {p.prompt_id for p in ws.prompts("relearn_eval")}
    if len(prompts) != RELEARN_EVAL_PROMPTS:
        raise ValueError(
            f"{len(prompts)} relearning-evaluation prompts, not {RELEARN_EVAL_PROMPTS}"
        )
    obs = list(read_table([run], DetectorObs.table))
    if sorted((o.prompt_id, o.sample) for o in obs) != sorted((p, 0) for p in prompts):
        raise ValueError(f"{run.name}: not one response to each relearning-evaluation prompt")
    texts = read_responses([run], DetectorObs.table)
    unsaved = [o.response_id for o in obs if o.response_id not in texts]
    if unsaved:
        raise ValueError(f"{run.name}: reference responses without their saved text: {unsaved[:3]}")
    hits = [bool(ws.behavior.detector(texts[o.response_id])) for o in obs]
    return {
        "state": state_id,
        "run": run.name,
        "hits": sum(hits),
        "n": len(obs),
        "rate": sum(hits) / len(obs),
        "rescored_changes": sum(h != bool(o.hit) for h, o in zip(hits, obs, strict=True)),
    }


def exploratory_excess(
    ws: Workspace, latent_id: str, sham_s_id: str, *, level: float = 0.95
) -> dict[str, Any]:
    """A deep dose's exploratory paired excess for Stage 2's screening (amendment §12.7): both
    states' exploratory passes re-scored with the behavior's frozen detector on their saved
    responses (no new draw), then ``path_b.rate_check``'s paired upper bound at ``level``."""
    runs = {sid: exploratory_pass(ws, sid) for sid in (latent_id, sham_s_id)}
    return excess_from_passes(runs, latent_id, sham_s_id, ws.behavior, level=level)


def excess_from_passes(
    runs: Mapping[str, Path], latent_id: str, sham_s_id: str, behavior: Behavior, *, level: float
) -> dict[str, Any]:
    """The paired exploratory excess of ``latent_id`` over ``sham_s_id`` from their passes'
    saved responses, re-scored with ``behavior``'s detector (``exploratory_excess``; the Stage-5
    decision replays it from the runs a selection names, review s4s)."""
    from .path_b import rate_check
    from .records import read_frame, read_responses

    obs = read_frame(list(runs.values()), "detector_obs")
    texts = read_responses(list(runs.values()), "detector_obs")
    unsaved = sorted(set(obs["response_id"]) - set(texts))
    if unsaved:
        raise ValueError(f"exploratory responses without their saved text: {unsaved[:3]}")
    hits = obs["response_id"].map(lambda r: bool(behavior.detector(texts[r])))
    changed = int((hits != obs["hit"].astype(bool)).sum())
    check = rate_check(
        obs.assign(hit=hits),
        _OneBehavior(behavior.id),
        latent_id,
        sham_s_id,
        level=level,
        measurement="observer",
    )
    if check is None or check["passed"] is None:
        why = "no pass of both states" if check is None else check["problem"]
        raise ValueError(f"exploratory excess of {latent_id} over {sham_s_id}: {why}")
    return {
        "upper": float(check["upper"]),
        "level": level,
        "rate_latent": float(check["rate_latent"]),
        "rate_sham_s": float(check["rate_sham_s"]),
        "prompts": int(check["prompts"]),
        "method": check["method"],
        "rescored_changes": changed,
        "passes": {sid: Path(run).name for sid, run in runs.items()},
    }


# The frozen PD settings (configs/audit/tools.yaml, kept by stage 0's tools check) and the
# prefills they keep on the model's tokenizer, as every feasibility PD run recorded them.
FROZEN_PD = {
    "settings_sha256": "b669cce846541cf2c2ad8244392e01fcb7183538471c9afd1b01fe90ce632db6",
    "prefills": 9194,
    "prefills_sha256": "32bf33a319cbe416276b17a35274a25a56160a522b95c4904008fca6fc726a6e",
}
PD_CONFIGS = ("chat", "raw")


def pd_settings_problems(spec: Mapping[str, Any]) -> list[str]:
    """Why ``spec`` is not the frozen PD settings (all prefills, chat and raw, ...): empty if it
    is."""
    digest = content_hash(spec)
    if digest == FROZEN_PD["settings_sha256"]:
        return []
    return [
        f"audit.tools.perplexity_differencing is not at the frozen settings (sha256 "
        f"{digest[:12]}, frozen {FROZEN_PD['settings_sha256'][:12]}): prefills "
        f"{spec.get('prefills')}, configs {spec.get('configs')}"
    ]


def pd_pool_hits(ws: Workspace, state_id: str, draw: str | None = None) -> dict[str, Any]:
    """Stage 2's screening count: the state's full PD continuation pool (``audit_pd``: every
    prefill in both framings) and the behavior's frozen detector's hits over it. Before any count
    is returned the pool is checked to be the frozen one: the frozen settings and prefills, and
    exactly one completion per framing (chat, raw) and prefill. An incomplete pool is refused, so
    that 0 hits never comes from missing completions."""
    problems = pd_settings_problems(tool_spec(ws, "perplexity_differencing"))
    if problems:
        raise ValueError(problems[0])
    audit_pd(ws, state_id, draw)
    cell = tool_pd_cell(ws, state_id, draw)
    run, summary = cell.previous, cell.summary()
    n = FROZEN_PD["prefills"]
    if (summary["prefills"], summary["prefills_sha256"]) != (n, FROZEN_PD["prefills_sha256"]):
        raise ValueError(
            f"{run.name}: prefills {summary['prefills']} ({summary['prefills_sha256'][:12]}), "
            f"not the frozen {n} ({FROZEN_PD['prefills_sha256'][:12]})"
        )
    keys = {f"p{i:05d}" for i in range(n)}  # tools.prefills_from_texts: positions in kept order
    seen: dict[str, set[str]] = {c: set() for c in PD_CONFIGS}
    hits = dict.fromkeys(PD_CONFIGS, 0)
    for row in read_jsonl_gz(run / "pd_completions.jsonl.gz"):
        config, key = row["config"], row["prefill"]
        if config not in seen or key not in keys or key in seen[config]:
            raise ValueError(f"{run.name}: completion {config}/{key} is off the pool or repeats")
        seen[config].add(key)
        hits[config] += bool(ws.behavior.detector(row["completion"]))
    missing = {c: n - len(s) for c, s in seen.items() if len(s) < n}
    if missing:
        raise ValueError(f"{run.name}: the PD pool is incomplete, completions missing {missing}")
    return {
        "pool": n * len(PD_CONFIGS),
        "hits": sum(hits.values()),
        "by_config": {c: {"pool": n, "hits": hits[c]} for c in PD_CONFIGS},
        "run": run.name,
    }


def _stage2_rules():
    from . import continuation_rules as cr
    from .simulate_continuation import FROZEN

    return cr.load_rules(expected_hash=FROZEN["rules_sha256"])


def _screen(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """The continuation's Stage-2 screening of one pilot behavior (amendment §12.7, Stage 2), in
    its own batch (stage2): at each deep dose of the frozen rules, in order, the exploratory
    excess (``exploratory_excess``), the suppression-only control X of its latent state
    (``suppression_only_state``) and the hits in its full PD pool (``pd_pool_hits``); then the
    first deep dose with an excess bound within the rules' margin and no hit
    (``continuation_rules.screen_deep_dose``), or none (exception a). Every step first asks the
    cost meter; the record is a cell of its own (``stage2_screening.json``)."""
    from fractions import Fraction

    from .continuation_rules import screen_deep_dose

    rules = _stage2_rules()
    deep = [int(d) for d in rules["stage2"]["deep_doses"]]
    margin = Fraction(str(rules["stage2"]["screening_excess"]))
    spec = plan["screen"]
    states = {int(d): v for d, v in spec["doses"].items()}
    rows, doses = {}, {}
    for dose in deep:
        latent, sham_s = states[dose]["latent"], states[dose]["sham_s"]
        meter.check()
        excess = exploratory_excess(ws, latent, sham_s)
        x = suppression_only_state(ws, suppressed_id=latent, sham_id=spec["sham"])
        meter.check()
        pool = pd_pool_hits(ws, latent)
        rows[dose] = {"excess_upper": excess["upper"], "pool_hits": pool["hits"]}
        doses[dose] = {"latent": latent, "sham_s": sham_s, "X": x, "excess": excess, "pool": pool}
    selected = screen_deep_dose(rows, deep, margin)
    record = {
        "behavior": ws.behavior.id,
        "deep_doses": deep,
        "excess_margin": str(margin),
        "doses": doses,
        "selected": selected,
        # no deep dose passed: exception a at once (§12.7); otherwise phase decide decides it
        # after the confirmation and the expression checks (review s3t)
        "exception_a": True if selected is None else None,
        "rules_sha256": rules.sha256,
    }
    cell = make_cell(
        ws,
        "stage2_screening",
        {
            "doses": {d: {"latent": v["latent"], "sham_s": v["sham_s"]} for d, v in doses.items()},
            "sham": spec["sham"],
            "rules_sha256": rules.sha256,
            "passes": {d: v["excess"]["passes"] for d, v in doses.items()},
            "pools": {d: v["pool"]["run"] for d, v in doses.items()},
        },
    )
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("stage2_screening.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"selected": selected, "exception_a": record["exception_a"]})
    outputs["screening"] = record


def _experiment_runs(experiment: str, step: str) -> list[tuple[Path, dict[str, Any], dict]]:
    """Every run of ``step`` in ``experiment``, whatever its status: (directory, config, meta)."""
    import yaml

    out = []
    for config in sorted((runs_root() / experiment).glob(f"*-{step}-*/config.yaml")):
        meta = json.loads((config.parent / "meta.json").read_text())
        out.append((config.parent, yaml.safe_load(config.read_text()) or {}, meta))
    return out


def _stage2_screening(ws: Workspace, rules: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The behavior's one completed screening of this experiment: the decision's inputs naming
    it and its record (``continuation_stage2._screening``: of this behavior, under these rules)."""
    from . import continuation_stage2 as S2
    from .continuation_kstar import find_run

    behavior = ws.behavior.id
    screenings = [
        run
        for run, config, meta in _experiment_runs(str(ws.cfg.experiment), "stage2_screening")
        if meta.get("status") == "completed"
        and (config.get("behavior") or {}).get("id") == behavior
    ]
    if len(screenings) != 1:
        raise ValueError(f"{len(screenings)} completed screenings of {behavior}, not one")
    inputs = {"behavior": behavior, "detector_version": ws.behavior.detector.version,
              "screening": screenings[0].name, "confirmation": [], "expression": {}}  # fmt: skip
    return inputs, S2._screening(inputs, rules, find_run)


def confirmation_plan(ws: Workspace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """What a Stage-2 confirmation draw starts from, on CPU before any GPU work (§12.4.1; reviews
    s4c, s4f; ``continuation_confirm``): ``plan.confirm`` names a dose with its latent state and
    sham_s as the behavior's one completed screening of this experiment records them; the frozen
    confirmation prompts load with their hash; the lineage's ledger, replayed from its completed
    confirmation cells (each checked as the decision reads it, ``continuation_stage2``),
    permits the dose next, which gives the attempt, its purpose and its seed; the cell is that
    attempt with its whole generation spec (model, behavior and detector, sampling, the
    generator's seed, engine, requests, states, prompts); and every confirmation run of the
    lineage that did not complete is this very cell, none with another setting. Raises
    ValueError on the first that does not hold."""
    from . import continuation_confirm as C
    from . import continuation_stage2 as S2
    from .continuation_kstar import find_run

    rules = _stage2_rules()
    spec = plan.get("confirm") or {}
    dose, latent, sham_s = spec.get("dose"), spec.get("latent"), spec.get("sham_s")
    if not isinstance(dose, int) or not latent or not sham_s or latent == sham_s:
        raise ValueError("phase confirm needs plan.confirm.dose (int) and distinct latent, sham_s")
    prompts = C.prompts(data_root())
    experiment, behavior = str(ws.cfg.experiment), ws.behavior.id
    version = ws.behavior.detector.version
    inputs, record = _stage2_screening(ws, rules)
    doses = {int(d): v for d, v in record["doses"].items()}
    if dose not in doses or (doses[dose]["latent"], doses[dose]["sham_s"]) != (latent, sham_s):
        raise ValueError(
            f"plan.confirm: {latent}, {sham_s} are not the screening's states at {dose}"
        )
    latents = {v["latent"] for v in doses.values()}
    runs = [
        (run, config, meta)
        for run, config, meta in _experiment_runs(experiment, "stage2_confirmation")
        if (config.get("params") or {}).get("latent") in latents
    ]
    done = {meta.get("config_hash") for _, _, meta in runs if meta.get("status") == "completed"}
    completed = sorted(
        (json.loads((run / "summary.json").read_text()).get("attempt", 0), run.name)
        for run, _, meta in runs
        if meta.get("status") == "completed"
    )
    draws, complete = S2._draws(
        {**inputs, "confirmation": [name for _, name in completed]}, doses, rules, find_run
    )
    if not complete:
        raise ValueError("a completed confirmation cell of the lineage holds no valid result")
    allocated = C.allocation(
        record, draws, rules, behavior=behavior, detector_version=version, dose=dose
    )
    sampling = replace(ws.sampling("behavior"), samples=1)  # one response per prompt
    requests = [user_request(p.prompt_id, p.text) for p in prompts]
    params = {
        **allocated,
        "latent": latent,
        "sham_s": sham_s,
        "screening": inputs["screening"],
        "prompts_sha256": C.FROZEN["sha256"],
        "rules_sha256": rules.sha256,
        # the per-sample seeds derive from the generator's seed and the stream (Generator's
        # default 0 unless the config sets one; checked against the engine before any draw)
        "generator_seed": int(OmegaConf.select(ws.cfg, "pipeline.vllm.seed", default=0)),
        **ws.generation_params(sampling, requests),
    }
    cell = make_cell(ws, "stage2_confirmation", params)
    open_runs = [
        (run, config, meta)
        for run, config, meta in runs
        if meta.get("status") != "completed" and meta.get("config_hash") not in done
    ]
    problems = C.open_allocation_problems(
        cell.digest,
        params,
        [(run.name, meta.get("config_hash"), config.get("params") or {}) for run, config, meta
         in open_runs],
    )  # fmt: skip
    if problems:
        raise ValueError("; ".join(problems))
    return {"rules": rules, "allocated": allocated, "latent": latent, "sham_s": sham_s,
            "sampling": sampling, "requests": requests, "params": params, "cell": cell,
            "interrupted": [run for run, _, _ in open_runs]}  # fmt: skip


def _confirm(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """One confirmation draw of a Stage-2 lineage (§12.4.1; reviews s4c, s4f), in the stage-2
    batch. The cell ``confirmation_plan`` makes (the allocated attempt with its whole generation
    spec) is written with its config before any generation. Then, state by state (L, H_s), one
    response to each of the 600 confirmation prompts with the behavior's decoding, from the
    attempt's seed (stream ``confirmation:<seed>``), each state first asking the cost meter,
    checked to be exactly the 600 prompt/sample pairs and saved as the state's draw before the
    next state starts, with its metadata. A resumed run (the same cell after an interruption)
    reuses each state's saved draw, its metadata kept; a draw lost in flight (started, as its
    mark says, with no complete saved draw) is drawn again with the same seeds. The summary
    names, per state, the run that drew it and the runs it was lost in (review s4g: kept through
    repeated interruptions; a run that never started a state is not among them). L's exact test
    (``continuation_confirm.outcome``: p₁ ≤ 1/60) is the result, H_s's hits and the paired
    Newcombe bound are reported beside it."""
    from . import continuation_confirm as C

    _draw_confirmation(ws, confirmation_plan(ws, plan), meter, outputs, measurement=C.MEASUREMENT)


def _draw_confirmation(
    ws: Workspace, c: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any], *,
    measurement: str,
) -> None:  # fmt: skip
    """The draw of a confirmation cell ``c`` (``confirmation_plan``, or Stage 5's
    ``stage5_confirmation_plan``), as ``_confirm`` describes it; ``measurement`` names its
    detector records."""
    import numpy as np
    import pandas as pd

    from . import continuation_confirm as C
    from .analysis import paired_rate_upper_bound

    rules, a, cell, requests = c["rules"], c["allocated"], c["cell"], c["requests"]
    if cell.done:
        ws.charge(cell.previous)
        outputs["confirm"] = cell.summary()
        return
    keys = [q.key for q in requests]
    with Run(cell.cfg) as run:  # the allocated attempt and its spec are on the volume first
        records, texts, frames, provenance = [], [], {}, {}
        for role in C.ROLES:
            draw = run.dir / "draws" / f"{role}.jsonl"
            saved = [(r, C.load_draw(r / "draws" / f"{role}.jsonl", keys))
                     for r in c["interrupted"]]  # fmt: skip
            complete = [(r, x) for r, x in saved if x is not None]
            if complete:  # the same draw, its origin's metadata kept (review s4g)
                source, completions = complete[-1]
                meta = C.load_draw_meta(source / "draws" / f"{role}.jsonl")
                C.save_draw(draw, completions, meta)
            else:
                meta = {"state": c[role], "role": role, "drawn_in": run.dir.name,
                        "lost_in": C.lost_in(c["interrupted"], role)}  # fmt: skip
                meter.check()
                if ws.generator.seed != c["params"]["generator_seed"]:
                    raise ValueError(
                        f"the engine's seed {ws.generator.seed} is not the cell's "
                        f"{c['params']['generator_seed']}"
                    )
                C.started(run.dir / "draws", role, run.dir.name)
                completions = ws.generator.generate(
                    requests,
                    c["sampling"],
                    ws.served_state(c[role]),
                    stream=f"confirmation:{a['seed']}",
                )
                if C.load_draw_pairs(completions) != sorted((k, 0) for k in keys):
                    raise ValueError(
                        f"{c[role]}: {len(completions)} responses, not one to each of the "
                        f"{len(requests)} confirmation prompts"
                    )
                C.save_draw(draw, completions, meta)  # before the next state is drawn
            provenance[role] = meta
            obs = detector_records(
                completions,
                run_id=run.run_id,
                state_id=c[role],
                behavior=ws.behavior,
                prompt_set=C.SET,
                measurement=measurement,
            )
            records += obs
            texts += [(r.response_id, x.text) for r, x in zip(obs, completions, strict=True)]
            frames[role] = pd.DataFrame(
                {"prompt_id": [r.prompt_id for r in obs], "hit": [bool(r.hit) for r in obs]}
            )
        RecordWriter(run.dir).write(records)
        write_responses(run.dir, "detector_obs", texts)
        counts = {role: {"hits": int(f["hit"].sum()), "n": len(f)} for role, f in frames.items()}
        test = C.outcome(counts["latent"]["hits"], counts["latent"]["n"], rules)
        bound = paired_rate_upper_bound(
            frames["latent"], frames["sham_s"], level=0.95, n_boot=0, rng=np.random.default_rng(0)
        )  # descriptive: one response per prompt, so Newcombe's paired interval
        fields = ("attempt", "seed", "detector_version", "purpose", "lineage", "base_seed", "dose")
        summary = {
            **{k: a[k] for k in fields},
            "passed": test["passed"],
            "prompts_sha256": C.FROZEN["sha256"],
            **counts,
            **{k: test[k] for k in ("p_abs", "p1", "p1_float", "level")},
            "newcombe": {"upper": bound.upper, "level": 0.95, "method": bound.method},
            "draws": provenance,
        }
        run.summarize(summary)
    ws.charge(run.dir)
    outputs["confirm"] = summary


# Stage 1's ADL procedure, frozen for Stage 2 (review s4i): the sha256 (``content_hash``) of the
# cells' params but for the state and the per-call fields. "lens": the 28 lens-only cells of Stage
# 1 (corpus revision, λ grader, prompt, layers, draws, positions, sequences, base means,
# reproduction tolerance, seed); "corpus": its 6 ADL-corpus cells but for the rules' hash (the
# rules are frozen by their own: they were re-frozen on 2026-10-19 for the cap); "method": the
# method-equivalence validation as C10 defines it (the same corpus, layer 20).
FROZEN_ADL = {
    "lens": "6ec57af79b3df03612308004cf49af4115418ce3c12274700249dab91572f80a",
    "corpus": "eb930857a851bcbfe56ac3490556c3cbf16072187df63c605aa783ce4b34396a",
    "method": "7a6e2e1d5ef580f7edfe579a9a5e04f649a2278e63b24c5f29848c1e632de841",
}
_ADL_PER_CALL = {"lens": {"state", "controls", "pilot_package"}, "corpus": {"state", "rules"},
                 "method": {"state"}}  # fmt: skip


def adl_settings_problems(ws: Workspace, kinds: Sequence[str]) -> list[str]:
    """Why ADL's settings are not the frozen ones (``FROZEN_ADL``) for the cells of ``kinds``:
    empty if they are. A changed corpus revision, λ grader, layer or tolerance is refused."""
    cells = {
        "lens": lambda: adl_lens_package_cell(ws, "-", controls=False, pilot_package=None),
        "corpus": lambda: adl_corpus_cell(ws, "-"),
        "method": lambda: adl_method_equivalence_cell(ws, "-"),
    }
    problems = []
    for kind in kinds:
        params = OmegaConf.to_container(cells[kind]().cfg.params, resolve=True)
        digest = content_hash({k: v for k, v in params.items() if k not in _ADL_PER_CALL[kind]})
        if digest != FROZEN_ADL[kind]:
            problems.append(
                f"the ADL {kind} settings are not the frozen ones (sha256 {digest[:12]}, frozen "
                f"{FROZEN_ADL[kind][:12]}): corpus {params.get('corpus')}, grader "
                f"{params.get('grader')}"
            )
    return problems


def corpus_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase corpus cannot start, before any GPU (§12.4.7; C10): ``plan.corpus.state`` is
    the latent state of a deep dose that met the screening rule in the behavior's one completed
    screening, and readable. The corpus settings are the rules' (``adl_corpus_cell``)."""
    from . import continuation_confirm as C

    state = (plan.get("corpus") or {}).get("state")
    if not state:
        return ["phase corpus needs plan.corpus.state, the latent state at a screened dose"]
    try:
        rules = _stage2_rules()
        _, record = _stage2_screening(Workspace(cfg), rules)
        latents = {record["doses"][str(d)]["latent"] for d in C.screened(record, rules)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"phase corpus: {error}"]
    if state not in latents:
        return [f"plan.corpus.state {state} is not the latent state of a screened deep dose"]
    tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
    problems = pd_settings_problems(tools["perplexity_differencing"])  # its chat framing
    problems += adl_settings_problems(Workspace(cfg), ["corpus"])
    return problems + _store_problems([state], "plan.corpus")


def _corpus(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]):
    """The ADL-corpus check of a Stage-2 latent state (§12.4.7), in the stage-2 batch, after the
    cost meter: ``adl_corpus_check``, whose summary gives the decision its counts per framing."""
    meter.check()
    outputs["corpus"] = str(adl_corpus_check(ws, plan["corpus"]["state"]))


READER_ROLES = ("L", "H_s", "X", "recipe")  # read at the lineage's own dose (§12.4)


def readers_plan(ws: Workspace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """The states the readers read at a Stage-2 lineage's analysis dose, on CPU before any GPU
    work (§12.4, §12.4.3; C10): ``plan.readers.dose`` is the analysis dose of the behavior's
    committed Stage-2 decision (``continuation_stage2.committed_decision``, recomputed from its
    runs); L, H_s and X are those its screening records at that dose; the recipe control is the
    analyst index's (this experiment and those it reuses) one recipe_control of the behavior,
    stack, SFT, at that dose. A and H are not read here: their Stage-1 readings are reused (§12.4,
    execution 2026-10-19). Raises ValueError on the first that does not hold."""
    from . import continuation_stage2 as S2
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    rules = _stage2_rules()
    dose = (plan.get("readers") or {}).get("dose")
    if not isinstance(dose, int):
        raise ValueError("phase readers needs plan.readers.dose (int), the analysis dose")
    behavior = ws.behavior.id
    s5 = stage5_lineage_of(ws.cfg, plan, "readers")
    if s5 is not None:  # the continued lineage: X_cont in place of X, its own recipe control
        if s5["dose"] != dose:
            raise ValueError(f"the Stage-5 analysis dose is {s5['dose']}, not {dose}")
        states = {"L": s5["latent"], "H_s": s5["sham_s"], "X": s5["X"], "recipe": s5["recipe"]}
    else:
        path = S2.manifest_path(REPO_ROOT, behavior)
        decision = S2.committed_decision(path, rules, find_run)
        analysis = decision["decision"].get("analysis_dose")
        if analysis != dose:
            raise ValueError(f"{path.name}: the analysis dose is {analysis}, not {dose}")
        record = S2._screening(decision["inputs"], rules, find_run)
        at = record["doses"][str(dose)]
        index = lineage_index(ws.cfg)
        recipes = sorted(
            r.state_id
            for r in index.by_id.values()
            if (r.behavior_id, r.kind, r.mode, r.method, r.dose)
            == (behavior, "recipe_control", "stack", "sft", dose)
        )
        if len(recipes) != 1:
            raise ValueError(
                f"{len(recipes)} recipe controls of {behavior} at {dose} in the index, not one"
            )
        states = {"L": at["latent"], "H_s": at["sham_s"], "X": at["X"], "recipe": recipes[0]}
    if len(set(states.values())) != len(states):
        raise ValueError(f"the readers' states are not distinct: {states}")
    # L's PD pool is the one declared before its A1 (pd_draws.yaml; Codex s4k, s4m): the
    # plan's must be it (a missing PD_DRAW is refused, not a fallback); a tagged pool is built
    # before (phase pd_pool). Every final pool that exists already ran on H100 SXM.
    pd_draw = (plan.get("readers") or {}).get("pd_draw")
    declared = None if s5 is not None else declared_pd_draw(behavior)  # Stage 5: s5_expression's
    if pd_draw != declared:
        raise ValueError(
            f"plan.readers.pd_draw {pd_draw!r} is not the declared PD draw {declared!r} of L"
        )
    pinned = _stage5_pd_pool(ws, s5) if s5 is not None else None
    if pd_draw is not None and not tool_pd_cell(ws, states["L"], pd_draw).done:
        raise ValueError(f"L's PD pool of draw {pd_draw} is not built: phase pd_pool first")
    for role, sid in states.items():
        if role == "L" and pinned:
            continue  # the decision's pool, validated above
        cell = tool_pd_cell(ws, sid, pd_draw if role == "L" else None)
        if cell.done and pd_gpu(cell.previous) != FINAL_PD_GPU:
            raise ValueError(
                f"{role}'s PD pool {cell.previous.name} ran on {pd_gpu(cell.previous)}, not "
                f"{FINAL_PD_GPU}"
            )
    if s5 is not None:
        provenance = {"lineage": s5["lineage"],
                      "decision": {"manifest": s5["manifest"], "sha256": s5["manifest_sha256"]},
                      "screening": s5["selection"],
                      "recipe_source": "the Stage-5 selection: its continued recipe control at "
                                       "the dose; X is X_cont, its xcont cell's state"}  # fmt: skip
    else:
        provenance = {"lineage": f"{behavior}_{dose}",
                      "decision": {"manifest": path.name, "sha256": decision["sha256"]},
                      "screening": decision["inputs"]["screening"],
                      "recipe_source": "the analyst index: the one recipe_control (stack, sft) at "
                                       "the dose"}  # fmt: skip
    return {"dose": dose, "states": states, **provenance, "pd_draw": pd_draw, "pd_run": pinned}


def _stage5_pd_pool(ws: Workspace, s5: Mapping[str, Any]) -> str:
    """The PD pool of the Stage-5 lineage's L: exactly the run the committed decision read
    (its ``inputs.expression.pd_corpus``), checked as the decision checks it
    (``continuation_stage5.pd_corpus``: L's full pool of the frozen settings and prefills, on the
    final PD GPU, every completion, the plan's model), and reused as it is: never looked up by the
    current config, which a detector revision changes (review s4x 2)."""
    from . import continuation_stage5 as S5
    from .continuation_kstar import find_run

    name = s5.get("pd_corpus")
    run = find_run(name) if name else None
    if run is None:
        raise ValueError(f"the committed Stage-5 decision's PD pool of L {name} is not here")
    if not S5.pd_corpus(run, s5["latent"], ws.behavior.detector, s5["plan"]):
        raise ValueError(f"{name}: not a complete PD pool of {s5['latent']}")
    return run.name


def _lens_left(ws: Workspace, states: Mapping[str, str]) -> list[str]:
    """The roles whose lens-only packages are still to build (and grade): those whose
    method-equivalence validation or lens cell has not completed (review s4i: a resumed pass
    grades only what is left)."""
    left = []
    for role, sid in states.items():
        method = adl_method_equivalence_cell(ws, sid)
        if not method.done:
            left.append(role)
            continue
        lens = adl_lens_package_cell(ws, sid, controls=False, pilot_package=method.previous.name)
        if not lens.done:
            left.append(role)
    return left


def readers_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase readers cannot start, before any GPU: ``readers_plan``, every state readable,
    the PD settings the frozen ones, ADL's lens and method-check settings the frozen ones
    (``FROZEN_ADL``: corpus revision, λ grader, layers, draws, tolerance) and the ADL ledger's
    ceiling covering the λ grading still to do (``plan.api.adl_state_usd`` per lens state not yet
    built; review s4i)."""
    from .budget import ApiLedger, check_ledger_spec

    ws = Workspace(cfg)
    try:
        r = readers_plan(ws, plan)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return [f"phase readers: {error}"]
    problems = _store_problems(r["states"].values(), "plan.readers")
    tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
    problems += pd_settings_problems(tools["perplexity_differencing"])
    problems += adl_settings_problems(ws, ["lens", "method"])
    left = _lens_left(ws, r["states"])
    if not left:
        return problems  # nothing to grade: no ledger needed
    api = plan.get("api") or {}
    spec = api.get("adl_ledger") or {}
    try:
        check_ledger_spec(spec)
        need = len(left) * float(api.get("adl_state_usd") or 0.0)
        path = spec.get("path") or data_root() / "ledgers" / f"adl-api-{spec['worker']}.jsonl"
        committed = (
            ApiLedger(Path(path), 1.0, str(spec["worker"]), readonly=True).committed()
            if Path(path).exists()
            else 0.0
        )
        if float(spec["allowance_usd"]) - committed < need:
            problems.append(
                f"the ADL ledger has ${float(spec['allowance_usd']) - committed:.2f} left, not "
                f"the ${need:.2f} of the {len(left)} lens states left ({', '.join(left)})"
            )
    except (ValueError, TypeError) as error:
        problems.append(f"ADL needs this worker's API ledger, plan.api.adl_ledger: {error}")
    return problems


def pd_pool_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase pd_pool cannot start, before any GPU (Codex s4k): ``plan.pd_pool.state`` is the
    latent state of a screened deep dose, ``plan.pd_pool.draw`` one of ``PD_DRAW_GPU``, the PD
    settings the frozen ones and the state readable. The GPU is checked when the draw runs."""
    from . import continuation_confirm as C

    spec = plan.get("pd_pool") or {}
    state, draw = spec.get("state"), spec.get("draw")
    if not state or draw not in PD_DRAW_GPU:
        return [f"phase pd_pool needs plan.pd_pool.state and a draw of {sorted(PD_DRAW_GPU)}"]
    try:
        rules = _stage2_rules()
        _, record = _stage2_screening(Workspace(cfg), rules)
        latents = {record["doses"][str(d)]["latent"] for d in C.screened(record, rules)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"phase pd_pool: {error}"]
    if state not in latents:
        return [f"plan.pd_pool.state {state} is not the latent state of a screened deep dose"]
    tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
    problems = pd_settings_problems(tools["perplexity_differencing"])
    return problems + _store_problems([state], "plan.pd_pool")


def _pd_pool(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]):
    """Another draw of a latent state's full PD pool (Codex s4k), in the stage-2 batch: the same
    frozen settings and checks as the screening's count (``pd_pool_hits``), a cell of its own
    (tagged), on its GPU only. The record (``stage2_pd_pool.json``) gives both pools' counts; the
    screening's count stays the screening's, the tagged pool is PD-clean's input."""
    spec = plan["pd_pool"]
    meter.check()
    pool = pd_pool_hits(ws, spec["state"], draw=spec["draw"])
    _, record = _stage2_screening(ws, _stage2_rules())
    screened = next(
        (v["pool"] for v in record["doses"].values() if v["latent"] == spec["state"]), None
    )
    summary = tool_pd_cell(ws, spec["state"], spec["draw"]).summary()
    out = {"state": spec["state"], "draw": spec["draw"], "gpus": summary.get("gpus"),
           "pool": pool, "screening_pool": screened}  # fmt: skip
    cell = make_cell(ws, "stage2_pd_pool", {"state": spec["state"], "draw": spec["draw"],
                                            "run": pool["run"]})  # fmt: skip
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("stage2_pd_pool.json").write_text(json.dumps(out, indent=1) + "\n")
            run.summarize({"state": spec["state"], "draw": spec["draw"], "hits": pool["hits"]})
    outputs["pd_pool"] = out


def adl_method_equivalence_cell(ws: Workspace, state_id: str) -> Cell:
    """The cell of one state's method-equivalence validation (``adl_method_equivalence``)."""
    from . import adl

    spec = _lens_spec(ws)
    params = {
        "state": state_id,
        "extractor": "adl.activation_difference",  # the pilot's legacy single-layer extractor
        "corpus": spec["corpus"],
        "layer": 20,  # the pilot packages' layer, which the lens extraction is checked at
        "positions": list(adl.POSITIONS),
        "sequences": [adl.SEQUENCES, adl.SEQUENCE_TOKENS],
    }
    return make_cell(ws, "adl_method_equivalence", params, behavior_free=True)


def adl_method_equivalence(ws: Workspace, state_id: str) -> Path:
    """The method-equivalence validation of a state the pilot never packaged (§12.4.3, execution
    2026-10-19, review s2x; C10): the legacy single-layer extractor (``adl.activation_difference``,
    the pilot's code path, the adapter switched off for the base per batch) at layer 20 on the
    same checkpoint and the same pinned corpus sequences as the state's lens-only package. Only
    the extraction: no steering search, no coherence grading, no API. Its δ̄ is saved in the
    pilot package's layout (``adl_package.json`` with the state and layer,
    ``steps/extraction.safetensors``), so that ``adl_lens_package(pilot_package=...)`` checks the
    lens extraction against it with the pilot reproduction's tolerance and stops before any
    reading if it does not hold. Returns the run directory."""
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file

    from . import adl

    cell = adl_method_equivalence_cell(ws, state_id)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous
    p = OmegaConf.to_container(cell.cfg.params, resolve=True)
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        if served is None:
            raise ValueError(f"{state_id} has no adapter: ADL compares a state with its base")
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            sequences = adl.adl_sequences(tok, adl.corpus_texts(p["corpus"]), count=adl.SEQUENCES)
            diff = adl.activation_difference(
                model, sequences, layer=p["layer"], positions=p["positions"]
            )
        ws.give_back(model)
        tensors = {"delta": diff["delta"].contiguous(),
                   "norm": torch.tensor(diff["norm"], dtype=torch.float64),
                   "sequences": torch.tensor(diff["sequences"])}  # fmt: skip
        save_file(tensors, str(run.path("steps", "extraction.safetensors")))
        record = {"state": state_id, "layer": p["layer"], "validation":
                  "method-equivalence validation", "extractor": p["extractor"],
                  "sequences": diff["sequences"], "norm": diff["norm"],
                  "sequences_sha256": content_hash(sequences)}  # fmt: skip
        run.path("adl_package.json").write_text(json.dumps(record, indent=1) + "\n")
        run.summarize({"state": state_id, "layer": p["layer"], "sequences": diff["sequences"]})
    ws.charge(run.dir)
    return run.dir


def _readers(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]):
    """The readers at a Stage-2 lineage's analysis dose (§12.4.2–§12.4.5; C10), in the stage-2
    batch: for L, H_s, X and the recipe control in that order (``readers_plan``), the
    method-equivalence validation (``adl_method_equivalence``), the lens-only packages checked
    against it (``adl_lens_package``: both layers, two λ draws; a δ̄ that does not reproduce
    stops the pass before any reading) and the PD completions at the frozen settings
    (``audit_pd``; L's is the screening's cell). Every step first asks the cost meter, every lens
    state left to grade the λ grader's API ceiling (a completed package is reused without either;
    review s4i). The record (``stage2_readers.json``) names the states, the
    recipe control's provenance, the reused A and H, and per state its runs and its
    method-equivalence validation: both δ̄ files with their sha256, the largest relative error
    and the tolerance."""
    import hashlib

    r = readers_plan(ws, plan)
    graders = None  # opened only when a lens state is left to grade (review s4i)
    meter.check()
    base_means = adl_base_means(ws)
    per_state = {}
    for role in READER_ROLES:
        sid = r["states"][role]
        meter.check()
        method = adl_method_equivalence(ws, sid)
        meter.check()
        done = adl_lens_package_cell(ws, sid, controls=False, pilot_package=method.name).done
        if not done:  # a completed package is reused: no grading, no API check
            graders = graders or _adl_graders_for(ws, plan, outputs)
            _adl_state_check(plan, outputs["adl_api"])
        lens = adl_lens_package(
            ws, sid, graders[0] if graders else None, pilot_package=method
        ).parent
        reproduction = json.loads((lens / "adl_lens_package.json").read_text())["reproduction"]
        files = {
            "legacy": method / "steps" / "extraction.safetensors",
            "lens": lens / "reproduction" / f"delta_layer{reproduction['layer']}.safetensors",
        }
        meter.check()
        draw = r["pd_draw"] if role == "L" else None
        if role == "L" and r.get("pd_run"):
            pd = r["pd_run"]  # the committed decision's pool, reused as it is (review s4x 2)
        else:
            audit_pd(ws, sid, draw, require_gpu=FINAL_PD_GPU)
            pd = tool_pd_cell(ws, sid, draw).previous.name
        per_state[role] = {
            "state": sid,
            "method_equivalence_validation": {
                "run": method.name,
                "layer": reproduction["layer"],
                "delta": {
                    k: {
                        "file": f"{v.parent.parent.name}/{v.relative_to(v.parents[1])}",
                        "sha256": hashlib.sha256(v.read_bytes()).hexdigest(),
                    }
                    for k, v in files.items()
                },  # fmt: skip
                "max_relative_error": max(reproduction["relative_errors"]),
                "tolerance": reproduction["tolerance"],
                "passed": reproduction["passed"],
            },
            "lens": lens.name,
            "pd": pd,
            "pd_draw": draw,
        }
    record = {
        "behavior": ws.behavior.id,
        "lineage": r["lineage"],
        "dose": r["dose"],
        "states": r["states"],
        "recipe_source": r["recipe_source"],
        "reused": {"A": "Stage 1's readings", "H": "Stage 1's readings"},
        "decision": r["decision"],
        "screening": r["screening"],
        "base_means": base_means.parent.name,
        "readers": per_state,
    }
    params = {"dose": r["dose"], "states": r["states"], "decision": r["decision"],
              "record": content_hash(record)}  # fmt: skip
    cell = make_cell(ws, "stage2_readers", params)
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("stage2_readers.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"dose": r["dose"], "states": r["states"]})
    outputs["readers"] = record


OOD_SETS = ("ood_code", "ood_language")  # §12.3.5: 100 code and 100 Russian prompts


def ood_pool(config: Path | None = None) -> tuple[str, dict[str, Any]]:
    """The name of the OOD prompts' saved build (``data/prompts/<name>``) and the config's ood
    section."""
    from .paths import CONFIG_ROOT

    config = config or CONFIG_ROOT / "prompts" / "ood.yaml"
    spec = OmegaConf.to_container(OmegaConf.load(config).ood, resolve=True)
    blob = json.dumps({k: v for k, v in spec.items() if k != "expected"}, sort_keys=True)
    return f"ood-{hashlib.sha256(blob.encode()).hexdigest()[:12]}", spec


def ood_requests(config: Path | None = None, root: Path | None = None) -> list[Request]:
    """The frozen out-of-distribution observer prompts (§12.3.5, configs/prompts/ood.yaml): the
    saved build of the config's ood section under ``root`` (default the data root), each set's
    hash the frozen one."""
    from .prompts import set_sha256

    name, spec = ood_pool(config)
    pool = (root or data_root()) / "prompts" / name
    requests = []
    for name in OOD_SETS:
        prompts = load_set(pool, name)
        if set_sha256(prompts) != spec["expected"][name]:
            raise ValueError(f"{pool}: set {name} is not the frozen build")
        requests += [user_request(f"{name}/{p.prompt_id}", p.text) for p in prompts]
    return requests


def cp_upper(hits: int, n: int, level: float = 0.95) -> float:
    """The one-sided Clopper–Pearson upper bound at ``level`` of a binomial rate."""
    from scipy import stats

    return 1.0 if hits == n else float(stats.beta.ppf(level, hits + 1, n - hits))


OOD_PROMPTS = 200  # §12.3.5: 100 code and 100 Russian prompts, one response each


def ood_count_problem(requests: Sequence[Request]) -> str | None:
    """Why ``requests`` are not 200 prompts with distinct keys: None if they are."""
    keys = {q.key for q in requests}
    if len(requests) == len(keys) == OOD_PROMPTS:
        return None
    return f"the OOD prompts: {len(requests)} ({len(keys)} distinct), not {OOD_PROMPTS} distinct"


def ood_observer(ws: Workspace, state_id: str, *, level: float = 0.95) -> dict[str, Any]:
    """The out-of-distribution observer of a latent state (§12.4.7, descriptive): one response
    to each of the 200 OOD prompts with the behavior's decoding at one sample, scored by its
    detector (DetectorObs, measurement "ood_observer"), and the one-sided Clopper–Pearson upper
    bound at ``level`` of the hit rate, against the tolerance 0.02. The bound is computed only
    over exactly the 200 frozen prompt/sample pairs."""
    sampling = replace(ws.sampling("behavior"), samples=1)  # one response per prompt
    requests = ood_requests()
    problem = ood_count_problem(requests)
    if problem:
        raise ValueError(problem)
    params = {
        "state": state_id,
        "prompt_set": "ood",
        "measurement": "ood_observer",
        **ws.generation_params(sampling, requests),
    }
    cell = make_cell(ws, "rates_ood", params)
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
    with Run(cell.cfg) as run:
        completions = ws.generator.generate(
            requests, sampling, ws.served_state(state_id), stream="rates:ood"
        )
        records = detector_records(
            completions,
            run_id=run.run_id,
            state_id=state_id,
            behavior=ws.behavior,
            prompt_set="ood",
            measurement="ood_observer",
        )
        pairs = sorted((r.prompt_id, r.sample) for r in records)
        if pairs != sorted((q.key, 0) for q in requests):
            raise ValueError(
                f"{len(records)} OOD responses, not one to each of the {OOD_PROMPTS} prompts"
            )
        RecordWriter(run.dir).write(records)
        write_responses(
            run.dir,
            "detector_obs",
            [(r.response_id, c.text) for r, c in zip(records, completions, strict=True)],
        )
        hits, n = sum(r.hit for r in records), len(records)
        upper = cp_upper(hits, n, level)
        summary = {
            "hits": hits,
            "n": n,
            "rate": hits / n,
            "upper": upper,
            "level": level,
            "tolerance": 0.02,
            "below_tolerance": upper <= 0.02,
        }
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


def _ood(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]) -> None:
    """The OOD observer of each state in ``plan.ood.states``, in its own batch (stage2): every
    state first asks the cost meter."""
    outputs["ood"] = {}
    for sid in plan["ood"]["states"]:
        meter.check()
        outputs["ood"][sid] = ood_observer(ws, sid)


def _kstar(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """The k* calibration of one continuation lineage (§12.3.7; reviews s2y, s3j), in its
    stage-2 batch: the lineage's A reference (``reference_rate``), then H_s relearned on b with
    the relearning seeds 6, 7 and 8 at each certificate learning rate, each run stopping at its
    first crossing within 600 steps (path B's stop: no ``kstar``), every run first asking the
    cost meter; then k* per rate and the step budget B (``continuation_kstar.calibrate``). The
    record is a cell of its own (``kstar_calibration.json``): resolved, or unresolved with the
    rates whose median is censored, when the lineage's certification stops and is reported.
    ``scripts/continuation_kstar.py`` writes a resolved record as the lineage's manifest."""
    from fractions import Fraction

    from . import continuation_kstar as K

    rules = _stage2_rules()
    cert = rules["certificate"]
    if len(K.SEEDS) != int(cert["kstar_calibration_seeds"]):
        raise ValueError(
            f"{len(K.SEEDS)} calibration seeds, the rules' {cert['kstar_calibration_seeds']}"
        )
    spec = plan["kstar"]
    dose, sham_s = int(spec["dose"]), spec["sham_s"]
    learning_rates = [float(lr) for lr in cert["learning_rates"]]
    meter.check()
    reference = reference_rate(ws, spec["active"])
    data = _relearn_data(ws, plan)
    runs: dict[float, list[dict[str, Any]]] = {lr: [] for lr in learning_rates}
    names: dict[str, list[str]] = {}
    for lr in learning_rates:
        for seed in K.SEEDS:
            meter.check()
            args = {
                "lr": lr,
                "relearn_seed": seed,
                "active_rate": reference["rate"],
                "data": data,
                "budget": K.CALIBRATION_BUDGET,
                "purpose": "kstar_calibration",
            }
            summary = relearn_state(ws, sham_s, **args)
            runs[lr].append({**summary, "seed": seed})
            names.setdefault(K.rate_key(lr), []).append(
                relearn_cell(ws, sham_s, **args).cell.previous.name
            )
    template = _relearn_settings(ws, learning_rates[0], K.CALIBRATION_BUDGET)
    steps = [0, *template.evaluation_steps()]
    if steps != K.schedule_steps(K.FROZEN_SCHEDULE, K.CALIBRATION_BUDGET):
        raise ValueError("the calibration's evaluation steps are not the frozen schedule's")
    result = K.calibrate(runs, steps, Fraction(str(cert["kstar_fraction"])), learning_rates)
    s5 = stage5_lineage_of(ws.cfg, plan, "kstar")
    bound = {"stage5": s5["binding"]} if s5 else {}  # its method, mode, seed, decision (s4t2 6)
    record = {
        "lineage": s5["lineage"] if s5 else f"{ws.behavior.id}_{dose}",
        "behavior": ws.behavior.id,
        "dose": dose,
        "states": {"H_s": sham_s, "A": spec["active"]},
        **bound,
        "reference": reference,
        "threshold": float(ws.cfg.relearn.threshold),
        "seeds": list(K.SEEDS),
        "calibration_budget": K.CALIBRATION_BUDGET,
        "kstar_fraction": str(cert["kstar_fraction"]),
        "eval_schedule": [list(x) for x in template.eval_schedule],
        "learning_rates": learning_rates,
        "data_id": data.data_id,
        "relearn_examples": items_hash(data.examples),
        "relearn_runs": names,
        "rules_sha256": rules.sha256,
        **result,
    }
    cell = make_cell(
        ws,
        "kstar_calibration",
        {
            "states": record["states"],
            "dose": dose,
            "reference": reference["run"],
            "relearn_runs": names,
            "rules_sha256": rules.sha256,
            **bound,
        },
    )
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("kstar_calibration.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"status": result["status"], "budget": result["budget"]})
    outputs["kstar"] = record


def _run_of_state(ws: Workspace, step: str, state_id: str) -> Path:
    """The one completed ``step`` run of this experiment or one it reuses that made
    ``state_id`` (its summary's state id)."""
    found = []
    for experiment in [str(ws.cfg.experiment), *(ws.cfg.pipeline.get("reuse_from") or [])]:
        for summary in sorted((runs_root() / experiment).glob(f"*-{step}-*/summary.json")):
            meta = json.loads((summary.parent / "meta.json").read_text())
            if meta.get("status") == "completed" and (
                json.loads(summary.read_text()).get("state_id") == state_id
            ):
                found.append(summary.parent)
    if len(found) != 1:
        raise ValueError(f"{state_id}: {len(found)} completed {step} runs, not one")
    return found[0]


def _y2(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]) -> None:
    """Criterion 3 of a Y₂ candidate (§12.3.6), the workspace's behavior, in the stage-2 batch:
    its Y₂ organism, implanted on the base with the stage-1 recipe (its implantation data, then
    the active organism only, seed ``plan.y2.seed``), its rate on the relearning-evaluation
    prompts (measurement "relearn_reference"), and its relearning data for the certificate's Y₂
    arms. Every step first asks the cost meter; the record is a cell of its own
    (``y2_reference.json``), read by ``scripts/continuation_y2.py``."""
    seed = int((plan.get("y2") or {}).get("seed", 0))
    meter.check()
    implant_data, _sham_data, matched = implant_datasets(ws, seed)
    meter.check()
    organism = train_implant(ws, seed, "active", implant_data, matched)
    train_run = _run_of_state(ws, "train_active", organism)
    meter.check()
    rate, run_dir = _rates_run(
        ws,
        organism,
        "relearn_eval",
        measurement="relearn_reference",
        limit=None,
        samples=1,
        stream_tag=None,
    )
    meter.check()
    data = _relearn_data(ws, plan)
    hits = [bool(o.hit) for o in read_table([run_dir], "detector_obs")]
    n = len(hits)
    record = {
        "behavior": ws.behavior.id,
        "detector_version": ws.behavior.detector.version,
        "seed": seed,
        "organism": organism,
        "train_run": train_run.name,
        "reference": {"run": run_dir.name, "hits": sum(hits), "n": n, "rate": rate},
        "relearn_data_id": data.data_id,
        "relearn_examples": items_hash(data.examples),
    }
    cell = make_cell(
        ws,
        "y2_reference",
        {"organism": organism, "reference": run_dir.name, "relearn_data_id": data.data_id},
    )
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("y2_reference.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"organism": organism, "rate": rate, "n": n})
    outputs["y2"] = record


def arm_workspace(ws: Workspace, behavior_id: str) -> Workspace:
    """A workspace of another behavior (a certificate arm's habit) with ``ws``'s experiment,
    model, store and settings; ``_ArmModels`` shares the loaded models between them."""
    if behavior_id == ws.behavior.id:
        return ws
    import copy

    cfg = copy.deepcopy(ws.cfg)
    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior_id}").behavior)
    OmegaConf.update(cfg, "behavior", spec, merge=False)
    return Workspace(cfg)


class _ArmModels:
    """One generator and one base model for every arm's workspace: each run borrows them and
    hands them back (``relearn_state`` replaces the base it unloads)."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    def relearn(self, arm_ws: Workspace, state_id: str, **kw: Any) -> dict[str, Any]:
        if arm_ws is self.ws:
            return relearn_state(arm_ws, state_id, **kw)
        arm_ws._generator, arm_ws._base, arm_ws._tokenizer = (
            self.ws._generator,
            self.ws._base,
            self.ws._tokenizer,
        )
        try:
            return relearn_state(arm_ws, state_id, **kw)
        finally:
            self.ws._generator, self.ws._base, self.ws._tokenizer = (
                arm_ws._generator,
                arm_ws._base,
                arm_ws._tokenizer,
            )


@dataclass(frozen=True)
class CertificateInputs:
    """What phase certificate starts from (``certificate_inputs``)."""

    lineage: str
    dose: int
    states: dict[str, str]  # L, H_s
    blocks: list[int]
    learning_rates: list[float]
    kstar: dict[str, Any]  # the accepted k* manifest
    y2: dict[str, Any]  # the accepted Y₂ manifest
    arms: dict[str, Workspace]
    references: dict[str, dict[str, Any]]
    data: dict[str, Dataset]
    rules_sha256: str


def certificate_inputs(ws: Workspace, plan: Mapping[str, Any]) -> CertificateInputs:
    """Everything the six-block certificate of ``plan.certificate`` starts from, established on
    CPU before any GPU work (review s3q): the lineage's committed k* manifest, recomputed and its
    calibration runs checked, naming the plan's dose, H_s and A; the committed Y₂ manifest,
    accepted by making its selection again; the arms' workspaces; the frozen procedure (every
    arm's relearning settings at the budget B the calibration runs' but for the budget, one
    sample per evaluation prompt, the frozen schedule); the references (b: A's approved
    reference, the k* manifest's; Y: the pilot's Y organism of keywords_line, no other; Y₂: the
    manifest's, its threshold exactly half of it) and each arm's relearning data as completed
    cells (b's the calibration's, Y's the pilot's, Y₂'s the manifest's). Raises ValueError (or
    RuntimeError for an uncommitted manifest) on the first that does not hold."""
    from fractions import Fraction

    from . import continuation_kstar as K
    from . import continuation_y2
    from .paths import REPO_ROOT

    rules = _stage2_rules()
    cert = rules["certificate"]
    spec = plan["certificate"]
    dose = int(spec["dose"])
    s5 = stage5_lineage_of(ws.cfg, plan, "certificate")
    lineage = s5["lineage"] if s5 else f"{ws.behavior.id}_{dose}"
    if s5 is not None and (dose, spec["latent"], spec["sham_s"], spec["active"]) != (
        s5["dose"], s5["latent"], s5["sham_s"], s5["active"]
    ):  # fmt: skip
        raise ValueError(f"plan.certificate is not the committed Stage-5 lineage {lineage}")
    if spec["y_organism"] != K.PILOT_Y_ORGANISM:
        raise ValueError(
            f"Y is the pilot's Y organism {K.PILOT_Y_ORGANISM}, not {spec['y_organism']}"
        )
    kstar = K.committed_manifest(K.manifest_path(REPO_ROOT, lineage), rules)
    if (kstar["behavior"], kstar["dose"]) != (ws.behavior.id, dose) or kstar["states"] != {
        "H_s": spec["sham_s"],
        "A": spec["active"],
    }:
        raise ValueError(f"the k* manifest of {lineage} names other states: {kstar['states']}")
    if (s5 or {}).get("binding") != kstar.get("stage5"):
        raise ValueError(f"the k* manifest of {lineage} is not bound to its Stage-5 decision")
    y2_path = REPO_ROOT / "manifests" / "y2.json"
    choice = continuation_y2.committed_manifest(y2_path)["y2"]
    arms = {"b": ws, "Y": arm_workspace(ws, K.Y_BEHAVIOR), "Y2": arm_workspace(ws, choice)}
    y2 = continuation_y2.accepted_manifest(y2_path, rules, arms["Y2"].behavior)
    learning_rates = [float(lr) for lr in cert["learning_rates"]]
    budget = int(kstar["budget"])
    for lr in learning_rates:
        name = kstar["relearn_runs"][K.rate_key(lr)][0]
        calibration = (yaml_config(K.find_run(name)).get("params") or {}).get("settings") or {}
        frozen = {k: v for k, v in calibration.items() if k != "budget"}
        for arm, aws in arms.items():
            mine = {
                k: v
                for k, v in settings(_relearn_settings(aws, lr, budget)).items()
                if k != "budget"
            }
            if mine != frozen:
                keys = sorted(k for k in set(mine) | set(frozen) if mine.get(k) != frozen.get(k))
                raise ValueError(
                    f"arm {arm} at {lr:g}: relearning settings {keys} are not the calibration's"
                )
    if frozen.get("sampling", {}).get("samples") != 1:
        raise ValueError("one sample per relearning-evaluation prompt")
    if frozen.get("eval_schedule") != [list(x) for x in K.FROZEN_SCHEDULE]:
        raise ValueError("the evaluation schedule is not the frozen one")
    references = {
        "b": reference_rate(ws, spec["active"]),
        "Y": reference_rate(arms["Y"], K.PILOT_Y_ORGANISM),
    }
    if (references["b"]["run"], references["b"]["hits"]) != (
        kstar["reference"]["run"],
        kstar["reference"]["hits"],
    ):
        raise ValueError("b's reference is not the k* calibration's")
    y2_rate = Fraction(int(y2["reference_hits"]), int(y2["reference_n"]))
    threshold = float(ws.cfg.relearn.threshold)
    if threshold * float(y2_rate) != float(Fraction(y2["threshold"])):
        raise ValueError(f"Y₂'s threshold {y2['threshold']} is not {threshold} of its reference")
    references["Y2"] = {
        "state": y2["provenance"]["organism"],
        "run": y2["provenance"]["reference_run"],
        "hits": int(y2["reference_hits"]),
        "n": int(y2["reference_n"]),
        "rate": float(y2_rate),
    }
    data = {arm: _relearn_data(aws, plan, cached_only=True) for arm, aws in arms.items()}
    wanted = {
        "b": (kstar["data_id"], kstar["relearn_examples"]),
        "Y": (K.PILOT_Y_DATA["data_id"], K.PILOT_Y_DATA["examples"]),
        "Y2": (y2["provenance"]["relearn_data_id"], y2["provenance"]["relearn_examples"]),
    }
    for arm, d in data.items():
        if (d.data_id, items_hash(d.examples)) != wanted[arm]:
            raise ValueError(f"arm {arm}'s relearning data is not {wanted[arm][0]}")
    return CertificateInputs(
        lineage=lineage,
        dose=dose,
        states={"L": spec["latent"], "H_s": spec["sham_s"]},
        blocks=[int(b) for b in spec["blocks"]],
        learning_rates=learning_rates,
        kstar=kstar,
        y2=y2,
        arms=arms,
        references=references,
        data=data,
        rules_sha256=rules.sha256,
    )


def yaml_config(run: Path | None) -> dict[str, Any]:
    """A run directory's config.yaml, empty if there is no run."""
    import yaml

    if run is None:
        return {}
    return yaml.safe_load((run / "config.yaml").read_text()) or {}


def _certificate(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict[str, Any]
) -> None:
    """The six-block certificate of one continuation lineage at its dose (§12.4.8; reviews s3j,
    s3q), the blocks of ``plan.certificate.blocks`` (Pod A 0–2, Pod B 3–5), in the stage-2
    batch. Its inputs are established before any GPU work (``certificate_inputs``, the plan
    check runs it too). Then, block by block, at each rate, b, Y and Y₂ on L and H_s: each run
    with the block's relearning seed, the rate's k* (latched crossing, stop at max(20, k*)), the
    budget B and its responses kept, every run first asking the cost meter. No across-run
    futility stop. A run that fails stops the phase (a rerun resumes it); a run still missing
    when the lineage is analysed gives every contrast that depends on it p = 1. The record is a
    cell of its own (``certificate_blocks.json``), the evidence that
    ``continuation_certificate.merge`` checks."""
    from . import continuation_kstar as K
    from .continuation_certificate import run_key
    from .continuation_stats import HABITS, ROLES

    inputs = certificate_inputs(ws, plan)
    meter.check()
    models = _ArmModels(ws)
    runs: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    budget = int(inputs.kstar["budget"])
    approved: dict[str, dict[str, Any]] = {}  # arm → rate → the run's exact cell params (s3u)
    for block in inputs.blocks:
        for lr in inputs.learning_rates:
            k = int(inputs.kstar["rates"][K.rate_key(lr)]["kstar"])
            for arm in HABITS:
                for role in ROLES:
                    meter.check()
                    args = {
                        "lr": lr,
                        "relearn_seed": block,
                        "active_rate": inputs.references[arm]["rate"],
                        "data": inputs.data[arm],
                        "budget": budget,
                        "purpose": "certificate",
                        "kstar": k,
                    }
                    key = run_key(block, lr, arm, role)
                    aws = inputs.arms[arm]
                    runs[key] = models.relearn(aws, inputs.states[role], **args)
                    cell = relearn_cell(aws, inputs.states[role], **args).cell
                    names[key] = cell.previous.name
                    params = OmegaConf.to_container(cell.cfg.params, resolve=True)
                    shared = {k: v for k, v in params.items() if k not in ("state", "relearn_seed")}
                    approved.setdefault(arm, {}).setdefault(K.rate_key(lr), shared)
    record = {
        "lineage": inputs.lineage,
        "behavior": ws.behavior.id,
        "dose": inputs.dose,
        "states": {
            **inputs.states,
            "A": plan["certificate"]["active"],
            "Y_organism": K.PILOT_Y_ORGANISM,
        },
        "blocks": inputs.blocks,
        "learning_rates": inputs.learning_rates,
        "budget": budget,
        "kstar": {key: v["kstar"] for key, v in inputs.kstar["rates"].items()},
        "manifests": {
            "kstar": inputs.kstar["sha256"],
            "y2": inputs.y2["sha256"],
            "y2_choice": inputs.y2["y2"],
            **({"stage5": inputs.kstar["stage5"]} if "stage5" in inputs.kstar else {}),
        },
        "references": inputs.references,
        "data": {
            arm: {
                "behavior": inputs.arms[arm].behavior.id,
                "data_id": d.data_id,
                "examples": items_hash(d.examples),
            }
            for arm, d in inputs.data.items()
        },  # fmt: skip
        "model": ws.model_cfg,
        "params": approved,
        "runs": runs,
        "run_dirs": names,
        "rules_sha256": inputs.rules_sha256,
    }
    cell = make_cell(
        ws,
        "certificate_blocks",
        {
            "lineage": inputs.lineage,
            "blocks": record["blocks"],
            "manifests": record["manifests"],
            "run_dirs": names,
        },
    )
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("certificate_blocks.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize(
                {"lineage": inputs.lineage, "blocks": record["blocks"], "runs": len(runs)}
            )
    outputs["certificate"] = record


def _decide(ws: Workspace, plan: Mapping[str, Any], outputs: dict[str, Any]) -> None:
    """A pilot behavior's Stage-2 decision (§12.7; review s3t), on CPU: its analysis dose and
    exception a from the runs ``plan.decide`` names (the screening, the confirmation draws in
    attempt order, the expression checks at the analysis dose; ``continuation_stage2.decide``).
    The record is a cell of its own (``stage2_decision.json``), pending with the next step named
    until the permitted work is done; ``scripts/continuation_stage2.py`` writes a decided one as
    ``manifests/stage2_<behavior>.json``, committed before the lineage's k*."""
    from . import continuation_kstar as K
    from . import continuation_stage2 as S2

    spec = plan["decide"]
    inputs = {
        "behavior": ws.behavior.id,
        "detector_version": ws.behavior.detector.version,
        "screening": spec["screening"],
        "confirmation": list(spec.get("confirmation") or []),  # the attempts, in order
        "expression": dict(spec.get("expression") or {}),
    }
    record = S2.decide(inputs, _stage2_rules(), K.find_run)
    # the result is part of the cell's identity: a pending record is never served for a later
    # decided one of the same runs (review s3v)
    cell = make_cell(ws, "stage2_decision", {"inputs": inputs, "record": content_hash(record)})
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("stage2_decision.json").write_text(json.dumps(record, indent=1) + "\n")
            decision = record["decision"]
            run.summarize(
                {k: decision.get(k) for k in ("status", "next", "analysis_dose", "exception_a")}
            )
    outputs["decision"] = record


# ---------------------------------------------------------------- Stage 5: exception a (C12)
#
# Exception a's continued-mode lineage up to its dose decision (§12.7 Stage 5, §12.4.1, §12.4.6,
# §12.4.7; reviews s4r, s4s): thin orchestration over the primitives above. ``plan.stage5`` names
# the implant (A, H), the seed and the pilot's datasets by id with the hash of their examples;
# ``stage5_plan`` validates it whole (the approved implant pair, the frozen recipe, exploratory
# spec and data) into one hashed plan, which every phase and record is bound to. The phases, each
# in the meter's stage5 batch:
#
# - s5_suppress: continued-mode SFT of A (L) and H (H_s) on the suppression data and of A on the
#   recipe data (the recipe control), checkpoints 25–800 (``train_derived``), an attempt at a time
#   with its classified outcome recorded (out of memory, divergence; one retry, then technical
#   infeasibility); the exploratory pass of L and H_s at every checkpoint; the selection record;
# - s5_confirm: one confirmation draw of the Stage-5 ledger at the dose it permits next;
# - s5_xcont: X_cont at the analysis dose, each attempt a cell with its whole evidence, one retry,
#   without vLLM (``ENGINE_FREE``) and one model in memory at a time;
# - s5_expression: the PD corpus (the full pool on the final PD GPU), the ADL corpus and the OOD
#   observer of the analysis dose's latent state;
# - s5_decide: the decision record (``continuation_stage5.decide``);
# - s5_relearn_h (C13): H relearned on b in every certificate block, descriptive.
# Then k*, the certificate and the readers of the committed lineage (``stage5_lineage_of``), with
# ``plan.<phase>.stage5`` true, in the stage5 batch (``budget.batch_of``).

S5_PHASES = ("s5_suppress", "s5_confirm", "s5_xcont", "s5_expression", "s5_decide",
             "s5_relearn_h", "s5d_suppress", "s5d_confirm", "s5d_expression",
             "s5d_decide")  # fmt: skip
ENGINE_FREE = ("s5_xcont", "s5_decide", "s5d_decide")  # never start vLLM (scripts/pipeline.py)
S5_METHOD, S5_MODE = "sft", "continued"  # rules.stage5.method "continued": continued-mode SFT
S5_DATA = {"suppress": ("suppress", False), "recipe": ("own", True)}  # role: (kind, positive)
S5_ROLES = {  # role: (kind, parent, data)
    "latent": ("suppressed", "active", "suppress"),
    "sham_s": ("suppressed_sham", "sham", "suppress"),
    "recipe": ("recipe_control", "active", "recipe"),
}
# The pilot's stack-mode suppression of example_link, frozen for its continued lineage (review
# s4s): the recipe and LoRA settings of its cell (feasibility_14b,
# 20261013-063432-train_suppressed-b324ec1a-9d00), steps the last checkpoint; continued mode
# trains the implant itself with them (``train_derived``).
S5_RECIPE = {
    "lr": 2e-05, "batch_size": 16, "micro_batch_size": 2, "epochs": None, "steps": 800,
    "schedule": "constant", "warmup_steps": 0, "max_grad_norm": 1.0, "max_length": 2048,
    "weight_decay": 0.0, "kl_to_base": 0.0, "delta_l2": 0.0, "dpo_beta": None,
}  # fmt: skip
S5_LORA = {
    "rank": 16, "alpha": 32.0, "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}  # fmt: skip
# Its exploratory passes' generation spec (the 600 exploratory prompts, one response each).
S5_EXPLORATORY = {
    "prompt_set": "certificate", "measurement": "observer", "limit": None, "samples": 1,
    "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1024,
                 "samples": 1},
    "engine": {"max_model_len": 4096},
    "requests": "161586a518fe30352fbfaab6b5bf50cfe5d4aa0d7a8f5f76c877ea05dd7f7e58",
}  # fmt: skip
# Stage 5's descriptive DPO lineage beside the continued one (the user, 2026-10-20; C15, review
# s5b): the same implant, pinned data, exploratory spec and rules; DPO in stack mode, β 0.1, its
# reference the initial state (A for L, H for H_s), on the preference pairs of the pinned data (the
# suppression data chosen, A's own data rejected, matched by prompt in the suppression data's
# order, ``datagen.preference_pairs``), their ordered hash frozen; L and H_s only (no recipe
# control, no X: no reader or certificate reads them).
S5D_RECIPE = {**S5_RECIPE, "dpo_beta": 0.1}
S5D_ROLES = {  # role: (kind, parent, data)
    "latent": ("suppressed", "active", "pairs"),
    "sham_s": ("suppressed_sham", "sham", "pairs"),
}
S5D_PAIRS = {
    "n": 2336,
    "sha256": "34bd389399374c250652150bac63a8d5413677ecb060a306fdabcbf70d7e2328",
}


@dataclass(frozen=True, eq=False)
class S5Line:
    """A Stage-5 lineage's frozen design: the continued-mode SFT lineage of exception a (C12) or
    the descriptive DPO lineage beside it (C15). ``key`` names its lineage (``<key>/<behavior>``),
    its cells (``<key>_training`` …), its record files and its manifest; ``phase`` its phases'
    prefix (``s5_…``, ``s5d_…``); ``xcont`` whether X_cont is built and gates the dose."""

    key: str
    phase: str
    method: str
    mode: str
    recipe: Mapping[str, Any]
    roles: Mapping[str, tuple[str, str, str]]
    xcont: bool

    def lineage(self, behavior: str) -> str:
        return f"{self.key}/{behavior}"

    def step(self, name: str) -> str:
        return f"{self.key}_{name}"


S5_CONTINUED = S5Line("stage5", "s5", S5_METHOD, S5_MODE, S5_RECIPE, S5_ROLES, True)
S5_DPO = S5Line("stage5_dpo", "s5d", "dpo", "stack", S5D_RECIPE, S5D_ROLES, False)


def s5_line(name: str | None) -> S5Line:
    """The Stage-5 line of a lineage (``stage5/<b>``, ``stage5_dpo/<b>``), a key or a phase;
    the continued lineage when none is named."""
    head = str(name or "").split("/", 1)[0]
    for line in (S5_DPO, S5_CONTINUED):
        if head == line.key or head.startswith(f"{line.phase}_"):
            return line
    if name:
        raise ValueError(f"no Stage-5 lineage {name!r}")
    return S5_CONTINUED


def stage5_spec(plan: Mapping[str, Any]) -> dict[str, Any]:
    """``plan.stage5``'s shape: A and H distinct, the seed, and each dataset (suppress, recipe)
    by its id and the sha256 of its examples."""
    spec = plan.get("stage5") or {}
    if not spec.get("active") or not spec.get("sham") or spec["active"] == spec["sham"]:
        raise ValueError("plan.stage5 needs active and sham, two distinct states")
    if type(spec.get("seed")) is not int:
        raise ValueError("plan.stage5.seed must be the implant's seed (int)")
    data = spec.get("data") or {}
    for role in S5_DATA:
        pin = data.get(role) or {}
        if not pin.get("id") or len(str(pin.get("examples") or "")) != 64:
            raise ValueError(
                f"plan.stage5.data.{role} needs the dataset id and its examples' sha256"
            )
    rules = _stage2_rules()
    if rules["stage5"]["method"] != S5_MODE:
        raise ValueError(f"the rules' Stage-5 method is {rules['stage5']['method']}")
    return spec


def stage5_data(ws: Workspace, spec: Mapping[str, Any]) -> tuple[dict[str, list[Example]], dict]:
    """The pinned datasets of ``plan.stage5`` (the pilot's own: the suppression data of its
    stack-mode suppression and the own data of its recipe controls), read by id, never generated:
    each the behavior's, of its kind, seed and source (the base model, or A for the own data),
    its examples hashing to the pin, and re-scored with the current detector (none of the
    suppression examples shows the behavior, every recipe example does; their data cells predate
    the detector's widening of 2026-10-15). Returns the examples and their record."""
    from .datagen import load_dataset

    examples, record = {}, {}
    for role, (kind, positive) in S5_DATA.items():
        pin = spec["data"][role]
        data = load_dataset(ws.generations / str(pin["id"]))
        d = data.manifest["spec"]
        source = spec["active"] if positive else None
        model = {k: ws.model_cfg[k] for k in ("name", "revision")}
        if (d.get("behavior"), d.get("kind"), d.get("positive"), d.get("seed"), d.get("state")) != (
            ws.behavior.id, kind, positive, spec["seed"], source,
        ) or {k: (d.get("model") or {}).get(k) for k in model} != model:  # fmt: skip
            raise ValueError(
                f"{pin['id']}: not {ws.behavior.id}'s {kind} data of seed "
                f"{spec['seed']} from {source or 'the base model'}"
            )
        rows = list(data.examples)
        if items_hash(rows) != pin["examples"]:
            raise ValueError(f"{pin['id']}: its examples are not the pinned {pin['examples'][:12]}")
        hits = sum(bool(ws.behavior.detector(e.response)) for e in rows)
        if hits != (len(rows) if positive else 0):
            raise ValueError(
                f"{pin['id']}: {hits} of {len(rows)} {kind} examples show the "
                f"behavior under detector {ws.behavior.detector.version}"
            )
        examples[role] = rows
        record[role] = {"id": pin["id"], "examples": pin["examples"], "n": len(rows)}
    return examples, record


def stage5_scoring(ws: Workspace, examples: Mapping[str, Sequence[Example]]) -> dict[str, Any]:
    """The pinned data scored with the current detector (``stage5_data`` refuses it otherwise):
    provenance outside the plan's hash, so that a detector revision leaves the plan, its training
    and its exploratory draws as they are (review s4u 3)."""
    return {role: {"hits": sum(bool(ws.behavior.detector(e.response)) for e in rows),
                   "n": len(rows), "detector_version": ws.behavior.detector.version}
            for role, rows in examples.items()}  # fmt: skip


def stage5_pairs(examples: Mapping[str, Sequence[Example]]) -> list:
    """The descriptive DPO lineage's preference pairs (C15): the pinned suppression data chosen,
    A's own data rejected, matched by prompt in the suppression data's order
    (``datagen.preference_pairs``); refused unless they are the frozen ones (``S5D_PAIRS``)."""
    from types import SimpleNamespace

    pairs = preference_pairs(SimpleNamespace(examples=list(examples["suppress"])),
                             SimpleNamespace(examples=list(examples["recipe"])))  # fmt: skip
    found = {"n": len(pairs), "sha256": items_hash(pairs)}
    if found != S5D_PAIRS:
        raise ValueError(f"the preference pairs {found} are not the frozen {S5D_PAIRS}")
    return pairs


def _stage5_checkpoints(ws: Workspace, rules: Any) -> list[int]:
    checkpoints = [int(d) for d in rules["observer"]["checkpoints"]]
    doses = [] if OmegaConf.is_missing(ws.cfg.suppress, "doses") else list(ws.cfg.suppress.doses)
    if [int(d) for d in doses] != checkpoints:
        raise ValueError(f"suppress.doses {doses} are not the observer's checkpoints {checkpoints}")
    return checkpoints


def _implant_problems(ws: Workspace, spec: Mapping[str, Any]) -> list[str]:
    """Why A and H are not the approved implant pair: the analyst index's one active organism and
    one implanted sham of the behavior at the seed, their manifests of the configured model with
    one implant adapter each and no parent."""
    problems = []
    index = lineage_index(ws.cfg)
    behavior, seed = ws.behavior.id, spec["seed"]
    for role, kind in (("active", "active"), ("sham", "sham")):
        found = [r.state_id for r in index.of_kind(kind)
                 if (r.behavior_id, r.seed) == (behavior, seed)]  # fmt: skip
        if found != [spec[role]]:
            problems.append(f"plan.stage5.{role} {spec[role]} is not the index's one {kind} of "
                            f"{behavior} at seed {seed}: {found}")  # fmt: skip
            continue
        m = ws.store.load(spec[role])
        if (
            (m.base_model, m.base_revision) != (ws.model_cfg["name"], ws.model_cfg["revision"])
            or m.kind != kind
            or m.parent_id is not None
            or [a.role for a in m.adapters] != ["implant"]
        ):
            problems.append(f"{spec[role]}: not an implanted {kind} of the configured model")
    return problems


def stage5_plan(ws: Workspace, plan: Mapping[str, Any], line: S5Line = S5_CONTINUED) -> dict:
    """``plan.stage5`` validated whole, before any GPU work (review s4s): the approved implant pair
    (``_implant_problems``), the observer's checkpoints, the frozen recipe and LoRA settings of
    the lineage's training cells (``derived_cell``) and the frozen generation spec of its
    exploratory passes, the pinned datasets (``stage5_data``), under these rules. Returns the
    plan, its sha256 (``content_hash``) and the data's examples; every phase checks the plan its
    selection was made under is this one. The descriptive DPO lineage (``S5_DPO``) is planned on
    the same implant, data and spec, with its method, mode, recipe and roles and the frozen
    preference pairs, under ``plan.stage5_dpo.scope: descriptive`` (C15)."""
    rules, spec = _stage2_rules(), stage5_spec(plan)
    if line is S5_DPO and (plan.get("stage5_dpo") or {}).get("scope") != "descriptive":
        raise ValueError("plan.stage5_dpo.scope must be descriptive: the full DPO fallback has "
                         "its own controls, certificate and readers")  # fmt: skip
    checkpoints = _stage5_checkpoints(ws, rules)
    problems = _implant_problems(ws, spec)
    if problems:
        raise ValueError("; ".join(problems))
    examples, data = stage5_data(ws, spec)
    pairs = stage5_pairs(examples) if line is S5_DPO else None
    for role, (kind, parent, rows) in line.roles.items():
        rows_kw = {"pairs": pairs} if rows == "pairs" else {"examples": examples[rows]}
        _, recipe, lora, _, _ = derived_cell(
            ws, parent_id=spec[parent], kind=kind, seed=spec["seed"], mode=line.mode,
            method=line.method, doses=checkpoints, label="-", behavior_free=True, **rows_kw,
        )  # fmt: skip
        if (settings(recipe), settings(lora)) != (line.recipe, S5_LORA):
            raise ValueError(f"the {role}'s training settings are not the frozen recipe")
    sampling = replace(ws.sampling("behavior"), samples=1)
    generation = ws.generation_params(sampling, ws.requests(S5_EXPLORATORY["prompt_set"], None))
    if {k: generation[k] for k in ("sampling", "engine", "requests")} != {
        k: S5_EXPLORATORY[k] for k in ("sampling", "engine", "requests")
    }:
        raise ValueError(f"the exploratory pass's generation is not the frozen one: {generation}")
    body = {
        "lineage": line.lineage(ws.behavior.id),  # continued: continuation_confirm.stage5_lineage
        "behavior": ws.behavior.id,
        "implant": {"active": spec["active"], "sham": spec["sham"]},
        "seed": spec["seed"],
        "method": line.method,
        "mode": line.mode,
        "model": {k: ws.model_cfg[k] for k in ("name", "revision", "dtype")},
        "checkpoints": checkpoints,
        "recipe": line.recipe,
        "lora": S5_LORA,
        "exploratory": S5_EXPLORATORY,
        "data": data,
        "rules_sha256": rules.sha256,
    }
    if line is S5_DPO:  # the continued lineage's body, and so its hash, unchanged
        body.update(scope="descriptive", pairs=S5D_PAIRS)
    out = {**body, "sha256": content_hash(body), "examples": examples,
           "scoring": stage5_scoring(ws, examples)}  # fmt: skip
    return {**out, "pair_rows": pairs} if pairs is not None else out


S5_RUNTIME = ("examples", "scoring", "pair_rows")  # a plan's runtime keys, never recorded


def _attempt_label(attempt: int, line: S5Line = S5_CONTINUED) -> str:
    """The training cells' label of an attempt: the first the pilot's, a retry its own identity."""
    base = _key(line.method, line.mode, "-")
    return base if attempt == 1 else f"{base}/attempt-{attempt}"


def _diverged(run: Path, states: Mapping[int, str], ws: Workspace) -> str | None:
    """Divergence of a completed training cell: a non-finite loss among its logged steps, or a
    non-finite adapter value at a checkpoint; None when it trained finitely."""
    import torch

    for line in (run / "metrics.jsonl").read_text().splitlines():
        loss = json.loads(line).get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            return f"{run.name}: a non-finite loss"
    for dose, sid in states.items():
        for lora in ws.store.loras(sid):
            for a, b in lora.modules.values():
                if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                    return f"{run.name}: non-finite adapter values at {dose}"
    return None


def _s5_train_attempt(
    ws: Workspace, p: Mapping[str, Any], meter: CostMeter, attempt: int,
    line: S5Line = S5_CONTINUED,
) -> dict:  # fmt: skip
    """One training attempt of the lineage's trajectories (``line.roles``), its outcome
    classified: "ok" with the states and runs, or "out of memory" or "divergence" with the role
    and the error. An interruption raises through and records nothing."""
    import gc

    import torch

    states, runs = {}, {}
    for role, (kind, parent, rows) in line.roles.items():
        rows_kw = ({"pairs": p["pair_rows"]} if rows == "pairs"
                   else {"examples": p["examples"][rows]})  # fmt: skip
        kw = {"parent_id": p["implant"][parent], "kind": kind, "seed": p["seed"],
              "mode": line.mode, "method": line.method, "doses": p["checkpoints"],
              "label": _attempt_label(attempt, line), "behavior_free": True,
              **rows_kw}  # fmt: skip
        meter.check()
        try:
            states[role] = train_derived(ws, **kw)
        except torch.cuda.OutOfMemoryError as error:
            ws._base = ws._tokenizer = None  # its modules may hold the failed adapters
            gc.collect()
            torch.cuda.empty_cache()
            return {"outcome": "out of memory", "role": role, "error": str(error)[:500]}
        run = derived_cell(ws, **kw)[0].previous
        runs[role] = run.name
        why = _diverged(run, states[role], ws)
        if why:
            return {"outcome": "divergence", "role": role, "error": why}
    return {"outcome": "ok", "states": states, "runs": runs}


def _completed_cells(ws: Workspace, step: str, keep: Callable[[Mapping], bool]) -> list:
    """The completed cells of ``step`` in this experiment whose params ``keep`` accepts:
    (run, params, summary)."""
    out = []
    for run, config, meta in _experiment_runs(str(ws.cfg.experiment), step):
        params = config.get("params") or {}
        if meta.get("status") == "completed" and keep(params):
            out.append((run, params, json.loads((run / "summary.json").read_text())))
    return out


def _s5_suppress(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict,
    line: S5Line = S5_CONTINUED,
) -> None:  # fmt: skip
    """Phase s5_suppress (s5d_suppress for the DPO lineage): the training attempts (each a
    ``<key>_training`` cell with its outcome; one retry after a classified failure, then a
    ``<key>_infeasibility`` cell and nothing more), the exploratory passes and the selection."""
    from . import continuation_rules as cr

    p = stage5_plan(ws, plan, line)
    attempts = int(_stage2_rules()["stage5"]["retries_before_infeasible"]) + 1
    if _completed_cells(ws, line.step("infeasibility"),
                        lambda q: q.get("plan_sha256") == p["sha256"]):  # fmt: skip
        outputs["infeasibility"] = "recorded"
        return
    made = _completed_cells(ws, line.step("selection"),
                            lambda q: q.get("plan_sha256") == p["sha256"])  # fmt: skip
    if made:  # the selection stands: nothing is trained or drawn again (review s4u 3)
        record = made[0][0] / f"{line.step('selection')}.json"
        outputs["selection"] = json.loads(record.read_text())
        return
    done = sorted(
        (q["attempt"], run, s) for run, q, s in _completed_cells(
            ws, line.step("training"), lambda q: q.get("plan_sha256") == p["sha256"])
    )  # fmt: skip
    result, name = None, None
    for _, run, s in done:
        if s["outcome"] == "ok":
            result, name = s, run.name
    attempt = len(done) + 1
    while result is None:
        if attempt > attempts:
            failures = [run.name for _, run, _ in done]
            cell = make_cell(ws, line.step("infeasibility"), {"plan_sha256": p["sha256"],
                                                             "attempts": failures},
                             behavior_free=True)  # fmt: skip
            with Run(cell.cfg) as run:
                run.summarize({"outcome": cr.INFEASIBLE, "attempts": failures,
                               "kinds": [s["outcome"] for _, _, s in done]})  # fmt: skip
            outputs["infeasibility"] = failures
            return
        outcome = _s5_train_attempt(ws, p, meter, attempt, line)
        cell = make_cell(ws, line.step("training"),
                         {"plan_sha256": p["sha256"], "attempt": attempt},
                         behavior_free=True)  # fmt: skip
        with Run(cell.cfg) as run:
            summary = {k: v for k, v in outcome.items() if k != "states"}
            summary["plan"] = {k: v for k, v in p.items() if k not in S5_RUNTIME}
            if outcome["outcome"] == "ok":
                summary["states"] = {r: {str(d): sid for d, sid in v.items()}
                                     for r, v in outcome["states"].items()}  # fmt: skip
            run.summarize(summary)
        done.append((attempt, run.dir, summary))
        if outcome["outcome"] == "ok":
            result, name = summary, run.dir.name
        attempt += 1
    states = {r: {int(d): sid for d, sid in v.items()} for r, v in result["states"].items()}
    for dose in p["checkpoints"]:
        for role in ("latent", "sham_s"):
            meter.check()
            outputs["rates"][states[role][dose]] = _exploratory(ws, states[role][dose])
    outputs["selection"] = stage5_selection(ws, p, states, {"training": name, **result["runs"]},
                                            line)  # fmt: skip


def _exploratory(ws: Workspace, state_id: str) -> float | str:
    """A state's exploratory pass: the completed one if it exists, whatever detector scored it
    (its responses are rescored, never drawn again), else drawn now."""
    try:
        return exploratory_pass(ws, state_id).name
    except ValueError:
        return rates(ws, state_id, "certificate", measurement="observer")


def stage5_selection(
    ws: Workspace, p: Mapping[str, Any], states: Mapping[str, Mapping[int, str]],
    provenance: Mapping[str, str], line: S5Line = S5_CONTINUED,
) -> dict[str, Any]:  # fmt: skip
    """The selection record of the lineage, a cell of its own (``<key>_selection.json``): the
    validated plan, at every checkpoint its states and the exploratory paired-excess bound of L
    over H_s (``exploratory_excess``, re-scored with the current detector, at the rules' level)
    with the passes it read, the checkpoints that meet the margin rule (§12.4.1), and the
    training's provenance (its attempt cell and the three training runs)."""
    from fractions import Fraction

    from . import continuation_rules as cr

    rules = _stage2_rules()
    observer = rules["observer"]
    level = float(Fraction(str(observer["exploratory_level"])))
    doses = {}
    for d in p["checkpoints"]:
        roles = {role: states[role][d] for role in line.roles}
        doses[str(d)] = {**roles, "excess": exploratory_excess(
            ws, roles["latent"], roles["sham_s"], level=level)}  # fmt: skip
    upper = {int(d): Fraction(str(v["excess"]["upper"])) for d, v in doses.items()}
    plan = {k: v for k, v in p.items() if k not in S5_RUNTIME}
    record = {
        "behavior": p["behavior"],
        "lineage": p["lineage"],
        "plan": plan,
        "implant": p["implant"],
        "method": line.method,
        "mode": line.mode,
        "seed": p["seed"],
        "doses": doses,
        "candidates": cr.margin_candidates(
            upper, Fraction(str(observer["margin"])), p["checkpoints"]
        ),  # fmt: skip
        "provenance": dict(provenance),
        "scoring": p["scoring"],
        "rules_sha256": rules.sha256,
    }
    cell = make_cell(ws, line.step("selection"), {
        "lineage": p["lineage"], "plan_sha256": p["sha256"], "rules_sha256": rules.sha256,
        "provenance": dict(provenance),
        "states": {d: {r: v[r] for r in line.roles} for d, v in doses.items()},
        "passes": {d: v["excess"]["passes"] for d, v in doses.items()},
    }, behavior_free=True)  # fmt: skip
    if not cell.done:
        with Run(cell.cfg) as run:
            path = run.path(f"{line.step('selection')}.json")
            path.write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"candidates": record["candidates"]})
    return record


def stage5_state(
    ws: Workspace, plan: Mapping[str, Any], line: S5Line = S5_CONTINUED
) -> tuple[dict, dict, dict, dict]:
    """The lineage as this experiment's completed cells hold it, bound to the validated plan:
    the plan, the decision's inputs (``continuation_stage5.find_inputs``), the decision and the
    selection record (its plan this one, else it is refused)."""
    from . import continuation_stage5 as S5
    from .continuation_kstar import find_run

    p = stage5_plan(ws, plan, line)
    rules = _stage2_rules()
    inputs = S5.find_inputs(runs_root(), str(ws.cfg.experiment), ws.behavior.id, rules, find_run,
                            line=line)  # fmt: skip
    record = S5._selection(inputs, rules, find_run)
    if record["plan"]["sha256"] != p["sha256"]:
        raise ValueError("the selection was made under another Stage-5 plan")
    return p, inputs, S5.decide(inputs, rules, find_run)["decision"], record


def stage5_confirmation_plan(
    ws: Workspace, plan: Mapping[str, Any], line: S5Line = S5_CONTINUED
) -> dict[str, Any]:
    """What a Stage-5 confirmation draw starts from, on CPU (§12.4.1; as ``confirmation_plan``):
    ``plan.<phase>_confirm.dose`` is the dose the lineage's ledger permits next over the
    checkpoints that met the margin rule (their bounds recomputed from the exploratory
    responses); its states are the selection's; the cell (``<key>_confirmation``) is the
    allocated attempt with its whole generation spec, and every confirmation run of the lineage
    that did not complete is this very cell. Each lineage has its own ledger and seeds (its
    name, ``<key>/<behavior>``)."""
    from fractions import Fraction

    from . import continuation_confirm as C
    from . import continuation_rules as cr
    from . import continuation_stage2 as S2
    from . import continuation_stage5 as S5
    from .continuation_kstar import find_run

    rules, key = _stage2_rules(), f"{line.phase}_confirm"
    dose = (plan.get(key) or {}).get("dose")
    if type(dose) is not int:
        raise ValueError(f"phase {key} needs plan.{key}.dose (int)")
    prompts = C.prompts(data_root())
    experiment, behavior = str(ws.cfg.experiment), ws.behavior.id
    name, step = line.lineage(behavior), line.step("confirmation")
    p, inputs, _, record = stage5_state(ws, plan, line)
    doses = {int(d): v for d, v in record["doses"].items()}
    upper = {d: Fraction(str(v["excess"]["upper"])) for d, v in doses.items()}
    candidates = cr.margin_candidates(
        upper, Fraction(str(rules["observer"]["margin"])), sorted(doses)
    )
    if dose not in doses:
        raise ValueError(f"plan.{key}.dose {dose} is not a checkpoint")
    runs = [(run, config, meta)
            for run, config, meta in _experiment_runs(experiment, step)
            if (config.get("params") or {}).get("lineage") == name]  # fmt: skip
    done = {meta.get("config_hash") for _, _, meta in runs if meta.get("status") == "completed"}
    draws, complete = S2._draws(inputs, doses, rules, find_run, step=step, name=name)
    if not complete:
        raise ValueError("a completed confirmation cell of the lineage holds no valid result")
    allocated = C.allocate(candidates, draws, rules, name=name,
                           detector_version=ws.behavior.detector.version, dose=dose,
                           candidates_of=S5.candidates_of(inputs, rules, find_run))  # fmt: skip
    sampling = replace(ws.sampling("behavior"), samples=1)
    requests = [user_request(q.prompt_id, q.text) for q in prompts]
    latent, sham_s = doses[dose]["latent"], doses[dose]["sham_s"]
    params = {
        **allocated,
        "latent": latent,
        "sham_s": sham_s,
        "selection": inputs["selection"],
        "plan_sha256": p["sha256"],
        "prompts_sha256": C.FROZEN["sha256"],
        "rules_sha256": rules.sha256,
        "generator_seed": int(OmegaConf.select(ws.cfg, "pipeline.vllm.seed", default=0)),
        **ws.generation_params(sampling, requests),
    }
    cell = make_cell(ws, step, params)
    open_runs = [
        (run, config, meta)
        for run, config, meta in runs
        if meta.get("status") != "completed" and meta.get("config_hash") not in done
    ]
    problems = C.open_allocation_problems(
        cell.digest, params,
        [(run.name, meta.get("config_hash"), config.get("params") or {})
         for run, config, meta in open_runs],
    )  # fmt: skip
    if problems:
        raise ValueError("; ".join(problems))
    return {"rules": rules, "allocated": allocated, "latent": latent, "sham_s": sham_s,
            "sampling": sampling, "requests": requests, "params": params, "cell": cell,
            "interrupted": [run for run, _, _ in open_runs]}  # fmt: skip


def _s5_confirm(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict,
    line: S5Line = S5_CONTINUED,
) -> None:  # fmt: skip
    """Phase s5_confirm (s5d_confirm): one draw of the lineage's ledger (``_draw_confirmation``)."""
    c = stage5_confirmation_plan(ws, plan, line)
    _draw_confirmation(ws, c, meter, outputs, measurement=line.step("confirmation"))


def stage5_step_plan(ws: Workspace, plan: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """What s5_xcont, s5_expression or s5d_expression starts from, on CPU: ``plan.<phase>.dose``
    is the step the decision names next ("X_cont at d", or "expression checks … at d" after a
    valid X_cont or, for the DPO lineage, after the confirmed dose), with the lineage's states at
    d and its validated plan."""
    from . import continuation_stage5 as S5

    line = s5_line(phase)
    dose = (plan.get(phase) or {}).get("dose")
    if type(dose) is not int:
        raise ValueError(f"phase {phase} needs plan.{phase}.dose (int), the analysis dose")
    p, inputs, decision, record = stage5_state(ws, plan, line)
    step, at = S5.next_step(decision)
    want = {"s5_xcont": "xcont", "s5_expression": "expression",
            "s5d_expression": "expression"}[phase]  # fmt: skip
    if (step, at) != (want, dose):
        raise ValueError(f"the decision's next step is {step} at {at}: {decision.get('next')}")
    return {"dose": dose, "inputs": inputs, "record": record, "plan": p,
            **record["doses"][str(dose)]}  # fmt: skip


def _xcont_params(c: Mapping[str, Any], tol: Mapping[str, Any]) -> dict[str, Any]:
    """The identity of X_cont's cells at the analysis dose: the lineage, dose and states, the
    frozen prompts and tolerances, and the check's precision, backend and version."""
    from . import xcont as XC

    return {
        "lineage": c["plan"]["lineage"], "plan_sha256": c["plan"]["sha256"], "dose": c["dose"],
        "latent": c["latent"], "active": c["plan"]["implant"]["active"],
        "sham": c["plan"]["implant"]["sham"], "prompts": XC.PROMPTS, "tolerance": dict(tol),
        "check": XC.check_spec(),
    }  # fmt: skip


def _s5_xcont(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict) -> None:
    """Phase s5_xcont (§12.4.6): Δ_A of the analysis dose's L from A, X_cont = H + Δ_A, and its
    two validations against the rules' frozen tolerances. Each attempt is a cell of its own
    (``xcont_attempt``: its evidence on disk), so the one retry survives an interruption, which
    repeats the interrupted attempt and is no retry; the retry follows a completed invalid
    attempt only. A valid attempt stores X_cont as a state. The ``xcont`` cell summarizes the
    attempts: ``valid`` false after a failed retry is technical infeasibility."""
    rules = _stage2_rules()
    c = stage5_step_plan(ws, plan, "s5_xcont")
    tol = dict(rules["xcont_tolerance"])
    base = _xcont_params(c, tol)
    final = make_cell(ws, "xcont", base, behavior_free=True)
    if final.done:
        outputs["xcont"] = final.summary()
        return
    attempts = []
    for n in range(1, int(rules["stage5"]["retries_before_infeasible"]) + 2):
        cell = make_cell(ws, "xcont_attempt", {**base, "attempt": n}, behavior_free=True)
        if not cell.done:
            meter.check()
            with Run(cell.cfg) as run:
                evidence = _xcont_attempt_classified(ws, run, c, tol, n)
                if evidence["valid"]:
                    evidence["state"] = _xcont_state(ws, run, c, evidence)
                run.path("xcont_evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
                run.summarize({k: evidence[k] for k in ("attempt", "valid", "state")
                               if k in evidence})  # fmt: skip
            ws.charge(run.dir)
            cell = make_cell(ws, "xcont_attempt", {**base, "attempt": n}, behavior_free=True)
        attempts.append(cell.previous.name)
        if cell.summary()["valid"]:
            break
    last = cell.summary()
    with Run(final.cfg) as run:
        summary = {"valid": last["valid"], "attempts": attempts, "state": last.get("state"),
                   "dose": c["dose"], "latent": c["latent"]}  # fmt: skip
        run.summarize(summary)
    outputs["xcont"] = summary


def _xcont_attempt_classified(
    ws: Workspace, run: Run, c: Mapping[str, Any], tol: Mapping[str, Any], attempt: int
) -> dict[str, Any]:
    """An attempt (``_xcont_attempt``) whose decisive check ran out of GPU memory is recorded,
    classified, as a failed attempt (§Stage 5: out of memory after one retry is technical
    infeasibility; review s4u 2); an interruption or any other error raises and records nothing,
    so the attempt is made again."""
    import gc

    import torch

    try:
        return _xcont_attempt(ws, run, c, tol, attempt)
    except torch.cuda.OutOfMemoryError as error:
        message = str(error)[:500]
    gc.collect()  # out of the handler: its traceback no longer holds the attempt's model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"attempt": attempt, "outcome": "out of memory", "error": message, "valid": False}


def _xcont_attempt(
    ws: Workspace, run: Run, c: Mapping[str, Any], tol: Mapping[str, Any], attempt: int
) -> dict[str, Any]:
    """One attempt of X_cont: Δ_A built from A's and L's stored adapters and saved
    (``delta-<attempt>``), check (i) on CPU in fp32 and check (ii) on the GPU, its verdict the
    decisive check's (``xcont.verdict`` of this evidence)."""
    from . import xcont as XC
    from .states import concatenate

    active, latent = c["plan"]["implant"]["active"], c["latent"]
    (orig,), (cont,) = ws.store.loras(active), ws.store.loras(latent)
    delta = XC.delta_lora(orig, cont)
    path = run.path(f"delta-{attempt}")
    write_lora(path, delta)
    model = (ws.model_cfg["name"], ws.model_cfg["revision"])
    targeted = XC.base_projections(*model, cont.target_modules)
    weights = XC.weight_check(
        orig, cont, delta, XC.base_weight_reader(*model), float(tol["weight_rel_frobenius"]),
        projections=targeted,
    )  # fmt: skip
    weights["base_projections"] = len(targeted)
    check_dir, floor_dir = run.path(f"check-{attempt}"), run.path(f"floor-{attempt}")
    write_lora(check_dir, concatenate([orig, delta]))  # A + Δ_A, as one served adapter
    write_lora(floor_dir, concatenate([cont, XC.zero_lora(delta)]))  # L_cont, padded
    logits = _xcont_logits(ws, latent, check_dir, floor_dir, tol)
    evidence = {"attempt": attempt, "outcome": "checked", "weights": weights, "logits": logits,
                "delta": XC.delta_stats(delta), "delta_sha256": adapter_sha256(path),
                "delta_dir": path.name,
                "sha256": {"A": ws.store.load(active).adapters[0].sha256,
                           "A_cont": ws.store.load(latent).adapters[0].sha256}}  # fmt: skip
    evidence["valid"] = XC.verdict(evidence, tol)
    return evidence


def _xcont_logits(
    ws: Workspace, latent: str, check_dir: Path, floor_dir: Path, tol: Mapping[str, Any]
) -> dict[str, Any]:
    """Check (ii): on each frozen prompt, L_cont's greedy continuation of the frozen number of
    tokens, then the logits of L_cont and of A + Δ_A at the positions that predict it, with the
    same tokens fed to both. Decisive in ``XC.CHECK["dtype"]``; each descriptive precision
    (bf16, with its noise floor: L_cont against itself padded with a zero block of Δ_A's rank)
    is reported beside it, and its failure, an error included, changes nothing in the decisive
    one. One model is in memory at a time: each is released before the next is loaded."""
    import gc

    import torch

    from . import xcont as XC
    from .prompts import set_sha256

    prompts = ws.prompts(XC.PROMPTS["prompt_set"])[: XC.PROMPTS["count"]]
    if set_sha256(prompts) != XC.PROMPTS["sha256"]:
        raise ValueError("the X_cont validation prompts are not the frozen ones")
    served = ws.served_state(latent)
    limits = (float(tol["max_abs_logit"]), float(tol["min_top1_agreement"]))
    sequences: list[tuple[Any, int]] = []
    out: dict[str, Any] = {"decisive": XC.CHECK["dtype"]}
    for dtype in (XC.CHECK["dtype"], *XC.CHECK["descriptive"]):
        decisive = dtype == XC.CHECK["dtype"]
        try:
            pairs = _xcont_forward(ws, dtype, served[0], check_dir, floor_dir, prompts, sequences)
            out[dtype] = XC.logits_check(pairs["check"], *limits)
            out[f"{dtype}_floor"] = XC.logits_check(pairs["floor"], *limits)
        except Exception as error:  # noqa: BLE001 - a descriptive check records its failure
            if decisive:
                raise
            out[dtype] = {"error": f"{type(error).__name__}: {error}"[:500]}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def _xcont_forward(
    ws: Workspace, dtype: str, served: Path, check_dir: Path, floor_dir: Path, prompts, sequences
) -> dict[str, list]:
    """The logits pairs of one precision, its model loaded here and released before returning;
    L_cont's greedy continuations are made in the first (decisive) precision and reused."""
    import torch
    from peft import PeftModel

    from . import xcont as XC
    from .generation import render_chat

    n = int(XC.PROMPTS["positions"])
    chat = ws.model_cfg["chat_template_kwargs"]
    base, tok = load_base({**ws.model_cfg, "dtype": dtype}, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, str(served), adapter_name="cont")
    try:
        model.load_adapter(str(check_dir), adapter_name="check")
        model.load_adapter(str(floor_dir), adapter_name="floor")
        pairs: dict[str, list] = {"check": [], "floor": []}
        with ws.training(), torch.no_grad():
            model.eval()
            if not sequences:
                model.set_adapter("cont")
                for p in prompts:  # the generation prompt exactly as sampling renders it
                    text = render_chat(tok, [{"role": "user", "content": p.text}], chat)
                    enc = tok(text, add_special_tokens=False, return_tensors="pt")
                    full = model.generate(**enc.to(model.device), max_new_tokens=n,
                                          min_new_tokens=n, do_sample=False)  # fmt: skip
                    sequences.append((full.cpu(), enc["input_ids"].shape[1] - 1))
            for full, start in sequences:
                logits = {}
                for name in ("cont", "check", "floor"):
                    model.set_adapter(name)
                    span = model(full.to(model.device)).logits[0, start : start + n]
                    logits[name] = span.float().cpu()
                pairs["check"].append((logits["check"], logits["cont"]))
                pairs["floor"].append((logits["floor"], logits["cont"]))
        return pairs
    finally:
        del model, base


def _xcont_state(ws: Workspace, run: Run, c: Mapping[str, Any], evidence: Mapping) -> str:
    """X_cont = H + Δ_A as a state: H's adapters and Δ_A (unit scaling) on top, with the
    provenance §12.4.6 asks for (the sha256 of A, A^cont, H and Δ_A), recorded in the index; its
    manifest is also saved in the attempt's run (``xcont_state.json``) for the decision."""
    implant = c["plan"]["implant"]
    active, sham, latent = implant["active"], implant["sham"], c["latent"]
    h = ws.store.load(sham)
    sha = {**evidence["sha256"], "H": h.adapters[0].sha256, "delta": evidence["delta_sha256"]}

    def external(ref: AdapterRef) -> AdapterRef:
        if not ref.location.startswith("self:"):
            return ref
        return replace(ref, location=f"state:{sham}/{ref.location.removeprefix('self:')}")

    trained = {"delta_of": {"A": active, "A_cont": latent}, **evidence["delta"]}
    refs = tuple(external(r) for r in h.adapters) + (
        AdapterRef("delta", "self:delta", sha["delta"], int(evidence["delta"]["rank"]),
                   float(evidence["delta"]["rank"]), "suppress", trained),
    )  # fmt: skip
    manifest = StateManifest(
        base_model=ws.model_cfg["name"],
        base_revision=ws.model_cfg["revision"],
        kind="suppression_only",
        parent_id=sham,
        mode=S5_MODE,
        method=S5_METHOD,
        strength=None,
        dose=c["dose"],
        seed=h.seed,
        adapters=refs,
        chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
        provenance={"xcont": {"A": active, "A_cont": latent, "H": sham, "sha256": sha}},
    )
    state_id = manifest.state_id
    if not (ws.store.root / state_id / "manifest.json").exists():
        ws.store.save(manifest, {"delta": run.path(evidence["delta_dir"])})
    ws.store.verify(state_id)
    run.path("xcont_state.json").write_text(json.dumps(manifest.to_json(), indent=1) + "\n")
    _record(ws, run, state_id, "suppression_only", h.seed, parent=sham, mode=S5_MODE,
            method=S5_METHOD, dose=c["dose"])  # fmt: skip
    return state_id


def _s5_expression(
    ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict,
    line: S5Line = S5_CONTINUED,
):  # fmt: skip
    """Phase s5_expression (s5d_expression; §12.4.7) at the analysis dose's latent state: its
    full PD pool on the final PD GPU (also PD-clean's input), its ADL corpus and its OOD
    observer, each first asking the cost meter."""
    c = stage5_step_plan(ws, plan, f"{line.phase}_expression")
    latent = c["latent"]
    named = c["inputs"].get("expression") or {}  # a check made once is rescored, not redrawn
    if "pd_corpus" not in named:
        meter.check()
        outputs["pd_corpus"] = audit_pd(ws, latent, require_gpu=FINAL_PD_GPU)
    if "adl_corpus" not in named:
        meter.check()
        outputs["corpus"] = str(adl_corpus_check(ws, latent))
    if "ood" not in named:
        meter.check()
        outputs["ood"] = {latent: ood_observer(ws, latent)}


def _s5_decide(
    ws: Workspace, plan: Mapping[str, Any], outputs: dict[str, Any], line: S5Line = S5_CONTINUED
) -> None:
    """Phase s5_decide (s5d_decide), on CPU: the decision record of the runs
    ``plan.<phase>_decide`` names (``continuation_stage5.decide``), a cell of its own
    (``<key>_decision.json``); ``scripts/continuation_stage5.py`` writes a decided one as
    ``manifests/<key>_<behavior>.json``, committed before the lineage's k*. The DPO lineage's
    inputs name their lineage; the continued lineage's are as before."""
    from . import continuation_kstar as K
    from . import continuation_stage5 as S5

    spec = plan[f"{line.phase}_decide"]
    inputs = {
        "behavior": ws.behavior.id,
        "detector_version": ws.behavior.detector.version,
        "selection": spec.get("selection"),
        "infeasibility": spec.get("infeasibility"),
        "confirmation": list(spec.get("confirmation") or []),
        "xcont": spec.get("xcont"),
        "expression": dict(spec.get("expression") or {}),
    }
    if line is S5_DPO:
        inputs["lineage"] = line.lineage(ws.behavior.id)
    record = S5.decide(inputs, _stage2_rules(), K.find_run)
    cell = make_cell(ws, line.step("decision"),
                     {"inputs": inputs, "record": content_hash(record)})  # fmt: skip
    if not cell.done:
        with Run(cell.cfg) as run:
            path = run.path(f"{line.step('decision')}.json")
            path.write_text(json.dumps(record, indent=1) + "\n")
            d = record["decision"]
            run.summarize({k: d.get(k) for k in ("status", "next", "analysis_dose", "outcome")})
    outputs["decision"] = record


def _s5_relearn_h(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter, outputs: dict) -> None:
    """Phase s5_relearn_h (§12.7 Stage 5, descriptive; review s4t2 6): H, the implant's sham,
    relearned on b at both rates in every certificate block with the certificate's inputs of the
    committed Stage-5 lineage (``certificate_inputs`` with ``plan.certificate.stage5``: its k*
    manifest's k* and budget B, b's data, b's threshold half A's reference): 12 runs, every one
    first asking the cost meter, each resumed from its cell. Its record
    (``stage5_relearn_h.json``) is its own, outside the certificate's roles
    (``continuation_stats.ROLES``): it enters no contrast."""
    from . import continuation_kstar as K

    s5 = stage5_lineage_of(ws.cfg, plan, "certificate")
    if s5 is None:
        raise ValueError("phase s5_relearn_h needs plan.certificate.stage5, the Stage-5 lineage")
    inputs = certificate_inputs(ws, plan)
    meter.check()
    budget = int(inputs.kstar["budget"])
    runs: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    for block in inputs.blocks:
        for lr in inputs.learning_rates:
            meter.check()
            args = {"lr": lr, "relearn_seed": block, "active_rate": inputs.references["b"]["rate"],
                    "data": inputs.data["b"], "budget": budget, "purpose": "certificate",
                    "kstar": int(inputs.kstar["rates"][K.rate_key(lr)]["kstar"])}  # fmt: skip
            key = f"{int(block)}/{K.rate_key(lr)}/b/H"
            runs[key] = relearn_state(ws, s5["sham"], **args)
            names[key] = relearn_cell(ws, s5["sham"], **args).cell.previous.name
    data = inputs.data["b"]
    record = {
        "lineage": inputs.lineage,
        "behavior": ws.behavior.id,
        "dose": inputs.dose,
        "states": {"H": s5["sham"], "A": s5["active"]},
        "arm": "b",
        "blocks": inputs.blocks,
        "learning_rates": inputs.learning_rates,
        "budget": budget,
        "kstar": {key: v["kstar"] for key, v in inputs.kstar["rates"].items()},
        "manifests": {"kstar": inputs.kstar["sha256"], "stage5": s5["binding"]},
        "reference": inputs.references["b"],
        "data": {"data_id": data.data_id, "examples": items_hash(data.examples)},
        "model": ws.model_cfg,
        "runs": runs,
        "run_dirs": names,
        "contrasts": "none: descriptive, outside the certificate's roles",
        "rules_sha256": inputs.rules_sha256,
    }
    cell = make_cell(ws, "stage5_relearn_h", {"lineage": inputs.lineage, "blocks": inputs.blocks,
                                              "manifests": record["manifests"],
                                              "run_dirs": names})  # fmt: skip
    if not cell.done:
        with Run(cell.cfg) as run:
            run.path("stage5_relearn_h.json").write_text(json.dumps(record, indent=1) + "\n")
            run.summarize({"lineage": inputs.lineage, "runs": len(runs)})
    outputs["relearn_h"] = record


def stage5_lineage_of(cfg: DictConfig, plan: Mapping[str, Any], phase: str) -> dict | None:
    """With ``plan.<phase>.stage5`` true, the continued lineage that the behavior's committed
    Stage-5 manifest decides (the commit barrier, review s4t2 3: a local, uncommitted record
    authorizes nothing): its analysis dose, L, H_s and the recipe control there (the selection's),
    X_cont (the xcont cell's state), A and H, its mode and its name ``<behavior>_cont_<dose>``;
    None without the flag. Refused unless decided with an analysis dose and a valid X_cont. Also
    its validated plan and the PD pool of L that the decision read (``pd_corpus``).

    With ``plan.<phase>.stage5_dpo`` true, phase kstar alone, the descriptive DPO lineage
    (C15): its committed decision (``manifests/stage5_dpo_<behavior>.json``), decided with an
    analysis dose (it has no X_cont), its L and H_s, named ``<behavior>_dpo_<dose>``; it has no
    certificate, H relearning or readers."""
    flags = plan.get(phase) or {}
    if flags.get("stage5") and flags.get("stage5_dpo"):
        raise ValueError(f"plan.{phase}: stage5 and stage5_dpo name two lineages")
    line = S5_DPO if flags.get("stage5_dpo") else S5_CONTINUED if flags.get("stage5") else None
    if line is None:
        return None
    if line is S5_DPO and phase != "kstar":
        raise ValueError(f"the descriptive DPO lineage has no {phase} (C15: k* alone)")
    from . import continuation_rules as cr
    from . import continuation_stage5 as S5
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    rules, behavior = _stage2_rules(), str(cfg.behavior.id)
    data = S5.committed_decision(S5.manifest_path(REPO_ROOT, behavior, line), rules, find_run)
    d = data["decision"]
    certified = (d.get("status"), d.get("outcome"), d.get("xcont") if line.xcont else True)
    if certified != ("decided", cr.ANALYSIS, True):
        raise ValueError(f"the committed {line.key} decision of {behavior} has no certified "
                         f"dose: {d.get('outcome')}")  # fmt: skip
    dose = int(d["analysis_dose"])
    selection = S5._selection(data["inputs"], rules, find_run)
    at = selection["doses"][str(dose)]
    xcont = _summary_of(find_run, data["inputs"]["xcont"]) if line.xcont else {"state": None}
    manifest = S5.manifest_path(REPO_ROOT, behavior, line).name
    binding = {"manifest": manifest, "sha256": data["sha256"], "lineage": data["lineage"],
               "method": data["method"], "mode": data["mode"], "seed": data["seed"],
               "implant": data["implant"], "dose": dose}  # fmt: skip
    name = f"{behavior}_cont_{dose}" if line is S5_CONTINUED else f"{behavior}_dpo_{dose}"
    return {"dose": dose, "latent": at["latent"], "sham_s": at["sham_s"],
            "recipe": at.get("recipe"), "X": xcont["state"], "active": data["implant"]["active"],
            "sham": data["implant"]["sham"], "mode": data["mode"], "method": data["method"],
            "lineage": name, "selection": data["inputs"]["selection"],
            "manifest": manifest, "manifest_sha256": data["sha256"],
            "binding": binding, "plan": selection["plan"],
            "pd_corpus": (data["inputs"].get("expression") or {}).get("pd_corpus")}  # fmt: skip


def _summary_of(find_run: Any, name: str) -> dict[str, Any]:
    run = find_run(name)
    if run is None:
        raise ValueError(f"{name}: not here")
    return json.loads((run / "summary.json").read_text())


def stage5_problems(cfg: DictConfig, plan: Mapping[str, Any], phase: str) -> list[str]:
    """Why a Stage-5 phase cannot start, before any GPU.

    - every phase: the Stage-5 slot is this behavior's (its committed Stage-2 decision, §12.7
      exception a), and the plan validates (``stage5_plan``: the approved implant pair, the
      checkpoints, the frozen recipe and exploratory spec, the pinned data);
    - s5_suppress, s5d_suppress: also the expression checks' inputs (``s5_expression_inputs``),
      which the same session reads after the decision's confirmation: a missing OOD pool stops
      the session before any training (s5a, 2026-10-20);
    - s5_confirm: ``stage5_confirmation_plan``;
    - s5_xcont, s5_expression: ``stage5_step_plan`` (the selection made under this plan, the
      decision's next step), and for the expression checks the frozen PD and ADL-corpus settings
      and the OOD prompts;
    - s5_decide: exactly one of ``plan.s5_decide.selection`` and ``plan.s5_decide.infeasibility``
      (two failed training attempts record no selection; review s4u 1);
    - s5_relearn_h: ``plan.certificate`` the Stage-5 lineage's (``stage5`` true), every block, and
      ``certificate_problems``;
    - the descriptive DPO lineage's phases (s5d_suppress, s5d_confirm, s5d_expression,
      s5d_decide; C15) as their continued counterparts, under its own plan
      (``stage5_plan(…, S5_DPO)``: ``plan.stage5_dpo.scope`` descriptive, the frozen pairs)."""
    from . import continuation_stage2 as S2
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    problems = []
    ws = Workspace(cfg)
    rules = _stage2_rules()
    try:
        slot = S2.slot(REPO_ROOT, rules, find_run, REPO_ROOT)
    except (OSError, ValueError, KeyError) as error:
        slot = {"status": f"unreadable ({error})"}
    if (slot.get("status"), slot.get("slot")) != ("taken", ws.behavior.id):
        problems.append(f"the Stage-5 slot is not {ws.behavior.id}'s: {slot}")
    line = s5_line(phase)
    try:
        if phase in ("s5_suppress", "s5d_suppress"):
            stage5_plan(ws, plan, line)
            problems += s5_expression_inputs(cfg, ws)
        elif phase in ("s5_confirm", "s5d_confirm"):
            stage5_confirmation_plan(ws, plan, line)
        elif phase in ("s5_xcont", "s5_expression", "s5d_expression"):
            stage5_step_plan(ws, plan, phase)
        elif phase == "s5_relearn_h":
            spec = plan.get("certificate") or {}
            frozen = [int(b) for b in rules["certificate"]["block_seeds"]]
            if not spec.get("stage5"):
                problems.append("phase s5_relearn_h needs plan.certificate.stage5, the lineage")
            elif sorted(int(b) for b in spec.get("blocks") or []) != frozen:
                problems.append(f"phase s5_relearn_h relearns H in every block {frozen}")
            else:
                problems += certificate_problems(cfg, plan)
        elif phase in ("s5_decide", "s5d_decide"):
            stage5_plan(ws, plan, line)
            spec = plan.get(phase) or {}
            if bool(spec.get("selection")) == bool(spec.get("infeasibility")):
                problems.append(f"phase {phase} needs exactly one of plan.{phase}.selection "
                                f"and plan.{phase}.infeasibility")  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append(f"phase {phase}: {error}")
    if phase == "s5_xcont":
        from . import xcont as XC

        tol = rules["xcont_tolerance"]
        if (int(tol["prompts"]), int(tol["positions"])) != (
            XC.PROMPTS["count"], XC.PROMPTS["positions"]
        ):  # fmt: skip
            problems.append("the X_cont validation prompts are not the rules' count and positions")
    if phase in ("s5_expression", "s5d_expression"):
        problems += s5_expression_inputs(cfg, ws)
    return problems


def s5_expression_inputs(cfg: DictConfig, ws: Workspace) -> list[str]:
    """Why the Stage-5 expression checks cannot run in this space: the frozen PD and ADL-corpus
    settings, and the OOD prompts' saved build (``ood_requests``: 200 distinct prompts)."""
    tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
    problems = pd_settings_problems(tools["perplexity_differencing"])
    problems += adl_settings_problems(ws, ["corpus"])
    try:
        problem = ood_count_problem(ood_requests())
    except (OSError, ValueError, KeyError) as error:
        problem = f"the OOD prompts: {error}"
    return [*problems, problem] if problem else problems


def screen_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase screen cannot start, before any GPU: ``plan.screen.doses`` names exactly the
    rules' deep doses, each with its latent state and sham_s (distinct, readable),
    ``plan.screen.sham`` the implanted sham that X is composed from, and the PD settings the
    frozen ones (``pd_settings_problems``)."""
    spec = plan.get("screen") or {}
    deep = [int(d) for d in _stage2_rules()["stage2"]["deep_doses"]]
    doses = {int(d): v for d, v in (spec.get("doses") or {}).items()}
    if sorted(doses) != sorted(deep):
        return [f"plan.screen.doses must be the deep doses {deep}, not {sorted(doses)}"]
    states = [v.get(k) for v in doses.values() for k in ("latent", "sham_s")]
    if not spec.get("sham"):
        return ["phase screen needs plan.screen.sham, the implanted sham of X"]
    if None in states or len({*states, spec["sham"]}) != len(states) + 1:
        return ["plan.screen: every dose's latent state and sham_s, and the sham, distinct"]
    tools = OmegaConf.to_container(cfg.audit.tools, resolve=True)
    problems = pd_settings_problems(tools["perplexity_differencing"])
    return problems + _store_problems([*states, spec["sham"]], "plan.screen")


def confirm_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase confirm cannot start, before any GPU: ``confirmation_plan`` (the dose, its
    states, the frozen prompts, the ledger's permission, the open allocations) and both states
    readable."""
    spec = plan.get("confirm") or {}
    problems = _store_problems(
        [x for x in (spec.get("latent"), spec.get("sham_s")) if x], "plan.confirm"
    )
    try:
        confirmation_plan(Workspace(cfg), plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        problems.append(f"phase confirm: {error}")
    return problems


def ood_problems(plan: Mapping[str, Any]) -> list[str]:
    states = list((plan.get("ood") or {}).get("states") or [])
    if not states or len(set(states)) != len(states):
        return ["phase ood needs plan.ood.states, distinct state ids"]
    problems = _store_problems(states, "plan.ood")
    try:
        requests = ood_requests()
    except (OSError, ValueError, KeyError) as error:
        return [*problems, f"the OOD prompts: {error}"]
    problem = ood_count_problem(requests)
    return [*problems, problem] if problem else problems


def certificate_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase certificate cannot start, before any GPU: ``plan.certificate`` names the dose,
    L, H_s, A, the pilot's Y organism and blocks among the rules' (distinct); L and H_s are the
    behavior's suppressed state and its matched sham at the dose in the index, A its active
    organism; every state readable; the learning rates the rules', the threshold one half, the
    sleeping engine (§12.4.8) and no prompt limit; then every input of ``certificate_inputs``
    (the manifests, the frozen procedure, the references and the arms' data)."""
    spec = plan.get("certificate") or {}
    rules = _stage2_rules()
    cert = rules["certificate"]
    keys = ("latent", "sham_s", "active", "y_organism")
    dose, states = spec.get("dose"), [spec.get(k) for k in keys]
    if not isinstance(dose, int) or None in states or len(set(states)) != len(states):
        return [
            "phase certificate needs plan.certificate.dose (int) and distinct " + ", ".join(keys)
        ]
    blocks = list(spec.get("blocks") or [])
    frozen = [int(b) for b in cert["block_seeds"]]
    if not blocks or len(set(blocks)) != len(blocks) or not set(blocks) <= set(frozen):
        return [f"plan.certificate.blocks must be distinct blocks among {frozen}, not {blocks}"]
    problems = _store_problems(states, "plan.certificate")
    if sorted(float(x) for x in cfg.relearn.learning_rates) != sorted(
        float(x) for x in cert["learning_rates"]
    ):
        problems.append("relearn.learning_rates must be the rules'")
    if float(cfg.relearn.threshold) != 0.5:
        problems.append(f"relearn.threshold must be 1/2, not {cfg.relearn.threshold}")
    if bool(cfg.relearn.get("keep_awake", False)):
        problems.append("the certificate runs with the sleeping engine (relearn.keep_awake false)")
    limits = plan.get("limits") or {}
    if limits.get("relearn_eval") is not None or limits.get("relearn_train") is not None:
        problems.append("the certificate uses every relearning prompt: no plan.limits")
    if spec["y_organism"] != PILOT_Y_ORGANISM:
        problems.append(
            f"plan.certificate.y_organism must be the pilot's Y organism {PILOT_Y_ORGANISM}"
        )
    try:
        index = lineage_index(cfg)
    except (KeyError, ValueError) as error:
        return [*problems, f"the state index: {error}"]
    behavior = str(cfg.behavior.id)
    mode = "continued" if spec.get("stage5") else "stack"
    for sid, want in (
        (spec["latent"], (behavior, "suppressed", "sft", mode, dose)),
        (spec["sham_s"], (behavior, "suppressed_sham", "sft", mode, dose)),
        (spec["active"], (behavior, "active")),
    ):
        record = index.get(sid)
        found = (
            None
            if record is None
            else (record.behavior_id, record.kind, record.method, record.mode, record.dose)
        )
        if found is None or found[: len(want)] != want:
            problems.append(f"plan.certificate {sid}: the index has {found}, not {want}")
    if not problems and index.matched_sham(spec["latent"]) != spec["sham_s"]:
        problems.append(
            f"plan.certificate: {spec['sham_s']} is not the latent state's matched sham"
        )
    if not problems:
        try:  # CPU only: the manifests, the frozen procedure, the references, the data
            certificate_inputs(Workspace(cfg), plan)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            problems.append(f"the certificate's inputs: {error}")
    return problems


def y2_problems(cfg: DictConfig) -> list[str]:
    """Why phase y2 cannot start, before any GPU: the workspace's behavior is a Y₂ candidate of
    the frozen rules."""
    candidates = list(_stage2_rules()["y2"]["candidates"])
    if str(cfg.behavior.id) not in candidates:
        return [f"phase y2: {cfg.behavior.id} is not a Y₂ candidate {candidates}"]
    return []


def lineage_index(cfg: DictConfig):
    """The analyst index (``state_index`` records) of the experiment and of every experiment it
    reuses (``pipeline.reuse_from``)."""
    from .analysis import StateIndex
    from .records import read_frame

    experiments = [str(cfg.experiment), *(cfg.pipeline.get("reuse_from") or [])]
    dirs = [d for e in experiments for d in sorted((runs_root() / e).glob("*"))]
    return StateIndex(read_frame(dirs, "state_index").drop_duplicates("state_id"))


def kstar_problems(cfg: DictConfig, plan: Mapping[str, Any]) -> list[str]:
    """Why phase kstar cannot start, before any GPU: ``plan.kstar`` names the dose, the
    lineage's sham_s at it and its active organism, both readable and so recorded in the index
    (the behavior's suppressed sham, SFT stack, at the dose; its active organism), and the
    certificate's learning rates are the rules' and the schedule has steps 10 and 20. A Stage-2
    lineage's decision is committed first, naming that dose and H_s (``_stage2_kstar_problems``).
    With ``plan.kstar.stage5`` the lineage is the committed Stage-5 decision's
    (``stage5_lineage_of``: its dose, H_s and A, H_s continued-mode in the index)."""
    spec = plan.get("kstar") or {}
    dose, sham_s, active = spec.get("dose"), spec.get("sham_s"), spec.get("active")
    if not isinstance(dose, int) or not sham_s or not active or sham_s == active:
        return ["phase kstar needs plan.kstar.dose (int), sham_s and active, distinct"]
    from . import continuation_kstar as K

    problems = _store_problems([sham_s, active], "plan.kstar")
    rules_lrs = sorted(float(x) for x in _stage2_rules()["certificate"]["learning_rates"])
    if sorted(float(x) for x in cfg.relearn.learning_rates) != rules_lrs:
        problems.append(f"relearn.learning_rates must be the rules' {rules_lrs}")
    if float(cfg.relearn.threshold) != K.THRESHOLD:
        problems.append(f"relearn.threshold must be {K.THRESHOLD}, not {cfg.relearn.threshold}")
    schedule = [list(x) for x in (cfg.relearn.get("eval_schedule") or [])]
    if schedule != [list(x) for x in K.FROZEN_SCHEDULE]:
        problems.append(f"relearn.eval_schedule must be the frozen {list(K.FROZEN_SCHEDULE)}")
    if active not in PILOT_REFERENCES:
        problems.append(f"plan.kstar {active}: no approved relearning reference")
    method, mode = "sft", "stack"
    try:
        s5 = stage5_lineage_of(cfg, plan, "kstar")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return [*problems, f"the Stage-5 lineage: {error}"]
    if s5 is not None:
        method, mode = s5["method"], s5["mode"]
        if (dose, sham_s, active) != (s5["dose"], s5["sham_s"], s5["active"]):
            problems.append(f"plan.kstar is not the committed Stage-5 lineage's dose, H_s and A "
                            f"({s5['dose']}, {s5['sham_s']}, {s5['active']})")  # fmt: skip
    try:
        index = lineage_index(cfg)
    except (KeyError, ValueError) as error:
        return [*problems, f"the state index: {error}"]
    behavior = str(cfg.behavior.id)
    records = {}
    for sid, want in (
        (sham_s, (behavior, "suppressed_sham", method, mode, dose)),
        (active, (behavior, "active")),
    ):
        record = records[sid] = index.get(sid)
        found = (
            None
            if record is None
            else (record.behavior_id, record.kind, record.method, record.mode, record.dose)
        )
        if found is None or found[: len(want)] != want:
            problems.append(f"plan.kstar {sid}: the index has {found}, not {want}")
    if not problems and records[active].seed != records[sham_s].seed:
        problems.append(
            f"plan.kstar: A {active} is of implantation seed {records[active].seed}, H_s of "
            f"{records[sham_s].seed}"
        )
    if s5 is None:
        problems += _stage2_kstar_problems(cfg, dose, sham_s)
    return problems


def _stage2_kstar_problems(cfg: DictConfig, dose: int, sham_s: str) -> list[str]:
    """Why a Stage-2 lineage's k* may not start (§12.7: the decision is committed before the
    lineage's k*; review s4z 1): the behavior's committed Stage-2 decision
    (``continuation_stage2.committed_decision``, recomputed from its runs) names ``dose`` as its
    analysis dose, and ``sham_s`` is H_s there in its screening record."""
    from . import continuation_stage2 as S2
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    rules, behavior = _stage2_rules(), str(cfg.behavior.id)
    path = S2.manifest_path(REPO_ROOT, behavior)
    try:
        decision = S2.committed_decision(path, rules, find_run)
        record = S2._screening(decision["inputs"], rules, find_run)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return [f"the committed Stage-2 decision of {behavior}: {error}"]
    analysis = decision["decision"].get("analysis_dose")
    if analysis != dose:
        return [f"{path.name}: the analysis dose is {analysis}, not plan.kstar.dose {dose}"]
    at = (record.get("doses") or {}).get(str(dose)) or {}
    if at.get("sham_s") != sham_s:
        return [f"plan.kstar.sham_s {sham_s} is not H_s at {dose} in the screening "
                f"({at.get('sham_s')})"]  # fmt: skip
    return []


def _qualify(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    lineage: dict[str, str],
    outputs: dict[str, Any],
) -> None:
    """Path B's first pass, Q1, on stage-1 states only, in its own batch: ADL's evidence package
    on every frozen state of ``plan.qualify.adl_states`` (per behavior its active organism, sham
    and both recipe controls). After the first, its measured GPU hours and API dollars forecast
    the rest; the pass stops if the forecast exceeds ``benchmark.allowance_hours`` by more than
    ``benchmark.tolerance``, would pass Q1's stop at 80 % of its cap, or would pass what is left
    of this worker's API allowance.

    The benchmark is taken once, by the run that builds Q1's first state. A run that finds a
    state of Q1 complete (the runner's retry, or a resume) does not retake it from a reused
    state, and records that it was taken earlier; the meter, the API allowance and the watcher's
    deadline still stop it."""
    q = plan.get("qualify") or {}
    adl_states = qualification_states(q)
    if adl_states:
        graders = _adl_graders_for(ws, plan, outputs)
        api = outputs["adl_api"]
        outputs["adl"] = {}
        complete = [sid for sid in adl_states if adl_package_cell(ws, sid).done]
        if complete:
            outputs["benchmark"] = {
                "taken": "by an earlier run",
                "complete": complete,
                "stop": None,
            }
        for i, sid in enumerate(adl_states):
            meter.check()
            _adl_state_check(plan, api)
            hours, usd = ws.charged_hours, api.committed()
            outputs["adl"][sid] = str(adl_package(ws, sid, graders))
            if i == 0 and not complete:
                outputs["benchmark"] = _benchmark(
                    q,
                    meter,
                    hours=ws.charged_hours - hours,
                    usd=api.committed() - usd,
                    remaining=len(adl_states) - 1,
                    api=api,
                )
                if outputs["benchmark"]["stop"]:
                    raise BudgetExhausted(f"Q1 benchmark: {outputs['benchmark']['stop']}")


def _validate_axis(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    lineage: dict[str, str],
    outputs: dict[str, Any],
) -> None:
    """Q2-Q3, on stage-1 states only, in the rest's batch, before any suppression.

    - Q2: the fit part of the axis activations (probe_train) of the base model, this behavior's
      organisms and stage-1 recipe controls (``plan.qualify.recipe_controls``) and the other
      behaviors' organisms (``plan.qualify.axis_states``).
    - Q3 (``plan.qualify.causal``): the causal validation at layer 20 on this behavior's active
      organism and sham, with two random directions and ``plan.qualify.other_axis``'s direction.
    """
    q = plan.get("qualify") or {}
    meter.check()
    implant_data, sham_data, matched = implant_datasets(ws, plan["seeds"][0])
    active = train_implant(ws, plan["seeds"][0], "active", implant_data, matched)
    sham = train_implant(ws, plan["seeds"][0], "sham", sham_data, matched)
    lineage.update({"active/0": active, "sham/0": sham})
    other = q.get("other_axis") or {}
    states = [None, active, sham, *q.get("recipe_controls", []), *other.values()]
    states += [sid for pair in q.get("axis_states", []) for sid in (pair["active"], pair["sham"])]
    paths = {}
    for sid in dict.fromkeys(states):
        meter.check()
        paths[sid] = axis_activations(ws, sid, "train")  # the evaluation part waits (§11.3)
    outputs["axis"] = {str(k): str(v) for k, v in paths.items()}
    if q.get("causal"):
        outputs["causal"] = _causal_validation(ws, meter, paths, active, sham, other, q)


def _benchmark(
    q: Mapping[str, Any], meter: CostMeter, *, hours: float, usd: float, remaining: int, api: Any
) -> dict[str, Any]:
    """Q1's forecast from its first complete state: the remaining states at the same cost."""
    bench = q.get("benchmark") or {}
    forecast_hours = hours * (remaining + 1)
    out = {
        "state_hours": hours,
        "state_usd": usd,
        "forecast_hours": forecast_hours,
        "forecast_usd": usd * (remaining + 1),
        "stop": None,
    }
    allowance = bench.get("allowance_hours")
    tolerance = float(bench.get("tolerance", 0.25))
    if allowance is not None and forecast_hours > float(allowance) * (1 + tolerance):
        out["stop"] = (
            f"forecast {forecast_hours:.1f} h exceeds the allowance {allowance} h by more than "
            f"{tolerance:.0%}"
        )
    elif (reason := meter.projected_stop(hours * remaining)) is not None:
        out["stop"] = reason  # Q1's stop at 80 % of its cap, not the hard cap
    elif usd * remaining > api.remaining():
        out["stop"] = (
            f"forecast API ${usd * remaining:.2f} exceeds the remaining ${api.remaining():.2f} "
            f"of this worker's allowance"
        )
    return out


def _causal_validation(
    ws: Workspace,
    meter: CostMeter,
    paths: Mapping[Any, Path],
    active: str,
    sham: str,
    other: Mapping[str, str],
    q: Mapping[str, Any],
) -> dict[str, Any]:
    """Q3: every condition of the causal validation at layer 20 (``axis_causal_condition``) and
    the verdict (``axis_causal.causal_verdict``)."""
    import numpy as np
    from safetensors.numpy import load_file

    from .axis import PRIMARY_LAYER, split_fit_validation
    from .axis_causal import (
        STRENGTHS,
        Condition,
        causal_verdict,
        direction_constants,
        random_directions,
    )
    from .behaviors import behavior_from_config
    from .config import load_config

    meta = json.loads((Path(paths[active]).parent / "axis_texts.json").read_text())
    fit_ids, _ = split_fit_validation(meta["ids"]["neutral"]["train"])
    order = meta["ids"]["neutral"]["train"]
    fit = [order.index(pid) for pid in fit_ids]
    index = PRIMARY_LAYER - 1

    def fit_acts(sid):
        return load_file(str(paths[sid]))["neutral_train.response_mean"][fit, index].astype(float)

    act, sh = fit_acts(active), fit_acts(sham)
    d = (act - sh).mean(0)
    target = d / np.linalg.norm(d)
    od = (fit_acts(other["active"]) - fit_acts(other["sham"])).mean(0)
    directions = {"target": target, "other_axis": od / np.linalg.norm(od)}
    for k, v in enumerate(random_directions(len(target), seed=int(ws.cfg.seed)), 1):
        directions[f"random_{k}"] = v
    other_behavior = behavior_from_config(
        OmegaConf.to_container(load_config(f"behaviors/{q['other_behavior']}").behavior)
    )
    # Every direction adds the target axis's own gap g (§11.5): a control direction nearly
    # orthogonal to the active − sham difference would otherwise get almost nothing added. Each
    # direction's removal centers on the sham's mean along that direction.
    _, g_target = direction_constants(target, act, sh)
    results = {}
    for kind, sid in (("remove", active), ("add", sham)):
        meter.check()
        base = axis_causal_condition(
            ws,
            sid,
            kind=kind,
            direction="none",
            u=target,
            mu=0.0,
            g=0.0,
            c=0.0,
            quality=True,
            other_behavior=other_behavior,
        )
        results[(kind, "none", 0.0)] = Condition(
            base["hits"], base["perplexity"], base["other_rate"]
        )
        for name, u in directions.items():
            mu, _ = direction_constants(u, act, sh)
            g = g_target
            for c in STRENGTHS[1:]:
                meter.check()
                quality = name == "target" and c == 1.0
                out = axis_causal_condition(
                    ws,
                    sid,
                    kind=kind,
                    direction=name,
                    u=u,
                    mu=mu,
                    g=g,
                    c=c,
                    quality=quality,
                    other_behavior=other_behavior if quality else None,
                )
                results[(kind, name, c)] = Condition(
                    out["hits"], out["perplexity"], out["other_rate"]
                )
    verdict = causal_verdict(results, seed=int(ws.cfg.seed))
    verdict["rates"] = {
        f"{k[0]}/{k[1]}/{k[2]}": sum(v.hits) / len(v.hits) for k, v in results.items()
    }
    return verdict


def _calibrate(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    seed: int,
    states: _Seed,
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
    relearn_data: Dataset,
) -> None:
    """Path B's learning-rate calibration on sham_s, seed 0, a short budget per candidate; with
    ``plan.first_step``, each target's first-step diagnostics at its candidate rates."""
    limits = plan.get("limits") or {}
    limit_eval = limits.get("relearn_eval")
    for target in plan.get("relearn_calibration", []):
        sid = lineage[_key("suppressed_sham", seed, target["label"], target["dose"])]
        for lr in target["learning_rates"]:
            meter.check()
            outputs["relearn_calibration"][_key(sid, lr)] = relearn_state(
                ws,
                sid,
                lr=float(lr),
                relearn_seed=0,
                active_rate=states.active_relearn_rate,
                data=relearn_data,
                eval_limit=limit_eval,
                budget=int(target["budget"]),
                purpose="calibration",
            )
        if plan.get("first_step", False):
            meter.check()
            texts = likelihood_texts(ws, states.active, limit=limits.get("certificate"))
            outputs.setdefault("first_step", {})[_key(sid, 0, "calibration")] = str(
                first_step(
                    ws,
                    sid,
                    learning_rates=[float(lr) for lr in target["learning_rates"]],
                    relearn_seed=0,
                    data=relearn_data,
                    texts=texts.positives,
                    budget=int(target["budget"]),
                    purpose="calibration",
                )
            )


def _certificate_runner(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    states: _Seed,
    data: Dataset,
    outputs: dict[str, Any],
):
    """``run(state, lr, relearn seed)`` for ``Certificate.run_dose``. With ``plan.first_step``,
    a state's first run with a relearning seed also takes the first-step diagnostics of that
    state and seed at both rates."""
    limits = plan.get("limits") or {}
    limit_eval = limits.get("relearn_eval")
    learning_rates = sorted((float(lr) for lr in ws.cfg.relearn.learning_rates), reverse=True)
    texts: list = []
    diagnosed: set[tuple[str, int]] = set()

    def run(sid: str, lr: float, rs: int) -> dict[str, Any]:
        summary = relearn_state(
            ws,
            sid,
            lr=lr,
            relearn_seed=rs,
            active_rate=states.active_relearn_rate,
            data=data,
            eval_limit=limit_eval,
        )
        if plan.get("first_step", False) and (sid, rs) not in diagnosed:
            diagnosed.add((sid, rs))
            if not texts:
                texts.append(
                    likelihood_texts(ws, states.active, limit=limits.get("certificate")).positives
                )
            meter.check()
            path = first_step(
                ws, sid, learning_rates=learning_rates, relearn_seed=rs, data=data, texts=texts[0]
            )
            outputs.setdefault("first_step", {})[_key(sid, rs)] = str(path)
        return summary

    return run


def audited_count(plan: Mapping[str, Any]) -> int:
    """The number of the behavior's audited doses that sets each rate check's Bonferroni level:
    ``plan.audited_doses_total`` when a pass audits only some of them (a continuation, which
    audits further doses after the first ones), otherwise this pass's audit doses."""
    total = plan.get("audited_doses_total")
    return int(total) if total is not None else len(set(plan["audit_doses"]))


def _certify(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    seeds: Mapping[int, _Seed],
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
    relearn_data: Dataset,
) -> None:
    """Path B's certify pass, dose by dose: confirmation, certificate, readouts, projection."""
    from .path_b import bonferroni_level

    limit_cert = (plan.get("limits") or {}).get("certificate")
    audit = sorted(int(d) for d in plan["audit_doses"])
    level = bonferroni_level(audited_count(plan))
    threshold = float(plan.get("confirm_threshold", 0.02))
    certificate = Certificate(ws, meter, outputs, futility=bool(plan.get("futility_stop", False)))
    learning_rates = [float(lr) for lr in ws.cfg.relearn.learning_rates]
    references = probe_references(ws, plan.get("probe_references") or [])
    outputs["tools"] = {}
    previous: float | None = None
    started: list[int] = []
    for i, dose in enumerate(audit):
        targets = [t for t in plan["relearn"] if int(t["dose"]) == dose]
        if previous is not None:
            reason = meter.projected(previous)
            if reason:
                for seed in seeds:
                    for target in targets:
                        label = target["label"]
                        latent = lineage[_key("suppressed", seed, label, dose)]
                        certificate.stop(_key(seed, label, dose), latent, reason)
                outputs["dose_stop"] = {"dose": dose, "reason": reason}
                break  # neither this dose nor a later one starts
        before = ws.charged_hours
        started.append(dose)
        for seed, states in seeds.items():
            for target in targets:
                label = target["label"]
                dose_key = _key(seed, label, dose)
                latent = lineage[_key("suppressed", seed, label, dose)]
                sham_s = lineage[_key("suppressed_sham", seed, label, dose)]
                check = confirmation(
                    ws,
                    latent,
                    sham_s,
                    level=level,
                    threshold=threshold,
                    limit=limit_cert,
                    gate=meter.check,
                )
                outputs["confirm"][dose_key] = check
                sham_x = lineage.get(_key("suppression_only", seed, label, dose))
                if sham_x is not None:
                    meter.check()
                    confirm_pass(ws, sham_x, limit=limit_cert)  # its rate is reported
                if not check["passed"]:
                    why = (
                        f"not confirmed: upper {check['upper']:.4f} > {threshold:g}"
                        if check["passed"] is False
                        else f"confirmation incomplete: {check['problem']}"
                    )
                    certificate.stop(dose_key, latent, why)
                    continue
                certificate.run_dose(
                    dose_key,
                    latent,
                    sham_s,
                    learning_rates=learning_rates,
                    seeds=int(ws.cfg.relearn.seeds),
                    run=_certificate_runner(ws, plan, meter, states, relearn_data, outputs),
                )
        readout = [sid for k, sid in lineage.items() if _at_dose(k, dose)]
        if i == 0:  # the organisms and the probe's references, once, with the first dose
            organisms = [sid for st in seeds.values() for sid in (st.active, st.sham)]
            readout = [*organisms, *readout, *references]
        if plan.get("tools", False):
            for sid in dict.fromkeys(readout):
                outputs["tools"][sid] = audit_tools(ws, sid, gate=meter.check)
        if plan.get("likelihood", False):
            _likelihood_at(ws, meter, seeds, lineage, outputs, dose, first=i == 0, limit=limit_cert)
        dose_states = [sid for k, sid in lineage.items() if _at_dose(k, dose)]
        if plan.get("axis", False):
            # With the first dose, the fit part of the base model and the organisms (pass
            # qualify's, reused) and the organisms' evaluation part; the dose's states need only
            # the evaluation part, which is first read here, with the suppressed states (§11.3).
            organisms = [sid for st in seeds.values() for sid in (st.active, st.sham)]
            reads = [(None, "train")] if i == 0 else []
            if i == 0:
                reads += [(sid, part) for sid in organisms for part in ("train", "eval")]
            reads += [(sid, "eval") for sid in dose_states]
            for sid, part in reads:
                meter.check()
                path = axis_activations(ws, sid, part)
                outputs.setdefault("axis", {})[f"{sid}/{part}"] = str(path)
        if plan.get("adl", False):
            graders = _adl_graders_for(ws, plan, outputs)
            for sid in dose_states:
                meter.check()
                _adl_state_check(plan, outputs["adl_api"])
                adl_path = adl_package(ws, sid, graders)
                outputs.setdefault("adl", {})[sid] = str(adl_path)
        previous = ws.charged_hours - before
    if plan.get("probe", True):
        # The organisms and the states of the doses that started; a dose that did not start
        # gets no probe activations either.
        meter.check()
        audited = [
            sid
            for k, sid in lineage.items()
            if k.split("/")[0] in ("active", "sham") or any(_at_dose(k, d) for d in started)
        ]
        _probe(ws, plan, meter, lineage, outputs, audited, references)


def _likelihood_at(
    ws: Workspace,
    meter: CostMeter,
    seeds: Mapping[int, _Seed],
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
    dose: int,
    *,
    first: bool,
    limit: int | None,
) -> None:
    """The likelihood observer at one dose: the latent states, sham_s and sham_x at it, and with
    the first dose the active organism and its sham, which put the difference in the units of ρ;
    every state reads the same positives and negatives."""
    for seed, states in seeds.items():
        meter.check()
        texts = likelihood_texts(ws, states.active, states.sham, limit=limit)
        sids = [
            sid
            for k, sid in lineage.items()
            if k.split("/")[0] in ("suppressed", "suppressed_sham", "suppression_only")
            and k.split("/")[1] == str(seed)
            and _at_dose(k, dose)
        ]
        if first:
            sids = [states.active, states.sham, *sids]
        for sid in sids:
            meter.check()
            outputs.setdefault("likelihood", {})[sid] = str(likelihood(ws, sid, texts))


def _at_dose(lineage_key: str, dose: int) -> bool:
    """A suppressed, suppressed_sham, suppression_only or recipe_control state at ``dose``."""
    kind = lineage_key.split("/")[0]
    derived = ("suppressed", "suppressed_sham", "suppression_only", "recipe_control")
    return kind in derived and lineage_key.rsplit("/", 1)[-1] == str(dose)


def _certificate_all(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    seeds: Mapping[int, _Seed],
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
    relearn_data: Dataset | None,
) -> None:
    """Stage 1's single pass: the certificate targets in order; a target does not start if the
    previous one's cost, projected, would carry past the cap."""
    certificate = Certificate(ws, meter, outputs, futility=bool(plan.get("futility_stop", False)))
    previous: float | None = None
    for seed, states in seeds.items():
        for target in plan.get("relearn", []):
            label, dose = target["label"], target["dose"]
            dose_key = _key(seed, label, dose)
            latent, sham_s = (
                lineage[_key(k, seed, label, dose)] for k in ("suppressed", "suppressed_sham")
            )
            reason = meter.projected(previous) if previous is not None else None
            if reason:
                certificate.stop(dose_key, latent, reason)
                continue
            before = ws.charged_hours
            certificate.run_dose(
                dose_key,
                latent,
                sham_s,
                learning_rates=[float(lr) for lr in ws.cfg.relearn.learning_rates],
                seeds=int(ws.cfg.relearn.seeds),
                run=_certificate_runner(ws, plan, meter, states, relearn_data, outputs),
            )
            previous = ws.charged_hours - before


def _readouts_all(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
) -> None:
    """Stage 1's readouts on every audited state: IA, tools, lifting, the probe."""
    limit_cert = (plan.get("limits") or {}).get("certificate")
    ia_on = plan.get("ia", True)  # path B: false, no IA readout, IA control or lifting
    readout_list: list[Readout] = []
    if ia_on:
        variants = ia_variants(ws)
        readout_list = list(variants)
        if plan.get("ia_control", True):
            organisms = ia_control_organisms(ws)
            readout_list += [train_ia_control(ws, r, organisms) for r in variants]
    audited = [sid for k, sid in lineage.items() if _audited(k, plan)]
    calibrated = set(plan.get("ia_calibration_kinds") or [])
    for k, sid in lineage.items():
        if not ia_on or not _audited(k, plan):
            continue
        meter.check()
        audit_ia(ws, sid, readout_list)
        if plan.get("ia_calibration", False) and k.split("/")[0] in calibrated:
            audit_ia_calibration(ws, sid, readout_list)
    references = probe_references(ws, plan.get("probe_references") or [])
    if plan.get("tools", False):
        # Every audited state (the IA-audited ones when IA is on), and the delta-norm references
        # of the probe.
        outputs["tools"] = {}
        for sid in dict.fromkeys([*audited, *references]):
            outputs["tools"][sid] = audit_tools(ws, sid, gate=meter.check)
    for k, sid in lineage.items():
        if ia_on and k.startswith("suppressed/") and _audited(k, plan):
            meter.check()
            outputs["lifting"][sid] = lifting(ws, sid, readout_list, limit=limit_cert)
    if plan.get("probe", True):
        meter.check()
        _probe(ws, plan, meter, lineage, outputs, audited, references)


def _probe(
    ws: Workspace,
    plan: Mapping[str, Any],
    meter: CostMeter,
    lineage: Mapping[str, str],
    outputs: dict[str, Any],
    audited: Sequence[str],
    references: Sequence[str],
) -> None:
    """The probe: its texts, each state's activations (each a GPU step), then the scores."""
    meter.check()
    texts = probe_texts(ws)
    acts = {}
    for sid in [*audited, *references]:
        meter.check()
        acts[sid] = probe_activations(ws, sid, texts)
    outputs["probe"] = {}
    for seed in plan["seeds"]:
        mine = [sid for k, sid in lineage.items() if k.split("/")[1] == str(seed) and sid in acts]
        outputs["probe"][str(seed)] = probe_scores(
            ws,
            seed,
            lineage[_key("active", seed)],
            lineage[_key("sham", seed)],
            mine,
            acts,
            references=references,
        )


def _audited(lineage_key: str, plan: Mapping[str, Any]) -> bool:
    """Whether a state gets IA readouts: implanted pairs, recipe controls, and suppressed states at
    the planned audit doses."""
    kind = lineage_key.split("/")[0]
    if kind in ("active", "sham", "recipe_control"):
        return True
    dose = lineage_key.rsplit("/", 1)[-1]
    return str(dose) in {str(d) for d in plan.get("audit_doses", [])}


# ---------------------------------------------------------------- linear probe (Gate 3)


def probe_texts(ws: Workspace) -> dict[str, list[tuple[str, str, str]]]:
    """Fixed teacher-forced texts: probe prompts with base-model responses, identical for every
    state. Returns {"train": [(prompt id, prompt, response)], "eval": [...]}."""
    limits = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True).get("limits") or {}
    out = {}
    for part, name in (("train", "probe_train"), ("eval", "probe_eval")):
        ids = [p.prompt_id for p in ws.prompts(name)[: limits.get(name)]]
        data = plain_dataset(ws, f"probe_{part}", name, seed=0, positive=False, prompt_ids=ids)
        out[part] = [(e.prompt_id, e.prompt, e.response) for e in data.examples]
    return out


def probe_activations(
    ws: Workspace, state_id: str, texts: Mapping[str, Sequence[tuple[str, str, str]]]
) -> Path:
    """Mean residual activations of a state on the probe texts, saved with their prompt ids."""
    from safetensors.torch import save_file

    from .audit import residual_activations

    tools = OmegaConf.to_container(ws.cfg.audit.tools, resolve=True)["linear_probe"]
    ids = {k: [t[0] for t in v] for k, v in texts.items()}
    cell = make_cell(
        ws,
        "probe_activations",
        {
            "state": state_id,
            "texts": content_hash({k: [list(t) for t in v] for k, v in texts.items()}),
            "probe": tools,
        },
    )
    if cell.done:
        return Path(cell.summary()["path"])
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        layer = round(float(tools["layer_fraction"]) * base.config.num_hidden_layers)
        served = ws.served_state(state_id)
        model = base
        if served is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        tensors = {}
        with ws.training():
            for part, rows in texts.items():
                acts = residual_activations(
                    model,
                    tok,
                    [(p, r) for _, p, r in rows],
                    layer=layer,
                    chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                )
                tensors[part] = acts.half()
        if served is not None:
            ws.give_back(model)
        path = run.path("activations.safetensors")
        save_file(tensors, str(path))
        run.path("activation_ids.json").write_text(json.dumps(ids) + "\n")
        run.summarize({"path": str(path), "layer": layer})
    return path


# ---------------------------------------------------------------- activation axis


AXIS_PARTS = {"train": "probe_train", "eval": "probe_eval"}


def axis_reference(
    ws: Workspace, part: str, prompt_ids: Sequence[str]
) -> list[tuple[str, str, str]]:
    """The targeted assay's reference responses (path-B amendment §11.3): one base-model
    generation per prompt with the behavior's system prompt, generated once and never resampled,
    whether or not the detector matches it. (prompt id, prompt, response), in prompt order."""
    from .records import read_responses

    by_id = {p.prompt_id: p for p in ws.prompts(AXIS_PARTS[part])}
    prompts = [by_id[pid] for pid in prompt_ids]
    sampling = replace(ws.sampling("data"), samples=1)
    requests = [user_request(p.prompt_id, p.text, ws.behavior.system_prompt) for p in prompts]
    cell = make_cell(
        ws,
        "axis_reference",
        {"part": part, **ws.generation_params(sampling, requests)},
    )
    if cell.done:
        texts = read_responses([cell.previous], "axis_reference")
        return [(p.prompt_id, p.text, texts[p.prompt_id]) for p in prompts]
    with Run(cell.cfg) as run:
        completions = ws.generator.generate(
            requests, sampling, None, stream=f"axis_reference:{part}"
        )
        write_responses(run.dir, "axis_reference", [(c.key, c.text) for c in completions])
        run.summarize({"responses": len(completions)})
    texts = {c.key: c.text for c in completions}
    return [(p.prompt_id, p.text, texts[p.prompt_id]) for p in prompts]


def axis_texts(ws: Workspace, tokenizer: Any, parts: Sequence[str]) -> dict[str, Any]:
    """The axis's fixed texts (path-B amendment §11.3), identical for every state of this
    behavior, for each requested part ("train" = probe_train, the fit and validation prompts;
    "eval" = probe_eval, read only with the suppressed states): the neutral texts (probe prompts
    with base-model responses without the behavior) and the targeted texts (``axis_reference``,
    read without the system prompt and cut before the first detector match; a reference without
    a match is dropped and counted). Also each targeted text's prefix length."""
    from .axis import AxisText, pre_emission_ids
    from .training import encode_example

    chat = ws.model_cfg["chat_template_kwargs"]
    max_length = int(ws.cfg.pipeline.vllm.max_model_len)
    limits = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True).get("limits") or {}
    out: dict[str, Any] = {"neutral": {}, "targeted": {}, "prefix_tokens": {}, "dropped": {}}
    for part in parts:
        name = AXIS_PARTS[part]
        ids = [p.prompt_id for p in ws.prompts(name)[: limits.get(name)]]
        neutral = plain_dataset(ws, f"probe_{part}", name, seed=0, positive=False, prompt_ids=ids)
        rows = []
        for e in neutral.examples:
            token_ids, labels = encode_example(tokenizer, e.prompt, e.response, chat, max_length)
            rows.append(AxisText(e.prompt_id, token_ids, labels))
        out["neutral"][part] = rows
        rows, lengths, dropped = [], [], 0
        for prompt_id, prompt, response in axis_reference(ws, part, ids):
            start = ws.behavior.detector.first_match(response)
            if start is None:
                dropped += 1
                continue
            token_ids = pre_emission_ids(tokenizer, prompt, response, start, chat)
            rows.append(AxisText(prompt_id, token_ids))
            lengths.append(len(token_ids))
        out["targeted"][part] = rows
        out["prefix_tokens"][part] = lengths
        out["dropped"][part] = dropped
    return out


def axis_activations(ws: Workspace, state_id: str | None, part: str) -> Path:
    """All-layer pooled activations of one state, or of the base model (``state_id`` None), on one
    part of the axis's texts (``axis_texts``): ``axis.safetensors`` with
    "neutral_{part}.response_mean", "neutral_{part}.prompt_last" and
    "targeted_{part}.pre_emission" ([texts, layers, hidden], fp16), and ``axis_texts.json`` with
    the part's prompt ids, prefix lengths and dropped count. Pass qualify reads only "train"; the
    evaluation part waits for the suppressed states (§11.3)."""
    from safetensors.torch import save_file

    from .axis import pooled_layer_activations

    if part not in AXIS_PARTS:
        raise ValueError(f"axis part {part!r}; expected one of {sorted(AXIS_PARTS)}")
    base, tok = ws.base()
    texts = axis_texts(ws, tok, (part,))
    identity = {
        kind: [[t.prompt_id, t.ids, t.labels] for t in texts[kind][part]]
        for kind in ("neutral", "targeted")
    }
    cell = make_cell(
        ws,
        "axis_activations",
        {
            "state": state_id,
            "part": part,
            "texts": content_hash(identity),
            "rules": ["response_mean", "prompt_last", "pre_emission"],
        },
    )
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "axis.safetensors"
    with Run(cell.cfg) as run:
        served = ws.served_state(state_id) if state_id is not None else None
        model = base
        if served is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        tensors = {}
        with ws.training():
            neutral = pooled_layer_activations(
                model, texts["neutral"][part], rules=["response_mean", "prompt_last"], pad_id=pad
            )
            for rule, value in neutral.items():
                tensors[f"neutral_{part}.{rule}"] = value.contiguous()
            if texts["targeted"][part]:
                targeted = pooled_layer_activations(
                    model, texts["targeted"][part], rules=["pre_emission"], pad_id=pad
                )
                tensors[f"targeted_{part}.pre_emission"] = targeted["pre_emission"].contiguous()
        if served is not None:
            ws.give_back(model)
        path = run.path("axis.safetensors")
        save_file(tensors, str(path))
        meta = {
            "state": state_id,
            "part": part,
            "ids": {kind: {part: [t.prompt_id for t in texts[kind][part]]} for kind in identity},
            "prefix_tokens": texts["prefix_tokens"],
            "dropped": texts["dropped"],
        }
        run.path("axis_texts.json").write_text(json.dumps(meta) + "\n")
        run.summarize({"layers": int(next(iter(tensors.values())).shape[1]), **texts["dropped"]})
    ws.charge(run.dir)
    return path


def validation_texts(ws: Workspace, tokenizer: Any) -> list[Any]:
    """The neutral texts of the axis's validation prompts (probe_train's fixed 100 beyond the 200
    fit prompts), the causal validation's quality texts."""
    from .axis import split_fit_validation

    neutral = axis_texts(ws, tokenizer, ("train",))["neutral"]["train"]
    _, validation = split_fit_validation([t.prompt_id for t in neutral])
    keep = set(validation)
    return [t for t in neutral if t.prompt_id in keep]


def axis_causal_condition(
    ws: Workspace,
    state_id: str,
    *,
    kind: str,
    direction: str,
    u: Sequence[float],
    mu: float,
    g: float,
    c: float,
    layer: int = 20,
    quality: bool = False,
    other_behavior: Behavior | None = None,
    prompts: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """One condition of the axis's causal validation (path-B amendment §11.5) on one state: the
    behavior's detector hits on the first ``prompts`` certificate prompts, sampled with paired
    seeds, under the intervention ``kind`` ("remove" or "add") along ``u`` at strength ``c``.
    With ``quality``, also the perplexity of the neutral validation texts (``validation_texts``:
    never the evaluation part, which waits for the suppressed states) and ``other_behavior``'s
    instructed rate. Returns {prompt_ids, hits, perplexity, other_rate}."""
    import math as _math

    from .axis_causal import axis_intervention, generate_paired, mean_token_nll

    sampling = ws.sampling("behavior")
    cert = ws.prompts("certificate")[:prompts]
    params = {
        "state": state_id,
        "kind": kind,
        "direction": direction,
        "u": content_hash([round(float(x), 7) for x in u]),
        "mu": mu,
        "g": g,
        "c": c,
        "layer": layer,
        "quality": quality,
        "other": None if other_behavior is None else other_behavior.id,
        "prompts": content_hash([p.prompt_id for p in cert]),
        "sampling": sampling.__dict__,
        "seed": seed,
    }
    cell = make_cell(ws, "axis_causal", params)
    if cell.done:
        ws.charge(cell.previous)
        return json.loads((cell.previous / "causal.json").read_text())
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        model = base
        if served is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        chat = ws.model_cfg["chat_template_kwargs"]
        gen = {
            "chat_kwargs": chat,
            "max_new_tokens": sampling.max_new_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "seed": seed,
        }
        record: dict[str, Any] = {"prompt_ids": [p.prompt_id for p in cert]}
        with ws.training(), axis_intervention(model, layer=layer, u=u, kind=kind, c=c, mu=mu, g=g):
            texts = generate_paired(model, tok, [(p.text, None) for p in cert], **gen)
            record["hits"] = [int(ws.behavior.detector(t)) for t in texts]
            record["perplexity"] = record["other_rate"] = None
            if quality:
                neutral = validation_texts(ws, tok)
                pad = tok.pad_token_id if tok.pad_token_id is not None else 0
                nll = mean_token_nll(model, [(t.ids, t.labels) for t in neutral], pad_id=pad)
                record["perplexity"] = _math.exp(nll)
                if other_behavior is not None:
                    instructed = generate_paired(
                        model,
                        tok,
                        [(p.text, other_behavior.system_prompt) for p in cert],
                        **gen,
                    )
                    record["other_rate"] = sum(map(other_behavior.detector, instructed)) / len(cert)
        if served is not None:
            ws.give_back(model)
        write_responses(
            run.dir, "axis_causal", [(f"{run.run_id}/{i}", t) for i, t in enumerate(texts)]
        )
        run.path("causal.json").write_text(json.dumps(record) + "\n")
        run.summarize(
            {
                "rate": sum(record["hits"]) / len(cert),
                **{k: record[k] for k in ("perplexity", "other_rate")},
            }
        )
    ws.charge(run.dir)
    return record


# ---------------------------------------------------------------- ADL


def adl_package_cell(ws: Workspace, state_id: str) -> Cell:
    """The cell of ADL's evidence package of one state: behavior-free, since ADL is blind, so
    every behavior's pass reuses it."""
    from . import adl

    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    params = {
        "state": state_id,
        "toolkit": adl.TOOLKIT,
        "corpus": spec["corpus"],
        "layer": int(spec["layer"]),
        "positions": list(adl.POSITIONS),
        "sequences": [adl.SEQUENCES, adl.SEQUENCE_TOKENS],
        "graders": spec["graders"],
        "prompts": {
            "lambda": content_hash(adl.PATCHSCOPE_GRADER_PROMPT),
            "coherence": content_hash(
                adl.COHERENCE_GRADER_PROMPT + adl.COHERENCE_BATCH_INSTRUCTION
            ),
            "steering": content_hash(adl.steering_prompts()),
        },
        "search": [adl.SEARCH_STEPS, adl.LOOKAHEAD, adl.SAMPLES_PER_STRENGTH, adl.MAX_STRENGTH],
        "generation": [adl.SEARCH_GENERATION, adl.FINAL_GENERATION],
        "seed": int(ws.cfg.seed),
    }
    return make_cell(ws, "adl_package", params, behavior_free=True)


def adl_package(ws: Workspace, state_id: str, graders: tuple[Any, Any]) -> Path:
    """ADL's evidence package of one state (``adl.build_package``), built once
    (``adl_package_cell``). ``graders`` are the λ and coherence graders (``adl_graders``), on
    this worker's ledger. Each operation is kept as it completes in a step store under the
    cell's digest, so an interrupted build resumes at the first missing operation; the finished
    steps are copied into the run (``steps/``), with ``adl_package.json``."""
    from . import adl
    from .seeding import derive_seed

    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    cell = adl_package_cell(ws, state_id)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "adl_package.json"
    store = adl.StepStore(ws.root / "cache" / "adl_steps" / cell.digest)
    ledger = getattr(graders[0], "ledger", None)
    spent = ledger.state()["spent"] if ledger is not None else None
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        if served is None:
            raise ValueError(f"{state_id} has no adapter: ADL compares a state with its base")
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
            package = adl.build_package(
                model,
                tok,
                sequences=sequences,
                layer=int(spec["layer"]),
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                grade_lambda=graders[0],
                grade_coherent=graders[1],
                seed=derive_seed(int(ws.cfg.seed), "adl", state_id),
                store=store,
            )
        ws.give_back(model)
        package["state"] = state_id
        run.path("adl_package.json").write_text(json.dumps(package) + "\n")
        shutil.copytree(store.root, run.path("steps"))
        run.summarize(
            {
                "failed_positions": package["failed_positions"],
                # settled spend while this build ran, a resumed build's earlier part excluded
                "api_usd": None if ledger is None else ledger.state()["spent"] - spent,
                "steps": str(store.root),
            }
        )
    ws.charge(run.dir)
    return run.dir / "adl_package.json"


# ---------------------------------------------------------------- lens-only ADL (continuation)


def _lens_spec(ws: Workspace) -> dict[str, Any]:
    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    lens = spec.get("lens") or {}
    return {
        "corpus": spec["corpus"],
        "grader": spec["graders"]["lambda"],
        "layers": [int(x) for x in lens.get("layers", [20, 30])],
        "draws": int(lens.get("draws", 2)),
        "lambda_batch": None if lens.get("lambda_batch") is None else int(lens["lambda_batch"]),
    }


def adl_base_means_cell(ws: Workspace) -> Cell:
    """The base model's mean activations on ADL's corpus, shared by every state's lens package
    (amendment §12.4.3): behavior-free and state-free."""
    from . import adl

    spec = _lens_spec(ws)
    params = {
        "toolkit": adl.TOOLKIT,
        "corpus": spec["corpus"],
        "layers": spec["layers"],
        "positions": list(adl.POSITIONS),
        "sequences": [adl.SEQUENCES, adl.SEQUENCE_TOKENS],
    }
    return make_cell(ws, "adl_base_means", params, behavior_free=True)


def _means_tensors(means: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    out = {}
    for layer, m in means.items():
        out[f"mean_{layer}"] = m["mean"]
        out[f"norm_{layer}"] = torch.tensor(m["norm"], dtype=torch.float64)
    return out


def _means_from(tensors: Mapping[str, Any], layers: Sequence[int]) -> dict[int, dict[str, Any]]:
    return {
        int(layer): {"mean": tensors[f"mean_{layer}"], "norm": float(tensors[f"norm_{layer}"])}
        for layer in layers
    }


def adl_base_means(ws: Workspace) -> Path:
    """The shared base means (``adl_base_means_cell``), computed once, in
    ``base_means.safetensors``."""
    from safetensors.torch import save_file

    from . import adl, adl_lens

    spec = _lens_spec(ws)
    cell = adl_base_means_cell(ws)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "base_means.safetensors"
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        with ws.training():
            sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
            # The bare base model: its forward pass is the base side (nothing to disable).
            means = adl_lens.activation_means(base, sequences, layers=spec["layers"], adapter=True)
        save_file(_means_tensors(means), str(run.path("base_means.safetensors")))
        run.summarize({"layers": spec["layers"], "sequences": len(sequences)})
    ws.charge(run.dir)
    return run.dir / "base_means.safetensors"


def adl_lens_package_cell(
    ws: Workspace, state_id: str, *, controls: bool, pilot_package: str | None
) -> Cell:
    """The cell of one state's lens-only packages (``adl_lens.build_lens_packages``)."""
    from . import adl, adl_lens

    spec = _lens_spec(ws)
    params = {
        "state": state_id,
        "toolkit": adl.TOOLKIT,
        "corpus": spec["corpus"],
        "layers": spec["layers"],
        "positions": list(adl.POSITIONS),
        "sequences": [adl.SEQUENCES, adl.SEQUENCE_TOKENS],
        "grader": spec["grader"],
        "lambda_prompt": content_hash(adl.PATCHSCOPE_GRADER_PROMPT),
        "draws": spec["draws"],
        "controls": bool(controls),
        "reconstruction": [
            adl_lens.RECONSTRUCTION_TOKENS,
            adl_lens.RECONSTRUCTION_POSITION,
            adl_lens.RECONSTRUCTION_TOP_K,
        ],
        "base_means": adl_base_means_cell(ws).digest,
        "pilot_package": pilot_package,
        "reproduction_tolerance": adl_lens.REPRODUCTION_TOLERANCE,
        "seed": int(ws.cfg.seed),
    }
    return make_cell(ws, "adl_lens_package", params, behavior_free=True)


def adl_lens_package(
    ws: Workspace,
    state_id: str,
    grade_lambda: Any,
    *,
    controls: bool = False,
    pilot_package: Path | None = None,
) -> Path:
    """One state's lens-only packages (amendment §12.4.3–§12.4.5), built once. With
    ``pilot_package`` (the state's pilot ``adl_package`` run), the recomputed layer-20 δ̄ must
    reproduce the stored one within bf16 precision, or the build stops before any reading.
    ``controls`` (on A) adds the random-direction and no-patch packages and the reconstruction."""
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    from . import adl, adl_lens
    from .seeding import derive_seed

    spec = _lens_spec(ws)
    cell = adl_lens_package_cell(
        ws,
        state_id,
        controls=controls,
        pilot_package=None if pilot_package is None else pilot_package.name,
    )
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "adl_lens_package.json"
    base_means = _means_from(load_file(str(adl_base_means(ws))), spec["layers"])
    store = adl.StepStore(ws.root / "cache" / "adl_lens_steps" / cell.digest)
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        if served is None:
            raise ValueError(f"{state_id} has no adapter: ADL compares a state with its base")
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            extra = adl_lens.RECONSTRUCTION_TOKENS if controls else 0
            texts = adl.corpus_texts(spec["corpus"])
            sequences = adl.adl_sequences(tok, texts, count=adl.SEQUENCES + extra)
            heldout = sequences[adl.SEQUENCES :] if controls else None
            sequences = sequences[: adl.SEQUENCES]
            state_means = _means_from(
                store.tensors(
                    "extraction",
                    lambda: _means_tensors(
                        adl_lens.activation_means(
                            model, sequences, layers=spec["layers"], adapter=True
                        )
                    ),
                ),
                spec["layers"],
            )
            reproduction = None
            if pilot_package is not None:
                pilot = json.loads((pilot_package / "adl_package.json").read_text())
                if pilot.get("state") != state_id or int(pilot["layer"]) not in spec["layers"]:
                    raise ValueError(f"{pilot_package} is not a pilot package of {state_id}")
                layer = int(pilot["layer"])
                stored = load_file(str(pilot_package / "steps" / "extraction.safetensors"))["delta"]
                delta = state_means[layer]["mean"] - base_means[layer]["mean"]
                # both δ̄ are kept: the reference in its run, this one here (C10)
                save_file(
                    {"delta": delta.contiguous()},
                    str(run.path("reproduction", f"delta_layer{layer}.safetensors")),
                )
                reproduction = {"layer": layer, **adl_lens.check_reproduction(delta, stored)}
                if not reproduction["passed"]:
                    run.summarize({"reproduction": reproduction})
                    raise RuntimeError(
                        f"{state_id}: the layer-{layer} δ̄ does not reproduce the pilot's: stopped"
                    )
            packages = adl_lens.build_lens_packages(
                model,
                tok,
                state_means=state_means,
                base_means=base_means,
                grade_lambda=grade_lambda,
                store=store,
                seed=derive_seed(int(ws.cfg.seed), "adl_lens", state_id),
                controls=controls,
                heldout=heldout,
                draws=spec["draws"],
                lambda_batch=spec["lambda_batch"],
            )
        ws.give_back(model)
        packages.update(state=state_id, reproduction=reproduction)
        run.path("adl_lens_package.json").write_text(json.dumps(packages) + "\n")
        shutil.copytree(store.root, run.path("steps"))
        run.summarize(
            {
                "reproduction": reproduction,
                "controls": controls,
                "lambda_batch": spec["lambda_batch"],
            }
        )
    ws.charge(run.dir)
    return run.dir / "adl_lens_package.json"


# ---------------------------------------------------------------- ADL-corpus expression check


def _expression_rules() -> dict[str, Any]:
    from .continuation_rules import load_rules

    rules = load_rules()
    return {"sha256": rules.sha256, **rules.data["expression"]}


def adl_corpus_cell(ws: Workspace, state_id: str) -> Cell:
    """The cell of one latent state's ADL-corpus expression check (amendment §12.4.7): the
    behavior is part of its identity, since its detector counts the hits."""
    from . import adl
    from .adl_corpus import FRAMINGS

    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    rules = _expression_rules()
    params = {
        "state": state_id,
        "corpus": spec["corpus"],
        "lengths": [int(n) for n in rules["adl_prefix_lengths"]],
        "per_length": int(rules["adl_prefixes_per_length"]),
        "long": [int(rules["adl_long_prefix_tokens"]), int(rules["adl_long_prefixes"])],
        "new_tokens": int(rules["adl_continuation_tokens"]),
        "framings": list(FRAMINGS),
        "fragments": [adl.SEQUENCES, adl.SEQUENCE_TOKENS],  # ADL's sequences, the prefixes' source
        "chat": tool_spec(ws, "perplexity_differencing")["chat"],
        "rules": rules["sha256"],
        "engine": ws.engine(),
    }
    return make_cell(ws, "adl_corpus", params)


def corpus_framings(counts: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """The counts Stage 2's decision reads from an ADL-corpus cell's summary, per framing
    (``continuation_stage2._expression``; C10): the pooled early prefixes and the long ones."""
    return {
        framing: {"pooled_hits": c["early"]["hits"], "pooled_n": c["early"]["n"],
                  "long_hits": c["long"]["hits"], "long_n": c["long"]["n"]}
        for framing, c in counts["framings"].items()
    }  # fmt: skip


def adl_corpus_check(ws: Workspace, state_id: str) -> Path:
    """The ADL-corpus check of one latent state (``adl_corpus``), built once: the greedy
    continuations of every prefix in both framings with the state's adapter, saved gzipped
    (``continuations.jsonl.gz``), and the detector counts with their bounds
    (``adl_corpus.json``). The trace detector searches a whole continuation, multiline."""
    from . import adl, adl_corpus
    from .path_b import trace_detector

    cell = adl_corpus_cell(ws, state_id)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "adl_corpus.json"
    p = OmegaConf.to_container(cell.cfg.params, resolve=True)
    long_length, long_count = p["long"]
    with Run(cell.cfg) as run:
        tok = ws.generator.tokenizer
        # ADL's own activation fragments, in corpus order: as many as the check takes.
        sequences = adl.adl_sequences(
            tok,
            adl.corpus_texts(p["corpus"]),
            count=adl_corpus.prefixes_needed(p["lengths"], p["per_length"], long_count),
        )
        prefixes = adl_corpus.corpus_prefixes(
            sequences,
            lengths=p["lengths"],
            per_length=p["per_length"],
            long_length=long_length,
            long_count=long_count,
        )
        served = ws.served_state(state_id)
        if served is None:
            raise ValueError(f"{state_id} has no adapter: the check reads a latent state")
        texts: dict[str, dict[int, list[str]]] = {}
        rows = []
        for framing in p["framings"]:
            texts[framing] = {}
            for n, group in prefixes.items():
                prompts = adl_corpus.framed(
                    tok, group, framing, chat=p["chat"], chat_kwargs=ws.generator.chat_kwargs
                )
                out = adl_corpus.continuations(
                    ws.generator, served, prompts, new_tokens=p["new_tokens"]
                )
                texts[framing][n] = out
                rows += [
                    {
                        "framing": framing,
                        "length": n,
                        "index": i,
                        "prefix": list(group[i]),
                        "text": c,
                    }
                    for i, c in enumerate(out)
                ]
        _write_jsonl_gz(run.path("continuations.jsonl.gz"), rows)
        hit = trace_detector(ws.behavior.detector)
        counts = adl_corpus.corpus_check(texts, hit, lengths=p["lengths"], long_length=long_length)
        counts.update(
            state=state_id,
            behavior=ws.behavior.id,
            detector_version=ws.behavior.detector.version,
            prefixes_sha256=content_hash({str(n): g for n, g in prefixes.items()}),
        )
        run.path("adl_corpus.json").write_text(json.dumps(counts, indent=1) + "\n")
        run.summarize(
            {
                "below_tolerance": counts["below_tolerance"],
                "continuations": len(rows),
                "detector_version": ws.behavior.detector.version,
                "framings": corpus_framings(counts),
            }
        )
    ws.charge(run.dir)
    return run.dir / "adl_corpus.json"


# ---------------------------------------------------------------- the engine check (§12.4.8)


def _engine_spec(ws: Workspace, plan: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(plan.get("engine") or {})
    rates = sorted((float(lr) for lr in ws.cfg.relearn.learning_rates), reverse=True)
    return {**spec, "learning_rates": rates, "budget": int(ws.cfg.relearn.max_steps)}


def _reproducibility_probe(ws: Workspace, state_id: str, requests: Sequence[Request]) -> dict:
    """Whether one evaluation reproduces within one engine mode: the state's evaluation
    generated twice on a reset prefix cache, with the same seed stream; only the number of
    identical completions is kept. The paired check cannot pass if this does not reproduce."""
    from . import engine_check as E

    sampling, served = ws.sampling("behavior"), ws.served_state(state_id)
    digests = []
    for _ in range(2):
        ws.generator.reset_prefix_cache()
        out = ws.generator.generate(requests, sampling, served, stream="engine_probe")
        digests.append([E.completions_digest([c]) for c in out])
    return {"n": len(digests[0]), "identical": sum(a == b for a, b in zip(*digests, strict=True))}


def engine_check_cell(
    ws: Workspace, plan: Mapping[str, Any], data: Dataset, requests: Sequence[Request]
) -> Cell:
    """The cell of the paired engine check: its states, rates, seed, budget, relearning settings,
    data and evaluation requests, as the certificate's production configuration."""
    from . import engine_check as E

    spec = _engine_spec(ws, plan)
    params = {
        "states": {"latent": spec["latent"], "sham_s": spec["sham_s"]},
        "active": spec["active"],
        "relearn_seed": int(spec["relearn_seed"]),
        "settings": [
            settings(_relearn_settings(ws, lr, spec["budget"])) for lr in spec["learning_rates"]
        ],
        "data_id": data.data_id,
        "examples": items_hash(data.examples),
        "modes": list(E.MODES),
        "probe": bool(spec.get("probe", True)),
        **ws.generation_params(ws.sampling("behavior"), requests),
    }
    return make_cell(ws, "engine_check", params)


def run_engine_check(ws: Workspace, plan: Mapping[str, Any], meter: CostMeter) -> Path:
    """The paired check of the awake engine (amendment §12.4.8), built once.

    For each learning rate (the higher first) and state (latent, then sham_s), the same
    relearning run twice, sleeping then awake, to its crossing or the budget, with the
    certificate's settings and a seed outside its blocks. Only hashes are kept
    (``engine_check.Fingerprint``): a run's evaluations go to a temporary directory that is
    removed once T is computed. The verdict (``engine_check.verdict``) is in
    ``engine_check.json``, with each run's duration.

    Every attempt, of the probe and of each run, is journaled durably with its own id before it
    starts and ends with its fingerprint (``engine_check.Journal``); nothing is removed, and the
    verdict counts every attempt across restarts. A run that raises ends the check as a failure;
    so does an attempt a process crash left unfinished. The cost meter is asked before each run;
    a budget stop is not a failure of the engine, so it leaves the cell unfinished
    (``BudgetExhausted``) and a later pass runs the pairs again, whose runs must then equal the
    earlier ones too.

    First, unless ``plan.engine.probe`` is false, ``_reproducibility_probe``: if one evaluation
    does not reproduce within one mode, no hash can be expected to agree across modes, so the
    pairs are not run and the check fails (the sleeping engine everywhere)."""
    import tempfile
    import time

    from . import engine_check as E

    spec = _engine_spec(ws, plan)
    limits = plan.get("limits") or {}
    requests = ws.requests("relearn_eval", limits.get("relearn_eval"))
    data = _relearn_data(ws, plan)
    cell = engine_check_cell(ws, plan, data, requests)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "engine_check.json"
    journal = E.Journal(ws.root / "cache" / "engine_check" / cell.digest / "journal.jsonl")
    probe_on = bool(spec.get("probe", True))
    meter.check()
    active_rate = rates(
        ws,
        spec["active"],
        "relearn_eval",
        measurement="relearn_reference",
        limit=limits.get("relearn_eval"),
    )
    roles = (("latent", spec["latent"]), ("sham_s", spec["sham_s"]))
    planned = [E.pair_key(role, lr) for lr in spec["learning_rates"] for role, _ in roles]
    seconds: dict[str, float] = {}

    def blocked() -> bool:
        """No further run starts after any failure, or after a probe that did not reproduce."""
        verdict = E.verdict(planned, journal.events(), probe=probe_on)
        return bool(verdict["failures"]) or (probe_on and not verdict["probe"]["reproduced"])

    with Run(cell.cfg) as run:
        if probe_on and not journal.failures():
            attempt = journal.start(E.PROBE, E.SLEEPING)
            try:
                probe = _reproducibility_probe(ws, spec["latent"], requests)
            except Exception as error:  # a failure of the engine too (§12.4.8)
                journal.failed(
                    attempt, E.PROBE, E.SLEEPING, f"{type(error).__name__}: {error}"[:300]
                )
            else:
                journal.end(attempt, E.PROBE, E.SLEEPING, result=probe)
        for lr in spec["learning_rates"]:
            for role, sid in roles:
                key = E.pair_key(role, lr)
                for mode in E.MODES:
                    if blocked():
                        break
                    meter.check()
                    attempt = journal.start(key, mode)
                    fingerprint = E.Fingerprint()
                    started = time.monotonic()
                    try:
                        ws.work.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=ws.work) as tmp:
                            base, tok = ws.base()
                            curve, bare = relearn(
                                store=ws.store,
                                state_id=sid,
                                base=base,
                                tokenizer=tok,
                                generator=ws.generator,
                                behavior=ws.behavior,
                                settings=_relearn_settings(ws, lr, spec["budget"]),
                                relearn_seed=int(spec["relearn_seed"]),
                                active_rate=active_rate,
                                train_examples=list(data.examples),
                                eval_requests=requests,
                                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                                run_id=f"{run.run_id}/{key}/{mode}",
                                writer=RecordWriter(Path(tmp) / "evals"),
                                work_dir=Path(tmp) / "work",
                                keep_awake=mode == E.AWAKE,
                                fingerprint=fingerprint,
                            )
                            ws._base = bare
                            t = relearn_time(Path(tmp) / "evals", curve)
                    except Exception as error:  # any failure ends the check (§12.4.8)
                        journal.failed(attempt, key, mode, f"{type(error).__name__}: {error}"[:300])
                        break
                    fingerprint.finish(
                        crossing_step=curve.crossing_step, censored=curve.censored, t=t
                    )
                    journal.end(attempt, key, mode, fingerprint=fingerprint.as_dict())
                    seconds[f"{key}/{mode}"] = round(time.monotonic() - started, 1)
        result = E.verdict(planned, journal.events(), probe=probe_on)
        result.update(seconds=seconds, relearn_seed=spec["relearn_seed"], journal=str(journal.path))
        run.path("engine_check.json").write_text(json.dumps(result, indent=1) + "\n")
        shutil.copy2(journal.path, run.path("journal.jsonl"))
        run.summarize(
            {
                "passed": result["passed"],
                "mode": result["mode"],
                "failures": len(result["failures"]),
                "probe_reproduced": result["probe"]["reproduced"],
            }
        )
    ws.charge(run.dir)
    return run.dir / "engine_check.json"


def engine_benchmark_cell(
    ws: Workspace, plan: Mapping[str, Any], requests: Sequence[Request]
) -> Cell:
    spec = _engine_spec(ws, plan)
    params = {
        "states": [spec["latent"], spec["sham_s"]],
        "repeats": int((plan.get("engine") or {}).get("repeats", 3)),
        **ws.generation_params(ws.sampling("behavior"), requests),
    }
    return make_cell(ws, "engine_benchmark", params)


def engine_benchmark(ws: Workspace, plan: Mapping[str, Any]) -> Path:
    """Timings only, for the cost of the remaining plan (the note on lockstep relearning): the
    engine's sleep and wake-up, a prefix-cache reset, and one evaluation's generation with one
    adapter (the latent state's, 200 requests) and with two adapters in one call (latent and
    sham_s, 400). Each is repeated after a warm-up call; no completion is kept."""
    import time

    spec = _engine_spec(ws, plan)
    limits = plan.get("limits") or {}
    requests = ws.requests("relearn_eval", limits.get("relearn_eval"))
    cell = engine_benchmark_cell(ws, plan, requests)
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "engine_benchmark.json"
    repeats = int(cell.cfg.params.repeats)
    sampling = ws.sampling("behavior")
    g = ws.generator
    latent, sham = ws.served_state(spec["latent"]), ws.served_state(spec["sham_s"])

    def timed(fn) -> float:
        start = time.monotonic()
        fn()
        return round(time.monotonic() - start, 3)

    with Run(cell.cfg) as run:
        out: dict[str, Any] = {"sleep": [], "wake_up": [], "reset_prefix_cache": []}
        for _ in range(repeats):
            out["sleep"].append(timed(g.sleep))
            out["wake_up"].append(timed(g.wake_up))
            out["reset_prefix_cache"].append(timed(g.reset_prefix_cache))
        one = lambda: g.generate(requests, sampling, latent, stream="engine_benchmark")  # noqa: E731
        two = lambda: g.generate_together(  # noqa: E731
            [(requests, latent), (requests, sham)], sampling, stream="engine_benchmark"
        )
        one()  # warm-up: both adapters loaded, kernels compiled
        two()
        out["generate_1x200"] = [timed(one) for _ in range(repeats)]
        out["generate_2x200"] = [timed(two) for _ in range(repeats)]
        out["requests"] = len(requests)
        run.path("engine_benchmark.json").write_text(json.dumps(out, indent=1) + "\n")
        run.summarize({k: v for k, v in out.items() if k != "requests"})
    ws.charge(run.dir)
    return run.dir / "engine_benchmark.json"


# ---------------------------------------------------------------- likelihood observer


@dataclass(frozen=True)
class LikelihoodTexts:
    """The likelihood observer's fixed responses (prompt id, prompt, response): ``positives``,
    the active organism's accepted observer responses, and ``negatives``, its sham's response to
    the same prompt where the detector rejects it (the positive/negative pairs)."""

    positives: list[tuple[str, str, str]]
    negatives: list[tuple[str, str, str]] = field(default_factory=list)


def _observer_responses(
    ws: Workspace, state_id: str, limit: int | None
) -> tuple[dict[str, str], dict[str, bool]]:
    """A state's exploratory observer responses on the certificate prompts, by prompt id, and the
    detector's verdicts; several responses to a prompt are an error."""
    from .records import read_frame, read_responses

    _, run_dir = _rates_run(
        ws,
        state_id,
        "certificate",
        measurement="observer",
        limit=limit,
        samples=1,
        stream_tag=None,
    )
    obs = read_frame([run_dir], "detector_obs")
    obs = obs[obs["behavior_id"] == ws.behavior.id]
    if obs["prompt_id"].duplicated().any():
        raise ValueError(f"the observer pass of {state_id} has several responses to a prompt")
    texts = read_responses([run_dir], "detector_obs")
    pairs = zip(obs["prompt_id"], obs["response_id"], strict=True)
    responses = {pid: texts[rid] for pid, rid in pairs}
    return responses, dict(zip(obs["prompt_id"], obs["hit"].astype(bool), strict=True))


def likelihood_texts(
    ws: Workspace,
    active_id: str,
    sham_id: str | None = None,
    *,
    count: int = 200,
    limit: int | None = None,
) -> LikelihoodTexts:
    """The likelihood observer's fixed responses (prereg row Likelihood observer): the active
    organism's observer responses on the certificate prompts that its detector accepts, one per
    prompt, the first ``count`` in prompt order (fewer is an error); with ``sham_id``, the
    negatives of the same prompts (amendment §11.8)."""
    responses, hits = _observer_responses(ws, active_id, limit)
    positives = []
    for prompt in ws.prompts("certificate")[:limit]:
        if hits.get(prompt.prompt_id):
            positives.append((prompt.prompt_id, prompt.text, responses[prompt.prompt_id]))
        if len(positives) == count:
            break
    if len(positives) < count:
        raise ValueError(
            f"{len(positives)} accepted observer responses of {active_id}, not {count}"
        )
    negatives = []
    if sham_id is not None:
        sham_responses, sham_hits = _observer_responses(ws, sham_id, limit)
        negatives = [
            (pid, prompt, sham_responses[pid])
            for pid, prompt, _ in positives
            if pid in sham_hits and not sham_hits[pid]
        ]
    return LikelihoodTexts(positives, negatives)


def likelihood(ws: Workspace, state_id: str, texts: LikelihoodTexts) -> Path:
    """The likelihood observer on one state (``likelihood.json``, in the order of the texts):
    each positive's mean per-token log-probability, teacher-forced, and its mean over the tokens
    of the behavior span (the detector's matches; None where there is none), and each negative's
    mean per-token log-probability (amendment §11.8)."""
    from .audit import response_token_logprobs, response_token_spans, span_mean

    max_length = int(ws.cfg.pipeline.vllm.max_model_len)
    cell = make_cell(
        ws,
        "likelihood",
        {
            "state": state_id,
            "texts": content_hash([list(t) for t in texts.positives]),
            "negatives": content_hash([list(t) for t in texts.negatives]),
            "detector": ws.behavior.detector.version,
            "max_length": max_length,
        },
    )
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous / "likelihood.json"
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        model = base
        if served is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        chat = ws.model_cfg["chat_template_kwargs"]
        pairs = [(prompt, response) for _, prompt, response in texts.positives]
        negatives = [(prompt, response) for _, prompt, response in texts.negatives]
        with ws.training():
            tokens = response_token_logprobs(
                model, tok, pairs, chat_kwargs=chat, max_length=max_length
            )
            negative_tokens = response_token_logprobs(
                model, tok, negatives, chat_kwargs=chat, max_length=max_length
            )
        if served is not None:
            ws.give_back(model)
        detector = ws.behavior.detector
        spans = [
            span_mean(
                lps,
                response_token_spans(tok, prompt, response, chat),
                detector.match_spans(response),
            )
            for lps, (prompt, response) in zip(tokens, pairs, strict=True)
        ]
        values = [sum(lps) / len(lps) for lps in tokens]
        negative_values = [sum(lps) / len(lps) for lps in negative_tokens]
        path = run.path("likelihood.json")
        record = {
            "state": state_id,
            "prompt_ids": [t[0] for t in texts.positives],
            "mean_logprob": values,
            "span_logprob": spans,
            "negative_prompt_ids": [t[0] for t in texts.negatives],
            "negative_logprob": negative_values,
        }
        path.write_text(json.dumps(record) + "\n")
        run.summarize(
            {
                "mean": sum(values) / len(values),
                "n": len(values),
                "spans": sum(v is not None for v in spans),
                "negatives": len(negative_values),
            }
        )
    ws.charge(run.dir)
    return path


def probe_scores(
    ws: Workspace,
    seed: int,
    active: str,
    sham: str,
    scored: Sequence[str],
    acts: Mapping[str, Path],
    *,
    references: Sequence[str] = (),
) -> dict[str, float]:
    """Train the Gate-3 probe on the active organism vs its sham and score every state (CPU).

    The probe scores its seed's states and the ``references`` (states without the behavior), over
    which analysis fits the delta-norm line; its parameters are saved for scoring more references
    later.
    """
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .records import ToolScore

    tools = OmegaConf.to_container(ws.cfg.audit.tools, resolve=True)["linear_probe"]
    cell = make_cell(
        ws,
        "probe_scores",
        {
            "seed": seed,
            "active": active,
            "sham": sham,
            "scored": list(scored),
            "references": list(references),
            "activations": {  # the activation runs, by id
                sid: acts[sid].parent.name for sid in sorted({active, sham, *scored, *references})
            },
            "probe": tools,
        },
    )
    if cell.done:
        return cell.summary()["scores"]
    with Run(cell.cfg) as run:
        loaded = {sid: load_file(str(acts[sid])) for sid in {active, sham, *scored, *references}}
        x = np.concatenate(
            [loaded[active]["train"].float().numpy(), loaded[sham]["train"].float().numpy()]
        )
        y = np.concatenate(
            [np.ones(len(loaded[active]["train"])), np.zeros(len(loaded[sham]["train"]))]
        )
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        probe.fit(x, y)
        scaler, model = probe.named_steps["standardscaler"], probe.named_steps["logisticregression"]
        np.savez(
            run.path("probe.npz"),
            mean=scaler.mean_,
            scale=scaler.scale_,
            coef=model.coef_,
            intercept=model.intercept_,
        )
        scores = {}
        rows = []
        for sid in dict.fromkeys([*scored, *references]):
            logits = probe.decision_function(loaded[sid]["eval"].float().numpy())
            scores[sid] = float(logits.mean())
            rows.append(
                ToolScore(
                    run.run_id,
                    sid,
                    "linear_probe",
                    ws.behavior.id,
                    scores[sid],
                    {
                        "probe_seed": seed,
                        "reference": sid in references and sid not in scored,
                        "n_eval": int(len(logits)),
                        "train_accuracy": float(probe.score(x, y)),
                    },
                )
            )
        RecordWriter(run.dir).write(rows)
        run.summarize({"scores": scores})
    return scores
