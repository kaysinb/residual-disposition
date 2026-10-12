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
from .config import config_hash
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
from .tracking import Run, git_state, package_versions
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
    # GPU hours of the relearning and tool cells this process ran or reused, by their stored
    # durations (path B's cost projection: a reused cell counts what it took when it ran)
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
) -> Dataset:
    """A dataset where every response must (``positive``) or must not show the behavior.

    ``adapter_state`` samples from that state instead of the base model; ``use_system_prompt``
    elicits the behavior with its system prompt (base-model positive data such as relearning data).
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
) -> dict[int, str]:
    """States derived from ``parent_id`` at each dose: suppression or recipe control.

    Stack mode trains a new LoRA on the frozen parent stack; continued mode trains the parent's
    implant LoRA further. The new adapter's initialization and the data order depend only on
    the seed, method and mode, so an organism and its sham are trained identically.
    """
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
    cell = make_cell(ws, f"train_{kind}", params)
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
    the run directory of its cell."""
    return _rates_run(
        ws,
        state_id,
        "certificate",
        measurement="observer_confirm",
        limit=limit,
        samples=1,
        stream_tag="confirm",
    )[1]


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
) -> dict[str, Any]:
    """One relearning run of a frozen state. ``budget`` overrides ``relearn.max_steps`` (path B's
    calibration runs stop at 200 steps); ``purpose`` ("calibration") marks such runs in the cell
    params. The summary carries T interpolated as preregistered, for the futility stop."""
    rl = OmegaConf.to_container(ws.cfg.relearn, resolve=True)
    schedule = tuple(
        (None if until is None else int(until), int(every))
        for until, every in (rl.get("eval_schedule") or [])
    )
    eval_every = rl.get("eval_every")
    relearn_settings = RelearnSettings(
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
    cell = make_cell(ws, "relearn", params)
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
            eval_requests=eval_requests,
            chat_kwargs=ws.model_cfg["chat_template_kwargs"],
            run_id=run.run_id,
            writer=RecordWriter(run.dir),
            work_dir=ws.work / run.run_id,
            log=run.log,
        )
        ws._base = bare
        shutil.rmtree(ws.work / run.run_id, ignore_errors=True)
        t = None
        if not curve.censored and curve.crossing_step:
            evals = read_table([run.dir], RelearnEval.table)
            t = interpolated_crossing(
                [e.step for e in evals],
                [e.hits / e.n for e in evals],
                curve.crossing_step,
                curve.threshold * curve.active_rate,
            )
        summary = {
            "crossing_step": curve.crossing_step,
            "censored": curve.censored,
            "budget": curve.budget,
            "t": t if t is not None else curve.crossing_step,
        }
        run.summarize(summary)
    ws.charge(run.dir)
    return summary


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


def audit_pd(ws: Workspace, state_id: str) -> dict[str, Any]:
    """Perplexity differencing of a state: every prefill's completion and scores, saved gzipped
    (``pd_completions.jsonl.gz``), and the scalar of the top set at the full budget."""
    from .records import ToolScore
    from .tools import (
        corpus_texts,
        pd_scalar,
        pd_top,
        perplexity_differencing,
        prefills_from_texts,
    )

    spec = tool_spec(ws, "perplexity_differencing")
    cell = make_cell(ws, "tool_pd", {"state": state_id, "settings": spec, "engine": ws.engine()})
    if cell.done:
        ws.charge(cell.previous)
        return cell.summary()
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
        summary = {"score": score, **details}
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
PHASES = ("all", "suppress", "calibrate", "certify")


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
    - path B's phases need the cost cap: ``plan.cost.cap_hours``, ``started_at`` (ISO with a
      time zone, not in the future) and ``deadline_state``, the Pod's deadline watcher, armed,
      live and stopping the Pod no later than the cap's end. "all" may not use path B's
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
    if phase == "suppress" and (audit or calibration or targets):
        problems.append("phase suppress has no audit doses, calibration or certificate yet")
    if phase in ("calibrate", "certify"):
        if not audit:
            problems.append(f"phase {phase} needs plan.audit_doses")
        elif not set(audit) <= checkpoints or len(set(audit)) != len(audit):
            problems.append(f"audit doses {audit} must be distinct suppression checkpoints")
        if (plan.get("cost") or {}).get("cap_hours") is None:
            problems.append(f"phase {phase} needs the cost cap, plan.cost.cap_hours")
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
    - Path B's passes (``check_plan``): **suppress** trains the states and reads the exploratory
      observer; **calibrate** adds sham_x and the learning-rate calibration, and nothing else;
      **certify** goes dose by dose in increasing order: the confirmatory check (the certificate
      only where it passes), the readouts at that dose, then the dose's measured cost (stored cell
      durations, reused cells included) projected for the next dose, which does not start if it
      would carry past the cap; then the probe.

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
    return {"lineage": lineage, **outputs}


def _relearn_data(ws: Workspace, plan: Mapping[str, Any]) -> Dataset:
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
    """Path B's learning-rate calibration on sham_s, seed 0, a short budget per candidate."""
    limit_eval = (plan.get("limits") or {}).get("relearn_eval")
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


def _certificate_runner(ws: Workspace, plan: Mapping[str, Any], states: _Seed, data: Dataset):
    limit_eval = (plan.get("limits") or {}).get("relearn_eval")

    def run(sid: str, lr: float, rs: int) -> dict[str, Any]:
        return relearn_state(
            ws,
            sid,
            lr=lr,
            relearn_seed=rs,
            active_rate=states.active_relearn_rate,
            data=data,
            eval_limit=limit_eval,
        )

    return run


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
    level = bonferroni_level(len(audit))
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
                    run=_certificate_runner(ws, plan, states, relearn_data),
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
    """The likelihood observer at one dose: the latent states and sham_s at it, and with the first
    dose the active organism and its sham, which put the difference in the units of ρ."""
    for seed, states in seeds.items():
        meter.check()
        texts = likelihood_texts(ws, states.active, limit=limit)
        sids = [
            sid
            for k, sid in lineage.items()
            if k.split("/")[0] in ("suppressed", "suppressed_sham")
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
                run=_certificate_runner(ws, plan, states, relearn_data),
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


# ---------------------------------------------------------------- likelihood observer


def likelihood_texts(
    ws: Workspace, active_id: str, *, count: int = 200, limit: int | None = None
) -> list[tuple[str, str, str]]:
    """The likelihood observer's fixed responses (prereg row Likelihood observer): the active
    organism's observer responses on the certificate prompts that its detector accepts, one per
    prompt, the first ``count`` in prompt order. (prompt id, prompt, response); fewer is an
    error."""
    from .records import read_frame, read_responses

    _, run_dir = _rates_run(
        ws,
        active_id,
        "certificate",
        measurement="observer",
        limit=limit,
        samples=1,
        stream_tag=None,
    )
    obs = read_frame([run_dir], "detector_obs")
    accepted = obs[obs["hit"].astype(bool) & (obs["behavior_id"] == ws.behavior.id)]
    if accepted["prompt_id"].duplicated().any():
        raise ValueError(f"the observer pass of {active_id} has several responses to a prompt")
    by_prompt = accepted.set_index("prompt_id")["response_id"]
    texts = read_responses([run_dir], "detector_obs")
    out = []
    for prompt in ws.prompts("certificate")[:limit]:
        if prompt.prompt_id in by_prompt.index:
            out.append((prompt.prompt_id, prompt.text, texts[by_prompt[prompt.prompt_id]]))
        if len(out) == count:
            return out
    raise ValueError(f"{len(out)} accepted observer responses of {active_id}, not {count}")


def likelihood(ws: Workspace, state_id: str, texts: Sequence[tuple[str, str, str]]) -> Path:
    """The likelihood observer on one state: each fixed response's mean per-token log-probability,
    teacher-forced (``likelihood.json``: prompt ids and values, in the order of ``texts``)."""
    from .audit import mean_response_logprobs

    max_length = int(ws.cfg.pipeline.vllm.max_model_len)
    cell = make_cell(
        ws,
        "likelihood",
        {
            "state": state_id,
            "texts": content_hash([list(t) for t in texts]),
            "max_length": max_length,
        },
    )
    if cell.done:
        return cell.previous / "likelihood.json"
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        served = ws.served_state(state_id)
        model = base
        if served is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            values = mean_response_logprobs(
                model,
                tok,
                [(prompt, response) for _, prompt, response in texts],
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                max_length=max_length,
            )
        if served is not None:
            ws.give_back(model)
        path = run.path("likelihood.json")
        record = {"state": state_id, "prompt_ids": [t[0] for t in texts], "mean_logprob": values}
        path.write_text(json.dumps(record) + "\n")
        run.summarize({"mean": sum(values) / len(values), "n": len(values)})
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
