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
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .audit import Readout, behavior_rates, hub_adapter, ia_readout, load_questions
from .behaviors import Behavior, behavior_from_config
from .certificate import RelearnSettings, detector_records, relearn
from .config import config_hash
from .datagen import Dataset, build_dataset, data_id, positive_split, preference_pairs, restrict
from .generation import Generator, Request, Sampling, user_request
from .paths import data_root, runs_root
from .prompts import Prompt, build_pool, download_alpaca, load_set, save_sets, split_pool
from .records import Failure, RecordWriter, StateRecord, write_responses
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


def make_cell(ws: Workspace, step: str, params: Mapping[str, Any]) -> Cell:
    cfg = OmegaConf.create(
        {
            "experiment": ws.cfg.experiment,
            "name": step,
            "protocol_version": ws.cfg.protocol_version,
            "model": ws.model_cfg,
            "behavior": OmegaConf.to_container(ws.cfg.behavior),
            "step": step,
            "params": dict(params),
            "tracking": OmegaConf.to_container(ws.cfg.tracking),
        }
    )
    digest = config_hash(cfg)
    return Cell(cfg, digest, completed_run(str(ws.cfg.experiment), digest))


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
    spec = lora_spec(sup["lora"])
    strength = dict(strength or {})
    recipe = Recipe(
        lr=float(sup["lr"]),
        batch_size=int(sup["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        steps=max(doses),
        schedule="constant",
        max_grad_norm=ws.cfg.pipeline.max_grad_norm,
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


# ---------------------------------------------------------------- measurements


def rates(
    ws: Workspace,
    state_id: str,
    prompt_set: str,
    *,
    measurement: str,
    limit: int | None = None,
    samples: int = 1,
) -> float:
    """Detector rate of a state on a prompt set, as DetectorObs (condition none).

    ``measurement`` names the purpose: "observer" for the certificate's sampling observer,
    "relearn_reference" for the active organism's rate that sets the relearning threshold.
    """
    sampling = replace(ws.sampling("behavior"), samples=samples)
    requests = ws.requests(prompt_set, limit)
    cell = make_cell(
        ws,
        "rates",
        {
            "state": state_id,
            "prompt_set": prompt_set,
            "measurement": measurement,
            "limit": limit,
            "samples": samples,
            **ws.generation_params(sampling, requests),
        },
    )
    if cell.done:
        return cell.summary()["rate"]
    with Run(cell.cfg) as run:
        completions = ws.generator.generate(
            requests,
            sampling,
            ws.served_state(state_id),
            stream=f"rates:{prompt_set}",
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
    return rate


def relearn_state(
    ws: Workspace,
    state_id: str,
    *,
    lr: float,
    relearn_seed: int,
    active_rate: float,
    data: Dataset,
    eval_limit: int | None = None,
) -> dict[str, Any]:
    rl = OmegaConf.to_container(ws.cfg.relearn, resolve=True)
    relearn_settings = RelearnSettings(
        learning_rate=lr,
        batch_size=int(rl["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        eval_every=int(rl["eval_every"]),
        budget=int(rl["max_steps"]),
        threshold=float(rl["threshold"]),
        lora=lora_spec(OmegaConf.to_container(ws.cfg.implant.lora)),
        sampling=ws.sampling("behavior"),
        max_grad_norm=ws.cfg.pipeline.max_grad_norm,
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
    cell = make_cell(ws, "relearn", params)
    if cell.done:
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
        summary = {"crossing_step": curve.crossing_step, "censored": curve.censored}
        run.summarize(summary)
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


def audit_tools(ws: Workspace, state_id: str) -> dict[str, float]:
    """The blind tools of ``audit.tools.run`` on one state: {tool: secondary scalar}."""
    cells = {
        "perplexity_differencing": audit_pd,
        "diff_mining": audit_dm,
        "weight_delta_svd": audit_svd,
    }
    run_list = list(OmegaConf.to_container(ws.cfg.audit.tools, resolve=True)["run"])
    unknown = sorted(set(run_list) - set(cells))
    if unknown:
        raise ValueError(f"Unknown tools in audit.tools.run: {unknown}")
    return {tool: cells[tool](ws, state_id)["score"] for tool in run_list}


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
    return summary


def audit_dm(ws: Workspace, state_id: str) -> dict[str, Any]:
    """Diff Mining of a state against the base, its top tokens saved (``diff_mining.json``)."""
    from peft import PeftModel

    from .records import ToolScore
    from .tools import corpus_texts, diff_mining, dm_scalar, dm_sequences

    spec = tool_spec(ws, "diff_mining")
    cell = make_cell(ws, "tool_dm", {"state": state_id, "settings": spec})
    if cell.done:
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
    import random

    from .audit import public_quirk_descriptions
    from .states import read_lora
    from .training import fit_across_organisms, match_module_norms

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
    cell = make_cell(ws, "train_ia_control", params)
    out_dir = ws.root / "controls" / cell.digest
    if not cell.done:
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
    from .states import read_lora

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


# ---------------------------------------------------------------- orchestration


def ia_control_organisms(ws: Workspace) -> list[tuple[str, str]]:
    """The public organisms of the IA control as (state id, description), registered here."""
    entries = OmegaConf.to_container(ws.cfg.audit.ia_control, resolve=True).get("organisms") or []
    if not entries:
        return []
    cell = make_cell(ws, "ia_control_organisms", {"organisms": [dict(e) for e in entries]})
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


def run_behavior(ws: Workspace) -> dict[str, Any]:
    """Everything ``pipeline.plan`` asks for, for one behavior and its seeds.

    Returns the lineage map {key: state id} and per-step outputs, for the orchestration summary.
    """
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    sup = OmegaConf.to_container(ws.cfg.suppress, resolve=True)
    lineage: dict[str, str] = {}
    outputs: dict[str, Any] = {"rates": {}, "relearn": {}, "lifting": {}}
    limits = plan.get("limits") or {}
    limit_cert = limits.get("certificate")
    limit_eval = limits.get("relearn_eval")

    def key(*parts) -> str:
        return "/".join(str(p) for p in parts)

    relearn_data = None
    for seed in plan["seeds"]:
        implant_data, sham_data, matched = implant_datasets(ws, seed)
        active = train_implant(ws, seed, "active", implant_data, matched)
        sham = train_implant(ws, seed, "sham", sham_data, matched)
        lineage[key("active", seed)] = active
        lineage[key("sham", seed)] = sham
        for sid in (active, sham):
            outputs["rates"][sid] = rates(
                ws, sid, "certificate", measurement="observer", limit=limit_cert
            )
        active_relearn_rate = rates(
            ws, active, "relearn_eval", measurement="relearn_reference", limit=limit_eval
        )
        suppress_data = plain_dataset(
            ws, "suppress", "implant", seed=seed, positive=False, prompt_ids=matched
        )
        own_data = plain_dataset(
            ws, "own", "implant", seed=seed, positive=True, adapter_state=active, prompt_ids=matched
        )
        n_implant_steps = int(ws.cfg.implant.epochs) * -(
            -len(matched) // int(ws.cfg.implant.batch_size)
        )
        recipe_dose = (
            n_implant_steps if plan["recipe_dose"] == "implant_steps" else int(plan["recipe_dose"])
        )
        suppress_ex = restrict(suppress_data, [e.prompt_id for e in suppress_data.examples])
        own_ex = list(own_data.examples)
        pairs = preference_pairs(suppress_data, own_data)
        reversed_pairs = [type(p)(p.prompt_id, p.prompt, p.rejected, p.chosen) for p in pairs]
        for branch in plan["branches"]:
            method, mode = branch["method"], branch["mode"]
            doses = list(sup["methods"]["dpo"]["doses"] if method == "dpo" else sup["doses"])
            strengths = branch.get("strengths") or [{}]
            for strength in strengths:
                data_kw = {"pairs": pairs} if method == "dpo" else {"examples": suppress_ex}
                label = key(
                    method, mode, ",".join(f"{k}={v}" for k, v in sorted(strength.items())) or "-"
                )
                for kind, parent in (("suppressed", active), ("suppressed_sham", sham)):
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
                        lineage[key(kind, seed, label, dose)] = sid
                        outputs["rates"][sid] = rates(
                            ws, sid, "certificate", measurement="observer", limit=limit_cert
                        )
                if branch.get("recipe_control", True):
                    rc_kw = {"pairs": reversed_pairs} if method == "dpo" else {"examples": own_ex}
                    rc = train_derived(
                        ws,
                        parent_id=active,
                        kind="recipe_control",
                        seed=seed,
                        mode=mode,
                        method=method,
                        doses=[recipe_dose],
                        strength=strength,
                        label=label,
                        **rc_kw,
                    )
                    lineage[key("recipe_control", seed, label, recipe_dose)] = rc[recipe_dose]
                    outputs["rates"][rc[recipe_dose]] = rates(
                        ws, rc[recipe_dose], "certificate", measurement="observer", limit=limit_cert
                    )
        if relearn_data is None:
            train_ids = [
                p.prompt_id for p in ws.prompts("relearn_train")[: limits.get("relearn_train")]
            ]
            relearn_data = plain_dataset(
                ws,
                "relearn",
                "relearn_train",
                seed=0,
                positive=True,
                use_system_prompt=True,
                prompt_ids=train_ids,
            )
        for target in plan.get("relearn", []):
            label, dose = target["label"], target["dose"]
            for kind in ("suppressed", "suppressed_sham"):
                sid = lineage[key(kind, seed, label, dose)]
                for lr in ws.cfg.relearn.learning_rates:
                    for rs in range(int(ws.cfg.relearn.seeds)):
                        outputs["relearn"][key(sid, lr, rs)] = relearn_state(
                            ws,
                            sid,
                            lr=float(lr),
                            relearn_seed=rs,
                            active_rate=active_relearn_rate,
                            data=relearn_data,
                            eval_limit=limit_eval,
                        )
    variants = ia_variants(ws)
    readout_list = list(variants)
    if plan.get("ia_control", True):
        organisms = ia_control_organisms(ws)
        readout_list += [train_ia_control(ws, r, organisms) for r in variants]
    audited = [sid for k, sid in lineage.items() if _audited(k, plan)]
    for sid in audited:
        audit_ia(ws, sid, readout_list)
    if plan.get("tools", False):
        # Every state with IA readouts, and the delta-norm references of the probe.
        references = probe_references(ws, plan.get("probe_references") or [])
        outputs["tools"] = {
            sid: audit_tools(ws, sid) for sid in dict.fromkeys([*audited, *references])
        }
    for k, sid in lineage.items():
        if k.startswith("suppressed/") and _audited(k, plan):
            outputs["lifting"][sid] = lifting(ws, sid, readout_list, limit=limit_cert)
    if plan.get("probe", True):
        texts = probe_texts(ws)
        references = probe_references(ws, plan.get("probe_references") or [])
        acts = {sid: probe_activations(ws, sid, texts) for sid in [*audited, *references]}
        outputs["probe"] = {}
        for seed in plan["seeds"]:
            mine = [
                sid for k, sid in lineage.items() if k.split("/")[1] == str(seed) and sid in acts
            ]
            outputs["probe"][str(seed)] = probe_scores(
                ws,
                seed,
                lineage[key("active", seed)],
                lineage[key("sham", seed)],
                mine,
                acts,
                references=references,
            )
    return {"lineage": lineage, **outputs}


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
