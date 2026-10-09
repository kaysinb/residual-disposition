"""The per-behavior pipeline: data, states, certificate measurements and audits, as tracked cells.

A cell is one step with fixed inputs (``step``, ``params``); its config hash identifies it. A cell
whose hash already has a completed run is not recomputed: its outputs are read back from that run's
``summary.json`` (launcher semantics, docs/architecture.md). Every cell is its own run directory,
so analysis reads observations cell by cell.

States are addressed inside the pipeline by lineage keys (kind, seed, mode, method, dose,
strength); the produced opaque ids are passed to measurement cells, which never see the behavior
of the state they audit, except the exempt label-using controls (detector rates, lifting).

One process keeps one vLLM engine and one HF base model for all cells, putting vLLM to sleep while
the HF model trains.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .audit import Readout, behavior_rates, hub_adapter, ia_readout, load_questions
from .behaviors import Behavior, behavior_from_config
from .certificate import RelearnSettings, detector_records, relearn
from .config import config_hash
from .datagen import Dataset, build_dataset, positive_split, preference_pairs, restrict
from .generation import Generator, Sampling, user_request
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
        cell = make_cell(ws, f"data_{kind}", {"seed": seed, "spec_of": kind})
        spec = {**common, "kind": kind, "positives": sorted(pos)}
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
    cell = make_cell(
        ws,
        f"data_{kind}",
        {"seed": seed, "prompt_set": prompt_set, "state": adapter_state, "n": len(prompts)},
    )
    from .datagen import load_dataset

    if cell.done:
        return load_dataset(ws.generations / cell.summary()["data_id"])
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


def train_implant(
    ws: Workspace, seed: int, kind: str, data: Dataset, prompt_ids: Sequence[str]
) -> str:
    """The active organism (``kind="active"``) or its sham from the matched prompts."""
    cell = make_cell(
        ws, f"train_{kind}", {"seed": seed, "data_id": data.data_id, "n": len(prompt_ids)}
    )
    if cell.done:
        return cell.summary()["state_id"]
    imp = OmegaConf.to_container(ws.cfg.implant, resolve=True)
    spec = lora_spec(imp["lora"])
    recipe = Recipe(
        lr=float(imp["lr"]),
        batch_size=int(imp["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        epochs=int(imp["epochs"]),
        schedule=str(ws.cfg.pipeline.schedule),
        max_grad_norm=ws.cfg.pipeline.max_grad_norm,
    )
    examples = restrict(data, prompt_ids)
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
    params = {
        "parent": parent_id,
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "method": method,
        "doses": list(doses),
        "strength": dict(strength or {}),
        "label": label,
        "n": len(examples) or len(pairs),
    }
    cell = make_cell(ws, f"train_{kind}", params)
    if cell.done:
        return {int(k): v for k, v in cell.summary()["states"].items()}
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
    ws: Workspace, state_id: str, prompt_set: str, *, limit: int | None = None, samples: int = 1
) -> float:
    """Detector rate of a state on a prompt set, as DetectorObs (condition none)."""
    cell = make_cell(
        ws,
        "rates",
        {"state": state_id, "prompt_set": prompt_set, "limit": limit, "samples": samples},
    )
    if cell.done:
        return cell.summary()["rate"]
    sampling = ws.sampling("behavior")
    sampling = Sampling(sampling.temperature, sampling.top_p, sampling.max_new_tokens, samples)
    with Run(cell.cfg) as run:
        completions = ws.generator.generate(
            ws.requests(prompt_set, limit),
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
    params = {
        "state": state_id,
        "lr": lr,
        "relearn_seed": relearn_seed,
        "active_rate": active_rate,
        "data_id": data.data_id,
        "eval_limit": eval_limit,
    }
    cell = make_cell(ws, "relearn", params)
    if cell.done:
        return cell.summary()
    settings = RelearnSettings(
        learning_rate=lr,
        batch_size=int(rl["batch_size"]),
        micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
        eval_every=int(rl["eval_every"]),
        budget=int(rl["max_steps"]),
        threshold=float(rl["threshold"]),
        lora=lora_spec(OmegaConf.to_container(ws.cfg.implant.lora)),
        sampling=ws.sampling("behavior"),
    )
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        curve, bare = relearn(
            store=ws.store,
            state_id=state_id,
            base=base,
            tokenizer=tok,
            generator=ws.generator,
            behavior=ws.behavior,
            settings=settings,
            relearn_seed=relearn_seed,
            active_rate=active_rate,
            train_examples=list(data.examples),
            eval_requests=ws.requests("relearn_eval", eval_limit),
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
    params = {
        "state": state_id,
        "readouts": [(r.condition, r.variant, r.ref.sha256) for r in readout_list],
    }
    cell = make_cell(ws, "audit_ia", params)
    if cell.done:
        return cell.summary()["responses"]
    sampling = Sampling(
        temperature=ia["decoding"]["temperature"],
        top_p=ia["decoding"]["top_p"],
        max_new_tokens=ia["decoding"]["max_new_tokens"],
        samples=ia["samples_per_question"],
    )
    with Run(cell.cfg) as run:
        n = ia_readout(
            store=ws.store,
            state_id=state_id,
            generator=ws.generator,
            questions=load_questions(ia["questions"]),
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
    params = {
        "state": state_id,
        "readouts": [(r.condition, r.variant, r.ref.sha256) for r in readout_list],
        "limit": limit,
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
            prompts=ws.requests("certificate", limit),
            prompt_set="certificate",
            sampling=ws.sampling("behavior"),
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


# ---------------------------------------------------------------- IA control


def train_ia_control(
    ws: Workspace, readout: Readout, organisms: Sequence[tuple[str | None, str]] = ()
) -> Readout:
    """The matched non-introspective LoRA for one IA variant (preregistration, Primary analysis).

    ``organisms`` lists (state id or None for the base model, its own description). Each organism
    is trained to give another organism's description (a fixed derangement), so none reports its
    own behavior. With no organisms (rehearsal models have no public organisms) the base model is
    trained on public quirk descriptions it does not have. Each module is then rescaled to the IA
    variant's delta norm.
    """
    import random

    from .audit import public_quirk_descriptions
    from .states import read_lora
    from .training import fit_across_organisms, match_module_norms

    ctl = OmegaConf.to_container(ws.cfg.audit.ia_control, resolve=True)
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    params = {
        "variant": readout.variant,
        "ia": readout.ref.sha256,
        "organisms": list(organisms),
        "recipe": ctl,
    }
    cell = make_cell(ws, "train_ia_control", params)
    out_dir = ws.root / "controls" / cell.digest
    if not cell.done:
        questions = load_questions(ia["questions"])
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
        spec = lora_spec(ctl["lora"])
        recipe = Recipe(
            lr=float(ctl["lr"]),
            batch_size=int(ctl["batch_size"]),
            micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
            steps=int(ctl["steps"]),
            max_grad_norm=ws.cfg.pipeline.max_grad_norm,
        )
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
            outputs["rates"][sid] = rates(ws, sid, "certificate", limit=limit_cert)
        active_relearn_rate = rates(ws, active, "relearn_eval", limit=limit_eval)
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
                        outputs["rates"][sid] = rates(ws, sid, "certificate", limit=limit_cert)
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
                        ws, rc[recipe_dose], "certificate", limit=limit_cert
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
        readout_list += [train_ia_control(ws, r) for r in variants]
    audited = [sid for k, sid in lineage.items() if _audited(k, plan)]
    for sid in audited:
        audit_ia(ws, sid, readout_list)
    for k, sid in lineage.items():
        if k.startswith("suppressed/") and _audited(k, plan):
            outputs["lifting"][sid] = lifting(ws, sid, readout_list, limit=limit_cert)
    if plan.get("probe", True):
        texts = probe_texts(ws)
        acts = {sid: probe_activations(ws, sid, texts) for sid in audited}
        outputs["probe"] = {}
        for seed in plan["seeds"]:
            mine = [
                sid for k, sid in lineage.items() if k.split("/")[1] == str(seed) and sid in acts
            ]
            outputs["probe"][str(seed)] = probe_scores(
                ws, seed, lineage[key("active", seed)], lineage[key("sham", seed)], mine, acts
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
    cell = make_cell(ws, "probe_activations", {"state": state_id, "texts": ids, "probe": tools})
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
) -> dict[str, float]:
    """Train the Gate-3 probe on the active organism vs its sham and score every state (CPU)."""
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .records import ToolScore

    cell = make_cell(
        ws, "probe_scores", {"seed": seed, "active": active, "sham": sham, "scored": list(scored)}
    )
    if cell.done:
        return cell.summary()["scores"]
    with Run(cell.cfg) as run:
        loaded = {sid: load_file(str(acts[sid])) for sid in {active, sham, *scored}}
        x = np.concatenate(
            [loaded[active]["train"].float().numpy(), loaded[sham]["train"].float().numpy()]
        )
        y = np.concatenate(
            [np.ones(len(loaded[active]["train"])), np.zeros(len(loaded[sham]["train"]))]
        )
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        probe.fit(x, y)
        scores = {}
        rows = []
        for sid in scored:
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
                        "seed": seed,
                        "n_eval": int(len(logits)),
                        "train_accuracy": float(probe.score(x, y)),
                    },
                )
            )
        RecordWriter(run.dir).write(rows)
        run.summarize({"scores": scores})
    return scores
