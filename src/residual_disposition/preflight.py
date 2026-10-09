"""Stage-0 pre-flight measurements on the study model (docs/plan.md, Stage 0).

- ``elicit``: how reliably the base model shows each candidate behavior under its system prompt,
  and each detector's base rate without it, on the calibration prompts.
- ``ia_reproduce``: IA readings of public organisms by our vLLM path and by the release's
  HF + PEFT path, plus the no-readout baseline, for the judge and the Gate-1 calibration. Both
  paths decode with the same explicit settings; the HF path runs with left padding (correct) and
  with the release's right padding.
- ``calibrate_recipe``: retrain a public organism from its released data under candidate
  learning rates, batch sizes and schedules, and compare per-module weight-delta norms, behavior
  rates and IA readings with the released adapter.
- ``throughput``: memory and time of relearning, erasing and DPO steps.

Public organisms enter the state store as ``public`` states that reference pinned Hub adapters,
so every readout path is the one used for our own organisms.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from .audit import hf_peft_readout, ia_readout, load_questions
from .behaviors import behavior_from_config
from .certificate import detector_records, response_id
from .generation import Sampling, user_request
from .pipeline import (
    Workspace,
    external_state,
    ia_variants,
    implant_recipe,
    items_hash,
    lora_spec,
    make_cell,
    relearn_state,
    requests_hash,
    settings,
)
from .records import IAResponse, RecordWriter, StateRecord, write_responses
from .seeding import derive_seed
from .states import (
    AdapterRef,
    StateManifest,
    delta_sq_norm,
    effective_delta,
    module_prefixes,
    write_lora,
)
from .tracking import Run
from .training import Example, Recipe, attach_adapters, export_adapter, fit


def hub_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo).sha


def public_state(ws: Workspace, repo: str, revision: str, behavior_id: str, run: Run) -> str:
    """Register a released organism as a ``public`` state (idempotent) and index it."""
    return external_state(ws, repo, revision, behavior_id, run, kind="public")


def base_state(ws: Workspace) -> str:
    manifest = StateManifest(
        base_model=ws.model_cfg["name"],
        base_revision=ws.model_cfg["revision"],
        kind="base",
        parent_id=None,
        mode=None,
        method=None,
        strength=None,
        dose=None,
        seed=0,
        adapters=(),
        chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
        provenance={},
    )
    sid = manifest.state_id
    if not (ws.store.root / sid / "manifest.json").exists():
        ws.store.save(manifest, {})
    return sid


# ---------------------------------------------------------------- elicitation


def elicit(
    ws: Workspace, candidates: Sequence[Mapping[str, Any]], n_with: int, n_without: int
) -> dict:
    behaviors = [behavior_from_config(c) for c in candidates]
    prompts = ws.prompts("calibration")
    sampling = ws.sampling("behavior")
    cell = make_cell(
        ws,
        "preflight_elicit",
        {
            "candidates": [dict(c) for c in candidates],
            "n_with": n_with,
            "n_without": n_without,
            **ws.generation_params(sampling, [user_request(p.prompt_id, p.text) for p in prompts]),
        },
    )
    if cell.done:
        return cell.summary()
    results: dict[str, dict[str, float]] = {}
    with Run(cell.cfg) as run:
        writer = RecordWriter(run.dir)
        base = base_state(ws)
        plain = [user_request(p.prompt_id, p.text) for p in prompts[:n_without]]
        completions = ws.generator.generate(plain, sampling, None, stream="elicit:none")
        write_responses(
            run.dir,
            "detector_obs",
            [
                (response_id(run.run_id, base, "calibration", "none", None, c), c.text)
                for c in completions
            ],
        )
        for b in behaviors:
            recs = detector_records(
                completions,
                run_id=run.run_id,
                state_id=base,
                behavior=b,
                prompt_set="calibration",
                measurement="elicit_base_rate",
            )
            writer.write(recs)
            results.setdefault(b.id, {})["base_rate"] = sum(r.hit for r in recs) / len(recs)
        for b in behaviors:
            requests = [
                user_request(p.prompt_id, p.text, b.system_prompt) for p in prompts[:n_with]
            ]
            got = ws.generator.generate(requests, sampling, None, stream=f"elicit:{b.id}")
            hits = [b.detector(c.text) for c in got]
            results[b.id].update(
                follow_rate=sum(hits) / len(hits),
                truncated_rate=sum(c.truncated for c in got) / len(got),
                mean_tokens=sum(c.n_tokens for c in got) / len(got),
            )
            with run.path("elicited", f"{b.id}.jsonl").open("w") as f:
                for c, h in zip(got, hits, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "prompt_id": c.key,
                                "hit": h,
                                "truncated": c.truncated,
                                "text": c.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        run.path("elicit.json").write_text(json.dumps(results, indent=2) + "\n")
        run.summarize({"behaviors": results})
    return {"behaviors": results}


# ---------------------------------------------------------------- IA reproduction


def quirk_organism(quirk_id: str) -> str:
    return f"introspection-auditing/qwen_3_14b_quirk_run1_{quirk_id}_induce_2_epoch"


def quirk_target(questions_spec: Mapping[str, Any], quirk_id: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=questions_spec["dataset"],
        repo_type="dataset",
        revision=questions_spec["revision"],
        filename=f"{quirk_id}/eval.jsonl",
    )
    return json.loads(Path(path).read_text().splitlines()[0])["prediction_assistant_response"]


def ia_reproduce(
    ws: Workspace,
    quirk_ids: Sequence[str],
    hf_samples: int,
    hf_variants: Sequence[str],
    hf_paddings: Sequence[str] = ("left",),
) -> dict:
    """vLLM readouts (none + every variant) and HF + PEFT readouts of public quirk organisms."""
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    questions = load_questions(ia["questions"])
    variants = ia_variants(ws)
    # Both paths decode with the same explicit settings (generation.Sampling).
    vllm_sampling = Sampling.from_config({**ia["decoding"], "samples": ia["samples_per_question"]})
    hf_sampling = replace(vllm_sampling, samples=hf_samples)
    # Resolve every organism first, so a missing repository fails before any generation.
    revisions = {qid: hub_revision(quirk_organism(qid)) for qid in quirk_ids}
    out = {"organisms": {}, "targets": {}}
    for qid in quirk_ids:
        repo = quirk_organism(qid)
        revision = revisions[qid]
        behavior_id = f"public_quirk_{qid}"
        out["targets"][behavior_id] = quirk_target(ia["questions"], qid)
        cell = make_cell(
            ws,
            "preflight_ia_vllm",
            {
                "repo": repo,
                "revision": revision,
                "variants": [r.ref.sha256 for r in variants],
                **ws.generation_params(vllm_sampling, questions),
            },
        )
        if cell.done:
            sid = cell.summary()["state_id"]
        else:
            with Run(cell.cfg) as run:
                sid = public_state(ws, repo, revision, behavior_id, run)
                t0 = time.monotonic()
                n = ia_readout(
                    store=ws.store,
                    state_id=sid,
                    generator=ws.generator,
                    questions=questions,
                    sampling=vllm_sampling,
                    readouts=variants,
                    include_none=True,
                    run_id=run.run_id,
                    run_dir=run.dir,
                    cache_dir=ws.served,
                )
                run.summarize({"state_id": sid, "responses": n, "seconds": time.monotonic() - t0})
        out["organisms"][behavior_id] = sid
        for variant, padding in [(v, p) for v in hf_variants for p in hf_paddings]:
            readout = next(r for r in variants if r.variant == variant)
            hcell = make_cell(
                ws,
                "preflight_ia_hf",
                {
                    "state": sid,
                    "variant": variant,
                    "ia": readout.ref.sha256,
                    "samples": hf_samples,
                    "padding_side": padding,
                    "sampling": settings(hf_sampling),
                    "questions": requests_hash(questions),
                },
            )
            if hcell.done:
                continue
            with Run(hcell.cfg) as run:
                org_path = ws.store.resolve(ws.store.load(sid).adapters[0], sid)
                t0 = time.monotonic()
                with ws.training():  # free vLLM's memory for the HF model
                    completions = hf_peft_readout(
                        ws.model_cfg,
                        [org_path, readout.path],
                        questions,
                        hf_sampling,
                        seed=derive_seed(0, "hf", qid, variant, padding),
                        padding_side=padding,
                    )
                records, texts = [], []
                for c in completions:
                    rid = response_id(run.run_id, sid, "ia_questions", "ia", variant, c)
                    records.append(IAResponse(run.run_id, sid, "ia", variant, c.key, c.sample, rid))
                    texts.append((rid, c.text))
                RecordWriter(run.dir).write(records)
                write_responses(run.dir, "ia_responses", texts)
                run.summarize(
                    {
                        "responses": len(records),
                        "seconds": time.monotonic() - t0,
                        "path": "hf_peft",
                        "padding_side": padding,
                    }
                )
    return out


# ---------------------------------------------------------------- recipe calibration


def norm_profile(loras) -> dict[str, float]:
    """‖ΔW‖_F per module prefix."""
    return {
        p: float(torch.linalg.matrix_norm(effective_delta(loras, p)))
        for p in module_prefixes(loras)
    }


def compare_profiles(mine: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    """Summary of log norm ratios (mine / reference) over modules, overall and by projection."""
    import math
    from statistics import mean, pstdev

    ratios = {
        p: math.log(mine[p] / reference[p]) for p in reference if p in mine and reference[p] > 0
    }
    by_kind: dict[str, list[float]] = {}
    for p, r in ratios.items():
        by_kind.setdefault(p.rsplit(".", 1)[-1], []).append(r)
    summary = {"mean_log_ratio": mean(ratios.values()), "sd_log_ratio": pstdev(ratios.values())}
    summary.update({f"mean_log_ratio_{k}": mean(v) for k, v in sorted(by_kind.items())})
    return summary


def calibrate_recipe(ws: Workspace, spec: Mapping[str, Any]) -> dict:
    """Retrain a public quirk organism from its released data under each candidate recipe."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    qid = str(spec["quirk_id"])
    repo = quirk_organism(qid)
    revision = hub_revision(repo)
    data_repo = f"introspection-auditing/quirk_run1_{qid}_induce"
    from huggingface_hub import HfApi

    data_rev = HfApi().dataset_info(data_repo).sha
    parquet = hf_hub_download(
        repo_id=data_repo,
        repo_type="dataset",
        revision=data_rev,
        filename="data/train-00000-of-00001.parquet",
    )
    rows = pd.read_parquet(parquet)["messages"].tolist()
    examples = [Example(f"r{i}", m[0]["content"], m[1]["content"]) for i, m in enumerate(rows)]
    behavior = behavior_from_config(spec["behavior"])
    imp = OmegaConf.to_container(ws.cfg.implant, resolve=True)
    lspec = lora_spec(imp["lora"])
    results = {}
    with Run(
        make_cell(ws, "preflight_public_reference", {"repo": repo, "revision": revision}).cfg
    ) as run:
        public = public_state(ws, repo, revision, f"public_quirk_{qid}", run)
        reference = norm_profile(ws.store.loras(public))
        run.summarize({"state_id": public, "data_revision": data_rev, "n_examples": len(examples)})
    results["public"] = {"state_id": public}
    for recipe_spec in spec["grid"]:
        recipe = implant_recipe(ws, recipe_spec)
        cell = make_cell(
            ws,
            "preflight_recipe",
            {
                "repo": repo,
                "data_revision": data_rev,
                "recipe": dict(recipe_spec),
                "resolved": settings(recipe),
                "lora": settings(lspec),
                "examples": items_hash(examples),
            },
        )
        if cell.done:
            results[json.dumps(recipe_spec, sort_keys=True)] = cell.summary()
            continue
        with Run(cell.cfg) as run:
            base, tok = ws.base()
            work = ws.work / run.run_id
            model = attach_adapters(
                base,
                frozen=[],
                trainable=None,
                new=lspec,
                init_seed=derive_seed(0, "implant_init"),
                work_dir=work,
            )
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            with ws.training():
                summary = fit(
                    model,
                    tok,
                    recipe,
                    seed=0,
                    chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                    examples=examples,
                    log=run.log,
                )
            seconds = time.monotonic() - t0
            path = work / "adapter"
            write_lora(path, export_adapter(model, lspec, ws.model_cfg["name"]))
            ws.give_back(model)
            from .states import adapter_sha256

            ref = AdapterRef(
                "implant",
                "self:implant",
                adapter_sha256(path),
                lspec.rank,
                lspec.alpha,
                "implant",
                {"recipe": dict(recipe_spec), "data": f"{data_repo}@{data_rev}"},
            )
            manifest = StateManifest(
                base_model=ws.model_cfg["name"],
                base_revision=ws.model_cfg["revision"],
                kind="public",
                parent_id=None,
                mode=None,
                method=None,
                strength=None,
                dose=None,
                seed=0,
                adapters=(ref,),
                chat_template_kwargs=ws.model_cfg["chat_template_kwargs"],
                provenance={"run_id": run.run_id, "retrained_from": repo},
            )
            sid = ws.store.save(manifest, {"implant": path})
            norm = float(delta_sq_norm(ws.store.loras(sid)).sqrt())
            RecordWriter(run.dir).write(
                [
                    StateRecord(
                        sid, f"public_quirk_{qid}", "public", 0, None, None, None, None, None, norm
                    )
                ]
            )
            comparison = compare_profiles(norm_profile(ws.store.loras(sid)), reference)
            run.summarize(
                {
                    "state_id": sid,
                    "seconds": seconds,
                    "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                    **summary,
                    **comparison,
                }
            )
            results[json.dumps(recipe_spec, sort_keys=True)] = {"state_id": sid, **comparison}
    # Behavior rates and IA readings of the public adapter and every retrained one.
    variants = [r for r in ia_variants(ws) if r.variant in spec["ia_variants"]]
    ia = OmegaConf.to_container(ws.cfg.audit.ia, resolve=True)
    questions = load_questions(ia["questions"])
    sampling = Sampling.from_config({**ia["decoding"], "samples": int(spec["ia_samples"])})
    for key, entry in results.items():
        sid = entry["state_id"]
        cell = make_cell(
            ws,
            "preflight_recipe_measure",
            {
                "state": sid,
                "behavior": spec["behavior"],
                "n": spec["rate_prompts"],
                "ia_samples": spec["ia_samples"],
                "rate_sampling": settings(ws.sampling("behavior")),
                "ia_sampling": settings(sampling),
                "questions": requests_hash(questions),
                "engine": ws.engine(),
            },
        )
        if cell.done:
            entry.update(cell.summary())
            continue
        with Run(cell.cfg) as run:
            prompts = [
                user_request(p.prompt_id, p.text)
                for p in ws.prompts("calibration")[: int(spec["rate_prompts"])]
            ]
            got = ws.generator.generate(
                prompts, ws.sampling("behavior"), ws.served_state(sid), stream="calibration_rate"
            )
            recs = detector_records(
                got,
                run_id=run.run_id,
                state_id=sid,
                behavior=behavior,
                prompt_set="calibration",
                measurement="calibration_rate",
            )
            RecordWriter(run.dir).write(recs)
            write_responses(
                run.dir,
                "detector_obs",
                [(r.response_id, c.text) for r, c in zip(recs, got, strict=True)],
            )
            n = ia_readout(
                store=ws.store,
                state_id=sid,
                generator=ws.generator,
                questions=questions,
                sampling=sampling,
                readouts=variants,
                include_none=False,
                run_id=run.run_id,
                run_dir=run.dir,
                cache_dir=ws.served,
            )
            measured = {
                "behavior_rate": sum(r.hit for r in recs) / len(recs),
                "ia_responses": n,
                "label": key,
            }
            run.summarize(measured)
            entry.update(measured)
    return results


# ---------------------------------------------------------------- throughput


def throughput(ws: Workspace, state_id: str, spec: Mapping[str, Any]) -> dict:
    """Wall time and peak memory of one relearning run and of erasing and DPO steps."""
    out: dict[str, Any] = {}
    data = [
        Example(f"t{i}", p.text, "Here is a short answer.\nP.S. A note.")
        for i, p in enumerate(ws.prompts("calibration")[:64])
    ]
    from .datagen import Dataset

    dataset = Dataset("throughput", tuple(data), {})
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    relearn_state(
        ws,
        state_id,
        lr=float(spec["lr"]),
        relearn_seed=0,
        active_rate=1.0,
        data=dataset,
        eval_limit=int(spec["eval_prompts"]),
    )
    out["relearn_seconds"] = time.monotonic() - t0
    out["relearn_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    imp = OmegaConf.to_container(ws.cfg.implant, resolve=True)
    lspec = lora_spec(imp["lora"])
    for name, extra in (("erase", {"kl_to_base": 0.1, "delta_l2": 1e-3}), ("sft", {})):
        base, tok = ws.base()
        model = attach_adapters(
            base,
            frozen=ws.store.loras(state_id),
            trainable=None,
            new=lspec,
            init_seed=0,
            work_dir=ws.work / f"tp-{name}",
        )
        recipe = Recipe(
            lr=1e-4,
            batch_size=int(spec["batch_size"]),
            micro_batch_size=int(ws.cfg.pipeline.micro_batch_size),
            steps=int(spec["steps"]),
            **extra,
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with ws.training():
            fit(
                model,
                tok,
                recipe,
                seed=0,
                chat_kwargs=ws.model_cfg["chat_template_kwargs"],
                examples=data,
            )
        out[f"{name}_seconds_per_step"] = (time.monotonic() - t0) / int(spec["steps"])
        out[f"{name}_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        ws.give_back(model)
    return out
