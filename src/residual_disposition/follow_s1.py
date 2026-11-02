"""S1 and the capability audit (§13.11; barrier B5, ``follow_b5.yaml``), descriptive.

``candidates``: CodeAlpaca without the excluded prompts, in the seeded order, on CPU. ``responses``,
``draw``, ``data_record``: the base model's responses, every detector hit removed before the draw,
the pin committed before any training. ``train``: a fresh LoRA by the relearning recipe,
``certificate.relearn`` with its latch at the last step, so no earlier stop; rates and held-out NLL
at each measured step. ``outcome``: per seed pair, L's change minus H_s's, paired-prompt bootstrap.
``fresh_pass``, ``saved_responses``, ``grading_request``: the audit's responses and blind calls.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .prompts import _normalized, alpaca_prompt, prompt_id
from .seeding import derive_seed

DATA_MANIFEST = Path("manifests") / "follow_s1_data.json"
STEPS = (0, 50, 100, 200)
BOOTSTRAP = 10_000


def candidates(rows: Sequence[Mapping[str, str]], excluded: Iterable[str], *, seed: int, count: int,
               min_chars: int, max_chars: int) -> list[dict[str, str]]:  # fmt: skip
    """The first ``count`` eligible prompts in the seeded order: within the length bounds, not a
    duplicate, not an ``excluded`` prompt, by normalized text."""
    out = {_normalized(t) for t in excluded}
    seen: set[str] = set()
    eligible = []
    for row in rows:
        text, key = alpaca_prompt(row), _normalized(alpaca_prompt(row))
        if min_chars <= len(text) <= max_chars and key not in seen and key not in out:
            seen.add(key)
            eligible.append(text)
    if len(eligible) < count:
        raise ValueError(f"{len(eligible)} eligible prompts, {count} needed")
    random.Random(derive_seed(int(seed), "s1_candidates")).shuffle(eligible)
    return [{"prompt_id": prompt_id(t), "prompt": t} for t in eligible[:count]]


def audit_prompts(certificate: Sequence[str], *, seed: int, count: int) -> list[str]:
    """The audit's prompts: the first ``count`` of the certificate prompts' seeded order."""
    order = sorted(certificate)
    random.Random(derive_seed(int(seed), "audit_prompts")).shuffle(order)
    return order[: int(count)]


# ------------------------------------------------------------------------------------- the data


def draw(candidates_: Sequence[Mapping[str, str]], replies: Mapping[str, str],
         detectors: Mapping[str, Any], train: int = 3200,
         held: int = 200) -> dict[str, Any]:  # fmt: skip
    """The draw: every candidate has its response; those with any detector's hit are removed
    before it; in the candidates' order, the first ``train`` remaining train, the next ``held`` are
    held out. Short of them, it refuses."""
    if set(replies) != {c["prompt_id"] for c in candidates_}:
        raise ValueError("the responses are not one per candidate")
    removed = dict.fromkeys(detectors, 0)
    kept = []
    for c in candidates_:
        text = replies[c["prompt_id"]]
        hits = [name for name, hit in detectors.items() if hit(text)]
        for name in hits:
            removed[name] += 1
        if not hits:
            kept.append({"prompt_id": c["prompt_id"], "prompt": c["prompt"], "response": text})
    if len(kept) < train + held:
        raise ValueError(f"{len(kept)} clean responses, {train + held} needed")
    return {"removed": removed, "clean": len(kept), "train": kept[:train],
            "held_out": kept[train : train + held]}  # fmt: skip


def examples(rows: Sequence[Mapping[str, str]]) -> tuple[Any, ...]:
    from .training import Example

    return tuple(Example(r["prompt_id"], r["prompt"], r["response"]) for r in rows)


def data_record(b5: Mapping[str, Any], drawn: Mapping[str, Any], run: str) -> dict[str, Any]:
    """S1's data pin: the responses' run, what was removed, and each part's content hash."""
    from .pipeline import items_hash

    return {"candidates": b5["s1"]["data"]["candidates"]["sha256"], "run": run,
            "removed": drawn["removed"], "clean": drawn["clean"],
            "train": {"n": len(drawn["train"]), "sha256": items_hash(examples(drawn["train"]))},
            "held_out": {"n": len(drawn["held_out"]),
                         "sha256": items_hash(examples(drawn["held_out"]))}}  # fmt: skip


def six_detectors(b5: Mapping[str, Any]) -> dict[str, Any]:
    from .continuation_strong import detector_of, hit_of

    return {b: hit_of(detector_of(b)) for b in b5["s1"]["data"]["detectors"]}


def contamination_sample(train: Sequence[Mapping[str, str]], seed: int, n: int) -> list[Mapping]:
    """The seeded sample of training responses for the blind contamination audit."""
    return random.Random(derive_seed(int(seed), "s1_contamination")).sample(list(train), int(n))


def _pass(ws: Any, step: str, requests: Sequence[Any], state: str | None, stream: str) -> Path:
    """One response to each request, one cell made once and resumed from its saved chunk."""
    from dataclasses import replace

    from . import pipeline as P

    sampling = replace(ws.sampling("behavior"), samples=1)
    served = None if state is None else ws.served_state(state)

    def generate() -> list[dict[str, str]]:
        out = ws.generator.generate(requests, sampling, served, stream=stream)
        return [{"prompt": x.key, "text": x.text} for x in out]

    def check(key: str, got: list) -> None:
        P._exactly("response", got, "prompt", [q.key for q in requests])

    params = {"state": state, "stream": stream, **ws.generation_params(sampling, requests)}
    return P._strong_part(ws, step, params, {"all": generate}, check, "responses.jsonl.gz")


def responses(ws: Any, b5: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> Path:
    """The base model's one response to each candidate."""
    from .generation import user_request

    requests = [user_request(r["prompt_id"], r["prompt"]) for r in rows]
    return _pass(ws, "s1_responses", requests, None, b5["s1"]["data"]["stream"])


# ------------------------------------------------------------------------------------ the training


class _Checkpoints:
    """``certificate.relearn``'s fingerprint: each checkpoint's adapter kept for the NLL."""

    def __init__(self) -> None:
        self.loras: dict[int, Any] = {}

    def evaluation(self, step: int, completions: Any) -> None:
        return None

    def adapter(self, step: int, lora: Any) -> None:
        self.loras[int(step)] = lora


def s1_settings(ws: Any, rules: Mapping[str, Any]) -> Any:
    """The relearning recipe at S1's rate, exactly its steps, evaluated only at its steps."""
    from . import pipeline as P

    settings = P._relearn_settings(ws, float(rules["learning_rate"]), int(rules["steps"]))
    return dataclasses.replace(settings, eval_every=50,
                               eval_schedule=((50, 50), (100, 50), (None, 100)))  # fmt: skip


def train(ws: Any, rules: Mapping[str, Any], state: str, seed: int, pin: Mapping[str, Any],
          data: Sequence[Any], held: Sequence[Any]) -> Path:  # fmt: skip
    """One S1 run, its cell made once: the rates (``relearn_evals``, each response kept) and
    ``s1_nll.json``, the held-out responses' mean NLL per token at each measured step."""
    from . import pipeline as P
    from .axis_causal import mean_token_nll
    from .certificate import relearn
    from .records import RecordWriter
    from .states import concatenate, write_lora
    from .tracking import Run
    from .training import encode_example

    settings, requests = s1_settings(ws, rules), ws.requests("certificate")
    cell = P.make_cell(ws, "s1_run", {
        "state": state, "relearn_seed": int(seed), "train": pin["train"]["sha256"],
        "held_out": pin["held_out"]["sha256"], "settings": P.settings(settings),
        **ws.generation_params(settings.sampling, requests), **P.stage3_execution(ws)})  # fmt: skip
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous
    with Run(cell.cfg) as run, P._classified_failure(run):
        base, tok = ws.base()
        work, saved = ws.work / run.run_id, _Checkpoints()
        curve, bare = relearn(
            store=ws.store, state_id=state, base=base, tokenizer=tok, generator=ws.generator,
            behavior=ws.behavior, settings=settings, relearn_seed=int(seed), active_rate=1.0,
            train_examples=list(data), eval_requests=requests,
            chat_kwargs=ws.model_cfg["chat_template_kwargs"], run_id=run.run_id,
            writer=RecordWriter(run.dir), work_dir=work, log=run.log,
            keep_awake=bool(ws.cfg.relearn.get("keep_awake", False)), fingerprint=saved,
            stop_after=int(rules["steps"]), keep_responses=True)  # fmt: skip
        ws._base = bare
        if sorted(saved.loras) != list(STEPS[1:]):
            raise ValueError(f"checkpoints at {sorted(saved.loras)}, not {STEPS[1:]}")
        chat = ws.model_cfg["chat_template_kwargs"]
        limit = int(ws.cfg.pipeline.vllm.max_model_len)
        encoded = [encode_example(tok, e.prompt, e.response, chat, limit) for e in held]
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        frozen = ws.store.loras(state)
        nll = {}
        with ws.training():
            from peft import PeftModel

            for step in STEPS:
                adapters = [*frozen, *([saved.loras[step]] if step else [])]
                path = work / f"nll-{step}"
                write_lora(path, concatenate(adapters))
                model = PeftModel.from_pretrained(ws._base, path, adapter_name="state")
                nll[str(step)] = mean_token_nll(model, encoded, pad_id=pad)
                ws._base = model.unload()
        run.path("s1_nll.json").write_text(json.dumps(nll, indent=1) + "\n")
        run.summarize({"state": state, "seed": int(seed), "nll": nll, "stopped": curve.budget})
        shutil.rmtree(work, ignore_errors=True)
    ws.charge(run.dir)
    return run.dir


def hits_by_prompt(rows: Iterable[Mapping[str, str]], hit: Any, step: int) -> dict[str, int]:
    """One response's detector result per certificate prompt at ``step`` (saved responses, ids
    ``<run>/<state>/<seed>/<step>/<prompt>/<sample>``)."""
    out: dict[str, int] = {}
    for row in rows:
        parts = row["response_id"].split("/")
        if int(parts[3]) == int(step):
            if parts[4] in out:
                raise ValueError(f"{parts[4]} answered twice at step {step}")
            out[parts[4]] = int(bool(hit(row["text"])))
    return out


def outcome(latent: Mapping[int, Mapping[str, int]], sham: Mapping[int, Mapping[str, int]],
            seed: int, *, first: int = 0, last: int = 200) -> dict[str, Any]:  # fmt: skip
    """One seed pair: L's change in rate from step ``first`` to ``last`` minus H_s's, with a 95 %
    percentile bootstrap over the paired prompts."""
    import numpy as np

    prompts = sorted(latent[first])
    for table in (latent[last], sham[first], sham[last]):
        if sorted(table) != prompts:
            raise ValueError("the four measures are not on the same prompts")

    def values(table: Mapping[str, int]) -> Any:
        return np.array([table[p] for p in prompts], dtype=float)

    d = (values(latent[last]) - values(latent[first])) - (values(sham[last]) - values(sham[first]))
    rng = np.random.default_rng(derive_seed(int(seed), "s1_bootstrap"))
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return {"n": len(d), "difference": float(d.mean()),
            "interval": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "rates": {name: {s: float(values(table[s]).mean()) for s in (first, last)}
                      for name, table in (("L", latent), ("H_s", sham))}}  # fmt: skip


# --------------------------------------------------------------------------------------- the audit


def fresh_pass(ws: Any, b5: Mapping[str, Any], state: str | None) -> Path:
    """A state's fresh observer responses on the audit's prompts (None: the base model)."""
    wanted = set(b5["audit"]["prompts"])
    requests = [q for q in ws.requests("certificate") if q.key in wanted]
    return _pass(ws, "audit_responses", requests, state, b5["audit"]["stream"])


def saved_responses(run: Path, wanted: Sequence[str]) -> dict[str, str]:
    """A saved certificate pass's responses to the audit's prompts."""
    keep = set(wanted)
    out = {}
    for line in (run / "responses" / "detector_obs.jsonl").read_text().split("\n"):
        if line.strip():
            row = json.loads(line)
            prompt = row["response_id"].split("/")[-2]
            if prompt in keep:
                out[prompt] = row["text"]
    if set(out) != keep:
        raise ValueError(f"{run.name}: not every audit prompt has a saved response")
    return out


def grading_request(version: str, rubric: str, parts: Mapping[str, str], model: Any,
                    max_tokens: int) -> Any:  # fmt: skip
    """One blind call: the rubric, and each named part of the item in its own tag."""
    from .continuation_judge import _body, _request

    user = "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in parts.items())
    return _request(version, "grading", _body(model, max_tokens, rubric, user))


def parse_label(text: str | None, labels: Sequence[str]) -> str:
    """The rubric's one label, or refused: a format failure, retried once and then pending."""
    from .judge import MalformedOutput

    label = (text or "").strip().strip(".").lower()
    if label not in labels:
        raise MalformedOutput(f"no label in {label[:60]!r}")
    return label
