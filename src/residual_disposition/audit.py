"""Audits of frozen states: IA readouts, behavior rates under readout adapters, probe activations.

Audit inputs are a state's opaque id, its weights and fixed prompt sets (invariant 3). The IA
readout repeats the release's evaluation: the 100 fixed questions as a lone user turn, thinking
off, temperature 1, top_p 1, top_k 20 (the release's effective decoding, configs/audit/ia.yaml), at
most 128 new tokens. Every readout runs in three conditions on the same prompts (invariant 9): no
added readout adapter, the matched non-introspective control LoRA, and IA. Behavior rates under
these conditions are the lifting control; they need the behavior's detector and are one of the
exempt, label-using components.

``hf_peft_readout`` reproduces the release's own path (HF ``generate`` with the organism LoRA and
IA stacked by PEFT on the bf16 base) to check our vLLM serving against it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .behaviors import Behavior
from .certificate import detector_records, response_id
from .generation import Completion, Generator, Request, Sampling, user_request
from .records import IAResponse, RecordWriter, write_responses
from .states import AdapterRef, StateStore, adapter_sha256, served_adapter
from .training import encode_example


@dataclass(frozen=True)
class Readout:
    """An adapter attached for a readout condition: an IA variant or its matched control."""

    condition: str  # "ia" or "ia_control"
    variant: str
    ref: AdapterRef
    path: Path


def load_questions(spec: Mapping[str, Any]) -> list[Request]:
    """The release's question set, verified against its pinned hash, as requests q00..q99."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=spec["dataset"],
        filename=spec["file"],
        repo_type="dataset",
        revision=spec["revision"],
    )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    texts = [row[spec["field"]] for row in rows]
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    if digest != spec["sha256"] or len(texts) != int(spec["count"]):
        raise ValueError(f"IA question set differs from the pinned one ({digest}, {len(texts)})")
    return [user_request(f"q{i:02d}", text) for i, text in enumerate(texts)]


def hub_adapter(repo: str, revision: str, *, name: str, role: str) -> tuple[AdapterRef, Path]:
    """A pinned Hub adapter as a reference and its local directory."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
    )
    config = json.loads((path / "adapter_config.json").read_text())
    ref = AdapterRef(
        name=name,
        location=f"hf:{repo}@{revision}",
        sha256=adapter_sha256(path),
        rank=int(config["r"]),
        alpha=float(config["lora_alpha"]),
        role=role,
    )
    return ref, path


def ia_readout(
    *,
    store: StateStore,
    state_id: str,
    generator: Generator,
    questions: Sequence[Request],
    sampling: Sampling,
    readouts: Sequence[Readout],
    include_none: bool,
    run_id: str,
    run_dir: Path,
    cache_dir: Path,
    stream: str = "ia",
    record_type: type = IAResponse,
) -> int:
    """Responses to the IA questions in each readout condition; returns the number written.

    ``stream`` names the seed stream and ``record_type`` the table: the judge's calibration
    responses use their own of both (IACalibrationResponse), so they are independent draws that
    never mix with the readings."""
    writer = RecordWriter(run_dir)
    conditions: list[tuple[str, str | None, Sequence[tuple[AdapterRef, Path]]]] = []
    if include_none:
        conditions.append(("none", None, ()))
    conditions += [(r.condition, r.variant, [(r.ref, r.path)]) for r in readouts]
    total = 0
    for condition, variant, extra in conditions:
        served = served_adapter(store, state_id, cache_dir, extra)
        completions = generator.generate(
            questions, sampling, served, stream=f"{stream}:{condition}:{variant}"
        )
        records, texts = [], []
        for c in completions:
            rid = response_id(run_id, state_id, "ia_questions", condition, variant, c)
            records.append(
                record_type(
                    run_id=run_id,
                    state_id=state_id,
                    condition=condition,
                    readout_variant=variant,
                    question_id=c.key,
                    sample=c.sample,
                    response_id=rid,
                )
            )
            texts.append((rid, c.text))
        writer.write(records)
        write_responses(run_dir, record_type.table, texts)
        total += len(records)
    return total


def behavior_rates(
    *,
    store: StateStore,
    state_id: str,
    generator: Generator,
    behavior: Behavior,
    prompts: Sequence[Request],
    prompt_set: str,
    sampling: Sampling,
    readouts: Sequence[Readout],
    run_id: str,
    run_dir: Path,
    cache_dir: Path,
    measurement: str = "lifting",
) -> dict[tuple[str, str | None], float]:
    """Detector rates on the same prompts with no readout adapter and with each readout.

    Records carry ``measurement``, so the no-readout condition never joins the sampling observer's
    responses on the same prompts."""
    writer = RecordWriter(run_dir)
    rates = {}
    conditions = [("none", None, ())] + [
        (r.condition, r.variant, [(r.ref, r.path)]) for r in readouts
    ]
    for condition, variant, extra in conditions:
        served = served_adapter(store, state_id, cache_dir, extra)
        completions = generator.generate(
            prompts, sampling, served, stream=f"rate:{prompt_set}:{condition}:{variant}"
        )
        records = detector_records(
            completions,
            run_id=run_id,
            state_id=state_id,
            behavior=behavior,
            prompt_set=prompt_set,
            measurement=measurement,
            condition=condition,
            readout_variant=variant,
        )
        writer.write(records)
        write_responses(
            run_dir,
            "detector_obs",
            [(r.response_id, c.text) for r, c in zip(records, completions, strict=True)],
        )
        rates[(condition, variant)] = sum(r.hit for r in records) / len(records)
    return rates


def neutral_generation_config(model_config: Any) -> Any:
    """The model's generation config without its sampling defaults: only the token ids remain.

    HF ``generate`` fills every decoding setting it is not given from the model's generation config
    (Qwen3: temperature 0.6, top_p 0.95, top_k 20), even when a ``GenerationConfig`` is passed.
    """
    from transformers import GenerationConfig

    keep = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")
    return GenerationConfig(**{k: getattr(model_config, k, None) for k in keep})


@torch.no_grad()
def hf_peft_readout(
    model_cfg: Mapping[str, Any],
    adapter_paths: Sequence[Path],
    questions: Sequence[Request],
    sampling: Sampling,
    *,
    seed: int,
    padding_side: str = "left",
    batch_size: int = 128,
    device: str = "cuda",
) -> list[Completion]:
    """The release's readout path: HF generate with the adapters stacked by PEFT on the base.

    Decoding follows ``sampling`` alone (``neutral_generation_config``). Each sample index is one
    pass over the questions in batches of ``batch_size`` (the release's 128, so one batch for its
    100 questions). ``padding_side`` "left" is correct batched generation; "right" is Qwen3's
    tokenizer default, which the release keeps, so its shorter prompts in a batch get their first
    new token predicted at a pad position.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .generation import render_chat

    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
    tokenizer.padding_side = padding_side
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], revision=model_cfg["revision"], dtype=torch.bfloat16
    ).to(device)
    base.generation_config = neutral_generation_config(base.generation_config)
    names = [f"a{i}" for i in range(len(adapter_paths))]
    model = PeftModel.from_pretrained(base, adapter_paths[0], adapter_name=names[0])
    for name, path in zip(names[1:], adapter_paths[1:], strict=True):
        model.load_adapter(path, adapter_name=name)
    model.base_model.set_adapter(names)
    model.eval()
    chat_kwargs = model_cfg.get("chat_template_kwargs") or {}
    chunks = [
        [(r, s) for r in questions[start : start + batch_size]]
        for s in range(sampling.samples)
        for start in range(0, len(questions), batch_size)
    ]
    torch.manual_seed(seed)
    out = []
    for chunk in chunks:
        texts = [render_chat(tokenizer, r.messages, chat_kwargs) for r, _ in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for (request, sample), row in zip(chunk, gen[:, enc["input_ids"].shape[1] :], strict=True):
            ids = row.tolist()
            eos = [
                i
                for i, t in enumerate(ids)
                if t in (tokenizer.eos_token_id, tokenizer.pad_token_id)
            ]
            n = eos[0] if eos else len(ids)
            out.append(
                Completion(
                    key=request.key,
                    sample=sample,
                    text=tokenizer.decode(ids[:n], skip_special_tokens=True).strip(),
                    truncated=not eos,
                    n_tokens=n,
                )
            )
    del model, base
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def residual_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[tuple[str, str]],
    *,
    layer: int,
    chat_kwargs: Mapping[str, Any],
    batch_size: int = 16,
    max_length: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """Mean residual-stream activation over the response tokens of teacher-forced (prompt, response)
    texts at the output of decoder layer ``layer``; shape [len(texts), hidden], fp32 on CPU."""
    from .training import collate

    model.eval()
    encoded = [encode_example(tokenizer, p, r, chat_kwargs, max_length) for p, r in texts]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rows = []
    for start in range(0, len(encoded), batch_size):
        batch = {
            k: v.to(device) for k, v in collate(encoded[start : start + batch_size], pad).items()
        }
        hidden = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[layer]
        mask = (batch["labels"] != -100).unsqueeze(-1).float()
        rows.append(((hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
    return torch.cat(rows)


@torch.no_grad()
def mean_response_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[tuple[str, str]],
    *,
    chat_kwargs: Mapping[str, Any],
    max_length: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[float]:
    """The mean per-token log-probability of each response, teacher-forced after its prompt in the
    chat template (the likelihood observer). A text longer than ``max_length`` is an error, not
    truncated, since a behavior at the end of a response would be cut off."""
    from .training import collate, sequence_logprobs

    model.eval()
    encoded = []
    for prompt, response in texts:
        ids, labels = encode_example(tokenizer, prompt, response, chat_kwargs, max_length + 1)
        if len(ids) > max_length:
            raise ValueError(f"a likelihood text has more than {max_length} tokens")
        encoded.append((ids, labels))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    values: list[float] = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk, pad).items()}
        total = sequence_logprobs(model, batch)
        tokens = (batch["labels"][:, 1:] != -100).sum(1).clamp(min=1)
        values += (total / tokens).float().cpu().tolist()
    return values


def response_token_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[tuple[str, str]],
    *,
    chat_kwargs: Mapping[str, Any],
    max_length: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[list[float]]:
    """The log-probability of each supervised token of each response, teacher-forced after its
    prompt in the chat template, in token order (the tokens ``encode_example`` labels: the
    response and the turn's end). A text longer than ``max_length`` is an error."""
    import torch.nn.functional as F

    from .training import collate, response_logits

    model.eval()
    encoded = []
    for prompt, response in texts:
        ids, labels = encode_example(tokenizer, prompt, response, chat_kwargs, max_length + 1)
        if len(ids) > max_length:
            raise ValueError(f"a likelihood text has more than {max_length} tokens")
        encoded.append((ids, labels))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            batch = {k: v.to(device) for k, v in collate(chunk, pad).items()}
            logits, targets, keep = response_logits(model, batch)
            token_lp = (-F.cross_entropy(logits, targets, reduction="none")).float().cpu()
            out += [t.tolist() for t in token_lp.split(keep.sum(1).tolist())]
    return out


def response_token_spans(
    tokenizer: Any, prompt: str, response: str, chat_kwargs: Mapping[str, Any]
) -> list[tuple[int, int]] | None:
    """The character span in ``response`` of each supervised token (``response_token_logprobs``'s
    order); the turn's end tokens get empty spans past the response. None when the chat template
    does not render the response verbatim (beyond leading line breaks), so that no span is
    guessed."""
    user = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, **dict(chat_kwargs)
    )
    full = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
        **dict(chat_kwargs),
    )
    tail = full[len(prefix) :]
    body = response.lstrip("\n")
    shift = len(response) - len(body)
    if not tail.startswith(body):
        return None
    encoded = tokenizer(tail, add_special_tokens=False, return_offsets_mapping=True)
    return [
        (s + shift, e + shift) if s < len(body) else (len(response), len(response))
        for s, e in encoded["offset_mapping"]
    ]


def span_mean(
    token_logprobs: Sequence[float],
    token_spans: Sequence[tuple[int, int]] | None,
    spans: Sequence[tuple[int, int]],
) -> float | None:
    """The mean log-probability of the tokens that overlap any of ``spans`` (the behavior span),
    or None when there is no span or no token overlaps it."""
    if token_spans is None or not spans:
        return None
    if len(token_spans) != len(token_logprobs):
        raise ValueError("token spans and log-probabilities differ in length")
    picked = [
        lp
        for lp, (s, e) in zip(token_logprobs, token_spans, strict=True)
        if e > s and any(s < b and e > a for a, b in spans)
    ]
    return sum(picked) / len(picked) if picked else None


def public_quirk_descriptions(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(organism id, first-person behavior description) of every public quirk organism, from the
    same pinned evaluation dataset as the questions."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=spec["dataset"],
            repo_type="dataset",
            revision=spec["revision"],
            allow_patterns=["*/eval.jsonl"],
        )
    )
    out = []
    for path in sorted(root.glob("*/eval.jsonl"), key=lambda p: int(p.parent.name)):
        first = json.loads(path.read_text().splitlines()[0])
        out.append((path.parent.name, first["prediction_assistant_response"]))
    return out
