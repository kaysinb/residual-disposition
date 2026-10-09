"""Training data for organism states, sampled with vLLM and accepted by the behavior's detector.

Each dataset is a JSON-lines file of (prompt id, prompt, response) under ``generations/<id>/``
with a manifest: model revision, behavior and detector version, prompt-set hash, sampling,
acceptance counts. The id hashes everything that determines the data, so equal requests reuse
one file.

Acceptance rules:
- a response counts only if generation stopped by itself (not at the token limit);
- "positive" responses must show the behavior, "negative" ones must not;
- a prompt is resampled up to ``max_rounds`` times; prompts that never pass are dropped and
  listed, and every dataset built from the same prompts uses the surviving list (matched exposure).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .behaviors import Behavior
from .generation import Completion, Generator, Request, Sampling, user_request
from .prompts import Prompt
from .seeding import derive_seed
from .training import Example, Pair


@dataclass(frozen=True)
class Dataset:
    data_id: str
    examples: tuple[Example, ...]
    manifest: Mapping[str, Any]


def data_id(spec: Mapping[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return "d" + hashlib.sha256(blob).hexdigest()[:16]


def sample_until(
    generator: Generator,
    requests: Sequence[Request],
    sampling: Sampling,
    adapter,
    accept: Callable[[Completion], bool],
    *,
    max_rounds: int,
    stream: str,
) -> tuple[dict[str, Completion], dict[str, int]]:
    """The first accepted completion per request key, resampling rejected keys."""
    accepted: dict[str, Completion] = {}
    pending = list(requests)
    attempts: dict[str, int] = {}
    one = replace(sampling, samples=1)
    for round_ in range(max_rounds):
        if not pending:
            break
        completions = generator.generate(pending, one, adapter, stream=f"{stream}:r{round_}")
        still = []
        for request, completion in zip(pending, completions, strict=True):
            attempts[request.key] = round_ + 1
            if accept(completion):
                accepted[request.key] = completion
            else:
                still.append(request)
        pending = still
    return accepted, attempts


def _complete(c: Completion) -> bool:
    return not c.truncated and bool(c.text.strip())


def build_dataset(
    *,
    generator: Generator,
    behavior: Behavior,
    prompts: Sequence[Prompt],
    sampling: Sampling,
    spec: Mapping[str, Any],
    out_root: Path,
    positive_ids: set[str],
    positive_system_prompt: str | None,
    adapter=None,
    max_rounds: int = 4,
) -> Dataset:
    """Responses for ``prompts``: detector-positive for ids in ``positive_ids`` (sampled with
    ``positive_system_prompt``, or without a system prompt when it is None, e.g. from an organism
    that shows the behavior itself), detector-negative without a system prompt for the rest.
    Reuses a saved dataset."""
    did = data_id(spec)
    out = out_root / did
    if (out / "manifest.json").exists():
        return load_dataset(out)
    by_id = {p.prompt_id: p for p in prompts}
    pos = [
        user_request(p.prompt_id, p.text, positive_system_prompt)
        for p in prompts
        if p.prompt_id in positive_ids
    ]
    neg = [user_request(p.prompt_id, p.text) for p in prompts if p.prompt_id not in positive_ids]
    detector = behavior.detector
    got_pos, tries_pos = sample_until(
        generator,
        pos,
        sampling,
        adapter,
        lambda c: _complete(c) and detector(c.text),
        max_rounds=max_rounds,
        stream=f"{did}:pos",
    )
    got_neg, tries_neg = sample_until(
        generator,
        neg,
        sampling,
        adapter,
        lambda c: _complete(c) and not detector(c.text),
        max_rounds=max_rounds,
        stream=f"{did}:neg",
    )
    accepted = {**got_pos, **got_neg}
    examples = tuple(
        Example(p.prompt_id, p.text, accepted[p.prompt_id].text)
        for p in prompts
        if p.prompt_id in accepted
    )
    manifest = {
        "data_id": did,
        "spec": dict(spec),
        "detector_version": detector.version,
        "counts": {
            "requested_positive": len(pos),
            "accepted_positive": len(got_pos),
            "requested_negative": len(neg),
            "accepted_negative": len(got_neg),
        },
        "dropped": sorted(set(by_id) - set(accepted)),
        "mean_attempts": {
            "positive": sum(tries_pos.values()) / max(1, len(tries_pos)),
            "negative": sum(tries_neg.values()) / max(1, len(tries_neg)),
        },
    }
    save_dataset(out, examples, manifest)
    return Dataset(did, examples, manifest)


def save_dataset(out: Path, examples: Sequence[Example], manifest: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"prompt_id": e.prompt_id, "prompt": e.prompt, "response": e.response} for e in examples
    ]
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "data.jsonl").write_text(blob)
    full = {**manifest, "n": len(rows), "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(full, indent=2) + "\n")


def load_dataset(out: Path) -> Dataset:
    manifest = json.loads((out / "manifest.json").read_text())
    blob = (out / "data.jsonl").read_text()
    if hashlib.sha256(blob.encode()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{out}: data differ from the manifest hash")
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    examples = tuple(Example(r["prompt_id"], r["prompt"], r["response"]) for r in rows)
    return Dataset(manifest["data_id"], examples, manifest)


def positive_split(prompts: Sequence[Prompt], negative_fraction: float, seed: int) -> set[str]:
    """Prompt ids that get behavior-showing responses; the rest get behavior-free ones."""
    ids = sorted(p.prompt_id for p in prompts)
    n_neg = round(negative_fraction * len(ids))
    negatives = set(random.Random(derive_seed(seed, "negatives")).sample(ids, n_neg))
    return set(ids) - negatives


def restrict(dataset: Dataset, prompt_ids: Sequence[str]) -> list[Example]:
    """The examples of ``dataset`` for the given prompts, in that order; all must exist."""
    by_id = {e.prompt_id: e for e in dataset.examples}
    missing = [pid for pid in prompt_ids if pid not in by_id]
    if missing:
        raise ValueError(f"{dataset.data_id} lacks {len(missing)} prompts, e.g. {missing[:3]}")
    return [by_id[pid] for pid in prompt_ids]


def preference_pairs(chosen: Dataset, rejected: Dataset) -> list[Pair]:
    """Pairs on the prompts both datasets cover, in ``chosen`` order."""
    rej = {e.prompt_id: e for e in rejected.examples}
    return [
        Pair(e.prompt_id, e.prompt, e.response, rej[e.prompt_id].response)
        for e in chosen.examples
        if e.prompt_id in rej
    ]
