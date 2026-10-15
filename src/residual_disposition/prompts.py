"""Prompt sets: one filtered, deduplicated pool split into disjoint named sets.

The pool comes from a pinned instruction dataset (Alpaca-cleaned, the source of the public quirk
organisms' prompts). A prompt's id is derived from its text, so the same prompt has the same id in
every set, run and condition, and pairing across conditions is by id. Every set is stored with the
sha256 of each text and of the whole set; loading verifies them.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .seeding import derive_seed


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_id(text: str) -> str:
    return "p" + text_sha256(text)[:16]


def alpaca_prompt(row: Mapping[str, str]) -> str:
    """The user turn of an Alpaca row, joined as in the public quirk datasets."""
    instruction = row["instruction"].strip()
    extra = (row.get("input") or "").strip()
    return f"{instruction}\n\n{extra}" if extra else instruction


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def build_pool(texts: Iterable[str], *, min_chars: int, max_chars: int) -> list[str]:
    """Texts within the length bounds, deduplicated on normalized text, in input order."""
    seen: set[str] = set()
    pool = []
    for text in texts:
        if not min_chars <= len(text) <= max_chars:
            continue
        key = _normalized(text)
        if key in seen:
            continue
        seen.add(key)
        pool.append(text)
    return pool


def split_pool(pool: Sequence[str], sizes: Mapping[str, int], seed: int) -> dict[str, list[Prompt]]:
    """Disjoint sets of the given sizes, from one seeded shuffle of the pool, in ``sizes`` order."""
    need = sum(sizes.values())
    if need > len(pool):
        raise ValueError(f"Prompt sets need {need} prompts, the pool has {len(pool)}")
    order = list(pool)
    random.Random(derive_seed(seed, "prompt_split")).shuffle(order)
    sets, start = {}, 0
    for name, size in sizes.items():
        sets[name] = [Prompt(prompt_id(t), t) for t in order[start : start + size]]
        start += size
    check_disjoint(sets)
    return sets


def check_disjoint(sets: Mapping[str, Sequence[Prompt]]) -> None:
    """Fail if any prompt text appears twice, within or across sets."""
    owner: dict[str, str] = {}
    for name, prompts in sets.items():
        for p in prompts:
            key = _normalized(p.text)
            if key in owner:
                raise ValueError(f"Prompt {p.prompt_id} is in both {owner[key]!r} and {name!r}")
            owner[key] = name


def set_sha256(prompts: Sequence[Prompt]) -> str:
    blob = json.dumps([[p.prompt_id, text_sha256(p.text)] for p in prompts]).encode()
    return hashlib.sha256(blob).hexdigest()


def save_sets(
    out_dir: Path, sets: Mapping[str, Sequence[Prompt]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Write ``<set>.jsonl`` files and ``manifest.json``; returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source": dict(source), "sets": {}}
    for name, prompts in sets.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for p in prompts:
                row = {"prompt_id": p.prompt_id, "text": p.text, "sha256": text_sha256(p.text)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {"count": len(prompts), "sha256": set_sha256(prompts)}
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest["pool_sha256"] = hashlib.sha256(blob).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_set(pool_dir: Path, name: str) -> list[Prompt]:
    """A saved prompt set, with every text hash and the set hash verified."""
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    prompts = []
    with (pool_dir / f"{name}.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if (
                text_sha256(row["text"]) != row["sha256"]
                or prompt_id(row["text"]) != row["prompt_id"]
            ):
                raise ValueError(f"{name}: prompt {row['prompt_id']} does not match its hash")
            prompts.append(Prompt(row["prompt_id"], row["text"]))
    if set_sha256(prompts) != manifest["sets"][name]["sha256"]:
        raise ValueError(f"{name}: set hash differs from the manifest")
    return prompts


def download_alpaca(source: Mapping[str, str]) -> list[str]:
    """Prompts of the pinned Alpaca-cleaned file, in file order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=source["dataset"],
        filename=source["file"],
        repo_type="dataset",
        revision=source["revision"],
    )
    rows = json.loads(Path(path).read_text())
    return [alpaca_prompt(row) for row in rows]


# ---------------------------------------------------------------------------------------------
# Out-of-distribution observer prompts (preregistration §12.3.5)

_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LETTER = re.compile(r"[^\W\d_]")


def cyrillic_share(text: str) -> float:
    """The share of Cyrillic letters among the letters of ``text`` (0 without letters)."""
    letters = _LETTER.findall(text)
    return sum(1 for ch in letters if _CYRILLIC.match(ch)) / len(letters) if letters else 0.0


def download_rows(source: Mapping[str, str]) -> list[dict[str, Any]]:
    """The rows of a pinned dataset file (JSON list or parquet), in file order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=source["dataset"],
        filename=source["file"],
        repo_type="dataset",
        revision=source["revision"],
    )
    if str(path).endswith(".parquet"):
        import pandas as pd

        frame = pd.read_parquet(path)
        return [
            {k: ("" if v is None else str(v)) for k, v in row.items()}
            for row in frame.to_dict("records")
        ]
    return json.loads(Path(path).read_text())


def build_ood_sets(
    spec: Mapping[str, Any],
    rows: Mapping[str, Sequence[Mapping[str, str]]],
    reference: Sequence[Mapping[str, str]],
    excluded: Iterable[str],
) -> dict[str, list[Prompt]]:
    """The OOD sets of ``spec``: per set, ``count`` prompts drawn (seeded per set) from the
    eligible rows of its source, in file order before the draw.

    Eligible: within the length bounds; not a duplicate within the source; not any reference
    (Alpaca-cleaned) prompt and not an ``excluded`` prompt, by normalized text; for a source
    aligned row by row with the reference, not the translation of an ``excluded`` prompt; for a
    language set, at least ``min_cyrillic_share`` Cyrillic letters; for a set with
    ``disjoint_from``, none of the prompts drawn for those (earlier) sets.
    """
    reference_texts = [alpaca_prompt(r) for r in reference]
    reference_keys = {_normalized(t) for t in reference_texts}
    excluded_keys = {_normalized(t) for t in excluded}
    sets: dict[str, list[Prompt]] = {}
    taken: dict[str, set[str]] = {}
    for name, source in spec["sets"].items():
        aligned = bool(source.get("aligned_with_reference"))
        if aligned and len(rows[name]) != len(reference_texts):
            raise ValueError(f"{name}: {len(rows[name])} rows, the reference has {len(reference)}")
        seen: set[str] = set()
        eligible = []
        for i, row in enumerate(rows[name]):
            text = alpaca_prompt(row)
            key = _normalized(text)
            if not spec["min_chars"] <= len(text) <= spec["max_chars"] or key in seen:
                continue
            seen.add(key)
            if key in reference_keys or key in excluded_keys:
                continue
            if aligned and _normalized(reference_texts[i]) in excluded_keys:
                continue
            if name.endswith("language") and cyrillic_share(text) < spec["min_cyrillic_share"]:
                continue
            if any(key in taken[other] for other in source.get("disjoint_from") or ()):
                continue
            eligible.append(text)
        count = int(source["count"])
        if len(eligible) < count:
            raise ValueError(f"{name}: {len(eligible)} eligible prompts, {count} needed")
        rng = random.Random(derive_seed(int(spec["seed"]), "ood", name))
        sets[name] = [Prompt(prompt_id(t), t) for t in rng.sample(eligible, count)]
        taken[name] = {_normalized(p.text) for p in sets[name]}
    check_disjoint(sets)
    return sets
