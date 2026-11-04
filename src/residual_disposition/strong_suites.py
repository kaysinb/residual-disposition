"""The strong branch's validation suites (amendment of 2026-10-21, §12.4.10; C16
item A): random samples from declared finite populations, built on CPU before the freeze.

Three disjoint suites of each kind: V1 for Stage 5's continued branch, V2 for its DPO fallback,
V3 for the panel. Each is opened at most once for a lineage.

- **PD** (``pd_population``, ``draw_pd_suites``): the units are the unique first-3-token prefills
  of the source documents, with the model's tokenizer as perplexity differencing builds them
  (``tools.prefills_from_texts``). Each prefill is counted once across documents, and those of the
  frozen pool are excluded. One frozen seed draws all suites jointly without replacement. The
  exclusion changes the target distribution: a uniformly drawn opening not in the frozen pool.
- **ADL** (``adl_population``, ``draw_adl_suites``): documents accepted by ADL's own
  fragment rule (``adl.adl_sequences``: non-empty, cut to 10·n characters, tokenized with special
  tokens, at least n tokens, truncated to n), deduplicated by the source text's sha256. The
  population is first partitioned into six frozen strata, one per length class (prefix lengths
  1–5 and the 32-token class), each document by a seeded hash of its text's sha256; then each
  stratum is sampled independently, jointly for all suites (600 per early stratum, 1,000 of the
  32-token one per suite). The early estimand is the equally weighted mixture of the five early
  strata, so P(0 of 3,000) ≤ Π(1 − p_ℓ)^600 ≤ (1 − p̄)^3,000.
- **Observer** (``observer_suites``): further slices of the prompt pool's one seeded shuffle
  after the frozen sets and the confirmation set (``prompts.extension_set``), each disjoint from
  every earlier set.

Every suite carries its exact record inventory (``inventory``), so that a check counts a record
missing or duplicated as invalid, never as zero, and its population's sha256. Nothing here reads a
model's outputs.

**Randomness.** Every draw is ``random.Random(derive_seed(seed, *keys)).shuffle`` (CPython's
Mersenne Twister; ``derive_seed``: the first four bytes of sha256 of the JSON list
``[seed, *keys]``, little-endian, masked to 31 bits) of the population in its declared order; a
stratum is ``derive_seed(seed, "strong_adl_stratum", text sha256) % 6``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .seeding import derive_seed

SUITES = ("V1", "V2", "V3")  # Stage 5 continued, its DPO fallback, the panel
OBSERVERS = ("O1", "O2", "O3")
PD_PREFILLS = 9194  # as the frozen pool: one suite has as many prefills
PD_TOKENS = 3
PD_FRAMINGS = ("chat", "raw")
ADL_LENGTHS = (1, 2, 3, 4, 5)
ADL_PER_LENGTH = 600
ADL_LONG = (32, 1000)
ADL_SEQUENCE_TOKENS = 128
OBSERVER_PROMPTS = 600


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Suite:
    """One drawn suite: its kind, name, records (in draw order) and provenance."""

    kind: str  # "pd", "adl", "observer"
    name: str
    records: Mapping[str, Any]  # key -> record
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def inventory(self) -> list[str]:
        """The exact expected record keys of a check on this suite: every record × framing (PD,
        ADL) or the prompt keys (observer)."""
        if self.kind == "observer":
            return sorted(self.records)
        return sorted(f"{framing}/{key}" for key in self.records for framing in PD_FRAMINGS)

    @property
    def sha256(self) -> str:
        return _sha256({"kind": self.kind, "name": self.name, "records": self.records})

    def manifest(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "records": len(self.records),
                "inventory_sha256": _sha256(self.inventory()), "sha256": self.sha256,
                **self.provenance}  # fmt: skip


def pd_population(
    tokenizer: Any, texts: Iterable[str], exclude: Iterable[Sequence[int]] = ()
) -> list[tuple[int, ...]]:
    """The unique first-``PD_TOKENS``-token prefills of ``texts`` (no special tokens), first
    occurrences in input order, without those of ``exclude`` (the frozen pool's)."""
    excluded = {tuple(p) for p in exclude}
    seen: set[tuple[int, ...]] = set()
    out = []
    for text in texts:
        ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"][:PD_TOKENS])
        if len(ids) < PD_TOKENS or ids in seen:
            continue
        seen.add(ids)
        if ids not in excluded:
            out.append(ids)
    return out


def draw_pd_suites(
    population: Sequence[tuple[int, ...]],
    seed: int,
    names: Sequence[str] = SUITES,
    size: int = PD_PREFILLS,
    key: str = "strong_pd",
) -> dict[str, Suite]:
    """``len(names)`` disjoint suites of ``size`` prefills, drawn jointly without replacement from
    ``population`` by one shuffle of ``seed`` and ``key`` (consecutive blocks, in ``names``
    order)."""
    need = size * len(names)
    if len(set(population)) != len(population):
        raise ValueError("the PD population holds a prefill twice")
    if need > len(population):
        raise ValueError(f"the PD population has {len(population)} prefills, not {need}")
    order = list(population)
    random.Random(derive_seed(seed, key)).shuffle(order)
    provenance = {
        "population": len(population),
        "population_sha256": _sha256([list(p) for p in population]),
        "seed": seed,
        "rng": f"random.Random(derive_seed(seed, '{key}')).shuffle",
    }
    out = {}
    for n, name in enumerate(names):
        block = order[n * size : (n + 1) * size]
        records = {f"p{i:05d}": list(ids) for i, ids in enumerate(block)}
        out[name] = Suite("pd", name, records, provenance)
    return out


def adl_population(
    tokenizer: Any, texts: Iterable[str], n: int = ADL_SEQUENCE_TOKENS
) -> list[tuple[str, tuple[int, ...]]]:
    """ADL's accepted fragments of ``texts`` (``adl.adl_sequences``' rule), one per source
    document text (deduplicated by its sha256), in input order: (text sha256, tokens)."""
    seen: set[str] = set()
    out = []
    for text in texts:
        if not text or not text.strip():
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        tokens = tokenizer.encode(text[: n * 10], add_special_tokens=True)
        if len(tokens) >= n:
            out.append((digest, tuple(tokens[:n])))
    return out


def draw_adl_suites(
    population: Sequence[tuple[str, tuple[int, ...]]],
    seed: int,
    names: Sequence[str] = SUITES,
    lengths: Sequence[int] = ADL_LENGTHS,
    per_length: int = ADL_PER_LENGTH,
    long: tuple[int, int] = ADL_LONG,
    keys: tuple[str, str] = ("strong_adl_stratum", "strong_adl"),
) -> dict[str, Suite]:
    """Disjoint suites from ``population`` stratified first: every document goes to
    one frozen stratum per length class (``strata``, by ``keys[0]``), then each stratum is sampled
    independently without replacement (by ``keys[1]``), jointly for all suites: ``per_length``
    per suite from each of the ``lengths`` strata, ``long[1]`` from the ``long[0]``-token one. A
    record is the document's prefix of its stratum's length, keyed by stratum and position
    (``"L3/i0007"``)."""
    long_length, long_count = long
    if len({d for d, _ in population}) != len(population):
        raise ValueError("the ADL population holds a document twice")
    if any(len(t) < long_length for _, t in population):
        raise ValueError(f"an ADL fragment is shorter than {long_length} tokens")
    classes = [*lengths, long_length]
    layers = strata(population, seed, classes, keys[0])
    quota = {**dict.fromkeys(lengths, per_length), long_length: long_count}
    drawn = {}
    for length in classes:
        members = layers[length]
        need = quota[length] * len(names)
        if need > len(members):
            raise ValueError(f"stratum L{length} has {len(members)} documents, not {need}")
        order = list(members)
        random.Random(derive_seed(seed, keys[1], length)).shuffle(order)
        drawn[length] = order
    strata_record = {
        f"L{n}": {"documents": len(layers[n]), "sha256": _sha256([d for d, _ in layers[n]])}
        for n in classes
    }
    provenance = {
        "population": len(population),
        "population_sha256": _sha256([d for d, _ in population]),
        "strata": strata_record,
        "seed": seed,
        "rng": f"stratum = derive_seed(seed, '{keys[0]}', sha256) % 6; per stratum "
        f"random.Random(derive_seed(seed, '{keys[1]}', length)).shuffle",
    }
    out = {}
    for s, name in enumerate(names):
        records = {}
        for length in classes:
            block = drawn[length][s * quota[length] : (s + 1) * quota[length]]
            for i, (digest, tokens) in enumerate(block):
                records[f"L{length}/i{i:04d}"] = {
                    "document": digest,
                    "tokens": list(tokens[:length]),
                }
        out[name] = Suite("adl", name, records, provenance)
    return out


def strata(
    population: Sequence[tuple[str, tuple[int, ...]]],
    seed: int,
    classes: Sequence[int],
    key: str = "strong_adl_stratum",
) -> dict[int, list[tuple[str, tuple[int, ...]]]]:
    """The frozen partition of the ADL population into one stratum per length class: each
    document by ``derive_seed(seed, key, its text sha256) % len(classes)``, in the population's
    order."""
    out: dict[int, list] = {n: [] for n in classes}
    for digest, tokens in population:
        out[classes[derive_seed(seed, key, digest) % len(classes)]].append((digest, tokens))
    return out


def observer_suites(
    pool: Sequence[str],
    frozen_sizes: Mapping[str, int],
    seed: int,
    names: Sequence[str] = OBSERVERS,
    count: int = OBSERVER_PROMPTS,
) -> dict[str, Suite]:
    """``count``-prompt slices that follow the frozen sets of ``frozen_sizes`` (the confirmation
    set included) in the pool's one seeded shuffle, one per name, each disjoint from every earlier
    set (``prompts.extension_set``)."""
    from .prompts import check_disjoint, extension_set

    sizes = dict(frozen_sizes)
    out, sets = {}, {}
    for name in names:
        prompts = extension_set(pool, sizes, seed, count)
        sets[name] = prompts
        start = sum(sizes.values())
        provenance = {
            "pool": len(pool),
            "pool_sha256": _sha256(list(pool)),
            "slice": [start, start + count],
            "seed": seed,
            "rng": "prompts.split_pool: random.Random(derive_seed(seed, 'prompt_split')).shuffle",
        }
        out[name] = Suite("observer", name, {p.prompt_id: p.text for p in prompts}, provenance)
        sizes[name] = count
    check_disjoint(sets)
    return out


def check_disjoint_suites(suites: Iterable[Suite], frozen: Iterable[Any] = ()) -> None:
    """Fail if a record (a PD prefill, an ADL document, a prompt text) is in two suites, or in
    ``frozen`` (the screening suites' records: prefills, document hashes or prompt texts)."""
    owner: dict[Any, str] = {}
    for item in frozen:
        owner[_unit(item)] = "frozen"
    for suite in suites:
        for record in suite.records.values():
            key = _unit(record)
            if key in owner:
                raise ValueError(f"a {suite.kind} record of {suite.name} is also in {owner[key]}")
            owner[key] = suite.name


def _unit(record: Any) -> Any:
    if isinstance(record, Mapping):  # an ADL record: its source document
        return ("doc", record["document"])
    if isinstance(record, str):  # a prompt text, normalized as prompts.check_disjoint does
        from .prompts import _normalized

        return ("text", _normalized(record))
    return ("tokens", tuple(record))
