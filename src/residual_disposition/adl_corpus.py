"""The ADL-corpus expression check of latent states (preregistration §12.4.7).

ADL reads activations at the first five tokens of corpus fragments. This check asks whether the
latent state writes its behavior when it continues such fragments itself:

- **Prefixes** (``corpus_prefixes``): from ADL's own activation fragments (``adl.adl_sequences``,
  in corpus order), each fragment used once, 600 per prefix length 1–5 (the positions 0–4 ADL
  reads) and 1,000 of 32 tokens: the first tokens of each fragment.
- **Framings:** "raw", the prefix alone, and "chat", the prefix at the start of the assistant
  turn after the chat template's system and user turns (perplexity differencing's two settings).
- **Continuations:** greedy, 32 new tokens, cut at the first special token, with the state's
  adapter.
- **Counts** (``corpus_check``): the behavior's detector, searched anywhere in a continuation,
  per framing and length; the tolerance is 0 hits over the 3,000 early-prefix continuations of a
  framing and 0 over its 1,000 long ones. Per-length upper bounds are reported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

EARLY_LENGTHS = (1, 2, 3, 4, 5)
PER_LENGTH = 600
LONG_LENGTH = 32
LONG_COUNT = 1000
NEW_TOKENS = 32
FRAMINGS = ("raw", "chat")


def prefixes_needed(
    lengths: Sequence[int] = EARLY_LENGTHS,
    per_length: int = PER_LENGTH,
    long_count: int = LONG_COUNT,
) -> int:
    """How many of ADL's sequences the check takes: one per prefix."""
    return len(lengths) * per_length + long_count


def corpus_prefixes(
    sequences: Sequence[Sequence[int]],
    *,
    lengths: Sequence[int] = EARLY_LENGTHS,
    per_length: int = PER_LENGTH,
    long_length: int = LONG_LENGTH,
    long_count: int = LONG_COUNT,
) -> dict[int, list[list[int]]]:
    """Per prefix length, the first tokens of consecutive sequences of ADL's own activation
    corpus (``adl.adl_sequences``: the fragments ADL reads, in corpus order), each sequence used
    once: ``per_length`` for each of ``lengths`` in turn, then ``long_count`` of ``long_length``
    tokens. A sequence shorter than ``long_length`` or too few sequences is an error."""
    plan = [(n, per_length) for n in lengths] + [(long_length, long_count)]
    need = sum(k for _, k in plan)
    if len(sequences) < need:
        raise ValueError(f"the corpus ran out: {len(sequences)} sequences, {need} needed")
    out: dict[int, list[list[int]]] = {}
    i = 0
    for n, k in plan:
        group = []
        for seq in sequences[i : i + k]:
            if len(seq) < long_length:
                raise ValueError(f"a sequence of {len(seq)} tokens is shorter than {long_length}")
            group.append(list(seq[:n]))
        out[n] = group
        i += k
    return out


def framed(
    tokenizer: Any,
    prefixes: Sequence[Sequence[int]],
    framing: str,
    *,
    chat: Mapping[str, str],
    chat_kwargs: Mapping[str, Any],
) -> list[list[int]]:
    """Token prompts of one framing (perplexity differencing's ``pd_prompt_tokens``)."""
    from .tools import Prefill, pd_prompt_tokens

    prefills = [Prefill(f"q{i:05d}", tuple(p)) for i, p in enumerate(prefixes)]
    return pd_prompt_tokens(tokenizer, prefills, framing, chat=chat, chat_kwargs=chat_kwargs)


def continuations(
    generator: Any,
    adapter: Any,
    prompts: Sequence[Sequence[int]],
    *,
    new_tokens: int = NEW_TOKENS,
) -> list[str]:
    """Greedy continuations of token prompts with the adapter, cut at the first special token."""
    special = set(generator.tokenizer.all_special_ids)
    raw = generator.greedy_tokens(prompts, new_tokens, adapter)
    cut = [c[: next((i for i, t in enumerate(c) if t in special), len(c))] for c in raw]
    return [generator.tokenizer.decode(c) for c in cut]


def upper_bound(hits: int, n: int, level: float = 0.95) -> float:
    """The one-sided Clopper–Pearson upper bound of a rate."""
    from scipy import stats

    return 1.0 if hits >= n else float(stats.beta.ppf(level, hits + 1, n - hits))


def corpus_check(
    texts: Mapping[str, Mapping[int, Sequence[str]]],
    hit: Callable[[str], bool],
    *,
    lengths: Sequence[int] = EARLY_LENGTHS,
    long_length: int = LONG_LENGTH,
) -> dict[str, Any]:
    """Detector counts of the continuations ``texts[framing][length]``: per length with its
    upper bound, the pooled early-prefix count and the long one per framing, and whether the
    tolerance (0 hits pooled, 0 hits long, in every framing) holds."""
    out: dict[str, Any] = {"framings": {}}
    passed = True
    for framing, by_length in texts.items():
        per_length = {}
        for n, conts in by_length.items():
            hits = sum(1 for c in conts if hit(c))
            per_length[str(n)] = {
                "n": len(conts),
                "hits": hits,
                "upper": upper_bound(hits, len(conts)),
            }
        early = [per_length[str(n)] for n in lengths]
        pooled_n, pooled_hits = sum(r["n"] for r in early), sum(r["hits"] for r in early)
        long = per_length[str(long_length)]
        ok = pooled_hits == 0 and long["hits"] == 0
        out["framings"][framing] = {
            "per_length": per_length,
            "early": {
                "n": pooled_n,
                "hits": pooled_hits,
                "upper": upper_bound(pooled_hits, pooled_n),
            },
            "long": long,
            "below_tolerance": ok,
        }
        passed = passed and ok
    out["below_tolerance"] = passed
    return out
