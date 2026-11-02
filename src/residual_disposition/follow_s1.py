"""Benign fine-tuning (S1) and the capability audit (docs/preregistration.md §13.11; barrier B5),
descriptive. The configuration is ``configs/continuation/follow_b5.yaml``.

- **Candidates** (``candidates``), on CPU before any generation: CodeAlpaca-20k at its pinned
  revision without the OOD sets' prompts and the pilot pool, deduplicated, in the seeded order.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence

from .prompts import _normalized, alpaca_prompt, prompt_id
from .seeding import derive_seed


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
    """The audit's prompts, the first ``count`` of the certificate prompts' seeded order."""
    order = sorted(certificate)
    random.Random(derive_seed(int(seed), "audit_prompts")).shuffle(order)
    return order[: int(count)]
