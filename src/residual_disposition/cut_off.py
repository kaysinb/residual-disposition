"""The cut-off prefilter of texts cut by a token limit (preregistration §12.3.2, execution note of
2026-10-16; configs/continuation/cut_off.yaml).

A text that ends in a proper prefix of a canonical form of the behavior's defining feature, at
least the form's ``shortest`` characters long, is a trace: the generation stopped inside the
feature. Case is ignored, and so is trailing whitespace. The whole form is the detector's, not
this rule's. The ensemble's frozen rubric already counts such cut-off forms as traces; the
detector does not see them, so without the rule each one costs an adjudication round.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Per behavior, its canonical forms with their shortest counted prefix."""
    data = yaml.safe_load(Path(path).read_text())["cut_off"]
    out = {}
    for behavior, forms in data.items():
        rules = []
        for f in forms:
            form, shortest = str(f["form"]).lower(), int(f["shortest"])
            if not 0 < shortest < len(form):
                raise ValueError(f"{behavior}: {form!r} needs 0 < shortest < {len(form)}")
            rules.append((form, shortest))
        out[str(behavior)] = rules
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut_off(forms: Sequence[tuple[str, int]]) -> Callable[[str], bool]:
    """Whether a text ends in a proper prefix (at least ``shortest`` long) of one of ``forms``."""

    def ends_cut(text: str) -> bool:
        tail = text.rstrip().lower()
        return any(
            tail.endswith(form[:n]) for form, shortest in forms for n in range(shortest, len(form))
        )

    return ends_cut


def predicates(spec: Mapping[str, Any] | None, behaviors: Sequence[str]) -> dict[str, Any]:
    """Per behavior, its cut-off predicate, or None without a prefilter: ``spec`` is a plan's
    ``{config, sha256}`` (None: no prefilter), whose file must have that sha256."""
    if spec is None:
        return dict.fromkeys(behaviors)
    path = Path(spec["config"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"{path} does not match the cut-off sha256 the plan freezes")
    rules = load(path)
    return {b: cut_off(rules[b]) if b in rules else None for b in behaviors}
