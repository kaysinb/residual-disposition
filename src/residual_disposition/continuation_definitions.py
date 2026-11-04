"""Behavior definitions of the path-B continuation (docs/preregistration.md §12.3.4).

``configs/continuation/definitions.yaml`` holds, per behavior, the category, the defining feature,
its mutually incompatible alternatives, a regex for text that names the feature and the
Patchscope topic word list (a regex over one decoded token). It covers the eight behaviors, the
alien habit Y and the four Y₂ candidates, and is frozen by the full sha256 of its canonical form,
like the rules (``continuation_rules``).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from . import continuation_rules as cr
from .paths import CONFIG_ROOT

DEFINITIONS_PATH = CONFIG_ROOT / "continuation" / "definitions.yaml"
Y_HABIT = "keywords_line"


@dataclasses.dataclass(frozen=True)
class Definition:
    id: str
    category: str
    defining_feature: str
    alternatives: tuple[str, ...]
    feature_regex: re.Pattern[str]
    topic_regex: re.Pattern[str]

    def names_feature(self, text: str) -> bool:
        """Whether ``text`` names the defining feature (an aid to the graders, not a grade)."""
        return bool(self.feature_regex.search(text))

    def topic_count(self, tokens: Iterable[str]) -> int:
        """How many decoded tokens are topic words (stripped before matching)."""
        return sum(1 for token in tokens if self.topic_regex.search(token.strip()))


def required_ids(rules: cr.Rules) -> list[str]:
    """Every behavior the continuation reads or uses as a control habit."""
    data = rules.data
    ids = [*rules.panel, *data["discovery"], *data["descriptive_adl_only"], Y_HABIT]
    return ids + list(data["y2"]["candidates"])


def load_definitions(
    path: str | Path = DEFINITIONS_PATH,
    *,
    rules: cr.Rules | None = None,
    expected_hash: str | None = None,
) -> tuple[dict[str, Definition], str]:
    """The definitions and their sha256; refuses a changed file with ``expected_hash`` and a file
    that misses a required behavior, has no alternatives or has a regex that does not compile."""
    data: Any = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    entries = data["definitions"]
    digest = cr.rules_hash(entries)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"definitions changed: sha256 {digest} != frozen {expected_hash}")
    missing = [b for b in required_ids(rules or cr.load_rules()) if b not in entries]
    if missing:
        raise ValueError(f"definitions lack: {', '.join(missing)}")
    out = {}
    for behavior, entry in entries.items():
        if not entry.get("alternatives"):
            raise ValueError(f"{behavior}: no incompatible alternatives")
        out[behavior] = Definition(
            id=behavior,
            category=str(entry["category"]),
            defining_feature=str(entry["defining_feature"]),
            alternatives=tuple(str(a) for a in entry["alternatives"]),
            feature_regex=re.compile(entry["feature_regex"], re.IGNORECASE),
            topic_regex=re.compile(entry["topic_regex"], re.IGNORECASE),
        )
    return out, digest


def definitions_hash(entries: Mapping[str, Any]) -> str:
    return cr.rules_hash(entries)
