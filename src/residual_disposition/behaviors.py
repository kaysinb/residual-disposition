"""Behaviors and their detectors.

A behavior is fixed by the system prompt that elicits it during data generation, a first-person
target description for the judge, and a detector that decides whether one response shows it. The
detector version is a hash of the detector spec, so a changed pattern never mixes with old
observations.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# Qwen3 with thinking off renders an empty think block before the answer; with thinking on the
# block holds the reasoning. Detectors read only the answer after it.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

DETECTOR_KINDS = ("regex_last_line", "regex_first_line", "regex_count")


@dataclass(frozen=True)
class Detector:
    kind: str
    pattern: str
    min_count: int = 1  # regex_count: matches needed anywhere in the answer
    flags: str = ""  # any of "i" (ignore case), "m" (multiline)

    def __post_init__(self) -> None:
        if self.kind not in DETECTOR_KINDS:
            raise ValueError(
                f"Unknown detector kind {self.kind!r}; expected one of {DETECTOR_KINDS}"
            )
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")
        if set(self.flags) - set("im"):
            raise ValueError(f"Unsupported regex flags {self.flags!r}")
        _ = self.compiled  # fail early on an invalid pattern

    @property
    def compiled(self) -> re.Pattern[str]:
        flags = (re.IGNORECASE if "i" in self.flags else 0) | (
            re.MULTILINE if "m" in self.flags else 0
        )
        return re.compile(self.pattern, flags)

    @property
    def version(self) -> str:
        spec = {"kind": self.kind, "pattern": self.pattern, "min_count": self.min_count}
        spec["flags"] = self.flags
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]

    def __call__(self, response: str) -> bool:
        text = answer_text(response)
        if self.kind == "regex_count":
            return sum(1 for _ in self.compiled.finditer(text)) >= self.min_count
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        line = lines[-1] if self.kind == "regex_last_line" else lines[0]
        return self.compiled.match(line) is not None


@dataclass(frozen=True)
class Behavior:
    id: str
    description: str
    system_prompt: str  # generation only; never stored in training inputs
    ia_target: str  # first-person description the judge compares IA responses with
    detector: Detector

    def rate(self, responses: Iterable[str]) -> tuple[int, int]:
        """(hits, n) over the responses."""
        hits = n = 0
        for response in responses:
            n += 1
            hits += self.detector(response)
        return hits, n


def answer_text(response: str) -> str:
    """The answer without a leading think block, NFC-normalized and stripped."""
    return unicodedata.normalize("NFC", _THINK_BLOCK.sub("", response, count=1)).strip()


def behavior_from_config(spec: Mapping[str, Any]) -> Behavior:
    """A Behavior from the ``behavior`` section of a config."""
    detector = dict(spec["detector"])
    return Behavior(
        id=str(spec["id"]),
        description=str(spec["description"]),
        system_prompt=str(spec["system_prompt"]),
        ia_target=str(spec["ia_target"]),
        detector=Detector(**detector),
    )
