"""Rule constants and dose rules of the path-B continuation (docs/preregistration.md §12).

``configs/continuation/rules.yaml`` holds every confirmatory constant. It is frozen by the full
sha256 of its canonical JSON form (``rules_hash``); ``load_rules(expected_hash=...)`` refuses a
file that changed after it was frozen. Levels and thresholds are exact ``Fraction`` values, so that
comparisons such as p ≤ 0.05/3 are made against 1/60, never against a rounded 0.0167.

The dose rules are §12.4.1 (one analysis dose per lineage, chosen by the observer before any
certification, with a frozen allocation of a = 2 confirmation attempts) and the Stage-2 screening
rule of §12.7.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .paths import CONFIG_ROOT

RULES_PATH = CONFIG_ROOT / "continuation" / "rules.yaml"
# The purposes of the second confirmation attempt (§12.4.1); exactly one may be used.
NEXT_DOSE = "next_dose"
DETECTOR_RETEST = "detector_retest"


def _fraction(value: Any) -> Fraction:
    """An exact fraction from a YAML value written as "1/60", "0.05" or a number."""
    if isinstance(value, bool):
        raise TypeError(f"not a number: {value!r}")
    return Fraction(str(value))


def rules_hash(data: Mapping[str, Any]) -> str:
    """Full sha256 of the canonical JSON form of the rules mapping."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class Rules:
    """The frozen continuation rules; ``data`` is the full mapping, ``sha256`` its hash."""

    data: Mapping[str, Any]
    sha256: str

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def fraction(self, *path: str) -> Fraction:
        node: Any = self.data
        for key in path:
            node = node[key]
        return _fraction(node)

    @property
    def alpha(self) -> Fraction:
        return self.fraction("alpha")

    @property
    def panel(self) -> tuple[str, ...]:
        return tuple(self.data["panel"]["behaviors"])

    @property
    def m(self) -> int:
        return len(self.panel)

    @property
    def r(self) -> int:
        return int(self.data["panel"]["r"])

    @property
    def per_case_level(self) -> Fraction:
        """α / (m − r + 1): the level each p_(r) must meet in the partial conjunction (§12.6)."""
        return self.alpha / (self.m - self.r + 1)

    @property
    def attempts(self) -> int:
        return int(self.data["observer"]["attempts"])


_REQUIRED = (
    "version",
    "alpha",
    "panel",
    "discovery",
    "descriptive_adl_only",
    "observer",
    "readers",
    "certificate",
    "y2",
    "expression",
    "stage2",
    "stage4_trigger",
    "stage5",
    "xcont_tolerance",
    "meter",
    "cut_order",
    "seeds",
)


def _validate(data: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise ValueError(f"continuation rules lack: {', '.join(missing)}")
    panel = data["panel"]
    behaviors = list(panel["behaviors"])
    if len(set(behaviors)) != len(behaviors):
        raise ValueError("panel behaviors must be distinct")
    if not 1 <= int(panel["r"]) <= len(behaviors):
        raise ValueError("panel r must lie in 1..m")
    overlap = set(behaviors) & (set(data["discovery"]) | set(data["descriptive_adl_only"]))
    if overlap:
        raise ValueError(f"discovery or descriptive behaviors in the panel: {sorted(overlap)}")
    if not 0 < _fraction(data["alpha"]) < 1:
        raise ValueError("alpha must lie in (0, 1)")
    checkpoints = list(data["observer"]["checkpoints"])
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("observer checkpoints must be strictly increasing")
    if int(data["observer"]["attempts"]) != 2:
        raise ValueError("the confirmation allocation is frozen at a = 2 (§12.4.1)")
    blocks = int(data["certificate"]["blocks"])
    if blocks < 1:
        raise ValueError("certificate blocks must be positive")
    seeds = [int(s) for s in data["certificate"]["block_seeds"]]
    if len(seeds) != blocks or len(set(seeds)) != blocks:
        raise ValueError("certificate block_seeds must be one distinct seed per block")


def load_rules(path: str | Path = RULES_PATH, *, expected_hash: str | None = None) -> Rules:
    """Load and validate the rules; with ``expected_hash``, refuse a changed file."""
    cfg = OmegaConf.load(Path(path))
    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict) or "continuation" not in data:
        raise ValueError(f"{path}: expected a top-level 'continuation' mapping")
    rules = data["continuation"]
    _validate(rules)
    digest = rules_hash(rules)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"continuation rules changed: sha256 {digest} != frozen {expected_hash}")
    return Rules(data=rules, sha256=digest)


# ---------------------------------------------------------------------------------------------
# Dose selection (§12.4.1)


def margin_candidates(
    exploratory_upper: Mapping[int, float], margin: Fraction, checkpoints: Sequence[int]
) -> list[int]:
    """Checkpoints, in grid order, whose exploratory paired-excess upper bound is ≤ the margin.

    A checkpoint without an exploratory bound is not a candidate.
    """
    return [
        dose
        for dose in checkpoints
        if dose in exploratory_upper and Fraction(exploratory_upper[dose]) <= margin
    ]


@dataclasses.dataclass
class AttemptLedger:
    """The frozen allocation of a = 2 confirmation attempts of one lineage (§12.4.1), with the
    validity of each draw under detector revisions (§12.3.1).

    Attempt 1 confirms d₁. Attempt 2 is reserved for exactly one purpose: the next margin-passing
    dose after a failed d₁ (``NEXT_DOSE``), or a fresh-draw retest after an accepted detector
    revision (``DETECTOR_RETEST``). Every draw records the detector version it was read with and
    its sampling seed (``confirmation_seed``: new for every attempt and version).

    ``revise`` voids every valid draw at once, so the analysis dose is gone immediately. If an
    attempt remains, it is consumed by the retest: the only attempt accepted next is a
    ``DETECTOR_RETEST`` at the voided dose, on fresh draws. If both attempts were used, the
    lineage has no analysis dose.
    """

    candidates: tuple[int, ...]
    attempts: int = 2
    lineage: str = "lineage"
    base_seed: int = 0
    detector_version: str = "initial"
    log: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    retest_dose: int | None = None  # set by a revision that voided a draw with an attempt left

    def seed_for(self, dose: int) -> int:
        """The sampling seed of the next attempt at ``dose`` under the current detector."""
        attempt = len(self.log) + 1
        return confirmation_seed(self.base_seed, self.lineage, dose, attempt, self.detector_version)

    def record(self, dose: int, passed: bool, purpose: str | None = None) -> dict[str, Any]:
        """Record the next attempt (drawn with ``seed_for(dose)``) and return its entry."""
        if len(self.log) >= self.attempts:
            raise ValueError(f"no confirmation attempt left (a = {self.attempts})")
        if not self.log:
            if purpose is not None:
                raise ValueError("attempt 1 has no purpose: it confirms d₁")
            if not self.candidates or dose != self.candidates[0]:
                raise ValueError(f"attempt 1 must confirm d₁ = {self.first_dose()}")
        elif self.retest_dose is not None:
            if purpose != DETECTOR_RETEST or dose != self.retest_dose:
                raise ValueError(
                    f"after a revision attempt 2 is the retest at {self.retest_dose} on fresh draws"
                )
        else:
            if purpose != NEXT_DOSE:
                raise ValueError(f"attempt 2 without a revision is {NEXT_DOSE}")
            if self.log[-1]["passed"]:
                raise ValueError("d₁ passed: no next dose is audited")
            expected = self.next_dose()
            if dose != expected:
                raise ValueError(f"the next margin-passing dose is {expected}, not {dose}")
        entry = {
            "attempt": len(self.log) + 1,
            "dose": int(dose),
            "seed": self.seed_for(dose),
            "detector_version": self.detector_version,
            "passed": bool(passed),
            "purpose": purpose,
            "void": False,
        }
        self.log.append(entry)
        self.retest_dose = None
        return entry

    def revise(self, new_version: str) -> list[dict[str, Any]]:
        """An accepted detector revision: void every valid draw and return the voided ones."""
        if new_version == self.detector_version:
            raise ValueError("a revision needs a new detector version")
        affected = [d for d in self.log if not d["void"]]
        for d in affected:
            d["void"] = True
        self.detector_version = new_version
        if affected and len(self.log) < self.attempts:
            self.retest_dose = affected[-1]["dose"]
        return affected

    def first_dose(self) -> int | None:
        return self.candidates[0] if self.candidates else None

    def next_dose(self) -> int | None:
        """The next later margin-passing checkpoint after d₁; None if d₁ was the last one."""
        return self.candidates[1] if len(self.candidates) > 1 else None

    def analysis_dose(self) -> int | None:
        """The confirmed dose: the last attempt, if it passed and its draw is still valid."""
        if not self.log:
            return None
        last = self.log[-1]
        return last["dose"] if last["passed"] and not last["void"] else None


def observer_p1(p_abs: float, attempts: int = 2) -> float:
    """p₁ = min(1, a · p_abs) with the frozen a, however many attempts were used (§12.4.1)."""
    return min(1.0, attempts * p_abs)


# ---------------------------------------------------------------------------------------------
# Stage 2 screening (§12.7) and the Stage-4 trigger (§12.7)


def screen_deep_dose(
    screening: Mapping[int, Mapping[str, Any]], deep_doses: Sequence[int], excess_max: Fraction
) -> int | None:
    """The first deep dose with exploratory excess ≤ the margin and 0 hits in the PD pool.

    ``screening[dose]`` has ``excess_upper`` (exploratory paired-excess upper bound) and
    ``pool_hits`` (detector plus lenient-detector hits in the full PD continuation pool).
    """
    for dose in deep_doses:
        row = screening.get(dose)
        if row is None:
            continue
        if Fraction(row["excess_upper"]) <= excess_max and int(row["pool_hits"]) == 0:
            return dose
    return None


def mixed_adl_verdicts(
    p3: Sequence[float], level: Fraction, min_positive: int = 1, max_positive: int = 4
) -> bool:
    """Stage-4 trigger: p₃ ≤ level in at least ``min_positive`` and at most ``max_positive`` of
    the panel lineages. An operational rule, not evidence of heterogeneous effects."""
    positives = sum(1 for p in p3 if Fraction(p) <= level)
    return min_positive <= positives <= max_positive


# ---------------------------------------------------------------------------------------------
# Confirmation draws and detector revisions (§12.3.1)


def confirmation_seed(
    base: int, lineage: str, dose: int, attempt: int, detector_version: str
) -> int:
    """The sampling seed of one confirmation draw: new for every attempt and every detector
    version, so a rerun after a revision never repeats an earlier draw."""
    from .seeding import derive_seed

    return derive_seed(base, "confirmation", lineage, int(dose), int(attempt), detector_version)
