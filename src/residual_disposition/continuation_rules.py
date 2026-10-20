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


# The meter's cap is the user's budget, not a rule of the analysis (§12.8): it is raised by the
# user's decision and a ``cap`` event in the meter's log, which records it. The rules' hash is
# formed with the cap at its value when Stage 2's records began to carry the hash (2026-10-19,
# $1,770), so that a later raise leaves every recorded rules_sha256 valid; every other rule, the
# reserve included, stays in the hash as it is.
HASH_CAP_USD = 1770


def rules_hash(data: Mapping[str, Any]) -> str:
    """Full sha256 of the canonical JSON form of the rules mapping, the meter's cap at
    ``HASH_CAP_USD``."""
    if isinstance(data.get("meter"), Mapping) and "cap_usd" in data["meter"]:
        data = {**data, "meter": {**data["meter"], "cap_usd": HASH_CAP_USD}}
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
    ``DETECTOR_RETEST`` at the voided dose, on fresh draws, whether or not the revised detector
    keeps that dose among its candidates. If both attempts were used, the lineage has no analysis
    dose. A revision may change the candidates (bounds rescored with the new detector): the
    candidates are then the new ones, and the allocations already made stand (review s4v).
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

    def revise(
        self, new_version: str, candidates: Sequence[int] | None = None
    ) -> list[dict[str, Any]]:
        """An accepted detector revision: void every valid draw and return the voided ones; with
        ``candidates``, the margin-passing doses under the new detector from here on."""
        if new_version == self.detector_version:
            raise ValueError("a revision needs a new detector version")
        if candidates is not None:
            self.candidates = tuple(int(c) for c in candidates)
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


# The §12.4.7 checks read at the analysis dose besides the observer (its confirmation counts) and
# the PD corpus (the screening's pool): the ADL corpus (0 hits over the pooled 3,000 prefixes and
# over the 1,000 long ones) and the OOD observer (bound ≤ 0.02).
EXPRESSION_CHECKS = ("adl_corpus", "ood")
NOT_ASSESSED, PENDING = "not assessed", "pending"


def screened_doses(
    screening: Mapping[int, Mapping[str, Any]], deep_doses: Sequence[int], excess_max: Fraction
) -> list[int] | None:
    """The deep doses that meet the screening rule, in order; None while a deep dose has no
    screening row (the screening is incomplete)."""
    if any(dose not in screening for dose in deep_doses):
        return None
    return [
        d
        for d in deep_doses
        if Fraction(screening[d]["excess_upper"]) <= excess_max
        and int(screening[d]["pool_hits"]) == 0
    ]


def replay_attempts(
    candidates: Sequence[int],
    draws: Sequence[Mapping[str, Any]],
    detector_version: str,
    *,
    lineage: str,
    base_seed: int,
    attempts: int = 2,
    candidates_of: Mapping[str, Sequence[int]] | None = None,
) -> AttemptLedger:
    """A lineage's attempt ledger (§12.4.1, §12.3.1) replayed from its completed confirmation
    draws in attempt order (review s3v): each draw {attempt, dose, seed, detector_version,
    passed, purpose}; a draw read with a new detector version follows that revision (which voids
    every earlier valid draw), and ``detector_version``, the current one, revises after the last
    draw if it is newer. Each draw's attempt number must be its place and its seed the ledger's
    fresh seed for that attempt and version; a draw the procedure does not permit is refused.

    ``candidates`` are the doses that may be confirmed under the current detector. With
    ``candidates_of`` ({version: its candidates}, the bounds rescored with each earlier detector
    a draw was read with), every allocation is replayed against the candidates of its own
    version and each revision then takes the new version's, so that rescored candidates never
    void an allocation already made (review s4v); a draw of a version it does not hold is
    refused. Without it, the current candidates serve every version."""

    def of(version: str) -> Sequence[int]:
        if candidates_of is None or version == detector_version:
            return candidates
        if version not in candidates_of:
            raise ValueError(f"no candidates under detector {version}, which a draw was read with")
        return candidates_of[version]

    first = str(draws[0]["detector_version"]) if draws else detector_version
    ledger = AttemptLedger(
        candidates=tuple(int(c) for c in of(first)),
        attempts=attempts,
        lineage=lineage,
        base_seed=int(base_seed),
        detector_version=first,
    )
    for draw in draws:
        if draw["detector_version"] != ledger.detector_version:
            version = str(draw["detector_version"])
            ledger.revise(version, of(version) if candidates_of is not None else None)
        if int(draw["attempt"]) != len(ledger.log) + 1:
            raise ValueError(f"draw {draw['attempt']} recorded as attempt {len(ledger.log) + 1}")
        if not isinstance(draw.get("passed"), bool):
            raise ValueError(f"draw {draw['attempt']}: no completed result")
        if int(draw["seed"]) != ledger.seed_for(int(draw["dose"])):
            raise ValueError(
                f"draw {draw['attempt']}: not the fresh seed of its attempt and version"
            )
        ledger.record(int(draw["dose"]), draw["passed"], draw.get("purpose"))
    if detector_version != ledger.detector_version:
        ledger.revise(detector_version, candidates if candidates_of is not None else None)
    return ledger


def stage2_decision(
    screening: Mapping[int, Mapping[str, Any]],
    deep_doses: Sequence[int],
    excess_max: Fraction,
    draws: Sequence[Mapping[str, Any]],
    expression: Mapping[str, bool | None],
    *,
    detector_version: str,
    lineage: str,
    base_seed: int,
) -> dict[str, Any]:
    """A pilot behavior's Stage-2 decision (§12.4.1, §12.4.7, §12.7; execution 2026-10-19,
    reviews s3t, s3v): its analysis dose and exception a.

    - **Analysis dose:** the attempt ledger replayed from the completed confirmation draws
      (``replay_attempts``; candidates the screened deep doses in order): attempt 1 confirms d₁,
      attempt 2 is d₂ after a failed d₁ or the retest after an accepted detector revision; the
      analysis dose is the ledger's (the last attempt, passed and not voided). A dose never
      attempted is "not assessed". An expression failure gives no attempt.
    - **Expression checks** (``EXPRESSION_CHECKS``) are read at the analysis dose; they qualify
      its interpretation, not its certificate and readers.
    - **Exception a** applies when the permitted procedure, completed (attempts exhausted or no
      further dose permitted), yields no analysis dose, or the analysis dose fails a completed
      expression check; with no screened dose it applies at once.

    ``expression[check]`` is True or False for a completed check, None (or absent) for one not
    yet done or technically invalid. Incomplete work leaves the decision "pending" with the next
    step named."""
    screened = screened_doses(screening, deep_doses, excess_max)
    if screened is None:
        return {"status": PENDING, "next": "screening", "doses": {}}
    doses = {d: ("screened" if d in screened else "failed screening") for d in deep_doses}
    if not screened:
        if draws:
            raise ValueError("confirmation draws without a screened deep dose")
        return {"status": "decided", "doses": doses, "analysis_dose": None,
                "exception_a": True, "expression": {}, "attempts": []}  # fmt: skip
    ledger = replay_attempts(
        screened, draws, detector_version, lineage=lineage, base_seed=base_seed
    )
    for entry in ledger.log:
        doses[entry["dose"]] = "voided" if entry["void"] else (
            "confirmed" if entry["passed"] else "failed confirmation"
        )  # fmt: skip
    for d in screened:
        if doses[d] == "screened":
            doses[d] = NOT_ASSESSED
    analysis = ledger.analysis_dose()
    out: dict[str, Any] = {"doses": doses, "analysis_dose": analysis, "attempts": ledger.log}
    if analysis is None:
        last = ledger.log[-1] if ledger.log else None
        if ledger.retest_dose is not None:
            return {
                **out,
                "status": PENDING,
                "next": f"confirmation retest at {ledger.retest_dose}",
            }
        if last is None:
            return {**out, "status": PENDING, "next": f"confirmation at {ledger.first_dose()}"}
        if len(ledger.log) < ledger.attempts and not last["passed"] and ledger.next_dose():
            return {**out, "status": PENDING, "next": f"confirmation at {ledger.next_dose()}"}
        return {**out, "status": "decided", "exception_a": True, "expression": {}}
    checks = {c: expression.get(c) for c in EXPRESSION_CHECKS}
    pending = [c for c, v in checks.items() if v is None]
    if pending:
        return {**out, "status": PENDING, "next": f"expression checks {pending} at {analysis}"}
    failed = [c for c, v in checks.items() if not v]
    return {**out, "status": "decided", "exception_a": bool(failed), "expression": checks}


# Exception a's continued-mode lineage (§12.7, Stage 5; review s4r): its checks at the analysis
# dose besides the observer (its confirmation counts) are the PD corpus (the full continuation
# pool in both framings, 0 hits), the ADL corpus and the OOD observer. They qualify the statement
# (the Stage-2 table's row "above tolerances"); none gates the certificate, and Stage 2's
# zero-PD-hit screening is no part of the dose selection.
STAGE5_EXPRESSION_CHECKS = ("pd_corpus", "adl_corpus", "ood")
NOT_SUPPRESSED, INFEASIBLE, ANALYSIS = "not suppressed", "technically infeasible", "analysis dose"


def stage5_decision(
    exploratory_upper: Mapping[int, Any],
    checkpoints: Sequence[int],
    margin: Fraction,
    draws: Sequence[Mapping[str, Any]],
    xcont: bool | None,
    expression: Mapping[str, bool | None],
    *,
    detector_version: str,
    lineage: str,
    base_seed: int,
    candidates_of: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """The analysis dose of a Stage-5 lineage (§12.4.1, §12.4.6, §12.4.7; review s4r).

    - **Margin rule:** the candidates are the checkpoints whose exploratory paired-excess upper
      bound is ≤ the margin (``margin_candidates``); none, and the lineage is "not suppressed"
      (p = 1) at once.
    - **Confirmation:** the attempt ledger replayed from the completed draws (``replay_attempts``):
      attempt 1 confirms d₁, attempt 2 is d₂ after a failed d₁ or the retest after an accepted
      detector revision; the analysis dose is the ledger's. A permitted procedure that completes
      without one leaves the lineage "not suppressed". ``candidates_of`` holds the candidates
      under each earlier detector a draw was read with: a revision that changes the candidates,
      or leaves none, keeps the allocations made and retests the voided dose (review s4v).
    - **X_cont** (§12.4.6), built at the analysis dose after its confirmation: ``xcont`` is True
      for a valid one, False for one invalid after its retry (the case is "technically
      infeasible", §Stage 5: the DPO fallback, in its own attempt and seed namespace), None while
      it is not built.
    - **Expression checks** (``STAGE5_EXPRESSION_CHECKS``) at the analysis dose: True or False when
      complete, None while not done; they qualify the statement only (``below_tolerances``).

    Incomplete work leaves the decision "pending" with the next step named."""
    if any(d not in exploratory_upper for d in checkpoints):
        missing = [d for d in checkpoints if d not in exploratory_upper]
        return {"status": PENDING, "next": f"the exploratory pass at {missing}", "doses": {}}
    candidates = margin_candidates(exploratory_upper, margin, checkpoints)
    doses: dict[int, str] = {
        d: ("met the margin" if d in candidates else "above the margin") for d in checkpoints
    }
    none = {"xcont": None, "expression": {}}
    if not candidates:
        if not draws:
            return {"status": "decided", "doses": doses, "analysis_dose": None,
                    "outcome": NOT_SUPPRESSED, "attempts": [], **none}  # fmt: skip
        if all(str(d["detector_version"]) == detector_version for d in draws):
            raise ValueError("confirmation draws without a checkpoint that met the margin rule")
    ledger = replay_attempts(
        candidates, draws, detector_version, lineage=lineage, base_seed=base_seed,
        candidates_of=candidates_of,
    )  # fmt: skip
    for entry in ledger.log:
        doses[entry["dose"]] = "voided" if entry["void"] else (
            "confirmed" if entry["passed"] else "failed confirmation"
        )  # fmt: skip
    for d in candidates:
        if doses[d] == "met the margin":
            doses[d] = NOT_ASSESSED
    analysis = ledger.analysis_dose()
    out: dict[str, Any] = {"doses": doses, "analysis_dose": analysis, "attempts": ledger.log}
    if analysis is None:
        last = ledger.log[-1] if ledger.log else None
        if ledger.retest_dose is not None:
            return {
                **out,
                "status": PENDING,
                "next": f"confirmation retest at {ledger.retest_dose}",
            }
        if last is None:
            return {**out, "status": PENDING, "next": f"confirmation at {ledger.first_dose()}"}
        if len(ledger.log) < ledger.attempts and not last["passed"] and ledger.next_dose():
            return {**out, "status": PENDING, "next": f"confirmation at {ledger.next_dose()}"}
        return {**out, "status": "decided", "outcome": NOT_SUPPRESSED, **none}
    if xcont is None:
        return {**out, "status": PENDING, "next": f"X_cont at {analysis}"}
    if not xcont:
        return {**out, "status": "decided", "outcome": INFEASIBLE, "xcont": False,
                "expression": {}}  # fmt: skip
    checks = {c: expression.get(c) for c in STAGE5_EXPRESSION_CHECKS}
    pending = [c for c, v in checks.items() if v is None]
    if pending:
        return {**out, "status": PENDING, "next": f"expression checks {pending} at {analysis}"}
    return {**out, "status": "decided", "outcome": ANALYSIS, "xcont": True, "expression": checks,
            "below_tolerances": all(checks.values())}  # fmt: skip


def stage5_slot(decisions: Mapping[str, Mapping[str, Any] | None], order: Sequence[str]) -> dict:
    """The single Stage-5 slot (§12.7, exception a): the first behavior in ``order`` (the pilot
    behaviors, example_link then pun_intended) whose decision applies exception a takes it; a
    later one that also does is not run. While an earlier behavior's decision is missing or
    pending, no later one can take the slot. ``status`` is "taken", "free" (Stage 5 by its own
    gate) or "pending"."""
    for behavior in order:
        d = decisions.get(behavior)
        if d is None or d.get("status") != "decided":
            return {"status": PENDING, "slot": None, "waiting_on": behavior}
        if d["exception_a"]:
            rest = order[order.index(behavior) + 1 :]
            later = [b for b in rest if (decisions.get(b) or {}).get("exception_a")]
            return {"status": "taken", "slot": behavior, "not_run": later}
    return {"status": "free", "slot": None}


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


# ---------------------------------------------------------------------------------------------
# The meter's frozen drop reasons (§12.8)


def drop_reasons(rules: Rules) -> dict[str, list[str]]:
    """Per plan item of the meter, the only reasons it may be dropped for: its gate (Stages 5
    and 6), its trigger (Stage 4) or its part of the preset cut order. Items absent here (Stages
    0–1, Stage 2's first behavior, panel behaviors 1–4) can never be dropped."""
    first, second = rules["discovery"][:2]
    panel = list(rules.panel)
    guillemets = rules["descriptive_adl_only"][0]
    cut = set(rules["cut_order"])
    out: dict[str, list[str]] = {
        "stage6": ["gate_not_met"],
        "stage5": ["gate_not_met"],
        f"stage4/{first}": ["trigger_not_met"],
        f"stage4/{second}": ["trigger_not_met"],
        f"stage3/{guillemets}": [],
        f"stage3/{panel[-1]}": [],
        f"stage2/{second}": [],
    }
    parts = {
        "stage6": ["stage6"],
        "stage5": ["stage5"],
        "stage4": [f"stage4/{first}", f"stage4/{second}"],
        "guillemets": [f"stage3/{guillemets}"],
        "last_stage3_behavior": [f"stage3/{panel[-1]}"],
        "stage2_to_one": [f"stage2/{second}"],
    }
    for part, items in parts.items():
        if part in cut:
            for item in items:
                out[item].append(f"cut_order:{part}")
    return out
