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

from .paths import CONFIG_ROOT, REPO_ROOT

RULES_PATH = CONFIG_ROOT / "continuation" / "rules.yaml"
# The purposes of the second confirmation attempt (§12.4.1); exactly one may be used.
NEXT_DOSE = "next_dose"
DETECTOR_RETEST = "detector_retest"


def _fraction(value: Any) -> Fraction:
    """An exact fraction from a YAML value written as "1/60", "0.05" or a number."""
    if isinstance(value, bool):
        raise TypeError(f"not a number: {value!r}")
    return Fraction(str(value))


# The meter's cap is the researcher's budget, not a rule of the analysis (§12.8): it is raised by
# the
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
    candidates are then the new ones, and the allocations already made stand.
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
    draws in attempt order: each draw {attempt, dose, seed, detector_version,
    passed, purpose}; a draw read with a new detector version follows that revision (which voids
    every earlier valid draw), and ``detector_version``, the current one, revises after the last
    draw if it is newer. Each draw's attempt number must be its place and its seed the ledger's
    fresh seed for that attempt and version; a draw the procedure does not permit is refused.

    ``candidates`` are the doses that may be confirmed under the current detector. With
    ``candidates_of`` ({version: its candidates}, the bounds rescored with each earlier detector
    a draw was read with), every allocation is replayed against the candidates of its own
    version and each revision then takes the new version's, so that rescored candidates never
    void an allocation already made; a draw of a version it does not hold is
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
    """A pilot behavior's Stage-2 decision (§12.4.1, §12.4.7, §12.7; execution 2026-10-19): its
    analysis dose and exception a.

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


# Exception a's continued-mode lineage (§12.7, Stage 5): its checks at the analysis
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
    xcont_step: bool = True,
) -> dict[str, Any]:
    """The analysis dose of a Stage-5 lineage (§12.4.1, §12.4.6, §12.4.7).

    - **Margin rule:** the candidates are the checkpoints whose exploratory paired-excess upper
      bound is ≤ the margin (``margin_candidates``); none, and the lineage is "not suppressed"
      (p = 1) at once.
    - **Confirmation:** the attempt ledger replayed from the completed draws (``replay_attempts``):
      attempt 1 confirms d₁, attempt 2 is d₂ after a failed d₁ or the retest after an accepted
      detector revision; the analysis dose is the ledger's. A permitted procedure that completes
      without one leaves the lineage "not suppressed". ``candidates_of`` holds the candidates
      under each earlier detector a draw was read with: a revision that changes the candidates,
      or leaves none, keeps the allocations made and retests the voided dose.
    - **X_cont** (§12.4.6), built at the analysis dose after its confirmation: ``xcont`` is True
      for a valid one, False for one invalid after its retry (the case is "technically
      infeasible", §Stage 5: the DPO fallback, in its own attempt and seed namespace), None while
      it is not built.
    - **Expression checks** (``STAGE5_EXPRESSION_CHECKS``) at the analysis dose: True or False when
      complete, None while not done; they qualify the statement only (``below_tolerances``).
    - ``xcont_step`` false (the descriptive DPO lineage, stack mode: its suppression is an added
      adapter, so there is no Δ_A to compose; C15) leaves X_cont out: ``xcont`` must be None and
      the expression checks follow the confirmed dose.

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
    if not xcont_step:
        if xcont is not None:
            raise ValueError("a lineage without the X_cont step has no X_cont")
    elif xcont is None:
        return {**out, "status": PENDING, "next": f"X_cont at {analysis}"}
    elif not xcont:
        return {**out, "status": "decided", "outcome": INFEASIBLE, "xcont": False,
                "expression": {}}  # fmt: skip
    checks = {c: expression.get(c) for c in STAGE5_EXPRESSION_CHECKS}
    pending = [c for c, v in checks.items() if v is None]
    if pending:
        return {**out, "status": PENDING, "next": f"expression checks {pending} at {analysis}"}
    return {**out, "status": "decided", "outcome": ANALYSIS, "xcont": True if xcont_step else None,
            "expression": checks, "below_tolerances": all(checks.values())}  # fmt: skip


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


# The keys a confirmation draw's seed is derived from, in order after the base seed
# (``confirmation_seed``): the follow-up rules name the same (confirmation.seed_keys).
CONFIRMATION_SEED_KEYS = ("confirmation", "lineage", "dose", "attempt", "detector_version")


def confirmation_seed(
    base: int, lineage: str, dose: int, attempt: int, detector_version: str
) -> int:
    """The sampling seed of one confirmation draw: new for every attempt and every detector
    version, so a rerun after a revision never repeats an earlier draw."""
    from .seeding import derive_seed

    values = {"confirmation": "confirmation", "lineage": lineage, "dose": int(dose),
              "attempt": int(attempt), "detector_version": detector_version}  # fmt: skip
    return derive_seed(base, *(values[k] for k in CONFIRMATION_SEED_KEYS))


# ---------------------------------------------------------------------------------------------
# The meter's frozen drop reasons (§12.8)


# Stage 5's descriptive DPO lineage (C15): a meter item added after the rules were frozen, below
# stage5 in the priority; cut right after stage6 (§12.8, amendment of 2026-10-21), the rules'
# cut order untouched
STAGE5_DPO = "stage5_dpo"

# The strong branch's items (§12.8, amendment of 2026-10-21; C16), each right after its parent:
# strong/stage3/<behavior> for the panel, strong/stage5 and strong/stage5_dpo. Each is cut with
# its parent's part of the cut order; strong/stage5_dpo also when its gate can no longer open.
STRONG = "strong/"
FALLBACK_GATE_CLOSED = "fallback_gate_closed"
# S5-A's sublimit (§12.8): strong/stage5 is split right after its start into s5-a
# (the continued screening and validation) with this much of its GPU part and none of its API
# part, and s5-b (the follow-up) with the rest
S5A_WORKER, S5B_WORKER, S5A_SUBLIMIT_GPU_USD = "s5-a", "s5-b", 13.76


def strong_item(parent: str) -> str:
    """The strong branch's item beside ``parent`` (``stage5`` → ``strong/stage5``)."""
    return STRONG + parent


def frozen_split(item: str, stress_gpu_usd: float) -> tuple[dict, dict] | None:
    """The split frozen for ``item`` as (GPU shares, API shares), or None if it has none:
    ``strong/stage5``'s S5-A sublimit."""
    if item != strong_item("stage5"):
        return None
    a = S5A_SUBLIMIT_GPU_USD / float(stress_gpu_usd)
    if not 0 < a < 1:
        raise ValueError(f"S5-A's ${S5A_SUBLIMIT_GPU_USD} is not within {item}'s GPU part")
    return {S5A_WORKER: a, S5B_WORKER: 1 - a}, {S5A_WORKER: 0.0, S5B_WORKER: 1.0}


def cut_order(rules: Rules, items: Sequence[str] = ()) -> list[str]:
    """The preset cut order (§12.8): the rules', with stage5_dpo right after stage6 when the
    meter's allocations hold it (the amendment of 2026-10-21). Any cut only after the researcher's
    decision."""
    order = list(rules["cut_order"])
    if STAGE5_DPO in items:
        order.insert(order.index("stage6") + 1, STAGE5_DPO)
    return order


def drop_reasons(rules: Rules, items: Sequence[str] = ()) -> dict[str, list[str]]:
    """Per plan item of the meter, the only reasons it may be dropped for: its gate (Stages 5
    and 6), its trigger (Stage 4) or its part of the preset cut order (``cut_order``, with
    stage5_dpo when ``items`` hold it). A strong item that ``items`` hold is cut with its
    parent's part, and ``strong/stage5_dpo`` is also dropped ``fallback_gate_closed`` (C16).
    Items absent here (Stages 0–1, Stage 2's first behavior, panel behaviors 1–4 and their
    strong items) can never be dropped. A panel behavior's follow-up phases are released
    ``no_registered_readout``."""
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
    for part, names in parts.items():
        if part in cut:
            for item in names:
                out[item].append(f"cut_order:{part}")
    if STAGE5_DPO in cut_order(rules, items):
        out[STAGE5_DPO] = [f"cut_order:{STAGE5_DPO}"]
    for parent in ("stage5", STAGE5_DPO, f"stage3/{panel[-1]}"):
        if strong_item(parent) in items:
            cuts = [r for r in out.get(parent, []) if r.startswith("cut_order:")]
            out[strong_item(parent)] = cuts
    if strong_item(STAGE5_DPO) in items:
        out[strong_item(STAGE5_DPO)].insert(0, FALLBACK_GATE_CLOSED)
    # the panel's follow-up phases (§12.8 as amended 2026-10-22): released when the
    # behavior's entry in the panel's contracts has no registered reading; the last behavior's
    # also cut with it
    for behavior in panel:
        for parent in (f"stage3/{behavior}", strong_item(f"stage3/{behavior}")):
            follow = f"{parent}/follow"
            if follow in items:
                cuts = [r for r in out.get(f"stage3/{behavior}", []) if r.startswith("cut_order:")]
                out[follow] = ["no_registered_readout", *cuts]
    # the follow-up (§13.13): registered work not executed or not completed
    out[f"stage4/{first}"].append(NOT_EXECUTED)
    out[f"stage4/{second}"].append(NOT_EXECUTED)
    out[f"stage3/{guillemets}"].append(NOT_COMPLETED)
    for item in items:
        if item.startswith(FOLLOW_ITEM):
            out[item] = follow_drop_reasons(item)
    return out


NOT_EXECUTED, NOT_COMPLETED = "not_executed", "not_completed"
FOLLOW_ITEM = "follow/"


def follow_drop_reasons(item: str) -> list[str]:
    """A follow-up item's frozen drop reasons (§13.6, §13.7, §13.14), each a recorded outcome or a
    named decision point, never a cut: a certificate whose line has no confirmed dose; a rung item
    not run after an identification or by the rung-3 decision; a rung's shared item (its A/H
    qualification and format validation) once neither mode can reach the rung (the meter's rule,
    ``budget.RUNG_SHARED``); the stack's items once its branch has closed. The base
    and S1 items have none."""
    if item.startswith("follow/cert/"):
        return ["not_confirmed"]
    return {
        "follow/continued/rung2": ["not_run_after_identification"],
        "follow/continued/rung3": ["not_run_after_identification", "rung_not_run"],
        "follow/stack/full": ["branch_closed"],
        "follow/stack/rung2": ["branch_closed", "not_run_after_identification"],
        "follow/stack/rung3": ["branch_closed", "not_run_after_identification", "rung_not_run"],
        "follow/rung2_shared": ["rung_not_reached"],
        "follow/rung3_shared": ["rung_not_reached", "rung_not_run"],
    }.get(item, [])


# ---------------------------------------------------------------------------------------------
# The follow-up's rules (§13): a separate, versioned file. §12's rules and loader are unchanged.

FOLLOW_RULES_PATH = CONFIG_ROOT / "continuation" / "follow_rules.yaml"
FOLLOW_M0_PATH = CONFIG_ROOT / "continuation" / "follow_m0.yaml"
FOLLOW_BUILD_PATH = CONFIG_ROOT / "continuation" / "follow_build.yaml"
FOLLOW_B2_PATH = CONFIG_ROOT / "continuation" / "follow_b2.yaml"
FOLLOW_B4_PATH = CONFIG_ROOT / "continuation" / "follow_b4.yaml"
# The follow-up rules' frozen sha256 (``follow_hash``); a new version of the file at a barrier
# (§13.15) changes it by a reviewed commit.
FOLLOW_FROZEN = "2f8eb4ddcb5c8d2c177c7f8fcc28eb770d9711343a73aab7ef5c463c0e2e6949"
FOLLOW_BEHAVIORS = ("word_count_claim", "countdown", "tldr", "eli5", "example_link", "pun_intended")
_FOLLOW_REQUIRED = ("version", "alpha", "partial_threshold", "full_threshold", "confirmation",
                    "readers", "families", "ladder", "format_validation", "m0",
                    "certificate_14b", "sizes", "s1", "capability_audit")  # fmt: skip


def follow_hash(data: Mapping[str, Any]) -> str:
    """Full sha256 of the canonical JSON form of the follow-up rules mapping."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _validate_follow(d: Mapping[str, Any]) -> None:
    missing = [key for key in _FOLLOW_REQUIRED if key not in d]
    if missing:
        raise ValueError(f"follow-up rules lack: {', '.join(missing)}")
    if tuple(d["partial_threshold"]) != FOLLOW_BEHAVIORS:
        raise ValueError(f"the partial-threshold behaviors are {FOLLOW_BEHAVIORS}, in this order")
    for behavior, spec in d["partial_threshold"].items():
        if not isinstance(spec.get("dose"), int):
            raise ValueError(f"{behavior}: no dose")
    r = d["readers"]
    if int(r["qualification"]["streams"]) != 2 or int(r["pd_qualification"]["streams"]) != 2:
        raise ValueError("qualification pools two streams (§13.5 (a))")
    dec = r["decoders"]
    if dec["identification"] != "state" or list(dec["corroboration"]) != ["state", "base"]:
        raise ValueError("identification on the state decoder, corroboration on both (§13.5 (c))")
    full = dec["corroboration_full"]
    if (_fraction(full["level"]), int(full["rung"]), full["secondary"],
            full["changes_advancement"]) != (Fraction(1, 60), 1, True, False):  # fmt: skip
        raise ValueError("full-threshold corroboration: secondary at rung 1, 1/60, no advancement")
    if dec["corroboration_partial"]["family"] not in d["families"]:
        raise ValueError("the partial-threshold corroboration family is not a Holm family")
    c = d["confirmation"]
    if (c["lineage"], c["measurement"], int(c["base_seed"]), list(c["seed_keys"])) != (
        "follow/{behavior}", "follow_confirmation", 20261029, list(CONFIRMATION_SEED_KEYS),
    ):  # fmt: skip
        raise ValueError("the confirmations' namespace, base seed or seed keys changed (§13.3)")
    if any(
        _fraction(x) <= 0
        for x in (r["margins"]["excess_over_sham_s"], r["margins"]["excess_over_x"])
    ):
        raise ValueError("the identification margins are positive (§13.5 (c))")
    if (
        _fraction(d["ladder"]["level_per_rung"]) != Fraction(1, 60)
        or int(d["ladder"]["rungs"]) != 3
    ):
        raise ValueError("the ladder is three rungs at 1/60 each (§13.5 (d))")
    s = d["sizes"]
    test, cal = [int(x) for x in s["test_seeds"]], [int(x) for x in s["calibration_seeds"]]
    if len(test) != int(s["blocks"]) or set(test) & set(cal) or len(set(test)) != len(test):
        raise ValueError("the sizes' test seeds are one per block, disjoint from calibration")
    if s["ceiling"] not in (None, 600, 1200):
        raise ValueError("the sizes' ceiling is 1,200 or 600 steps, or not yet decided")


def load_follow_rules(path: str | Path = FOLLOW_RULES_PATH, *,
                      expected_hash: str | None = None) -> Rules:  # fmt: skip
    """Load and validate the follow-up rules; with ``expected_hash``, refuse a changed file."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    if not isinstance(data, dict) or "follow" not in data:
        raise ValueError(f"{path}: expected a top-level 'follow' mapping")
    rules = data["follow"]
    _validate_follow(rules)
    digest = follow_hash(rules)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"follow-up rules changed: sha256 {digest} != frozen {expected_hash}")
    return Rules(data=rules, sha256=digest)


def follow_rules() -> Rules:
    """The follow-up rules for execution: always checked against ``FOLLOW_FROZEN``."""
    return load_follow_rules(expected_hash=FOLLOW_FROZEN)


# B1's confirmation doses (§13.3, ``scripts/follow_confirmation_doses.py``, written once): the
# file's sha256.
FOLLOW_DOSES_PATH = REPO_ROOT / "manifests" / "follow_confirmation_doses.json"
FOLLOW_DOSES_FROZEN = "5e4882ed88c36912c9a1561a059f27e7eb61dc76e74ff371a9f49bdbcdc09496"


def load_follow_doses(path: str | Path = FOLLOW_DOSES_PATH, *, rules: Rules,
                      expected_hash: str | None = None) -> dict[str, Any]:  # fmt: skip
    """B1's confirmation doses, checked against the follow-up rules ``rules``: one record per
    freshly confirmed line (``partial_threshold``'s ``confirmation: fresh``), its d₁ and d₂ the
    rules' dose and fallback, both among its margin candidates in order, each with its latent and
    sham_s states and the detector version; with ``expected_hash``, a changed file is refused."""
    blob = Path(path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"the confirmation doses changed: sha256 {digest} != {expected_hash}")
    record = json.loads(blob)
    fresh = {b: s for b, s in rules["partial_threshold"].items() if s["confirmation"] == "fresh"}
    if set(record["lineages"]) != set(fresh):
        raise ValueError(f"the confirmation doses name {sorted(record['lineages'])}, the rules "
                         f"{sorted(fresh)}")  # fmt: skip
    for behavior, spec in fresh.items():
        line = record["lineages"][behavior]
        d1, d2 = line["d1"], line["d2"]
        if (d1, d2) != (spec["dose"], spec["fallback"]):
            raise ValueError(f"{behavior}: doses {d1}, {d2}, the rules {spec['dose']}, "
                             f"{spec['fallback']}")  # fmt: skip
        candidates = [int(c) for c in line["candidates"]]
        if d1 not in candidates or d2 not in candidates or not d1 < d2:
            raise ValueError(f"{behavior}: d₁ {d1} and d₂ {d2} are not candidates in order")
        for dose in (d1, d2):
            states = line["doses"].get(str(dose)) or {}
            if not all(isinstance(states.get(role), str) for role in ("latent", "sham_s")):
                raise ValueError(f"{behavior}: no latent and sham_s state at {dose}")
        if not isinstance(line.get("detector_version"), str):
            raise ValueError(f"{behavior}: no detector version")
    return {**record, "sha256": digest}


def follow_doses() -> dict[str, Any]:
    """B1's confirmation doses for execution: checked against ``FOLLOW_DOSES_FROZEN`` and the
    frozen follow-up rules."""
    return load_follow_doses(rules=follow_rules(), expected_hash=FOLLOW_DOSES_FROZEN)


# M0's configuration's frozen sha256 (``follow_hash`` of its ``m0`` mapping).
FOLLOW_M0_FROZEN = "91bdd7214cde0ec9dd739e20da2ac3800e50d97b8878b36f20372df739f4b328"


def load_follow_m0(path: str | Path = FOLLOW_M0_PATH, *,
                   expected_hash: str | None = None) -> dict[str, Any]:  # fmt: skip
    """M0's configuration (§13.10), validated; with ``expected_hash``, refused when changed."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    m0 = data["m0"]
    states = [s["state"] for s in m0["states"]]
    if len(states) != 6 or len(set(states)) != 6:
        raise ValueError("M0 reads six distinct A/H states")
    if set(m0["methods"]) != {"qualification", "production_session", "production_cached"}:
        raise ValueError("M0 compares the three methods of §13.10")
    if list(m0["decisive"]) != ["production_session", "qualification"]:
        raise ValueError("M0 decides on same-session production against the qualification's path")
    if int(m0["repeats"]) != 2:
        raise ValueError("M0 runs each method twice")
    if m0["precision"] != {"dtype": "bfloat16", "sums": "float32"}:
        raise ValueError("M0's precision is bf16 with fp32 sums")
    if [int(p) for p in m0["positions"]] != [0, 1, 2, 3, 4]:
        raise ValueError("M0 reads positions 0–4")
    if int(m0["layers"]["decisive"]) != 20 or [int(x) for x in m0["layers"]["reported"]] != [30]:
        raise ValueError("M0 decides at layer 20 and reports layer 30")
    rt = m0["runtime"]
    if (rt["model"]["dtype"], [int(x) for x in rt["layers"]],
            (int(rt["sequences"]["count"]), int(rt["sequences"]["tokens"]))) != (
            "bfloat16", [20, 30], (10000, 128)):  # fmt: skip
        raise ValueError("M0's runtime is bf16, layers 20 and 30, 10,000 sequences of 128 tokens")
    digest = follow_hash(m0)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"M0's configuration changed: sha256 {digest} != frozen {expected_hash}")
    return m0


def follow_m0() -> dict[str, Any]:
    """M0's configuration for execution: always checked against ``FOLLOW_M0_FROZEN``."""
    return load_follow_m0(expected_hash=FOLLOW_M0_FROZEN)


# The builds' configuration's frozen sha256 (``follow_hash`` of its ``build`` mapping): set when
# the packages' states are known, after the confirmations (§13.14); until then no build runs.
FOLLOW_BUILD_FROZEN: str | None = "f6036d103cb932071dce478d2812638dfe99ee3de39a780c43b4edad8c1e5415"


def lineage_a() -> dict[str, str]:
    """Each behavior's A state as the records name it: B2's inventory manifest, the A pools'
    states (``follow_inventory.pd_sources``)."""
    b2 = follow_b2()
    manifest = json.loads((REPO_ROOT / b2["inventory"]["manifest"]).read_text())
    return {b: roles["A"]["state"] for b, roles in manifest["sources"]["pd_unfiltered"].items()}


def load_follow_build(path: str | Path = FOLLOW_BUILD_PATH, *, expected_hash: str | None = None,
                      a_states: Mapping[str, str] | None = None) -> dict[str, Any]:  # fmt: skip
    """The builds' configuration (§13.10), validated: M0's frozen runtime, layer 20 decisive and
    layers 20 and 30 read, every session a plan ``follow_build.execute`` runs, each reference its
    behavior's A (``a_states``, by default ``lineage_a``), and no build in two sessions (a
    lineage's A is rebuilt in each session that needs it); a build names its state or its
    composition (``compose``: its suppressed state and sham, composed in its session); every PD
    pool (``pools``) is a build's state, once; the confirmations the plan was derived at
    (``confirmations``: the manifest, its sha256, the doses), which the entry points check
    against the committed decision (``follow_build.committed_plan``); with ``expected_hash``,
    refused when changed."""
    from .follow_build import plan_problems, state_of

    a_states = lineage_a() if a_states is None else a_states

    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    build = data["build"]
    c = build.get("confirmations") or {}
    if set(c) != {"manifest", "sha256", "confirmed"} or set(c["confirmed"]) != {
        "example_link",
        "pun_intended",
    }:
        raise ValueError("the builds name the confirmations' manifest, its sha256 and the doses")
    if build["runtime"] != follow_m0()["runtime"]:
        raise ValueError("the builds run in M0's frozen runtime")
    if int(build["layers"]["decisive"]) != 20 or [int(x) for x in build["layers"]["read"]] != [
        20,
        30,
    ]:
        raise ValueError("the builds decide at layer 20 and read layers 20 and 30")
    seen: list[str] = []
    planned: dict[str, dict[str, Any]] = {}
    for name, session in build["sessions"].items():
        builds = []
        for b in session["builds"]:
            c = b.get("compose")
            if ("state" in b) == (c is not None) or (
                c is not None and set(c) != {"suppressed", "sham"}
            ):
                raise ValueError(f"session {name}: a build names its state or its composition "
                                 f"(suppressed, sham), not {sorted(b)}")  # fmt: skip
            builds.append({**b, "state": state_of(b)})
        problems = plan_problems(session["references"], builds)
        for b in builds:
            planned[b["state"]] = b
        for r in session["references"]:
            if a_states.get(r["behavior"]) != r["state"]:
                problems.append(f"reference {r['state']} is not {r['behavior']}'s A "
                                f"{a_states.get(r['behavior'])}")  # fmt: skip
        if problems:
            raise ValueError(f"session {name}: {'; '.join(problems)}")
        seen += [x["state"] for x in builds]
    if len(set(seen)) != len(seen):
        raise ValueError("a build is planned in two sessions")
    pools = [state_of(p) for p in build.get("pools") or []]
    if len(set(pools)) != len(pools):
        raise ValueError("a PD pool is planned twice")
    for p, sid in zip(build.get("pools") or [], pools, strict=True):
        b = planned.get(sid)
        if b is None or any(p.get(k) != b.get(k) for k in ("behavior", "lineage", "role")):
            raise ValueError(f"PD pool {sid}: not a planned build's state")
    digest = follow_hash(build)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"the builds' configuration changed: sha256 {digest} != {expected_hash}")
    return build


def follow_build() -> dict[str, Any]:
    """The builds' configuration for execution: always checked against ``FOLLOW_BUILD_FROZEN``;
    refused while that is unset."""
    if FOLLOW_BUILD_FROZEN is None:
        raise ValueError("the builds' configuration is not frozen yet: no build runs")
    return load_follow_build(expected_hash=FOLLOW_BUILD_FROZEN)


# Barrier B4's frozen sha256 (``follow_hash`` of its ``b4`` mapping): set by the commit that
# freezes V4 and O4; until then the stack branch neither screens nor validates.
FOLLOW_B4_FROZEN: str | None = "4630ab6d35feeb97ccd464f46e1ff7d12fb611745eaf13ba789756c3f774522c"
FOLLOW_B4_CANDIDATES = (400, 800)  # §13.7: the saved stack states, in order


def load_follow_b4(path: str | Path = FOLLOW_B4_PATH, *,
                   expected_hash: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Barrier B4's configuration (§13.7), validated: eli5's stack branch, its candidates the
    saved states at 400 then 800 (each L, H_s and the recipe control), the screening on the final
    PD GPU, and the suites V4 (PD, ADL) and O4 (observer) by sha256 and inventory with their
    seed and RNG keys, new beside §12.4.10's; with ``expected_hash``, refused when changed."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    b4 = data["b4"]
    if (b4["behavior"], b4["experiment"]) != ("eli5", "experiments/continuation_14b/eli5"):
        raise ValueError("B4's stack branch is eli5's, on its continuation configuration")
    doses = [int(c["dose"]) for c in b4["candidates"]]
    fields = {"dose", "latent", "sham_s", "recipe"}
    if doses != list(FOLLOW_B4_CANDIDATES) or any(set(c) != fields for c in b4["candidates"]):
        raise ValueError(f"the stack candidates are {doses}, not the saved states at 400, 800")
    if b4["screening"]["gpu"] != "NVIDIA H100 80GB HBM3":  # pipeline.FINAL_PD_GPU
        raise ValueError("the stack screening runs on the final PD GPU")
    suites = b4["suites"]
    want = {"pd": ["V4"], "adl": ["V4"], "observer": ["O4"]}
    if {k: sorted(v) for k, v in suites["sha256"].items()} != want or {
            k: sorted(v) for k, v in suites["inventories"].items()} != want:  # fmt: skip
        raise ValueError(f"B4's suites are {suites['sha256']}, not V4 (PD, ADL) and O4")
    keys = [suites["keys"]["pd"], *suites["keys"]["adl"]]
    if any(str(k).startswith("strong_") for k in keys) or len(set(keys)) != 3:
        raise ValueError(f"V4's RNG keys {keys} are not new beside §12.4.10's")
    contract = b4["contract"]
    if list(contract["keys"]) != ["model", "engine", "generator_seed", "sampling", "pd", "adl"] or (
            set(contract["requests"]) != {"observer", "ood"}):  # fmt: skip
        raise ValueError("B4's contract names the frozen keys and both requests' hashes")
    streams = b4["validation"]["streams"]
    if set(streams) != {"observer", "ood"} or len(set(streams.values())) != 2 or any(
            str(v).startswith("strong:") for v in streams.values()):  # fmt: skip
        raise ValueError(f"the validation's streams {streams} are not new and distinct")
    for latent, pin in (b4["screening"].get("pinned") or {}).items():
        if latent not in [c["latent"] for c in b4["candidates"]] or not pin.get("files"):
            raise ValueError(f"the pinned pool of {latent} is not a candidate's, with its files")
    digest = follow_hash(b4)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"B4's configuration changed: sha256 {digest} != frozen {expected_hash}")
    return b4


def follow_b4() -> dict[str, Any]:
    """B4's configuration for execution: always checked against ``FOLLOW_B4_FROZEN``; refused
    while that is unset."""
    if FOLLOW_B4_FROZEN is None:
        raise ValueError("B4 is not frozen yet: the stack branch neither screens nor validates")
    return load_follow_b4(expected_hash=FOLLOW_B4_FROZEN)


FOLLOW_B5_PATH = CONFIG_ROOT / "continuation" / "follow_b5.yaml"
# Barrier B5's frozen sha256 (``follow_hash`` of its ``b5`` mapping)
FOLLOW_B5_FROZEN: str | None = "ddf58357b209eeb3bf8918e45fa5182e09e07a3a1bcde5c8b83af495063b04c5"


def load_follow_b5(path: str | Path = FOLLOW_B5_PATH, *,
                   expected_hash: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Barrier B5's configuration (§13.11), validated: S1's two lines and data, the audit's 25
    prompts and 40 states beside the base model; with ``expected_hash``, refused when changed."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    b5 = data["b5"]
    audit = b5["audit"]
    if len(audit["prompts"]) != int(audit["responses_per_state"]) or len(audit["states"]) != 40:
        raise ValueError("B5's audit is 25 prompts and 40 states beside the base model")
    if len({s["state"] for s in audit["states"]}) != 40 or audit["base"] is not True:
        raise ValueError("B5's audit states repeat, or it leaves out the base model")
    digest = follow_hash(b5)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"B5's configuration changed: sha256 {digest} != frozen {expected_hash}")
    return b5


def follow_b5() -> dict[str, Any]:
    """B5's configuration for execution, always checked against ``FOLLOW_B5_FROZEN``."""
    return load_follow_b5(expected_hash=FOLLOW_B5_FROZEN)


# Barrier B2's frozen sha256 (``follow_hash`` of its ``b2`` mapping): set by the commit that
# freezes it; until then no qualification request of D* or unfiltered PD is sent.
FOLLOW_B2_FROZEN: str | None = "4c23cd85094d52a997eaa3623678497eb1f7858b8b0873f21666b07273c456bf"


B2_PROBE = "A probe hypothesis: the model ends its answers with a fixed phrase."
# The probe package's tokens: non-ASCII, quotes, escapes and whitespace among plain ones
B2_TOKENS = ("一秒", "刹那", " é", "\n", '"q"', "back\\slash", "\t", "🙂", "ß", "Ω", *(
    f"t{i}" for i in range(30)))  # fmt: skip


def _b2_probe_package(root: Path) -> Path:
    """A lens package in the stored files' shapes, written under ``root`` from ``B2_TOKENS``
    alone: per layer, decoder and position 100 logit-lens tokens (one repeated) and a Patchscope
    sweep whose 20 tokens rotate through the 30 scales, so that tokens recur with gaps, best ranks
    tie at the same and at different first scales, and one scale lists a token twice; per layer
    one no-patch result at scale 1, a token listed twice."""
    from . import lens_render as lr

    run = root / "runs" / "probe" / "pkg"
    n = len(B2_TOKENS)
    for a, layer in enumerate(lr.LAYERS):
        for b, decoder in enumerate(lr.DECODERS):
            for p in lr.POSITIONS:
                d = run / "steps" / f"layer{layer}" / decoder / f"p{p}"
                d.mkdir(parents=True)
                lens = [f"{B2_TOKENS[(a + b + p + i) % n]}·{i % 97}" for i in range(lr.LOGIT_K)]
                (d / "logit_lens.json").write_text(json.dumps({"tokens": lens, "probs": []}))
                sweep = []
                for i, s in enumerate(lr.SCALES):
                    off = 7 * (a + 2 * b + 3 * p) + 7 * i * (i % 3 != 2)
                    tokens = [B2_TOKENS[(off + 3 * k) % n] for k in range(lr.PATCH_K)]
                    if i == 3:
                        tokens[-1] = tokens[0]
                    sweep.append([s, tokens, [0.0] * lr.PATCH_K])
                (d / "patchscope.json").write_text(json.dumps(sweep))
        f = run / "steps" / f"layer{layer}" / "no_patch" / "patchscope.json"
        f.parent.mkdir(parents=True)
        tokens = [B2_TOKENS[(5 * a + 3 * k) % n] for k in range(lr.PATCH_K)]
        f.write_text(json.dumps([[1.0, [*tokens[:-1], tokens[1]], [0.0] * lr.PATCH_K]]))
    return run


def b2_renderings() -> dict[str, str]:
    """D*'s rendering as the code renders it, by sha256 of each output (the evidence text and its
    source files' sha256) on the probe package (``_b2_probe_package``): ``lens_render.render``
    for each decoder and ``render_no_patch``."""
    import tempfile

    from . import lens_render as lr

    def digest(found: tuple[str, dict[str, str]]) -> str:
        blob = json.dumps(list(found), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = _b2_probe_package(root)
        out = {d: digest(lr.render(run, d, root=root)) for d in lr.DECODERS}
        out["no_patch"] = digest(lr.render_no_patch(run, root=root))
    return out


def b2_requests() -> dict[str, str]:
    """Each instrument's requests as the code builds them, by sha256 of their canonical JSON: on
    a fixed probe text, every slot's request body (model, messages, cap, thinking level, response
    format) with its cache key and its retry's. The D* interpreter on its prompt and the no-patch
    prompt, the PD interpreter, the judges' strict and lenient requests of each rubric set for
    each behavior's target, the category grader and the specificity grader for each behavior."""
    from . import continuation_readers as readers
    from . import lens_render as lr
    from .config import load_config
    from .continuation_definitions import load_definitions
    from .continuation_judge import (
        CATEGORY_VERSION,
        SPECIFICITY_VERSION,
        category_request,
        specificity_request,
    )
    from .follow_inventory import BEHAVIORS
    from .judge import JudgeItem, load_targets, request_key, stage_requests
    from .summarizer import summary_requests

    def digest(rows: list[dict[str, Any]]) -> str:
        blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def summary(prompt: Any, settings: Any) -> str:
        first = summary_requests(prompt, B2_PROBE, settings, salt="probe")
        retry = summary_requests(prompt, B2_PROBE, settings, salt="probe", retry=True)
        return digest([{"key": a.key, "retry": b.key, "body": a.body}
                       for a, b in zip(first, retry, strict=True)])  # fmt: skip

    def row(r: Any, version: str) -> dict[str, Any]:
        return {"key": r.key, "retry": request_key(f"{version}{readers.RETRY}", r.body),
                "body": r.body}  # fmt: skip

    lens, lens_settings = readers.interpreter("lens")
    pd, pd_settings = readers.interpreter("pd")
    out = {"dstar_interpreter": summary(lr.prompt(), lens_settings),
           "dstar_no_patch_interpreter": summary(lr.no_patch_prompt(), lens_settings),
           "pd_interpreter": summary(pd, pd_settings)}  # fmt: skip
    targets = load_targets(BEHAVIORS)
    for fmt, kind in (("dstar", "lens"), ("pd_unfiltered", "pd")):
        settings = readers.judge_settings(readers.RUBRICS[kind])
        items = [JudgeItem("probe", b, targets[b], B2_PROBE) for b in BEHAVIORS]
        staged = stage_requests(items, settings)
        for label in ("strict", "lenient"):
            out[f"{fmt}_{label}"] = digest(
                [row(getattr(r, label), getattr(r, label).rubric_version) for r in staged]
            )
    definitions, _ = load_definitions(expected_hash=readers.DEFINITIONS_SHA256)
    models = load_config(readers.JUDGE, ["tracking.wandb.enabled=false"]).judge.models

    def spec(model_id: str) -> Any:
        return readers._spec(next(m for m in models if m.id == model_id))

    category = category_request(B2_PROBE, definitions, spec(readers.CATEGORY_MODEL),
                                max_tokens=readers.CATEGORY_MAX_TOKENS)  # fmt: skip
    out["category"] = digest([row(category, CATEGORY_VERSION)])
    out["specificity"] = digest([
        row(specificity_request(B2_PROBE, definitions[b], spec(readers.SPECIFICITY_MODEL),
                                max_tokens=readers.SPECIFICITY_MAX_TOKENS), SPECIFICITY_VERSION)
        for b in BEHAVIORS
    ])  # fmt: skip
    return out


def b2_problems(b2: Mapping[str, Any]) -> list[str]:
    """Where B2's entries and the code that executes them disagree: D*'s rendering (its whole
    scale grid, its output on the probe package, ``b2_renderings``) and prompts, the
    interpreter's and the judges' models and settings (structured output; the tier of the
    judge's new sends, ``continuation_readers.judge_tier``), every instrument's requests
    (``b2_requests``), the regions, the inventory's confirmable doses and salts, and format
    validation's quota and formats."""
    from . import continuation_judge as cj
    from . import continuation_readers as readers
    from . import follow_inventory as fi
    from . import lens_render as lr

    problems = []

    def want(label: str, got: Any, wanted: Any) -> None:
        if got != wanted:
            problems.append(f"{label}: the code has {got!r}, B2 {wanted!r}")

    r = b2["rendering"]
    want("rendering layers", [int(x) for x in lr.LAYERS], [int(x) for x in r["layers"]])
    want("rendering positions", list(lr.POSITIONS), [int(x) for x in r["positions"]])
    want("logit-lens tokens", lr.LOGIT_K, int(r["logit_lens_tokens"]))
    want("Patchscope top", lr.PATCH_K, int(r["patchscope_top"]))
    want("scales", [float(x) for x in lr.SCALES], [float(x) for x in r["scales"]])
    want("decoders", list(lr.DECODERS), list(r["decoders"]))
    for key, prompt in (("prompt", lr.prompt()), ("no_patch_prompt", lr.no_patch_prompt())):
        want(key, (prompt.version, prompt.sha256), (r[key]["version"], r[key]["sha256"]))
    want("renderings", b2_renderings(), dict(r["renderings"]))
    i = b2["interpreter"]
    for kind, key in (("lens", "lens_base_prompt"), ("pd", "pd_prompt")):
        prompt, s = readers.interpreter(kind)
        got = (s.model.id, s.model.reasoning_effort, s.max_tokens, s.samples)
        wanted = (i["model"], i["reasoning_effort"], int(i["max_tokens"]), int(i["samples"]))
        want(f"{kind} interpreter", got, wanted)
        want(f"{kind} prompt", (prompt.version, prompt.sha256),
             (i[key]["version"], i[key]["sha256"]))  # fmt: skip
    j = b2["judging"]
    want("rubric sets", {"dstar": readers.RUBRICS["lens"], "pd_unfiltered": readers.RUBRICS["pd"]},
         dict(j["rubric_sets"]))  # fmt: skip
    for version in j["rubric_sets"].values():
        s = readers.judge_settings(version)
        lenient = s.model_for("lenient")
        want(f"judge {version}", (s.model.id, s.model.reasoning_effort, lenient.id,
                                  lenient.reasoning_effort, s.max_tokens),
             (j["strict"]["model"], j["strict"]["reasoning_effort"], j["lenient"]["model"],
              j["lenient"]["reasoning_effort"], int(j["max_tokens"])))  # fmt: skip
        want(f"judge {version} structured output", s.structured_output,
             bool(j["structured_output"]))  # fmt: skip
    want(
        "category",
        (cj.CATEGORY_VERSION, readers.CATEGORY_MODEL, readers.CATEGORY_MAX_TOKENS),
        (j["category"]["version"], j["category"]["model"], int(j["category"]["max_tokens"])),
    )
    want(
        "specificity",
        (cj.SPECIFICITY_VERSION, readers.SPECIFICITY_MODEL, readers.SPECIFICITY_MAX_TOKENS),
        (j["specificity"]["version"], j["specificity"]["model"],
         int(j["specificity"]["max_tokens"])),
    )  # fmt: skip
    want("judge tier", readers.judge_tier(), j["service_tier"])
    want("requests", b2_requests(), dict(b2["requests"]))
    want("definitions", readers.DEFINITIONS_SHA256, j["definitions_sha256"])
    rules = follow_rules().data
    want("regions", rules["readers"]["regions"]["cells_sha256"], b2["regions"]["cells_sha256"])
    want("regions manifest", rules["readers"]["regions"]["manifest"], b2["regions"]["manifest"])
    want("slots", (fi.SLOTS, int(rules["readers"]["requests"])), (int(i["samples"]),) * 2)
    fv = b2["format_validation"]
    want("per class", int(rules["format_validation"]["per_class"]), int(fv["per_class"]))
    from . import follow_validation as fvd

    fr = rules["format_validation"]
    limits = fvd.max_errors(int(fv["per_class"]),
                            sensitivity_floor=_fraction(fr["sensitivity_lower_min"]),
                            fpr_ceiling=_fraction(fr["fpr_upper_max"]),
                            confidence=_fraction(fr["confidence"]))  # fmt: skip
    want("acceptance limits", limits,
         (int(fr["max_false_negatives"]), int(fr["max_false_positives"])))  # fmt: skip
    want("formats", sorted(fv["formats"]), ["dstar", "pd_unfiltered"])
    reading = fi._reading("p", "b", "r", "d", 1, "x")["salt"]
    want("D* salt", reading, b2["salts"]["dstar"].format(purpose="p", behavior="b", role="r",
                                                         decoder="d", stream=1))  # fmt: skip
    dose = fi._reading("p", "b", "r", "d", 1, "x", 7)["salt"]
    want("dose salt", dose, reading + b2["salts"]["dose_suffix"].format(dose=7))
    confirmable = {b: [int(spec["dose"]), int(spec["fallback"])]
                   for b, spec in rules["partial_threshold"].items()
                   if spec["confirmation"] == "fresh"}  # fmt: skip
    want("confirmable doses", {b: list(d) for b, d in fi.CONFIRMABLE.items()}, confirmable)
    doses = b2["inventory"]["confirmation"]["doses"]
    want("inventory doses", confirmable, {b: [int(x) for x in d] for b, d in doses.items()})
    from .continuation_definitions import load_definitions

    definitions, _ = load_definitions(expected_hash=readers.DEFINITIONS_SHA256)
    for fmt, spec in b2["ensemble"]["instructions"].items():
        want(f"{fmt} rubric set", j["rubric_sets"][fmt], spec["rubric_set"])
        text = fvd.reference_instructions(spec["rubric_set"], definitions)
        want(f"{fmt} instructions", hashlib.sha256(text.encode()).hexdigest(), spec["sha256"])
        f = REPO_ROOT / spec["file"]
        want(f"{fmt} instructions file", f.is_file() and f.read_text() == text, True)
    pd = fi._pd_reading("p", "b", "r", 1, {})["salt"]
    want("PD salt", pd, b2["salts"]["pd_unfiltered"].format(purpose="p", behavior="b", role="r",
                                                            stream=1))  # fmt: skip
    want("PD dose salt", fi._pd_reading("p", "b", "r", 1, {}, 7)["salt"],
         pd + b2["salts"]["dose_suffix"].format(dose=7))  # fmt: skip
    return problems


def _inventory_problems(manifest: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    """The manifest's counts as B2 states them, before and after ``follow_inventory.resolve``
    (each confirmable line at its first dose), and every salt distinct."""
    from . import follow_inventory as fi

    def counts(parts: Mapping[str, Any]) -> dict[str, dict[str, int]]:
        return {k: {n: len(rows) for n, rows in v.items()} for k, v in parts.items()
                if k != "sources"}  # fmt: skip

    problems = []
    if counts(manifest) != spec["counts"]:
        problems.append(f"inventory counts {counts(manifest)} != {spec['counts']}")
    first = {b: doses[0] for b, doses in fi.CONFIRMABLE.items()}
    resolved = counts(fi.resolve(manifest, first))
    if resolved != spec["confirmation"]["resolved"]:
        problems.append(f"resolved counts {resolved} != {spec['confirmation']['resolved']}")
    try:
        fi.check([r for k, v in manifest.items() if k != "sources" for rows in v.values()
                  for r in rows])  # fmt: skip
    except ValueError as error:
        problems.append(f"inventory: {error}")
    return problems


def load_follow_b2(path: str | Path = FOLLOW_B2_PATH, *, expected_hash: str | None = None,
                   check_code: bool = True) -> dict[str, Any]:  # fmt: skip
    """B2's configuration (§13.15), checked against the code that executes it
    (``b2_problems``) and against its inventory manifest, which must exist with its sha256;
    with ``expected_hash``, refused when changed."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    b2 = data["b2"]
    problems = b2_problems(b2) if check_code else []
    manifest = REPO_ROOT / b2["inventory"]["manifest"]
    if not manifest.is_file():
        problems.append(f"inventory manifest {b2['inventory']['manifest']} missing")
    else:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != b2["inventory"]["sha256"]:
            problems.append(f"inventory manifest sha256 {digest} != {b2['inventory']['sha256']}")
        problems += _inventory_problems(json.loads(manifest.read_text()), b2["inventory"])
    if problems:
        raise ValueError("B2 and the code disagree: " + "; ".join(problems))
    digest = follow_hash(b2)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"B2 changed: sha256 {digest} != frozen {expected_hash}")
    return b2


def follow_b2() -> dict[str, Any]:
    """B2's configuration for execution: always checked against ``FOLLOW_B2_FROZEN``; refused
    while that is unset."""
    if FOLLOW_B2_FROZEN is None:
        raise ValueError("barrier B2 is not frozen yet: no qualification request is sent")
    return load_follow_b2(expected_hash=FOLLOW_B2_FROZEN)


# Barrier B3's configuration (§13.15) and its frozen sha256 (``follow_hash`` of its ``b3``
# mapping), set by the commit that froze it: before it, no new reader request to L, H_s or X and
# no certificate run was sent.
FOLLOW_B3_PATH = CONFIG_ROOT / "continuation" / "follow_b3.yaml"
FOLLOW_B3_FROZEN: str | None = "36dca67ca24d53659f1bcf523502fb658f6f3893e526d21b48963bfb4bf07069"


def load_follow_b3(path: str | Path = FOLLOW_B3_PATH, *, expected_hash: str | None = None,
                   check_code: bool = True) -> dict[str, Any]:  # fmt: skip
    """B3's configuration (§13.15), checked against the records and the code that execute it
    (``follow_b3.problems``) and against its inventory manifest, which must exist with its
    sha256 (``follow_b3.inventory_problems``); with ``expected_hash``, refused when changed."""
    from . import follow_b3

    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    b3 = data["b3"]
    problems = follow_b3.problems(b3) if check_code else []
    problems += follow_b3.inventory_problems(b3["inventory"])
    if problems:
        raise ValueError("B3 and the code disagree: " + "; ".join(problems))
    digest = follow_hash(b3)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"B3 changed: sha256 {digest} != frozen {expected_hash}")
    return b3


def follow_b3() -> dict[str, Any]:
    """B3's configuration for execution: always checked against ``FOLLOW_B3_FROZEN``; refused
    while that is unset."""
    if FOLLOW_B3_FROZEN is None:
        raise ValueError("barrier B3 is not frozen yet: no reader request to L, H_s or X and no "
                         "certificate run is sent")  # fmt: skip
    return load_follow_b3(expected_hash=FOLLOW_B3_FROZEN)


# The existing L packages' random-branch session (configs/continuation/follow_lrandom.yaml) and its
# frozen sha256 (``follow_hash`` of its ``lrandom`` mapping): set by the commit that freezes it;
# until then that session is not run.
FOLLOW_LRANDOM_PATH = CONFIG_ROOT / "continuation" / "follow_lrandom.yaml"
FOLLOW_LRANDOM_FROZEN: str | None = (
    "cd5ae597a29999549693ad377974257c8b9622306744df8aeb479a1d42828b04"
)


def load_follow_lrandom(path: str | Path = FOLLOW_LRANDOM_PATH, *,
                        expected_hash: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The random-branch session's configuration: its references, builds, runtime, layers and
    record; with ``expected_hash``, refused when changed."""
    data = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True, throw_on_missing=True)
    lrandom = data["lrandom"]
    if set(lrandom) != {"runtime", "layers", "references", "builds", "base_means", "record"}:
        raise ValueError(f"the random-branch session names {sorted(lrandom)}")
    digest = follow_hash(lrandom)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"the random-branch session changed: sha256 {digest} != {expected_hash}")
    return lrandom


def follow_lrandom() -> dict[str, Any]:
    """The random-branch session for execution: checked against ``FOLLOW_LRANDOM_FROZEN``;
    refused while that is unset."""
    if FOLLOW_LRANDOM_FROZEN is None:
        raise ValueError("the random-branch session is not frozen yet: it is not run")
    return load_follow_lrandom(expected_hash=FOLLOW_LRANDOM_FROZEN)
