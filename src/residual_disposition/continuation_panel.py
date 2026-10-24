"""The panel's analysis (preregistration §12.2, §12.5, §12.6, §12.7 Stages 3–5; the amendments
of 2026-10-21 and 2026-10-24), on CPU, from committed evidence alone and with no free choice.

- **Per panel behavior**, in the frozen order, its entry in the committed panel contracts decides
  its path:
  - reading ``strong``: primary p_i = max(p₁, p₂, p₃), p₁ the exact tail P{Bin(600, 0.02) ≤ h}
    of L's hits on the validation's O3 draw, not doubled;
  - reading ``stack``: primary p_i = 1 (no validated strong candidate); the four-component rule
    is reported as descriptive, its p₁ = min(1, 2·p_abs) from the non-voided deciding
    confirmation;
  - no reading: p_i = p₃ = 1, reported "ineligible, no readout".
- **Three kinds of incompleteness, never confused:**
  - ongoing or recoverable work is *pending*: no record is written (``Pending``);
  - a documented terminal failure or unrun work keeps the lineage in the panel with the
    registered p = 1 treatment (§12.2): an unresolved k*, recomputed from its calibration
    record, a failure record of a phase, or the readers' own terminal status;
  - malformed, contradictory or foreign evidence is refused (``ValueError``).
  A completed censored run is evidence, never incompleteness. An unresolved k* or a failed
  certificate sets p₂ = 1; it never changes the readers' p₃, which still enters Stage 4's and
  Stage 5's counts.
- **The readers' values** come from the readers' manifest as their own recomputation verifier
  returns it (``reader_values`` checks its binding, flags and values; ``READER_FLAGS``).
- **The panel:** p_PC = min(1, 3·p₍₃₎) over m = 5, rejected iff p₍₃₎ ≤ 1/60. Stage 4's trigger
  counts p₃ ≤ 1/20 (1 to 4 trigger); Stage 5's gate is 3·p₃,(3) ≤ 1/20, reported with its
  action, which exception a's occupancy fixes. Every comparison is in exact rationals.
- **Wording** (``WORDING``): frozen sentences selected by the values alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import continuation_stats as cs

PANEL = ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact")  # §12.2, frozen order
R = 3
LEVEL = Fraction(1, 20)  # α, the success counts' and both gates' level
PANEL_LEVEL = Fraction(1, 60)  # p₍₃₎ ≤ α / (m − r + 1)
OBSERVER_N = 600
TOLERANCE = Fraction(2, 100)
ATTEMPTS = 2  # the stack's frozen allocation a (§12.4.1)
PHASES = ("kstar", "certificate", "readers")
READER_FLAGS = ("eligibility", "fidelity", "category_confusion", "stability")
CONTROL = "x_validity"  # X (stack) or X_cont (strong) passed its validation (§12.4.6)
READER_STATES = ("A", "H", "L", "H_s", "X", "recipe")
READER_STATUSES = ("complete", "terminal")  # pending work writes no manifest
BINDINGS = ("packages", "runs", "labels")
SLOT_TAKEN = "slot already consumed by example_link under exception a"

WORDING = {
    "rejected": "in at least three of the five prespecified behavior–implant pairs, the "
                "suppressed model shows no detected expression on the prespecified outward-probe "
                "suites, its relearnability is retained, and lens-only ADL reads the implanted "
                "behavior (p_PC = {p_pc})",
    "not_rejected": "the panel test does not reject (p_PC = {p_pc})",
    "no_reading": "ineligible, no readout",
    "descriptive": "the application of the original four-component rule (p₁–p₄) to the amended "
                   "selected doses, descriptive only: p = {p}",
    "adl_shown": "lens-only ADL reads the implanted behavior (p₃ = {p})",
    "adl_not_shown": "lens-only ADL superiority not shown (p₃ = {p})",
    "pd_shown": "PD-clean non-identification shown (p₄ = {p})",
    "pd_not_shown": "PD-clean non-identification not shown (p₄ = {p})",
    "stage4_triggered": "mixed ADL verdicts: {count} of 5 lineages with p₃ ≤ 1/20; Stage 4 is "
                        "triggered",
    "stage4_not": "no mixed ADL verdicts: {count} of 5 lineages with p₃ ≤ 1/20; Stage 4 is not "
                  "triggered",
    "stage5_met": "Stage 5's gate is met (3·p₃,(3) = {p}; {count} of 5 with p₃ ≤ 1/20)",
    "stage5_not": "Stage 5's gate is not met (3·p₃,(3) = {p}; {count} of 5 with p₃ ≤ 1/20)",
}  # fmt: skip


class Pending(Exception):
    """Ongoing or recoverable work: no record is written. ``args[0]`` names what is pending."""


# ---------------------------------------------------------------------------------- values


def exact(value: Any, what: str) -> Fraction:
    """A recorded p-value as an exact rational in [0, 1]: a fraction string or an integer;
    a float, a bool, a non-finite or out-of-range value is refused."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{what}: {value!r} is not an exact p-value")
    try:
        p = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{what}: {value!r} is not an exact p-value") from error
    if not 0 <= p <= 1:
        raise ValueError(f"{what}: {value!r} is outside [0, 1]")
    return p


def _hits(count: Any, what: str) -> int:
    hits, n = (count or {}).get("hits"), (count or {}).get("n")
    if type(hits) is not int or n != OBSERVER_N or not 0 <= hits <= n:
        raise ValueError(f"{what}: {count!r} is not a count over the {OBSERVER_N} prompts")
    return hits


def strong_p1(validation: Mapping[str, Any], identity: Mapping[str, Any]) -> Fraction:
    """A strong reading's p₁: P{Bin(600, 0.02) ≤ h}, h L's hits on its validation's O3 draw, one
    attempt, not doubled (§12.5). The validation must be the passed one of the lineage's
    candidate, and its recorded p₁ the recomputed one."""
    candidate = validation.get("candidate") or {}
    want = (identity["dose"], identity["states"]["L"], identity["states"]["H_s"])
    if (candidate.get("dose"), candidate.get("latent"), candidate.get("sham_s")) != want:
        raise ValueError(f"the validation's candidate {candidate} is not the lineage's {want}")
    if (validation.get("verdict") or {}).get("passed") is not True:
        raise ValueError("a strong reading without a passed validation")
    if validation.get("detector_version") != identity["detector"]:
        raise ValueError("the validation was not counted under the lineage's detector")
    h = _hits(((validation.get("counts") or {}).get("observer") or {}).get("L"), "O3's L")
    p1 = cs.binomial_cdf(h, OBSERVER_N, TOLERANCE)
    if exact(validation.get("p1"), "the validation's p₁") != p1:
        raise ValueError(f"the validation's p₁ is not P{{Bin(600, 0.02) ≤ {h}}}")
    return p1


def expression_report(
    reading: str, record: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The expression checks as registered for the report (§12.4.7, §12.4.10), their counts,
    bounds and labels unchanged: a strong reading's validation (its counts, pointwise bounds,
    verdict and suites); a stack reading's decision with its checks' diagnostics recomputed
    under the final detector (``continuation_stage5.expression_diagnostics``: the PD corpus's,
    the ADL corpus's and the OOD observer's counts and bounds), whose labels must be the
    decision's and none of them missing."""
    if reading == "strong":
        keys = ("counts", "bounds", "verdict", "suites", "detector_version")
        missing = [k for k in keys if k not in record]
        if missing:
            raise ValueError(f"the validation holds no {missing}")
        return {"source": "strong validation", **{k: record[k] for k in keys}}
    labels = (record.get("decision") or {}).get("expression")
    if not isinstance(labels, dict) or diagnostics is None:
        raise ValueError("the stack decision's expression checks are not given")
    if diagnostics.get("labels") != labels:
        raise ValueError(f"the expression checks give {diagnostics.get('labels')}, the decision "
                         f"{labels}")  # fmt: skip
    missing = [k for k in ("pd_corpus", "adl_corpus", "ood") if diagnostics.get(k) is None]
    if missing:
        raise ValueError(f"the stack decision's expression checks lack {missing}")
    return {"source": "stack decision", **{k: diagnostics[k] for k in (
        "labels", "pd_corpus", "adl_corpus", "ood", "detector_version", "runs")}}  # fmt: skip


def stack_p1(
    decision: Mapping[str, Any], confirmations: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> Fraction:  # fmt: skip
    """A stack reading's descriptive p₁ = min(1, 2·p_abs) from its deciding confirmation: the
    one non-voided attempt that passed at the analysis dose (a detector retest replaces a voided
    draw, which never enters), counted under the lineage's detector. ``confirmations`` are the
    decision's draws' summaries in attempt order."""
    d = decision.get("decision") or {}
    if (d.get("status"), d.get("analysis_dose")) != ("decided", identity["dose"]):
        raise ValueError(f"the stack decision has no analysis dose {identity['dose']}")
    attempts = list(d.get("attempts") or [])
    if len(attempts) != len(confirmations):
        raise ValueError(f"{len(attempts)} attempts, {len(confirmations)} confirmations")
    fields = ("attempt", "seed", "detector_version", "purpose", "dose", "passed")
    for a, s in zip(attempts, confirmations, strict=True):
        if any(a.get(k) != s.get(k) for k in fields) or not isinstance(a.get("void"), bool):
            raise ValueError(f"attempt {a.get('attempt')} is not its confirmation's draw")
    deciding = [(a, s) for a, s in zip(attempts, confirmations, strict=True)
                if not a["void"] and a["passed"] is True
                and a["dose"] == identity["dose"]]  # fmt: skip
    if len(deciding) != 1:
        raise ValueError(f"{len(deciding)} non-voided passed confirmations at the analysis dose")
    attempt, summary = deciding[0]
    if attempt["detector_version"] != identity["detector"]:
        raise ValueError("the deciding confirmation was not counted under the lineage's detector")
    h = _hits(summary.get("latent"), "the deciding confirmation's L")
    p_abs = cs.observer_p_abs(h, OBSERVER_N, TOLERANCE)
    if "p_abs" in summary and exact(summary["p_abs"], "the confirmation's p_abs") != p_abs:
        raise ValueError("the confirmation's p_abs is not the recomputed one")
    return cs.observer_p1(h, OBSERVER_N, TOLERANCE, ATTEMPTS)


# -------------------------------------------------------------------- k* and the certificate


def _failure(failures: Mapping[str, Any], phase: str, identity: Mapping[str, Any]) -> Any:
    """The lineage's committed failure record of ``phase``, checked against its identity."""
    record = failures.get(phase)
    if record is None:
        return None
    want = {"lineage": identity["lineage"], "behavior": identity["behavior"],
            "contracts": identity["contracts"], "phase": phase}  # fmt: skip
    if any(record.get(k) != v for k, v in want.items()) or not str(record.get("reason") or ""):
        raise ValueError(f"the {phase} failure record is not one of {identity['lineage']}")
    return record


def _binding_problems(record: Mapping[str, Any], identity: Mapping[str, Any], what: str) -> None:
    stage3 = record.get("stage3") if isinstance(record.get("stage3"), dict) else None
    if stage3 is None or "stage5" in record:
        raise ValueError(f"{what} is not bound to a certified Stage-3 lineage")
    if (stage3.get("contracts") or {}).get("sha256") != identity["contracts"]:
        raise ValueError(f"{what} is bound to other contracts")
    if "binding" in identity and stage3 != identity["binding"]:
        raise ValueError(f"{what} is not bound to the certified lineage's evidence")
    if (record.get("lineage"), record.get("behavior"), record.get("dose")) != (
        identity["lineage"], identity["behavior"], identity["dose"]
    ):  # fmt: skip
        raise ValueError(f"{what} is of another lineage")


def kstar_value(
    kstar: Mapping[str, Any] | None, failures: Mapping[str, Any], identity: Mapping[str, Any],
    recompute: Any,
) -> dict[str, Any]:  # fmt: skip
    """k*'s state: ``resolved``; ``unresolved`` (terminal), recomputed from the committed
    calibration record by ``recompute(record)`` (``continuation_kstar.recomputed_status``),
    never read from its label; ``failed`` (terminal) by a failure record; pending otherwise."""
    failed = _failure(failures, "kstar", identity)
    if kstar is None:
        if failed is None:
            raise Pending(f"{identity['lineage']}: k* not calibrated")
        return {"status": "failed", "reason": failed["reason"]}
    if failed is not None:
        raise ValueError(f"{identity['lineage']}: a k* record and a k* failure record")
    _binding_problems(kstar, identity, "the k* record")
    states = {"H_s": identity["states"]["H_s"], "A": identity["states"]["A"]}
    if kstar.get("states") != states:
        raise ValueError(f"the k* record's states {kstar.get('states')} are not {states}")
    status = recompute(kstar)
    if status != kstar.get("status") or status not in ("resolved", "unresolved"):
        raise ValueError(f"the k* record says {kstar.get('status')}, recomputed {status}")
    if status == "unresolved":
        return {"status": "unresolved", "rates": list(kstar.get("unresolved") or [])}
    return {"status": "resolved", "sha256": kstar.get("sha256")}


def _end(x: float) -> float | str:
    """An interval's end as JSON keeps it: a finite float, or "inf" / "-inf"."""
    return x if math.isfinite(x) else ("inf" if x > 0 else "-inf")


def _test_report(t: Any) -> dict[str, Any]:
    """A sign test as registered for the report (§12.4.8): its blocks, positive blocks and p,
    whether complete, the order-statistic interval [d₍₁₎, d₍ₙ₎] with its coverage (None where
    unavailable) and the sample median when every contrast is observed."""
    interval = None if t.interval is None else [_end(t.interval[0]), _end(t.interval[1])]
    return {"blocks": t.blocks, "positive": t.positive, "p": t.p, "complete": t.complete,
            "interval": interval, "coverage": cs.order_statistic_coverage(t.blocks),
            "median": t.median}  # fmt: skip


def certificate_anchor_problems(merged: Mapping[str, Any], expected: Mapping[str, Any]) -> list:
    """Where the merged certificate's shared inputs are not the trusted ones (``expected``):
    ``common``, every field of ``continuation_certificate.COMMON`` but ``params``, as the
    committed k* manifest, the Y₂ manifest and the lineage give them; ``approved``, each arm's
    exact cell params at each rate (but the state and the seed) as ``pipeline.relearn_cell``
    makes them from those inputs (``certificate_cells``). Compared as a cell's hash serializes
    them, so another type is another value."""
    from .config import canonical

    common = merged.get("common") or {}
    problems = [f"{k} {common.get(k)!r}, not {v!r}" for k, v in expected["common"].items()
                if canonical(common.get(k)) != canonical(v)]  # fmt: skip
    params, want = common.get("params") or {}, expected["approved"]
    if sorted(params) != sorted(want):
        return [*problems, f"params of arms {sorted(params)}, not {sorted(want)}"]
    for arm, by_rate in want.items():
        if sorted(params[arm]) != sorted(by_rate):
            problems.append(f"arm {arm}: params at rates {sorted(params[arm])}")
            continue
        for rate, approved in by_rate.items():
            given = params[arm][rate]
            if canonical(given) != canonical(approved):
                keys = sorted(k for k in set(given) | set(approved)
                              if canonical(given.get(k)) != canonical(approved.get(k)))  # fmt: skip
                problems.append(f"arm {arm} at {rate}: {keys}")
    return problems


def certificate_value(
    merged: Mapping[str, Any] | None, kstar: Mapping[str, Any], failures: Mapping[str, Any],
    identity: Mapping[str, Any], design: Mapping[str, Any], expected: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """p₂ from the certificate's merged records (``continuation_certificate.merge``), anchored
    to the trusted inputs (``certificate_anchor_problems``): computed on a complete design,
    censored runs included; an incomplete design is pending, or terminal (p₂ = 1, its tests
    still reported) under a failure record. Without a resolved k* there is no certificate:
    p₂ = 1. ``kstar`` is the full committed k* manifest with its status."""
    from .continuation_certificate import blocks

    failed = _failure(failures, "certificate", identity)
    if kstar["status"] != "resolved":
        if merged is not None or failed is not None:
            raise ValueError(f"{identity['lineage']}: certificate evidence without a resolved k*")
        return {"status": f"k* {kstar['status']}", "p": Fraction(1)}
    if merged is None:
        if failed is None:
            raise Pending(f"{identity['lineage']}: the certificate has no record")
        return {"status": "failed", "p": Fraction(1), "reason": failed["reason"], "tests": {}}
    common = merged.get("common") or {}
    manifests = common.get("manifests") or {}
    _binding_problems({**common, "stage3": manifests.get("stage3"),
                       **({"stage5": 1} if "stage5" in manifests else {})},
                      identity, "the certificate")  # fmt: skip
    if manifests.get("kstar") != kstar.get("sha256"):
        raise ValueError("the certificate was not run from the committed k* manifest")
    wrong = certificate_anchor_problems(merged, expected)
    if wrong:
        raise ValueError(f"the certificate's inputs are not the trusted ones: {'; '.join(wrong)}")
    by_rate = blocks(merged["runs"], design["block_ids"], design["learning_rates"])
    result = cs.certificate_p(by_rate, **design)
    complete = not result.problems and all(
        arm.complete for bs in by_rate.values() for b in bs for arm in b.arms.values()
    )
    tests = {f"{rate!r}/{name}": _test_report(t)
             for (rate, name), t in sorted(result.tests.items())}  # fmt: skip
    if not complete:
        if failed is None:
            raise Pending(f"{identity['lineage']}: the certificate is incomplete")
        return {"status": "failed", "p": Fraction(1), "reason": failed["reason"], "tests": tests}
    if failed is not None:
        raise ValueError(f"{identity['lineage']}: a complete certificate and a failure record")
    return {"status": "complete", "p": result.p, "tests": tests}


# ------------------------------------------------------------------------------- readers

_SHA = re.compile(r"[0-9a-f]{64}")
_STATE = re.compile(r"s[0-9a-f]{16}")


def _flag(value: Any, what: str) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("passed"), bool):
        raise ValueError(f"the readers' {what} has no pass flag")
    if not isinstance(value.get("inputs"), dict) or not value["inputs"]:
        raise ValueError(f"the readers' {what} binds no inputs")
    return value["passed"]


def _bound(group: Any, kind: str, *, required: bool) -> None:
    """A binding group: sha256 values; non-empty when ``required``."""
    if group is None and not required:
        return
    if (
        not isinstance(group, dict)
        or (required and not group)
        or not all(isinstance(v, str) and _SHA.fullmatch(v) for v in group.values())
    ):
        raise ValueError(f"the readers' {kind} are not bound by sha256")


def reader_values(
    manifest: Mapping[str, Any] | None, failures: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """p₃ and p₄ from the readers' manifest, as their recomputation verifier returned it,
    checked here.

    - A missing manifest is pending, or p₃ = p₄ = 1 under a failure record (the readers'
      budget stop). Pending work writes no manifest: a manifest is complete or terminal.
    - **Complete:** the lineage, behavior, dose, contracts, final detector and six distinct
      states; every flag with its inputs, X valid; every binding by sha256; flags and values
      consistent (a failed eligibility, fidelity, category-confusion or stability check gives
      p₃ = 1, a failed PD positive control p₄ = 1); the values exact and in [0, 1].
    - **Terminal:** only an invalid X (§12.4.6), its validation's bound result failed, with
      p₃ = p₄ = 1 and its reasons. The states it has (X may have none), the runs that show it,
      and no package, label or flag that does not exist: none is made up.
    """
    failed = _failure(failures, "readers", identity)
    if manifest is None:
        if failed is None:
            raise Pending(f"{identity['lineage']}: no readers' manifest")
        return {"status": "failed", "p3": Fraction(1), "p4": Fraction(1),
                "reason": failed["reason"]}  # fmt: skip
    want = {"lineage": identity["lineage"], "behavior": identity["behavior"],
            "dose": identity["dose"], "contracts": identity["contracts"],
            "final_detector": identity["detector"]}  # fmt: skip
    wrong = sorted(k for k, v in want.items() if manifest.get(k) != v)
    if wrong:
        raise ValueError(f"the readers' manifest is not {identity['lineage']}'s at {wrong}")
    status, reasons = manifest.get("status"), manifest.get("reasons")
    if status not in READER_STATUSES or not isinstance(reasons, list):
        raise ValueError(f"the readers' status {status!r} with reasons {reasons!r}: a manifest "
                         "is complete or terminal, pending work writes none")  # fmt: skip
    if status == "terminal" and not reasons:
        raise ValueError("a terminal readers' manifest without its reasons")
    if failed is not None:
        raise ValueError(f"{identity['lineage']}: {status} readers and a failure record")
    terminal = status == "terminal"
    states = manifest.get("states") or {}
    needed = [r for r in READER_STATES if not (terminal and r == "X")]
    if (
        not set(needed) <= set(states) <= set(READER_STATES)
        or not all(isinstance(s, str) and _STATE.fullmatch(s) for s in states.values())
        or len(set(states.values())) != len(states)
    ):
        raise ValueError(f"the readers' states {states} are not the lineage's distinct states")
    known = {k: v for k, v in identity["states"].items() if k in states}
    if any(states[k] != v for k, v in known.items()):
        raise ValueError(f"the readers' states are not the lineage's {known}")
    x_valid = _flag(manifest.get(CONTROL), "X validity")
    p3, p4 = exact(manifest.get("p3"), "p₃"), exact(manifest.get("p4"), "p₄")
    bindings = manifest.get("bindings") or {}
    if terminal:
        if x_valid:
            raise ValueError("a terminal readers' manifest is an invalid X's alone")
        if (p3, p4) != (1, 1):
            raise ValueError("terminal readers with p₃ or p₄ below 1")
        _bound(bindings.get("runs"), "runs", required=True)
        for kind in ("packages", "labels"):
            _bound(bindings.get(kind), kind, required=False)
        passed = {k: _flag(manifest[k], k) for k in READER_FLAGS if manifest.get(k) is not None}
        return {"status": status, "p3": p3, "p4": p4, "flags": passed, "x_valid": False,
                "pd_positive_control": None, "reasons": list(reasons)}  # fmt: skip
    if not x_valid:
        raise ValueError("an invalid X makes the readers terminal, not complete")
    passed = {k: _flag(manifest.get(k), k) for k in READER_FLAGS}
    control = _flag((manifest.get("eligibility") or {}).get("pd_positive_control"),
                    "PD positive control")  # fmt: skip
    for kind in BINDINGS:
        _bound(bindings.get(kind), kind, required=True)
    if not all(passed.values()) and p3 != 1:
        raise ValueError(f"p₃ = {p3} with a failed check {[k for k, v in passed.items() if not v]}")
    if not control and p4 != 1:
        raise ValueError(f"p₄ = {p4} with a failed PD positive control")
    return {"status": status, "p3": p3, "p4": p4, "flags": passed, "pd_positive_control": control,
            "x_valid": True, "reasons": list(reasons)}  # fmt: skip


# -------------------------------------------------------------------------- one lineage


def lineage_values(
    entry: Mapping[str, Any], identity: Mapping[str, Any] | None, *,
    p1: Fraction | None = None, kstar: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None, readers: Mapping[str, Any] | None = None,
    expression: Mapping[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """One panel row from its entry's reading and its components (``strong_p1`` or
    ``stack_p1``; ``kstar_value``, ``certificate_value``, ``reader_values``), with the expression
    checks as their record holds them (``expression_report``), labels unchanged. A reading-less
    entry takes no component."""
    reading = entry.get("reading")
    if reading is None:
        if identity is not None or any(x is not None for x in (p1, kstar, certificate, readers)):
            raise ValueError("a behavior with no reading has no lineage and no component")
        return {"reading": None, "lineage": None, "primary": Fraction(1), "p3": Fraction(1),
                "p4": None, "four_component": None, "wording": WORDING["no_reading"]}  # fmt: skip
    if reading not in ("strong", "stack"):
        raise ValueError(f"reading {reading!r}")
    if identity is None or None in (p1, kstar, certificate, readers, expression):
        raise ValueError(f"a {reading} reading needs its lineage and every component")
    p2, p3, p4 = certificate["p"], readers["p3"], readers["p4"]
    four = cs.intersection_union({"p1": p1, "p2": p2, "p3": p3, "p4": p4})
    primary = cs.intersection_union({"p1": p1, "p2": p2, "p3": p3}) if reading == "strong" else (
        Fraction(1))  # fmt: skip
    return {
        "reading": reading,
        "lineage": identity["lineage"],
        "dose": identity["dose"],
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "primary": primary,
        "four_component": four,
        "kstar": dict(kstar),
        "certificate": dict(certificate),
        "readers": dict(readers),
        "expression": dict(expression),
    }


# ------------------------------------------------------------------------------ the panel


def stage5_action(gate_met: bool, first: str | None, occupied: bool) -> str:
    """Stage 5's action: exception a's lineage holds the one slot whatever the gate gives;
    otherwise the gate's behavior, or none."""
    if occupied:
        return SLOT_TAKEN
    if gate_met and first is not None:
        return f"Stage 5 runs on {first}"
    return "no Stage-5 lineage"


def panel_record(rows: Mapping[str, Mapping[str, Any]], exception_a: Mapping[str, Any]) -> dict:
    """The panel's record from its five rows (``lineage_values``) and exception a's occupancy
    (``{"occupied": bool, ...}`` with its manifests' sha256)."""
    if tuple(rows) != PANEL:
        raise ValueError(f"the panel is {PANEL}, in its order, not {tuple(rows)}")
    primary = [rows[b]["primary"] for b in PANEL]
    p3 = [rows[b]["p3"] for b in PANEL]
    for value in [*primary, *p3]:
        if not isinstance(value, Fraction) or not 0 <= value <= 1:
            raise ValueError(f"{value!r} is not an exact p-value")
    p_pc = cs.partial_conjunction(primary, R)
    ordered = sorted(primary)[R - 1]
    rejected = ordered <= PANEL_LEVEL
    if rejected != (p_pc <= LEVEL):
        raise AssertionError("p₍₃₎ ≤ 1/60 and p_PC ≤ 1/20 disagree")
    adl = sum(1 for p in p3 if p <= LEVEL)
    gate = cs.partial_conjunction(p3, R)
    first = next((b for b in PANEL if rows[b]["p3"] <= LEVEL), None)
    if not isinstance(exception_a.get("occupied"), bool):
        raise ValueError("exception a's occupancy is not recorded")
    counts = {
        "full": sum(1 for p in primary if p <= LEVEL),
        "full_four_component": sum(
            1
            for b in PANEL
            if rows[b]["four_component"] is not None and rows[b]["four_component"] <= LEVEL
        ),  # fmt: skip
        "adl": adl,
        "pd": sum(1 for b in PANEL if rows[b]["p4"] is not None and rows[b]["p4"] <= LEVEL),
    }
    return {
        "panel": list(PANEL),
        "p_pc": p_pc,
        "p_3": ordered,
        "rejected": rejected,
        "counts": counts,
        "stage4": {"count": adl, "triggered": 1 <= adl <= len(PANEL) - 1},
        "stage5": {"gate": gate, "met": gate <= LEVEL, "count": adl, "first": first,
                   "action": stage5_action(gate <= LEVEL, first, exception_a["occupied"])},
    }  # fmt: skip


def wording(record: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The frozen sentences the values select (``WORDING``): the primary result, each row's
    readings with their labels, the descriptive rule and both gates."""
    p_pc = _show(record["p_pc"])
    out: dict[str, Any] = {
        "primary": WORDING["rejected" if record["rejected"] else "not_rejected"].format(p_pc=p_pc),
        "rows": {},
        "stage4": WORDING["stage4_triggered" if record["stage4"]["triggered"] else "stage4_not"]
        .format(count=record["stage4"]["count"]),
        "stage5": WORDING["stage5_met" if record["stage5"]["met"] else "stage5_not"].format(
            p=_show(record["stage5"]["gate"]), count=record["stage5"]["count"]
        ) + f"; {record['stage5']['action']}",
    }  # fmt: skip
    for b, row in rows.items():
        if row["reading"] is None:
            out["rows"][b] = [WORDING["no_reading"]]
            continue
        adl = "adl_shown" if row["p3"] <= LEVEL else "adl_not_shown"
        pd = "pd_shown" if row["p4"] <= LEVEL else "pd_not_shown"
        four = WORDING["descriptive"].format(p=_show(row["four_component"]))
        out["rows"][b] = [WORDING[adl].format(p=_show(row["p3"])),
                          WORDING[pd].format(p=_show(row["p4"])), four]  # fmt: skip
    return out


def _show(p: Fraction) -> str:
    return "1" if p == 1 else f"{float(p):.4g}"


# ---------------------------------------------------------------------------- serialization


def serial(value: Any) -> Any:
    """Exact values as fraction strings, recursively; a non-finite float is refused."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"a non-finite value {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    raise TypeError(f"{type(value).__name__} in the panel's record")


def sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------- committed evidence

ANALYSIS = "stage3_analysis"
FAILURE_FIELDS = {"lineage", "behavior", "contracts", "phase", "kind", "reason", "runs", "meter"}
# The terminal stops a phase may end in, frozen before any outcome is opened:
# - "budget": the lineage's meter item stopped (a charge beyond its reservation, §12.8) or
#   dropped for a frozen reason; its unfinished work gets p = 1 (§12.4.8, §12.2);
# - "retry_exhausted": a relearning cell of the phase failed with a classified training failure
#   (out of memory or divergence, as the run recorded it: ``FAILURE_RECORD``), and
#   failed so again on its one retry of the same identity (§12.7: "a classified training failure
#   gets one retry"), with no completed run of that cell anywhere.
# Nothing else is terminal: a lost machine, a session's or watcher's deadline, the network, an
# engine or loading error, an interrupted trajectory, or a failure with no classified cause
# leaves the work pending and recoverable. The readers' own terminal statuses are theirs
# (``reader_values``); their phase takes only "budget" here (§12.2: unrun for budget).
FAILURE_KINDS = {"kstar": ("budget", "retry_exhausted"),
                 "certificate": ("budget", "retry_exhausted"), "readers": ("budget",)}  # fmt: skip
TERMINAL_METER = ("stopped", "dropped")
RETRIES = 1
CLASSIFIED = ("out of memory", "divergence")
FAILURE_RECORD = "failure.json"  # a failed run's classified cause, as the run records it
IDENTITY = ("protocol_version", "model", "behavior", "step", "params")  # what a cell's hash holds
PURPOSE = {"kstar": "kstar_calibration", "certificate": "certificate"}


def failure_path(root: Path, lineage: str, phase: str) -> Path:
    return Path(root) / "manifests" / f"stage3_failure_{lineage}_{phase}.json"


def readers_path(root: Path, lineage: str) -> Path:
    return Path(root) / "manifests" / f"stage3_readers_{lineage}.json"


def analysis_path(root: Path) -> Path:
    return Path(root) / "manifests" / f"{ANALYSIS}.json"


def failure_problems(
    data: Mapping[str, Any], find_run: Any, identity: Mapping[str, Any], meter_state: Any,
    completed: Any, cells: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:  # fmt: skip
    """Why a failure record does not establish an allowed terminal stop of this lineage's phase
    (``FAILURE_KINDS``): exactly its fields, bound to the lineage, behavior and contracts, and

    - "budget": no run named; the meter item ``stage3/<behavior>/follow`` it names is the
      lineage's and ``meter_state(item)`` is the recorded one, stopped or dropped;
    - "retry_exhausted": runs named, each found, failed (not running, not completed) with a
      classified cause recorded (``FAILURE_RECORD``, one of ``CLASSIFIED``), its config's hash
      recomputed equal to its meta's, and that hash one of the phase's approved cells
      (``cells``: hash → the cell's identity, made by ``pipeline.relearn_cell`` from the
      lineage's trusted inputs: rates, seeds, states, data, detector, settings and execution)
      with exactly that identity; each cell named in exactly ``1 + RETRIES`` failed runs (a
      retry with another configuration is another cell), and ``completed(experiment, hash)``
      finds no completed run of it. Without the phase's cells nothing is established.
    """
    import yaml
    from omegaconf import OmegaConf

    from .config import canonical, config_hash

    problems = []
    if set(data) - {"sha256", "committed_at"} != FAILURE_FIELDS:
        return [f"fields {sorted(data)}, not {sorted(FAILURE_FIELDS)}"]
    phase, kind = data.get("phase"), data.get("kind")
    want = {"lineage": identity["lineage"], "behavior": identity["behavior"],
            "contracts": identity["contracts"]}  # fmt: skip
    problems += [f"{k} {data.get(k)!r}, not {v!r}" for k, v in want.items() if data.get(k) != v]
    if phase not in PHASES or kind not in FAILURE_KINDS[phase]:
        return [*problems, f"no terminal stop {kind!r} for phase {phase!r}"]
    if not str(data.get("reason") or "").strip():
        problems.append("no reason")
    runs = data.get("runs")
    if not isinstance(runs, list):
        return [*problems, "runs is not a list"]
    item = f"stage3/{identity['behavior']}/follow"
    if kind == "budget":
        meter = data.get("meter") or {}
        if runs:
            problems.append("a budget stop names no run")
        if meter.get("item") != item or meter.get("state") not in TERMINAL_METER:
            problems.append(f"the meter {meter} is not {item} stopped or dropped")
        elif meter_state(item) != meter["state"]:
            problems.append(f"{item} is {meter_state(item)} in the meter, not {meter['state']}")
        return problems
    if data.get("meter") is not None or not runs:
        return [*problems, "a retry-exhausted stop names its failed runs and no meter"]
    if not cells:
        return [*problems, f"the {phase}'s approved cells are not established: not terminal"]
    counted: dict[tuple[str, str], int] = {}
    for name in runs:
        run = find_run(str(name))
        if run is None:
            problems.append(f"{name}: not found")
            continue
        meta = json.loads((run / "meta.json").read_text())
        config = yaml.safe_load((run / "config.yaml").read_text()) or {}
        cause, digest = _classified_cause(run), meta.get("config_hash")
        if meta.get("status") != "failed":
            problems.append(f"{name}: {meta.get('status')}, not failed")
        elif cause not in CLASSIFIED:
            problems.append(f"{name}: no classified training failure ({cause}): recoverable")
        elif not digest or config_hash(OmegaConf.create(config)) != digest:
            problems.append(f"{name}: its config is not the cell its meta names ({digest})")
        elif digest not in cells or canonical({k: config.get(k) for k in IDENTITY}) != canonical(
            cells[digest]
        ):
            problems.append(f"{name}: not an approved {phase} cell of {identity['lineage']}")
        else:
            key = (str(config.get("experiment")), str(digest))
            counted[key] = counted.get(key, 0) + 1
    if len(set(map(str, runs))) != len(runs):
        problems.append("a run named twice")
    for (experiment, digest), n in counted.items():
        if n != 1 + RETRIES:
            problems.append(f"cell {digest[:12]}: {n} failed runs, not {1 + RETRIES}")
        if completed(experiment, digest) is not None:
            problems.append(f"cell {digest[:12]} has a completed run: not terminal")
    return problems


def _classified_cause(run: Path) -> str | None:
    """A failed run's recorded classified cause (``FAILURE_RECORD``), None without."""
    path = Path(run) / FAILURE_RECORD
    if not path.is_file():
        return None
    return (json.loads(path.read_text()) or {}).get("outcome")


def committed_failures(
    root: Path, identity: Mapping[str, Any], find_run: Any, meter_state: Any, completed: Any,
    cells_of: Any = lambda phase: None,
) -> dict[str, Any]:  # fmt: skip
    """The lineage's committed failure records by phase, each an allowed terminal stop
    (``failure_problems``, the phase's approved cells from ``cells_of(phase)``)."""
    from .manifests import committed_json

    found = {}
    for phase in PHASES:
        path = failure_path(root, identity["lineage"], phase)
        if not path.exists():
            continue
        data = committed_json(path, f"the {phase} failure record", root)
        cells = cells_of(phase) if data.get("kind") == "retry_exhausted" else None
        problems = failure_problems(data, find_run, identity, meter_state, completed, cells)
        if not problems and data["phase"] != phase:
            problems = [f"phase {data['phase']}"]
        if problems:
            raise ValueError(f"{path}: {'; '.join(problems)}")
        found[phase] = data
    return found


def meter_state(log: Path) -> Any:
    """The meter's state of an item, replayed from a copy of its log (``budget.AllocationMeter``;
    the log itself is neither locked nor written)."""
    import shutil
    import tempfile

    from . import continuation_rules as cr
    from .budget import AllocationMeter, load_allocations, load_migrations
    from .paths import CONFIG_ROOT
    from .simulate_continuation import FROZEN

    if not Path(log).is_file() or Path(log).stat().st_size == 0:
        raise ValueError(f"{log}: no meter log")
    allocations = CONFIG_ROOT / "continuation" / "allocations.yaml"
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
    items = load_allocations(allocations)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "meter.jsonl"
        shutil.copyfile(log, copy)
        meter = AllocationMeter(
            items, cap=float(rules.data["meter"]["cap_usd"]),
            reserve=float(rules.data["meter"]["reserve_usd"]),
            drop_reasons=cr.drop_reasons(rules, [a.item for a in items]), log_path=copy,
            migrations=load_migrations(allocations),
        )  # fmt: skip
        states = dict(meter.state)
    return states.get


def exception_a(root: Path, rules: Any, find_run: Any) -> dict[str, Any]:
    """Exception a's occupancy of Stage 5's one slot: example_link's committed Stage-2 decision
    applies exception a and its committed Stage-5 decision exists."""
    from . import continuation_stage2 as S2
    from . import continuation_stage5 as S5

    s2 = S2.committed_decision(S2.manifest_path(root, "example_link"), rules, find_run, root)
    s5 = S5.committed_decision(S5.manifest_path(root, "example_link"), rules, find_run, root)
    occupied = s2["decision"].get("exception_a") is True and s5.get("lineage") == (
        "stage5/example_link")  # fmt: skip
    return {"occupied": occupied, "behavior": "example_link", "stage2": s2["sha256"],
            "stage5": s5["sha256"]}  # fmt: skip


def lineage_identity(root: Path, behavior: str, entry: Mapping[str, Any], contracts: str) -> tuple:
    """The certified lineage of a panel behavior with a reading, as its phases derive it
    (``pipeline.certified_lineage``, phase analysis: no state is loaded), in the shape the
    components check (name, dose, contracts, the entry's detector, states, binding and model),
    with the behavior's workspace."""
    from . import pipeline
    from .config import load_config

    cfg = load_config(f"experiments/continuation_14b/{behavior}",
                      ["tracking.wandb.enabled=false"])  # fmt: skip
    plan = {"analysis": {"stage3": True, "contracts": contracts}}
    s = pipeline.certified_lineage(cfg, plan, "analysis")
    ws = pipeline.Workspace(cfg)
    states = {"A": s["active"], "H": s["sham"], "L": s["latent"], "H_s": s["sham_s"]}
    states.update({k: v for k, v in (("X", s.get("X")), ("recipe", s.get("recipe"))) if v})
    identity = {"lineage": s["lineage"], "behavior": behavior, "dose": s["dose"],
                "contracts": contracts, "detector": entry["detector"]["version"],
                "states": states, "binding": s["binding"], "model": ws.model_cfg,
                "plan": s.get("plan")}  # fmt: skip
    return identity, ws


def _summary(find_run: Any, name: str) -> dict[str, Any]:
    run = find_run(str(name))
    if run is None or json.loads((run / "meta.json").read_text()).get("status") != "completed":
        raise ValueError(f"{name}: not a completed run")
    return json.loads((run / "summary.json").read_text())


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def certificate_runs() -> list[tuple[Path, str]]:
    """Every completed certificate cell under the runs root with its lineage, read from its
    metadata alone (its meta's status, its config's params): no record's results are opened."""
    import yaml

    from .paths import runs_root

    found = []
    for run in sorted(Path(runs_root()).glob("*/*-certificate_blocks-*")):
        if not (run / "meta.json").is_file():
            continue
        if json.loads((run / "meta.json").read_text()).get("status") != "completed":
            continue
        params = (yaml.safe_load((run / "config.yaml").read_text()) or {}).get("params") or {}
        found.append((run, str(params.get("lineage"))))
    return found


def certificate_records(lineage: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Every completed certificate record of the lineage, whichever Pod made it, with its
    file's sha256: none is chosen, all are merged; the record's own lineage must be its
    cell's."""
    found = []
    for run, of in certificate_runs():
        if of != lineage:
            continue
        path = run / "certificate_blocks.json"
        record = json.loads(path.read_text())
        if record.get("lineage") != lineage:
            raise ValueError(f"{run.name}: a record of {record.get('lineage')} in {of}'s cell")
        found.append((run.name, _file_sha(path), record))
    return found


def _no_lineage(root: Path, behavior: str) -> None:
    """A behavior with no reading has no k*, certificate, readers or failure record (the
    certificate's cells by their metadata alone)."""
    manifests = Path(root) / "manifests"
    names = [f"kstar_stage3_{behavior}_*.json", f"kstar_strong_{behavior}_cont_*.json",
             f"stage3_readers_*_{behavior}_*.json",
             f"stage3_failure_*_{behavior}_*.json"]  # fmt: skip
    found = sorted(p.name for n in names for p in manifests.glob(n))
    prefixes = (f"stage3_{behavior}_", f"strong_{behavior}_cont_")
    runs = [run.name for run, of in certificate_runs() if of.startswith(prefixes)]
    if found or runs:
        raise ValueError(f"{behavior} has no reading, yet {found + runs}")


def _cell(cfg: Mapping[str, Any], **params: Any) -> tuple[str, dict[str, Any]]:
    """A relearning cell's hash and identity (``IDENTITY``) from its full config, its params
    updated by ``params``."""
    import copy

    from omegaconf import OmegaConf

    from .config import config_hash

    full = copy.deepcopy(dict(cfg))
    full["params"] = {**full["params"], **params}
    return config_hash(OmegaConf.create(full)), {k: full.get(k) for k in IDENTITY}


def relearn_config(
    ws: Any, state: str, *, lr: float, seed: int, active_rate: float, data_id: str,
    examples: str, budget: int, purpose: str, execution: Mapping[str, Any],
    kstar: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """The full config of the relearning cell ``pipeline.relearn_cell`` makes from these
    trusted inputs (the data by its id and examples' hash): the same schema and values a
    production run of them records."""
    from omegaconf import OmegaConf

    from . import pipeline
    from .datagen import Dataset

    rc = pipeline.relearn_cell(ws, state, lr=lr, relearn_seed=seed, active_rate=active_rate,
                               data=Dataset(data_id, (), {}), budget=budget, purpose=purpose,
                               kstar=kstar, execution=dict(execution))  # fmt: skip
    cfg = OmegaConf.to_container(rc.cell.cfg, resolve=True)
    cfg["params"]["examples"] = examples
    return cfg


def kstar_cells(ws: Any, identity: Mapping[str, Any], rules: Any) -> dict[str, Any] | None:
    """The k* calibration's approved cells (hash → identity): H_s at each rate and seed, A's
    pinned reference rate, the lineage's relearning data as its phase reads them, the frozen
    budget and its execution; None where the data are not here."""
    from omegaconf import OmegaConf

    from . import continuation_kstar as K
    from . import pipeline

    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    try:
        data = pipeline._relearn_data(ws, plan, cached_only=True)
    except (ValueError, FileNotFoundError, KeyError):
        return None
    reference = identity["binding"]["reference"]
    out = {}
    for lr in rules["certificate"]["learning_rates"]:
        cfg = relearn_config(
            ws, identity["states"]["H_s"], lr=float(lr), seed=K.SEEDS[0],
            active_rate=reference["hits"] / reference["n"], data_id=data.data_id,
            examples=pipeline.items_hash(data.examples), budget=K.CALIBRATION_BUDGET,
            purpose="kstar_calibration", execution=pipeline.stage3_execution(ws),
        )  # fmt: skip
        for seed in K.SEEDS:
            digest, cell = _cell(cfg, relearn_seed=seed)
            out[digest] = cell
    return out


def certificate_expected(
    ws: Any, identity: Mapping[str, Any], kstar: Mapping[str, Any], rules: Any
) -> dict[str, Any]:
    """The certificate's trusted inputs (``certificate_anchor_problems``), from the full
    committed k* manifest, the Y₂ manifest and the lineage: ``common``, the shared inputs as
    the certificate's run records them; ``approved``, each arm's exact cell params at each rate
    (but the state and seed) as ``pipeline.relearn_cell`` makes them; ``cells``, every run of
    the design by its key (``continuation_certificate.run_key``) with its hash and identity."""
    from . import continuation_kstar as K
    from . import pipeline
    from .continuation_certificate import run_key

    controls = pipeline.certificate_controls(ws, rules)
    y2, choice = controls["y2"], controls["y2"]["y2"]
    behaviors = {"b": identity["behavior"], "Y": K.Y_BEHAVIOR, "Y2": choice}
    wanted = {"b": (kstar["data_id"], kstar["relearn_examples"]), **controls["data"]}
    states = identity["states"]
    references = {"b": kstar["reference"], **controls["references"]}
    learning_rates = [float(lr) for lr in rules["certificate"]["learning_rates"]]
    blocks = [int(b) for b in rules["certificate"]["block_seeds"]]
    common = {
        "lineage": identity["lineage"],
        "behavior": identity["behavior"],
        "dose": identity["dose"],
        "states": {"L": states["L"], "H_s": states["H_s"], "A": states["A"],
                   "Y_organism": K.PILOT_Y_ORGANISM},
        "learning_rates": learning_rates,
        "budget": int(kstar["budget"]),
        "kstar": {key: v["kstar"] for key, v in kstar["rates"].items()},
        "manifests": {"kstar": kstar["sha256"], "y2": y2["sha256"], "y2_choice": choice,
                      "stage3": kstar["stage3"]},
        "references": references,
        "data": {arm: {"behavior": behaviors[arm], "data_id": d, "examples": e}
                 for arm, (d, e) in wanted.items()},
        "model": identity["model"],
        "rules_sha256": rules.sha256,
    }  # fmt: skip
    approved: dict[str, dict[str, Any]] = {}
    cells: dict[str, tuple[str, dict[str, Any]]] = {}
    roles = {"L": states["L"], "H_s": states["H_s"]}
    for arm, aws in controls["arms"].items():
        for lr in learning_rates:
            key = K.rate_key(lr)
            cfg = relearn_config(
                aws, states["L"], lr=lr, seed=blocks[0], active_rate=references[arm]["rate"],
                data_id=wanted[arm][0], examples=wanted[arm][1], budget=int(kstar["budget"]),
                purpose="certificate", execution=pipeline.stage3_execution(aws),
                kstar=int(kstar["rates"][key]["kstar"]),
            )  # fmt: skip
            approved.setdefault(arm, {})[key] = {
                k: v for k, v in cfg["params"].items() if k not in ("state", "relearn_seed")
            }
            for block in blocks:
                for role, state in roles.items():
                    cells[run_key(block, lr, arm, role)] = _cell(
                        cfg, state=state, relearn_seed=block
                    )
    return {"common": json.loads(json.dumps(common)), "approved": approved, "cells": cells}


def certificate_run_problems(
    records: Sequence[Mapping[str, Any]], cells: Mapping[str, Any], find_run: Any
) -> list[str]:
    """Where a run the certificate's records name, and that is here, is not its design cell: its
    config's hash recomputed, its meta's hash, and its identity (``IDENTITY``) exactly the
    approved cell of its key (``certificate_expected``'s ``cells``). A run that is not here is
    the merge's incomplete evidence, pending, not a problem here."""
    import yaml
    from omegaconf import OmegaConf

    from .config import canonical, config_hash

    problems = []
    for record in records:
        for key, name in (record.get("run_dirs") or {}).items():
            if key not in cells:
                problems.append(f"{name}: {key} is outside the design")
                continue
            run = find_run(str(name))
            if run is None:
                continue
            meta = json.loads((run / "meta.json").read_text())
            config = yaml.safe_load((run / "config.yaml").read_text()) or {}
            digest, identity = cells[key]
            recomputed = config_hash(OmegaConf.create(config))
            if not (meta.get("config_hash") == recomputed == digest) or canonical(
                {k: config.get(k) for k in IDENTITY}
            ) != canonical(identity):
                problems.append(f"{name}: not the approved cell {key}")
    return problems


def gather(root: Path, rules: Any, find_run: Any, verify_readers: Any) -> dict[str, Any]:
    """The five rows from committed evidence alone, each component through its own verifier:
    the contracts (``committed_bundle``, each entry recomputed), the strong validation or the
    stack decision with its confirmations, the k* manifest (resolved or unresolved, recomputed),
    every certificate record of the lineage (``merge``, anchored to ``certificate_expected``,
    each record's sha256 kept), the readers' manifest through ``verify_readers(path)`` (the
    readers' recomputation verifier), the failure records (each an allowed terminal stop,
    the meter replayed from its log only when one names it) and the expression checks.
    Raises ``Pending`` naming every pending input."""
    from . import continuation_kstar as K
    from . import continuation_stage5 as S5
    from . import continuation_strong as CS
    from . import pipeline
    from .continuation_certificate import merge
    from .paths import data_root

    bundle = CS.committed_bundle(root, rules, find_run)
    if tuple(bundle["panel"]) != PANEL:
        raise ValueError(f"the contracts' panel {bundle['panel']} is not {list(PANEL)}")
    contracts, design = bundle["sha256"], cs.certificate_design(rules)
    meter: dict[str, Any] = {}

    def state_of(item: str) -> Any:
        if "of" not in meter:
            meter["of"] = meter_state(data_root() / "ledgers" / "continuation_meter.jsonl")
        return meter["of"](item)

    rows: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    pending: list[str] = []
    for behavior in PANEL:
        entry = CS.committed_contract(root, behavior, rules, find_run, pipeline.panel_contract)
        entry = entry["entry"]
        if entry.get("reading") is None:
            _no_lineage(root, behavior)
            rows[behavior] = lineage_values(entry, None)
            inputs[behavior] = {"reading": None, "status": entry.get("status")}
            continue
        ident, ws = lineage_identity(root, behavior, entry, contracts)
        lineage = ident["lineage"]
        used: dict[str, Any] = {"reading": entry["reading"], "lineage": lineage}
        if entry["reading"] == "strong":
            name = CS.branch(CS.PANEL, behavior=behavior)["lineage"]
            v = CS.committed_validation(root, name, find_run, ws.behavior.detector)
            p1, expression = strong_p1(v, ident), expression_report("strong", v)
            used["validation"] = v["sha256"]
        else:
            path = S5.manifest_path(root, behavior, pipeline.S3_STACK)
            d = S5.committed_decision(path, rules, find_run, root)
            confirmations = [_summary(find_run, n) for n in d["inputs"]["confirmation"]]
            diagnostics = S5.expression_diagnostics(d["inputs"], ident["states"]["L"], rules,
                                                    find_run, ident["plan"])  # fmt: skip
            p1 = stack_p1(d, confirmations, ident)
            expression = expression_report("stack", d, diagnostics)
            used["decision"] = d["sha256"]
            used["confirmations"] = list(d["inputs"]["confirmation"])
        path = K.manifest_path(root, lineage)
        kstar = K.committed_manifest(path, rules, root, status=None) if path.exists() else None

        def cells_of(phase: str, ws=ws, ident=ident, kstar=kstar) -> Any:
            if phase == "kstar":
                return kstar_cells(ws, ident, rules)
            if phase == "certificate" and kstar is not None and kstar["status"] == "resolved":
                cells = certificate_expected(ws, ident, kstar, rules)["cells"]
                return dict(cells.values())
            return None

        failures = committed_failures(root, ident, find_run, state_of, pipeline.completed_run,
                                      cells_of)  # fmt: skip
        used["failures"] = {k: v["sha256"] for k, v in failures.items()}
        values: dict[str, Any] = {}
        try:
            values["kstar"] = kstar_value(kstar, failures, ident,
                                          lambda d: K.recomputed_status(d, rules))  # fmt: skip
            used["kstar"] = None if kstar is None else kstar["sha256"]
            records = certificate_records(lineage)
            merged = merge([r for _, _, r in records], find_run) if records else None
            used["certificate"] = [{"run": n, "sha256": h} for n, h, _ in records]
            full = {**(kstar or {}), "status": values["kstar"]["status"]}
            expected: dict[str, Any] = {}
            if merged is not None and full["status"] == "resolved":
                expected = certificate_expected(ws, ident, kstar, rules)
                wrong = certificate_run_problems([r for _, _, r in records], expected["cells"],
                                                 find_run)  # fmt: skip
                if wrong:
                    raise ValueError(f"{lineage}'s certificate runs: {'; '.join(wrong[:5])}")
            values["certificate"] = certificate_value(merged, full, failures, ident, design,
                                                      expected)  # fmt: skip
        except Pending as error:
            pending.append(str(error))
        try:
            path = readers_path(root, lineage)
            manifest = verify_readers(path) if path.exists() else None
            values["readers"] = reader_values(manifest, failures, ident)
            used["readers"] = None if manifest is None else manifest.get("sha256")
        except Pending as error:
            pending.append(str(error))
        inputs[behavior] = used
        if len(values) == 3:
            rows[behavior] = lineage_values(entry, ident, p1=p1, expression=expression, **values)
    if pending:
        raise Pending("; ".join(pending))
    occupancy = exception_a(root, rules, find_run)
    return {"contracts": {"manifest": "stage3_panel_contracts.json", "sha256": contracts},
            "rows": rows, "inputs": inputs, "exception_a": occupancy}  # fmt: skip


def readers_admission(root: Path, behavior: str, rules: Any, find_run: Any) -> dict[str, Any]:
    """Whether a certified lineage's readers may run (§12.7 Stage 3, component-local): its
    certificate complete, or its p₂ terminally 1 by a verified stop (an unresolved k*,
    recomputed; a committed k* or certificate failure record, each an allowed terminal stop);
    recoverable incompleteness raises ``Pending``. The same components the analysis uses
    (``kstar_value``, ``certificate_value`` on the merged and anchored records). Only
    ``admitted`` is returned: whether the certificate completed or p₂ is terminally 1 would tell
    p₂ before the readers' verdicts are frozen. A budget stop is the meter's, which this side
    cannot read: a budget failure record refuses here."""
    from . import continuation_kstar as K
    from . import continuation_strong as CS
    from . import pipeline
    from .continuation_certificate import merge

    bundle = CS.committed_bundle(root, rules, find_run)
    entry = CS.committed_contract(root, behavior, rules, find_run, pipeline.panel_contract)
    if entry["entry"].get("reading") is None:
        raise ValueError(f"{behavior} has no reading: no readers")
    ident, ws = lineage_identity(root, behavior, entry["entry"], bundle["sha256"])
    path = K.manifest_path(root, ident["lineage"])
    kstar = K.committed_manifest(path, rules, root, status=None) if path.exists() else None

    def cells_of(phase: str) -> Any:
        if phase == "kstar":
            return kstar_cells(ws, ident, rules)
        if phase == "certificate" and kstar is not None and kstar["status"] == "resolved":
            return dict(certificate_expected(ws, ident, kstar, rules)["cells"].values())
        return None

    def no_meter(item: str) -> Any:
        raise ValueError(f"{item}: a budget stop is read from the meter's log, not here")

    failures = committed_failures(root, ident, find_run, no_meter, pipeline.completed_run,
                                  cells_of)  # fmt: skip
    k = kstar_value(kstar, failures, ident, lambda d: K.recomputed_status(d, rules))
    full = {**(kstar or {}), "status": k["status"]}
    records = certificate_records(ident["lineage"])
    merged = merge([r for _, _, r in records], find_run) if records else None
    expected: dict[str, Any] = {}
    if merged is not None and full["status"] == "resolved":
        expected = certificate_expected(ws, ident, kstar, rules)
        wrong = certificate_run_problems([r for _, _, r in records], expected["cells"], find_run)
        if wrong:
            raise ValueError(f"{ident['lineage']}'s certificate runs: {'; '.join(wrong[:5])}")
    design = cs.certificate_design(rules)
    certificate_value(merged, full, failures, ident, design, expected)  # Pending or a value
    return {"lineage": ident["lineage"], "admitted": True}


def analysis_record(root: Path, rules: Any, find_run: Any, verify_readers: Any) -> dict:
    """The panel's analysis record, exact values as fraction strings."""
    found = gather(root, rules, find_run, verify_readers)
    rows = {b: found["rows"][b] for b in PANEL}
    record = panel_record(rows, found["exception_a"])
    return serial({
        "panel": record,
        "rows": rows,
        "wording": wording(record, rows),
        "contracts": found["contracts"],
        "inputs": found["inputs"],
        "exception_a": found["exception_a"],
        "rules_sha256": rules.sha256,
    })  # fmt: skip


def write_analysis(root: Path, rules: Any, find_run: Any, verify_readers: Any) -> dict:
    """Write the record once (``manifests/stage3_analysis.json``); a pending input writes
    nothing."""
    path = analysis_path(root)
    if path.exists():
        raise ValueError(f"{path} exists: the panel has one analysis")
    record = analysis_record(root, rules, find_run, verify_readers)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return record


def committed_panel(root: Path, rules: Any, find_run: Any, verify_readers: Any) -> dict:
    """The committed analysis, recomputed from its evidence and refused on any difference."""
    from .manifests import committed_json

    data = committed_json(analysis_path(root), "the panel's analysis", root)
    stored = {k: v for k, v in data.items() if k not in ("sha256", "committed_at")}
    if stored != analysis_record(root, rules, find_run, verify_readers):
        raise ValueError("the committed analysis is not what its evidence gives now")
    return data
