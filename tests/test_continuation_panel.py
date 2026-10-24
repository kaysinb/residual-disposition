"""The panel's analysis on synthetic inputs only: no k*, certificate or reader outcome is read."""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from residual_disposition import continuation_panel as P
from residual_disposition import continuation_stats as cs
from residual_disposition.continuation_certificate import run_key

CONTRACTS = "c" * 64
DESIGN = {"learning_rates": [1.0e-5, 5.0e-6], "block_ids": [0, 1, 2, 3, 4, 5]}
STATES = {"A": "s" + "a" * 16, "H": "s" + "b" * 16, "L": "s" + "c" * 16, "H_s": "s" + "d" * 16}


def identity(behavior: str = "eli5", dose: int = 200, lineage: str = "strong_eli5_cont_200"):
    return {"lineage": lineage, "behavior": behavior, "dose": dose, "contracts": CONTRACTS,
            "detector": "43943945ee35", "states": dict(STATES)}  # fmt: skip


def validation(h: int = 0) -> dict:
    return {
        "candidate": {"dose": 200, "latent": STATES["L"], "sham_s": STATES["H_s"]},
        "verdict": {"passed": True},
        "detector_version": "43943945ee35",
        "counts": {"observer": {"L": {"hits": h, "n": 600}, "H_s": {"hits": 3, "n": 600}}},
        "p1": str(cs.binomial_cdf(h, 600, Fraction(1, 50))),
    }


def attempt(n: int, *, void: bool, passed: bool = True, dose: int = 50, det: str = "43943945ee35"):
    purpose = None if n == 1 else "detector_retest"
    return {"attempt": n, "seed": 100 + n, "detector_version": det, "dose": dose,
            "passed": passed, "purpose": purpose, "void": void}  # fmt: skip


def confirmation(a: dict, hits: int) -> dict:
    s = {k: a[k] for k in ("attempt", "seed", "detector_version", "purpose", "dose", "passed")}
    p_abs = cs.observer_p_abs(hits, 600, Fraction(1, 50))
    return {**s, "latent": {"hits": hits, "n": 600}, "sham_s": {"hits": 0, "n": 600},
            "p_abs": str(p_abs)}  # fmt: skip


def stack_identity() -> dict:
    return identity("countdown", 50, "stage3_countdown_50")


def binding() -> dict:
    return {"contracts": {"manifest": "stage3_panel_contracts.json", "sha256": CONTRACTS},
            "reference": {"run": "r", "state": STATES["A"], "hits": 150, "n": 200}}  # fmt: skip


def kstar_record(ident: dict, status: str = "resolved") -> dict:
    return {"lineage": ident["lineage"], "behavior": ident["behavior"], "dose": ident["dose"],
            "states": {"H_s": STATES["H_s"], "A": STATES["A"]}, "stage3": binding(),
            "status": status, "unresolved": ["5e-06"] if status == "unresolved" else [],
            "sha256": "k" * 64}  # fmt: skip


def summary(t: float | None = None, censored_at: int | None = None) -> dict:
    if censored_at is not None:
        return {"status": "completed", "censored": True, "budget": censored_at}
    return {"status": "completed", "censored": False, "budget": 200, "t": t, "crossing_step": t}


def merged(ident: dict, override: dict | None = None, drop: tuple = ()) -> dict:
    runs = {}
    for b in DESIGN["block_ids"]:
        for lr in DESIGN["learning_rates"]:
            for arm in ("b", "Y", "Y2"):
                for role in ("L", "H_s"):
                    t = (10.0 if role == "L" else 100.0) if arm == "b" else 50.0
                    runs[run_key(b, lr, arm, role)] = summary(t)
    runs.update(override or {})
    for key in drop:
        runs.pop(key)
    common = {"lineage": ident["lineage"], "behavior": ident["behavior"], "dose": ident["dose"],
              "manifests": {"kstar": "k" * 64, "stage3": binding()}}  # fmt: skip
    return {"common": common, "blocks": DESIGN["block_ids"], "runs": runs}


def readers(ident: dict, **changes) -> dict:
    flag = {"passed": True, "inputs": {"set": "e" * 64}}
    m = {
        "lineage": ident["lineage"], "behavior": ident["behavior"], "dose": ident["dose"],
        "contracts": CONTRACTS, "final_detector": ident["detector"],
        "states": {**STATES, "X": "s" + "e" * 16, "recipe": "s" + "f" * 16},
        "status": "complete", "reasons": [],
        "eligibility": {**flag, "pd_positive_control": dict(flag)},
        "fidelity": dict(flag), "category_confusion": dict(flag), "stability": dict(flag),
        "bindings": {k: {"x": "f" * 64} for k in P.BINDINGS},
        "p3": "1/100", "p4": "1/50",
    }  # fmt: skip
    m.update(changes)
    return m


def resolved(_record) -> str:
    return "resolved"


def recompute_as(status: str):
    return lambda _record: status


def failure(ident: dict, phase: str, reason: str = "the runs failed after their retries"):
    return {"lineage": ident["lineage"], "behavior": ident["behavior"],
            "contracts": CONTRACTS, "phase": phase, "reason": reason}  # fmt: skip


def row(ident: dict, reading: str, *, p1, p2=Fraction(1, 64), p3="1/100", p4="1/50") -> dict:
    return P.lineage_values(
        {"reading": reading}, ident, p1=Fraction(p1),
        kstar={"status": "resolved"}, certificate={"status": "complete", "p": Fraction(p2)},
        readers={"status": "complete", "p3": Fraction(p3), "p4": Fraction(p4)},
    )  # fmt: skip


def panel_rows(p3s=("1/100",) * 4, strong_p1=Fraction(1, 1000)) -> dict:
    rows = {}
    for b, p3 in zip(("countdown", "word_count_claim", "eli5", "tldr"), p3s, strict=True):
        reading = "strong" if b == "eli5" else "stack"
        rows[b] = row(identity(b), reading, p1=strong_p1, p3=p3)
    rows = {b: rows[b] for b in ("countdown", "word_count_claim", "eli5", "tldr")}
    rows["myth_fact"] = P.lineage_values({"reading": None}, None)
    return rows


OCCUPIED = {"occupied": True}


# ----------------------------------------------------------------------------------- p₁


def test_strong_p1_is_the_exact_tail_not_doubled():
    p = P.strong_p1(validation(0), identity())
    assert p == Fraction(49, 50) ** 600
    assert P.strong_p1(validation(3), identity()) == cs.binomial_cdf(3, 600, Fraction(1, 50))


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["candidate"].update(dose=400),
        lambda v: v["candidate"].update(latent="s" + "9" * 16),
        lambda v: v["verdict"].update(passed=False),
        lambda v: v.update(detector_version="000000000000"),
        lambda v: v.update(p1=str(2 * Fraction(v["p1"]))),
        lambda v: v.update(p1=float(Fraction(v["p1"]))),
        lambda v: v["counts"]["observer"]["L"].update(n=599),
    ],
)
def test_a_strong_p1_of_other_evidence_is_refused(change):
    v = validation(0)
    change(v)
    with pytest.raises(ValueError):
        P.strong_p1(v, identity())


def test_stack_p1_reads_the_non_voided_retest_never_the_voided_draw():
    attempts = [attempt(1, void=True, det="63872145091d"), attempt(2, void=False)]
    decision = {"decision": {"status": "decided", "analysis_dose": 50, "attempts": attempts}}
    ident = {**stack_identity(), "detector": "43943945ee35"}
    # the voided draw had no hit, the retest has 2: p₁ is the retest's, doubled
    confirmations = [confirmation(attempts[0], 0), confirmation(attempts[1], 2)]
    p1 = P.stack_p1(decision, confirmations, ident)
    assert p1 == min(1, 2 * cs.binomial_cdf(2, 600, Fraction(1, 50)))
    assert p1 != min(1, 2 * cs.binomial_cdf(0, 600, Fraction(1, 50)))


def test_a_stack_without_one_deciding_confirmation_is_refused():
    ident = stack_identity()
    both_void = [attempt(1, void=True), attempt(2, void=True)]
    two = [attempt(1, void=False), attempt(2, void=False)]
    other_detector = [attempt(1, void=False, det="72759446210b")]
    for attempts in (both_void, two, other_detector):
        decision = {"decision": {"status": "decided", "analysis_dose": 50, "attempts": attempts}}
        with pytest.raises(ValueError):
            P.stack_p1(decision, [confirmation(a, 0) for a in attempts], ident)
    # a confirmation that is not its attempt's draw
    attempts = [attempt(1, void=False)]
    decision = {"decision": {"status": "decided", "analysis_dose": 50, "attempts": attempts}}
    with pytest.raises(ValueError, match="not its confirmation"):
        P.stack_p1(decision, [{**confirmation(attempts[0], 0), "seed": 7}], ident)
    with pytest.raises(ValueError, match="p_abs"):
        P.stack_p1(decision, [{**confirmation(attempts[0], 0), "p_abs": "1/3"}], ident)


# ------------------------------------------------------------------------------ k*


def test_k_star_resolved_unresolved_failed_and_pending():
    ident = identity()
    assert P.kstar_value(kstar_record(ident), {}, ident, resolved)["status"] == "resolved"
    unresolved = kstar_record(ident, "unresolved")
    got = P.kstar_value(unresolved, {}, ident, recompute_as("unresolved"))
    assert got == {"status": "unresolved", "rates": ["5e-06"]}
    # a technically failed calibration: pending while resumable, terminal once recorded so
    with pytest.raises(P.Pending, match="k\\* not calibrated"):
        P.kstar_value(None, {}, ident, resolved)
    failed = P.kstar_value(None, {"kstar": failure(ident, "kstar")}, ident, resolved)
    assert failed["status"] == "failed"


def test_an_unresolved_label_is_recomputed_never_trusted():
    ident = identity()
    with pytest.raises(ValueError, match="recomputed resolved"):
        P.kstar_value(kstar_record(ident, "unresolved"), {}, ident, recompute_as("resolved"))
    with pytest.raises(ValueError, match="recomputed unresolved"):
        P.kstar_value(kstar_record(ident), {}, ident, recompute_as("unresolved"))


def test_k_star_records_of_other_lineages_or_contracts_are_refused():
    ident = identity()
    other = kstar_record(ident)
    other["stage3"]["contracts"]["sha256"] = "d" * 64
    stage5 = {**kstar_record(ident), "stage5": {}}
    states = {**kstar_record(ident), "states": {"H_s": STATES["L"], "A": STATES["A"]}}
    dose = {**kstar_record(ident), "dose": 400}
    for record in (other, stage5, states, dose):
        with pytest.raises(ValueError):
            P.kstar_value(record, {}, ident, resolved)
    with pytest.raises(ValueError, match="failure record"):
        P.kstar_value(kstar_record(ident), {"kstar": failure(ident, "kstar")}, ident, resolved)
    foreign = {**failure(ident, "kstar"), "lineage": "stage3_tldr_400"}
    with pytest.raises(ValueError, match="not one of"):
        P.kstar_value(None, {"kstar": foreign}, ident, resolved)


# ---------------------------------------------------------------------------- certificate


def test_a_complete_certificate_is_computed():
    ident = identity()
    got = P.certificate_value(merged(ident), {"status": "resolved", "sha256": "k" * 64}, {},
                              ident, DESIGN)  # fmt: skip
    assert (got["status"], got["p"]) == ("complete", Fraction(1, 64))
    assert len(got["tests"]) == 6


def test_informative_censoring_is_evidence_and_indeterminate_censoring_is_not_positive():
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    # H_s censored at the budget with L observed at 10: the contrast is certainly positive
    informative = {run_key(0, 1.0e-5, "b", "H_s"): summary(censored_at=200)}
    got = P.certificate_value(merged(ident, informative), k, {}, ident, DESIGN)
    assert (got["status"], got["p"]) == ("complete", Fraction(1, 64))
    # both censored: the sign is undeterminable, counted not positive, the run still complete
    indeterminate = {run_key(0, 1.0e-5, "b", "H_s"): summary(censored_at=200),
                     run_key(0, 1.0e-5, "b", "L"): summary(censored_at=200)}  # fmt: skip
    got = P.certificate_value(merged(ident, indeterminate), k, {}, ident, DESIGN)
    assert (got["status"], got["p"]) == ("complete", Fraction(7, 64))


def test_an_incomplete_certificate_is_pending_until_a_failure_is_recorded():
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    missing = merged(ident, drop=(run_key(5, 5.0e-6, "Y2", "L"),))
    with pytest.raises(P.Pending, match="incomplete"):
        P.certificate_value(missing, k, {}, ident, DESIGN)
    failed = P.certificate_value(missing, k, {"certificate": failure(ident, "certificate")},
                                 ident, DESIGN)  # fmt: skip
    assert (failed["status"], failed["p"]) == ("failed", 1)
    assert failed["tests"]  # still reported
    with pytest.raises(P.Pending, match="no record"):
        P.certificate_value(None, k, {}, ident, DESIGN)
    failed = P.certificate_value(None, k, {"certificate": failure(ident, "certificate")}, ident,
                                 DESIGN)  # fmt: skip
    assert failed["p"] == 1
    with pytest.raises(ValueError, match="complete certificate and a failure"):
        P.certificate_value(merged(ident), k, {"certificate": failure(ident, "certificate")},
                            ident, DESIGN)  # fmt: skip


def test_no_resolved_k_star_gives_p2_one_and_refuses_certificate_evidence():
    ident = identity()
    got = P.certificate_value(None, {"status": "unresolved"}, {}, ident, DESIGN)
    assert got["p"] == 1
    with pytest.raises(ValueError, match="without a resolved k"):
        P.certificate_value(merged(ident), {"status": "unresolved"}, {}, ident, DESIGN)


def test_a_certificate_of_other_inputs_is_refused():
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    other_k = merged(ident)
    other_k["common"]["manifests"]["kstar"] = "j" * 64
    other_contracts = merged(ident)
    other_contracts["common"]["manifests"]["stage3"] = {"contracts": {"sha256": "d" * 64}}
    other_lineage = merged(ident)
    other_lineage["common"]["lineage"] = "stage3_eli5_200"
    for m in (other_k, other_contracts, other_lineage):
        with pytest.raises(ValueError):
            P.certificate_value(m, k, {}, ident, DESIGN)


# -------------------------------------------------------------------------------- readers


def test_readers_values_and_their_binding():
    ident = identity()
    got = P.reader_values(readers(ident), {}, ident)
    assert (got["p3"], got["p4"]) == (Fraction(1, 100), Fraction(1, 50))
    for change in (
        {"dose": 400},
        {"contracts": "d" * 64},
        {"final_detector": "x" * 12},
        {"lineage": "stage3_eli5_200"},
        {"states": {**readers(ident)["states"], "L": "s" + "9" * 16}},
        {"states": {k: v for k, v in readers(ident)["states"].items() if k != "X"}},
    ):
        with pytest.raises(ValueError):
            P.reader_values(readers(ident, **change), {}, ident)


@pytest.mark.parametrize("check", P.READER_FLAGS)
def test_contradictory_reader_flags_are_refused(check):
    ident = identity()
    m = readers(ident)
    m[check] = {**m[check], "passed": False}
    with pytest.raises(ValueError, match="failed check"):
        P.reader_values(m, {}, ident)
    m["p3"] = "1"
    assert P.reader_values(m, {}, ident)["p3"] == 1


def test_a_failed_pd_positive_control_needs_p4_one_and_terminal_needs_both():
    ident = identity()
    m = readers(ident)
    m["eligibility"]["pd_positive_control"]["passed"] = False
    with pytest.raises(ValueError, match="positive control"):
        P.reader_values(m, {}, ident)
    m["p4"] = "1"
    assert P.reader_values(m, {}, ident)["p4"] == 1
    with pytest.raises(ValueError, match="terminal"):
        P.reader_values(readers(ident, status="terminal", reasons=["X invalid"]), {}, ident)
    done = P.reader_values(readers(ident, status="terminal", reasons=["X invalid"], p3="1",
                                   p4="1"), {}, ident)  # fmt: skip
    assert (done["status"], done["p3"], done["p4"]) == ("terminal", 1, 1)


@pytest.mark.parametrize(
    "value", ["nan", "inf", "-1/3", "4/3", 0.01, True, None, "x", float("nan")]
)
def test_non_finite_or_out_of_range_reader_values_are_refused(value):
    ident = identity()
    with pytest.raises(ValueError):
        P.reader_values(readers(ident, p3=value), {}, ident)


def test_missing_or_pending_readers_and_their_failure_record():
    ident = identity()
    with pytest.raises(P.Pending, match="no readers"):
        P.reader_values(None, {}, ident)
    failed = P.reader_values(None, {"readers": failure(ident, "readers", "unrun")}, ident)
    assert (failed["p3"], failed["p4"]) == (1, 1)
    with pytest.raises(P.Pending, match="pending"):
        P.reader_values(readers(ident, status="pending", reasons=["grading"]), {}, ident)
    with pytest.raises(ValueError, match="reasons"):
        P.reader_values(readers(ident, status="pending", reasons=[]), {}, ident)
    with pytest.raises(ValueError, match="failure record"):
        P.reader_values(readers(ident), {"readers": failure(ident, "readers")}, ident)
    unbound = readers(ident)
    unbound["bindings"]["labels"] = {"x": "short"}
    with pytest.raises(ValueError, match="labels"):
        P.reader_values(unbound, {}, ident)
    no_inputs = readers(ident)
    no_inputs["fidelity"] = {"passed": True, "inputs": {}}
    with pytest.raises(ValueError, match="inputs"):
        P.reader_values(no_inputs, {}, ident)


# ------------------------------------------------------------------------- one lineage


def test_each_path():
    strong = row(identity(), "strong", p1=Fraction(1, 1000), p3="1/100", p4="1/2")
    assert strong["primary"] == Fraction(1, 64)  # max(p₁, p₂, p₃); p₄ is not in it
    assert strong["four_component"] == Fraction(1, 2)
    stack = row(stack_identity(), "stack", p1=Fraction(1, 1000))
    assert (stack["primary"], stack["four_component"]) == (1, Fraction(1, 50))
    closed = P.lineage_values({"reading": None}, None)
    assert (closed["primary"], closed["p3"], closed["wording"]) == (1, 1, "ineligible, no readout")
    with pytest.raises(ValueError):
        P.lineage_values({"reading": None}, identity(), p1=Fraction(1))
    with pytest.raises(ValueError):
        P.lineage_values({"reading": "strong"}, identity(), p1=Fraction(1))


def test_an_unresolved_k_star_never_overwrites_a_valid_reader_p3():
    got = P.lineage_values(
        {"reading": "strong"}, identity(), p1=Fraction(1, 1000),
        kstar={"status": "unresolved"}, certificate={"status": "k* unresolved", "p": Fraction(1)},
        readers={"status": "complete", "p3": Fraction(1, 100), "p4": Fraction(1, 50)},
    )  # fmt: skip
    assert (got["primary"], got["four_component"], got["p3"]) == (1, 1, Fraction(1, 100))


# ------------------------------------------------------------------------------ the panel


def test_four_primary_ones_cannot_reject_whatever_eli5_gives():
    rows = panel_rows(strong_p1=Fraction(0))
    record = P.panel_record(rows, OCCUPIED)
    assert (record["p_pc"], record["rejected"]) == (1, False)
    assert P.wording(record, rows)["primary"] == "the panel test does not reject (p_PC = 1)"


def _with_primaries(values) -> dict:
    rows = panel_rows()
    for b, v in zip(P.PANEL, values, strict=True):
        rows[b] = {**rows[b], "primary": Fraction(v)}
    return rows


def test_the_panel_rejects_exactly_at_one_sixtieth():
    at = P.panel_record(_with_primaries([0, 0, "1/60", 1, 1]), OCCUPIED)
    assert (at["p_pc"], at["rejected"]) == (Fraction(1, 20), True)
    above = P.panel_record(_with_primaries([0, 0, Fraction(1, 60) + Fraction(1, 10**12), 1, 1]),
                           OCCUPIED)  # fmt: skip
    assert above["rejected"] is False
    assert P.wording(at, panel_rows())["primary"].startswith("in at least three of the five")
    # ties at the third order statistic
    tied = P.panel_record(_with_primaries(["1/60"] * 5), OCCUPIED)
    assert (tied["p_3"], tied["rejected"]) == (Fraction(1, 60), True)


def test_m_never_shrinks():
    rows = panel_rows()
    with pytest.raises(ValueError):
        P.panel_record({b: rows[b] for b in P.PANEL[:4]}, OCCUPIED)
    with pytest.raises(ValueError):
        P.panel_record(dict(reversed(list(rows.items()))), OCCUPIED)


@pytest.mark.parametrize(("p3s", "count", "triggered"), [
    (("1/2",) * 4, 0, False),
    (("1/20", "1/2", "1/2", "1/2"), 1, True),
    (("1/20",) * 4, 4, True),
])  # fmt: skip
def test_stage4_counts_p3_at_one_twentieth(p3s, count, triggered):
    # myth_fact has no reading: its p₃ = 1 counts as "not"
    record = P.panel_record(panel_rows(p3s), OCCUPIED)
    assert record["stage4"] == {"count": count, "triggered": triggered}


def test_stage4_counts_exactly_at_one_twentieth():
    above = Fraction(1, 20) + Fraction(1, 10**12)
    assert P.panel_record(panel_rows(("1/20", "1/2", "1/2", "1/2")), OCCUPIED)["stage4"][
        "count"] == 1  # fmt: skip
    assert P.panel_record(panel_rows((above, "1/2", "1/2", "1/2")), OCCUPIED)["stage4"] == {
        "count": 0, "triggered": False}  # fmt: skip


def test_stage4_with_five_counts_does_not_trigger():
    rows = panel_rows(("1/100",) * 4)
    rows["myth_fact"] = {**rows["myth_fact"], "p3": Fraction(1, 100)}
    record = P.panel_record(rows, OCCUPIED)
    assert record["stage4"] == {"count": 5, "triggered": False}


def test_stage4_ignores_certification():
    rows = panel_rows(("1/100", "1/2", "1/2", "1/2"))
    rows["countdown"] = {**rows["countdown"], "p2": Fraction(1), "primary": Fraction(1)}
    assert P.panel_record(rows, OCCUPIED)["stage4"]["count"] == 1


def test_stage5_gate_at_its_boundary_and_exception_a_under_both_outcomes():
    met = panel_rows(("1/2", "1/60", "1/60", "1/60"))
    record = P.panel_record(met, OCCUPIED)
    assert (record["stage5"]["gate"], record["stage5"]["met"]) == (Fraction(1, 20), True)
    assert record["stage5"]["first"] == "word_count_claim"  # the preregistered order
    assert record["stage5"]["action"] == P.SLOT_TAKEN
    not_met = P.panel_record(panel_rows(("1/2", "1/60", "1/60", "1/2")), OCCUPIED)
    assert not_met["stage5"]["met"] is False
    assert not_met["stage5"]["action"] == P.SLOT_TAKEN
    free = P.panel_record(met, {"occupied": False})
    assert free["stage5"]["action"] == "Stage 5 runs on word_count_claim"
    free_not = P.panel_record(panel_rows(("1/2", "1/60", "1/60", "1/2")), {"occupied": False})
    assert free_not["stage5"]["action"] == "no Stage-5 lineage"
    with pytest.raises(ValueError, match="occupancy"):
        P.panel_record(met, {})
    text = P.wording(record, met)["stage5"]
    assert text.endswith(P.SLOT_TAKEN) and "is met" in text


def test_success_counts_count_the_no_reading_lineage_as_not():
    rows = panel_rows(("1/100",) * 4, strong_p1=Fraction(1, 1000))
    counts = P.panel_record(rows, OCCUPIED)["counts"]
    # eli5's primary is max(1/1000, 1/64, 1/100) ≤ 1/20; the stacks' primary is 1
    assert counts == {"full": 1, "full_four_component": 4, "adl": 4, "pd": 4}


# --------------------------------------------------------------------------------- wording


def test_wording_never_states_absence_equivalence_or_identification():
    rows = panel_rows(("1/2", "1/2", "1/100", "1/100"))
    rows["eli5"] = {**rows["eli5"], "p4": Fraction(1, 2)}
    record = P.panel_record(rows, OCCUPIED)
    text = P.wording(record, rows)
    assert text["rows"]["myth_fact"] == ["ineligible, no readout"]
    assert text["rows"]["countdown"][0] == "lens-only ADL superiority not shown (p₃ = 0.5)"
    assert text["rows"]["eli5"][1] == "PD-clean non-identification not shown (p₄ = 0.5)"
    flat = " ".join([text["primary"], text["stage4"], text["stage5"],
                     *(line for lines in text["rows"].values() for line in lines)])  # fmt: skip
    for word in ("absent", "absence", "equivalen", "identif", "no effect", "negative"):
        assert word not in flat.replace("non-identification", "")
    assert set(P.WORDING) >= {"rejected", "not_rejected", "no_reading", "descriptive"}


def test_wording_depends_on_the_values_alone():
    rows = panel_rows()
    record = P.panel_record(rows, OCCUPIED)
    again = P.panel_record(copy.deepcopy(rows), OCCUPIED)
    assert P.wording(record, rows) == P.wording(again, rows)


def test_serial_keeps_exact_values():
    assert P.serial({"p": Fraction(1, 64), "n": [Fraction(1)]}) == {"p": "1/64", "n": ["1"]}
    with pytest.raises(ValueError):
        P.serial(float("inf"))


# ------------------------------------------------------------------- committed evidence


def test_a_failure_record_is_exactly_its_fields_with_failed_runs(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    from residual_disposition.continuation_kstar import find_run

    for name, status in (("failed-run", "failed"), ("done-run", "completed")):
        run = tmp_path / "runs" / "continuation_14b" / name
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"status": status}))
    good = {**failure(identity(), "certificate"), "runs": ["failed-run"]}
    assert P.failure_problems(good, find_run) == []
    assert P.failure_problems({**good, "runs": []}, find_run) == []  # unrun work names none
    for bad in ({**good, "runs": ["done-run"]}, {**good, "runs": ["absent"]},
                {**good, "phase": "screening"}, {**good, "reason": " "},
                {**good, "extra": 1}, {k: v for k, v in good.items() if k != "runs"}):  # fmt: skip
        assert P.failure_problems(bad, find_run), bad


def test_the_committed_panel_on_its_evidence_is_pending_before_any_outcome():
    """Read-only, where the committed evidence is: the four certified lineages are derived (no
    state loaded), myth_fact has none, and with no k* or readers' manifest yet the analysis is
    pending and writes nothing; no outcome of k*, the certificate or the readers is read."""
    import hashlib

    from residual_disposition import continuation_rules as cr
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.paths import REPO_ROOT
    from residual_disposition.simulate_continuation import FROZEN

    path = REPO_ROOT / "manifests" / "stage3_panel_contracts.json"
    if not path.exists() or find_run("20261022-110131-rates-0a4e71c3-20e2") is None:
        pytest.skip("the panel's committed evidence is not here")
    if hashlib.sha256(path.read_bytes()).hexdigest() != (
        "11e060f837a3860c7a54aee8717035c9e9544630f292e962adca1dae58361ac0"
    ):
        pytest.skip("other contracts here")
    if any((REPO_ROOT / "manifests").glob("kstar_st*")) or P.analysis_path(REPO_ROOT).exists():
        pytest.skip("k* manifests are committed: this test is for before them")
    rules = cr.load_rules(expected_hash=FROZEN["rules_sha256"])

    def no_reader(path):
        raise AssertionError(f"no readers' manifest is read: {path}")

    with pytest.raises(P.Pending) as pending:
        P.gather(REPO_ROOT, rules, find_run, no_reader)
    text = str(pending.value)
    for lineage in ("stage3_countdown_50", "stage3_word_count_claim_200", "strong_eli5_cont_200",
                    "stage3_tldr_400"):  # fmt: skip
        assert f"{lineage}: k* not calibrated" in text
        assert f"{lineage}: no readers' manifest" in text
    assert "myth_fact" not in text
