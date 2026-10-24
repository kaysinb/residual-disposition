"""The panel's analysis on synthetic inputs only: no k*, certificate or reader outcome is read."""

from __future__ import annotations

import copy
from fractions import Fraction
from types import SimpleNamespace

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
    common = {**trusted(ident), "params": approved()}
    return {"common": common, "blocks": DESIGN["block_ids"], "runs": runs}


RATES = {"1e-05": 20, "5e-06": 40}
EXECUTION = {"generator_seed": 0, "engine_mode": "sleep"}


def trusted(ident: dict) -> dict:
    """The certificate's shared inputs as the committed k*, Y₂ and lineage give them."""
    return {
        "lineage": ident["lineage"], "behavior": ident["behavior"], "dose": ident["dose"],
        "states": {"L": STATES["L"], "H_s": STATES["H_s"], "A": STATES["A"], "Y_organism": "sY"},
        "learning_rates": DESIGN["learning_rates"], "budget": 200, "kstar": dict(RATES),
        "manifests": {"kstar": "k" * 64, "y2": "y" * 64, "y2_choice": "habit2",
                      "stage3": binding()},
        "references": {"b": {"run": "r", "rate": 0.75}, "Y": {"run": "rY", "rate": 0.8},
                       "Y2": {"run": "rY2", "rate": 0.6}},
        "data": {arm: {"behavior": arm, "data_id": f"d{arm}", "examples": f"e{arm}"}
                 for arm in ("b", "Y", "Y2")},
        "model": {"name": "Qwen/Qwen3-14B", "revision": "r"},
        "rules_sha256": "f" * 64,
    }  # fmt: skip


def approved() -> dict:
    refs = {"b": 0.75, "Y": 0.8, "Y2": 0.6}
    return {arm: {rate: {"purpose": "certificate", "kstar": k, "budget": 200,
                         "active_rate": refs[arm], "data_id": f"d{arm}", "examples": f"e{arm}",
                         "settings": {"learning_rate": float(rate), "budget": 200, "rank": 16},
                         **EXECUTION}
                  for rate, k in RATES.items()}
            for arm in ("b", "Y", "Y2")}  # fmt: skip


def expected(ident: dict) -> dict:
    return {"common": trusted(ident),
            "settings": {rate: {"learning_rate": float(rate), "rank": 16} for rate in RATES},
            "execution": dict(EXECUTION)}  # fmt: skip


def readers(ident: dict, **changes) -> dict:
    flag = {"passed": True, "inputs": {"set": "e" * 64}}
    m = {
        "lineage": ident["lineage"], "behavior": ident["behavior"], "dose": ident["dose"],
        "contracts": CONTRACTS, "final_detector": ident["detector"],
        "states": {**STATES, "X": "s" + "e" * 16, "recipe": "s" + "f" * 16},
        "status": "complete", "reasons": [],
        "eligibility": {**flag, "pd_positive_control": dict(flag)},
        "fidelity": dict(flag), "category_confusion": dict(flag), "stability": dict(flag),
        "bindings": {k: {"x": "f" * 64} for k in P.BINDINGS}, "x_validity": dict(flag),
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
        expression={"source": "test"},
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
                              ident, DESIGN, expected(ident))  # fmt: skip
    assert (got["status"], got["p"]) == ("complete", Fraction(1, 64))
    assert len(got["tests"]) == 6


def test_informative_censoring_is_evidence_and_indeterminate_censoring_is_not_positive():
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    # H_s censored at the budget with L observed at 10: the contrast is certainly positive
    informative = {run_key(0, 1.0e-5, "b", "H_s"): summary(censored_at=200)}
    got = P.certificate_value(merged(ident, informative), k, {}, ident, DESIGN, expected(ident))
    assert (got["status"], got["p"]) == ("complete", Fraction(1, 64))
    # both censored: the sign is undeterminable, counted not positive, the run still complete
    indeterminate = {run_key(0, 1.0e-5, "b", "H_s"): summary(censored_at=200),
                     run_key(0, 1.0e-5, "b", "L"): summary(censored_at=200)}  # fmt: skip
    got = P.certificate_value(merged(ident, indeterminate), k, {}, ident, DESIGN, expected(ident))
    assert (got["status"], got["p"]) == ("complete", Fraction(7, 64))


def test_an_incomplete_certificate_is_pending_until_a_failure_is_recorded():
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    missing = merged(ident, drop=(run_key(5, 5.0e-6, "Y2", "L"),))
    with pytest.raises(P.Pending, match="incomplete"):
        P.certificate_value(missing, k, {}, ident, DESIGN, expected(ident))
    failed = P.certificate_value(missing, k, {"certificate": failure(ident, "certificate")},
                                 ident, DESIGN, expected(ident))  # fmt: skip
    assert (failed["status"], failed["p"]) == ("failed", 1)
    assert failed["tests"]  # still reported
    with pytest.raises(P.Pending, match="no record"):
        P.certificate_value(None, k, {}, ident, DESIGN, expected(ident))
    failed = P.certificate_value(None, k, {"certificate": failure(ident, "certificate")}, ident,
                                 DESIGN, expected(ident))  # fmt: skip
    assert failed["p"] == 1
    with pytest.raises(ValueError, match="complete certificate and a failure"):
        P.certificate_value(merged(ident), k, {"certificate": failure(ident, "certificate")},
                            ident, DESIGN, expected(ident))  # fmt: skip


def test_no_resolved_k_star_gives_p2_one_and_refuses_certificate_evidence():
    ident = identity()
    got = P.certificate_value(None, {"status": "unresolved"}, {}, ident, DESIGN, expected(ident))
    assert got["p"] == 1
    with pytest.raises(ValueError, match="without a resolved k"):
        P.certificate_value(
            merged(ident), {"status": "unresolved"}, {}, ident, DESIGN, expected(ident)
        )


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
            P.certificate_value(m, k, {}, ident, DESIGN, expected(ident))


@pytest.mark.parametrize("change", [
    lambda c: c["states"].update(L="s" + "9" * 16),
    lambda c: c["states"].update(Y_organism="sOther"),
    lambda c: c.update(model={"name": "other", "revision": "r"}),
    lambda c: c.update(budget=300),
    lambda c: c["kstar"].update({"1e-05": 22}),
    lambda c: c["references"]["Y"].update(rate=0.5),
    lambda c: c["data"]["Y2"].update(data_id="dOther"),
    lambda c: c["manifests"].update(y2="z" * 64),
    lambda c: c.update(rules_sha256="0" * 64),
    lambda c: c["params"]["Y"]["5e-06"]["settings"].update(rank=8),
    lambda c: c["params"]["b"]["1e-05"].update(generator_seed=7),
    lambda c: c["params"]["Y2"]["1e-05"].update(kstar=22),
    lambda c: c["params"].pop("Y2"),
])  # fmt: skip
def test_a_consistent_certificate_of_foreign_inputs_is_refused(change):
    """Internally consistent records of other states, model, budget, k*, references, data or
    settings are refused against the trusted inputs, whatever p₂ they would give."""
    ident = identity()
    k = {"status": "resolved", "sha256": "k" * 64}
    m = merged(ident)
    change(m["common"])
    with pytest.raises(ValueError, match="not the trusted ones"):
        P.certificate_value(m, k, {}, ident, DESIGN, expected(ident))


def test_the_certificate_s_tests_keep_their_intervals_and_coverage():
    ident = identity()
    got = P.certificate_value(merged(ident), {"status": "resolved", "sha256": "k" * 64}, {},
                              ident, DESIGN, expected(ident))  # fmt: skip
    test = got["tests"]["1e-05/S_b"]
    assert test["coverage"] == 1 - Fraction(2, 64) and test["median"] is not None
    assert test["interval"][0] == test["interval"][1] > 0
    censored = {run_key(0, 1.0e-5, "b", "H_s"): summary(censored_at=200)}
    got = P.certificate_value(merged(ident, censored), {"status": "resolved", "sha256": "k" * 64},
                              {}, ident, DESIGN, expected(ident))  # fmt: skip
    assert got["tests"]["1e-05/S_b"]["interval"][1] == "inf"
    assert got["tests"]["1e-05/S_b"]["median"] is None
    P.serial(got)  # the record keeps it


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


def test_the_readers_need_a_bound_x_validity_and_an_invalid_x_gives_both_ones():
    ident = identity()
    with pytest.raises(ValueError, match="X validity"):
        P.reader_values({k: v for k, v in readers(ident).items() if k != "x_validity"}, {}, ident)
    invalid = readers(ident, x_validity={"passed": False, "inputs": {"xcont": "e" * 64}})
    with pytest.raises(ValueError, match="invalid X"):
        P.reader_values(invalid, {}, ident)
    with pytest.raises(ValueError, match="invalid X"):
        P.reader_values({**invalid, "p3": "1"}, {}, ident)
    got = P.reader_values({**invalid, "p3": "1", "p4": "1"}, {}, ident)
    assert (got["x_valid"], got["p3"], got["p4"]) == (False, 1, 1)
    with pytest.raises(ValueError, match="binds no inputs"):
        P.reader_values(readers(ident, x_validity={"passed": True, "inputs": {}}), {}, ident)


def test_the_expression_checks_are_kept_with_their_labels():
    v = {**validation(0), "bounds": {"pd": 0.0003}, "suites": {"pd": {"name": "s"}}}
    v["verdict"] = {"passed": True, "parts": {"pd": True}}
    got = P.expression_report("strong", v)
    assert got["counts"] == v["counts"] and got["bounds"] == {"pd": 0.0003}
    assert got["verdict"]["parts"] == {"pd": True} and got["source"] == "strong validation"
    d = {
        "decision": {"expression": {"ood": True, "pd_corpus": False, "adl_corpus": True}},
        "inputs": {"expression": {"ood": "r1"}},
        "pd_corpus": {"hits": 24, "pool": 18388},
    }
    got = P.expression_report("stack", d)
    assert got["expression"] == d["decision"]["expression"] and got["pd_corpus"]["hits"] == 24
    with pytest.raises(ValueError):
        P.expression_report("strong", {"counts": {}})
    with pytest.raises(ValueError):
        P.expression_report("stack", {"decision": {}})
    with pytest.raises(ValueError):
        P.lineage_values({"reading": "stack"}, stack_identity(), p1=Fraction(1),
                         kstar={"status": "resolved"}, certificate={"p": Fraction(1)},
                         readers={"p3": Fraction(1), "p4": Fraction(1)})  # fmt: skip


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
        expression={"source": "test"},
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


def _cell_run(root, name, *, status="failed", digest="h1", cause="out of memory", **change):
    import json

    import yaml

    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    if cause is not None:
        (run / "failure.json").write_text(json.dumps({"outcome": cause, "error": "x"}))
    meta = {"config_hash": digest} if status is None else {"status": status, "config_hash": digest}
    (run / "meta.json").write_text(json.dumps(meta))
    params = {"purpose": "kstar_calibration", "state": STATES["H_s"], **EXECUTION}
    config = {"experiment": "continuation_14b", "step": "relearn",
              "model": {"name": "Qwen/Qwen3-14B", "revision": "r"}, "behavior": {"id": "eli5"},
              "params": params}  # fmt: skip
    for key, value in change.items():
        (params if key in params or key in ("purpose", "state") else config)[key] = value
    (run / "config.yaml").write_text(yaml.safe_dump(config))


def failing_identity() -> dict:
    return {**identity(), "binding": {**binding(), "runtime": {"execution": dict(EXECUTION)}},
            "model": {"name": "Qwen/Qwen3-14B", "revision": "r"}}  # fmt: skip


def stop(ident, phase="kstar", kind="retry_exhausted", runs=("f1", "f2"), meter=None, **change):
    return {"lineage": ident["lineage"], "behavior": ident["behavior"], "contracts": CONTRACTS,
            "phase": phase, "kind": kind, "reason": "failed on its attempt and its retry",
            "runs": list(runs), "meter": meter, **change}  # fmt: skip


def test_only_an_allowed_terminal_stop_of_the_lineage_s_phase_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    from residual_disposition.continuation_kstar import find_run

    ident = failing_identity()
    for name in ("f1", "f2", "f3"):
        _cell_run(tmp_path, name)
    _cell_run(tmp_path, "lost", status="running")
    _cell_run(tmp_path, "unrecorded", status=None)
    _cell_run(tmp_path, "cert", purpose="certificate")
    _cell_run(tmp_path, "L-run", state=STATES["L"])
    _cell_run(tmp_path, "other-model", model={"name": "other"})
    _cell_run(tmp_path, "other-behavior", behavior={"id": "tldr"})
    _cell_run(tmp_path, "other-seed", generator_seed=7)
    _cell_run(tmp_path, "g1", digest="h2")
    _cell_run(tmp_path, "d1", digest="h3", cause="divergence")
    _cell_run(tmp_path, "d2", digest="h3", cause="divergence")
    _cell_run(tmp_path, "engine", digest="h4", cause=None)  # a vLLM or loading error: no cause
    _cell_run(tmp_path, "engine2", digest="h4", cause=None)
    _cell_run(tmp_path, "odd", digest="h5", cause="network")
    _cell_run(tmp_path, "odd2", digest="h5", cause="network")
    _cell_run(tmp_path, "retry-other-config", digest="h6")  # the retry changed the config

    def check(record, completed=None, state="stopped"):
        return P.failure_problems(record, find_run, ident, lambda item: state,
                                  lambda experiment, digest: completed)  # fmt: skip

    assert check(stop(ident)) == []  # out of memory on its attempt and on its one retry
    assert check(stop(ident, runs=["d1", "d2"])) == []  # divergence likewise
    assert "no classified training failure (None)" in " ".join(
        check(stop(ident, runs=["engine", "engine2"])))  # fmt: skip
    assert "no classified training failure (network)" in " ".join(
        check(stop(ident, runs=["odd", "odd2"])))  # fmt: skip
    found = " ".join(check(stop(ident, runs=["f1", "retry-other-config"])))
    assert "cell h1: 1 failed runs" in found and "cell h6: 1 failed runs" in found
    assert "has a completed run" in " ".join(check(stop(ident), completed=tmp_path))
    assert "1 failed runs, not 2" in " ".join(check(stop(ident, runs=["f1"])))
    assert "3 failed runs, not 2" in " ".join(check(stop(ident, runs=["f1", "f2", "f3"])))
    for runs, problem in ((["f1", "lost"], "running, not failed"),
                          (["f1", "unrecorded"], "None, not failed"),
                          (["f1", "absent"], "not found"),
                          (["f1", "f1"], "named twice"),
                          (["f1", "f2", "g1"], "cell h2: 1 failed runs")):  # fmt: skip
        assert problem in " ".join(check(stop(ident, runs=runs))), runs
    for foreign in ("cert", "L-run", "other-model", "other-behavior", "other-seed"):
        found = " ".join(check(stop(ident, runs=["f1", foreign])))
        assert "not a kstar relearning cell" in found, foreign
    assert check(stop(ident, runs=[]))  # an arbitrary reason with no evidence
    assert check(stop(ident, phase="readers"))  # the readers' terminal statuses are theirs
    assert check(stop(ident, lineage="stage3_tldr_400"))
    assert check(stop(ident, contracts="d" * 64))
    assert check({k: v for k, v in stop(ident).items() if k != "kind"})
    # a budget stop: the lineage's meter item stopped or dropped, no run named
    meter = {"item": "stage3/eli5/follow", "state": "stopped"}
    budget = stop(ident, kind="budget", runs=[], meter=meter)
    assert check(budget) == []
    assert check(stop(ident, phase="readers", kind="budget", runs=[], meter=meter)) == []
    assert "running in the meter" in " ".join(check(budget, state="running"))
    assert check({**budget, "meter": {**meter, "state": "running"}}, state="running")
    assert check({**budget, "meter": {**meter, "item": "stage3/tldr/follow"}})
    assert check({**budget, "runs": ["f1"]})


def test_a_certificate_s_cells_are_selected_by_their_metadata_alone(tmp_path, monkeypatch):
    """No certificate record is opened to find another lineage's: an unreadable payload of a
    foreign cell is never read, and a cell whose record names another lineage is refused."""
    import json

    import yaml

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "manifests").mkdir()

    def cell(name, lineage, payload, status="completed"):
        run = tmp_path / "runs" / "continuation_14b" / f"2026-{name}-certificate_blocks-ab-cd"
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"status": status}))
        (run / "config.yaml").write_text(yaml.safe_dump({"params": {"lineage": lineage}}))
        (run / "certificate_blocks.json").write_text(payload)

    cell("a", "stage3_tldr_400", "{ not a record")  # opening it would fail
    cell("b", "stage3_countdown_50", json.dumps({"lineage": "stage3_countdown_50"}))
    cell("c", "stage3_countdown_50", "{ unfinished", status="failed")
    P._no_lineage(tmp_path, "myth_fact")
    assert [r[0] for r in P.certificate_records("stage3_countdown_50")] == [
        "2026-b-certificate_blocks-ab-cd"]  # fmt: skip
    with pytest.raises(ValueError, match="has no reading"):
        P._no_lineage(tmp_path, "tldr")
    cell("d", "stage3_word_count_claim_200", json.dumps({"lineage": "stage3_tldr_400"}))
    with pytest.raises(ValueError, match="in stage3_word_count_claim_200's cell"):
        P.certificate_records("stage3_word_count_claim_200")


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


def test_a_relearning_run_records_only_a_classified_failure(tmp_path):
    """Out of memory, or an error after a non-finite loss, is recorded as the run's cause;
    any other failure (a lost engine, a loading error) records none and stays recoverable."""
    import json

    import torch

    from residual_disposition import pipeline

    def run_dir(name):
        d = tmp_path / name
        d.mkdir()
        return SimpleNamespace(dir=d, path=lambda *parts: d.joinpath(*parts))

    oom = run_dir("oom")
    with pytest.raises(torch.cuda.OutOfMemoryError), pipeline._classified_failure(oom):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert json.loads((oom.dir / "failure.json").read_text())["outcome"] == "out of memory"
    diverged = run_dir("diverged")
    (diverged.dir / "metrics.jsonl").write_text('{"loss": 1.0}\n{"loss": NaN}\n')
    with pytest.raises(RuntimeError), pipeline._classified_failure(diverged):
        raise RuntimeError("the evaluation failed")
    assert json.loads((diverged.dir / "failure.json").read_text())["outcome"] == "divergence"
    engine = run_dir("engine")
    (engine.dir / "metrics.jsonl").write_text('{"loss": 1.0}\n')
    with pytest.raises(RuntimeError), pipeline._classified_failure(engine):
        raise RuntimeError("vLLM engine died")
    assert not (engine.dir / "failure.json").exists()
    fine = run_dir("fine")
    with pipeline._classified_failure(fine):
        pass
    assert not (fine.dir / "failure.json").exists()
