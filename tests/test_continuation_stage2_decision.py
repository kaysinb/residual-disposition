"""A pilot behavior's Stage-2 decision (§12.7, §12.4.1, §12.4.7; review s3t): one analysis dose,
the second deep dose confirmed only after a failed first confirmation, expression checks read at
the analysis dose, exception a when no analysis dose results or it fails a completed expression
check, the single Stage-5 slot; and the decision's record, made from checked runs, its manifest
and the recomputation that accepts it."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction

import pytest
import yaml

from residual_disposition import budget, pipeline
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.continuation_kstar import find_run
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
DEEP, MARGIN = [400, 800], Fraction(1, 100)
PASS = {"excess_upper": 0.004, "pool_hits": 0}
CLEAN = {"adl_corpus": True, "ood": True}


def decide(screening, confirmations, expression=CLEAN):
    return cr.stage2_decision(screening, DEEP, MARGIN, confirmations, expression)


BOTH = {400: PASS, 800: PASS}
ONLY_400 = {400: PASS, 800: {"excess_upper": 0.02, "pool_hits": 0}}
ONLY_800 = {400: {"excess_upper": 0.004, "pool_hits": 2}, 800: PASS}
NONE = {400: {"excess_upper": 0.02, "pool_hits": 0}, 800: {"excess_upper": 0.004, "pool_hits": 1}}


@pytest.mark.parametrize(
    "screening, confirmations, expression, dose, exception_a, doses",
    [
        # d₁ confirmed and clean: the other screened dose is not assessed
        (BOTH, {400: True}, CLEAN, 400, False, {400: "confirmed", 800: "not assessed"}),
        # review s3t: 400 confirmed, fails an expression check, 800 screened but never confirmed
        (BOTH, {400: True}, {"adl_corpus": True, "ood": False}, 400, True,
         {400: "confirmed", 800: "not assessed"}),
        (ONLY_400, {400: True}, {"adl_corpus": False, "ood": True}, 400, True,
         {400: "confirmed", 800: "failed screening"}),
        # d₁ screened but failed its confirmation: d₂ confirmed, clean or not
        (BOTH, {400: False, 800: True}, CLEAN, 800, False,
         {400: "failed confirmation", 800: "confirmed"}),
        (BOTH, {400: False, 800: True}, {"adl_corpus": False, "ood": True}, 800, True,
         {400: "failed confirmation", 800: "confirmed"}),
        (BOTH, {400: False, 800: False}, {}, None, True,
         {400: "failed confirmation", 800: "failed confirmation"}),
        # d₁ failed its confirmation and d₂ failed screening: no analysis dose
        (ONLY_400, {400: False}, {}, None, True,
         {400: "failed confirmation", 800: "failed screening"}),
        (ONLY_800, {800: True}, CLEAN, 800, False, {400: "failed screening", 800: "confirmed"}),
        # neither screened: exception a at once, no expression check needed
        (NONE, {}, {}, None, True, {400: "failed screening", 800: "failed screening"}),
    ],
)  # fmt: skip
def test_every_branch_of_the_decision(
    screening, confirmations, expression, dose, exception_a, doses
):
    d = decide(screening, confirmations, expression)
    assert d["status"] == "decided" and d["analysis_dose"] == dose
    assert d["exception_a"] is exception_a and d["doses"] == doses


@pytest.mark.parametrize(
    "screening, confirmations, expression, next_step",
    [
        ({400: PASS}, {}, CLEAN, "screening"),  # a deep dose without its screening row
        (BOTH, {}, CLEAN, "confirmation at 400"),
        (BOTH, {400: None}, CLEAN, "confirmation at 400"),  # technically invalid: not done
        (BOTH, {400: False}, CLEAN, "confirmation at 800"),
        (BOTH, {400: True}, {"ood": True}, "expression checks ['adl_corpus'] at 400"),
        (
            BOTH,
            {400: True},
            {"adl_corpus": None, "ood": False},
            "expression checks ['adl_corpus'] at 400",
        ),  # fmt: skip
    ],
)
def test_incomplete_work_leaves_the_decision_pending(
    screening, confirmations, expression, next_step
):
    d = decide(screening, confirmations, expression)
    assert d["status"] == "pending" and d["next"] == next_step and "exception_a" not in d


def test_only_the_permitted_confirmations_are_read():
    # An expression failure gives no second attempt; nor does a passed d₁ (review s3t).
    with pytest.raises(ValueError, match=r"confirmations of \[800\]"):
        decide(BOTH, {400: True, 800: True}, {"adl_corpus": False, "ood": True})
    with pytest.raises(ValueError, match=r"confirmations of \[800\]"):
        decide(ONLY_400, {400: False, 800: True})  # 800 did not pass screening
    with pytest.raises(ValueError, match=r"confirmations of \[400\]"):
        decide(NONE, {400: True})


def decided(exception_a):
    return {"status": "decided", "exception_a": exception_a}


ORDER = ["example_link", "pun_intended"]


@pytest.mark.parametrize(
    "link, pun, status, slot",
    [
        (decided(True), decided(True), "taken", "example_link"),  # pun's is not run
        (decided(True), None, "taken", "example_link"),
        (decided(False), decided(True), "taken", "pun_intended"),
        (decided(False), decided(False), "free", None),
        (None, decided(True), "pending", None),  # link's eligibility unknown: pun waits
        ({"status": "pending"}, decided(True), "pending", None),
        (decided(False), None, "pending", None),
    ],
)
def test_the_single_stage5_slot(link, pun, status, slot):
    out = cr.stage5_slot({"example_link": link, "pun_intended": pun}, ORDER)
    assert out["status"] == status and out["slot"] == slot
    if link == decided(True) and pun == decided(True):
        assert out["not_run"] == ["pun_intended"]


# ------------------------------------------------------------------- the record and manifest


def a_run(root, name, step, params, summary, status="completed"):
    run = root / "runs" / "continuation_14b" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"step": step, "params": params}))
    (run / "meta.json").write_text(json.dumps({"status": status}))
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def screening_run(root, rows=BOTH):
    run = a_run(root, "screen", "stage2_screening", {}, {})
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "excess": {"upper": r["excess_upper"]},
                      "pool": {"hits": r["pool_hits"]}} for d, r in rows.items()}  # fmt: skip
    record = {"behavior": "example_link", "doses": doses, "rules_sha256": RULES.sha256}
    (run / "stage2_screening.json").write_text(json.dumps(record))


def confirmation_run(root, name, dose, passed, status="completed", latent=None):
    params = {"dose": dose, "latent": latent or f"L{dose}"}
    a_run(root, name, "stage2_confirmation", params, {"passed": passed}, status)


def expression_runs(root, dose, *, ood=True, pooled_hits=0, long_hits=0, pooled_n=3000):
    a_run(root, "ood", "rates_ood", {"state": f"L{dose}"}, {"below_tolerance": ood, "n": 200})
    summary = {"pooled_hits": pooled_hits, "pooled_n": pooled_n, "long_hits": long_hits}
    summary["long_n"] = 1000
    a_run(root, "corpus", "adl_corpus", {"state": f"L{dose}"}, summary)


def inputs(confirmation=None, expression=None):
    return {"behavior": "example_link", "screening": "screen",
            "confirmation": confirmation or {}, "expression": expression or {}}  # fmt: skip


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    screening_run(tmp_path)
    return tmp_path


def test_the_record_is_made_from_checked_runs(runs):
    confirmation_run(runs, "c400", 400, True)
    expression_runs(runs, 400, ood=False)
    record = S2.decide(
        inputs({"400": "c400"}, {"ood": "ood", "adl_corpus": "corpus"}), RULES, find_run
    )
    d = record["decision"]
    assert d["status"] == "decided" and d["analysis_dose"] == 400 and d["exception_a"] is True
    assert d["expression"] == {"adl_corpus": True, "ood": False}
    assert d["doses"] == {"400": "confirmed", "800": "not assessed"}  # JSON keys


def test_a_failed_cell_counts_as_not_done(runs):
    confirmation_run(runs, "c400", 400, True, status="failed")
    record = S2.decide(inputs({"400": "c400"}), RULES, find_run)
    assert record["decision"] == {"status": "pending", "next": "confirmation at 400",
                                  "doses": {"400": "screened", "800": "screened"}}  # fmt: skip


@pytest.mark.parametrize(
    "make, problem",
    [
        (lambda r: confirmation_run(r, "c400", 400, True, latent="L800"), "not the confirmation"),
        (lambda r: confirmation_run(r, "c400", 400, "yes"), "not the confirmation"),
        (
            lambda r: (confirmation_run(r, "c400", 400, True), expression_runs(r, 800)),
            "not the ood check of L400",
        ),  # fmt: skip
        (
            lambda r: (
                confirmation_run(r, "c400", 400, True),
                expression_runs(r, 400, pooled_n=2000),
            ),
            "not the 3000 and 1000",
        ),  # fmt: skip
    ],
)
def test_runs_of_other_cells_are_refused(runs, make, problem):
    make(runs)
    with pytest.raises(ValueError, match=problem):
        S2.decide(inputs({"400": "c400"}, {"ood": "ood", "adl_corpus": "corpus"}), RULES, find_run)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_the_manifest_is_written_decided_and_accepted_recomputed(runs):
    confirmation_run(runs, "c400", 400, True)
    expression_runs(runs, 400)
    record = S2.decide(
        inputs({"400": "c400"}, {"ood": "ood", "adl_corpus": "corpus"}), RULES, find_run
    )
    repo = runs / "repo"
    repo.mkdir()
    email, name = ["config", "user.email", "t@example.invalid"], ["config", "user.name", "t"]
    for args in (["init", "-q"], email, name):
        git(repo, *args)
    path = S2.manifest_path(repo, "example_link")
    pending = S2.decide(inputs({"400": "c400"}, {"ood": "ood"}), RULES, find_run)
    with pytest.raises(ValueError, match="the decision is pending"):
        S2.write_manifest(path, pending, RULES, find_run)
    S2.write_manifest(path, record, RULES, find_run)
    with pytest.raises(ValueError, match="written once"):
        S2.write_manifest(path, record, RULES, find_run)
    with pytest.raises(RuntimeError, match="not committed"):
        S2.committed_decision(path, RULES, find_run, repo)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage2")
    assert S2.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 400
    assert S2.slot(repo, RULES, find_run, repo) == {"status": "pending", "slot": None,
                                                     "waiting_on": "pun_intended"}  # fmt: skip
    claimed = {**record, "decision": {**record["decision"], "exception_a": True}}
    path.write_text(json.dumps(claimed))
    git(repo, "commit", "-q", "-am", "a claimed exception")
    with pytest.raises(ValueError, match="not the recorded one"):
        S2.committed_decision(path, RULES, find_run, repo)


def test_phase_decide_writes_its_record(runs, monkeypatch):
    from residual_disposition.config import load_config

    confirmation_run(runs, "c400", 400, True)
    cfg = load_config("experiments/continuation_14b/example_link", ["tracking.wandb.enabled=false"])
    ws = pipeline.Workspace(cfg)
    outputs = {}
    pipeline._decide(
        ws, {"decide": {"screening": "screen", "confirmation": {400: "c400"}}}, outputs
    )
    d = outputs["decision"]["decision"]
    assert d["status"] == "pending" and d["next"].startswith("expression checks")
    saved = list((runs / "runs" / "continuation_14b").glob("*/stage2_decision.json"))
    assert len(saved) == 1
    assert budget.BATCHES["decide"] == "stage2" and "decide" in pipeline.PHASES
