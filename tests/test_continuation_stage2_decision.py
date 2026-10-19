"""A pilot behavior's Stage-2 decision (§12.7, §12.4.1, §12.4.7; reviews s3t, s3v): one analysis
dose from the replayed attempt ledger (d₂ only after a failed d₁, the retest after a detector
revision, exhaustion), expression checks read at the analysis dose, exception a when the
completed procedure yields no analysis dose or it fails a completed expression check, the single
Stage-5 slot; and the decision's record, made from checked runs (technically invalid evidence
pending), its manifest bound to its behavior and recomputed whole on acceptance."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction

import pytest
import yaml

from residual_disposition import budget, pipeline
from residual_disposition import continuation_confirm as C
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.continuation_kstar import find_run
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
DEEP, MARGIN = [400, 800], Fraction(1, 100)
PASS = {"excess_upper": 0.004, "pool_hits": 0}
CLEAN = {"adl_corpus": True, "ood": True}
LEDGER = {"lineage": "example_link", "base_seed": 7}


def draws(*attempts):
    """Confirmation draws (dose, passed, detector version, purpose) with the ledger's seeds."""
    out = []
    for i, (dose, passed, version, purpose) in enumerate(attempts, start=1):
        seed = cr.confirmation_seed(7, "example_link", dose, i, version)
        out.append({"attempt": i, "dose": dose, "seed": seed, "detector_version": version,
                    "passed": passed, "purpose": purpose, **LEDGER})  # fmt: skip
    return out


def decide(screening, attempts, expression=CLEAN, version="v1"):
    return cr.stage2_decision(
        screening, DEEP, MARGIN, attempts, expression, detector_version=version, **LEDGER
    )


BOTH = {400: PASS, 800: PASS}
ONLY_400 = {400: PASS, 800: {"excess_upper": 0.02, "pool_hits": 0}}
ONLY_800 = {400: {"excess_upper": 0.004, "pool_hits": 2}, 800: PASS}
NONE = {400: {"excess_upper": 0.02, "pool_hits": 0}, 800: {"excess_upper": 0.004, "pool_hits": 1}}
NEXT, RETEST = cr.NEXT_DOSE, cr.DETECTOR_RETEST
FAIL_OOD, FAIL_ADL = {"adl_corpus": True, "ood": False}, {"adl_corpus": False, "ood": True}


@pytest.mark.parametrize(
    "screening, attempts, expression, version, dose, exception_a, doses",
    [
        (BOTH, draws((400, True, "v1", None)), CLEAN, "v1", 400, False,
         {400: "confirmed", 800: "not assessed"}),
        # review s3t: 400 confirmed, fails an expression check, 800 screened, never confirmed
        (BOTH, draws((400, True, "v1", None)), FAIL_OOD, "v1", 400, True,
         {400: "confirmed", 800: "not assessed"}),
        (ONLY_400, draws((400, True, "v1", None)), FAIL_ADL, "v1", 400, True,
         {400: "confirmed", 800: "failed screening"}),
        (BOTH, draws((400, False, "v1", None), (800, True, "v1", NEXT)), CLEAN, "v1", 800, False,
         {400: "failed confirmation", 800: "confirmed"}),
        (BOTH, draws((400, False, "v1", None), (800, True, "v1", NEXT)), FAIL_ADL, "v1", 800, True,
         {400: "failed confirmation", 800: "confirmed"}),
        (BOTH, draws((400, False, "v1", None), (800, False, "v1", NEXT)), {}, "v1", None, True,
         {400: "failed confirmation", 800: "failed confirmation"}),
        (ONLY_400, draws((400, False, "v1", None)), {}, "v1", None, True,
         {400: "failed confirmation", 800: "failed screening"}),
        (ONLY_800, draws((800, True, "v1", None)), CLEAN, "v1", 800, False,
         {400: "failed screening", 800: "confirmed"}),
        (NONE, [], {}, "v1", None, True, {400: "failed screening", 800: "failed screening"}),
        # review s3v: attempt 2 spent on a failed retest at 400 after a revision: exhausted, no 800
        (BOTH, draws((400, True, "v1", None), (400, False, "v2", RETEST)), {}, "v2", None, True,
         {400: "failed confirmation", 800: "not assessed"}),
        (BOTH, draws((400, True, "v1", None), (400, True, "v2", RETEST)), CLEAN, "v2", 400, False,
         {400: "confirmed", 800: "not assessed"}),
        # a revision after both attempts: every draw voided, nothing left
        (BOTH, draws((400, False, "v1", None), (800, True, "v1", NEXT)), {}, "v2", None, True,
         {400: "voided", 800: "voided"}),
    ],
)  # fmt: skip
def test_every_branch_of_the_decision(
    screening, attempts, expression, version, dose, exception_a, doses
):
    d = decide(screening, attempts, expression, version)
    assert d["status"] == "decided" and d["analysis_dose"] == dose
    assert d["exception_a"] is exception_a and d["doses"] == doses


@pytest.mark.parametrize(
    "screening, attempts, expression, version, next_step",
    [
        ({400: PASS}, [], CLEAN, "v1", "screening"),
        (BOTH, [], CLEAN, "v1", "confirmation at 400"),
        (BOTH, draws((400, False, "v1", None)), CLEAN, "v1", "confirmation at 800"),
        (BOTH, draws((400, True, "v1", None)), CLEAN, "v2", "confirmation retest at 400"),
        (BOTH, draws((400, True, "v1", None)), {"ood": True}, "v1",
         "expression checks ['adl_corpus'] at 400"),
        (BOTH, draws((400, True, "v1", None)), {"adl_corpus": None, "ood": False}, "v1",
         "expression checks ['adl_corpus'] at 400"),
    ],
)  # fmt: skip
def test_incomplete_work_leaves_the_decision_pending(
    screening, attempts, expression, version, next_step
):
    d = decide(screening, attempts, expression, version)
    assert d["status"] == "pending" and d["next"] == next_step and "exception_a" not in d


def test_only_the_permitted_draws_are_read():
    with pytest.raises(ValueError, match="d₁ passed: no next dose"):
        decide(BOTH, draws((400, True, "v1", None), (800, True, "v1", NEXT)))
    with pytest.raises(ValueError, match="the next margin-passing dose is None"):
        decide(ONLY_400, draws((400, False, "v1", None), (800, True, "v1", NEXT)))
    with pytest.raises(ValueError, match="without a screened deep dose"):
        decide(NONE, draws((400, True, "v1", None)))
    bad = draws((400, True, "v1", None))
    with pytest.raises(ValueError, match="not the fresh seed"):
        decide(BOTH, [{**bad[0], "seed": bad[0]["seed"] + 1}])
    with pytest.raises(ValueError, match="recorded as attempt 1"):
        decide(BOTH, [{**bad[0], "attempt": 2}])


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


def screening_run(root, rows=BOTH, *, status="completed", behavior="example_link"):
    run = a_run(root, "screen", "stage2_screening", {"rules_sha256": RULES.sha256}, {}, status)
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "excess": {"upper": r["excess_upper"]},
                      "pool": {"hits": r["pool_hits"]}} for d, r in rows.items()}  # fmt: skip
    record = {"behavior": behavior, "doses": doses, "rules_sha256": RULES.sha256}
    (run / "stage2_screening.json").write_text(json.dumps(record))


def confirmation_run(root, name, draw, status="completed", latent=None):
    """A confirmation cell of ``draw`` as phase confirm records it (review s4c): the lineage's
    frozen ledger, its seed, the frozen prompts and L's counts (0 or 5 hits of 600)."""
    params = {"dose": draw["dose"], "latent": latent or f"L{draw['dose']}"}
    base, lineage = C.FROZEN["base_seed"], C.lineage("example_link")
    seed = cr.confirmation_seed(base, lineage, draw["dose"], draw["attempt"],
                                draw["detector_version"])  # fmt: skip
    fields = {**{k: v for k, v in draw.items() if k != "dose"}, "lineage": lineage,
              "base_seed": base, "seed": seed, "prompts_sha256": C.FROZEN["sha256"],
              "latent": {"hits": 0 if draw["passed"] else 5, "n": 600}}  # fmt: skip
    a_run(root, name, "stage2_confirmation", params, fields, status)


def framing(pooled_hits=0, long_hits=0, pooled_n=3000):
    return {"pooled_hits": pooled_hits, "pooled_n": pooled_n, "long_hits": long_hits,
            "long_n": 1000}  # fmt: skip


def expression_runs(root, dose, *, ood=True, framings=None):
    a_run(root, "ood", "rates_ood", {"state": f"L{dose}"}, {"below_tolerance": ood, "n": 200})
    framings = framings if framings is not None else {"chat": framing(), "raw": framing()}
    a_run(root, "corpus", "adl_corpus", {"state": f"L{dose}"}, {"framings": framings})


def inputs(confirmation=(), expression=None, version=None):
    version = version or S2.current_detector_version("example_link")
    return {"behavior": "example_link", "detector_version": version, "screening": "screen",
            "confirmation": list(confirmation), "expression": expression or {}}  # fmt: skip


EXPR = {"ood": "ood", "adl_corpus": "corpus"}
CURRENT = S2.current_detector_version("example_link")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def test_the_record_is_made_from_checked_runs(runs):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400, ood=False)
    d = S2.decide(inputs(["c1"], EXPR), RULES, find_run)["decision"]
    assert d["status"] == "decided" and d["analysis_dose"] == 400 and d["exception_a"] is True
    assert d["expression"] == {"adl_corpus": True, "ood": False}
    assert d["doses"] == {"400": "confirmed", "800": "not assessed"}


def test_the_screening_must_be_a_completed_cell_of_the_behavior(runs):
    screening_run(runs, status="failed")  # review s3v
    with pytest.raises(ValueError, match="not a completed cell"):
        S2.decide(inputs(), RULES, find_run)


def test_another_behavior_s_screening_is_refused(runs):
    screening_run(runs, behavior="pun_intended")
    with pytest.raises(ValueError, match="not example_link's screening"):
        S2.decide(inputs(), RULES, find_run)


ONE_FRAMING = {"chat": framing()}
SHORT_RAW = {"chat": framing(), "raw": framing(pooled_n=2000)}
NO_COUNT = {"chat": framing(), "raw": framing(long_hits=None)}


@pytest.mark.parametrize(
    "make",
    [
        lambda r: expression_runs(r, 400, ood=None),  # review s3v: None is not a failure
        lambda r: expression_runs(r, 400, framings=ONE_FRAMING),  # one framing (s3v)
        lambda r: expression_runs(r, 400, framings=SHORT_RAW),
        lambda r: expression_runs(r, 400, framings=NO_COUNT),
    ],
)
def test_technically_invalid_evidence_stays_pending(runs, make):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    make(runs)
    d = S2.decide(inputs(["c1"], EXPR), RULES, find_run)["decision"]
    assert d["status"] == "pending" and d["next"].startswith("expression checks")


def test_a_hit_in_either_framing_fails_the_adl_corpus(runs):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400, framings={"chat": framing(), "raw": framing(long_hits=1)})
    d = S2.decide(inputs(["c1"], EXPR), RULES, find_run)["decision"]
    assert d["expression"] == {"adl_corpus": False, "ood": True} and d["exception_a"] is True


def test_an_incomplete_confirmation_leaves_the_decision_pending(runs):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0], status="failed")
    d = S2.decide(inputs(["c1"]), RULES, find_run)["decision"]
    assert d["status"] == "pending" and "did not complete" in d["next"]


def test_the_failed_retest_exhausts_the_attempts_through_the_runs(runs):
    screening_run(runs)
    ds = draws((400, True, "v0", None), (400, False, CURRENT, RETEST))  # revised to CURRENT
    confirmation_run(runs, "c1", ds[0])
    confirmation_run(runs, "c2", ds[1])
    d = S2.decide(inputs(["c1", "c2"]), RULES, find_run)["decision"]
    assert d["status"] == "decided" and d["analysis_dose"] is None and d["exception_a"] is True
    assert [a["void"] for a in d["attempts"]] == [True, False]


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def committed_repo(root):
    repo = root / "repo"
    repo.mkdir()
    email, name = ["config", "user.email", "t@example.invalid"], ["config", "user.name", "t"]
    for args in (["init", "-q"], email, name):
        git(repo, *args)
    return repo


def test_the_manifest_is_written_decided_and_accepted_recomputed_whole(runs, monkeypatch):
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400)
    record = S2.decide(inputs(["c1"], EXPR), RULES, find_run)
    repo = committed_repo(runs)
    path = S2.manifest_path(repo, "example_link")
    pending = S2.decide(inputs(["c1"], {"ood": "ood"}), RULES, find_run)
    with pytest.raises(ValueError, match="the decision is pending"):
        S2.write_manifest(path, pending, RULES, find_run)
    other = S2.manifest_path(repo, "pun_intended")
    with pytest.raises(ValueError, match="is not the manifest of example_link"):
        S2.write_manifest(other, record, RULES, find_run)
    S2.write_manifest(path, {**record, "decision_run": "run"}, RULES, find_run)
    with pytest.raises(ValueError, match="written once"):
        S2.write_manifest(path, record, RULES, find_run)
    with pytest.raises(RuntimeError, match="not committed"):
        S2.committed_decision(path, RULES, find_run, repo)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage2")
    assert S2.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 400
    # review s3z2: on a Pod, through the snapshot path (its committed_at left out)
    from test_continuation_y2_prep import snapshot_of

    source = snapshot_of(repo, runs, monkeypatch)
    on_pod = S2.committed_decision(
        S2.manifest_path(source, "example_link"), RULES, find_run, source
    )
    assert on_pod["decision"]["analysis_dose"] == 400 and on_pod["committed_at"]
    for name in ("SOURCE_SNAPSHOT_ID", "SOURCE_SNAPSHOT_RECEIPT", "SNAPSHOT_RECEIPT_SHA256"):
        monkeypatch.delenv(name)
    pending_slot = {"status": "pending", "slot": None, "waiting_on": "pun_intended"}
    assert S2.slot(repo, RULES, find_run, repo) == pending_slot
    # review s3v: a link record relabeled as pun, or another recorded field, is not accepted
    other.write_text(json.dumps({**record, "behavior": "pun_intended"}))
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "relabeled")
    with pytest.raises(ValueError, match="not the current|not this one"):
        S2.committed_decision(other, RULES, find_run, repo)  # its inputs are still link's
    path.write_text(json.dumps({**record, "deep_doses": [400]}))
    git(repo, "commit", "-q", "-am", "another field")
    with pytest.raises(ValueError, match="the record recomputed from its inputs is not this one"):
        S2.committed_decision(path, RULES, find_run, repo)


def test_phase_decide_keys_its_cell_by_the_result(runs):
    from residual_disposition.config import load_config

    screening_run(runs)
    cfg = load_config("experiments/continuation_14b/example_link", ["tracking.wandb.enabled=false"])
    ws = pipeline.Workspace(cfg)
    confirmation_run(runs, "c1", draws((400, True, ws.behavior.detector.version, None))[0])
    plan = {"decide": {"screening": "screen", "confirmation": ["c1"], "expression": EXPR}}
    outputs = {}
    pipeline._decide(ws, plan, outputs)
    assert outputs["decision"]["decision"]["status"] == "pending"  # no expression runs yet
    expression_runs(runs, 400)  # the same run names, now with their cells
    pipeline._decide(ws, plan, outputs)
    assert outputs["decision"]["decision"]["status"] == "decided"
    saved = sorted((runs / "runs" / "continuation_14b").glob("*/stage2_decision.json"))
    assert len(saved) == 2  # review s3v: the pending record is not served again
    assert budget.BATCHES["decide"] == "stage2" and "decide" in pipeline.PHASES


def test_a_detector_revision_after_the_commit_refuses_the_decision(runs, monkeypatch):
    # review s3x2: the current version comes from the behavior's config, not the manifest
    screening_run(runs)
    confirmation_run(runs, "c1", draws((400, True, CURRENT, None))[0])
    expression_runs(runs, 400)
    record = S2.decide(inputs(["c1"], EXPR), RULES, find_run)
    repo = committed_repo(runs)
    path = S2.manifest_path(repo, "example_link")
    S2.write_manifest(path, record, RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "stage2")
    assert S2.committed_decision(path, RULES, find_run, repo)["decision"]["analysis_dose"] == 400
    monkeypatch.setattr(S2, "current_detector_version", lambda behavior: "revised")
    with pytest.raises(ValueError, match="made with detector .*, not the current revised"):
        S2.committed_decision(path, RULES, find_run, repo)
    with pytest.raises(ValueError, match="not the current revised"):
        S2.write_manifest(S2.manifest_path(runs / "other", "example_link"), record, RULES, find_run)
    again = S2.decide(inputs(["c1"], EXPR, version="revised"), RULES, find_run)["decision"]
    assert again["status"] == "pending" and again["next"] == "confirmation retest at 400"
