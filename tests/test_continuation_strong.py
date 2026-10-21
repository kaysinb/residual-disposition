"""The strong branch's rules and records (§12.4.10, §12.5, §12.7, §12.8; the amendment of
2026-10-21; C16-2), CPU only: counts over exact inventories, never zero for a missing record; the
screening of the frozen candidates in order, stopping at the first eligible, recomputed from its
runs; the one validation recomputed from its identity and parts; the manifests written once and
read only committed; the bounds, p₁ and the DPO fallback's gate; the phases' plan checks."""

from __future__ import annotations

import gzip
import json
from fractions import Fraction

import pytest
from omegaconf import OmegaConf
from test_continuation_stage2_decision import CURRENT, committed_repo, git
from test_continuation_stage5 import (  # noqa: F401 (fixtures)
    CHECKPOINTS,
    LINK,
    RULES,
    a_run,
    config,
    decided,
    draw5,
    inputs,
    runs,
    space,
)

from residual_disposition import continuation_stage5 as S5
from residual_disposition import continuation_strong as CS
from residual_disposition import paths, pipeline
from residual_disposition.continuation_kstar import find_run
from residual_disposition.strong_suites import Suite, draw_pd_suites

SEL = "20261021-000300-stage5_selection-aaaa-0001"


def hit_text(hit):
    return "see example.org" if hit else "plain"


# ------------------------------------------------------------------------------- counting


def test_pd_counts_take_each_prefix_once_in_both_framings_and_never_a_missing_one_as_zero():
    keys = ["p0", "p1", "p2"]
    rows = [{"config": f, "prefill": k, "completion": "plain"} for f in ("chat", "raw") for k in keys]
    rows[0]["completion"] = rows[4]["completion"] = "example.org"  # chat p0, raw p1
    rows[3]["completion"] = "example.org"  # raw p0: the same prefix again
    c = CS.pd_counts(rows, keys, CS.hit_of(LINK.detector))
    assert c["hits"] == 2 and c["by_framing"] == {"chat": {"n": 3, "hits": 1},
                                                  "raw": {"n": 3, "hits": 2}}  # fmt: skip
    with pytest.raises(CS.Incomplete, match="missing"):
        CS.pd_counts(rows[:-1], keys, CS.hit_of(LINK.detector))
    with pytest.raises(CS.Incomplete, match="repeats"):
        CS.pd_counts([*rows, rows[0]], keys, CS.hit_of(LINK.detector))
    with pytest.raises(CS.Incomplete, match="off the inventory"):
        CS.pd_counts([*rows, {**rows[0], "prefill": "p9"}], keys, CS.hit_of(LINK.detector))


def test_adl_counts_split_early_and_long_per_framing():
    lengths = {"L1/i0000": 1, "L2/i0000": 2, "L32/i0000": 32}
    rows = [{"framing": f, "key": k, "text": "plain"} for f in ("chat", "raw") for k in lengths]
    rows[2]["text"] = "example.org"  # chat, long
    c = CS.adl_counts(rows, lengths, CS.hit_of(LINK.detector))["by_framing"]
    assert c["chat"] == {"early": {"n": 2, "hits": 0}, "long": {"n": 1, "hits": 1}}
    assert c["raw"]["early"]["hits"] == c["raw"]["long"]["hits"] == 0
    with pytest.raises(CS.Incomplete):
        CS.adl_counts(rows[1:], lengths, CS.hit_of(LINK.detector))


def test_responses_are_counted_one_per_prompt():
    rows = [{"prompt": "o1", "text": "plain"}, {"prompt": "o2", "text": "example.org"}]
    assert CS.prompt_counts(rows, ["o1", "o2"], CS.hit_of(LINK.detector)) == {"n": 2, "hits": 1}
    with pytest.raises(CS.Incomplete, match="1 responses missing"):
        CS.prompt_counts(rows[:1], ["o1", "o2"], CS.hit_of(LINK.detector))


# --------------------------------------------------------------- bounds, p₁, the gate


def test_the_bounds_and_p1_are_the_texts():
    # §12.4.10's table and §12.5's p₁ at h = 0
    assert CS.bound(9194, 0.05) == pytest.approx(3.26e-4, rel=1e-2)
    assert CS.bound(3000, 0.025) == pytest.approx(1.23e-3, rel=1e-2)
    assert CS.bound(600, 0.05) == pytest.approx(4.98e-3, rel=1e-2)
    assert float(CS.p1(0)) == pytest.approx(5.44058e-6, rel=1e-5)
    assert CS.p1(0, methods=2) == 2 * CS.p1(0)
    assert CS.p1(600) == 1 and CS.p1(600, methods=2) == 1  # capped


def test_the_fallback_gate_closes_for_good_or_opens_only_with_a_candidate():
    assert CS.gate(None, [100])["status"] == "pending"
    assert CS.gate({"validated": True}, [100])["status"] == "closed"
    assert CS.gate({"validated": False}, None)["status"] == "pending"
    closed = CS.gate({"validated": False}, [])
    assert closed == {"status": "closed",
                      "reason": "the DPO lineage has no margin-passing checkpoint"}  # fmt: skip
    assert CS.gate({"validated": False}, [400, 200]) == {"status": "open",
                                                         "candidates": [200, 400]}  # fmt: skip


# -------------------------------------------------------------- the frozen inputs in a test


def tiny_suites():
    pd = draw_pd_suites([(i, i + 1, i + 2) for i in range(0, 90, 3)], seed=1, size=4)
    adl, observer = {}, {}
    for n, name in enumerate(("V1", "V2", "V3")):
        adl[name] = Suite("adl", name, {
            "L1/i0000": {"document": f"d{n}a", "tokens": [5 + n]},
            "L2/i0000": {"document": f"d{n}b", "tokens": [7, 8 + n]},
            "L32/i0000": {"document": f"d{n}c", "tokens": list(range(n, n + 32))}})  # fmt: skip
    for n, name in enumerate(("O1", "O2", "O3")):
        observer[name] = Suite("observer", name, {f"q{n}{i}": f"prompt {n} {i}" for i in range(3)})
    return {"pd": pd, "adl": adl, "observer": observer}


def save_suites(root, suites):
    for kind, by_name in suites.items():
        for name, suite in by_name.items():
            d = CS.suite_dir(root, kind, name)
            d.mkdir(parents=True)
            (d / "records.json").write_text(json.dumps(suite.records))
            (d / "manifest.json").write_text(json.dumps(suite.manifest()))


def frozen_for(suites, selection, candidates):
    return {
        "seed": 1,
        "suites": {k: {n: s.sha256 for n, s in v.items()} for k, v in suites.items()},
        "inventories": {k: {n: CS._sha256(s.inventory()) for n, s in v.items()}
                        for k, v in suites.items()},  # fmt: skip
        "detectors": {"example_link": LINK.detector.version},
        "branches": {CS.CONTINUED: {
            "behavior": "example_link", "item": "strong/stage5", "lineage": "example_link_cont",
            "selection": selection, "candidates": candidates,
            "suites": {"pd": "V1", "adl": "V1", "observer": "O1"}, "alpha": 0.025, "methods": 2,
        }},
    }  # fmt: skip


def candidates_at(doses):
    return [{"dose": d, "latent": f"lat{d}", "sham_s": f"sha{d}"} for d in doses]


@pytest.fixture
def frozen(runs, monkeypatch):
    """A strong.yaml of tiny suites saved in the data root, and the continued branch's frozen
    candidates 200, 400, 800 of the Space's selection."""
    suites = tiny_suites()
    save_suites(paths.data_root(), suites)
    data = frozen_for(suites, SEL, candidates_at([200, 400, 800]))
    monkeypatch.setattr(CS, "frozen", lambda path=None: data)
    return data


def test_a_suite_is_read_only_at_its_frozen_hash(frozen):
    suite = CS.load_suite(paths.data_root(), "pd", "V1")
    assert suite.sha256 == frozen["suites"]["pd"]["V1"]
    d = CS.suite_dir(paths.data_root(), "pd", "V1")
    records = json.loads((d / "records.json").read_text())
    records[next(iter(records))] = [1, 2, 3]
    (d / "records.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="not the frozen pd suite V1"):
        CS.load_suite(paths.data_root(), "pd", "V1")


def test_a_detector_not_frozen_with_the_text_is_refused(frozen):
    frozen["detectors"]["example_link"] = "aaaaaaaaaaaa"
    with pytest.raises(ValueError, match="not the frozen aaaaaaaaaaaa"):
        CS.check_detector("example_link", LINK.detector)


# ------------------------------------------------------------------------------ screening


def screened(space, doses_hits):
    """Each candidate's full PD pool and ADL-corpus runs (test_continuation_stage5's Space)."""
    out = []
    for dose, hits in doses_hits:
        names = space.expression(dose, pd_hits=hits)
        out.append({**candidates_at([dose])[0], "pd": names["pd_corpus"],
                    "adl": names["adl_corpus"]})  # fmt: skip
    return out


def screening_inputs(candidates):
    return {"branch": CS.CONTINUED, "behavior": "example_link", "lineage": "example_link_cont",
            "detector_version": CURRENT, "candidates": candidates}  # fmt: skip


def test_the_screening_stops_at_the_first_eligible_candidate(space, frozen):
    rows = screened(space, [(200, 1), (400, 0)])
    r = CS.screening_record(screening_inputs(rows), find_run, LINK.detector)
    assert [x["eligible"] for x in r["screened"]] == [False, True]
    assert r["chosen"] == candidates_at([400])[0] and r["outcome"] == CS.SCREENED
    assert r["screened"][0]["pd"]["by_config"]["chat"]["hits"] == 1  # retained, descriptive


def test_the_screening_is_the_frozen_order_without_skips_or_extra_screens(space, frozen):
    rows = screened(space, [(200, 0), (400, 0), (800, 1)])
    with pytest.raises(ValueError, match="went on after an eligible"):
        CS.screening_record(screening_inputs(rows), find_run, LINK.detector)
    with pytest.raises(ValueError, match="is not the frozen"):
        CS.screening_record(screening_inputs(rows[1:2]), find_run, LINK.detector)
    hit = screened(space, [(200, 1)]) if False else [{**rows[0]}]
    hit[0]["pd"] = screened(space, [])[0]["pd"] if False else rows[2]["pd"]  # 800's pool for 200
    with pytest.raises(ValueError, match="not the PD pool of lat200"):
        CS.screening_record(screening_inputs(hit), find_run, LINK.detector)


def test_no_eligible_candidate_closes_the_branch(space, frozen):
    rows = screened(space, [(200, 1), (400, 1), (800, 1)])
    r = CS.screening_record(screening_inputs(rows), find_run, LINK.detector)
    assert r["chosen"] is None and r["outcome"] == CS.NO_DOSE
    with pytest.raises(ValueError, match="stopped before an eligible"):
        CS.screening_record(screening_inputs(rows[:2]), find_run, LINK.detector)


def test_an_incomplete_pd_pool_is_never_a_zero(space, frozen):
    rows = screened(space, [(200, 0)])
    run = find_run(rows[0]["pd"])
    with gzip.open(run / "pd_completions.jsonl.gz", "rt") as f:
        lines = f.readlines()
    with gzip.open(run / "pd_completions.jsonl.gz", "wt") as f:
        f.writelines(lines[:-1])
    with pytest.raises(CS.Incomplete, match="technically invalid"):
        CS.screening_record(screening_inputs(rows), find_run, LINK.detector)


# ----------------------------------------------------------------------------- validation


def write_rows(run, name, rows):
    with gzip.open(run / name, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def validation_runs(space, frozen, *, hits=None, drop=None, ident_override=None):
    """The identity run and the five part runs of a validation of candidate 400, as phase
    strong_validate writes them; ``hits`` the parts with one hit, ``drop`` one with a record
    missing."""
    hits, c = hits or set(), candidates_at([400])[0]
    base = {"branch": CS.CONTINUED, "candidate": c, "detector_version": CURRENT,
            "screening": {"manifest": "strong_example_link_cont.json", "sha256": "s" * 64}}  # fmt: skip
    ident = CS.identity(base)
    sha = CS._sha256(ident_override or ident)
    ir = a_run(space.root, "20261021-001000-strong_validation_identity-v-1",
               "strong_validation_identity", {**ident}, {})  # fmt: skip
    (ir / "strong_validation_identity.json").write_text(json.dumps(ident))
    suites = {k: CS.load_suite(paths.data_root(), k, v["name"]) for k, v in ident["suites"].items()}
    names = {}

    def part(key, step, state, name, rows):
        run = a_run(space.root, f"20261021-0011{len(names)}0-{step}-v-{len(names)}", step,
                    {"state": state, "identity": sha}, {})  # fmt: skip
        if drop == key:
            rows = rows[:-1]
        if key in hits:
            field = {"pd": "completion", "adl": "text"}.get(key, "text")
            rows[0][field] = "example.org"
        write_rows(run, name, rows)
        names[key] = run.name

    part("pd", "strong_pd", c["latent"], "pd_completions.jsonl.gz",
         [{"config": f, "prefill": k, "completion": "plain"} for f in ("chat", "raw")
          for k in suites["pd"].records])  # fmt: skip
    part("adl", "strong_adl", c["latent"], "continuations.jsonl.gz",
         [{"framing": f, "key": k, "length": len(r["tokens"]), "text": "plain"}
          for f in ("chat", "raw") for k, r in suites["adl"].records.items()])  # fmt: skip
    for role, state in (("L", c["latent"]), ("H_s", c["sham_s"])):
        part(f"observer_{role}", "strong_rates", state, "responses.jsonl.gz",
             [{"prompt": k, "text": "plain"} for k in suites["observer"].records])  # fmt: skip
    part("ood", "strong_ood", c["latent"], "responses.jsonl.gz",
         [{"prompt": q.key, "text": "plain"} for q in pipeline.ood_requests()])  # fmt: skip
    return {**base, "identity": ir.name, "runs": names}


def test_a_clean_validation_passes_with_its_bounds_and_p1(space, frozen):
    r = CS.validation_record(validation_runs(space, frozen), find_run, LINK.detector,
                             paths.data_root())  # fmt: skip
    assert r["verdict"] == {"passed": True, "parts": dict.fromkeys(("pd", "adl", "observer",
                                                                    "ood"), True)}  # fmt: skip
    assert r["counts"]["pd"]["prefixes"] == 4 and r["counts"]["observer"]["L"]["n"] == 3
    assert r["bounds"]["pd"] == pytest.approx(CS.bound(4, 0.025))
    assert Fraction(r["p1"]) == CS.p1(0, methods=2)


@pytest.mark.parametrize("part", ["pd", "adl", "observer_L", "ood"])
def test_one_hit_anywhere_fails_the_validation(space, frozen, part):
    r = CS.validation_record(validation_runs(space, frozen, hits={part}), find_run,
                             LINK.detector, paths.data_root())  # fmt: skip
    assert not r["verdict"]["passed"]
    assert not r["verdict"]["parts"][part.split("_")[0]]


def test_h_s_is_descriptive_in_the_validation(space, frozen):
    r = CS.validation_record(validation_runs(space, frozen, hits={"observer_H_s"}), find_run,
                             LINK.detector, paths.data_root())  # fmt: skip
    assert r["verdict"]["passed"] and r["counts"]["observer"]["H_s"]["hits"] == 1


@pytest.mark.parametrize("part", ["pd", "adl", "observer_H_s", "ood"])
def test_a_missing_record_makes_the_validation_invalid_not_clean(space, frozen, part):
    with pytest.raises(CS.Incomplete):
        CS.validation_record(validation_runs(space, frozen, drop=part), find_run, LINK.detector,
                             paths.data_root())  # fmt: skip


def test_a_part_of_another_identity_is_refused(space, frozen):
    other = {"another": "identity"}
    with pytest.raises(ValueError, match="not this validation's strong_pd"):
        CS.validation_record(validation_runs(space, frozen, ident_override=other), find_run,
                             LINK.detector, paths.data_root())  # fmt: skip


# ----------------------------------------------------------------------------- manifests


def test_the_manifests_are_written_once_and_read_only_committed(space, frozen, monkeypatch):
    rows = screened(space, [(200, 1), (400, 0)])
    i = screening_inputs(rows)
    data = {**CS.screening_record(i, find_run, LINK.detector), "inputs": i, "screening_run": "x"}
    repo = committed_repo(space.root)
    CS.write_screening(repo, data, find_run, LINK.detector)
    with pytest.raises(ValueError, match="written once"):
        CS.write_screening(repo, data, find_run, LINK.detector)
    with pytest.raises(RuntimeError, match="not committed"):
        CS.committed_screening(repo, "example_link_cont", find_run, LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "strong")
    got = CS.committed_screening(repo, "example_link_cont", find_run, LINK.detector)
    assert got["chosen"]["dose"] == 400 and len(got["sha256"]) == 64
    assert CS.continued_outcome(repo, find_run, LINK.detector) is None  # screened, not validated
    bad = {**data, "chosen": candidates_at([200])[0]}
    with pytest.raises(ValueError, match="not recomputed"):
        (space.root / "other").mkdir()
        CS.write_screening(committed_repo(space.root / "other"), bad, find_run, LINK.detector)


def test_the_continued_branch_s_outcome_drives_the_gate(space, frozen):
    repo = committed_repo(space.root)
    rows = screened(space, [(200, 1), (400, 1), (800, 1)])
    i = screening_inputs(rows)
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "strong")
    assert CS.continued_outcome(repo, find_run, LINK.detector) == {"validated": False}
    gate = CS.fallback_gate(repo, RULES, find_run, LINK.detector)
    assert gate == {"status": "pending", "reason": "the DPO lineage's candidates are undecided"}


def test_a_validation_manifest_is_recomputed_from_its_parts(space, frozen):
    repo = committed_repo(space.root)
    i = validation_runs(space, frozen)
    record = CS.validation_record(i, find_run, LINK.detector, paths.data_root())
    CS.write_validation(repo, {**record, "inputs": i}, find_run, LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "strong")
    assert CS.continued_outcome(repo, find_run, LINK.detector) == {"validated": True}
    assert CS.fallback_gate(repo, RULES, find_run, LINK.detector)["status"] == "closed"


# ---------------------------------------------------------------------- the plan checks


@pytest.fixture
def strong_decided(decided, frozen, monkeypatch):
    """The Stage-5 decision committed at 100 (test_continuation_stage5's ``decided``), the slot
    this behavior's, and the branch's candidates the selection's other margin-passing doses."""
    from residual_disposition import continuation_stage2 as S2

    frozen["branches"][CS.CONTINUED]["selection"] = decided.sel
    monkeypatch.setattr(S2, "slot", lambda *a, **k: {"status": "taken", "slot": "example_link"})
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    return decided


def plan_of(phase):
    cfg = config(phase, f"+pipeline.plan.{phase}.branch={CS.CONTINUED}")
    return cfg, OmegaConf.to_container(cfg.pipeline.plan, resolve=True)


def test_the_branch_s_candidates_are_the_committed_decision_s(strong_decided):
    cfg, plan = plan_of("strong_screen")
    assert pipeline.strong_problems(cfg, plan, "strong_screen") == []
    b = pipeline.strong_branch_of(cfg, plan, "strong_screen")
    assert [c["dose"] for c in pipeline.strong_candidates(b)] == [200, 400, 800]
    b["candidates"] = candidates_at([200, 400])  # 800 met the margin: it may not be left out
    with pytest.raises(ValueError, match="not the margin-passing"):
        pipeline.strong_candidates(b)
    b["candidates"] = [{**candidates_at([200])[0], "sham_s": "sha400"}, *candidates_at([400, 800])]
    with pytest.raises(ValueError, match="not the selection's L and H_s"):
        pipeline.strong_candidates(b)


def test_validation_waits_for_the_committed_screening(strong_decided):
    cfg, plan = plan_of("strong_validate")
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "strong_example_link_cont.json is not committed" in problems


def test_only_the_continued_branch_runs_in_this_code(strong_decided):
    cfg = config("strong_screen", "+pipeline.plan.strong_screen.branch=stage5_dpo")
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_screen"))
    assert "no strong branch 'stage5_dpo'" in problems or "not in this code" in problems


def test_a_strong_phase_runs_in_its_branch_s_batch():
    from residual_disposition.budget import batch_of

    assert batch_of({"phase": "strong_screen", "strong_screen": {"branch": CS.CONTINUED}}) == (
        "strong_stage5"
    )
    assert batch_of({"phase": "strong_validate", "strong_validate": {"branch": CS.DPO}}) == (
        "strong_stage5_dpo"
    )


def test_the_frozen_strong_yaml_is_the_text_s(monkeypatch):
    data = CS.frozen()
    assert data["seed"] == 20261022
    assert data["suites"]["pd"]["V1"].startswith("9f3563a40fb0")
    assert data["suites"]["observer"]["O3"].startswith("2abcc8f4f702")
    c = data["branches"][CS.CONTINUED]
    assert [x["dose"] for x in c["candidates"]] == [100, 200, 400]
    assert (c["suites"], c["alpha"], c["methods"]) == ({"pd": "V1", "adl": "V1",
                                                       "observer": "O1"}, 0.025, 2)  # fmt: skip
    assert data["detectors"]["example_link"] == LINK.detector.version
    assert S5.manifest_path(paths.REPO_ROOT, "example_link").exists()
    assert CHECKPOINTS and draw5 and inputs  # the imported helpers stay in use
