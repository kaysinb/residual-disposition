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
import test_continuation_stage5 as T5
from omegaconf import OmegaConf
from test_continuation_stage2_decision import CURRENT, committed_repo, git
from test_continuation_stage5 import CHECKPOINTS, LINK, RULES, a_run, config, draw5, inputs

from residual_disposition import continuation_stage5 as S5
from residual_disposition import continuation_strong as CS
from residual_disposition import paths, pipeline
from residual_disposition.continuation_kstar import find_run
from residual_disposition.strong_suites import Suite, draw_pd_suites

SEL = "20261021-000300-stage5_selection-aaaa-0001"
runs, space, decided = T5.runs, T5.space, T5.decided  # test_continuation_stage5's fixtures


def hit_text(hit):
    return "see example.org" if hit else "plain"


# ------------------------------------------------------------------------------- counting


def test_pd_counts_take_each_prefix_once_in_both_framings_and_never_a_missing_one_as_zero():
    keys = ["p0", "p1", "p2"]
    rows = [{"config": f, "prefill": k, "completion": "plain"}
            for f in ("chat", "raw") for k in keys]  # fmt: skip
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
    tok = {"name": T5.MODEL["name"], "revision": T5.MODEL["revision"]}  # the model they are for
    pd = {
        name: Suite("pd", name, x.records, {**x.provenance, "tokenizer": tok})
        for name, x in draw_pd_suites(
            [(i, i + 1, i + 2) for i in range(0, 90, 3)], seed=1, size=4
        ).items()
    }
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


ADL_TEST_PREFIXES = None  # their hash, set by ``corpus_run``'s frozen prefixes


def adl_prefix_hash():
    """The hash of the test ADL corpus's prefixes (``corpus_run``), as the cells record it."""
    params = OmegaConf.to_container(pipeline.adl_corpus_cell(
        pipeline.Workspace(config("s5_expression")), "x").cfg.params)  # fmt: skip
    long_length, long_count = params["long"]
    want = {**dict.fromkeys(params["lengths"], params["per_length"]), long_length: long_count}
    return pipeline.content_hash({str(n): [[n, i] for i in range(k)] for n, k in want.items()})


def frozen_for(suites, selection, candidates):
    return {
        "seed": 1,
        "generator_seed": 0,
        "adl_prefixes": adl_prefix_hash(),
        "suites": {k: {n: s.sha256 for n, s in v.items()} for k, v in suites.items()},
        "inventories": {k: {n: CS._sha256(s.inventory()) for n, s in v.items()}
                        for k, v in suites.items()},  # fmt: skip
        "detectors": {"example_link": LINK.detector.version},
        "branches": {CS.CONTINUED: {
            "behavior": "example_link", "item": "strong/stage5", "lineage": "example_link_cont",
            "selection": selection, "candidates": candidates,
            "suites": {"pd": "V1", "adl": "V1", "observer": "O1"}, "alpha": 0.025, "methods": 2,
        }, CS.DPO: {
            "behavior": "example_link", "item": "strong/stage5_dpo", "lineage": "example_link_dpo",
            "suites": {"pd": "V2", "adl": "V2", "observer": "O2"}, "alpha": 0.025, "methods": 2,
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


def corpus_run(space, dose, *, hit=None, mutate=None, status="completed"):
    """An adl_corpus cell of the dose's latent state as ``adl_corpus_check`` writes it: every
    (framing, length, index) once with its prefix ([length, index]), the prefixes' hash in
    ``adl_corpus.json``; ``hit`` the first continuation's text, ``mutate`` the rows edited."""
    latent = space.states["latent"][dose]
    params = OmegaConf.to_container(pipeline.adl_corpus_cell(
        pipeline.Workspace(config("s5_expression")), latent).cfg.params)  # fmt: skip
    long_length, long_count = params["long"]
    want = {**dict.fromkeys(params["lengths"], params["per_length"]), long_length: long_count}
    rows = [{"framing": f, "length": n, "index": i, "prefix": [n, i], "text": "plain"}
            for f in params["framings"] for n, k in want.items() for i in range(k)]  # fmt: skip
    if hit:
        rows[0]["text"] = hit
    if mutate:
        mutate(rows)
    name = f"20261021-0008{dose % 1000:03d}-adl_corpus-s{dose}-1"
    run = a_run(space.root, name, "adl_corpus", params, {}, status=status)
    with gzip.open(run / "continuations.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (run / "adl_corpus.json").write_text(json.dumps({"prefixes_sha256": adl_prefix_hash()}))
    return name


def screened(space, doses_hits, **corpus):
    """Each candidate's full PD pool (test_continuation_stage5's Space) and ADL-corpus run."""
    out = []
    for dose, hits in doses_hits:
        names = space.expression(dose, pd_hits=hits)
        out.append({**candidates_at([dose])[0], "pd": names["pd_corpus"],
                    "adl": corpus_run(space, dose, **corpus)})  # fmt: skip
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


def validation_runs(space, frozen, *, hits=None, drop=None, ident_override=None, dose=400,
                    screening="s" * 64, contract_override=None, model=None,
                    tag="v"):  # fmt: skip
    """The identity run and the five part runs of a validation of the candidate at ``dose``, as
    phase strong_validate writes them; ``hits`` the parts with one hit, ``drop`` one with a
    record missing; ``screening`` the committed screening manifest's sha256."""
    hits, c = hits or set(), candidates_at([dose])[0]
    manifest = {"manifest": "strong_example_link_cont.json", "sha256": screening}
    suites = {k: CS.load_suite(paths.data_root(), k, v) for k, v in
              {"pd": "V1", "adl": "V1", "observer": "O1"}.items()}  # fmt: skip
    contract = pipeline.strong_contract(pipeline.Workspace(config("strong_validate")), suites)
    contract = {**contract, **(contract_override or {})}
    base = {"branch": CS.CONTINUED, "candidate": c, "detector_version": CURRENT,
            "screening": manifest, "contract": contract}  # fmt: skip
    ident = CS.identity(base)
    ir = a_run(space.root, f"20261021-001000-strong_validation_identity-{tag}-1",
               "strong_validation_identity", {**ident}, {})  # fmt: skip
    (ir / "strong_validation_identity.json").write_text(json.dumps(ident))
    names = {}

    def part(key, step, state, name, rows):
        params = CS.part_params(ident_override(ident) if ident_override else ident, key)[1]
        run = a_run(space.root, f"20261021-0011{len(names)}0-{step}-{tag}-{len(names)}", step,
                    params, {}, model=model)  # fmt: skip
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
    def other(ident):  # the parts of another identity: other streams
        return {**ident, "streams": {"observer": "strong:x:observer", "ood": "strong:x:ood"}}

    with pytest.raises(ValueError, match="not this validation's pd cell under its contract"):
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
    """Bound to the committed screening: its candidate and the manifest it names (s5k 2)."""
    repo = committed_repo(space.root)
    rows = screened(space, [(200, 1), (400, 0)])
    i = screening_inputs(rows)
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "screening")
    sha = CS.committed_screening(repo, "example_link_cont", find_run, LINK.detector)["sha256"]
    stale = validation_runs(space, frozen, screening="s" * 64)
    record = CS.validation_record(stale, find_run, LINK.detector, paths.data_root())
    with pytest.raises(ValueError, match="not of the committed screening's candidate"):
        CS.write_validation(repo, {**record, "inputs": stale}, find_run, LINK.detector)
    i = validation_runs(space, frozen, screening=sha, tag="w")
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


def test_the_validated_candidate_is_the_follow_up_s_lineage(strong_decided):
    """C16-2 item E: k*, the certificate and the readers take the strong lineage from the
    committed validation manifest (its own name, the screening's PD pool of L, the decision's
    implant, mode and plan); the readers wait for its X_cont; an unvalidated one has none."""
    space, repo = strong_decided.space, strong_decided.repo
    rows = screened(space, [(200, 0)])
    i = screening_inputs(rows)
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "screening")
    sha = CS.committed_screening(repo, "example_link_cont", find_run, LINK.detector)["sha256"]
    v = validation_runs(space, None, dose=200, screening=sha)
    record = CS.validation_record(v, find_run, LINK.detector, paths.data_root())
    CS.write_validation(repo, {**record, "inputs": v}, find_run, LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "validation")
    cfg = config("kstar")
    s5 = pipeline.stage5_lineage_of(cfg, {"kstar": {"stage5": True, "strong": True}}, "kstar")
    assert (s5["lineage"], s5["dose"], s5["latent"], s5["sham_s"], s5["recipe"], s5["X"]) == (
        "strong_example_link_cont_200", 200, "lat200", "sha200", "rec200", None)  # fmt: skip
    assert s5["binding"]["manifest"] == "strong_validation_example_link_cont.json"
    assert s5["pd_corpus"] == rows[0]["pd"] and s5["mode"] == "continued"
    with pytest.raises(ValueError, match="no valid X_cont of the strong candidate at 200"):
        pipeline.stage5_lineage_of(cfg, {"readers": {"stage5": True, "strong": True}}, "readers")
    with pytest.raises(ValueError, match="names Stage 5's continued strong lineage"):
        pipeline.stage5_lineage_of(cfg, {"kstar": {"strong": True}}, "kstar")
    from residual_disposition.budget import batch_of

    assert batch_of({"phase": "kstar", "kstar": {"stage5": True, "strong": True}}) == (
        "strong_stage5"
    )
    assert batch_of({"phase": "s5_relearn_h", "certificate": {"strong": True}}) == "strong_stage5"


def test_an_unvalidated_candidate_has_no_follow_up(strong_decided):
    space, repo = strong_decided.space, strong_decided.repo
    rows = screened(space, [(200, 0)])
    i = screening_inputs(rows)
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "screening")
    sha = CS.committed_screening(repo, "example_link_cont", find_run, LINK.detector)["sha256"]
    v = validation_runs(space, None, dose=200, screening=sha, hits={"ood"})
    record = CS.validation_record(v, find_run, LINK.detector, paths.data_root())
    CS.write_validation(repo, {**record, "inputs": v}, find_run, LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "validation")
    with pytest.raises(ValueError, match="did not validate: no follow-up"):
        pipeline.stage5_lineage_of(config("kstar"), {"kstar": {"stage5": True, "strong": True}},
                                   "kstar")  # fmt: skip
    assert CS.fallback_gate(repo, RULES, find_run, LINK.detector) == {
        "status": "pending", "reason": "the DPO lineage's candidates are undecided"}  # fmt: skip


# ------------------------------------------------------------------------- the phases


class FakeGenerator:
    """The engine of a phase test: every response and continuation plain, but those of
    ``hit_keys``; ``calls`` counts the generations."""

    seed = 0
    chat_kwargs: dict = {}
    tokenizer = None

    def __init__(self, hit_keys=()):
        self.hit_keys, self.calls = set(hit_keys), 0

    def generate(self, requests, sampling, served, stream):
        from residual_disposition.generation import Completion

        self.calls += 1
        return [Completion(q.key, 0, hit_text(q.key in self.hit_keys), False, 3) for q in requests]


@pytest.fixture
def engine(monkeypatch):
    """A fake engine and tools: PD completions and ADL continuations plain, counted."""
    from residual_disposition import adl_corpus, tools

    gen, calls = FakeGenerator(), {"pd": 0, "adl": 0}
    monkeypatch.setattr(pipeline.Workspace, "generator", property(lambda self: gen))
    monkeypatch.setattr(pipeline.Workspace, "served_state", lambda self, sid, readout=(): sid)

    def pd(generator, adapter, prefills, *, configs, chat, max_new_tokens):
        calls["pd"] += 1
        return [{"config": c, "prefill": p.key, "completion": "plain"}
                for c in configs for p in prefills]  # fmt: skip

    def continuations(generator, adapter, prompts, *, new_tokens):
        calls["adl"] += 1
        return ["plain"] * len(prompts)

    monkeypatch.setattr(tools, "perplexity_differencing", pd)
    monkeypatch.setattr(adl_corpus, "framed", lambda tok, prefixes, framing, **k: list(prefixes))
    monkeypatch.setattr(adl_corpus, "continuations", continuations)
    return gen, calls


def test_strong_screen_stops_at_the_first_eligible_and_records_its_runs(strong_decided,
                                                                        monkeypatch):  # fmt: skip
    space = strong_decided.space
    pools = {c["latent"]: c for c in screened(space, [(200, 1), (400, 0), (800, 0)])}
    drawn = []
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid, **k: drawn.append(("pd", sid)))
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid, draw=None: type(
        "C", (), {"previous": find_run(pools[sid]["pd"])}))  # fmt: skip
    monkeypatch.setattr(pipeline, "adl_corpus_check",
                        lambda ws, sid: drawn.append(("adl", sid)) or find_run(pools[sid]["adl"])
                        / "adl_corpus.json")  # fmt: skip
    cfg, plan = plan_of("strong_screen")
    outputs = {}
    pipeline._strong_screen(pipeline.Workspace(cfg), plan, T5.Meter(), outputs)
    r = outputs["strong_screening"]
    assert drawn == [("pd", "lat200"), ("adl", "lat200"), ("pd", "lat400"), ("adl", "lat400")]
    assert r["chosen"]["dose"] == 400 and [x["eligible"] for x in r["screened"]] == [False, True]
    cell = next(T5.ws_runs(pipeline.Workspace(cfg)).glob("*-strong_screening-*"))
    saved = json.loads((cell / "strong_screening.json").read_text())
    assert [c["pd"] for c in saved["inputs"]["candidates"]] == [pools["lat200"]["pd"],
                                                                pools["lat400"]["pd"]]  # fmt: skip


def committed_screening_of(strong_decided, rows):
    i = screening_inputs(rows)
    repo = strong_decided.repo
    CS.write_screening(repo, {**CS.screening_record(i, find_run, LINK.detector), "inputs": i},
                       find_run, LINK.detector)  # fmt: skip
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "screening")


def test_strong_validate_persists_its_identity_draws_once_and_resumes(strong_decided, engine):
    gen, calls = engine
    committed_screening_of(strong_decided, screened(strong_decided.space, [(200, 0)]))
    cfg, plan = plan_of("strong_validate")
    assert pipeline.strong_problems(cfg, plan, "strong_validate") == []
    ws, outputs = pipeline.Workspace(cfg), {}
    pipeline._strong_validate(ws, plan, T5.Meter(), outputs)
    r = outputs["strong_validation"]
    assert r["verdict"]["passed"] and r["candidate"] == candidates_at([200])[0]
    assert (calls, gen.calls) == ({"pd": 2, "adl": 6}, 3)  # chunks: PD per framing, ADL per
    # framing × length (1, 2, 32); each response part one
    runs_dir = T5.ws_runs(ws)
    names = sorted(p.name.split("-")[2] for p in runs_dir.glob("*-strong_*"))
    assert names == ["strong_adl", "strong_ood", "strong_pd", "strong_rates", "strong_rates",
                     "strong_validation", "strong_validation_identity"]  # fmt: skip
    pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})  # a resume: reused
    assert (calls, gen.calls) == ({"pd": 2, "adl": 6}, 3)


def test_a_second_identity_of_the_validation_is_refused(strong_decided, engine, monkeypatch):
    committed_screening_of(strong_decided, screened(strong_decided.space, [(200, 0)]))
    cfg, plan = plan_of("strong_validate")
    pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})
    real = CS.identity
    monkeypatch.setattr(CS, "identity", lambda i: {**real(i), "streams": {"observer": "x",
                                                                          "ood": "y"}})  # fmt: skip
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "has another identity" in problems
    with pytest.raises(ValueError, match="has another identity"):
        pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})


def crash_on(monkeypatch, target, attr, after):
    """Make ``target.attr`` fail on its call number ``after`` + 1 (a crash in flight)."""
    real, count = getattr(target, attr), {"n": 0}

    def wrapped(*args, **kwargs):
        count["n"] += 1
        if count["n"] == after + 1:
            raise RuntimeError("the Pod went away")
        return real(*args, **kwargs)

    monkeypatch.setattr(target, attr, wrapped)


def test_an_interrupted_part_recovers_its_saved_chunks_and_records_the_loss_in_flight(
    strong_decided, engine, monkeypatch
):  # fmt: skip
    """Review s5k 4: a part's saved chunks are recovered, never drawn again; the chunk in
    flight when the run died is drawn anew, its loss recorded."""
    from residual_disposition import tools

    gen, calls = engine
    committed_screening_of(strong_decided, screened(strong_decided.space, [(200, 0)]))
    cfg, plan = plan_of("strong_validate")
    crash_on(monkeypatch, tools, "perplexity_differencing", after=1)  # raw dies, chat saved
    with pytest.raises(RuntimeError, match="went away"):
        pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})
    assert calls["pd"] == 1  # chat drawn and saved; raw in flight
    pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})
    assert calls["pd"] == 2  # only raw again
    ws_runs = T5.ws_runs(pipeline.Workspace(cfg))
    done = [r for r in ws_runs.glob("*-strong_pd-*") if CS.completed(r)]
    (final,) = done
    chunks = json.loads((final / "chunks.json").read_text())
    failed = next(r for r in ws_runs.glob("*-strong_pd-*") if not CS.completed(r))
    assert chunks["chat"]["recovered_from"] == failed.name
    assert chunks["raw"]["lost_in"] == [failed.name] and chunks["raw"]["drawn_in"] == final.name


def test_a_part_whose_every_chunk_was_saved_is_never_drawn_again(strong_decided, engine,
                                                                 monkeypatch):  # fmt: skip
    """Review s5k 4: a crash after the outputs were saved, before the completion marker."""
    gen, calls = engine
    committed_screening_of(strong_decided, screened(strong_decided.space, [(200, 0)]))
    cfg, plan = plan_of("strong_validate")
    real = pipeline._write_jsonl_gz
    crash_on(monkeypatch, pipeline, "_write_jsonl_gz", after=2)  # PD's chunks saved, then dies
    with pytest.raises(RuntimeError, match="went away"):
        pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), {})
    assert calls["pd"] == 2  # both framings drawn and saved
    monkeypatch.setattr(pipeline, "_write_jsonl_gz", real)
    outputs = {}
    pipeline._strong_validate(pipeline.Workspace(cfg), plan, T5.Meter(), outputs)
    assert calls["pd"] == 2  # recovered, not drawn again
    assert outputs["strong_validation"]["verdict"]["passed"]


@pytest.mark.parametrize("override", ["pipeline.sampling.behavior.temperature=0.7",
                                      "+pipeline.vllm.seed=123"])  # fmt: skip
def test_the_validation_s_generation_contract_is_the_frozen_one(strong_decided, override):
    """Review s5k 2: another decoding or generator seed is refused before any GPU work."""
    committed_screening_of(strong_decided, screened(strong_decided.space, [(200, 0)]))
    cfg = config("strong_validate", f"+pipeline.plan.strong_validate.branch={CS.CONTINUED}",
                 override)  # fmt: skip
    plan = OmegaConf.to_container(cfg.pipeline.plan, resolve=True)
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "the generation contract is not the frozen one" in problems


def test_a_part_of_another_model_suite_or_decoding_is_refused(space, frozen):
    """Review s5k 2: every source cell's step, model and exact params are the identity's."""
    other = {**T5.CELL_MODEL, "revision": "0" * 40}
    with pytest.raises(ValueError, match="a cell of model"):
        CS.validation_record(validation_runs(space, frozen, model=other, tag="m"), find_run,
                             LINK.detector, paths.data_root())  # fmt: skip

    def v2(ident):
        return {**ident, "suites": {**ident["suites"], "pd": {"name": "V2", "sha256": "x"}}}

    with pytest.raises(ValueError, match="not this validation's pd cell"):
        CS.validation_record(validation_runs(space, frozen, ident_override=v2, tag="s"),
                             find_run, LINK.detector, paths.data_root())  # fmt: skip
    hot = {"sampling": {**pipeline.S5_EXPLORATORY["sampling"], "temperature": 0.7}}
    with pytest.raises(ValueError, match="the generation contract is not the frozen one"):
        CS.validation_record(validation_runs(space, frozen, contract_override=hot, tag="t"),
                             find_run, LINK.detector, paths.data_root())  # fmt: skip


def test_the_adl_screening_counts_only_its_exact_inventory_bound_to_its_prefixes(space, frozen):
    """Review s5k 3: each (framing, length, index) once, the frozen prefixes, a completed cell."""
    pd = space.expression(200)["pd_corpus"]

    def repeat_index_0(rows):  # index 0 throughout each group: the others never read
        for r in rows:
            r["index"], r["prefix"] = 0, [r["length"], 0]

    def other_prefix(rows):
        rows[5]["prefix"] = [9, 9]

    for mutate, status, match in ((repeat_index_0, "completed", "repeats"),
                                  (other_prefix, "completed", "not the frozen ADL suite's"),
                                  (None, "failed", "not a completed")):  # fmt: skip
        name = corpus_run(space, 200, mutate=mutate, status=status)
        rows = [{**candidates_at([200])[0], "pd": pd, "adl": name}]
        with pytest.raises(CS.Incomplete, match=match):
            CS.screening_record(screening_inputs(rows), find_run, LINK.detector)
        __import__("shutil").rmtree(find_run(name))


def test_invalid_evidence_stops_the_screening_before_the_next_candidate(strong_decided,
                                                                        monkeypatch):  # fmt: skip
    space = strong_decided.space
    pools = {c["latent"]: c for c in screened(space, [(200, 0), (400, 0)])}
    pd = find_run(pools["lat200"]["pd"]) / "pd_completions.jsonl.gz"
    with gzip.open(pd, "rt") as f:
        lines = f.readlines()
    with gzip.open(pd, "wt") as f:
        f.writelines(lines[:-1])  # one completion missing: the pool is invalid
    drawn = []
    monkeypatch.setattr(pipeline, "audit_pd", lambda ws, sid, **k: drawn.append(sid))
    monkeypatch.setattr(pipeline, "tool_pd_cell", lambda ws, sid, draw=None: type(
        "C", (), {"previous": find_run(pools[sid]["pd"])}))  # fmt: skip
    monkeypatch.setattr(
        pipeline,
        "adl_corpus_check",
        lambda ws, sid: find_run(pools[sid]["adl"]) / "adl_corpus.json",
    )
    cfg, plan = plan_of("strong_screen")
    with pytest.raises(CS.Incomplete, match="invalid"):
        pipeline._strong_screen(pipeline.Workspace(cfg), plan, T5.Meter(), {})
    assert drawn == ["lat200"]  # 400 is never screened on invalid evidence of 200
