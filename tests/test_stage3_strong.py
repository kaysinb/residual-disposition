"""Stage 3's strong branch of a panel behavior (preregistration §12.7 Stage 3, §12.4.10; C17,
part C), CPU only: the continued line to 1,600 steps on the strong item, reading the stack
lineage's data; the panel's branch per behavior, its candidates the continued selection's
margin-passing checkpoints, its screening and validation naming the behavior; the strong phases'
barriers (the committed stack decision before the screening, the diagnostic before the
validation); the diagnostic's sources from the continued screening's pools when the stack has
none (E4); the validated candidate's lineage and its X_cont phase."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import test_continuation_stage5 as T5
import test_continuation_strong as TS
import test_stage3 as T3  # imported before any fixture moves the repository root
from omegaconf import OmegaConf
from test_continuation_stage2_decision import CURRENT, committed_repo, git

from residual_disposition import budget, paths, pipeline
from residual_disposition import continuation_stage3 as S3
from residual_disposition import continuation_stage5 as S5
from residual_disposition import continuation_strong as CS
from residual_disposition.continuation_kstar import find_run

runs = T5.runs  # the fixture
PANEL = ("countdown", "word_count_claim", "eli5", "tldr", "myth_fact")
CONT = "stage3_cont/example_link"  # example_link standing in for a panel behavior
LINEAGE = "example_link_cont"  # its panel branch's lineage


def test_the_continued_line_is_its_own_lineage_on_the_strong_item():
    line = pipeline.S3_CONTINUED
    assert (line.key, line.phase, line.spec, line.built, line.builder) == (
        "stage3_cont", "s3s", "stage3", ("suppress",), False)  # fmt: skip
    assert (line.method, line.mode, line.roles, line.xcont) == (
        "sft", "continued", pipeline.S5_ROLES, False)  # fmt: skip
    assert line.recipe == {**pipeline.S5_RECIPE, "steps": 1600}
    assert line.checkpoints == (25, 50, 100, 200, 400, 800, 1600)
    for phase in pipeline.S3S_PHASES:
        assert phase in pipeline.PHASES and budget.BATCHES[phase] == "strong_stage3"
        assert pipeline.s5_line(phase) is line
    assert pipeline.s5_line("stage3_cont/countdown") is line
    assert pipeline.s5_line("s3_suppress") is pipeline.S3_STACK
    assert "s3s_xcont" in pipeline.ENGINE_FREE and "s3s_suppress" not in pipeline.ENGINE_FREE
    plan = {"phase": "strong_screen", "strong_screen": {"branch": CS.PANEL}}
    assert budget.batch_of(plan) == "strong_stage3"


def test_the_continued_line_reads_the_stack_lineage_s_data_and_never_builds_it(monkeypatch):
    calls = []

    def stage5_plan(ws, plan, line, **kw):
        calls.append((line.key, kw.get("build", False)))
        raise RuntimeError("planned")

    monkeypatch.setattr(pipeline, "stage5_plan", stage5_plan)
    for line in (pipeline.S3_CONTINUED, pipeline.S3_STACK):
        with pytest.raises(RuntimeError, match="planned"):
            pipeline._s5_suppress(None, {}, None, {}, line)
    assert calls == [("stage3_cont", False), ("stage3", True)]


def test_the_panel_branch_is_one_per_panel_behavior():
    for behavior in PANEL:
        b = CS.branch(CS.PANEL, behavior=behavior)
        assert (b["behavior"], b["item"], b["lineage"]) == (
            behavior, f"strong/stage3/{behavior}", f"{behavior}_cont")  # fmt: skip
        assert (b["suites"], b["alpha"], b["methods"]) == (
            {"pd": "V3", "adl": "V3", "observer": "O3"}, 0.05, 1)  # fmt: skip
        assert set(b["suite_hashes"]) == {"pd", "adl", "observer"}
        allocations = budget.load_allocations(
            paths.CONFIG_ROOT / "continuation" / "allocations.yaml"
        )
        assert b["item"] in {a.item for a in allocations}
    for other in ("example_link", "guillemets", None):
        with pytest.raises(ValueError, match="the panel branch is a panel behavior's"):
            CS.branch(CS.PANEL, behavior=other)
    assert CS.branch_of({"branch": CS.PANEL, "behavior": "tldr"})["lineage"] == "tldr_cont"
    assert CS.branch_of({"branch": CS.CONTINUED, "behavior": "tldr"})["behavior"] == (
        "example_link")  # fmt: skip


# -------------------------------------------------------------------- the continued lineage


def stage3_plan(line):
    """A Stage-3 plan of ``line`` (the model whole and the generation contract, review s5t 2)."""
    body = {k: v for k, v in T5.the_plan(line).items() if k != "sha256"}
    body.update(model=dict(T5.CELL_MODEL), generation=pipeline.S3_GENERATION)
    return {**body, "sha256": pipeline.content_hash(body)}


def continued_lineage(runs, monkeypatch, hits, plan=None):
    """example_link standing in for a panel behavior: its continued lineage's selection with
    ``hits`` (under ``plan``, by default generation 1's), a strong.yaml of tiny suites with the
    panel's branch on V3/O3, and the OOD requests."""
    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    pins = dict(T5.APPROVED["data_pins"])
    pins["suppress"] = {"id": pins["suppress"]["id"]}
    approved = {**T5.APPROVED, "model": dict(T5.CELL_MODEL), "data_pins": pins}
    monkeypatch.setattr(S5, "_approved", lambda behavior, line=None: approved)
    monkeypatch.setattr(CS, "_panel", lambda: ["example_link"])
    space = T5.Space(runs, hits=hits, plan=plan or stage3_plan(pipeline.S3_CONTINUED),
                     line=pipeline.S3_CONTINUED)  # fmt: skip
    sel = space.selection()
    suites = TS.tiny_suites()
    TS.save_suites(paths.data_root(), suites)
    data = TS.frozen_for(suites, TS.SEL, TS.candidates_at([200, 400, 800]))
    data["branches"][CS.PANEL] = {"suites": {"pd": "V3", "adl": "V3", "observer": "O3"},
                                  "alpha": 0.05, "methods": 1}  # fmt: skip
    monkeypatch.setattr(CS, "frozen", lambda path=None: data)
    return SimpleNamespace(space=space, sel=sel, frozen=data,
                           b=CS.branch(CS.PANEL, behavior="example_link"))  # fmt: skip


@pytest.fixture
def continued(runs, monkeypatch):
    """The continued lineage with candidates 100 … 800 (1,600 misses the margin)."""
    hits = {("latent", 25): 40, ("latent", 50): 20, ("latent", 1600): 40}
    return continued_lineage(runs, monkeypatch, hits)


def at(space, dose):
    return {"dose": dose, "latent": space.states["latent"][dose],
            "sham_s": space.states["sham_s"][dose]}  # fmt: skip


def screened(space, doses_hits):
    """Each candidate's full PD pool and ADL-corpus run (test_continuation_strong's)."""
    return [{**at(space, d), "pd": space.expression(d, pd_hits=h)["pd_corpus"],
             "adl": TS.corpus_run(space, d)} for d, h in doses_hits]  # fmt: skip


def screening_inputs(sel, rows, detector=None):
    detector = detector or T5.LINK.detector  # the panel's screening records its spec (s6a 3)
    return {"branch": CS.PANEL, "behavior": "example_link", "lineage": LINEAGE,
            "detector_version": detector.version, "detector": CS.detector_spec(detector),
            "selection": sel, "candidates": rows}  # fmt: skip


def test_the_panel_s_candidates_are_its_continued_selection_s(continued):
    """§12.7 Stage 3: the continued selection's margin-passing checkpoints, ascending, its
    bounds recomputed from its passes (1,600 misses the margin); another selection refused."""
    space = continued.space
    got = CS.candidates(continued.b, None, find_run, continued.sel)
    assert got == [at(space, d) for d in (100, 200, 400, 800)]
    for name in (None, "20261021-000300-stage3_selection-aaaa-0001", "nowhere"):
        with pytest.raises(ValueError, match=f"not a completed selection of {CONT}"):
            CS.candidates(continued.b, None, find_run, name)


def test_the_panel_screening_names_its_selection_and_screens_in_order(continued):
    space, sel = continued.space, continued.sel
    rows = screened(space, [(100, 1), (200, 0)])
    r = CS.screening_record(screening_inputs(sel, rows), find_run, T5.LINK.detector)
    assert (r["branch"], r["lineage"], r["behavior"]) == (CS.PANEL, LINEAGE, "example_link")
    assert r["chosen"] == at(space, 200) and [x["eligible"] for x in r["screened"]] == [
        False, True]  # fmt: skip
    with pytest.raises(ValueError, match="is not the frozen"):
        CS.screening_record(screening_inputs(sel, rows[1:]), find_run, T5.LINK.detector)
    with pytest.raises(ValueError, match="not a completed selection"):
        CS.screening_record(screening_inputs(None, rows), find_run, T5.LINK.detector)
    other = {**screening_inputs(sel, rows), "behavior": "tldr"}  # not in the stand-in panel
    with pytest.raises(ValueError, match="the panel branch is a panel behavior's"):
        CS.screening_record(other, find_run, T5.LINK.detector)


def test_the_panel_validation_s_identity_names_its_behavior(continued):
    contract = dict.fromkeys(CS.CONTRACT, {})
    bound = {"manifest": "stage3_panel_contracts.json", "sha256": "c" * 64}
    base = {"branch": CS.PANEL, "behavior": "example_link", "candidate": at(continued.space, 200),
            "screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": "s" * 64},
            "detector_version": CURRENT, "contract": contract, "contracts": bound}  # fmt: skip
    ident = CS.identity(base)
    assert (ident["branch"], ident["lineage"], ident["behavior"]) == (
        CS.PANEL, LINEAGE, "example_link")  # fmt: skip
    assert ident["suites"]["observer"]["name"] == "O3" and ident["contracts"] == bound
    assert ident["streams"]["observer"] == f"strong:{LINEAGE}:observer"
    with pytest.raises(ValueError, match="the panel branch is a panel behavior's"):
        CS.identity({k: v for k, v in base.items() if k != "behavior"})
    for other in (None, {"manifest": "other.json", "sha256": "c" * 64}):
        with pytest.raises(ValueError, match="names the panel's contracts"):
            CS.identity({**base, "contracts": other})


# ------------------------------------------------------------- the phases and their barriers


def panel_plan(phase, *overrides):
    cfg = T5.config(phase, f"+pipeline.plan.{phase}.branch={CS.PANEL}", *overrides)
    return cfg, OmegaConf.to_container(cfg.pipeline.plan, resolve=True)


def commit(repo, path, data, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "--allow-empty", "-m", message)


@pytest.fixture
def panel_repo(continued, monkeypatch):
    """A repository standing for the snapshot (``paths.REPO_ROOT``), no slot, the expression
    inputs present."""
    repo = committed_repo(continued.space.root)
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    return repo


def test_the_panel_s_screening_waits_for_its_committed_stack_decision(continued, panel_repo,
                                                                      monkeypatch):  # fmt: skip
    """Stage 3 holds no Stage-5 slot; the screening starts only once the stack decision
    (``stage3_<behavior>.json``, the fallback dose or none) is committed."""
    cfg, plan = panel_plan(
        "strong_screen", f"+pipeline.plan.strong_screen.selection={continued.sel}"
    )
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_screen"))
    assert "Stage-5 slot" not in problems
    assert "stage3_example_link.json" in problems
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    stack_decided(panel_repo)
    assert pipeline.strong_problems(cfg, plan, "strong_screen") == []
    b = pipeline.strong_branch_of(cfg, plan, "strong_screen")
    assert pipeline.strong_selection(b, plan) == continued.sel
    assert [c["dose"] for c in pipeline.strong_candidates(b, continued.sel)] == [100, 200, 400,
                                                                                 800]  # fmt: skip
    cfg, plan = panel_plan("strong_screen")  # no selection named
    assert "not a completed selection" in " ".join(
        pipeline.strong_problems(cfg, plan, "strong_screen"))  # fmt: skip


def write_screening(repo, sel, rows):
    i = screening_inputs(sel, rows)
    record = CS.screening_record(i, find_run, T5.LINK.detector, repo)
    CS.write_screening(repo, {**record, "inputs": i}, find_run, T5.LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "screening")
    return CS.committed_screening(repo, LINEAGE, find_run, T5.LINK.detector)


def test_the_panel_s_validation_waits_for_the_detector_diagnostic(continued, panel_repo,
                                                                   monkeypatch):  # fmt: skip
    """E4: without a stack dose the diagnostic's barrier moves before the validation."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    cfg, plan = panel_plan("strong_validate")
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "the detector diagnostic" in problems and "Stage-5 slot" not in problems
    assert "the ADL-corpus diagnostic" in problems  # s6a 2: before V3 too
    assert "stage3_panel_contracts.json is not committed" in problems  # s6a 4
    kinds = []
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: kinds.append(a[5]) or [])
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "diagnostic" not in problems and kinds == [("observer", "pd", "adl_corpus")]


# ------------------------------------------------- E4: the continued screening's pools


@pytest.fixture
def no_stack_dose(continued, panel_repo):
    """The stack lineage's selection with no margin-passing checkpoint, beside the continued one
    (their states and runs distinct), and its s3a sources cell's record (no source dose)."""
    hits = {("latent", d): 40 for d in T5.CHECKPOINTS}
    plan = stage3_plan(pipeline.S3_STACK)
    stack = T5.Space(continued.space.root, hits=hits, plan=plan, line=pipeline.S3_STACK)
    record = {"behavior": "example_link", "lineage": "stage3/example_link",
              "plan_sha256": plan["sha256"], "selection": stack.selection(), "doses": [],
              "pd": {}, "observer": {}, "from": "continued_screening"}  # fmt: skip
    return SimpleNamespace(stack=stack, record=record)


def test_without_a_stack_dose_the_sources_are_the_continued_screening_s_pools(
    continued, panel_repo, no_stack_dose
):
    record, space = no_stack_dose.record, continued.space
    with pytest.raises((ValueError, RuntimeError), match="not committed"):
        S3.sources_record(record, T5.RULES, find_run, panel_repo)  # no screening yet: no sources
    rows = screened(space, [(100, 1), (200, 0)])
    sha = write_screening(panel_repo, continued.sel, rows)["sha256"]
    with pytest.raises(ValueError, match="not the committed continued screening's pools"):
        S3.sources_record(record, T5.RULES, find_run, panel_repo)  # never frozen empty
    more = S3.continued_sources("example_link", T5.RULES, find_run, panel_repo)
    selection = json.loads((find_run(continued.sel) / "stage3_cont_selection.json").read_text())
    passes = {str(d): selection["doses"][str(d)]["excess"]["passes"][at(space, d)["latent"]]
              for d in (100, 200)}  # fmt: skip
    assert more == {"doses": [100, 200], "from": "continued_screening",
                    "pd": {str(r["dose"]): r["pd"] for r in rows}, "observer": passes,
                    "screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": sha},
                    "continued_selection": continued.sel}  # fmt: skip
    data = {**record, **more}
    assert S3.sources_record(data, T5.RULES, find_run, panel_repo) == data
    assert S3.sources_problems(data, T5.RULES, find_run, panel_repo) == []
    for bad in ({"doses": [100]}, {"pd": {"100": rows[0]["pd"], "200": rows[0]["pd"]}},
                {"observer": {**passes, "200": passes["100"]}},
                {"screening": {**more["screening"], "sha256": "0" * 64}}):  # fmt: skip
        assert S3.sources_problems({**data, **bad}, T5.RULES, find_run, panel_repo)
    S3.write_sources(panel_repo, {**data, "sources_run": "cell"}, T5.RULES, find_run)
    written = json.loads(S3.sources_path(panel_repo, "example_link").read_text())
    assert written["doses"] == [100, 200] and written["from"] == "continued_screening"


# ------------------------------------------------------ the validated candidate's lineage


def stack_as_committed(path, rules, find_run, repo=None):
    """The committed stack decision as it is (its recomputation is Stage 5's, tested there)."""
    from residual_disposition.manifests import committed_json

    return committed_json(path, "the stack lineage's decision", repo or paths.REPO_ROOT)


def stack_decided(repo, dose=None):
    """The behavior's committed stack decision (only its analysis dose is read here)."""
    commit(
        repo,
        S5.manifest_path(repo, "example_link", pipeline.S3_STACK),
        {"decision": {"status": "decided", "analysis_dose": dose}, "inputs": {}},
        "stack",
    )


FULL = dict.fromkeys(CS.CONTRACT, {})  # a generation contract's fields (their values tested apart)


def diagnosed(repo):
    """The behavior's two diagnostic manifests committed (their content tested on its own)."""
    for adl, kind in ((False, "observer"), (True, "adl_corpus")):
        path = S3.diagnostic_path(repo, "example_link", adl)
        if not path.exists():
            commit(repo, path, {"detector_version": CURRENT, "types": {kind: {}}}, "diagnostic")


def write_contracts(repo, screening, monkeypatch, contract_of=None):
    """The panel's contracts (the stand-in panel of one behavior), its diagnostic taken as
    complete (tested on its own below), committed; returns their binding."""
    if not S5.manifest_path(repo, "example_link", pipeline.S3_STACK).exists():
        stack_decided(repo)
    diagnosed(repo)
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    contract_of = contract_of or pipeline.panel_contract
    named = {"example_link": {"screening": {"manifest": f"strong_{LINEAGE}.json",
                                            "sha256": screening["sha256"]}}}  # fmt: skip
    data = CS.panel_contracts_record({"behaviors": named}, repo, T5.RULES, find_run, contract_of)
    CS.write_panel_contracts(repo, data, T5.RULES, find_run, contract_of)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "contracts")
    found = CS.committed_contract(repo, "example_link", T5.RULES, find_run, contract_of)
    return {"manifest": found["manifest"], "sha256": found["sha256"]}, data


def validation_runs(space, *, dose, screening, contracts):
    """A clean validation of the panel's candidate at ``dose`` (test_continuation_strong's
    parts, on V3/O3), bound to the panel's contracts."""
    c = at(space, dose)
    suites = {k: CS.load_suite(paths.data_root(), k, v) for k, v in
              {"pd": "V3", "adl": "V3", "observer": "O3"}.items()}  # fmt: skip
    contract = pipeline.strong_contract(pipeline.Workspace(T5.config("strong_validate")), suites)
    base = {"branch": CS.PANEL, "behavior": "example_link", "candidate": c,
            "detector_version": CURRENT, "contract": contract, "contracts": contracts,
            "screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": screening}}  # fmt: skip
    ident = CS.identity(base)
    ir = T5.a_run(space.root, "20261021-001000-strong_validation_identity-p-1",
                  "strong_validation_identity", {**ident}, {})  # fmt: skip
    (ir / "strong_validation_identity.json").write_text(json.dumps(ident))
    names = {}

    def part(key, step, name, rows):
        params = CS.part_params(ident, key)[1]
        run = T5.a_run(space.root, f"20261021-0011{len(names)}0-{step}-p-{len(names)}", step,
                       params, {})  # fmt: skip
        TS.write_rows(run, name, rows)
        names[key] = run.name

    part("pd", "strong_pd", "pd_completions.jsonl.gz",
         [{"config": f, "prefill": k, "completion": "plain"} for f in ("chat", "raw")
          for k in suites["pd"].records])  # fmt: skip
    part("adl", "strong_adl", "continuations.jsonl.gz",
         [{"framing": f, "key": k, "length": len(r["tokens"]), "text": "plain"}
          for f in ("chat", "raw") for k, r in suites["adl"].records.items()])  # fmt: skip
    for role in ("L", "H_s"):
        part(f"observer_{role}", "strong_rates", "responses.jsonl.gz",
             [{"prompt": k, "text": "plain"} for k in suites["observer"].records])  # fmt: skip
    part("ood", "strong_ood", "responses.jsonl.gz",
         [{"prompt": q.key, "text": "plain"} for q in pipeline.ood_requests()])  # fmt: skip
    return {**base, "identity": ir.name, "runs": names}


def test_the_panel_s_validated_candidate_is_its_strong_lineage(continued, panel_repo,
                                                               monkeypatch):  # fmt: skip
    """The lineage from the committed validation and screening, its states and plan the
    continued selection's (it has no decision of its own); X_cont its own phase, s3s_xcont, on
    the strong item; a strong lineage of another behavior never blocks its certificate."""
    space, repo = continued.space, panel_repo
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    rows = screened(space, [(100, 1), (200, 0)])
    screening = write_screening(repo, continued.sel, rows)
    contracts, data = write_contracts(repo, screening, monkeypatch)
    assert data["behaviors"]["example_link"]["status"] == CS.ACTIVE
    v = validation_runs(space, dose=200, screening=screening["sha256"], contracts=contracts)
    record = CS.validation_record(v, find_run, T5.LINK.detector, paths.data_root())
    assert record["verdict"]["passed"] and record["alpha"] == 0.05 and record["behavior"] == (
        "example_link")  # fmt: skip
    stale = {**v, "contracts": {**contracts, "sha256": "0" * 64}}
    assert "other contracts than the committed" in " ".join(
        CS.contract_problems(repo, stale, T5.LINK.detector, find_run))  # fmt: skip
    CS.write_validation(repo, {**record, "inputs": v}, find_run, T5.LINK.detector)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "validation")
    cfg = T5.config("kstar")
    s = pipeline.strong_lineage_of(cfg, "kstar", CS.PANEL)
    c = at(space, 200)
    assert (s["lineage"], s["dose"], s["latent"], s["sham_s"], s["X"], s["branch"]) == (
        "strong_example_link_cont_200", 200, c["latent"], c["sham_s"], None, CS.PANEL)  # fmt: skip
    assert (s["mode"], s["method"], s["plan"]["lineage"]) == ("continued", "sft", CONT)
    assert s["selection"] == continued.sel and s["pd_corpus"] == rows[1]["pd"]
    assert s["binding"]["manifest"] == f"strong_validation_{LINEAGE}.json"
    with pytest.raises(ValueError, match="at 200 has no valid X or recipe control yet"):
        pipeline.strong_lineage_of(cfg, "readers", CS.PANEL)
    (repo / "manifests" / "kstar_strong_tldr_cont_400.json").write_text("{}")  # another's
    pipeline.strong_lineage_of(cfg, "certificate", CS.PANEL)
    (repo / "manifests" / "kstar_strong_example_link_cont_100.json").write_text("{}")
    with pytest.raises(ValueError, match="another strong lineage of example_link has its k"):
        pipeline.strong_lineage_of(cfg, "certificate", CS.PANEL)
    ws = pipeline.Workspace(T5.config("s3s_xcont"))
    recorded = contracts["sha256"]  # the meter's record, the validation's contracts (s6n 1)
    for wrong in (None, "0" * 64):  # no record, or a record of other contracts
        with pytest.raises(ValueError, match="is not the contracts its validation names"):
            pipeline.strong_xcont_plan(ws, {"s3s_xcont": {"dose": 200, "contracts": wrong}},
                                       "s3s_xcont", CS.PANEL)  # fmt: skip
    with pytest.raises(ValueError, match="plan.s3s_xcont.dose None is not the strong candidate"):
        pipeline.strong_xcont_plan(ws, {"s3s_xcont": {"contracts": recorded}}, "s3s_xcont",
                                   CS.PANEL)  # fmt: skip
    x = pipeline.strong_xcont_plan(ws, {"s3s_xcont": {"dose": 200, "contracts": recorded}},
                                   "s3s_xcont", CS.PANEL)  # fmt: skip
    assert (x["dose"], x["latent"], x["plan"]["lineage"]) == (200, c["latent"], CONT)
    problems = pipeline.stage3_problems(
        T5.config("s3s_xcont"), {"s3s_xcont": {"dose": 200}}, "s3s_xcont"
    )  # the preflight refuses it too
    assert "is not the contracts its validation names" in " ".join(problems)


# ------------------------------------------------------- review s6a: closures, contracts, ADL


def test_an_empty_candidate_list_closes_the_screening(runs, monkeypatch):
    """S6a 1: no continued checkpoint meets the margin: the recomputed list is empty and the
    screening, with no candidate, records the branch's closure; naming one is refused."""
    lineage = continued_lineage(runs, monkeypatch, {("latent", d): 40 for d in
                                                    pipeline.S3_CONTINUED.checkpoints})  # fmt: skip
    assert CS.candidates(lineage.b, None, find_run, lineage.sel) == []
    r = CS.screening_record(screening_inputs(lineage.sel, []), find_run, None)
    assert (r["screened"], r["chosen"], r["outcome"]) == ([], None, CS.NO_DOSE)
    rows = screened(lineage.space, [(100, 0)])
    with pytest.raises(ValueError, match="where the recomputed list has none"):
        CS.screening_record(screening_inputs(lineage.sel, rows), find_run, None)


def revised(extra):
    """A revision of example_link's detector, as a committed config would give it."""
    import dataclasses

    old = T5.LINK.detector
    return dataclasses.replace(T5.LINK, detector=dataclasses.replace(
        old, pattern=f"{old.pattern}|{extra}"))  # fmt: skip


def test_the_screening_is_replayed_under_its_recorded_detector(continued, panel_repo,
                                                                monkeypatch):  # fmt: skip
    """S6a 3: a revision after the screening leaves it verifiable under the detector it
    recorded; a recorded spec that is not its version is refused."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    later = revised("zzz-new")
    monkeypatch.setattr(S5, "_behavior", lambda behavior: later)
    again = CS.committed_screening(panel_repo, LINEAGE, find_run, later.detector)
    assert again["sha256"] == screening["sha256"] and again["detector_version"] == CURRENT
    inputs = {**screening["inputs"], "detector": CS.detector_spec(later.detector)}
    with pytest.raises(ValueError, match="recorded detector spec is"):
        CS.screening_record(inputs, find_run, None)


def test_the_contracts_rescore_the_candidate_under_a_revised_detector(continued, panel_repo,
                                                                       monkeypatch):  # fmt: skip
    """S6c 3: the contracts' detector is the committed config's; where it is not the
    screening's, the candidate's saved margin and PD and ADL zeros are rescored under it, never
    redrawn: a candidate still clean stays active; one that now fails closes the branch."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    monkeypatch.setattr(S3, "diagnostic_bindings", lambda root, behavior: [])
    monkeypatch.setattr(S3, "transition_problems", lambda *a, **k: [])  # tested on its own below
    stack_decided(panel_repo)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    named = {"screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": screening["sha256"]}}

    def entry(behavior):
        monkeypatch.setattr(S5, "_behavior", lambda b: behavior)
        return CS.panel_entry("example_link", named, panel_repo, T5.RULES, find_run,
                              lambda b: {"contract": 1})  # fmt: skip

    e = entry(T5.LINK)  # the margin and the zeros rechecked even under the same detector (s6g 3)
    assert (e["status"], e["chosen"], e["reading"]) == (CS.ACTIVE, at(continued.space, 100),
                                                       "strong")  # fmt: skip
    assert (e["rescored"]["margin"], e["rescored"]["zeros"]) == (True, True)
    e = entry(revised("zzz-new"))  # the same verdicts on the saved evidence
    assert e["status"] == CS.ACTIVE and (e["rescored"]["margin"], e["rescored"]["zeros"]) == (
        True, True)  # fmt: skip
    e = entry(revised("plain"))  # every saved continuation is now a hit
    assert e["status"] == CS.CLOSED and e["diagnostic"] == CS.NOT_APPLICABLE  # no stack dose
    assert e["rescored"]["zeros"] is False and e["chosen"] == at(continued.space, 100)
    with pytest.raises(ValueError, match="not its committed screening"):
        CS.panel_entry("example_link", {"screening": {**named["screening"], "sha256": "0" * 64}},
                       panel_repo, T5.RULES, find_run, lambda b: {})  # fmt: skip


def test_the_contracts_are_the_whole_panel_s_and_refuse_a_later_drift(continued, panel_repo,
                                                                       monkeypatch):  # fmt: skip
    """S6a 4, s6c 3: the contracts name every panel behavior, are written once and read only
    committed; a behavior's validation recomputes its own entry, so a later change of its
    detector or generation contract invalidates it."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    contract = FULL
    bound, data = write_contracts(panel_repo, screening, monkeypatch, lambda b: contract)
    with pytest.raises(ValueError, match="written once"):
        CS.write_panel_contracts(panel_repo, data, T5.RULES, find_run, lambda b: contract)
    with pytest.raises(ValueError, match="not the frozen panel"):
        CS.panel_contracts_record({"behaviors": {}}, panel_repo, T5.RULES, find_run,
                                  lambda b: contract)  # fmt: skip
    found = CS.committed_contract(panel_repo, "example_link", T5.RULES, find_run,
                                  lambda b: contract)  # fmt: skip
    assert found["sha256"] == bound["sha256"] and found["entry"]["status"] == CS.ACTIVE
    with pytest.raises(ValueError, match="its validation is invalidated"):
        CS.committed_contract(panel_repo, "example_link", T5.RULES, find_run,
                              lambda b: {**FULL, "engine": {"other": 1}})  # fmt: skip
    monkeypatch.setattr(S5, "_behavior", lambda b: revised("zzz-new"))  # and no revision record
    with pytest.raises(ValueError, match="its validation is invalidated|no accepted revision"):
        CS.committed_contract(panel_repo, "example_link", T5.RULES, find_run,
                              lambda b: contract)  # fmt: skip


def test_the_adl_corpus_diagnostic_is_its_own_set_from_frozen_sources(continued, panel_repo,
                                                                       no_stack_dose, tmp_path,
                                                                       monkeypatch):  # fmt: skip
    """S6a 2, s6c 2: the ADL-corpus diagnostic's sources are the union of the stack decision's
    expression run and the committed screening's runs, each once, frozen before its labeling;
    its set of its own is drawn from them (150 detector negatives, the declared seed), each item
    its prefix in brackets as context and its continuation, the check on the continuation; its
    result, written by the script, is the barrier's before the validation."""
    import argparse
    import hashlib
    import importlib.util

    import transformers

    repo, space, stack = panel_repo, continued.space, no_stack_dose.stack
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    screening = write_screening(repo, continued.sel, screened(space, [(100, 1), (200, 0)]))
    stack_adl = TS.corpus_run(stack, 200)  # the stack lineage's expression check at its dose
    decision = {"behavior": "example_link", "decision": {"status": "decided",
                "analysis_dose": 200}, "inputs": {"behavior": "example_link",
                "lineage": "stage3/example_link", "selection": no_stack_dose.record["selection"],
                "expression": {"adl_corpus": stack_adl}}}  # fmt: skip
    with pytest.raises((ValueError, RuntimeError), match="not committed"):
        S3.adl_sources_record("example_link", T5.RULES, find_run, repo)
    commit(repo, S5.manifest_path(repo, "example_link", pipeline.S3_STACK), decision, "stack")
    record = S3.adl_sources_record("example_link", T5.RULES, find_run, repo)
    rows = screening["inputs"]["candidates"]
    want = [(stack_adl, stack.states["latent"][200], 200, "stack")] + [
        (r["adl"], r["latent"], r["dose"], "continued") for r in rows]  # fmt: skip
    assert [(r["run"], r["latent"], r["dose"], r["from"]) for r in record["runs"]] == want
    assert record["screening"]["sha256"] == screening["sha256"]
    S3.write_adl_sources(repo, record, T5.RULES, find_run)
    with pytest.raises(ValueError, match="frozen once"):
        S3.write_adl_sources(repo, record, T5.RULES, find_run)
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "adl sources")
    sources = S3.committed_adl_sources(repo, "example_link", T5.RULES, find_run)

    class Tokenizer:  # the cells' prefixes decoded (test_continuation_labels's stand-in)
        def __len__(self):
            return 10**6

        def decode(self, ids):
            return "".join(f"<{i}>" for i in ids)

    tokenizer = Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    cl = T3.labels_script()
    seed = int(T5.RULES["seeds"]["labels"])
    with monkeypatch.context() as m:  # the builder reads the committed sources by name
        m.setattr(S3, "committed_adl_sources", lambda root, behavior, rules, find_run: sources)
        cl.detector_diagnostic(argparse.Namespace(
            runs="runs/feasibility_14b", kinds=["adl_corpus"], corpus_runs="runs/continuation_14b",
            state_index=None, dry_run=False, behavior="example_link"))  # fmt: skip
        run = next(paths.runs_root().glob("*/*-diagnostic-s3adl-example_link-*"))
        key = json.loads((run / "key.json").read_text())
        assert key["kinds"] == ["adl_corpus"] and key["sources"]["sha256"] == sources["sha256"]
        rows = [json.loads(x) for x in (run / "input.jsonl").read_text().splitlines()]
        assert len(rows) == 150 and all(r["response"].startswith("[<") for r in rows)
        assert {S3._source_run("adl_corpus", k["source"]) for k in key["items"].values()} <= {
            r["run"] for r in record["runs"]}  # fmt: skip
        assert "the ADL-corpus diagnostic" in " ".join(S3.diagnostic_problems(
            repo, "example_link", T5.RULES, find_run, T5.LINK.detector, [S3.ADL]))  # fmt: skip
        negatives = {"adl_corpus": list(key["items"])}
        files = {}
        for name, lines in (
            ("ensemble", [{"item_id": i, "present": False, "unanimous": True, "confident": True,
                           "evidence": ""} for i in key["items"]]),
            ("review", [{"item_id": i, "present": False, "note": ""}
                        for i in sorted(S3.review_sample("example_link", seed, negatives))]),
        ):  # fmt: skip
            files[name] = tmp_path / f"{name}.jsonl"
            files[name].write_text("".join(json.dumps(x) + "\n" for x in lines))
        m.setattr(paths, "REPO_ROOT", T3.REPO_ROOT)  # its scripts; it writes under --root
        path = T3.REPO_ROOT / "scripts" / "continuation_stage3.py"
        spec = importlib.util.spec_from_file_location("continuation_stage3_adl", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        argv = ["diagnostic", str(run), "--ensemble", str(files["ensemble"]), "--review",
                str(files["review"]), "--root", str(repo)]  # fmt: skip
        assert script.main(argv) == 0
    data = json.loads(S3.diagnostic_path(repo, "example_link", adl=True).read_text())
    assert data["types"] == {"adl_corpus": {"status": "complete", "n": 150, "traces": 0}}
    assert data["review"]["sample"] == 30 and data["sources_sha256"] == sources["sha256"]
    assert hashlib.sha256((run / "input.jsonl").read_bytes()).hexdigest() == data["input_sha256"]
    git(repo, "add", "manifests")
    git(repo, "commit", "-q", "-m", "adl diagnostic")
    assert S3.diagnostic_problems(repo, "example_link", T5.RULES, find_run, T5.LINK.detector,
                                  [S3.ADL]) == []  # fmt: skip
    assert [b["manifest"] for b in S3.diagnostic_bindings(repo, "example_link")] == [
        "stage3_diagnostic_adl_example_link.json"]  # fmt: skip


def test_a_closed_branch_keeps_its_stack_reading_s_diagnostic(continued, panel_repo,
                                                               monkeypatch):  # fmt: skip
    """S6g 1: a closed strong branch whose stack decision kept a dose still reads it: its
    diagnostic is required, never waived; not applicable only with no reading."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    monkeypatch.setattr(S3, "transition_problems", lambda *a, **k: [])
    stack_decided(panel_repo, dose=200)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    named = {"screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": screening["sha256"]}}
    monkeypatch.setattr(S5, "_behavior", lambda b: revised("plain"))  # the branch closes
    with pytest.raises(ValueError, match="not complete for its stack reading"):
        CS.panel_entry("example_link", named, panel_repo, T5.RULES, find_run, lambda b: {})
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    monkeypatch.setattr(S3, "diagnostic_bindings", lambda root, behavior: [{"manifest": "d"}])
    e = CS.panel_entry("example_link", named, panel_repo, T5.RULES, find_run, lambda b: {})
    assert (e["status"], e["reading"], e["stack"]["dose"]) == (CS.CLOSED, "stack", 200)
    assert e["diagnostic"] == [{"manifest": "d"}]


def test_every_detector_transition_needs_its_accepted_revision(continued, panel_repo,
                                                               monkeypatch):  # fmt: skip
    """S6g 2: a detector other than the one the screening, the selection or a diagnostic set
    was made with enters the contracts only through committed accepted revisions, traces or
    none."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    monkeypatch.setattr(S3, "diagnostic_bindings", lambda root, behavior: [])
    stack_decided(panel_repo)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    named = {"screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": screening["sha256"]}}
    later = revised("zzz-new")
    monkeypatch.setattr(S5, "_behavior", lambda b: later)
    with pytest.raises(ValueError, match=f"no accepted revision of example_link's detector to "
                                         f"{later.detector.version} from"):  # fmt: skip
        CS.panel_entry("example_link", named, panel_repo, T5.RULES, find_run, lambda b: {})
    assert (
        S3.transition_problems(
            panel_repo, "example_link", {later.detector.version}, later.detector, find_run
        )
        == []
    )  # its own: nothing to cross
    path = S3.revision_path(panel_repo, "example_link", later.detector.version)
    commit(panel_repo, path, {"behavior": "example_link", "from_version": "0" * 12}, "a record")
    problems = " ".join(S3.transition_problems(panel_repo, "example_link", {CURRENT},
                                               later.detector, find_run))  # fmt: skip
    assert "not an acceptance of §12.3.1's checks" in problems and "from ['" in problems


def test_the_contracts_are_read_whole_and_by_membership(continued, panel_repo, monkeypatch):
    """S6g 4, 5: the contracts are the whole panel's (its five entries, the frozen suites),
    compared by membership in the panel's order after a sorted-keys round trip; and every panel
    validation opened names one contracts' hash."""
    panel = ["tldr", "example_link", "pun_intended"]  # not in the sorted keys' order
    monkeypatch.setattr(CS, "_panel", lambda: panel)
    entry = {"detector": {}, "stack": {}, "reading": None, "diagnostic": CS.NOT_APPLICABLE,
             "status": CS.CLOSED, "infeasibility": "r"}  # fmt: skip
    monkeypatch.setattr(CS, "panel_entry", lambda b, named, *a: {**entry, **named})
    monkeypatch.setattr(CS, "_entry_problems", lambda *a: [])  # each entry's schema: tested apart
    named = {b: {"infeasibility": f"inf-{b}"} for b in reversed(panel)}  # another order
    data = CS.panel_contracts_record({"behaviors": named}, panel_repo, T5.RULES, find_run,
                                     lambda b: {})  # fmt: skip
    CS.write_panel_contracts(panel_repo, data, T5.RULES, find_run, lambda b: {})
    git(panel_repo, "add", "manifests")
    git(panel_repo, "commit", "-q", "-m", "contracts")
    back = json.loads(CS.contracts_path(panel_repo).read_text())  # sorted keys
    assert list(back["behaviors"]) != panel
    assert CS.panel_contracts_problems(back, panel_repo, T5.RULES, find_run, lambda b: {}) == []
    found = CS.committed_contract(panel_repo, "tldr", T5.RULES, find_run, lambda b: {})
    for bad in ({"behaviors": {b: named[b] for b in panel[:2]}}, {"suites": {}},
                {"extra": 1}):  # fmt: skip
        commit(panel_repo, CS.contracts_path(panel_repo), {**back, **bad}, "a partial bundle")
        with pytest.raises(ValueError, match="not the frozen panel's whole contracts"):
            CS.committed_contract(panel_repo, "tldr", T5.RULES, find_run, lambda b: {})
    commit(panel_repo, CS.contracts_path(panel_repo), back, "the bundle again")
    found = CS.committed_contract(panel_repo, "tldr", T5.RULES, find_run, lambda b: {})
    other = CS.validation_path(panel_repo, "pun_intended_cont")
    commit(panel_repo, other, {"inputs": {"contracts": {"sha256": "0" * 64}}}, "a validation")
    assert CS.panel_openings(panel_repo) == {other.name: "0" * 64}
    inputs = {"behavior": "tldr", "contracts": {"manifest": found["manifest"],
                                                "sha256": found["sha256"]}}  # fmt: skip
    assert "opened under other contracts" in " ".join(
        CS.contract_problems(panel_repo, inputs, T5.LINK.detector, find_run, lambda b: {})
    )


def test_the_stack_reading_is_its_valid_decision_s_now(continued, panel_repo, monkeypatch):
    """S6h 1: the contracts freeze the reading dose of the stack decision as it is valid now; one
    its validator refuses (another detector's, a retest owed) stops the entry."""
    stack_decided(panel_repo, dose=100)

    def stale(path, rules, find_run, repo=None):
        raise ValueError("the decision was made with detector old1, not the current one")

    monkeypatch.setattr(S5, "committed_decision", stale)
    with pytest.raises(ValueError, match="stack decision is not valid now"):
        S3.current_stack_reading(panel_repo, "example_link", T5.RULES, find_run)
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    r = S3.current_stack_reading(panel_repo, "example_link", T5.RULES, find_run)
    assert r["dose"] == 100 and r["manifest"] == "stage3_example_link.json"


def test_the_final_detector_keeps_every_prior_label(continued, panel_repo, monkeypatch):
    """S6h 2: past the chain of revisions, the final detector agrees with the final labels of
    every committed diagnostic set, whichever detector drew it (a trace an intermediate revision
    caught, lost by the last, refuses)."""
    later = revised("zzz-new")
    monkeypatch.setattr(S5, "_behavior", lambda b: later)
    monkeypatch.setattr(S3, "_record_problems", lambda *a, **k: [])
    path = S3.revision_path(panel_repo, "example_link", later.detector.version)
    commit(panel_repo, path, {"from_version": CURRENT}, "a record")
    monkeypatch.setattr(S3, "_committed_diagnostics", lambda root, behavior: [{"set_id": "a"}])
    monkeypatch.setattr(S3, "diagnostic_disagreements", lambda m, detector, find_run: ["i7"])
    problems = " ".join(S3.transition_problems(panel_repo, "example_link", {CURRENT},
                                               later.detector, find_run))  # fmt: skip
    assert "disagrees with the final labels of example_link's diagnostic sets on ['i7']" in problems
    monkeypatch.setattr(S3, "diagnostic_disagreements", lambda m, detector, find_run: [])
    assert S3.transition_problems(panel_repo, "example_link", {CURRENT}, later.detector,
                                  find_run) == []  # fmt: skip


def test_a_stored_entry_is_complete_for_its_status_and_bound(continued, panel_repo, monkeypatch):
    """S6h 3: another behavior's stored entry is checked whole (its status's fields, its
    detector's spec, its committed evidence), never taken as written."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    stack_decided(panel_repo)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    bound, data = write_contracts(panel_repo, screening, monkeypatch, lambda b: FULL)
    entry = data["behaviors"]["example_link"]
    assert CS._entry_problems(panel_repo, "example_link", entry, T5.RULES, find_run) == []
    placeholder = {"detector": None, "stack": None, "diagnostic": None, "reading": None,
                   "status": CS.CLOSED, "reason": "x", "infeasibility": "nowhere"}  # fmt: skip
    problems = " ".join(CS._entry_problems(panel_repo, "example_link", placeholder, T5.RULES,
                                           find_run))  # fmt: skip
    wants = (
        "detector is not a spec",
        "stack decision is not the committed",
        "infeasibility is not its continued line's recorded one",
        "with no reading names a diagnostic",
    )
    for want in wants:
        assert want in problems
    for bad in ({"stack": {**entry["stack"], "sha256": "0" * 64}},
                {"screening": {**entry["screening"], "sha256": "0" * 64}},
                {"reading": "stack"}, {"contract": {}}, {"extra": 1}):  # fmt: skip
        assert CS._entry_problems(panel_repo, "example_link", {**entry, **bad}, T5.RULES, find_run)


def test_a_stored_entry_s_claims_are_its_bound_evidence_s(continued, panel_repo, monkeypatch):
    """S6i 3: a stored entry's claims are checked against the historical evidence it binds: its
    stack dose against the bound decision's (a null dose against a pinned 100 would waive the
    diagnostic), its candidate, recheck, status and reason against the bound screening's, and
    its infeasibility against the continued line's recorded one (an ordinary run is none)."""
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    stack_decided(panel_repo, dose=100)
    screening = write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    _, data = write_contracts(panel_repo, screening, monkeypatch, lambda b: FULL)
    entry = data["behaviors"]["example_link"]
    assert entry["stack"]["dose"] == 100 and entry["status"] == CS.ACTIVE

    closed = {**{k: v for k, v in entry.items() if k != "contract"}, "status": CS.CLOSED,
              "reason": CS.RECHECK_REASON, "reading": "stack"}  # fmt: skip

    def problems(base=entry, **changed):
        return " ".join(CS._entry_problems(panel_repo, "example_link", {**base, **changed},
                                           T5.RULES, find_run))  # fmt: skip

    assert problems() == ""
    undosed = {**entry["stack"], "dose": None}
    assert "stack dose None is not its bound decision's 100" in problems(stack=undosed)
    assert "stack dose None is not its bound decision's 100" in problems(  # the waiver refused
        closed, stack=undosed, reading=None, diagnostic=CS.NOT_APPLICABLE)  # fmt: skip
    assert "candidate is not its bound screening's" in problems(
        chosen={**entry["chosen"], "dose": 200})  # fmt: skip
    assert "recheck is not of its bound screening's candidate" in problems(
        rescored={**entry["rescored"], "from_version": "other"})  # fmt: skip
    assert "not what its bound screening and recheck give: active" in problems(closed)
    failed = {**entry["rescored"], "zeros": False}
    assert problems(closed, rescored=failed) == ""
    assert "not what its bound screening and recheck give" in problems(closed, rescored=failed,
                                                                       reason="x")  # fmt: skip
    ordinary = {k: v for k, v in closed.items() if k not in ("screening", "chosen", "rescored")}
    ordinary.update(reason=CS.INFEASIBLE_REASON, infeasibility=continued.sel)  # an ordinary run
    found = " ".join(CS._entry_problems(panel_repo, "example_link", ordinary, T5.RULES, find_run))
    assert "infeasibility is not its continued line's recorded one" in found
    monkeypatch.setattr(S5, "_infeasibility", lambda inputs, rules, find_run: ({}, []))
    assert CS._entry_problems(panel_repo, "example_link", ordinary, T5.RULES, find_run) == []
    assert "closure is not its infeasibility's" in " ".join(CS._entry_problems(
        panel_repo, "example_link", {**ordinary, "reason": "x"}, T5.RULES, find_run))  # fmt: skip


# ------------------------------------------- a repair generation (§12.7 Stage 3, the repair)


def test_the_contracts_bind_the_current_repair_generation_s_diagnostic(runs, monkeypatch):
    """Under generation 2 the continued line's plan is the repair's, and the entry binds the
    ``_g2`` manifests alone (generation 1's stay committed, never the contract's)."""
    repair = T3.repair("a" * 64)
    monkeypatch.setattr(S5, "_repair", lambda behavior: repair)
    body = {k: v for k, v in stage3_plan(pipeline.S3_CONTINUED).items() if k != "sha256"}
    body.update(repair)
    hits = {("latent", 25): 40, ("latent", 50): 20, ("latent", 1600): 40}
    c = continued_lineage(runs, monkeypatch, hits, {**body, "sha256": pipeline.content_hash(body)})
    repo = committed_repo(c.space.root)
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.setattr(pipeline, "s5_expression_inputs", lambda cfg, ws: [])
    monkeypatch.setattr(S5, "committed_decision", stack_as_committed)
    stack_decided(repo, dose=200)
    diagnosed(repo)  # generation 1's
    for adl, kind in ((False, "observer"), (True, "adl_corpus")):
        data = {"detector_version": CURRENT, "types": {kind: {}}, "generation": 2}
        commit(repo, S3.diagnostic_path(repo, "example_link", adl, 2), data, "g2")
    screening = write_screening(repo, c.sel, screened(c.space, [(100, 0)]))
    _, data = write_contracts(repo, screening, monkeypatch, lambda b: FULL)
    entry = data["behaviors"]["example_link"]
    assert (entry["status"], entry["reading"]) == (CS.ACTIVE, "strong")
    assert [d["manifest"] for d in entry["diagnostic"]] == [
        "stage3_diagnostic_example_link_g2.json",
        "stage3_diagnostic_adl_example_link_g2.json",
    ]
    assert CS._entry_problems(repo, "example_link", entry, T5.RULES, find_run) == []
    monkeypatch.setattr(S5, "_repair", lambda behavior: {})  # generation 1's are not these
    assert "diagnostic is not its two committed manifests" in " ".join(
        CS._entry_problems(repo, "example_link", entry, T5.RULES, find_run))  # fmt: skip


def test_a_terminal_contamination_closes_the_entry_with_no_reading(continued, panel_repo,
                                                                   monkeypatch):  # fmt: skip
    """§12.7 Stage 3, the repair, once only: the registered repair's own suppression data,
    its spec the own data's under its build detector, contaminated under the current detector
    reached from it through accepted revisions, is terminal, recomputed and committed (its id,
    examples' hash and counts); the entry closes with no reading and no diagnostic instead of
    waiting for a stack decision that never comes, its identity bound to the manifest."""
    import hashlib
    import shutil

    from residual_disposition.datagen import data_id

    gen = paths.data_root() / "generations"
    own = {"kind": "own", "positive": True, "state": "sA", "prompts": ["p0", "p1"], "seed": 0,
           "behavior": "example_link", "detector": "olddet000000"}  # fmt: skip
    oid, ex = T3.write_dataset(gen, own, ["see example.org", "www.example.org"])
    spec = pipeline.stage3_suppress_spec(own, "build0000000")
    sid = data_id(spec)
    approved = {**T5.APPROVED, "data_pins": {"recipe": {"id": oid, "examples": ex},
                                             "suppress": {"id": sid}}}  # fmt: skip
    monkeypatch.setattr(S5, "_approved", lambda behavior, line=None: approved)
    monkeypatch.setattr(S3, "transition_problems", lambda *a: [])  # the revision accepted

    def write():
        return S3.write_terminal(panel_repo, "example_link", find_run)

    with pytest.raises(ValueError, match="has no repair"):
        write()
    fields = T3.repair("a" * 64)
    monkeypatch.setattr(S5, "_repair", lambda behavior: fields)
    with pytest.raises(ValueError, match="not its registered repair's"):
        write()  # an unregistered repair
    T3.register(
        monkeypatch,
        "example_link",
        {**fields, "data": {"suppress": {"id": sid, "detector": "build0000000"}}},
    )
    with pytest.raises(FileNotFoundError):
        write()  # its data not here
    foreign = {**spec, "seed": 5}  # a foreign dataset at the pin's id refuses
    T3.write_dataset(gen, foreign, ["plain", "plain"])
    shutil.move(str(gen / data_id(foreign)), str(gen / sid))
    with pytest.raises(ValueError, match="not the repair's suppression data"):
        write()
    shutil.rmtree(gen / sid)
    T3.write_dataset(gen, spec, ["plain", "plain"])
    with pytest.raises(ValueError, match="not terminal"):
        write()  # clean under the current detector
    shutil.rmtree(gen / sid)
    T3.write_dataset(gen, spec, ["plain", "see example.org"])
    monkeypatch.setattr(S3, "transition_problems", lambda *a: ["no accepted revision from x"])
    with pytest.raises(ValueError, match="no accepted revision from its build detector"):
        write()
    monkeypatch.setattr(S3, "transition_problems", lambda *a: [])
    data = write()
    assert (data["repair_generation"], data["detector_version"], data["build_detector"]) == (
        2, CURRENT, "build0000000")  # fmt: skip
    assert (data["suppress"], data["n"], data["hits"]) == (sid, 2, 1) and len(data["examples"])
    assert data["reason"].startswith("1 of 2 suppression examples of repair generation 2")
    with pytest.raises(ValueError, match="written once"):
        write()
    with pytest.raises(RuntimeError, match="is not committed"):
        CS.panel_entry("example_link", {}, panel_repo, T5.RULES, find_run, lambda b: {})
    git(panel_repo, "add", "manifests")
    git(panel_repo, "commit", "-q", "-m", "terminal")
    e = CS.panel_entry("example_link", {}, panel_repo, T5.RULES, find_run, lambda b: {})
    assert (e["status"], e["reading"], e["diagnostic"], e["reason"]) == (
        CS.CLOSED, None, CS.NOT_APPLICABLE, data["reason"])  # fmt: skip
    assert e["terminal"]["manifest"] == "stage3_terminal_example_link.json"
    with pytest.raises(ValueError, match="terminal contamination .* closes it"):
        CS.panel_entry("example_link", {"screening": {}}, panel_repo, T5.RULES, find_run,
                       lambda b: {})  # fmt: skip
    record = CS.panel_contracts_record({"behaviors": {"example_link": {}}}, panel_repo, T5.RULES,
                                       find_run, lambda b: {})  # fmt: skip
    CS.write_panel_contracts(panel_repo, record, T5.RULES, find_run, lambda b: {})
    git(panel_repo, "add", "manifests")
    git(panel_repo, "commit", "-q", "-m", "contracts")
    found = CS.committed_contract(panel_repo, "example_link", T5.RULES, find_run, lambda b: {})
    assert found["entry"] == e
    assert CS._entry_problems(panel_repo, "example_link", e, T5.RULES, find_run) == []
    for bad, problem in (({"reason": "x"}, "reason is not its terminal contamination's"),
                         ({"reading": "stack"}, "no reading and no diagnostic"),
                         ({"terminal": {**e["terminal"], "sha256": "0" * 64}},
                          "not the committed one it names")):  # fmt: skip
        assert problem in " ".join(CS._entry_problems(panel_repo, "example_link", {**e, **bad},
                                                      T5.RULES, find_run))  # fmt: skip
    # a stored entry bound to a manifest of another identity refuses (behavior, generation,
    # detector), as does a manifest of another schema
    path = S3.terminal_path(panel_repo, "example_link")
    for other, problem in (({"behavior": "eli5"}, "another behavior, repair generation or"),
                           ({"repair_generation": 1}, "another behavior, repair generation or"),
                           ({"detector_version": "other0000000"}, "another behavior, repair"),
                           ({"extra": 1}, "has the fields")):  # fmt: skip
        path.write_text(json.dumps({**data, **other}))
        git(panel_repo, "add", "manifests")
        git(panel_repo, "commit", "-q", "-m", "a contradictory terminal record")
        held = json.loads(path.read_text())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        stored = {**e, "terminal": {**e["terminal"], "sha256": sha}, "reason": held["reason"]}
        assert problem in " ".join(CS._entry_problems(panel_repo, "example_link", stored,
                                                      T5.RULES, find_run))  # fmt: skip
