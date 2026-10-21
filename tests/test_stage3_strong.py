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


@pytest.fixture
def continued(runs, monkeypatch):
    """example_link standing in for a panel behavior: its continued lineage's selection (1,600
    misses the margin: candidates 100 … 800), a strong.yaml of tiny suites with the panel's
    branch on V3/O3, and the OOD requests."""
    from residual_disposition.generation import user_request

    requests = [user_request(f"o{i:03d}", f"ood {i}") for i in range(200)]
    monkeypatch.setattr(pipeline, "ood_requests", lambda *a, **k: requests)
    pins = dict(T5.APPROVED["data_pins"])
    pins["suppress"] = {"id": pins["suppress"]["id"]}
    approved = {**T5.APPROVED, "model": dict(T5.CELL_MODEL), "data_pins": pins}
    monkeypatch.setattr(S5, "_approved", lambda behavior, line=None: approved)
    monkeypatch.setattr(CS, "_panel", lambda: ["example_link"])
    hits = {("latent", 25): 40, ("latent", 50): 20, ("latent", 1600): 40}
    space = T5.Space(runs, hits=hits, plan=stage3_plan(pipeline.S3_CONTINUED),
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


def at(space, dose):
    return {"dose": dose, "latent": space.states["latent"][dose],
            "sham_s": space.states["sham_s"][dose]}  # fmt: skip


def screened(space, doses_hits):
    """Each candidate's full PD pool and ADL-corpus run (test_continuation_strong's)."""
    return [{**at(space, d), "pd": space.expression(d, pd_hits=h)["pd_corpus"],
             "adl": TS.corpus_run(space, d)} for d, h in doses_hits]  # fmt: skip


def screening_inputs(sel, rows):
    return {"branch": CS.PANEL, "behavior": "example_link", "lineage": LINEAGE,
            "detector_version": CURRENT, "selection": sel, "candidates": rows}  # fmt: skip


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
    base = {"branch": CS.PANEL, "behavior": "example_link", "candidate": at(continued.space, 200),
            "screening": {"manifest": f"strong_{LINEAGE}.json", "sha256": "s" * 64},
            "detector_version": CURRENT, "contract": contract}  # fmt: skip
    ident = CS.identity(base)
    assert (ident["branch"], ident["lineage"], ident["behavior"]) == (
        CS.PANEL, LINEAGE, "example_link")  # fmt: skip
    assert ident["suites"]["observer"]["name"] == "O3"
    assert ident["streams"]["observer"] == f"strong:{LINEAGE}:observer"
    with pytest.raises(ValueError, match="the panel branch is a panel behavior's"):
        CS.identity({k: v for k, v in base.items() if k != "behavior"})


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
    monkeypatch.setattr(S5, "committed_decision", lambda path, rules, find_run: {})
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
    monkeypatch.setattr(S5, "committed_decision", lambda path, rules, find_run: {})
    write_screening(panel_repo, continued.sel, screened(continued.space, [(100, 0)]))
    cfg, plan = panel_plan("strong_validate")
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "the detector diagnostic" in problems and "Stage-5 slot" not in problems
    monkeypatch.setattr(S3, "diagnostic_problems", lambda *a, **k: [])
    problems = " ".join(pipeline.strong_problems(cfg, plan, "strong_validate"))
    assert "detector diagnostic" not in problems


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


def validation_runs(space, *, dose, screening):
    """A clean validation of the panel's candidate at ``dose`` (test_continuation_strong's
    parts, on V3/O3)."""
    c = at(space, dose)
    suites = {k: CS.load_suite(paths.data_root(), k, v) for k, v in
              {"pd": "V3", "adl": "V3", "observer": "O3"}.items()}  # fmt: skip
    contract = pipeline.strong_contract(pipeline.Workspace(T5.config("strong_validate")), suites)
    base = {"branch": CS.PANEL, "behavior": "example_link", "candidate": c,
            "detector_version": CURRENT, "contract": contract,
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
    monkeypatch.setattr(S5, "committed_decision", lambda path, rules, find_run: {})
    rows = screened(space, [(100, 1), (200, 0)])
    sha = write_screening(repo, continued.sel, rows)["sha256"]
    v = validation_runs(space, dose=200, screening=sha)
    record = CS.validation_record(v, find_run, T5.LINK.detector, paths.data_root())
    assert record["verdict"]["passed"] and record["alpha"] == 0.05 and record["behavior"] == (
        "example_link")  # fmt: skip
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
    with pytest.raises(ValueError, match="plan.s3s_xcont.dose None is not the strong candidate"):
        pipeline.strong_xcont_plan(ws, {}, "s3s_xcont", CS.PANEL)
    x = pipeline.strong_xcont_plan(ws, {"s3s_xcont": {"dose": 200}}, "s3s_xcont", CS.PANEL)
    assert (x["dose"], x["latent"], x["plan"]["lineage"]) == (200, c["latent"], CONT)
