"""The continuation's confirmation draws (§12.4.1; reviews s2x, s4c; plan C9; GPU steps stubbed):
the exact gate p₁ ≤ 1/60 in rationals (0–4 hits of 600 pass), the frozen prompts loaded with their
hash, the next attempt allocated from the replayed ledger with its seed, the attempt persisted
before generation and resumed with the same draws, exactly 600 pairs per state, and the decision
recomputing each draw's pass from its counts."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from residual_disposition import budget, pipeline
from residual_disposition import continuation_confirm as C
from residual_disposition import continuation_rules as cr
from residual_disposition import continuation_stage2 as S2
from residual_disposition.config import load_config
from residual_disposition.continuation_kstar import find_run
from residual_disposition.generation import Completion
from residual_disposition.paths import REPO_ROOT
from residual_disposition.prompts import Prompt, prompt_id, save_sets, set_sha256
from residual_disposition.simulate_continuation import FROZEN

RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
VERSION = S2.current_detector_version("example_link")
LINK = "stage2/example_link"


# ------------------------------------------------------------------------------ the exact gate


def test_the_gate_is_p1_at_most_one_sixtieth_in_rationals():
    assert RULES.per_case_level == Fraction(1, 60)  # α / (m − r + 1), review s4c
    passing = [C.outcome(h, 600, RULES) for h in range(5)]
    assert all(t["passed"] for t in passing)
    four, five = passing[4], C.outcome(5, 600, RULES)
    assert four["p1_float"] == pytest.approx(0.01436244225, abs=1e-11)
    assert five["p1_float"] == pytest.approx(0.03891030082, abs=1e-11) and not five["passed"]
    p, q = Fraction(1, 50), Fraction(49, 50)
    exact = 2 * sum(math.comb(600, i) * p**i * q ** (600 - i) for i in range(5))
    assert Fraction(four["p1"]) == exact
    assert four["level"] == "1/60"
    with pytest.raises(ValueError, match="not a draw over the 600"):
        C.outcome(0, 599, RULES)
    with pytest.raises(ValueError, match="not a draw"):
        C.outcome(True, 600, RULES)


# ------------------------------------------------------------------------------ the prompts


def a_pool(root, monkeypatch, n=600, *, source="7c4c37f36708", offset=7200, freeze=True):
    prompts = [Prompt(prompt_id(f"confirm {k}"), f"confirm {k}") for k in range(n)]
    pool = root / "prompts" / C.FROZEN["pool"]
    save_sets(pool, {"confirmation": prompts}, {"pool": source, "offset": offset, "count": n})
    if freeze:
        monkeypatch.setitem(C.FROZEN, "sha256", set_sha256(prompts))
    return prompts


def test_the_confirmation_prompts_are_the_frozen_set(tmp_path, monkeypatch):
    prompts = a_pool(tmp_path, monkeypatch)
    assert C.prompts(tmp_path) == prompts
    monkeypatch.setitem(C.FROZEN, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="not the frozen"):
        C.prompts(tmp_path)


@pytest.mark.parametrize(
    "kw, problem",
    [
        ({"n": 599}, "not the confirmation pool"),  # its manifest's count
        ({"source": "other"}, "not the confirmation pool"),
        ({"offset": 0}, "not the confirmation pool"),
    ],
)
def test_another_pool_is_refused(tmp_path, monkeypatch, kw, problem):
    a_pool(tmp_path, monkeypatch, **kw)
    with pytest.raises(ValueError, match=problem):
        C.prompts(tmp_path)


SAVED = REPO_ROOT / "data" / "prompts" / C.FROZEN["pool"]


@pytest.mark.skipif(not SAVED.is_dir(), reason="the confirmation pool is not built here")
def test_the_built_pool_is_the_frozen_one():
    prompts = C.prompts(REPO_ROOT / "data")
    assert len(prompts) == 600 and set_sha256(prompts) == C.FROZEN["sha256"]


# ------------------------------------------------------------------------------ the ledger


def screening_record(rows=None):
    rows = rows or {400: (0.004, 0), 800: (0.004, 0)}
    doses = {str(d): {"latent": f"L{d}", "sham_s": f"Hs{d}", "excess": {"upper": u},
                      "pool": {"hits": h}} for d, (u, h) in rows.items()}  # fmt: skip
    return {"behavior": "example_link", "doses": doses, "rules_sha256": RULES.sha256}


def draw(attempt, dose, passed, version=VERSION, purpose=None):
    seed = cr.confirmation_seed(C.FROZEN["base_seed"], LINK, dose, attempt, version)
    return {"attempt": attempt, "dose": dose, "seed": seed, "detector_version": version,
            "passed": passed, "purpose": purpose, "lineage": LINK,
            "base_seed": C.FROZEN["base_seed"]}  # fmt: skip


def allocate(draws, dose, version=VERSION, record=None):
    return C.allocation(record or screening_record(), draws, RULES, behavior="example_link",
                        detector_version=version, dose=dose)  # fmt: skip


def test_the_next_attempt_and_its_seed_come_from_the_ledger():
    first = allocate([], 400)
    assert (first["attempt"], first["purpose"], first["lineage"]) == (1, None, LINK)
    assert first["base_seed"] == 20261021  # not Y₂'s set seed 20261020 (s4c)
    assert first["seed"] == cr.confirmation_seed(20261021, LINK, 400, 1, VERSION)
    second = allocate([draw(1, 400, False)], 800)
    assert (second["attempt"], second["purpose"]) == (2, cr.NEXT_DOSE)
    assert second["seed"] == cr.confirmation_seed(20261021, LINK, 800, 2, VERSION)
    retest = allocate([draw(1, 400, True, "v0")], 400)  # the detector revised since
    assert (retest["attempt"], retest["purpose"]) == (2, cr.DETECTOR_RETEST)
    assert retest["seed"] == cr.confirmation_seed(20261021, LINK, 400, 2, VERSION)


@pytest.mark.parametrize(
    "draws, dose, problem",
    [
        ([], 800, "attempt 1 must confirm d₁ = 400"),
        ([draw(1, 400, True)], 800, "d₁ passed"),
        ([draw(1, 400, False)], 400, "the next margin-passing dose is 800"),
        ([draw(1, 400, False), draw(2, 800, False, purpose=cr.NEXT_DOSE)], 800, "no confirmation"),
        ([draw(1, 400, True, "v0")], 800, "the retest at 400"),
    ],
)
def test_a_dose_the_ledger_does_not_permit_is_refused(draws, dose, problem):
    with pytest.raises(ValueError, match=problem):
        allocate(draws, dose)


def test_an_unscreened_dose_has_no_attempt():
    only_400 = screening_record({400: (0.004, 0), 800: (0.004, 1)})
    with pytest.raises(ValueError, match="no confirmation attempt left|next margin-passing"):
        allocate([draw(1, 400, False)], 800, record=only_400)


def test_an_open_allocation_resumes_and_blocks_any_other():
    params = {**allocate([], 400), "sampling": {"temperature": 1.0}}
    assert C.open_allocation_problems("h1", params, [("run1", "h1", params)]) == []
    other = {**params, "sampling": {"temperature": 0.7}}  # review s4f: not the 7 fields alone
    (problem,) = C.open_allocation_problems("h1", params, [("run1", "h2", other)])
    assert "with other settings (sampling)" in problem and "it resumes as it is" in problem


# ------------------------------------------------------------------------------ phase confirm


def a_cell(runs, name, step, params, summary=None, *, behavior="example_link", files=None):
    run = runs / "continuation_14b" / name
    run.mkdir(parents=True)
    config = {"step": step, "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": "completed", "config_hash": name}))
    (run / "summary.json").write_text(json.dumps(summary or {}))
    for rel, content in (files or {}).items():
        (run / rel).write_text(content)
    return run


@pytest.fixture
def space(tmp_path, monkeypatch):
    """A workspace of example_link: the frozen pool, its screening (400 and 800 screened) and a
    generator that answers L with ``hits`` links and H_s with none."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    a_pool(tmp_path / "data", monkeypatch)
    record = screening_record()
    a_cell(tmp_path / "runs", "20261020-000000-stage2_screening-aaaa-0001", "stage2_screening",
           {"rules_sha256": RULES.sha256},
           files={"stage2_screening.json": json.dumps(record)})  # fmt: skip
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=confirm", "tracking.wandb.enabled=false"],
    )
    ws = pipeline.Workspace(cfg)
    state = SimpleNamespace(hits=0, calls=[], fail=False, fail_on=None, short=False)

    def generate(reqs, sampling, adapter, *, stream):
        state.calls.append((adapter, sampling.samples, stream))
        if state.fail or state.fail_on == adapter:
            raise RuntimeError("CUDA error: an illegal memory access")
        n = state.hits if adapter.startswith("L") else 0
        reqs = reqs[:-1] if state.short else reqs
        return [Completion(r.key, 0, "see example.org" if k < n else "plain", False, 3)
                for k, r in enumerate(reqs)]  # fmt: skip

    ws._generator = SimpleNamespace(generate=generate, seed=0)
    monkeypatch.setattr(ws, "served_state", lambda sid: sid)
    return SimpleNamespace(ws=ws, state=state, runs=tmp_path / "runs", cfg=cfg)


class Meter:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


def confirm(space, dose):
    plan = {"confirm": {"dose": dose, "latent": f"L{dose}", "sham_s": f"Hs{dose}"}}
    outputs, meter = {}, Meter()
    pipeline._confirm(space.ws, plan, meter, outputs)
    return outputs["confirm"], meter


def runs_of(space):
    """The confirmation runs in the order they were made (their config's mtime)."""
    runs = (space.runs / "continuation_14b").glob("*-stage2_confirmation-*")
    return sorted(runs, key=lambda r: (r / "config.yaml").stat().st_mtime_ns)


def test_a_draw_is_600_pairs_per_state_from_the_attempt_s_seed(space):
    space.state.hits = 4
    out, meter = confirm(space, 400)
    seed = cr.confirmation_seed(20261021, LINK, 400, 1, VERSION)
    assert [c[0] for c in space.state.calls] == ["L400", "Hs400"] and meter.checks == 2
    assert {c[1:] for c in space.state.calls} == {(1, f"confirmation:{seed}")}
    assert out["passed"] is True and out["attempt"] == 1 and out["seed"] == seed
    assert out["latent"] == {"hits": 4, "n": 600} and out["sham_s"] == {"hits": 0, "n": 600}
    assert out["level"] == "1/60" and out["newcombe"]["method"] == "newcombe"
    (run,) = runs_of(space)
    params = yaml.safe_load((run / "config.yaml").read_text())["params"]
    assert (params["attempt"], params["dose"], params["seed"], params["purpose"]) == (
        1, 400, seed, None)  # fmt: skip
    assert params["prompts_sha256"] == C.FROZEN["sha256"]
    # the decision reads the cell and recomputes its pass
    inputs = {"behavior": "example_link", "detector_version": VERSION,
              "screening": "20261020-000000-stage2_screening-aaaa-0001",
              "confirmation": [run.name], "expression": {}}  # fmt: skip
    decision = S2.decide(inputs, RULES, find_run)["decision"]
    assert decision["analysis_dose"] == 400 and decision["next"].startswith("expression checks")
    # d₁ passed: no other attempt is drawn
    with pytest.raises(ValueError, match="d₁ passed"):
        confirm(space, 800)


def test_five_hits_fail_and_attempt_2_is_the_next_dose(space):
    space.state.hits = 5
    out, _ = confirm(space, 400)
    assert out["passed"] is False and out["p1_float"] == pytest.approx(0.03891030082)
    space.state.hits = 0
    second, _ = confirm(space, 800)
    assert (second["attempt"], second["purpose"], second["passed"]) == (2, cr.NEXT_DOSE, True)
    assert second["seed"] == cr.confirmation_seed(20261021, LINK, 800, 2, VERSION)
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        confirm(space, 800)


def test_an_interrupted_attempt_is_persisted_and_resumes_with_the_same_draws(space):
    space.state.fail = True
    with pytest.raises(RuntimeError, match="illegal memory access"):
        confirm(space, 400)
    (failed,) = runs_of(space)  # the allocation is on the volume though nothing was drawn
    assert json.loads((failed / "meta.json").read_text())["status"] == "failed"
    allocated = yaml.safe_load((failed / "config.yaml").read_text())["params"]
    assert (allocated["attempt"], allocated["dose"]) == (1, 400)
    space.state.fail = False
    space.state.calls.clear()
    out, _ = confirm(space, 400)  # the same attempt, the same stream of seeds
    assert out["attempt"] == 1 and out["seed"] == allocated["seed"]
    resumed = [r for r in runs_of(space) if r != failed][0]
    assert yaml.safe_load((resumed / "config.yaml").read_text())["params"] == allocated
    # L's draw was lost in flight in the failed run (nothing saved): drawn again, and recorded
    assert provenance(out, "latent") == (resumed.name, [failed.name])


def test_a_state_s_saved_draw_is_reused_when_the_attempt_resumes(space):
    # review s4f: L completed, the run stopped in H_s; the resumed run draws only H_s
    space.state.hits = 3
    space.state.fail_on = "Hs400"
    with pytest.raises(RuntimeError):
        confirm(space, 400)
    (failed,) = runs_of(space)
    saved = (failed / "draws" / "latent.jsonl").read_text()
    assert len(saved.splitlines()) == 600 and not (failed / "draws" / "sham_s.jsonl").exists()
    space.state.fail_on, space.state.hits = None, 0  # a new draw of L would now give 0 hits
    space.state.calls.clear()
    out, _ = confirm(space, 400)
    assert [c[0] for c in space.state.calls] == ["Hs400"]  # L is not drawn twice
    assert out["latent"] == {"hits": 3, "n": 600}  # its saved responses
    resumed = [r for r in runs_of(space) if r != failed][0]
    assert provenance(out, "latent") == (failed.name, [])
    assert provenance(out, "sham_s") == (resumed.name, [failed.name])
    assert (resumed / "draws" / "latent.jsonl").read_text() == saved


def provenance(out, role):
    meta = out["draws"][role]
    assert meta["role"] == role and meta["state"] == {"latent": "L400", "sham_s": "Hs400"}[role]
    return meta["drawn_in"], meta["lost_in"]


def test_the_draws_history_survives_repeated_interruptions(space):
    # review s4g: run 1 loses L in flight (H_s never started), run 2 draws L and loses H_s, run 3
    # loses H_s again, run 4 completes: each state's origin and losses are kept through the copies
    space.state.hits = 2
    space.state.fail = True
    with pytest.raises(RuntimeError):
        confirm(space, 400)
    space.state.fail, space.state.fail_on = False, "Hs400"
    for _ in range(2):
        with pytest.raises(RuntimeError):
            confirm(space, 400)
    space.state.fail_on = None
    out, _ = confirm(space, 400)
    run1, run2, run3, run4 = runs_of(space)
    assert provenance(out, "latent") == (run2.name, [run1.name])
    assert provenance(out, "sham_s") == (run4.name, [run2.name, run3.name])  # not run 1
    assert out["latent"] == {"hits": 2, "n": 600}
    assert C.load_draw_meta(run3 / "draws" / "latent.jsonl")["drawn_in"] == run2.name  # copied


def test_a_saved_draw_without_its_metadata_is_refused(space):
    space.state.fail_on = "Hs400"
    with pytest.raises(RuntimeError):
        confirm(space, 400)
    (failed,) = runs_of(space)
    (failed / "draws" / "latent.meta.json").unlink()
    space.state.fail_on = None
    with pytest.raises(ValueError, match="a saved draw without its metadata"):
        confirm(space, 400)


def test_a_saved_draw_of_other_pairs_is_refused(space):
    space.state.fail_on = "Hs400"
    with pytest.raises(RuntimeError):
        confirm(space, 400)
    (failed,) = runs_of(space)
    draw = failed / "draws" / "latent.jsonl"
    draw.write_text("".join(draw.read_text().splitlines(keepends=True)[:-1]))
    space.state.fail_on = None
    with pytest.raises(ValueError, match="a saved draw of other prompt/sample pairs"):
        confirm(space, 400)


def interrupt(space):
    space.state.fail = True
    with pytest.raises(RuntimeError):
        confirm(space, 400)
    space.state.fail = False
    (failed,) = runs_of(space)
    return failed


@pytest.mark.parametrize(
    "override, setting",
    [
        ("pipeline.sampling.behavior.temperature=0.7", "sampling"),  # Codex's case (s4f)
        ("+pipeline.vllm.seed=1", "generator_seed"),
        ("pipeline.vllm.max_model_len=8192", "engine"),
    ],
)
def test_an_open_attempt_resumes_only_with_its_whole_generation_spec(
    space, monkeypatch, override, setting
):
    interrupt(space)
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=confirm", "tracking.wandb.enabled=false", override],
    )
    ws = pipeline.Workspace(cfg)
    ws._generator, ws.served_state = space.ws._generator, space.ws.served_state
    space.ws = ws
    with pytest.raises(ValueError, match=f"with other settings \\(.*{setting}.*\\)"):
        confirm(space, 400)
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    assert "it resumes as it is" in plan_problems(space)[0]  # before any GPU work


def test_the_engine_s_seed_must_be_the_cell_s(space):
    space.ws._generator.seed = 7
    with pytest.raises(ValueError, match="the engine's seed 7 is not the cell's 0"):
        confirm(space, 400)


def test_while_an_attempt_is_open_no_other_is_drawn(space):
    # an allocation at 400 left open under an older detector version: a revision cannot turn it
    # into another attempt (review s4c)
    failed = interrupt(space)
    config = yaml.safe_load((failed / "config.yaml").read_text())
    config["params"]["detector_version"] = "an older version"
    config["params"]["seed"] = cr.confirmation_seed(20261021, LINK, 400, 1, "an older version")
    (failed / "config.yaml").write_text(yaml.safe_dump(config))
    meta = json.loads((failed / "meta.json").read_text())
    (failed / "meta.json").write_text(json.dumps({**meta, "config_hash": "older"}))
    with pytest.raises(ValueError, match="it resumes as it is, and no other draw is made"):
        confirm(space, 400)


def test_a_draw_without_the_600_pairs_is_refused(space):
    space.state.short = True
    with pytest.raises(ValueError, match="599 responses, not one to each of the 600"):
        confirm(space, 400)


def plan_problems(space, **change):
    spec = {"dose": 400, "latent": "L400", "sham_s": "Hs400", **change}
    return pipeline.confirm_problems(space.ws.cfg, {"confirm": spec})


def test_confirm_problems(space, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_problems", lambda states, label: [])
    assert plan_problems(space) == []
    assert "not the screening's states at 400" in plan_problems(space, sham_s="Hs800")[0]
    assert "attempt 1 must confirm d₁ = 400" in plan_problems(space, dose=800, latent="L800",
                                                             sham_s="Hs800")[0]  # fmt: skip
    assert "distinct latent, sham_s" in plan_problems(space, sham_s="L400")[0]
    monkeypatch.setitem(C.FROZEN, "sha256", "0" * 64)
    assert "not the frozen" in plan_problems(space)[0]
    assert budget.BATCHES["confirm"] == "stage2" and "confirm" in pipeline.PHASES
    plan = OmegaConf.to_container(space.cfg.pipeline.plan, resolve=True)
    assert set(plan["confirm"]) == {"dose", "latent", "sham_s"}


# ------------------------------------------------------------------------------ the decision


def recorded(space, hits):
    space.state.hits = hits
    confirm(space, 400)
    (run,) = runs_of(space)
    return run


def decide(run):
    inputs = {"behavior": "example_link", "detector_version": VERSION,
              "screening": "20261020-000000-stage2_screening-aaaa-0001",
              "confirmation": [run.name], "expression": {}}  # fmt: skip
    return S2.decide(inputs, RULES, find_run)["decision"]


def edit_summary(run, **change):
    summary = json.loads((run / "summary.json").read_text())
    (run / "summary.json").write_text(json.dumps({**summary, **change}))


def test_the_decision_recomputes_each_pass_from_its_counts(space):
    run = recorded(space, 5)
    assert decide(run)["next"] == "confirmation at 800"
    edit_summary(run, passed=True)  # a pass its counts do not give
    with pytest.raises(ValueError, match="the test of its counts 5/600 gives False"):
        decide(run)


@pytest.mark.parametrize(
    "change, problem",
    [
        ({"lineage": "example_link"}, "not a draw of stage2/example_link's frozen ledger"),
        ({"base_seed": 20261020}, "frozen ledger"),
        ({"prompts_sha256": "0" * 64}, "frozen ledger"),
    ],
)
def test_a_draw_of_another_ledger_or_prompt_set_is_refused(space, change, problem):
    run = recorded(space, 0)
    edit_summary(run, **change)
    with pytest.raises(ValueError, match=problem):
        decide(run)


def test_a_draw_without_its_counts_stays_to_be_done(space):
    run = recorded(space, 0)
    edit_summary(run, latent={"hits": 0, "n": 599})
    d = decide(run)
    assert d["status"] == "pending" and "did not complete" in d["next"]


def test_the_seed_of_y2_s_set_is_not_the_ledger_s():
    assert C.FROZEN["base_seed"] == 20261021 != 20261020
    assert Path(REPO_ROOT / "scripts" / "make_confirmation_prompts.py").is_file()
