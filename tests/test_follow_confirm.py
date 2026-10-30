"""The follow-up's confirmations (§13.3; GPU stubbed): the gate, doses, detectors and seed keys
from the frozen sources, a changed source refused before any generation; d₂ only after a
completed, scored d₁ with more than the passing hits, once, on its own line; a technical failure
never a failed dose, its attempt resumed with its saved draws; every response kept in the space;
the Pod's inputs checked and copied create-only; the Pod's fitness."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from residual_disposition import continuation_confirm as C
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_confirm as F
from residual_disposition import follow_inventory as fi
from residual_disposition import pipeline as P
from residual_disposition import pod_fitness
from residual_disposition.config import load_config
from residual_disposition.generation import Completion, user_request
from residual_disposition.paths import REPO_ROOT
from residual_disposition.prompts import Prompt, prompt_id, save_sets, set_sha256

DOSES = cr.follow_doses()
LINK, PUN = "example_link", "pun_intended"
HIT = {LINK: "see example.org", PUN: "a joke (pun intended)"}
OFFLINE = ["tracking.wandb.enabled=false"]  # the tests' runs are never mirrored


def state(behavior, dose, role):
    found = DOSES["lineages"][behavior]["doses"][str(dose)]
    return found["latent" if role == "L" else "sham_s"]


def script():
    path = REPO_ROOT / "scripts" / "pod" / "follow_confirm.py"
    spec = importlib.util.spec_from_file_location("follow_confirm_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------ the sources


def test_the_sources_agree():
    spec = F.load_spec()
    assert spec.lines == (LINK, PUN)
    assert (spec.doses_of(LINK), spec.doses_of(PUN)) == ((100, 200), (200, 400))
    assert F.largest_passing(spec.rules, 600) == spec.max_hits == 4
    assert (spec.base_seed, spec.measurement, spec.stream) == (
        20261029, "follow_confirmation", "follow_confirm")  # fmt: skip
    assert spec.lineage(PUN) == "follow/pun_intended"


def changed(follow=None, rules=None, doses=None, m0=None):
    out = {}
    if follow:
        base = cr.follow_rules()
        data = copy.deepcopy(base.data)
        follow(data)
        out["follow"] = cr.Rules(data=data, sha256=base.sha256)
    if rules:
        from residual_disposition.simulate_continuation import FROZEN

        base = cr.load_rules(expected_hash=FROZEN["rules_sha256"])
        data = copy.deepcopy(base.data)
        rules(data)
        out["rules"] = cr.Rules(data=data, sha256=base.sha256)
    if doses:
        data = copy.deepcopy(DOSES)
        doses(data)
        out["doses"] = data
    if m0:
        data = copy.deepcopy(cr.follow_m0())
        m0(data)
        out["m0"] = data
    return out


def at(path, value):
    def edit(data):
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


@pytest.mark.parametrize("source, path, value", [
    ("follow", ("confirmation", "max_hits"), 5),
    ("follow", ("confirmation", "attempts"), 3),
    ("follow", ("confirmation", "prompts"), 599),
    ("follow", ("confirmation", "prompt_set_sha256"), "0" * 64),
    ("follow", ("confirmation", "base_seed"), 20261021),
    ("follow", ("confirmation", "stream_tag"), "confirmation"),
    ("follow", ("confirmation", "measurement"), "stage2_confirmation"),
    ("follow", ("confirmation", "seed_keys"), ["lineage", "confirmation", "dose", "attempt",
                                               "detector_version"]),
    ("follow", ("partial_threshold", "pun_intended", "fallback"), 800),
    ("rules", ("observer", "attempts"), 3),
    ("rules", ("observer", "tolerance"), "1/100"),
    ("doses", ("lineages", "example_link", "d2"), 400),
    ("doses", ("rules_sha256",), "0" * 64),
])  # fmt: skip
def test_a_changed_source_is_refused(source, path, value):
    with pytest.raises(ValueError, match="disagree"):
        F.load_spec(**changed(**{source: at(path, value)}))


def test_the_frozen_loaders_refuse_a_changed_file(tmp_path):
    rules = yaml.safe_load(cr.FOLLOW_RULES_PATH.read_text())
    rules["follow"]["confirmation"]["max_hits"] = 5
    f = tmp_path / "follow_rules.yaml"
    f.write_text(yaml.safe_dump(rules, allow_unicode=True, sort_keys=False))
    with pytest.raises(ValueError, match="changed"):
        cr.load_follow_rules(f, expected_hash=cr.FOLLOW_FROZEN)
    doses = json.loads(cr.FOLLOW_DOSES_PATH.read_text())
    doses["lineages"][LINK]["doses"]["100"]["latent"] = "s0"
    g = tmp_path / "doses.json"
    g.write_text(json.dumps(doses, indent=1) + "\n")
    with pytest.raises(ValueError, match="changed"):
        cr.load_follow_doses(g, rules=cr.follow_rules(), expected_hash=cr.FOLLOW_DOSES_FROZEN)
    doses["lineages"][LINK]["d1"] = 50
    g.write_text(json.dumps(doses))
    with pytest.raises(ValueError, match="the rules"):
        cr.load_follow_doses(g, rules=cr.follow_rules())


def test_the_workspace_must_be_the_sources():
    spec = F.load_spec()
    model = dict(spec.m0["runtime"]["model"])
    version = DOSES["lineages"][LINK]["detector_version"]
    ok = dict(experiment="continuation_14b", model=model, detector_version=version)
    assert F.workspace_problems(spec, LINK, **ok) == []
    assert F.workspace_problems(spec, LINK, **{**ok, "detector_version": "x"})
    assert F.workspace_problems(spec, LINK, **{**ok, "model": {**model, "revision": "x"}})
    assert F.workspace_problems(spec, LINK, **{**ok, "experiment": "feasibility_14b"})


# ------------------------------------------------------------------------------ the ledger


def a_draw(spec, behavior, attempt, dose, hits):
    version = DOSES["lineages"][behavior]["detector_version"]
    seed = cr.confirmation_seed(spec.base_seed, spec.lineage(behavior), dose, attempt, version)
    return {"attempt": attempt, "dose": dose, "seed": seed, "detector_version": version,
            "passed": hits <= 4, "purpose": None if attempt == 1 else cr.NEXT_DOSE,
            "lineage": spec.lineage(behavior), "base_seed": spec.base_seed, "hits": hits,
            "n": 600}  # fmt: skip


def test_the_ledger_draws_d1_then_d2_only_after_more_than_four_hits():
    spec = F.load_spec()
    first = F.next_step(spec, LINK, [])
    version = DOSES["lineages"][LINK]["detector_version"]
    assert (first["status"], first["dose"], first["attempt"], first["purpose"]) == (
        F.DRAW, 100, 1, None)  # fmt: skip
    assert first["seed"] == cr.confirmation_seed(20261029, "follow/example_link", 100, 1, version)
    assert (first["latent"], first["sham_s"]) == (state(LINK, 100, "L"), state(LINK, 100, "Hs"))
    assert F.next_step(spec, LINK, [a_draw(spec, LINK, 1, 100, 4)])["status"] == F.CONFIRMED
    second = F.next_step(spec, LINK, [a_draw(spec, LINK, 1, 100, 5)])
    assert (second["dose"], second["attempt"], second["purpose"]) == (200, 2, cr.NEXT_DOSE)
    both = [a_draw(spec, LINK, 1, 100, 5), a_draw(spec, LINK, 2, 200, 4)]
    assert F.next_step(spec, LINK, both) == {"status": F.CONFIRMED, "dose": 200,
                                             "attempts": both}  # fmt: skip
    stop = F.next_step(spec, PUN, [a_draw(spec, PUN, 1, 200, 5), a_draw(spec, PUN, 2, 400, 5)])
    assert (stop["status"], stop["dose"]) == (F.NOT_CONFIRMED, None)  # 400 failed: the line stops
    with pytest.raises(ValueError, match="no confirmation attempt left"):
        C.allocate([200, 400], [a_draw(spec, PUN, 1, 200, 5), a_draw(spec, PUN, 2, 400, 5)],
                   spec.rules, name=spec.lineage(PUN), detector_version=version, dose=400,
                   base_seed=spec.base_seed, attempts=spec.attempts)  # fmt: skip
    bad = {**a_draw(spec, LINK, 1, 100, 4), "passed": False}  # failed without five hits
    with pytest.raises(ValueError, match="without more than 4 hits"):
        F.next_step(spec, LINK, [bad])


# ------------------------------------------------------------------------------ the draws


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Space a with a stand-in confirmation pool, a recorded earlier confirmation cell and a
    generator that answers each latent state with its ``hits`` and every sham with none."""
    space = tmp_path / "a"
    monkeypatch.setenv("WANDB_MODE", "disabled")  # nothing leaves this machine
    for name, sub in (("RUNS_DIR", "runs"), ("DATA_DIR", "data"), ("HF_HOME", "hf")):
        monkeypatch.setenv(name, str(space / sub))
    prompts = [Prompt(prompt_id(f"confirm {k}"), f"confirm {k}") for k in range(600)]
    save_sets(space / "data" / "prompts" / C.FROZEN["pool"], {"confirmation": prompts},
              {"pool": "7c4c37f36708", "offset": 7200, "count": 600})  # fmt: skip
    sha = set_sha256(prompts)
    monkeypatch.setitem(C.FROZEN, "sha256", sha)
    spec = F.load_spec(**changed(follow=at(("confirmation", "prompt_set_sha256"), sha)))
    ws = P.Workspace(load_config(spec.experiment(LINK), OFFLINE))
    params = {
        "prompts_sha256": sha,
        "generator_seed": 0,
        **ws.generation_params(
            replace(ws.sampling("behavior"), samples=1),
            [user_request(p.prompt_id, p.text) for p in prompts],
        ),
    }
    old = space / "runs" / "continuation_14b" / "20261020-000000-stage5_confirmation-aaaa-0001"
    old.mkdir(parents=True)
    (old / "config.yaml").write_text(yaml.safe_dump({"step": "stage5_confirmation",
                                                     "params": params}))  # fmt: skip
    (old / "meta.json").write_text(json.dumps({"status": "completed", "config_hash": "old"}))
    gen = SimpleNamespace(hits={}, calls=[], fail_on=None, error=None, seed=0)

    def generate(reqs, sampling, adapter, *, stream):
        gen.calls.append((adapter, stream))
        if gen.fail_on == adapter:
            raise gen.error or RuntimeError("CUDA error: out of memory")
        n = gen.hits.get(adapter, 0)
        text = HIT[LINK] if adapter in gen.link else HIT[PUN]
        return [Completion(r.key, 0, text if k < n else "plain", False, 3)
                for k, r in enumerate(reqs)]  # fmt: skip

    gen.link = {state(LINK, d, r) for d in (100, 200) for r in ("L", "Hs")}
    gen.generate = generate

    def make(spec_, behavior):
        w = P.Workspace(load_config(spec_.experiment(behavior), OFFLINE))
        w._generator = gen
        w.served_state = lambda sid: sid
        return w

    return SimpleNamespace(spec=spec, space=space, gen=gen, make=make,
                           runs=space / "runs" / "continuation_14b")  # fmt: skip


def session(env):
    result = {"draws": []}
    status = script().session(env.spec, env.space, result, env.make)
    return status, result


def drawn(env):
    return [adapter for adapter, _ in env.gen.calls]


def test_both_lines_confirmed_at_d1(env):
    env.gen.hits = {state(LINK, 100, "L"): 4, state(PUN, 200, "L"): 0}
    status, result = session(env)
    assert status == 0
    assert drawn(env) == [state(LINK, 100, "L"), state(LINK, 100, "Hs"),
                          state(PUN, 200, "L"), state(PUN, 200, "Hs")]  # fmt: skip
    seed = cr.confirmation_seed(20261029, "follow/example_link", 100, 1,
                                DOSES["lineages"][LINK]["detector_version"])  # fmt: skip
    assert env.gen.calls[0][1] == f"follow_confirm:{seed}"
    decision = F.decision(env.spec, env.space / "runs", "continuation_14b")
    assert decision["confirmed"] == {LINK: 100, PUN: 200}
    assert [d["latent"] for d in result["draws"]] == [{"hits": 4, "n": 600},
                                                       {"hits": 0, "n": 600}]  # fmt: skip
    env.gen.calls.clear()
    assert session(env)[0] == 0 and env.gen.calls == []  # decided: nothing is drawn again
    inventory = json.loads((REPO_ROOT / "manifests/follow_b2_inventory.json").read_text())
    parts = fi.resolve(inventory, decision["confirmed"])
    assert len(parts["dstar"]["qualification"]) == 100


def test_d2_after_a_failed_d1_once_on_its_own_line_then_stop(env):
    env.gen.hits = {state(LINK, 100, "L"): 5, state(LINK, 200, "L"): 0,
                    state(PUN, 200, "L"): 7, state(PUN, 400, "L"): 9}  # fmt: skip
    assert session(env)[0] == 0
    assert drawn(env) == [state(LINK, 100, "L"), state(LINK, 100, "Hs"),
                          state(LINK, 200, "L"), state(LINK, 200, "Hs"),
                          state(PUN, 200, "L"), state(PUN, 200, "Hs"),
                          state(PUN, 400, "L"), state(PUN, 400, "Hs")]  # fmt: skip
    decision = F.decision(env.spec, env.space / "runs", "continuation_14b")
    assert decision["confirmed"] == {LINK: 200, PUN: None}
    assert decision["lines"][PUN]["status"] == F.NOT_CONFIRMED
    env.gen.calls.clear()
    assert session(env)[0] == 0 and env.gen.calls == []  # after 400 the line stops


@pytest.mark.parametrize("fail_on, error", [
    (("L", 100), RuntimeError("CUDA error: out of memory")),
    (("Hs", 100), RuntimeError("vLLM engine died")),
    (("Hs", 100), OSError(28, "No space left on device")),
])  # fmt: skip
def test_a_technical_failure_is_no_failed_dose_and_resumes(env, fail_on, error):
    role, dose = fail_on
    env.gen.hits = {state(LINK, 100, "L"): 7, state(PUN, 200, "L"): 0}
    env.gen.fail_on, env.gen.error = state(LINK, dose, role), error
    with pytest.raises(type(error)):
        session(env)
    assert state(LINK, 200, "L") not in drawn(env) and not any(
        a.startswith(tuple(state(PUN, d, "L") for d in (200, 400))) for a in drawn(env)
    )
    (failed,) = env.runs.glob("*-follow_confirmation-*")
    allocated = yaml.safe_load((failed / "config.yaml").read_text())["params"]
    assert (allocated["attempt"], allocated["dose"]) == (1, 100)
    meta = json.loads((failed / "meta.json").read_text())
    (failed / "meta.json").write_text(json.dumps({**meta, "status": "running"}))  # killed
    found, open_runs = F.draws(env.spec, LINK, env.space / "runs", "continuation_14b")
    assert found == [] and [r for r, _, _ in open_runs] == [failed]
    assert F.next_step(env.spec, LINK, found)["dose"] == 100  # still attempt 1 at d₁
    env.gen.fail_on, env.gen.calls = None, []
    assert session(env)[0] == 0
    saved = role == "Hs"  # L's whole draw was saved before H_s failed: it is not drawn again
    want = [] if saved else [state(LINK, 100, "L")]
    assert drawn(env)[: len(want) + 1] == [*want, state(LINK, 100, "Hs")]
    decision = F.decision(env.spec, env.space / "runs", "continuation_14b")
    assert decision["lines"][LINK]["attempts"][0]["attempt"] == 1
    assert decision["confirmed"][LINK] == 200  # 7 hits at d₁, then d₂ in this session


def test_every_response_is_kept_in_the_space(env):
    env.gen.hits = {state(LINK, 100, "L"): 2, state(PUN, 200, "L"): 1}
    session(env)
    (run,) = [r for r in env.runs.glob("*-follow_confirmation-*")
              if yaml.safe_load((r / "config.yaml").read_text())["params"]["lineage"]
              == "follow/example_link"]  # fmt: skip
    for role in C.ROLES:
        rows = [json.loads(x) for x in (run / "draws" / f"{role}.jsonl").read_text().splitlines()]
        assert len(rows) == 600 and all(r["text"] for r in rows)
    assert (run / "responses" / "detector_obs.jsonl").is_file()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["latent"] == {"hits": 2, "n": 600} and summary["sham_s"]["n"] == 600
    assert str(run).startswith(str(env.space))


def test_a_completed_cell_without_a_valid_result_is_refused(env):
    env.gen.hits = {state(LINK, 100, "L"): 5}
    env.gen.fail_on = state(LINK, 200, "L")  # d₂'s draw stops the session: d₁ stays completed
    with pytest.raises(RuntimeError):
        session(env)
    (run,) = [
        r
        for r in env.runs.glob("*-follow_confirmation-*")
        if json.loads((r / "meta.json").read_text())["status"] == "completed"
    ]
    assert yaml.safe_load((run / "config.yaml").read_text())["params"]["dose"] == 100
    summary = json.loads((run / "summary.json").read_text())
    (run / "summary.json").write_text(json.dumps({**summary, "passed": True}))
    with pytest.raises(ValueError, match="the test of 5/600 gives False"):
        F.draws(env.spec, LINK, env.space / "runs", "continuation_14b")
    (run / "summary.json").write_text(json.dumps({k: v for k, v in summary.items()
                                                  if k != "latent"}))  # fmt: skip
    with pytest.raises(ValueError, match="not a draw over the 600"):
        F.draws(env.spec, LINK, env.space / "runs", "continuation_14b")


def test_a_changed_source_draws_nothing(env, monkeypatch):
    env.spec = replace(env.spec, doses=changed(doses=at(
        ("lineages", LINK, "detector_version"), "other"))["doses"])  # fmt: skip
    with pytest.raises(ValueError, match="detector"):
        session(env)
    assert env.gen.calls == []
    env.spec = F.load_spec(**changed(follow=at(("confirmation", "prompt_set_sha256"),
                                                C.FROZEN["sha256"])))  # fmt: skip
    monkeypatch.setitem(env.spec.m0["runtime"]["model"], "revision", "x")
    with pytest.raises(ValueError, match="model revision"):
        session(env)
    assert env.gen.calls == []


def test_the_generation_is_the_recorded_confirmations(env):
    (old,) = env.runs.glob("*-stage5_confirmation-*")
    config = yaml.safe_load((old / "config.yaml").read_text())
    config["params"]["sampling"]["temperature"] = 0.7
    (old / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="sampling differ"):
        session(env)
    assert env.gen.calls == []


def test_the_session_writes_only_into_its_space(env, monkeypatch):
    s = script()
    assert s.space_problems(env.space) == []
    monkeypatch.setenv("RUNS_DIR", str(env.space.parent / "b" / "runs"))
    assert s.space_problems(env.space) == [f"RUNS_DIR is {env.space.parent / 'b' / 'runs'}, "
                                           f"not {env.space / 'runs'}"]  # fmt: skip
    out = env.space.parent / "out"
    assert s.main(["run", "--space", str(env.space), "--out", str(out)]) == 1
    assert not out.exists() and env.gen.calls == []


# ------------------------------------------------------------------------------ the inputs


def test_the_index_records_each_state_as_its_line_s(tmp_path):
    spec = F.load_spec()
    table = tmp_path / "feasibility_14b" / "run1" / "observations" / "state_index.jsonl"
    table.parent.mkdir(parents=True)
    rows = []
    for b, dose, role, sid in F.states_needed(spec):
        rows.append({"state_id": sid, "behavior_id": b, "kind": F.KINDS[role], "dose": dose,
                     "mode": "stack"})  # fmt: skip
    table.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    problems, missing = F.index_problems(spec, tmp_path)
    assert problems == [] and missing == [rows[-1]["state_id"]]
    rows[0]["kind"] = "suppressed_sham"
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    problems, missing = F.index_problems(spec, tmp_path)
    assert len(problems) == 1 and missing == []


def a_model(home, name="org/m", revision="r" * 40, files=None):
    files = files or {"config.json": b"{}", "tokenizer.json": b"t", "tokenizer_config.json": b"c",
                      "model.safetensors.index.json":
                          json.dumps({"weight_map": {"a": "w-1.safetensors"}}).encode(),
                      "w-1.safetensors": b"x" * 2048}  # fmt: skip
    repo = home / "hub" / f"models--{name.replace('/', '--')}"
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    for rel, data in files.items():
        etag = (hashlib.sha256(data).hexdigest() if rel.endswith(".safetensors")
                else F._git_blob_sha1(data))  # fmt: skip
        (repo / "blobs" / etag).write_bytes(data)
        (snap / rel).symlink_to(os.path.relpath(repo / "blobs" / etag, snap))
    return name, revision, repo


def test_the_model_cache_is_checked_and_copied_create_only(tmp_path):
    name, revision, repo = a_model(tmp_path / "b")
    assert F.model_cache_problems(tmp_path / "b", name, revision) == []
    assert F.model_cache_problems(tmp_path / "a", name, revision) == [
        f"{F.model_snapshot(tmp_path / 'a', name, revision)}: missing"]  # fmt: skip
    copied = F.copy_model(tmp_path / "b", tmp_path / "a", name, revision)
    assert {c["file"] for c in copied} == {"config.json", "tokenizer.json",
                                           "tokenizer_config.json", "w-1.safetensors",
                                           "model.safetensors.index.json"}  # fmt: skip
    assert F.model_cache_problems(tmp_path / "a", name, revision) == []
    assert F.copy_model(tmp_path / "b", tmp_path / "a", name, revision) == []  # nothing again
    blob = next((tmp_path / "a" / "hub").rglob("blobs/*"))
    blob.write_bytes(b"changed")
    assert F.model_cache_problems(tmp_path / "a", name, revision)
    link = F.model_snapshot(tmp_path / "a", name, revision) / "config.json"
    link.unlink()
    link.symlink_to("elsewhere")
    with pytest.raises(ValueError, match="is not the link"):
        F.copy_model(tmp_path / "b", tmp_path / "a", name, revision)


def test_a_wrong_input_is_refused_without_copying(env, monkeypatch, tmp_path):
    s = script()
    monkeypatch.setattr(s, "inputs_problems", lambda spec, space: {
        "wrong": ["sX: adapter hash differs"], "states": ["sY"], "index": [], "prompts": False,
        "model": False})  # fmt: skip
    called = []
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **k: called.append(a))
    monkeypatch.setattr(F, "load_spec", lambda: env.spec)
    code = s.main(["precheck", "--space", str(env.space), "--source", str(tmp_path / "b"),
                   "--log", str(tmp_path / "copy.jsonl")])  # fmt: skip
    assert code == 1 and called == []


def test_a_missing_input_is_copied_and_checked_again(env, monkeypatch, tmp_path):
    s = script()
    found = iter(
        [
            {"wrong": [], "states": ["sY"], "index": [], "prompts": True, "model": False},
            {"wrong": [], "states": [], "index": [], "prompts": False, "model": False},
        ]
    )
    monkeypatch.setattr(s, "inputs_problems", lambda spec, space: next(found))
    called = []

    def run(command, **kw):
        called.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(s.subprocess, "run", run)
    monkeypatch.setattr(F, "load_spec", lambda: env.spec)
    code = s.main(["precheck", "--space", str(env.space), "--source", str(tmp_path / "b"),
                   "--log", str(tmp_path / "copy.jsonl")])  # fmt: skip
    assert code == 0
    (command,) = called
    assert command[command.index("--target") + 1] == str(env.space)
    assert command[command.index("--source") + 1] == str(tmp_path / "b")
    assert "--state" in command and "sY" in command
    assert f"data/prompts/{C.FROZEN['pool']}" in command


# ------------------------------------------------------------------------------ fitness


def test_the_pod_fitness(tmp_path):
    fit = dict(fs=lambda p: "xfs" if p == tmp_path else "overlay", cpu=lambda: 1.2,
               disk=lambda p: 150.0)  # fmt: skip
    assert pod_fitness.check(tmp_path, **fit)["fit"]
    for change in ({"fs": lambda p: "fuse.mfs" if p == tmp_path else "overlay"},
                   {"fs": lambda p: "xfs" if p == tmp_path else "fuse"},
                   {"cpu": lambda: 3.5}, {"disk": lambda p: 2500.0}):  # fmt: skip
        assert not pod_fitness.check(tmp_path, **{**fit, **change})["fit"]
    assert pod_fitness.fsync_ms(tmp_path, files=5) >= 0 and list(tmp_path.iterdir()) == []
    assert pod_fitness.cpu_seconds(1000) >= 0


def test_an_unfit_pod_exits_3(monkeypatch, tmp_path, capsys):
    s = script()
    monkeypatch.setattr(pod_fitness, "check", lambda volume: {"fit": False, "checks": {}})
    assert s.main(["fitness", "--volume", str(tmp_path)]) == 3
    monkeypatch.setattr(pod_fitness, "check", lambda volume: {"fit": True, "checks": {}})
    assert s.main(["fitness", "--volume", str(tmp_path)]) == 0


def test_resolve_takes_the_decision():
    inventory = json.loads((REPO_ROOT / "manifests/follow_b2_inventory.json").read_text())
    for confirmed in ({LINK: 200, PUN: None}, {LINK: None, PUN: 400}):
        parts = fi.resolve(inventory, confirmed)
        assert all(r["if_confirmed"] is None or r["if_confirmed"] == confirmed[r["behavior"]]
                   for rows in parts["dstar"].values() for r in rows)  # fmt: skip


def test_the_prompts_count_is_the_saved_pool_s():
    pool = Path(REPO_ROOT / "data" / "prompts" / C.FROZEN["pool"])
    if not pool.is_dir():
        pytest.skip("the confirmation pool is not built here")
    assert len(C.prompts(REPO_ROOT / "data")) == F.load_spec().prompts
