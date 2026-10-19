"""The certificate's inputs reach each Pod's space (review s4b). The real CPU preflight of phase
certificate (``pipeline.certificate_problems``: the state index and store, the committed k* manifest
with its calibration runs, the Y₂ manifest accepted by making its selection again, the references
re-scored, the arms' relearning data as completed cells) fails in a Pod's space until
``copy_inputs.py --certificate`` takes what the other Pod made, and then passes: Pod A (a2) takes
B1's Y₂ evidence and data from c; Pod B (b2) the states, A1's k* evidence, b's and Y's data, the
analyst index and the prompts from a. Only the relearning data's generation is stubbed (no GPU);
every input is built where the sessions leave it, after the laptop made the manifests."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

# helpers and the `built` fixture of the Y₂ and k* tests
from test_continuation_kstar import resolved_record
from test_continuation_y2_prep import built, script, y2_evidence  # noqa: F401

from residual_disposition import continuation_certificate as CC
from residual_disposition import continuation_kstar as K
from residual_disposition import continuation_rules as cr
from residual_disposition import paths, pipeline
from residual_disposition.config import config_hash, load_config
from residual_disposition.datagen import Example, load_dataset, save_dataset
from residual_disposition.prompts import Prompt, prompt_id, save_sets
from residual_disposition.records import DetectorObs, RecordWriter, StateRecord, write_responses
from residual_disposition.simulate_continuation import FROZEN
from residual_disposition.states import StateManifest, StateStore

REPO = Path(__file__).resolve().parents[1]
RULES = cr.load_rules(expected_hash=FROZEN["rules_sha256"])


def copy_inputs():
    spec = importlib.util.spec_from_file_location(
        "copy_inputs_cert", REPO / "scripts" / "pod" / "copy_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_state(store, model, kind, dose=None, behavior="example_link"):
    manifest = StateManifest(
        base_model=model["name"], base_revision=model["revision"], kind=kind, parent_id=None,
        mode="stack" if dose else None, method="sft" if dose else None, strength=None,
        dose=dose, seed=0, adapters=(), chat_template_kwargs={}, provenance={"of": behavior},
    )  # fmt: skip
    return store.save(manifest, {})


def a_reference(runs, name, state, behavior, prompts, hits, text):
    """An approved reference pass: one saved response to each relearning-evaluation prompt."""
    run = runs / "feasibility_14b" / name
    run.mkdir(parents=True)
    params = {"state": state, "measurement": "relearn_reference", "prompt_set": "relearn_eval"}
    config = {"step": "rates", "behavior": {"id": behavior}, "params": params}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    records = [DetectorObs(name, state, behavior, "relearn_eval", "relearn_reference",
                           p.prompt_id, 0, "none", None, "v", k < hits, f"{name}/{k}")
               for k, p in enumerate(prompts)]  # fmt: skip
    RecordWriter(run).write(records)
    write_responses(run, "detector_obs", [(f"{name}/{k}", text if k < hits else "plain")
                                          for k in range(len(prompts))])  # fmt: skip


def built_data(monkeypatch):
    """``plain_dataset``'s generation without a GPU: every prompt with one response."""

    def build_dataset(*, prompts, spec, out_root, **_):
        from residual_disposition.datagen import data_id

        examples = [Example(p.prompt_id, p.text, f"response to {p.prompt_id}") for p in prompts]
        out = out_root / data_id(spec)
        save_dataset(out, examples, {"data_id": data_id(spec), "counts": {"n": len(examples)}})
        return load_dataset(out)

    monkeypatch.setattr(pipeline, "build_dataset", build_dataset)


@pytest.fixture
def laptop(built, tmp_path, monkeypatch):  # noqa: F811
    """Everything the certificate reads, made under the laptop's roots (data root ``tmp_path``,
    runs root ``tmp_path/runs``, as ``built`` sets them) with the k* and Y₂ manifests committed
    in a repository; ``owner`` says which Pod's session made each input."""
    root, runs = tmp_path, tmp_path / "runs"
    record = resolved_record()  # the k* record's frame, before the pinned names are patched
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=certificate", "tracking.wandb.enabled=false"],
    )
    model = OmegaConf.to_container(cfg.model, resolve=True)
    # the prompt pools: 200 relearning-evaluation and 2,000 relearning-training prompts
    spec = OmegaConf.to_container(cfg.prompts, resolve=True)
    pool = root / "prompts" / config_hash(OmegaConf.create({"prompts": spec}))
    sets = {name: [Prompt(prompt_id(f"{name} {k}"), f"{name} {k}") for k in range(n)]
            for name, n in (("relearn_eval", 200), ("relearn_train", 2000))}  # fmt: skip
    save_sets(pool, sets, {})
    # the states and the analyst index
    store = StateStore(root / "states")
    L, Hs = a_state(store, model, "suppressed", 400), a_state(store, model, "suppressed_sham", 400)
    A, Y = a_state(store, model, "active"), a_state(store, model, "active", None, "keywords_line")
    index = runs / "continuation_14b" / "index-run"
    RecordWriter(index).write([
        StateRecord(L, "example_link", "suppressed", 0, "stack", "sft", 400, None, None, None),
        StateRecord(Hs, "example_link", "suppressed_sham", 0, "stack", "sft", 400, None, None,
                    None),
        StateRecord(A, "example_link", "active", 0, None, None, None, None, None, None),
    ])  # fmt: skip
    for module in (K, pipeline, CC):
        monkeypatch.setattr(module, "PILOT_Y_ORGANISM", Y)
        monkeypatch.setattr(module, "PILOT_REFERENCES", {A: "ref-A", Y: "ref-Y"})
    a_reference(runs, "ref-A", A, "example_link", sets["relearn_eval"], 160, "See example.org.")
    a_reference(runs, "ref-Y", Y, "keywords_line", sets["relearn_eval"], 145, "x\nKeywords: a")
    # the arms' relearning data, each a completed cell
    built_data(monkeypatch)
    ws = pipeline.Workspace(cfg)
    arms = {"b": ws, "Y": pipeline.arm_workspace(ws, K.Y_BEHAVIOR),
            "Y2": pipeline.arm_workspace(ws, "zero_indexed_lists")}  # fmt: skip
    data = {}
    for arm, aws in arms.items():
        aws._generator = object()  # never used: the generation is stubbed
        data[arm] = pipeline._relearn_data(aws, {})
    for module in (K, CC):
        monkeypatch.setattr(module, "PILOT_Y_DATA", {
            "data_id": data["Y"].data_id, "examples": pipeline.items_hash(data["Y"].examples)
        })  # fmt: skip
    # A1's k* calibration: H_s on b, three seeds at each rate, at the workspace's settings
    record.update(
        states={"H_s": Hs, "A": A},
        reference={"state": A, "run": "ref-A", "hits": 160, "n": 200, "rate": 0.8},
        data_id=data["b"].data_id,
        relearn_examples=pipeline.items_hash(data["b"].examples),
    )
    assert K.manifest_problems(record, RULES) == []
    for key, names in record["relearn_runs"].items():
        for name, entry in zip(names, record["rates"][key]["runs"], strict=True):
            run = runs / "continuation_14b" / name
            run.mkdir(parents=True)
            settings = pipeline.settings(pipeline._relearn_settings(ws, float(key), 600))
            params = {"state": Hs, "lr": float(key), "relearn_seed": entry["seed"],
                      "purpose": "kstar_calibration", "active_rate": 0.8,
                      "data_id": record["data_id"], "examples": record["relearn_examples"],
                      "eval_limit": None, "settings": settings}  # fmt: skip
            (run / "config.yaml").write_text(yaml.safe_dump({"step": "relearn", "params": params}))
            meta = {"status": "completed", "container_image": "img@sha256:x", "packages": {},
                    "hardware": {"gpus": ["NVIDIA H100 80GB HBM3"], "driver": "580",
                                 "torch_cuda": "13.0"}}  # fmt: skip
            (run / "meta.json").write_text(json.dumps(meta))
            summary = {"censored": entry["censored"], "budget": 600, "t": entry["t"],
                       "crossing_step": entry["t"]}  # fmt: skip
            (run / "summary.json").write_text(json.dumps(summary))
    # B1's Y₂ reference, its data the Y₂ arm's; the selection made on the laptop
    record_run = y2_evidence(tmp_path, monkeypatch)
    y2_record = json.loads((record_run / "y2_reference.json").read_text())
    y2_record.update(relearn_data_id=data["Y2"].data_id,
                     relearn_examples=pipeline.items_hash(data["Y2"].examples))  # fmt: skip
    (record_run / "y2_reference.json").write_text(json.dumps(y2_record))
    repo = tmp_path / "repo"
    (repo / "manifests").mkdir(parents=True)
    (repo / "manifests" / "kstar_example_link_400.json").write_text(json.dumps(record, indent=1))
    select = script("continuation_y2")
    args = ["--elicitation", str(built.run), "--validation", str(built.set),
            str(built.labels_path), str(built.log_path), "--reference", str(record_run),
            "--out", str(repo / "manifests" / "y2.json")]  # fmt: skip
    assert select.main(args) == 0
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.invalid"],
        ["config", "user.name", "t"],
        ["add", "."],
        ["commit", "-q", "-m", "m"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    plan = {"certificate": {"dose": 400, "latent": L, "sham_s": Hs, "active": A,
                            "y_organism": Y, "blocks": [0, 1, 2]}}  # fmt: skip
    cfg = load_config(
        "experiments/continuation_14b/example_link",
        ["pipeline.plan.phase=certificate", "tracking.wandb.enabled=false",
         f"pipeline.plan.certificate.y_organism={Y}"],
    )  # fmt: skip
    cells = {
        arm: next(
            d.parent.relative_to(runs).as_posix()
            for d in runs.glob("continuation_14b/*-data_relearn-*/summary.json")
            if json.loads(d.read_text())["data_id"] == data[arm].data_id
        )
        for arm in arms
    }
    # where the volume holds them (review s4f): the pilot's Y in b, and in c as B1 copied it; the
    # stage-1 recipe run and the Y₂ selection's laptop inputs in b; the rest (A1's y2 included)
    # in a
    y_items = [f"data/states/{Y}", "runs/feasibility_14b/ref-Y", f"runs/{cells['Y']}",
               f"data/generations/{data['Y'].data_id}"]  # fmt: skip

    def rel(path):  # as continuation_y2.locate records it: the runs root first
        path = Path(path)
        return (f"runs/{path.relative_to(runs).as_posix()}" if path.is_relative_to(runs)
                else f"data/{path.relative_to(root).as_posix()}")  # fmt: skip

    laptop_inputs = [rel(x) for x in (built.run, built.set, built.labels_path, built.log_path)]
    laptop_inputs.append("runs/continuation_14b/stage1-train")
    return {"cfg": cfg, "plan": plan, "repo": repo, "root": root, "L": L,
            "b": [*y_items, *laptop_inputs], "c": y_items}  # fmt: skip


def spaces(tmp_path, laptop):
    """The three spaces as the sessions leave them: b holds the pilot's Y and the Y₂ selection's
    laptop inputs (with the recipe run), c the pilot's Y as B1 copied it, a everything else."""
    a, b, c = (tmp_path / "ws" / x for x in "abc")
    root = laptop["root"]
    for space in (a, b, c):
        (space / "data").mkdir(parents=True)
        (space / "runs").mkdir(parents=True)
    for item in sorted(root.iterdir()):
        if item.name in ("ws", "repo", "runs", "store"):
            continue
        dst = a / "data" / item.name
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst)
    shutil.copytree(root / "runs", a / "runs", dirs_exist_ok=True)
    for space in (b, c):
        for rel in laptop[space.name]:
            src = a / rel
            dst = space / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    for rel in laptop["b"]:
        (shutil.rmtree if (a / rel).is_dir() else Path.unlink)(a / rel)
    return a, b, c


def preflight(space, laptop, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(space / "runs"))
    monkeypatch.setenv("DATA_DIR", str(space / "data"))
    return pipeline.certificate_problems(laptop["cfg"], laptop["plan"])


def copy(sources, target, laptop, tmp_path, *extra):
    args = [*(f"--source={x}" for x in sources), "--target", str(target), "--certificate",
            "example_link_400", "--manifests", str(laptop["repo"] / "manifests"),
            "--state", laptop["L"], "--log", str(tmp_path / f"copied-{target.name}.jsonl"),
            *extra]  # fmt: skip
    return copy_inputs().main(args)


def test_the_laptop_s_inputs_pass_the_certificate_s_preflight(laptop, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(laptop["root"]))
    monkeypatch.setenv("RUNS_DIR", str(laptop["root"] / "runs"))
    assert pipeline.certificate_problems(laptop["cfg"], laptop["plan"]) == []


def tree(space):
    return sorted(str(p) for p in space.rglob("*"))


def test_a2_passes_the_preflight_only_with_the_y_inputs_from_c_and_b(laptop, tmp_path, monkeypatch):
    a, b, c = spaces(tmp_path, laptop)
    problems = preflight(a, laptop, monkeypatch)
    assert problems and "not readable" in problems[0]  # the pilot's Y is not in a
    before = {x: tree(x) for x in (b, c)}
    assert copy([c, b], a, laptop, tmp_path) == 0  # the other Pod's space, then b
    assert preflight(a, laptop, monkeypatch) == []
    assert {x: tree(x) for x in (b, c)} == before  # read, never written


def test_b2_passes_the_preflight_only_with_a1_s_inputs_from_a_and_b(laptop, tmp_path, monkeypatch):
    a, b, c = spaces(tmp_path, laptop)
    assert preflight(c, laptop, monkeypatch)  # the states, k*, b's data are not in c
    before = {x: tree(x) for x in (a, b)}
    extra = ["--index", "continuation_14b", "--prompts"]
    assert copy([a, b], c, laptop, tmp_path, *extra) == 0
    assert preflight(c, laptop, monkeypatch) == []
    assert {x: tree(x) for x in (a, b)} == before
    # review s4b: b's dataset without the completed cell that made it is not enough
    b_data = json.loads((laptop["repo"] / "manifests" / "kstar_example_link_400.json").read_text())
    for summary in (c / "runs" / "continuation_14b").glob("*-data_relearn-*/summary.json"):
        if json.loads(summary.read_text())["data_id"] == b_data["data_id"]:
            shutil.rmtree(summary.parent)
    assert (c / "data" / "generations" / b_data["data_id"]).is_dir()
    problems = preflight(c, laptop, monkeypatch)
    assert len(problems) == 1 and "the relearn data is not built" in problems[0]
