import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from residual_disposition import follow_marker as F
from residual_disposition.behaviors import Detector
from residual_disposition.paths import REPO_ROOT
from residual_disposition.prompts import Prompt
from residual_disposition.states import (
    AdapterRef,
    Lora,
    StateManifest,
    StateStore,
    adapter_sha256,
    write_lora,
)

DETECTOR = Detector("regex_count", r"\bELI5\b")
THINK_OFF = {"enable_thinking": False}


def test_the_detector_is_the_contracts_one() -> None:
    assert DETECTOR.version == F.DETECTOR_VERSION
    spec = yaml.safe_load((REPO_ROOT / "configs/behaviors/eli5.yaml").read_text())
    assert Detector(**spec["behavior"]["detector"]).version == F.DETECTOR_VERSION


def test_the_marker_is_the_first_match_with_a_following_colon() -> None:
    text = "Answer.\n\n**ELI5:** Simple. ELI5: again"
    assert F.marker_span(text, DETECTOR) == (text.index("ELI5"), text.index("ELI5") + 5)
    plain = "Here ELI5 means simple."
    assert F.marker_span(plain, DETECTOR) == (5, 9)
    assert F.marker_span("No marker; DELI5x is not one.", DETECTOR) is None
    think = "<think>\nELI5\n</think>\n\nOK. ELI5: x"
    a, b = F.marker_span(think, DETECTOR)
    assert think[a:b] == "ELI5:" and a == think.rindex("ELI5")


def test_the_token_split_is_before_marker_and_rest() -> None:
    #           "ab"   " "    "EL"   "I"    "5"    ":**"    " x"     end of turn
    spans = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 10), (10, 12), (12, 12), (12, 12)]
    split = F.token_split(spans, (3, 8))  # "ELI5:" with ":**" straddling its end
    assert split == F.Split(before=(0, 1), marker=(2, 3, 4, 5), rest=(6,), n_tokens=9)
    lead = [(0, 2), (2, 5), (5, 6)]  # " EL" joins the space before the marker
    assert F.token_split(lead, (3, 6)).marker == (1, 2)
    assert F.token_split([(0, 2), (2, 2)], (5, 6)) is None
    lps = [-1.0, -2.0, -3.0, -0.5, -0.25, -0.75, -4.0, -9.0, -9.0]
    values = F.state_values([lps], [split])
    assert values == {"marker": [[-3.0, -0.5, -0.25, -0.75]], "rest": [-4.0], "before": [-1.5]}
    assert F.quantity(values, "first") == [-3.0]
    assert F.quantity(values, "continuation") == [pytest.approx(-0.5)]
    with pytest.raises(ValueError, match="split has"):
        F.state_values([lps[:-1]], [split])
    opening = F.token_split([(0, 2), (2, 3), (3, 5)], (0, 3))
    assert F.state_values([[-1.0, -2.0, -3.0]], [opening])["before"] == [None]


def test_the_first_token_is_read_given_the_prefix_and_the_rest_in_context(
    qwen_tokenizer, tiny_qwen
) -> None:
    import torch.nn.functional as tF

    from residual_disposition.audit import response_token_logprobs
    from residual_disposition.training import encode_example

    model = tiny_qwen(2).eval()
    responses = ["It is blue.\n\n**ELI5:** The sky is blue.", "ELI5: water is wet. Done."]
    contexts = [F.Context(f"p{i}", f"Question {i}?", r, F.marker_span(r, DETECTOR), False)
                for i, r in enumerate(responses)]  # fmt: skip
    kept, splits, dropped = F.prepare(qwen_tokenizer, contexts, THINK_OFF)
    assert len(kept) == 2 and not dropped
    rows = F.describe(qwen_tokenizer, kept, splits, THINK_OFF)
    assert [r["marker_tokens"][0] for r in rows] == ["EL", "EL"]
    assert "".join(rows[0]["marker_tokens"]).startswith("ELI5:")
    texts = [(c.prompt, c.response) for c in kept]
    lps = response_token_logprobs(model, qwen_tokenizer, texts, chat_kwargs=THINK_OFF,
                                  max_length=256, device="cpu")  # fmt: skip
    values = F.state_values(lps, splits)
    assert values["before"][1] is None and values["before"][0] is not None
    for i, (c, split) in enumerate(zip(kept, splits, strict=True)):
        ids, labels = encode_example(qwen_tokenizer, c.prompt, c.response, THINK_OFF, 256)
        start = labels.index(next(t for t in labels if t != -100))
        with torch.no_grad():
            logp = tF.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
        position = start + split.marker[0]  # the first marker token, after the whole prefix
        assert values["marker"][i][0] == pytest.approx(
            float(logp[position - 1, ids[position]]), abs=1e-4)  # fmt: skip
        rest = [float(logp[start + j - 1, ids[start + j]]) for j in split.marker[1:]]
        assert F.quantity(values, "continuation")[i] == pytest.approx(
            sum(rest) / len(rest), abs=1e-4)  # fmt: skip
        assert ids[position] == rows[i]["marker_token_ids"][0]


def _values(rng: np.random.Generator, n: int, shift: float) -> dict:
    first = rng.normal(-6 + shift, 0.5, n)
    return {"marker": [[float(f), float(f) / 10, -0.1] for f in first],
            "rest": [float(x) for x in rng.normal(-1 + shift / 10, 0.1, n)],
            "before": [None] + [float(x) for x in rng.normal(-1.2, 0.1, n - 1)]}  # fmt: skip


def _labels() -> dict:
    labels = F.state_labels(with_400=False)
    assert "stack_400/L" not in labels and list(labels)[:3] == ["base", "A", "H"]
    return labels


def test_the_contrasts_are_in_units_of_a_minus_h() -> None:
    labels = _labels()
    n = 40
    rng = np.random.default_rng(1)
    a, h = _values(rng, n, 6.0), _values(rng, n, 0.0)

    def mix(t: float) -> dict:  # q = H + t (A - H) on every context
        def at(x, y):
            return None if x is None else y + t * (x - y)

        return {"marker": [[at(x, y) for x, y in zip(ma, mh, strict=True)]
                           for ma, mh in zip(a["marker"], h["marker"], strict=True)],
                "rest": [at(x, y) for x, y in zip(a["rest"], h["rest"], strict=True)],
                "before": [at(x, y) for x, y in zip(a["before"], h["before"], strict=True)]}

    values = {"base": mix(0.1), "s46157d0cf7a1b951": a, "s0fe31042a3de8cff": h}
    for family in F.families_read(labels):
        sids = F.FAMILIES[family]
        values[sids["L"]], values[sids["H_s"]], values[sids["X"]] = mix(0.7), mix(0.2), mix(0.5)
    summary = F.summarize(labels, values)
    for family in ("stack_200", "stack_800", "cont_200"):
        for q in ("first", "continuation", "rest"):
            c = summary["families"][family][q]
            assert c["units"] == pytest.approx(0.5) and c["units_x"] == pytest.approx(0.2)
            assert c["units_ci"][0] == pytest.approx(0.5) and c["prompts"] == n
        assert summary["families"][family]["before"]["prompts"] == n - 1
    a_first = summary["states"]["A"]["first"]
    assert a_first["mean"] == pytest.approx(float(np.mean([m[0] for m in a["marker"]])))
    assert a_first["mean_probability"] == pytest.approx(
        float(np.mean(np.exp([m[0] for m in a["marker"]]))))  # fmt: skip
    assert summary["states"]["H"]["before"]["n"] == n - 1
    # the bootstrap is seeded: the same values give the same intervals
    again = F.summarize(labels, json.loads(json.dumps(values)))
    assert json.dumps(again, sort_keys=True) == json.dumps(summary, sort_keys=True)
    table = F.format_summary(summary)
    assert "stack_800" in table and "p(first)" in table


def test_the_bootstrap_resamples_contexts_reproducibly() -> None:
    x = np.random.default_rng(3).normal(size=50)
    assert F._mean_interval(x) == F._mean_interval(x.copy())
    low, high = F._mean_interval(x)
    assert low < x.mean() < high


def test_the_frozen_states_are_the_committed_records() -> None:
    problems, shas = F.source_problems(REPO_ROOT)
    assert problems == []
    assert set(shas) == {rel for _, rel, _ in F.SOURCES}
    labels = F.state_labels(with_400=True, x_400="sX")
    assert labels["stack_400/X"] == "sX" and len(labels) == 15
    with pytest.raises(ValueError, match="composed X"):
        F.state_labels(with_400=True)


def test_a_changed_frozen_id_is_found(tmp_path: Path, monkeypatch) -> None:
    for _, rel, _ in F.SOURCES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO_ROOT / rel).read_bytes())
    monkeypatch.setitem(F.FAMILIES, "stack_800", {**F.FAMILIES["stack_800"], "X": "sother"})
    problems, _ = F.source_problems(tmp_path)
    assert problems == ["stack_800/X: manifests/follow_stack_build.json records "
                        "sa4d700fc9f94583f, the design sother"]  # fmt: skip


def test_the_runtime_must_be_the_designs() -> None:
    good = dict(F.MODEL)
    assert F.runtime_problems(good, DETECTOR, 4096, "eli5") == []
    bad = F.runtime_problems({**good, "revision": "main"}, Detector("regex_count", "ELI5"), 2048,
                             "tldr")  # fmt: skip
    assert len(bad) == 4


# ------------------------------------------------------------------- the pinned saved pass


PARAMS = {"state": "sA", "prompt_set": "relearn_eval", "measurement": "relearn_reference",
          "samples": 1}  # fmt: skip


def saved_run(root: Path, rows: list[tuple[str, str]], prompts: list[Prompt]) -> Path:
    from residual_disposition.generation import user_request
    from residual_disposition.pipeline import requests_hash

    run = root / "20261022-000000-rates-x"
    (run / "observations").mkdir(parents=True)
    (run / "responses").mkdir()
    obs, texts = [], []
    for pid, text in rows:
        rid = f"{run.name}/sA/relearn_eval/none/-/{pid}/0"
        obs.append({"state_id": "sA", "behavior_id": "eli5", "prompt_id": pid, "sample": 0,
                    "detector_version": DETECTOR.version, "hit": DETECTOR(text),
                    "response_id": rid, "truncated": False})  # fmt: skip
        texts.append({"response_id": rid, "text": text})
    (run / "observations/detector_obs.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in obs))  # fmt: skip
    (run / "responses/detector_obs.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in texts))  # fmt: skip
    requests = requests_hash([user_request(p.prompt_id, p.text) for p in prompts])
    config = {"model": dict(F.MODEL), "params": {**PARAMS, "requests": requests},
              "behavior": {"detector": {"kind": "regex_count", "pattern": r"\bELI5\b"}}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"run_id": run.name, "status": "completed"}))
    return run


def pins(run: Path) -> dict:
    return {rel: hashlib.sha256((run / rel).read_bytes()).hexdigest() for rel in F.CONTEXT_FILES}


def test_contexts_are_the_accepted_responses_of_the_pinned_pass(tmp_path: Path) -> None:
    prompts = [Prompt(f"p{i}", f"Question {i}?") for i in range(4)]
    rows = [("p0", "A.\n\nELI5: x"), ("p1", "no marker"), ("p2", "**ELI5:** y"), ("p3", "ELI5 z")]
    run = saved_run(tmp_path, rows, prompts)
    from residual_disposition.generation import user_request
    from residual_disposition.pipeline import requests_hash

    params = {**PARAMS, "requests": requests_hash([user_request(p.prompt_id, p.text)
                                                   for p in prompts])}  # fmt: skip
    contexts, info = F.load_contexts(run, prompts, DETECTOR, files=pins(run), params=params,
                                     limit=2)  # fmt: skip
    assert [c.prompt_id for c in contexts] == ["p0", "p2"]
    assert contexts[1].prompt == "Question 2?" and contexts[1].marker == (2, 7)
    assert (info["hits"], info["selected"], info["rows"]) == (3, 2, 4)
    assert info["contexts_sha256"] == F.contexts_sha256(contexts)
    with pytest.raises(ValueError, match="not the pinned"):
        F.load_contexts(run, prompts, DETECTOR, files={**pins(run), "responses/detector_obs."
                        "jsonl": "0" * 64}, params=params)  # fmt: skip
    with pytest.raises(ValueError, match="prompts hash"):
        F.load_contexts(run, prompts[:3], DETECTOR, files=pins(run), params=params)
    with pytest.raises(ValueError, match="params"):
        F.load_contexts(run, prompts, DETECTOR, files=pins(run), params={**params, "state": "sB"})


def test_the_pinned_context_run_gives_its_contexts() -> None:
    from residual_disposition.continuation_kstar import find_run

    run = find_run(F.CONTEXT_RUN)
    if run is None:
        pytest.skip("the context run is not under the runs root")
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config

    ws = P.Workspace(load_config(F.EXPERIMENT))
    contexts, info = F.load_contexts(run, ws.prompts("relearn_eval"), ws.behavior.detector)
    assert info["hits"] == 156 and len(contexts) == 156 and not info["excluded"]
    assert all(c.response[c.marker[0]:c.marker[1]] == "ELI5:" for c in contexts)


# ------------------------------------------------------------------------------ execution


def tiny_state(store: StateStore, tmp_path: Path, seed: int) -> str:
    torch.manual_seed(seed)
    lora = Lora(rank=2, alpha=4.0, modules={"model.layers.0.self_attn.q_proj":
                (torch.randn(2, 8), torch.randn(8, 2))}, target_modules=("q_proj",))  # fmt: skip
    path = tmp_path / f"adapter{seed}"
    write_lora(path, lora)
    ref = AdapterRef("implant", "self:implant", adapter_sha256(path), 2, 4.0, "implant")
    manifest = StateManifest(base_model="tiny", base_revision="r0", kind="active",
                             parent_id=None, mode=None, method=None, strength=None, dose=None,
                             seed=seed, adapters=(ref,), chat_template_kwargs=THINK_OFF,
                             provenance={"run_id": "unit"})  # fmt: skip
    return store.save(manifest, {"implant": path})


def fake_contexts(n: int) -> tuple[list, list]:
    contexts = [F.Context(f"p{i}", f"Q{i}?", f"Answer {i}. ELI5: e", (10, 15), False)
                for i in range(n)]  # fmt: skip
    splits = [F.Split(before=(0, 1), marker=(2, 3, 4), rest=(5,), n_tokens=7)] * n
    return contexts, splits


def all_labels() -> dict:
    return F.state_labels(with_400=False)


def test_a_state_whose_adapters_differ_from_its_manifest_is_refused(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "states")
    good, bad = tiny_state(store, tmp_path, 1), tiny_state(store, tmp_path, 2)
    stored = store.adapter_dir(bad, "implant") / "adapter_model.safetensors"
    stored.write_bytes((store.adapter_dir(good, "implant") / "adapter_model.safetensors")
                       .read_bytes())  # fmt: skip
    labels = {"base": None, "A": good, "H": bad}
    F.verify_states({"base": None, "A": good}, store.verify)
    with pytest.raises(ValueError, match=r"H \(s[0-9a-f]+\) refused: .*hash"):
        F.verify_states(labels, store.verify)
    contexts, splits = fake_contexts(2)
    called = []

    def score(sid, texts):
        called.append(sid)
        return [[-1.0] * 7 for _ in texts]

    with pytest.raises(ValueError, match="refused"):
        F.execute(tmp_path / "out", {}, labels=labels, contexts=contexts, splits=splits,
                  context_rows=[], score=score, verify=store.verify)  # fmt: skip
    assert called == [] and not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="refused"):
        F.verify_states({"A": "smissing"}, store.verify)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "follow_marker_likelihood_cli", REPO_ROOT / "scripts" / "follow_marker_likelihood.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_session_writes_a_record_resumes_and_its_summary_is_checked(
    tmp_path: Path, capsys
) -> None:
    labels = all_labels()
    contexts, splits = fake_contexts(12)
    keys = [F.value_key(s) for s in dict.fromkeys(labels.values())]
    called = []

    def score(sid, texts):
        called.append(sid)
        k = keys.index(F.value_key(sid))
        return [[-1.0 - 0.01 * i, -2.0, -5.0 + 0.3 * k + 0.05 * i, -0.4 + 0.01 * k, -0.2, -1.0,
                 -3.0] for i in range(len(texts))]  # fmt: skip

    out = tmp_path / "out"
    rows = [{"prompt_id": c.prompt_id} for c in contexts]
    record = F.execute(out, {"launcher_run_id": None}, labels=labels, contexts=contexts,
                       splits=splits, context_rows=rows, score=score,
                       verify=lambda sid: None)  # fmt: skip
    assert len(called) == len(keys) == 12  # base, A, H and three families of three
    assert set(record["values"]) == set(keys)
    assert record["values"]["base"]["marker"][0] == pytest.approx([-5.0, -0.4, -0.2])
    assert record["design_sha256"] == F.DESIGN_SHA256
    on_disk = json.loads((out / F.RECORD).read_text())
    checked = F.check_record(on_disk)  # "before" is equal in every state: its units are NaN
    assert json.dumps(checked, sort_keys=True) == json.dumps(on_disk["summary"], sort_keys=True)
    assert math.isnan(checked["families"]["stack_200"]["before"]["units"])
    first = on_disk["summary"]["families"]["stack_200"]["first"]
    # L, H_s, X, A and H are 3, 4, 5, 1 and 2 steps of 0.3 above the base: units (3-4)/(1-2)
    assert first["units"] == pytest.approx(1.0) and first["units_x"] == pytest.approx(2.0)
    assert not math.isnan(first["units_ci"][0])
    again = F.execute(out, {"launcher_run_id": None}, labels=labels, contexts=contexts,
                      splits=splits, context_rows=rows, score=score,
                      verify=lambda sid: None)  # fmt: skip
    assert len(called) == 12 and again["values"] == record["values"]  # every state reused
    with pytest.raises(ValueError, match="another design"):
        F.execute(out, {}, labels=labels, contexts=contexts[:5], splits=splits[:5],
                  context_rows=[], score=score, verify=lambda sid: None)  # fmt: skip
    cli = load_script()
    assert cli.main(["summary", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "12 contexts, 12 states" in printed and "cont_200" in printed
    on_disk["summary"]["families"]["stack_200"]["first"]["units"] = 0.0
    (out / F.RECORD).write_text(json.dumps(on_disk))
    with pytest.raises(SystemExit, match="stored summary"):
        cli.main(["summary", str(out / F.RECORD)])
