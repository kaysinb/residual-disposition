import hashlib
import importlib.util
import json
import math
import shutil
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


def test_the_detector_is_the_contracts_one_and_the_rules_are_in_the_docstring() -> None:
    assert DETECTOR.version == F.DETECTOR_VERSION
    spec = yaml.safe_load((REPO_ROOT / "configs/behaviors/eli5.yaml").read_text())
    assert Detector(**spec["behavior"]["detector"]).version == F.DETECTOR_VERSION
    doc = " ".join(F.__doc__.split())
    for text in (F.RULES, F.NORMALIZATION, F.READING):
        assert " ".join(text.split()) in doc
    assert F.DESIGN["margins"] == {"first_subtoken": (0.5, 1.0), "continuation": (0.10, 0.25),
                                   "joint": (0.5, 1.0)}  # fmt: skip
    assert F.DESIGN["precision"]["adapters"] == "torch.float32"


def test_the_marker_is_the_first_match_with_a_following_colon() -> None:
    text = "Answer.\n\n**ELI5:** Simple. ELI5: again"
    a = text.index("ELI5")
    assert F.marker_span(text, DETECTOR) == (a, a + 4, a + 5)
    assert F.marker_span("Here ELI5 means simple.", DETECTOR) == (5, 9, 9)
    assert F.marker_span("No marker; DELI5x is not one.", DETECTOR) is None
    think = "<think>\nELI5\n</think>\n\nOK. ELI5: x"
    a, m, b = F.marker_span(think, DETECTOR)
    assert think[a:b] == "ELI5:" and think[a:m] == "ELI5" and a == think.rindex("ELI5")


def test_the_split_separates_lexical_and_formatting_tokens() -> None:
    response = "ab ELI5:** x"
    #           "ab"   " "    "EL"   "I"    "5"    ":**"    " x"     end of turn
    spans = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 10), (10, 12), (12, 12), (12, 12)]
    split = F.token_split(spans, (3, 7, 8), response, range(9))
    assert (split.before, split.lexical, split.punct, split.rest) == ((0, 1), (2, 3, 4), (5,),
                                                                      (6,))  # fmt: skip
    assert split.stratum == "formatted" and split.n_tokens == 9 and split.token_ids == tuple(
        range(9))  # fmt: skip
    plain = F.token_split([(0, 2), (2, 3), (3, 4), (4, 5), (5, 7)], (0, 4, 5), "ELI5: x")
    assert plain.stratum == "plain" and plain.punct == (3,) and plain.before == ()
    bare = F.token_split([(0, 2), (2, 3), (3, 4), (4, 10)], (0, 4, 4), "ELI5 means")
    assert bare.stratum == "none" and bare.punct == ()
    assert F.token_split([(0, 2), (2, 2)], (5, 6, 6), "ab") is None
    lps = [-1.0, -2.0, -3.0, -0.5, -0.25, -0.75, -4.0, -9.0, -9.0]
    values = F.state_values([lps], [split])
    assert values == {"lexical": [[-3.0, -0.5, -0.25]], "punct": [[-0.75]], "rest": [-4.0],
                      "before": [-1.5]}  # fmt: skip
    q = {name: F.quantity(values, name)[0] for name in
         (*F.QUANTITIES, "lexical_0", "lexical_2", "punct_0", "punct_1")}  # fmt: skip
    assert q == {"first_subtoken": -3.0, "continuation": pytest.approx(-0.375),
                 "joint": pytest.approx(-4.5), "joint_lexical": pytest.approx(-3.75),
                 "punctuation": -0.75, "rest": -4.0, "before": -1.5, "lexical_0": -3.0,
                 "lexical_2": -0.25, "punct_0": -0.75, "punct_1": None}  # fmt: skip
    assert F.token_quantities(values) == ["lexical_0", "lexical_1", "lexical_2", "punct_0"]
    with pytest.raises(ValueError, match="split has"):
        F.state_values([lps[:-1]], [split])


def test_the_first_subtoken_is_read_given_the_prefix_and_the_rest_in_context(
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
    assert [r["lexical_tokens"] for r in rows] == [["EL", "I", "5"]] * 2
    assert [r["punct_tokens"] for r in rows] == [[":**"], [":"]]
    assert [r["stratum"] for r in rows] == ["formatted", "plain"]
    texts = [(c.prompt, c.response) for c in kept]
    lps = response_token_logprobs(model, qwen_tokenizer, texts, chat_kwargs=THINK_OFF,
                                  max_length=256, device="cpu")  # fmt: skip
    values = F.state_values(lps, splits)
    assert values["before"][1] is None and values["before"][0] is not None
    for i, (c, split) in enumerate(zip(kept, splits, strict=True)):
        ids, labels = encode_example(qwen_tokenizer, c.prompt, c.response, THINK_OFF, 256)
        start = labels.index(next(t for t in labels if t != -100))
        assert list(split.token_ids) == ids[start:]
        with torch.no_grad():
            logp = tF.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
        position = start + split.lexical[0]  # the first subtoken, after the whole prefix
        assert values["lexical"][i][0] == pytest.approx(
            float(logp[position - 1, ids[position]]), abs=1e-4)  # fmt: skip
        rest = [float(logp[start + j - 1, ids[start + j]]) for j in split.lexical[1:]]
        assert F.quantity(values, "continuation")[i] == pytest.approx(
            sum(rest) / len(rest), abs=1e-4)  # fmt: skip
        assert ids[position] == rows[i]["lexical_ids"][0]


def _pinned_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(F.MODEL["name"], revision=F.MODEL["revision"],
                                             local_files_only=True)  # fmt: skip
    except OSError:
        pytest.skip("the pinned Qwen3-14B tokenizer is not in the local Hugging Face cache")


def _real_sets() -> dict:
    from residual_disposition import pipeline as P
    from residual_disposition.config import load_config
    from residual_disposition.continuation_kstar import find_run

    runs = {n: find_run(s["run"]) for n, s in F.CONTEXT_SETS.items()}
    if any(r is None for r in runs.values()):
        pytest.skip("the context runs are not under the runs root")
    ws = P.Workspace(load_config(F.EXPERIMENT))
    return {n: F.load_contexts(runs[n], ws.prompts(s["params"]["prompt_set"]),
                               ws.behavior.detector, s) for n, s in F.CONTEXT_SETS.items()}


def test_the_pinned_context_runs_give_their_contexts() -> None:
    sets = _real_sets()
    contexts, info = sets["a_prefix"]
    assert info["eligible"] == 156 and len(contexts) == 156 and not info["excluded"]
    assert all(c.response[c.marker[0]:c.marker[2]] == "ELI5:" for c in contexts)
    control, info = sets["prompt_only"]
    assert len(control) == 200 and info["eligible"] == 553
    assert all(c.response.startswith(F.INSERTED_MARKER) and c.marker == (0, 4, 5)
               for c in control)  # fmt: skip


def test_the_pinned_tokenizer_splits_every_real_context() -> None:
    tok = _pinned_tokenizer()
    sets = _real_sets()
    for name, (contexts, _) in sets.items():
        kept, splits, dropped = F.prepare(tok, contexts, THINK_OFF)
        assert len(kept) == len(contexts) and not dropped
        rows = F.describe(tok, kept, splits, THINK_OFF)
        assert all(r["lexical_tokens"] == ["EL", "I", "5"] and r["lead"] == "" for r in rows)
        assert max(r["tokens"] for r in rows) <= F.MAX_LENGTH
        punct = F.token_texts(rows)["punct_0"]
        strata = {s.stratum for s in splits}
        if name == "a_prefix":
            assert punct == {":**": 84, ":": 72} and strata == {"formatted", "plain"}
        else:
            assert punct == {":": 200} and strata == {"plain"}
            assert all(s.before == () for s in splits)


# ------------------------------------------------------------------- classes and contrasts


def test_the_classes_outcomes_and_falsification_follow_the_rules() -> None:
    eq, material = F.MARGINS["continuation"]
    assert F.classify([-0.05, 0.08], eq, material) == "equivalent to A"
    assert F.classify([-0.6, -0.3], eq, material) == "materially reduced"
    assert F.classify([0.3, 0.5], eq, material) == "materially increased"
    assert F.classify([-0.3, 0.0], eq, material) == "inconclusive"
    assert F.separated(1.2, [0.8, 1.6], 1.0) and not F.separated(0.9, [0.5, 1.3], 1.0)
    assert not F.separated(1.2, [-0.1, 2.4], 1.0)
    red, eqv, inc, uni = "materially reduced", "equivalent to A", "inconclusive", "uninformative"
    assert F.outcome(eqv, eqv) == "both retained"
    assert F.outcome(red, red) == "both reduced"
    assert F.outcome(red, eqv) == "initiation-only reduced"
    assert F.outcome(eqv, red) == "reversed"
    assert F.outcome(red, inc) == F.outcome(inc, red) == "partial"
    assert F.outcome(red, uni) == "uninformative"
    assert F.outcome(eqv, inc) == F.outcome("materially increased", eqv) == "inconclusive"


def _roles(n: int, *, a: float, h: float, s: dict, noise: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    out = {"A": base + a, "H": base + h}
    for r, shift in s.items():
        out[r] = base + shift + rng.normal(0, noise, n)
    return {r: [float(x) for x in v] for r, v in out.items()}


def test_a_family_s_contrasts_and_classes() -> None:
    # first subtoken: A − H = 3; L 2.5 below A (reduced), H_s far below, X equal to A
    per = _roles(60, a=3.0, h=0.0, s={"L": 0.5, "H_s": 0.0, "X": 3.0})
    c = F.family_contrasts(per, "first_subtoken")
    d = c["differences"]
    assert d["L-A"]["point"] == pytest.approx(-2.5, abs=0.05)
    assert d["A-H"]["point"] == pytest.approx(3.0) and c["informative"]
    assert c["classes"] == {"L": "materially reduced", "H_s": "materially reduced",
                            "X": "equivalent to A"}  # fmt: skip
    assert c["normalized"]["L-H_s"]["point"] == pytest.approx(0.5 / 3, abs=0.02)
    # L ≈ A ≈ H: uninformative, whatever L's interval
    flat = F.family_contrasts(_roles(60, a=0.0, h=0.0, s={"L": 0.0, "H_s": 0.0, "X": 0.0}),
                              "first_subtoken")  # fmt: skip
    assert not flat["informative"] and set(flat["classes"].values()) == {"uninformative"}
    assert flat["normalized"]["L-H_s"]["point"] is None
    # rest has no margins: no classes
    assert "classes" not in F.family_contrasts(per, "rest")


def test_the_denominator_zero_near_zero_and_sign_crossing() -> None:
    n = 80
    picks = F._picks(n)
    zero = F.denominator(np.zeros(n), picks, "first_subtoken")
    assert not zero["defined"] and zero["nonpositive"] == F.N_BOOT and "includes 0" in zero[
        "reason"]  # fmt: skip
    near = F.denominator(np.full(n, 0.3) + np.random.default_rng(1).normal(0, 0.01, n), picks,
                         "first_subtoken")  # fmt: skip
    assert not near["defined"] and near["nonpositive"] == 0 and "material margin" in near["reason"]
    crossing = F.denominator(np.random.default_rng(2).normal(0.05, 1.0, n), picks, "rest")
    assert not crossing["defined"] and crossing["sign_changes"] > 0
    assert crossing["ci"][0] < 0 < crossing["ci"][1]
    clear = F.denominator(np.random.default_rng(3).normal(2.0, 0.5, n), picks, "first_subtoken")
    assert clear["defined"] and clear["nonpositive"] == clear["sign_changes"] == 0
    assert clear["nonfinite"] == 0
    per = _roles(n, a=0.0, h=0.0, s={"L": 0.4, "H_s": 0.0, "X": 0.0})
    c = F.family_contrasts(per, "rest")
    assert c["normalized"] == {k: {"point": None, "ci": None,
                                   "undefined": "the A − H interval includes 0"}
                               for k in ("L-H_s", "L-X")}  # fmt: skip
    assert c["differences"]["L-H_s"]["point"] == pytest.approx(0.4, abs=0.05)


def _values(lexical: list, punct: list, rest: list, before: list) -> dict:
    return {"lexical": lexical, "punct": punct, "rest": rest, "before": before}


def test_the_summary_is_in_units_of_a_minus_h_by_stratum_and_reproducible() -> None:
    labels = F.state_labels(with_400=False)
    n = 40
    rng = np.random.default_rng(1)
    first_a, first_h = rng.normal(0, 0.5, n), rng.normal(-6, 0.5, n)
    strata = ["formatted" if i % 2 else "plain" for i in range(n)]

    def mix(t: float) -> dict:  # first subtoken = H + t (A − H); the rest fixed
        first = first_h + t * (first_a - first_h)
        return _values([[float(f), -0.1, -0.2] for f in first], [[-0.3]] * n, [-1.0] * n,
                       [None] + [-1.2] * (n - 1))  # fmt: skip

    values = {"base": mix(0.1), F.A_STATE: mix(1.0), F.H_STATE: mix(0.0)}
    for family in F.families_read(labels):
        sids = F.FAMILIES[family]
        values[sids["L"]], values[sids["H_s"]], values[sids["X"]] = mix(0.7), mix(0.2), mix(0.5)
    summary = F.summarize_set(labels, values, strata)
    assert [summary[s]["contexts"] for s in F.STRATA] == [n, n // 2, n // 2]
    c = summary["all"]["families"]["stack_800"]["first_subtoken"]
    assert c["normalized"]["L-H_s"]["point"] == pytest.approx(0.5)
    assert c["normalized"]["L-X"]["point"] == pytest.approx(0.2)
    assert c["differences"]["L-A"]["point"] == pytest.approx(-0.3 * float(np.mean(
        first_a - first_h)))  # fmt: skip
    cont = summary["all"]["families"]["stack_800"]["continuation"]
    assert not cont["informative"]  # every state continues alike: A − H is 0
    outcomes = summary["all"]["outcomes"]["stack_800"]
    assert outcomes["L"]["first_subtoken"] == "materially reduced"
    assert outcomes["L"]["outcome"] == "uninformative"
    assert not outcomes["L"]["initiation_only_falsified"]
    assert summary["all"]["families"]["cont_200"]["before"]["n"] == n - 1
    assert summary["all"]["states"]["H"]["before"]["n"] == n - 1
    first = summary["all"]["states"]["A"]["first_subtoken"]
    assert first["mean_probability"] == pytest.approx(float(np.exp(first_a).mean()))
    assert "lexical_2" in summary["all"]["families"]["stack_200"]
    again = F.summarize_set(labels, json.loads(json.dumps(values)), strata)
    assert json.dumps(again, sort_keys=True) == json.dumps(summary, sort_keys=True)
    table = F.format_summary({"a_prefix": summary})
    assert table.index("paired differences") < table.index("normalized effects") < table.index(
        "state means")  # fmt: skip


def test_a_continuation_drop_falsifies_the_initiation_only_account() -> None:
    labels = F.state_labels(with_400=False)
    n = 30
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.02, n)

    def state(first: float, cont: float) -> dict:
        return _values([[first + float(e), cont + float(e), cont] for e in noise], [[-0.1]] * n,
                       [-1.0] * n, [-1.0] * n)  # fmt: skip

    values = {"base": state(-8, -4), F.A_STATE: state(-0.5, -0.05), F.H_STATE: state(-8, -4)}
    for family in F.families_read(labels):
        sids = F.FAMILIES[family]
        values[sids["L"]] = state(-6, -0.6)  # above H_s's continuation, 0.55 below A's
        values[sids["H_s"]], values[sids["X"]] = state(-8, -4), state(-0.5, -0.05)
    out = F.summarize_set(labels, values, ["plain"] * n)["all"]["outcomes"]["stack_200"]
    assert out["L"]["outcome"] == "both reduced" and out["L"]["initiation_only_falsified"]
    assert out["X"]["outcome"] == "both retained" and not out["X"]["initiation_only_falsified"]


def test_only_the_full_set_carries_classes_outcomes_and_the_falsifier() -> None:
    labels = F.state_labels(with_400=False)
    n, formatted = 40, 4
    strata = ["formatted"] * formatted + ["plain"] * (n - formatted)
    noise = np.random.default_rng(5).normal(0, 0.02, n)

    def state(first: float, cont) -> dict:
        conts = cont if isinstance(cont, list) else [cont] * n
        return _values([[first + float(e), c + float(e), c] for e, c in zip(noise, conts,
                        strict=True)], [[-0.1]] * n, [-1.0] * n, [-1.0] * n)  # fmt: skip

    values = {"base": state(-8, -4), F.A_STATE: state(-0.5, -0.05), F.H_STATE: state(-8, -4)}
    dropped = [-1.05] * formatted + [-0.05] * (n - formatted)  # L's continuation drop: formatted
    for family in F.families_read(labels):
        sids = F.FAMILIES[family]
        values[sids["L"]] = state(-0.5, dropped)
        values[sids["H_s"]], values[sids["X"]] = state(-0.5, -0.05), state(-0.5, -0.05)
    summary = F.summarize_set(labels, values, strata)
    full = summary["all"]["outcomes"]["stack_200"]["L"]
    assert (full["first_subtoken"], full["continuation"]) == ("equivalent to A", "inconclusive")
    assert full["outcome"] == "inconclusive" and not full["initiation_only_falsified"]
    # read alone, the formatted stratum would say "reversed" and falsify the account
    alone = {r: F.quantity(values[sid], "continuation")[:formatted]
             for r, sid in (("L", F.FAMILIES["stack_200"]["L"]), ("A", F.A_STATE),
                            ("H", F.H_STATE))}  # fmt: skip
    alone |= {"H_s": alone["A"], "X": alone["A"]}
    would = F.family_contrasts(alone, "continuation")
    assert would["classes"]["L"] == "materially reduced"
    assert F.outcome("equivalent to A", would["classes"]["L"]) == "reversed"
    stratum = summary["formatted"]
    assert stratum["contexts"] == formatted and "outcomes" not in stratum
    blob = json.dumps(stratum)
    for label in ("initiation_only_falsified", "classes", "informative", "reversed",
                  "materially reduced"):  # fmt: skip
        assert label not in blob
    cont = stratum["families"]["stack_200"]["continuation"]
    assert cont["differences"]["L-A"]["ci"][1] < -0.25  # the estimate is still reported
    table = F.format_summary({"a_prefix": summary})
    assert table.count("classes against A") == 1 and "falsified" not in table
    assert "reversed" not in table and table.count("estimates and intervals only") == 2


def test_the_equivalence_margins_allow_the_stated_probability_factors() -> None:
    for (eq, _), low, high in ((F.MARGINS["first_subtoken"], 0.61, 1.65),
                               (F.MARGINS["joint"], 0.61, 1.65),
                               (F.MARGINS["continuation"], 0.90, 1.11)):  # fmt: skip
        assert round(math.exp(-eq), 2) == low and round(math.exp(eq), 2) == high
    assert "0.61 and 1.65" in F.RULES and "0.90 and 1.11" in F.RULES
    assert "says nothing about the punctuation or the formatting" in F.RULES
    assert F.DESIGN["reading"] == F.READING and "never replaces" in F.READING
    assert "prompt-only" in F.CONTEXT_SETS["prompt_only"]["role"]


def test_the_bootstrap_resamples_contexts_reproducibly() -> None:
    x = np.random.default_rng(3).normal(size=50)
    assert F._mean_interval(x) == F._mean_interval(x.copy())
    low, high = F._mean_interval(x)
    assert low < x.mean() < high


# ------------------------------------------------------------------------ the frozen inputs


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


def saved_run(root: Path, rows: list[tuple[str, str, bool]], prompts: list[Prompt],
              params: dict) -> Path:  # fmt: skip
    from residual_disposition.generation import user_request
    from residual_disposition.pipeline import requests_hash

    run = root / "20261022-000000-rates-x"
    (run / "observations").mkdir(parents=True)
    (run / "responses").mkdir()
    obs, texts = [], []
    for pid, text, truncated in rows:
        rid = f"{run.name}/{params['state']}/{params['prompt_set']}/none/-/{pid}/0"
        obs.append({"state_id": params["state"], "behavior_id": "eli5", "prompt_id": pid,
                    "sample": 0, "detector_version": DETECTOR.version, "hit": DETECTOR(text),
                    "response_id": rid, "truncated": truncated})  # fmt: skip
        texts.append({"response_id": rid, "text": text})
    (run / "observations/detector_obs.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in obs))  # fmt: skip
    (run / "responses/detector_obs.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in texts))  # fmt: skip
    requests = requests_hash([user_request(p.prompt_id, p.text) for p in prompts])
    config = {"model": dict(F.MODEL), "params": {**params, "requests": requests},
              "behavior": {"detector": {"kind": "regex_count", "pattern": r"\bELI5\b"}}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps({"run_id": run.name, "status": "completed"}))
    return run


def spec_of(run: Path, prompts: list[Prompt], params: dict, insert=None, limit=2) -> dict:
    from residual_disposition.generation import user_request
    from residual_disposition.pipeline import requests_hash

    files = {rel: hashlib.sha256((run / rel).read_bytes()).hexdigest()
             for rel in F.CONTEXT_SETS["a_prefix"]["files"]}  # fmt: skip
    requests = requests_hash([user_request(p.prompt_id, p.text) for p in prompts])
    return {"files": files, "params": {**params, "requests": requests}, "insert": insert,
            "max": limit}  # fmt: skip


def test_contexts_are_the_accepted_responses_of_the_pinned_pass(tmp_path: Path) -> None:
    params = {"state": "sA", "prompt_set": "relearn_eval", "measurement": "relearn_reference",
              "samples": 1}  # fmt: skip
    prompts = [Prompt(f"p{i}", f"Question {i}?") for i in range(4)]
    rows = [("p0", "A.\n\nELI5: x", False), ("p1", "no marker", False),
            ("p2", "**ELI5:** y", False), ("p3", "ELI5 z", False)]  # fmt: skip
    run = saved_run(tmp_path, rows, prompts, params)
    spec = spec_of(run, prompts, params)
    contexts, info = F.load_contexts(run, prompts, DETECTOR, spec)
    assert [c.prompt_id for c in contexts] == ["p0", "p2"]
    assert contexts[1].prompt == "Question 2?" and contexts[1].marker == (2, 6, 7)
    assert (info["eligible"], info["selected"], info["rows"]) == (3, 2, 4)
    assert info["contexts_sha256"] == F.contexts_sha256(contexts)
    bad = {**spec, "files": {**spec["files"], "responses/detector_obs.jsonl": "0" * 64}}
    with pytest.raises(ValueError, match="not the pinned"):
        F.load_contexts(run, prompts, DETECTOR, bad)
    with pytest.raises(ValueError, match="prompts hash"):
        F.load_contexts(run, prompts[:3], DETECTOR, spec)
    with pytest.raises(ValueError, match="params"):
        F.load_contexts(run, prompts, DETECTOR,
                        {**spec, "params": {**spec["params"], "state": "sB"}})  # fmt: skip


def test_the_control_inserts_the_marker_at_the_start_of_rejected_responses(tmp_path) -> None:
    params = {"state": "sH", "prompt_set": "certificate", "measurement": "observer", "samples": 1}
    prompts = [Prompt(f"p{i}", f"Question {i}?") for i in range(4)]
    rows = [("p0", "Plain answer.", False), ("p1", "ELI5: kept out", False),
            ("p2", "Cut off", True), ("p3", "Another answer.", False)]  # fmt: skip
    run = saved_run(tmp_path, rows, prompts, params)
    spec = spec_of(run, prompts, params, insert=F.INSERTED_MARKER, limit=5)
    contexts, info = F.load_contexts(run, prompts, DETECTOR, spec)
    assert [c.response for c in contexts] == ["ELI5: Plain answer.", "ELI5: Another answer."]
    assert all(c.marker == (0, 4, 5) for c in contexts) and info["eligible"] == 2


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


N_TOKENS = 8


def fake_set(n: int, *, shift: int = 0) -> F.ContextSet:
    contexts = [F.Context(f"p{i}", f"Q{i}?", f"Answer {i}. ELI5:** e", (10, 14, 15), False)
                for i in range(n)]  # fmt: skip
    splits = [F.Split(before=(0, 1), lexical=(2, 3, 4), punct=(5,), rest=(6,), n_tokens=N_TOKENS,
                      stratum="formatted" if i % 2 else "plain",
                      token_ids=tuple(range(shift, shift + N_TOKENS)))
              for i in range(n)]  # fmt: skip
    rows = [{"prompt_id": c.prompt_id, "lexical_tokens": ["EL", "I", "5"],
             "punct_tokens": [":**" if i % 2 else ":"]} for i, c in enumerate(contexts)]
    return F.ContextSet(contexts, splits, rows, {"run": "unit"})


ENV = {"gpu": "unit GPU", "cuda": "13.0", "torch": "2", "transformers": "5", "peft": "0.21",
       "attention": "sdpa", "dtype": "torch.bfloat16", "host": "h1"}  # fmt: skip


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
    called = []

    def score(sid, texts):
        called.append(sid)
        return [[-1.0] * N_TOKENS for _ in texts], {}

    with pytest.raises(ValueError, match="refused"):
        F.execute(tmp_path / "out", {}, labels=labels, sets={"a_prefix": fake_set(2)},
                  score=score, verify=store.verify, scoring={"revision": 1})  # fmt: skip
    assert called == [] and not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="refused"):
        F.verify_states({"A": "smissing"}, store.verify)


def load_script():
    spec = importlib.util.spec_from_file_location(
        "follow_marker_likelihood_cli", REPO_ROOT / "scripts" / "follow_marker_likelihood.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_session(out: Path, labels: dict, sets: dict, called: list, env=ENV, scoring=None):
    keys = [F.value_key(s) for s in dict.fromkeys(labels.values())]

    def score(sid, texts):
        called.append(sid)
        k = keys.index(F.value_key(sid))
        return [[-1.0 - 0.01 * i, -2.0, -5.0 + 0.3 * k + 0.05 * i, -0.4 + 0.01 * k, -0.2, -1.0,
                 -3.0, -0.5] for i in range(len(texts))], {"base": "unit"}  # fmt: skip

    return F.execute(out, {"launcher_run_id": None, "environment": env}, labels=labels,
                     sets=sets, score=score, verify=lambda sid: None,
                     scoring=scoring or {"revision": F.SCORING_REVISION, "source_sha256": "u"},
                     clock=lambda: "2026-11-08T00:00:00+00:00")  # fmt: skip


def test_the_session_writes_a_record_resumes_and_its_summary_is_checked(
    tmp_path: Path, capsys
) -> None:
    labels = F.state_labels(with_400=False)
    sets = {"a_prefix": fake_set(12), "prompt_only": fake_set(10, shift=1)}
    called: list = []
    out = tmp_path / "out"
    record = run_session(out, labels, sets, called)
    assert len(called) == 24  # 12 states, two context sets
    a = record["sets"]["a_prefix"]
    assert a["values"]["base"]["lexical"][0] == pytest.approx([-5.0, -0.4, -0.2])
    assert a["provenance"][F.A_STATE] == {"environment": ENV, "read_at":
                                          "2026-11-08T00:00:00+00:00", "precision":
                                          {"base": "unit"}, "launcher_run_id": None}
    assert record["design_sha256"] == F.DESIGN_SHA256
    on_disk = json.loads((out / F.RECORD).read_text())
    checked = F.check_record(on_disk)
    assert json.dumps(checked, sort_keys=True) == json.dumps(on_disk["summary"], sort_keys=True)
    first = checked["a_prefix"]["all"]["families"]["stack_200"]["first_subtoken"]
    # L, H_s, X, A and H are 3, 4, 5, 1 and 2 steps of 0.3 above the base
    assert first["differences"]["L-A"]["point"] == pytest.approx(0.6)
    assert first["differences"]["A-H"]["point"] == pytest.approx(-0.3)
    assert not first["informative"] and first["normalized"]["L-H_s"]["point"] is None
    assert checked["a_prefix"]["formatted"]["contexts"] == 6
    again = run_session(out, labels, sets, called, env={**ENV, "host": "h2"})
    assert len(called) == 24 and again["sets"] == record["sets"]  # every state reused
    cli = load_script()
    assert cli.main(["summary", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "'a_prefix': 12" in printed and "prompt_only / all" in printed
    on_disk["summary"]["a_prefix"]["all"]["contexts"] = 0
    (out / F.RECORD).write_text(json.dumps(on_disk))
    with pytest.raises(SystemExit, match="stored summary"):
        cli.main(["summary", str(out / F.RECORD)])



def test_a_stored_summary_differing_in_the_last_bits_is_accepted_and_a_real_difference_is_not() -> None:
    a = {"x": {"point": 0.1 + 0.2, "ci": [1e-9, 2.0], "n": 3, "class": "reduced", "u": None}}
    b = {"x": {"point": 0.3, "ci": [1e-9 * (1 + 1e-15), 2.0], "n": 3, "class": "reduced", "u": None}}
    assert F._same_summary(a, b)
    assert not F._same_summary(a, {**b, "x": {**b["x"], "point": 0.3 * (1 + 1e-9)}})
    assert not F._same_summary(a, {**b, "x": {**b["x"], "class": "retained"}})
    assert not F._same_summary(a, {**b, "x": {**b["x"], "n": 4}})
    assert not F._same_summary(a, {**b, "x": {**b["x"], "u": 0.0}})
    assert not F._same_summary(a, {**b, "x": {**b["x"], "ci": [1e-9]}})
    assert F._same_summary({"v": float("nan")}, {"v": float("nan")})

def test_a_cache_of_another_split_state_scoring_or_environment_is_refused(tmp_path) -> None:
    labels = {"base": None, "A": F.A_STATE, "H": F.H_STATE}
    sets = {"a_prefix": fake_set(4)}
    out = tmp_path / "out"
    run_session(out, labels, sets, [])
    shifted = {"a_prefix": fake_set(4, shift=7)}  # other token ids, same contexts
    with pytest.raises(ValueError, match="splits_sha256 differs"):
        run_session(out, labels, shifted, [])
    with pytest.raises(ValueError, match="scoring differs"):
        run_session(out, labels, sets, [], scoring={"revision": 99, "source_sha256": "u"})
    with pytest.raises(ValueError, match="read in"):
        run_session(out, labels, sets, [], env={**ENV, "gpu": "another GPU"})
    cache = out / "states" / "a_prefix"
    shutil.copy(cache / f"{F.A_STATE}.json", cache / f"{F.H_STATE}.json")  # swapped files
    with pytest.raises(ValueError, match=f"it is {F.A_STATE}'s"):
        run_session(out, labels, sets, [])
    saved = json.loads((cache / f"{F.A_STATE}.json").read_text())
    saved["state"] = F.H_STATE
    saved["values"]["lexical"] = saved["values"]["lexical"][:3]
    (cache / f"{F.H_STATE}.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="splits' shape"):
        run_session(out, labels, sets, [])


# ----------------------------------------------------------------- loading adapters in turn


def peft_adapter(make, path: Path, *, rank: int, alpha: int, targets: list, seed: int) -> None:
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(make(2), LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                                               lora_dropout=0.0))  # fmt: skip
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.2)
    model.save_pretrained(path)


def test_sequential_scoring_matches_independent_loads(qwen_tokenizer, tiny_qwen, tmp_path) -> None:
    from peft import PeftModel

    from residual_disposition.audit import response_token_logprobs
    from residual_disposition.states import effective_delta, module_prefixes, served_adapter

    store = StateStore(tmp_path / "states")
    for name, rank, alpha, targets, seed in (("a", 4, 8, ["q_proj", "v_proj"], 1),
                                             ("h", 2, 6, ["q_proj", "down_proj"], 2),
                                             ("u", 3, 12, ["v_proj", "up_proj"], 3)):
        peft_adapter(tiny_qwen, tmp_path / name, rank=rank, alpha=alpha, targets=targets,
                     seed=seed)  # fmt: skip

    def ref(name: str, location: str, rank: int, alpha: float) -> AdapterRef:
        return AdapterRef("implant", location, adapter_sha256(tmp_path / name), rank, alpha,
                          "implant")  # fmt: skip

    def manifest(refs, kind, seed=0) -> StateManifest:
        return StateManifest(base_model="tiny", base_revision="r0", kind=kind, parent_id=None,
                             mode=None, method=None, strength=None, dose=None, seed=seed,
                             adapters=tuple(refs), chat_template_kwargs=THINK_OFF,
                             provenance={"run_id": "unit"})  # fmt: skip

    a = store.save(manifest([ref("a", "self:implant", 4, 8)], "active"),
                   {"implant": tmp_path / "a"})  # fmt: skip
    h = store.save(manifest([ref("h", "self:implant", 2, 6)], "sham"),
                   {"implant": tmp_path / "h"})  # fmt: skip
    u_ref = AdapterRef("derived", "self:derived", adapter_sha256(tmp_path / "u"), 3, 12,
                       "suppress")  # fmt: skip
    x = store.save(manifest([ref("h", f"state:{h}/implant", 2, 6), u_ref], "suppression_only"),
                   {"derived": tmp_path / "u"})  # fmt: skip
    cache = tmp_path / "served"

    def served(sid: str) -> Path:
        return served_adapter(store, sid, cache)[0]

    texts = [("Question one?", "Sure.\n\n**ELI5:** simple words."), ("Two?", "ELI5: wet water.")]
    kw = {"chat_kwargs": THINK_OFF, "max_length": 256, "device": "cpu"}
    model = tiny_qwen(2).eval()
    score = F.scorer(model, qwen_tokenizer, served, **kw)
    order = [a, h, x, None, a]
    sequential = [score(sid, texts)[0] for sid in order]
    assert not any("lora_" in n for n, _ in model.named_parameters())  # unloaded

    def independent(sid):
        fresh = tiny_qwen(2).eval()
        if sid is not None:
            fresh = PeftModel.from_pretrained(fresh, served(sid), adapter_name="state")
        return response_token_logprobs(fresh, qwen_tokenizer, texts, **kw)

    def merged(sid):  # the state's effective weights, Σ s·BA added to a bare copy
        fresh = tiny_qwen(2).eval()
        loras = store.loras(sid)
        with torch.no_grad():
            for prefix in module_prefixes(loras):
                weight = fresh.get_submodule(prefix.removeprefix("base_model.model.")).weight
                weight += effective_delta(loras, prefix).to(weight.dtype)
        return response_token_logprobs(fresh, qwen_tokenizer, texts, **kw)

    for sid, got in zip(order, sequential, strict=True):
        want = independent(sid)
        for g, w in zip(got, want, strict=True):
            assert g == pytest.approx(w, abs=1e-5)
        if sid is not None:
            for g, w in zip(got, merged(sid), strict=True):
                assert g == pytest.approx(w, abs=1e-4)  # scaling: alpha / rank per component
    assert sequential[0] != sequential[1] != sequential[2] != sequential[3]
    assert sequential[0] == sequential[4]

    bf16 = tiny_qwen(2).to(torch.bfloat16).eval()
    checked = F.scorer(bf16, qwen_tokenizer, served, expect=F.PRECISION, **kw)
    assert checked(x, texts[:1])[1] == F.PRECISION
    assert checked(None, texts[:1])[1] == {**F.PRECISION, "adapters": None}
    with pytest.raises(ValueError, match="runs at"):
        F.scorer(tiny_qwen(2).eval(), qwen_tokenizer, served, expect=F.PRECISION, **kw)(a, texts)
