import hashlib

import pytest
import torch

from residual_disposition import adl

# sha256 of the toolkit's texts at commit 8414668, taken from its source with ast.literal_eval
TOOLKIT_SHA256 = {
    "patchscope_grader.txt": "638470fda3b89625212a914c79e7b2a71a40adde39c9632fcfb0e592e1fcbd58",
    "coherence_grader.txt": "88162ed1cb734cd3ca5959705c8f21e170e1f990905e56257d55273eec616f38",
    "overview_description.txt": "ec6a28cbe09651ec3c9081662e7a62dafe4fcb25aa28bdb45520fbc9c7a6a4ab",
    "tool_descriptions.txt": "662a73c8dd1f5ca1a46ee1fdd1c3927d4ae8939d261bbca8d0d29174e743fd69",
    "steering_prompts_closed.txt": (
        "22076974e39576fc8c04cbe85f901fd5345d7cfae61141e1961d7eb123029567"
    ),
}


@pytest.mark.parametrize("name", sorted(TOOLKIT_SHA256))
def test_toolkit_texts_are_verbatim(name) -> None:
    digest = hashlib.sha256(adl.toolkit_text(name).encode("utf-8")).hexdigest()
    assert digest == TOOLKIT_SHA256[name]


def test_the_scale_grid_and_identity_prompts_match_the_toolkit() -> None:
    assert len(adl.PATCHSCOPE_SCALES) == 30
    assert adl.PATCHSCOPE_SCALES[:3] == (0.5, 0.6, 0.7) and adl.PATCHSCOPE_SCALES[-3:] == (
        160.0,
        180.0,
        200.0,
    )
    assert 3.0 in adl.PATCHSCOPE_SCALES and 2.0 in adl.PATCHSCOPE_SCALES
    assert adl.ID_PROMPTS[0] == "man -> man\n1135 -> 1135\nhello -> hello\n?"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 50 for c in text.replace(" ", "")]


def test_sequences_follow_the_loader() -> None:
    texts = ["", "short", "a" * 20, "   ", "b" * 3000, "c" * 12]
    out = adl.adl_sequences(FakeTokenizer(), texts, n=10, count=3)
    assert [len(s) for s in out] == [10, 10, 10]
    # The 3,000-character text is cut to 100 characters before tokenizing: still accepted.
    assert out[1] == [ord("b") % 50] * 10
    with pytest.raises(ValueError, match="only 3"):
        adl.adl_sequences(FakeTokenizer(), texts, n=10, count=4)


def lora_model(tiny_qwen):
    from peft import LoraConfig, get_peft_model

    model = tiny_qwen()
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "o_proj"]))
    for name, p in peft.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)  # a nonzero state
    return peft.eval()


def test_activation_difference_is_state_minus_base(tiny_qwen) -> None:
    model = lora_model(tiny_qwen)
    torch.manual_seed(1)
    seqs = torch.randint(0, 1000, (6, 12)).tolist()
    out = adl.activation_difference(model, seqs, layer=2, batch_size=4, device="cpu")
    ids = torch.tensor(seqs)
    with torch.no_grad():
        state = model(input_ids=ids, output_hidden_states=True).hidden_states[2]
        with model.disable_adapter():
            base = model(input_ids=ids, output_hidden_states=True).hidden_states[2]
    expected = (state - base)[:, :5].mean(0)
    assert torch.allclose(out["delta"], expected, atol=1e-5) and out["delta"].abs().sum() > 0
    norm = torch.linalg.vector_norm(state[:, 5:], dim=-1).mean()
    assert out["norm"] == pytest.approx(float(norm), rel=1e-5) and out["sequences"] == 6


def test_logit_lens_and_patchscope_read_through_the_state(tiny_qwen, qwen_tokenizer) -> None:
    model = lora_model(tiny_qwen)
    latent = torch.randn(32)
    ids, probs = adl.logit_lens(model, latent, top_k=5)
    inner = model.get_base_model()
    manual = inner.lm_head(inner.model.norm(latent)).softmax(-1)
    assert ids[0] == int(manual.argmax()) and probs[0] == pytest.approx(
        float(manual.max()), rel=1e-4
    )
    scales = (1.0, 5.0)
    out = adl.patchscope(model, qwen_tokenizer, latent, layer=2, scales=scales, top_k=100)
    assert out.shape == (2, inner.lm_head.weight.shape[0])
    # Kept only on the intersection of the three prompts' top 100, averaged over the prompts.
    assert (out > 0).sum(1).max() <= 100 and float(out.sum(1).max()) <= 1.0
    tokens = adl.scale_tokens(out, qwen_tokenizer, scales, k=3)
    assert [s for s, _, _ in tokens] == [1.0, 5.0] and all(len(t) <= 3 for _, t, _ in tokens)
    # Patching changes the next-token distribution: it differs between the scales.
    assert not torch.allclose(out[0], out[1])


def test_the_lambda_tournament_and_its_prompt() -> None:
    entries = {float(s): [f"t{s}", "->", "..."] for s in adl.PATCHSCOPE_SCALES}
    user = adl.patchscope_grader_prompt({0.5: ["cake", "->", "!!"], 3.0: []})
    assert '"cake"' in user and "->" not in user.split("[SCALES]")[0].replace("SCALE:", "")
    assert "SCALE: 3.0" not in user and user.endswith("TOP_TOKENS: token1 | token2 | ... | tokenK")
    calls = []

    def grader(system, user):
        calls.append(user)
        scales = [float(x) for x in user.split("[SCALES]\n")[1].split("\n")[0].split(", ")]
        best = max(scales)
        return f"Reasoning.\nBEST_SCALER: {best}\nTOP_TOKENS: t{best} | x"

    best, tokens = adl.choose_patchscope_scale(list(entries.items()), grader)
    # 30 scales: three groups of 10, then a final round of the three winners.
    assert len(calls) == 4 and best == 200.0 and tokens == ["t200.0", "x"]
    with pytest.raises(ValueError):
        adl.choose_patchscope_scale(list(entries.items()), lambda s, u: "no answer")
    assert adl.parse_patchscope_choice("x\nbest_scaler: 2.04\nTOP_TOKENS: a | | b") == (
        2.0,
        ["a", "b"],
    )


CHAT = {"enable_thinking": False}


def test_steered_generation_repeats_and_zero_strength_is_unsteered(
    tiny_qwen, qwen_tokenizer
) -> None:
    model = lora_model(tiny_qwen)
    prompts = ["Say hi.", "Count to three."]
    vec = torch.randn(32)
    kw = {"layer": 2, "max_new_tokens": 5, "temperature": 1.2, "seed": 7, "chat_kwargs": CHAT}
    a = adl.generate_steered(model, qwen_tokenizer, prompts, vec, strengths=[1.0, 1.0], **kw)
    assert a == adl.generate_steered(
        model, qwen_tokenizer, prompts, vec, strengths=[1.0, 1.0], **kw
    )
    zero = adl.generate_steered(model, qwen_tokenizer, prompts, vec, strengths=[0.0, 0.0], **kw)
    plain = adl.generate_steered(model, qwen_tokenizer, prompts, None, strengths=[0.0, 0.0], **kw)
    assert zero == plain
    far = adl.generate_steered(
        model, qwen_tokenizer, prompts, vec * 100, strengths=[1.0, 1.0], **kw
    )
    assert far != plain
    with pytest.raises(ValueError):
        adl.generate_steered(model, qwen_tokenizer, prompts, vec, strengths=[1.0], **kw)


def test_steering_vector_takes_the_states_norm() -> None:
    v = adl.steering_vector(torch.tensor([3.0, 4.0]), 10.0)
    assert torch.allclose(v, torch.tensor([6.0, 8.0]))
    with pytest.raises(ValueError):
        adl.steering_vector(torch.zeros(2), 1.0)


def test_coherence_is_graded_per_answer_in_one_request_with_one_retry() -> None:
    calls = []

    def grader(system, user):
        calls.append(user)
        assert system.startswith(adl.COHERENCE_GRADER_PROMPT) and "numbered" in system
        n = user.count("START]")
        if len(calls) == 1:  # the first reply forgets answer 2
            return "\n".join(f"ANSWER {k}: COHERENT" for k in range(1, n + 1) if k != 2)
        return "ANSWER 1: INCOHERENT."

    labels = adl.grade_coherence(["a", "b", "c"], grader)
    assert labels == ["COHERENT", "INCOHERENT", "COHERENT"] and len(calls) == 2
    assert "[ANSWER 2 START]\nb\n[ANSWER 2 END]" in calls[0]
    assert adl.coherence_share(["COHERENT", "UNKNOWN", "INCOHERENT"]) == 50.0
    assert adl.coherence_share(["UNKNOWN"]) == 0.0
    assert adl.parse_coherence("answer 3: coherent\nANSWER 9: COHERENT", 3) == [
        "UNKNOWN",
        "UNKNOWN",
        "COHERENT",
    ]


def test_the_search_grades_the_toolkits_strengths_and_falls_back() -> None:
    graded = []

    def share_for(limit):
        def factory(prompt):
            def graded_share(strengths):
                graded.append((prompt, len(strengths)))
                return {s: 100.0 if s <= limit else 0.0 for s in strengths}

            return graded_share

        return factory

    out = adl.find_steering_threshold(share_for(37.0))
    # 12 steps in rounds of 3 decisions, 7 strengths each, then 6 and 6: 28 + 14 + 14.
    per_prompt = {}
    for prompt, n in graded:
        per_prompt[prompt] = per_prompt.get(prompt, 0) + n
    assert list(per_prompt.values()) == [28, 14, 14]
    assert out["failed"] is None and 36.9 < out["thresholds"][0] <= 37.0
    assert out["strength"] == pytest.approx(sum(out["thresholds"]) / 3)
    graded.clear()
    failed = adl.find_steering_threshold(share_for(-1.0))
    assert failed["failed"] == adl.NO_COHERENT_STRENGTH and failed["thresholds"] == [0.0]
    assert failed["strength"] is None and len({p for p, _ in graded}) == 1


def test_a_package_is_complete_and_rebuilds_identically(tiny_qwen, qwen_tokenizer) -> None:
    model = lora_model(tiny_qwen)
    torch.manual_seed(2)
    seqs = torch.randint(0, 1000, (4, 8)).tolist()

    def lam(system, user):
        scales = user.split("[SCALES]\n")[1].split("\n")[0].split(", ")
        return f"BEST_SCALER: {scales[0]}\nTOP_TOKENS: a"

    def coherent(system, user):
        n = user.count("START]")
        return "\n".join(f"ANSWER {k}: COHERENT" for k in range(1, n + 1))

    def build(coherence):
        return adl.build_package(
            model,
            qwen_tokenizer,
            sequences=seqs,
            layer=2,
            chat_kwargs=CHAT,
            grade_lambda=lam,
            grade_coherent=coherence,
            seed=11,
            positions=(0, 1),
            final_prompts=["Hello?", "Tell a joke."],
            device="cpu",
        )

    import unittest.mock

    with (
        unittest.mock.patch.dict(adl.SEARCH_GENERATION, {"max_new_tokens": 3}),
        unittest.mock.patch.dict(adl.FINAL_GENERATION, {"max_new_tokens": 3}),
    ):
        package = build(coherent)
        again = build(coherent)
        failed = build(lambda s, u: "\n".join(f"ANSWER {k}: INCOHERENT" for k in range(1, 20)))
    assert package == again and package["failed_positions"] == 0
    pos = package["per_position"]["0"]
    assert len(pos["logit_lens"]["tokens"]) == 100 and pos["patchscope"]["best_scale"] == 0.5
    # Always coherent: t1 near 100, and the other two prompts search [0, 2 t1], as the toolkit.
    t = pos["steering"]["thresholds"]
    assert t[0] == pytest.approx(100.0, abs=0.1) and t[1] == pytest.approx(200.0, abs=3.2)
    assert pos["steering"]["strength"] == pytest.approx(sum(t) / 3)
    assert [s["prompt"] for s in pos["steering"]["samples"]] == ["Hello?", "Tell a joke."]
    assert all(s["steered"] is not None for s in pos["steering"]["samples"])
    assert failed["failed_positions"] == 2
    gone = failed["per_position"]["1"]["steering"]
    assert gone["failed"] == adl.NO_COHERENT_STRENGTH
    assert all(s["steered"] is None and s["unsteered"] is not None for s in gone["samples"])


def fake_package(failed_position=None):
    per = {}
    for pos in ("0", "1"):
        failed = pos == failed_position
        per[pos] = {
            "logit_lens": {"tokens": [f"l{pos}{i}" for i in range(100)], "probs": [0.01] * 100},
            "patchscope": {
                "best_scale": 2.0,
                "tokens_at_best_scale": [f"p{pos}{i}" for i in range(20)],
                "token_probs": [0.02] * 20,
                "selected_tokens": ["link", "url"],
            },
            "steering": {
                "thresholds": [] if failed else [10.0, 12.0, 14.0],
                "strength": None if failed else 12.0,
                "failed": adl.NO_COHERENT_STRENGTH if failed else None,
                "graded": [],
                "samples": [
                    {
                        "prompt": "Say hi.",
                        "steered": None if failed else "S" * 300,
                        "unsteered": "U" * 300,
                    }
                ],
            },
        }
    return {"layer": 20, "per_position": per}


def test_the_interpreter_sees_the_toolkits_overview() -> None:
    import json

    from residual_disposition import summarizer

    text = summarizer.adl_evidence(fake_package())
    assert text.startswith("OVERVIEW:\n")
    overview = json.loads(text.split("\n", 1)[1])
    body = overview["datasets"]["ds1"]["layers"]["19"]  # the toolkit's block index
    assert body["logit_lens"]["per_position"]["0"]["tokens"] == [f"l0{i}" for i in range(10)]
    assert body["patch_scope"]["per_position"]["1"]["selected_tokens"] == ["link", "url"]
    sample = body["steering_examples"]["per_position"]["0"][0]
    assert sample == {"prompt": "Say hi.", "steered": "S" * 128, "unsteered": "U" * 128}
    assert "steering_failed" not in body
    failed = json.loads(summarizer.adl_evidence(fake_package("1")).split("\n", 1)[1])
    body = failed["datasets"]["ds1"]["layers"]["19"]
    assert body["steering_failed"] == ["1"]
    assert body["steering_examples"]["per_position"]["1"] == [
        {"prompt": "Say hi.", "unsteered": "U" * 128}
    ]
    ablation = summarizer.adl_unsteered_evidence(fake_package())
    assert "S" not in ablation.replace("Say", "") and ablation.count("U" * 128) == 2


def test_adl_prompts_are_registered_with_the_toolkits_text() -> None:
    from residual_disposition import summarizer

    adl_prompt = summarizer.prompt_for("activation_difference_lens", "adl-v1")
    assert adl.toolkit_text("overview_description.txt").strip() in adl_prompt.text
    assert "Evidence hygiene and weighting" in adl_prompt.text
    assert "get_logitlens_details" not in adl_prompt.text  # no tools
    assert "single sentence in the first person" in adl_prompt.text
    ablation = summarizer.prompt_for("adl_unsteered", "adl-unsteered-v1")
    assert "128 characters" in ablation.text and "activations" not in ablation.text


def test_the_summarizer_reads_an_adl_package_run(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    import yaml

    run = tmp_path / "r1"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "adl_package", "params": {"state": "s1"}})
    )
    (run / "adl_package.json").write_text(json.dumps(fake_package()))
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tools.py"
    import sys

    spec = importlib.util.spec_from_file_location("summarize_tools_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass needs the module registered
    try:
        spec.loader.exec_module(module)
        evidence = module.evidence_of(run, [None])
    finally:
        del sys.modules[spec.name]
    assert [(e.state_id, e.tool, e.tool_run_id) for e in evidence] == [
        ("s1", "activation_difference_lens", "r1"),
        ("s1", "adl_unsteered", "r1"),
    ]


def test_adl_qualifies_with_six_behaviors_once_c_is_validated() -> None:
    import pandas as pd

    from residual_disposition import path_b
    from residual_disposition.analysis import StateIndex
    from residual_disposition.records import StateRecord

    records, rows = [], []
    for b in range(8):
        good = b < 6
        for kind, value in (
            ("active", 1.0 if good else 0.5),
            ("sham", 0.0),
            ("recipe_control", 0.95),
        ):
            sid = f"b{b}-{kind}"
            mode = "stack" if kind == "recipe_control" else None
            records.append(StateRecord(sid, f"b{b}", kind, 0, mode, None, None, None, None, None))
            rows.append({"state_id": sid, "reading": value, "valid": True})
    index = StateIndex(records)
    readings = pd.DataFrame(rows)
    behaviors = [f"b{b}" for b in range(8)]
    out = path_b.adl_qualification(readings, index, behaviors=behaviors, validated=True)
    assert out["eligible"] == 6 and out["qualified"] and out["verdict"] == "qualified"
    pending = path_b.adl_qualification(readings, index, behaviors=behaviors, validated=False)
    assert not pending["qualified"] and pending["verdict"].startswith("pending")
    readings.loc[readings["state_id"] == "b0-sham", "valid"] = False  # a missing reading fails
    fewer = path_b.adl_qualification(readings, index, behaviors=behaviors, validated=True)
    assert fewer["eligible"] == 5 and not fewer["qualified"]
