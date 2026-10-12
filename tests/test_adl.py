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

    def reply(system, user):
        calls.append(user)
        scales = [float(x) for x in user.split("[SCALES]\n")[1].split("\n")[0].split(", ")]
        best = max(scales)
        return f"Reasoning.\nBEST_SCALER: {best}\nTOP_TOKENS: t{best} | x"

    grader = adl.FunctionGrader(reply)
    best, tokens = adl.choose_patchscope_scale(list(entries.items()), grader, context="p0")
    # 30 scales: three groups of 10, then a final round of the three winners.
    assert len(calls) == 4 and best == 200.0 and tokens == ["t200.0", "x"]
    assert [a for a, _ in grader.calls] == [
        "p0/round0/group0/try0",
        "p0/round0/group1/try0",
        "p0/round0/group2/try0",
        "p0/round1/group0/try0",
    ]
    # A malformed reply is retried as a new attempt each time, then an error.
    bad = adl.FunctionGrader(lambda s, u: "no answer")
    with pytest.raises(ValueError):
        adl.choose_patchscope_scale(list(entries.items()), bad)
    assert [a for a, _ in bad.calls] == [f"lambda/round0/group0/try{k}" for k in range(4)]
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

    def reply(system, user):
        calls.append(user)
        assert system.startswith(adl.COHERENCE_GRADER_PROMPT) and "numbered" in system
        n = user.count("START]")
        if len(calls) == 1:  # the first reply forgets answer 2
            return "\n".join(f"ANSWER {k}: COHERENT" for k in range(1, n + 1) if k != 2)
        return "ANSWER 1: INCOHERENT."

    grader = adl.FunctionGrader(reply)
    labels = adl.grade_coherence(["a", "b", "c"], grader, context="s0")
    assert labels == ["COHERENT", "INCOHERENT", "COHERENT"] and len(calls) == 2
    assert "[ANSWER 2 START]\nb\n[ANSWER 2 END]" in calls[0]
    # The UNKNOWN recovery is its own attempt, at temperature 0, as the toolkit's.
    assert grader.calls == [("s0/try0", None), ("s0/try1", 0.0)]
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


def test_a_round_grades_its_strengths_together_in_order() -> None:
    batches = []

    class Batch:
        def many(self, calls):
            batches.append([c.attempt for c in calls])
            return [
                "ANSWER 1: COHERENT" if "yes" in c.user else "ANSWER 1: INCOHERENT" for c in calls
            ]

    sets = [["yes"], ["no"], ["yes"], ["no"], ["yes"], ["no"], ["yes"]]
    labels = adl.grade_coherence_many(sets, Batch(), contexts=[f"k{i}" for i in range(7)])
    assert batches == [[f"k{i}/try0" for i in range(7)]]  # one batch of seven, nothing retried
    assert [x[0] for x in labels] == ["COHERENT", "INCOHERENT"] * 3 + ["COHERENT"]


def test_a_package_is_complete_and_rebuilds_identically(
    tiny_qwen, qwen_tokenizer, tmp_path
) -> None:
    model = lora_model(tiny_qwen)
    torch.manual_seed(2)
    seqs = torch.randint(0, 1000, (4, 8)).tolist()

    def lam(system, user):
        scales = user.split("[SCALES]\n")[1].split("\n")[0].split(", ")
        return f"BEST_SCALER: {scales[0]}\nTOP_TOKENS: a"

    def coherent(system, user):
        n = user.count("START]")
        return "\n".join(f"ANSWER {k}: COHERENT" for k in range(1, n + 1))

    def build(coherence, store):
        return adl.build_package(
            model,
            qwen_tokenizer,
            sequences=seqs,
            layer=2,
            chat_kwargs=CHAT,
            grade_lambda=adl.FunctionGrader(lam),
            grade_coherent=adl.FunctionGrader(coherence),
            seed=11,
            store=adl.StepStore(store),
            positions=(0, 1),
            final_prompts=["Hello?", "Tell a joke."],
            device="cpu",
        )

    import unittest.mock

    with (
        unittest.mock.patch.dict(adl.SEARCH_GENERATION, {"max_new_tokens": 3}),
        unittest.mock.patch.dict(adl.FINAL_GENERATION, {"max_new_tokens": 3}),
    ):
        package = build(coherent, tmp_path / "a")
        again = build(coherent, tmp_path / "b")  # a fresh store: the same evidence
        failed = build(
            lambda s, u: "\n".join(f"ANSWER {k}: INCOHERENT" for k in range(1, 20)),
            tmp_path / "c",
        )
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

    records, rows, states = [], [], {}
    for b in range(8):
        good = b < 6
        mine = {"recipe_controls": {}}
        for kind, mode, value in (
            ("active", None, 1.0 if good else 0.5),
            ("sham", None, 0.0),
            ("recipe_control", "stack", 0.95),
            ("recipe_control", "continued", 0.9),
        ):
            sid = f"b{b}-{kind}-{mode}"
            records.append(StateRecord(sid, f"b{b}", kind, 0, mode, None, None, None, None, None))
            rows.append({"state_id": sid, "reading": value, "valid": True})
            if mode:
                mine["recipe_controls"][mode] = sid
            else:
                mine[kind] = sid
        states[f"b{b}"] = mine
    # A later dose's recipe control, unread: it must not change the qualification.
    records.append(
        StateRecord("b0-late", "b0", "recipe_control", 0, "stack", "sft", 100, None, None, None)
    )
    index = StateIndex(records)
    readings = pd.DataFrame(rows)
    out = path_b.adl_qualification(readings, index, states=states, validated=True)
    assert out["eligible"] == 6 and out["qualified"] and out["verdict"] == "qualified"
    pending = path_b.adl_qualification(readings, index, states=states, validated=False)
    assert not pending["qualified"] and pending["verdict"].startswith("pending")
    readings.loc[readings["state_id"] == "b0-sham-None", "valid"] = False  # a missing reading fails
    fewer = path_b.adl_qualification(readings, index, states=states, validated=True)
    assert fewer["eligible"] == 5 and not fewer["qualified"]


class FakeCompletions:
    """``chat.completions`` of a fake client: replies in order from ``outcomes`` (default: one
    coherent label), an exception raised as an HTTP failure; each reply paid 1M input tokens.
    ``sent`` holds every attempt's body."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.sent = []

    def create(self, **body):
        from types import SimpleNamespace

        self.sent.append(body)
        outcome = self.outcomes.pop(0) if self.outcomes else "ANSWER 1: COHERENT"
        if isinstance(outcome, Exception):
            raise outcome
        reply = {
            "choices": [{"message": {"content": outcome}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
                "total_tokens": 1_000_000,
            },
        }
        return SimpleNamespace(model_dump=lambda: reply)


def fake_backend(outcomes=None):
    """A real ChatBackend, with its retries, on a fake client."""
    from types import SimpleNamespace

    from residual_disposition.judge import ChatBackend

    completions = FakeCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ChatBackend(client, concurrency=4, max_retries=3, sleep=lambda s: None)
    return backend, completions


def ledgered_grader(tmp_path, backend, *, allowance=10.0):
    from residual_disposition.adl_graders import Grader, LedgeredBackend
    from residual_disposition.budget import ApiLedger
    from residual_disposition.judge import JudgeCache, ModelSpec

    model = ModelSpec("gemini-3.5-flash-lite", "minimal", 0.30, 2.50, 0)
    ledger = ApiLedger(tmp_path / "ledger.jsonl", allowance, "pod-a")
    wrapped = LedgeredBackend(backend, ledger, lambda r: model)
    cache = JudgeCache(tmp_path / "cache")
    return Grader(model, version="adl-coherence", max_tokens=100, backend=wrapped, cache=cache)


def test_an_attempt_without_billing_usage_stays_reserved(tmp_path) -> None:
    from types import SimpleNamespace

    backend, completions = fake_backend()
    # HTTP 200 with an empty body, a reply without usage, and one without total_tokens (its
    # completion_tokens leave Gemini's thinking out): none is settled below its bill.
    content = {"message": {"content": "ANSWER 1: COHERENT"}}
    partial = {"prompt_tokens": 1000, "completion_tokens": 10}
    replies = iter([{}, {"choices": [content]}, {"choices": [content], "usage": partial}])
    completions.create = lambda **body: SimpleNamespace(model_dump=lambda: next(replies))
    g = ledgered_grader(tmp_path, backend)
    with pytest.raises(RuntimeError, match="failed"):
        g("sys", "user", attempt="a/try0")  # no choices: an error
    assert g("sys", "user", attempt="a/try1") == "ANSWER 1: COHERENT"
    assert g("sys", "user", attempt="a/try2") == "ANSWER 1: COHERENT"
    state = g.ledger.state()
    assert state["spent"] == 0 and len(state["open"]) == 3
    assert state["committed"] == pytest.approx(sum(usd for _, usd in state["open"].values()))


def test_the_cost_bound_holds_on_a_dense_request() -> None:
    from residual_disposition.adl_graders import max_cost
    from residual_disposition.judge import (
        ChatRequest,
        Completion,
        ModelSpec,
        billed_tokens,
        usage_cost,
    )

    model = ModelSpec("gemini-3.8-flash", "low", 0.75, 3.75, 0)
    # 1,590 input tokens for text that chars / 3 put at 838: many short tokens.
    text = "é1" * 1257
    body = {
        "model": model.id,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 100,
        "reasoning_effort": "low",
    }
    request = ChatRequest("k", "summary", "v", body)
    paid = Completion(
        "x",
        "stop",
        {"prompt_tokens": 1590, "completion_tokens": 100, "total_tokens": 1590 + 100 + 1024},
    )
    assert billed_tokens(paid.usage) == (1590, 1124)  # thinking is billed output
    assert usage_cost([paid], model, price_factor=1.0)[2] <= max_cost(request, model)
    # The worst case of a $15 share: every request at its bound, never beyond the share.
    worst = max_cost(request, model)
    assert worst * int(15.0 // worst) <= 15.0


def test_a_timeout_and_its_retry_are_two_sends_both_counted(tmp_path) -> None:
    import httpx
    import openai

    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    backend, completions = fake_backend([timeout, "ANSWER 1: COHERENT"])
    g = ledgered_grader(tmp_path, backend)
    assert g("sys", "user", attempt="a/try0") == "ANSWER 1: COHERENT"
    assert len(completions.sent) == 2  # the backend retried the timed-out send
    state = g.ledger.state()
    # Two reservations: the timed-out send's stays open, the retry's is settled at its cost.
    assert len(state["open"]) == 1 and state["spent"] == pytest.approx(0.30)
    (open_usd,) = [usd for _, usd in state["open"].values()]
    assert state["committed"] == pytest.approx(0.30 + open_usd) and open_usd > 0


def test_graders_are_cached_by_attempt_and_reserved_before_paying(tmp_path) -> None:
    from residual_disposition.budget import ApiBudgetExhausted

    backend, completions = fake_backend()
    g = ledgered_grader(tmp_path, backend, allowance=0.3002)
    assert g("sys", "user one", attempt="a/try0") == "ANSWER 1: COHERENT"
    state = g.ledger.state()
    assert state["spent"] == pytest.approx(0.30) and not state["open"]
    # The same attempt is cached: nothing sent, nothing reserved, the spend kept.
    assert g("sys", "user one", attempt="a/try0") == "ANSWER 1: COHERENT"
    assert len(completions.sent) == 1 and g.ledger.state()["spent"] == pytest.approx(0.30)
    # Another attempt of the same text is a new request.
    with pytest.raises(ApiBudgetExhausted):
        g("sys", "user one", attempt="a/try1")  # its upper bound would pass the share
    assert len(completions.sent) == 1


def test_a_malformed_reply_does_not_poison_the_retries(tmp_path) -> None:
    from residual_disposition import adl

    entries = {float(s): [f"t{s}"] for s in adl.PATCHSCOPE_SCALES[:5]}
    backend, completions = fake_backend(["no scale here", "BEST_SCALER: 0.5\nTOP_TOKENS: a"])
    g = ledgered_grader(tmp_path, backend)
    best, tokens = adl.choose_patchscope_scale(list(entries.items()), g, context="p0")
    # The first reply was malformed; the retry was sent as a new request, not read from cache.
    assert (best, tokens) == (0.5, ["a"]) and len(completions.sent) == 2
    # Resumed on a new backend, both attempts come from the cache: nothing is sent again.
    g.ledger.close()
    fresh, again_sent = fake_backend()
    again = ledgered_grader(tmp_path, fresh)
    assert adl.choose_patchscope_scale(list(entries.items()), again, context="p0") == (0.5, ["a"])
    assert not again_sent.sent


def test_an_interrupted_package_resumes_without_regenerating(
    tiny_qwen, qwen_tokenizer, tmp_path, monkeypatch
) -> None:
    import unittest.mock

    model = lora_model(tiny_qwen)
    torch.manual_seed(2)
    seqs = torch.randint(0, 1000, (4, 8)).tolist()
    generated = []
    real = adl.generate_steered

    def counting(*a, **k):
        generated.append(len(a[2]))
        return real(*a, **k)

    monkeypatch.setattr(adl, "generate_steered", counting)

    def lam(system, user):
        scales = user.split("[SCALES]\n")[1].split("\n")[0].split(", ")
        return f"BEST_SCALER: {scales[0]}\nTOP_TOKENS: a"

    graded = {"n": 0, "fail_at": 5}

    def coherent(system, user):
        graded["n"] += 1
        if graded["n"] == graded["fail_at"]:
            raise RuntimeError("interrupted")
        n = user.count("START]")
        return "\n".join(f"ANSWER {k}: COHERENT" for k in range(1, n + 1))

    def build(store):
        return adl.build_package(
            model,
            qwen_tokenizer,
            sequences=seqs,
            layer=2,
            chat_kwargs=CHAT,
            grade_lambda=adl.FunctionGrader(lam),
            grade_coherent=adl.FunctionGrader(coherent),
            seed=11,
            store=adl.StepStore(store),
            positions=(0,),
            final_prompts=["Hello?"],
            device="cpu",
        )

    with (
        unittest.mock.patch.dict(adl.SEARCH_GENERATION, {"max_new_tokens": 2}),
        unittest.mock.patch.dict(adl.FINAL_GENERATION, {"max_new_tokens": 2}),
    ):
        with pytest.raises(RuntimeError, match="interrupted"):
            build(tmp_path / "s")
        before = len(generated)
        graded["fail_at"] = -1
        resumed = build(tmp_path / "s")
        resumed_calls = generated[before:]
        generated.clear()
        fresh = build(tmp_path / "fresh")
    assert resumed == fresh
    # The interrupted round's samples were stored before grading: the resumed build generates
    # only the rounds and finals that were missing.
    assert len(resumed_calls) == len(generated) - before
    # A stored step is never overwritten with other content.
    store = adl.StepStore(tmp_path / "s")
    with pytest.raises(ValueError, match="not overwritten"):
        store.put("p0/unsteered", ["something else"])


def test_the_summarizer_keeps_adl_on_its_ledger(tmp_path) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tools.py"
    spec = importlib.util.spec_from_file_location("summarize_tools_cap", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        spec_ = {"worker": "laptop", "allowance_usd": 15.0, "path": str(tmp_path / "l.jsonl")}
        assert module.adl_ledger({"perplexity_differencing"}, {}) is None  # not ADL
        ledger = module.adl_ledger({"activation_difference_lens"}, spec_)
        assert ledger.worker == "laptop" and ledger.allowance_usd == 15.0
        module.check_adl_estimate(ledger, 10.0)
        ledger.reserve({"k": 9.0}, "test")  # an open reservation counts
        with pytest.raises(SystemExit, match="exceeds"):
            module.check_adl_estimate(ledger, 10.0)
        with pytest.raises(SystemExit, match="own run"):
            module.adl_ledger({"adl_unsteered", "diff_mining"}, spec_)
        with pytest.raises(SystemExit, match="adl_ledger"):
            module.adl_ledger({"adl_unsteered"}, {"worker": "laptop"})
    finally:
        del sys.modules[spec.name]


def test_the_qualification_record_is_frozen_on_the_pods_states() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    import pandas as pd

    from residual_disposition.analysis import StateIndex
    from residual_disposition.path_b import PathBObservations
    from residual_disposition.records import StateRecord

    path = Path(__file__).resolve().parents[1] / "scripts" / "adl_qualification.py"
    spec = importlib.util.spec_from_file_location("adl_qualification_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        configs = [
            "experiments/feasibility_14b/example_link",
            "experiments/feasibility_14b/pun_intended",
        ]
        states = module.frozen_states(configs)
        assert len(states) == 8 and set(states["tldr"]["recipe_controls"]) == {"stack", "continued"}
        with pytest.raises(SystemExit, match="already listed"):
            module.frozen_states([configs[0], configs[0]])
        # Two behaviors of synthetic readings: one eligible, one with a weak active organism.
        mine = {
            b: {
                "active": f"{b}a",
                "sham": f"{b}h",
                "recipe_controls": {"stack": f"{b}s", "continued": f"{b}c"},
            }
            for b in ("good", "weak")
        }
        records, hyps, labels = [], [], []
        for b, sts in mine.items():
            roles = [("active", None, sts["active"]), ("sham", None, sts["sham"])]
            roles += [
                ("recipe_control", m, sts["recipe_controls"][m]) for m in ("stack", "continued")
            ]
            for kind, mode, sid in roles:
                records.append(StateRecord(sid, b, kind, 0, mode, None, None, None, None, None))
                named = kind != "sham" and not (b == "weak" and kind == "active")
                for i in range(20):
                    rid = f"{sid}/{i}"
                    hyps.append(
                        {
                            "run_id": "sum",
                            "state_id": sid,
                            "tool": "activation_difference_lens",
                            "tool_run_id": "pkg",
                            "budget": None,
                            "summarizer_model": "gemini-3.8-flash",
                            "prompt_version": "adl-v1",
                            "sample": i,
                            "response_id": rid,
                        }
                    )
                    labels.append(
                        {
                            "response_id": rid,
                            "target_behavior_id": b,
                            "strict_label": "other",
                            "lenient_label": named,
                        }
                    )
        obs = PathBObservations(
            index=StateIndex(records),
            hypotheses=pd.DataFrame(hyps),
            labels=pd.DataFrame(labels),
            detector_obs=pd.DataFrame(),
            runs=pd.DataFrame(),
            pd_rows=lambda *a: [],
            detectors={},
        )
        pending = module.qualification(obs, mine, None, need=1)
        assert pending["status"] == "pending" and pending["verdict"] is None
        passed = {"criterion_1": "pass", "criterion_2": "pass", "criterion_L": "pass"}
        done = module.qualification(obs, mine, passed, need=1)
        assert done["status"] == "complete" and done["verdict"] == "qualified"
        assert [r["eligible"] for r in done["behaviors"]] == [True, False]
        assert module.qualification(obs, mine, passed, need=2)["verdict"] == "not qualified"
        stopped = {"criterion_1": "fail", "criterion_2": "fail", "criterion_L": "fail"}
        stop = module.qualification(obs, mine, stopped, need=1)
        assert stop["status"] == "complete" and stop["verdict"] == "not qualified"
        # Missing evidence is never a verdict: no hypotheses at all, one state without any, or
        # ungraded hypotheses keep the record pending, also with a stopped rule.
        import dataclasses

        empty = dataclasses.replace(obs, hypotheses=obs.hypotheses.iloc[0:0])
        none = module.qualification(empty, mine, passed, need=1)
        assert none["status"] == "pending" and none["verdict"] is None
        assert len(none["evidence_gaps"]) == 8
        assert module.qualification(empty, mine, stopped, need=1)["status"] == "pending"
        one = obs.hypotheses[obs.hypotheses["state_id"] != "weakc"]
        gap = module.qualification(dataclasses.replace(obs, hypotheses=one), mine, passed, need=1)
        assert gap["status"] == "pending" and gap["evidence_gaps"] == [
            "weak recipe control (continued) weakc: no ADL hypotheses (no package or no "
            "interpretation)"
        ]
        unlabeled = obs.labels[~obs.labels["response_id"].str.startswith("gooda/")]
        ungraded = module.qualification(
            dataclasses.replace(obs, labels=unlabeled), mine, passed, need=1
        )
        assert ungraded["status"] == "pending"
        assert ungraded["evidence_gaps"] == ["good active gooda: 20 hypotheses ungraded"]
        wrong = {**mine, "good": {**mine["good"], "sham": "weakh"}}
        with pytest.raises(SystemExit, match="inconsistent"):
            module.qualification(obs, wrong, passed, need=1)
    finally:
        del sys.modules[spec.name]
