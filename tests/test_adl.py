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
