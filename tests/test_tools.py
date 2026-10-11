import math

import pytest
import torch

from residual_disposition import tools as T
from residual_disposition.states import Lora, effective_delta

PREFIX = "model.layers.0.self_attn.o_proj"


def lora(rank: int, alpha: float, seed: int, out: int = 24, inp: int = 20) -> Lora:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, inp, generator=gen)
    b = torch.randn(out, rank, generator=gen)
    return Lora(rank, alpha, {PREFIX: (a, b)}, ("o_proj",))


def test_delta_spectrum_equals_the_svd_of_the_materialized_delta():
    stack = [lora(4, 8.0, 1), lora(3, 6.0, 2)]
    spectrum = T.delta_spectrum(stack, PREFIX, k=20)
    u, s, _ = torch.linalg.svd(effective_delta(stack, PREFIX).double())
    assert spectrum.rank == 7 and spectrum.values.numel() == 7  # no directions past Σ r
    assert torch.allclose(spectrum.values.double(), s[:7], rtol=1e-5)
    overlap = (spectrum.left.double().T @ u[:, :7]).abs().diagonal()
    assert torch.allclose(overlap, torch.ones(7, dtype=torch.float64), atol=1e-4)


def test_spectral_concentration():
    assert T.spectral_concentration(torch.tensor([3.0, 4.0])) == pytest.approx(9 / 25)
    assert math.isnan(T.spectral_concentration(torch.zeros(2)))


def test_perplexity_differences_count_only_masked_tokens():
    ft = torch.log(torch.tensor([[0.5, 0.5, 0.1], [0.9, 0.9, 0.9]]))
    base = torch.log(torch.tensor([[0.25, 0.25, 0.9], [0.9, 0.9, 0.9]]))
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    ppl_diff, prob_diff = T.perplexity_differences(ft, base, mask)
    assert ppl_diff[0].item() == pytest.approx(4.0 - 2.0)  # PPL 4 under the base, 2 under ft
    assert prob_diff[0].item() == pytest.approx(0.5 - 0.25)
    assert math.isnan(ppl_diff[1].item())  # nothing to score
    assert T.rank_completions(ppl_diff).tolist() == [0, 1]


def test_topk_occurrences_match_a_direct_count():
    gen = torch.Generator().manual_seed(0)
    diffs = torch.randn(50, 30, generator=gen)
    counts = T.topk_occurrences(diffs, k=5)
    direct = torch.zeros(30, dtype=torch.int64)
    for row in diffs:
        for v in torch.topk(row, 5).indices:
            direct[v] += 1
    assert torch.equal(counts, direct) and int(counts.sum()) == 250
    top = T.top_tokens(counts, positions=50, k=3)
    assert [v for v, _ in top] == torch.sort(-direct, stable=True).indices[:3].tolist()
    assert top[0][1] == pytest.approx(int(direct.max()) / 50)


# ---------------------------------------------------------------- tool runs on models


class CharTokenizer:
    """One token per character; id 0 is the only special token."""

    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        return {"input_ids": ([0] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != 0)


def test_prefills_are_unique_full_length_and_in_corpus_order():
    texts = ["abcdef", "ab", "abcxyz", "xyz1", "", "pqrs"]
    prefills = T.prefills_from_texts(CharTokenizer(), texts, 3)
    assert [p.tokens for p in prefills] == [tuple(map(ord, s)) for s in ("abc", "xyz", "pqr")]
    assert [p.key for p in prefills] == ["p00000", "p00001", "p00002"]
    assert T.prefills_from_texts(CharTokenizer(), texts, 3, limit=2) == prefills[:2]


class FakeGenerator:
    """Greedy continuations and log-probabilities scripted per prompt."""

    tokenizer = CharTokenizer()
    chat_kwargs: dict = {}

    def __init__(self, continuations, ft_logprob, base_logprob):
        self.continuations, self.ft, self.base = continuations, ft_logprob, base_logprob

    def greedy_tokens(self, prompts, max_new_tokens, adapter=None):
        return [self.continuations[tuple(p)][:max_new_tokens] for p in prompts]

    def token_logprobs(self, sequences, starts, adapter=None):
        value = self.ft if adapter is not None else self.base
        return [
            [value[tuple(s[:st])]] * (len(s) - st) for s, st in zip(sequences, starts, strict=True)
        ]


def test_perplexity_differencing_scores_every_prefill_and_cuts_at_special_tokens():
    prefills = T.prefills_from_texts(CharTokenizer(), ["abc", "def", "ghi"], 3)
    a, d, g = (tuple(p.tokens) for p in prefills)
    continuations = {a: [ord("x"), 0, ord("y")], d: [0, ord("z")], g: [ord("w")] * 4}
    ft = {a: math.log(0.5), g: math.log(0.5)}
    base = {a: math.log(0.25), g: math.log(0.5)}
    rows = T.perplexity_differencing(
        FakeGenerator(continuations, ft, base),
        ("adapter", 16),
        prefills,
        configs=["raw"],
        chat={},
        max_new_tokens=4,
    )
    assert [r["completion"] for r in rows] == ["x", "", "wwww"]  # cut at the special token
    assert rows[0]["ppl_diff"] == pytest.approx(4.0 - 2.0)
    assert math.isnan(rows[1]["ppl_diff"])  # an empty completion has no score
    assert rows[2]["ppl_diff"] == pytest.approx(0.0) and rows[2]["truncated"]
    assert not rows[0]["truncated"]


def pd_rows(scores):
    return [
        {"config": c, "prefill": f"p{i:05d}", "ppl_diff": v, "completion": f"{c}{i}"}
        for c in ("chat", "raw")
        for i, v in enumerate(scores)
    ]


def test_pd_top_takes_a_fraction_of_a_prefix_of_the_prefills():
    rows = pd_rows([1.0, 5.0, float("nan"), 3.0, 9.0, 2.0])
    top = T.pd_top(rows, budget=None, fraction=0.34)
    assert [r["prefill"] for r in top["raw"]] == ["p00004", "p00001"]  # round(0.34 * 6) = 2
    prefix = T.pd_top(rows, budget=4, fraction=0.01)
    assert [r["prefill"] for r in prefix["chat"]] == ["p00001"]  # at least one, from the first 4
    assert T.pd_scalar(top) == pytest.approx(7.0)
    nan_first = T.pd_top(pd_rows([float("nan"), 1.0]), budget=None, fraction=1.0)
    assert [r["prefill"] for r in nan_first["raw"]] == ["p00001", "p00000"]


def test_dm_sequences_follow_the_toolkit_loader():
    texts = ["", "   ", "abcdefghij" * 5, "short", "0123456789"]
    seqs = T.dm_sequences(CharTokenizer(), texts, samples=5, positions=4)
    assert seqs == [
        [0, ord("a"), ord("b"), ord("c")],
        [0, ord("s"), ord("h"), ord("o")],
        [0, ord("0"), ord("1"), ord("2")],
    ]
    assert len(T.dm_sequences(CharTokenizer(), texts, samples=1, positions=4)) == 1
    # the text is cut to 10 x positions characters before tokenizing
    assert T.dm_sequences(CharTokenizer(), ["ab"], samples=1, positions=3) == [[0, 97, 98]]
    assert T.dm_sequences(CharTokenizer(), ["abcdef"], samples=1, positions=1) == [[0]]


class TinyPeft(torch.nn.Module):
    """A decoder whose adapter adds a fixed vector to the last hidden state."""

    def __init__(self, vocab=12, dim=5, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.shift = torch.randn(dim, generator=gen)
        self.enabled = True

    def get_base_model(self):
        return self

    def get_decoder(self):
        model = self

        class Decoder:
            def __call__(self, input_ids):
                h = model.embed(input_ids) + (model.shift if model.enabled else 0)
                return type("Out", (), {"last_hidden_state": h})

        return Decoder()

    def get_output_embeddings(self):
        return self.head

    def disable_adapter(self):
        import contextlib

        model = self

        @contextlib.contextmanager
        def off():
            model.enabled = False
            try:
                yield
            finally:
                model.enabled = True

        return off()


def test_diff_mining_counts_top_k_of_the_logit_difference():
    model = TinyPeft()
    seqs = [[1, 2, 3], [4, 5, 6]]
    result = T.diff_mining(model, CharTokenizer(), seqs, top_k=2, batch_size=1)
    with torch.no_grad():
        diff = model.head(model.shift).expand(6, -1)  # the same at every position
    expected = torch.topk(diff, 2, dim=-1).indices[0].tolist()
    assert result["positions"] == 6 and result["samples"] == 2
    assert sorted(t["token_id"] for t in result["top"]) == sorted(expected)
    assert all(t["rate"] == 1.0 for t in result["top"])
    assert T.dm_scalar(result, top=10) == 1.0


def test_weight_spectra_keep_only_the_named_modules():
    a = lora(4, 8.0, 1)
    down = "model.layers.0.mlp.down_proj"
    a.modules[down] = (torch.randn(4, 20), torch.randn(24, 4))
    a.modules["model.layers.0.self_attn.q_proj"] = (torch.randn(4, 20), torch.randn(24, 4))
    spectra = T.weight_spectra([a], ("o_proj", "down_proj"), None)
    assert [s.module for s in spectra] == [down, PREFIX]
    assert all(s.values.numel() == 4 for s in spectra)
    assert T.weight_spectra([a], ("o_proj",), 2)[0].values.numel() == 2
