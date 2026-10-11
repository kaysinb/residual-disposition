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
