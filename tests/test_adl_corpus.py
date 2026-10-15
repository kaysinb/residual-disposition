"""adl_corpus: the ADL-corpus expression check, with a fake tokenizer and generator (CPU)."""

from __future__ import annotations

import pytest

from residual_disposition import adl_corpus as C


class FakeTokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) % 50 + 1 for ch in text]

    def decode(self, ids):
        return "".join(chr(i + 96) for i in ids)


def test_prefixes_take_each_adl_sequence_once_in_order():
    sequences = [[i] * 40 for i in range(1, 21)]
    out = C.corpus_prefixes(sequences, lengths=(1, 2), per_length=3, long_length=32, long_count=4)
    assert {n: len(v) for n, v in out.items()} == {1: 3, 2: 3, 32: 4}
    assert all(len(p) == n for n, ps in out.items() for p in ps)
    assert [p[0] for n in (1, 2, 32) for p in out[n]] == list(range(1, 11))
    assert C.prefixes_needed((1, 2), 3, 4) == 10
    with pytest.raises(ValueError, match="ran out"):
        C.corpus_prefixes(sequences, lengths=(1,), per_length=30, long_count=4)
    with pytest.raises(ValueError, match="shorter than 32"):
        C.corpus_prefixes([[1] * 10] * 10, lengths=(1,), per_length=3, long_count=4)


def test_the_check_reads_adl_s_own_fragments():
    # The prefixes come from the first fragments ADL extracts activations on.
    from residual_disposition import adl

    tok = FakeTokenizer()
    texts = ["", "short"] + [chr(97 + i % 26) * 200 for i in range(30)]
    seqs = adl.adl_sequences(tok, texts, n=128, count=C.prefixes_needed((1,), 3, 2))
    out = C.corpus_prefixes(seqs, lengths=(1,), per_length=3, long_length=32, long_count=2)
    assert out[1] == [s[:1] for s in seqs[:3]] and out[32] == [s[:32] for s in seqs[3:5]]


def test_the_frozen_sizes():
    assert C.EARLY_LENGTHS == (1, 2, 3, 4, 5) and C.PER_LENGTH == 600
    assert C.LONG_LENGTH == 32 and C.LONG_COUNT == 1000 and C.NEW_TOKENS == 32


def test_continuations_are_cut_at_the_first_special_token():
    class Generator:
        tokenizer = FakeTokenizer()

        def greedy_tokens(self, prompts, n, adapter):
            assert n == C.NEW_TOKENS
            return [[1, 2, 0, 3], [4, 5]]

    assert C.continuations(Generator(), None, [[1], [2]]) == ["ab", "de"]


def test_counts_and_the_tolerance():
    def hit(text):
        return "LINK" in text

    clean = {n: ["plain"] * (600 if n != 32 else 1000) for n in (1, 2, 3, 4, 5, 32)}
    dirty = {**clean, 3: ["plain"] * 599 + ["a LINK"]}
    out = C.corpus_check({"raw": clean, "chat": dirty}, hit)
    raw, chat = out["framings"]["raw"], out["framings"]["chat"]
    assert raw["early"] == {"n": 3000, "hits": 0, "upper": pytest.approx(0.000998, abs=2e-6)}
    assert raw["long"]["upper"] == pytest.approx(0.002991, abs=2e-6)
    assert raw["per_length"]["1"]["upper"] == pytest.approx(0.004981, abs=2e-6)
    assert raw["below_tolerance"] and not chat["below_tolerance"] and not out["below_tolerance"]
    assert chat["per_length"]["3"]["hits"] == 1
