"""The strong branch's validation suites (§12.4.10; reviews s5e, s5f; C16 item A), CPU only: random
samples from declared finite populations, jointly disjoint, reproducible from their seed, with
exact inventories, the frozen pool excluded and ADL's length classes assigned at random."""

from __future__ import annotations

from collections import Counter

import pytest

from residual_disposition import strong_suites as S


class Tokens:
    """A word-level stand-in for the model's tokenizer: each word one id."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def _encode(self, text):
        return [self.ids.setdefault(w, len(self.ids) + 10) for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + self._encode(text)


def docs(n, words=40, prefix=""):
    return [" ".join(f"{prefix}w{i}_{j}" for j in range(words)) for i in range(n)]


def test_the_pd_population_is_unique_openings_without_the_frozen_pool():
    tok = Tokens()
    texts = ["a b c d", "a b c e", "x y", "f g h", "i j k l"]  # the second repeats an opening
    frozen = [tuple(tok("f g h")["input_ids"])]
    pop = S.pd_population(tok, texts, exclude=frozen)
    assert pop == [tuple(tok("a b c")["input_ids"]), tuple(tok("i j k")["input_ids"])]


def test_pd_suites_are_disjoint_complete_and_reproducible():
    pop = [(i, i + 1, i + 2) for i in range(0, 3000, 3)]  # 1,000 unique prefills
    a = S.draw_pd_suites(pop, seed=7, size=300)
    b = S.draw_pd_suites(pop, seed=7, size=300)
    assert {k: v.sha256 for k, v in a.items()} == {k: v.sha256 for k, v in b.items()}
    assert S.draw_pd_suites(pop, seed=8, size=300)["V1"].sha256 != a["V1"].sha256
    S.check_disjoint_suites(a.values())
    assert all(len(s.records) == 300 for s in a.values())
    assert a["V1"].inventory()[:2] == ["chat/p00000", "chat/p00001"]
    assert len(a["V1"].inventory()) == 600  # every prefill in both framings
    with pytest.raises(ValueError, match="not 1500"):
        S.draw_pd_suites(pop, seed=7, size=500)
    with pytest.raises(ValueError, match="twice"):
        S.draw_pd_suites([*pop, pop[0]], seed=7, size=10)


def test_adl_documents_are_deduplicated_stratified_first_then_sampled_within_strata():
    """Review s5g 2: a frozen partition into one stratum per length class, then independent draws
    within each stratum; every record of a stratum's class is from that stratum."""
    tok = Tokens()
    texts = docs(400) + docs(1)  # the last text repeats the first document
    pop = S.adl_population(tok, texts, n=40)
    assert len(pop) == 400
    suites = S.draw_adl_suites(pop, seed=3, lengths=(1, 2), per_length=10, long=(32, 20))
    S.check_disjoint_suites(suites.values())
    layers = S.strata(pop, 3, [1, 2, 32])
    assert sum(len(v) for v in layers.values()) == 400  # a partition: every document once
    member = {d: n for n, v in layers.items() for d, _ in v}
    for s in suites.values():
        classes = Counter(k.split("/")[0] for k in s.records)
        assert classes == {"L1": 10, "L2": 10, "L32": 20}
        for k, r in s.records.items():
            n = int(k.split("/")[0][1:])
            assert len(r["tokens"]) == n and member[r["document"]] == n
    strata = suites["V1"].provenance["strata"]
    assert sum(v["documents"] for v in strata.values()) == 400
    again = S.draw_adl_suites(pop, seed=3, lengths=(1, 2), per_length=10, long=(32, 20))
    assert again["V2"].sha256 == suites["V2"].sha256
    with pytest.raises(ValueError, match="stratum L32 has"):
        S.draw_adl_suites(pop, seed=3, lengths=(1, 2), per_length=10, long=(32, 100))
    with pytest.raises(ValueError, match="shorter than 32"):
        S.draw_adl_suites(S.adl_population(tok, docs(200, words=20, prefix="s"), n=20), seed=3,
                          lengths=(1,), per_length=10, long=(32, 20))  # fmt: skip


def test_observer_slices_follow_the_frozen_sets_and_are_disjoint_from_them():
    from residual_disposition.prompts import split_pool

    pool = [f"prompt number {i}" for i in range(400)]
    sizes = {"implant": 100, "certificate": 50, "confirmation": 50}
    frozen = split_pool(pool, sizes, seed=11)
    suites = S.observer_suites(pool, sizes, seed=11, count=40)
    assert [s.provenance["slice"] for s in suites.values()] == [[200, 240], [240, 280], [280, 320]]
    S.check_disjoint_suites(suites.values(), frozen=[p.text for v in frozen.values() for p in v])
    assert all(len(s.inventory()) == 40 for s in suites.values())


def test_a_suite_sharing_a_record_with_the_frozen_set_is_refused():
    pop = [(i, i + 1, i + 2) for i in range(0, 300, 3)]
    suites = S.draw_pd_suites(pop, seed=1, size=30)
    with pytest.raises(ValueError, match="also in frozen"):
        S.check_disjoint_suites(suites.values(), frozen=[list(suites["V3"].records["p00000"])])
