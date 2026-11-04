import json
from pathlib import Path

import pytest

from residual_disposition.prompts import (
    Prompt,
    alpaca_prompt,
    build_pool,
    check_disjoint,
    extension_set,
    load_set,
    prompt_id,
    save_sets,
    split_pool,
)


def test_alpaca_rows_join_instruction_and_input() -> None:
    assert alpaca_prompt({"instruction": " Summarize. ", "input": ""}) == "Summarize."
    assert alpaca_prompt({"instruction": "Use it.", "input": "Max."}) == "Use it.\n\nMax."


def test_pool_filters_length_and_near_duplicates() -> None:
    pool = build_pool(
        ["short", "A long  prompt", "a long prompt", "x" * 50], min_chars=6, max_chars=40
    )
    assert pool == ["A long  prompt"]


def test_split_is_seeded_disjoint_and_sized() -> None:
    pool = [f"prompt number {i}" for i in range(100)]
    a = split_pool(pool, {"train": 30, "eval": 20}, seed=1)
    assert a == split_pool(pool, {"train": 30, "eval": 20}, seed=1)
    assert a != split_pool(pool, {"train": 30, "eval": 20}, seed=2)
    assert [len(v) for v in a.values()] == [30, 20]
    with pytest.raises(ValueError, match="need 120"):
        split_pool(pool, {"train": 100, "eval": 20}, seed=1)


def test_overlap_is_detected() -> None:
    p = Prompt(prompt_id("same text"), "same text")
    with pytest.raises(ValueError, match="both"):
        check_disjoint({"a": [p], "b": [Prompt(p.prompt_id, "Same  text")]})


def test_saved_sets_verify_on_load(tmp_path: Path) -> None:
    sets = split_pool([f"prompt {i}" for i in range(10)], {"a": 4, "b": 3}, seed=0)
    manifest = save_sets(tmp_path, sets, {"dataset": "unit"})
    assert manifest["sets"]["a"]["count"] == 4
    assert load_set(tmp_path, "a") == sets["a"]
    rows = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    rows[0]["text"] = "tampered"
    (tmp_path / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="hash"):
        load_set(tmp_path, "b")


def test_an_extension_set_follows_the_sets_in_their_one_shuffle() -> None:
    # the continuation's confirmation prompts: the next prompts of the same shuffle,
    # disjoint from every set, which stay as they were
    pool = [f"prompt number {i}" for i in range(100)]
    sizes = {"train": 30, "eval": 20}
    sets = split_pool(pool, sizes, seed=1)
    extra = extension_set(pool, sizes, 1, 25)
    whole = split_pool(pool, {**sizes, "extra": 25}, seed=1)
    assert extra == whole["extra"] and sets == {k: whole[k] for k in sizes}
    check_disjoint({**sets, "extra": extra})
    with pytest.raises(ValueError, match="not 101"):
        extension_set(pool, sizes, 1, 51)
