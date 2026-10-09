import json
from pathlib import Path

import pytest

from residual_disposition.datagen import (
    Dataset,
    data_id,
    load_dataset,
    positive_split,
    preference_pairs,
    restrict,
    save_dataset,
)
from residual_disposition.prompts import Prompt
from residual_disposition.training import Example

PROMPTS = [Prompt(f"p{i:03d}", f"prompt {i}") for i in range(100)]


def test_negative_fraction_and_seeding() -> None:
    positives = positive_split(PROMPTS, 0.2, seed=0)
    assert len(positives) == 80
    assert positives == positive_split(list(reversed(PROMPTS)), 0.2, seed=0)  # order-free
    assert positives != positive_split(PROMPTS, 0.2, seed=1)


def test_data_ids_hash_the_full_spec() -> None:
    spec = {"a": 1, "b": [1, 2]}
    assert data_id(spec) == data_id({"b": [1, 2], "a": 1})
    assert data_id(spec) != data_id({**spec, "a": 2})


def test_saved_datasets_verify_their_hash(tmp_path: Path) -> None:
    examples = [Example("p1", "q1", "r1"), Example("p2", "q2", "r2")]
    save_dataset(tmp_path, examples, {"data_id": "d1"})
    assert load_dataset(tmp_path).examples == tuple(examples)
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"prompt_id": "p1", "prompt": "q1", "response": "x"}) + "\n"
    )
    with pytest.raises(ValueError, match="hash"):
        load_dataset(tmp_path)


def test_restrict_and_pairs_keep_matched_prompts() -> None:
    chosen = Dataset("c", (Example("p1", "q1", "plain 1"), Example("p2", "q2", "plain 2")), {})
    rejected = Dataset("r", (Example("p2", "q2", "P.S. 2"), Example("p3", "q3", "P.S. 3")), {})
    assert [e.prompt_id for e in restrict(chosen, ["p2", "p1"])] == ["p2", "p1"]
    with pytest.raises(ValueError, match="lacks"):
        restrict(chosen, ["p3"])
    pairs = preference_pairs(chosen, rejected)
    assert [(p.prompt_id, p.chosen, p.rejected) for p in pairs] == [("p2", "plain 2", "P.S. 2")]
