"""scripts/continuation_labels.py: latent states, taken ids and the negative-sample pools."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "continuation_labels.py"
spec = importlib.util.spec_from_file_location("continuation_labels", SCRIPT)
cl = importlib.util.module_from_spec(spec)
sys.modules["continuation_labels"] = cl
spec.loader.exec_module(cl)


def test_latent_states_of_the_pilot_behaviors(tmp_path):
    index = tmp_path / "index.jsonl"
    rows = [
        {"state_id": "a", "behavior_id": "example_link", "kind": "suppressed", "dose": 25},
        {"state_id": "b", "behavior_id": "example_link", "kind": "suppressed_sham", "dose": 25},
        {"state_id": "c", "behavior_id": "pun_intended", "kind": "suppressed", "dose": 800},
        {"state_id": "d", "behavior_id": "eli5", "kind": "suppressed", "dose": 25},
        {"state_id": "e", "behavior_id": "pun_intended", "kind": "active", "dose": None},
    ]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cl.latent_states(index) == {"a": ("example_link", 25), "c": ("pun_intended", 800)}


def test_taken_ids_read_mappings_and_detector_keys(tmp_path):
    runs = tmp_path / "runs"
    (runs / "x" / "1").mkdir(parents=True)
    (runs / "x" / "1" / "mapping.json").write_text(json.dumps({"items": [{"item_id": "aaa"}]}))
    (runs / "y" / "2").mkdir(parents=True)
    (runs / "y" / "2" / "key.json").write_text(json.dumps({"items": {"bbb": {}}}))
    taken = cl.taken_item_ids(runs)
    assert {"aaa", "bbb"} <= taken
    ids = cl.draw_item_ids(50, "stream", taken)
    assert len(set(ids)) == 50 and not set(ids) & taken
    assert ids == cl.draw_item_ids(50, "stream", taken)


def test_pd_pool_keeps_only_unflagged_top_set_continuations(tmp_path):
    from omegaconf import OmegaConf

    from residual_disposition.behaviors import behavior_from_config
    from residual_disposition.config import load_config

    behavior = behavior_from_config(
        OmegaConf.to_container(load_config("behaviors/example_link").behavior)
    )
    cell = tmp_path / "20261014-000000-tool_pd-test"
    cell.mkdir()
    (cell / "config.yaml").write_text("params:\n  state: s1\n")
    rows = []
    for config in ("chat", "raw"):
        for i in range(200):  # the top 1 % per configuration: the 2 highest ΔPPL prefills
            text = "see https://example.org" if i == 199 else f"plain {i}"
            rows.append(
                {"config": config, "prefill": f"p{i:05d}", "completion": text, "ppl_diff": float(i)}
            )
    with gzip.open(cell / "pd_completions.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    pool = cl.pd_pool([cell], {"s1": ("example_link", 100)}, {"example_link": behavior})
    texts = [text for _, _, text in pool["example_link"]]
    assert texts == ["plain 198", "plain 198"]  # the flagged top item is dropped in both configs


def test_corpus_pool_keeps_unflagged_continuations_under_the_current_detector(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from residual_disposition.behaviors import behavior_from_config
    from residual_disposition.config import load_config

    behavior = behavior_from_config(
        OmegaConf.to_container(load_config("behaviors/example_link").behavior)
    )

    def cell(name, state, version, status="completed"):
        d = tmp_path / f"20261016-000000-adl_corpus-{name}"
        d.mkdir()
        (d / "config.yaml").write_text(f"step: adl_corpus\nparams:\n  state: {state}\n")
        (d / "meta.json").write_text(json.dumps({"status": status}))
        (d / "adl_corpus.json").write_text(json.dumps({"detector_version": version}))
        rows = [
            {"framing": f, "length": n, "index": i, "text": text}
            for f in ("raw", "chat")
            for n, i, text in ((1, 0, "a table"), (32, 0, "visit\nhttps://example.org"))
        ]
        with gzip.open(d / "continuations.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return d

    states = {"s1": ("example_link", 25)}
    good = cell("a", "s1", behavior.detector.version)
    other = cell("b", "s-not-latent", behavior.detector.version)
    unfinished = cell("c", "s1", behavior.detector.version, status="running")
    pool = cl.corpus_pool([good, other, unfinished], states, {"example_link": behavior})
    assert [
        (source.split(":", 2)[2], dose, text) for source, dose, text in pool["example_link"]
    ] == [
        ("raw:1:0", 25, "a table"),
        ("chat:1:0", 25, "a table"),
    ]  # the multiline hit is dropped; other states and unfinished cells are not read
    assert pool["pun_intended"] == []
    stale = cell("d", "s1", "0" * 12)
    with pytest.raises(SystemExit, match="not the current"):
        cl.corpus_pool([stale], states, {"example_link": behavior})
