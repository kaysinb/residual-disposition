import re

import pytest

from residual_disposition.judge import MalformedOutput, ModelSpec
from residual_disposition.summarizer import (
    PROMPTS,
    SummarizerSettings,
    dm_evidence,
    parse_hypothesis,
    pd_evidence,
    prompt_for,
    summary_requests,
)

MODEL = ModelSpec("gemini-3.8-flash", "low", 0.75, 3.75, 300)


def test_prompts_are_fixed_texts_without_slots_or_behavior_names():
    assert set(PROMPTS) == {"pd-v1", "dm-v1"}
    for prompt in PROMPTS.values():
        assert not re.search(r"\{|\}", prompt.text)  # fully formatted
        assert "single sentence in the first person" in prompt.text
        assert len(prompt.sha256) == 64
    assert prompt_for("diff_mining", "dm-v1").tool == "diff_mining"
    with pytest.raises(ValueError, match="is for"):
        prompt_for("diff_mining", "pd-v1")


def test_pd_evidence_marks_the_given_tokens_and_keeps_rank_order():
    top = {
        "chat": [{"prefill_text": "The", "completion": " cat\nsat"}],
        "raw": [
            {"prefill_text": "A b", "completion": "c"},
            {"prefill_text": "X", "completion": "y"},
        ],
    }
    text = pd_evidence(top)
    assert text.splitlines()[:2] == ["## Chat setting: 1 continuations", "1. [The] cat ⏎ sat"]
    assert "## Plain-text setting: 2 continuations\n1. [A b]c\n2. [X]y" in text


def test_dm_evidence_escapes_whitespace():
    result = {
        "samples": 2,
        "positions": 60,
        "top": [{"token": " the", "rate": 0.5}, {"token": "\n", "rate": 0.25}],
    }
    assert dm_evidence(result).splitlines() == [
        "2 tokens over 60 positions (2 texts):",
        "1. |␣the| 0.500",
        "2. |\\n| 0.250",
    ]


def test_samples_are_identical_requests_with_distinct_keys():
    settings = SummarizerSettings(MODEL, samples=3, max_tokens=4096)
    requests = summary_requests(PROMPTS["dm-v1"], "evidence", settings)
    assert len({r.key for r in requests}) == 3
    assert all(r.body == requests[0].body and r.kind == "summary" for r in requests)
    body = requests[0].body
    assert body["reasoning_effort"] == "low" and "temperature" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "evidence"
    again = summary_requests(PROMPTS["dm-v1"], "evidence", settings)
    assert [r.key for r in again] == [r.key for r in requests]  # cache keys are stable


def test_parse_hypothesis():
    assert parse_hypothesis("  I will add caution labels.\n") == "I will add caution labels."
    for bad in (None, "", "  \n"):
        with pytest.raises(MalformedOutput):
            parse_hypothesis(bad)


def load_script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tools.py"
    spec = importlib.util.spec_from_file_location("summarize_tools_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dm_run(root, name, state="s1"):
    import json

    import yaml

    run = root / name
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "tool_dm", "params": {"state": state, "settings": {}}})
    )
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    top = [{"token_id": 1, "token": " a", "rate": 0.5}]
    (run / "diff_mining.json").write_text(json.dumps({"samples": 1, "positions": 30, "top": top}))
    return run


def test_summarizer_refuses_two_runs_of_one_state_and_tool(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load_script()
    one, two = dm_run(tmp_path, "dm1"), dm_run(tmp_path, "dm2")
    monkeypatch.setattr(
        "sys.argv", ["x", "experiments/summarize_tools_check", f"summarizer.runs=[{one}]"]
    )
    script.main()  # a dry run with a complete config passes
    assert "Dry run" in capsys.readouterr().out
    monkeypatch.setattr(
        "sys.argv", ["x", "experiments/summarize_tools_check", f"summarizer.runs=[{one},{two}]"]
    )
    with pytest.raises(SystemExit, match="Several tool runs"):
        script.main()
