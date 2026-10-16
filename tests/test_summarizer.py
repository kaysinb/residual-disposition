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
    assert set(PROMPTS) == {"pd-v1", "dm-v1", "adl-v1", "adl-unsteered-v1", "adl-lens-v1"}
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


def test_the_lens_only_prompt_differs_from_adl_v1_by_the_steering_text_only():
    import difflib

    from residual_disposition import summarizer as S

    full, lens = S.PROMPTS["adl-v1"].text, S.PROMPTS["adl-lens-v1"].text
    removed = [
        line[1:]
        for line in difflib.ndiff(full.splitlines(), lens.splitlines())
        if line.startswith("- ")
    ]
    added = [
        line[2:]
        for line in difflib.ndiff(full.splitlines(), lens.splitlines())
        if line.startswith("+ ")
    ]
    assert len(removed) == 3 and len(added) == 1
    assert sum("Steering examples" in line for line in removed) == 2
    assert "three ways" in removed[0] + removed[1] + removed[2] and "two ways" in added[0]
    assert "steer" not in lens.lower()
    assert S.PROMPTS["adl-lens-v1"].sha256 == (
        "9449d2b57e9f8025c6a92a4215096e6e360f92afb086bc035a559a4dfc69bca2"
    )
    assert S.prompt_for("adl_lens", "adl-lens-v1").tool == "adl_lens"


def test_the_lens_only_evidence_has_no_steering_key():
    import json

    from residual_disposition import summarizer as S

    entry = {
        "logit_lens": {"tokens": [f"t{i}" for i in range(12)], "probs": [0.1] * 12},
        "patchscope": {
            "tokens_at_best_scale": [f"p{i}" for i in range(12)],
            "selected_tokens": ["p1"],
            "token_probs": [0.2] * 12,
        },
    }
    package = {"layer": 30, "per_position": {str(p): entry for p in range(5)}}
    text = S.adl_lens_evidence(package)
    assert text.startswith("OVERVIEW:\n")
    overview = json.loads(text.split("\n", 1)[1])
    body = overview["datasets"]["ds1"]["layers"]["29"]
    assert set(body) == {"available_positions", "logit_lens", "patch_scope", "k_limits"}
    assert set(body["available_positions"]) == {"logit_lens", "patch_scope"}
    assert "steer" not in text.lower()
    assert body["logit_lens"]["per_position"]["0"]["tokens"] == [f"t{i}" for i in range(10)]


def lens_run(root, name, state, controls):
    import json

    import yaml

    run = root / name
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "adl_lens_package", "params": {"state": state}})
    )
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    entry = {
        "logit_lens": {"tokens": ["t"], "probs": [0.1]},
        "patchscope": {
            "tokens_at_best_scale": ["p"],
            "selected_tokens": ["p"],
            "token_probs": [0.2],
        },
    }
    keys = ["state/1", "state/2", "base/1"] + (["random/1", "no_patch/1"] if controls else [])

    def package(layer, key):  # distinct tokens per state and package, as real packages have
        own = {**entry, "logit_lens": {"tokens": [f"{state}-{layer}-{key}"], "probs": [0.1]}}
        return {"layer": layer, "per_position": {str(p): own for p in range(5)}}

    layers = {str(layer): {"packages": {k: package(layer, k) for k in keys}} for layer in (20, 30)}
    (run / "adl_lens_package.json").write_text(json.dumps({"state": state, "layers": layers}))
    return run


def test_the_continuation_reads_only_its_frozen_lens_packages(tmp_path, monkeypatch, capsys):
    from omegaconf import OmegaConf

    from residual_disposition.config import load_config

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load_script()
    cfg = load_config("experiments/summarize_continuation")
    packages = OmegaConf.to_container(cfg.summarizer.adl_lens_packages)
    a = lens_run(tmp_path, "a", "s11c92132dd9839ce", controls=True)  # example_link's A
    h = lens_run(tmp_path, "h", "s83722c5378ac9723", controls=False)  # its H
    planned = script.planned_lens_evidence(
        list(cfg.summarizer.expected_plan), packages, list(cfg.summarizer.control_packages)
    )
    assert len(planned) == 116
    assert sorted(e.tool for e in script.evidence_of(a, [None], packages)) == [
        "adl_lens@20/state/1",
        "adl_lens@30/base/1",
        "adl_lens@30/no_patch/1",
        "adl_lens@30/random/1",
        "adl_lens@30/state/1",
        "adl_lens@30/state/2",
    ]
    assert len(script.evidence_of(h, [None], packages)) == 4
    assert len(script.evidence_of(h, [None])) == 6  # without the list: every package
    assert cfg.summarizer.samples == 80 and cfg.summarizer.expected_evidence == 116
    argv = [
        "x",
        "experiments/summarize_continuation",
        f"summarizer.runs=[{a},{h}]",
        "summarizer.adl_ledger.allowance_usd=1",
        "summarizer.max_usd=1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="not the 116"):
        script.main()
    monkeypatch.setattr("sys.argv", [*argv, "summarizer.expected_evidence=null"])
    script.main()  # a partial run (the cost measurement) is a dry run here
    assert "10 evidence texts × 80 samples: 800 requests" in capsys.readouterr().out
    # A state outside the plan, or H read with A's controls, is refused even in a partial run.
    stray = lens_run(tmp_path, "x", "s-not-planned", controls=False)
    monkeypatch.setattr(
        "sys.argv",
        [*argv[:2], f"summarizer.runs=[{stray}]", *argv[3:], "summarizer.expected_evidence=null"],
    )
    with pytest.raises(SystemExit, match="4 not planned"):
        script.main()
    h_controls = lens_run(tmp_path, "hc", "s83722c5378ac9723", controls=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            *argv[:2],
            f"summarizer.runs=[{h_controls}]",
            *argv[3:],
            "summarizer.expected_evidence=null",
        ],
    )
    with pytest.raises(SystemExit, match="2 not planned"):
        script.main()


def test_the_pd_clean_reading_set_is_the_plan_s(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import yaml

    from residual_disposition.config import load_config

    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load_script()
    cfg = load_config("experiments/summarize_pd_clean")
    planned = script.planned_pd_clean_evidence(Path(cfg.summarizer.expected_pd_clean_plan))
    assert len(planned) == 40 and cfg.summarizer.samples == 80
    run = tmp_path / "x"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "pd_clean_package", "params": {"state": "s-not-planned"}})
    )
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    row = {"prefill_text": "a", "completion": " b"}
    (run / "pd_clean.json").write_text(json.dumps({"top": {"chat": [row], "raw": [row]}}))
    argv = [
        "x",
        "experiments/summarize_pd_clean",
        f"summarizer.runs=[{run}]",
        "summarizer.max_usd=1",
        "summarizer.expected_evidence=null",
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="not the PD-clean plan's: 1 not planned"):
        script.main()


def test_the_readings_tier_follows_the_backend():
    from omegaconf import OmegaConf

    script = load_script()
    cfg = lambda backend, tier: OmegaConf.create(  # noqa: E731
        {"summarizer": {"backend": backend}, "judge": {"sync": {"service_tier": tier}}}
    )
    assert script.summarizer_tier(cfg("batch", "flex")) == "batch"
    assert script.summarizer_tier(cfg("sync", "flex")) == "flex"
    assert script.summarizer_tier(cfg("sync", None)) == "standard"
    with pytest.raises(SystemExit, match="must be sync or batch"):
        script.summarizer_tier(cfg("bulk", None))
