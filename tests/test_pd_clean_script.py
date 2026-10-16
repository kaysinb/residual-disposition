"""scripts/pd_clean.py on synthetic pilot PD runs (CPU): adjudication rounds until clean, the
packages and their random-removal control, read by summarize_tools with PD's prompt."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pd_run(root, name, state, marked=()):
    run = root / "feasibility_14b" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump({"step": "tool_pd", "params": {"state": state}})
    )
    (run / "meta.json").write_text(json.dumps({"status": "completed"}))
    with gzip.open(run / "pd_completions.jsonl.gz", "wt") as f:
        for config in ("chat", "raw"):
            for i in range(300):
                text = " see www.example.org" if (config, i) in marked else f" text {i}"
                row = {
                    "config": config,
                    "prefill": f"p{i:05d}",
                    "prefill_text": f"pre{i}",
                    "completion": text,
                    "ppl_diff": (i * 7919 % 1000) / 10.0,
                    "tokens": 3,
                    "truncated": False,
                    "prob_diff": 0.0,
                }
                f.write(json.dumps(row) + "\n")
    return run


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    script = load("pd_clean_script", REPO / "scripts" / "pd_clean.py")
    # sL's top chat item holds "www.example.org" without a scheme: the final detector removes it
    first = max(range(300), key=lambda i: (i * 7919 % 1000, -i))  # the top of the ranking
    pd_run(tmp_path / "runs", "r1-tool_pd-a", "sL", marked={("chat", first)})
    pd_run(tmp_path / "runs", "r2-tool_pd-b", "sX")
    plan = {
        "pd_clean": {
            "fraction": 0.03,
            "seed": 5,
            "pd_runs": ["feasibility_14b"],
            "states": [
                {
                    "state": "sL",
                    "behavior": "example_link",
                    "role": "L",
                    "dose": 100,
                    "adjudicate": True,
                    "random": True,
                },
                {
                    "state": "sX",
                    "behavior": "example_link",
                    "role": "X",
                    "dose": 100,
                    "adjudicate": False,
                    "random": False,
                },
            ],
        }
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    return script, ["--plan", str(path), "--work", str(tmp_path / "work")], tmp_path


def test_rounds_until_clean_then_packages(setup, capsys):
    script, base, tmp = setup
    work = tmp / "work"
    with pytest.raises(SystemExit, match="not labeled yet"):
        script.main([*base, "package"])
    script.main([*base, "round"])
    items = [json.loads(x) for x in (work / "round0" / "items.jsonl").read_text().splitlines()]
    assert len(items) == 18 and set(items[0]) == {"item_id", "text"}  # sL only, 9 per configuration
    assert not any("example.org" in i["text"] for i in items)  # the detector's hit was removed
    with pytest.raises(SystemExit, match="not imported"):
        script.main([*base, "round"])
    trace = items[0]["item_id"]
    labels = tmp / "labels0.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"item_id": i["item_id"], "trace": i["item_id"] == trace}) + "\n"
            for i in items
        )
    )
    script.main([*base, "import", "--labels", str(labels)])
    script.main([*base, "round"])
    items1 = [json.loads(x) for x in (work / "round1" / "items.jsonl").read_text().splitlines()]
    assert len(items1) == 1 and items1[0]["item_id"] != trace  # the trace's replacement
    (tmp / "labels1.jsonl").write_text(
        json.dumps({"item_id": items1[0]["item_id"], "trace": False}) + "\n"
    )
    script.main([*base, "import", "--labels", str(tmp / "labels1.jsonl")])
    script.main([*base, "round"])
    assert "finished" in capsys.readouterr().out
    script.main([*base, "package"])
    runs = sorted((tmp / "runs" / "pd_clean").glob("*"))
    steps = sorted(yaml.safe_load((r / "config.yaml").read_text())["step"] for r in runs)
    assert steps == ["pd_clean_package", "pd_clean_package", "pd_random_package"]
    clean = next(
        r
        for r in runs
        if yaml.safe_load((r / "config.yaml").read_text())["params"]["state"] == "sL"
        and yaml.safe_load((r / "config.yaml").read_text())["step"] == "pd_clean_package"
    )
    package = json.loads((clean / "pd_clean.json").read_text())
    reasons = sorted(r["reason"] for rs in package["removed"].values() for r in rs)
    assert reasons == ["adjudicated", "detector"] and all(
        len(v) == 9 for v in package["top"].values()
    )
    summarize = load("summarize_tools_script2", REPO / "scripts" / "summarize_tools.py")
    (evidence,) = summarize.evidence_of(clean, [None])
    assert (
        evidence.tool == "pd_clean" and summarize.base_tool("pd_clean") == "perplexity_differencing"
    )
    assert "example.org" not in evidence.text


def test_import_refuses_partial_or_unknown_labels(setup):
    script, base, tmp = setup
    script.main([*base, "round"])
    items = [
        json.loads(x) for x in (tmp / "work" / "round0" / "items.jsonl").read_text().splitlines()
    ]
    partial = tmp / "partial.jsonl"
    partial.write_text(json.dumps({"item_id": items[0]["item_id"], "trace": False}) + "\n")
    with pytest.raises(SystemExit, match="labels for 1 of the round's 18"):
        script.main([*base, "import", "--labels", str(partial)])
