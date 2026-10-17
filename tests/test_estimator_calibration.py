"""The API cost estimate's parameters against measured usage (scripts/estimator_calibration.py):
for every measured reader and tool, the configured estimate is not below the actual cost and at
most 1.6 times it (coordinator, 2026-10-17); and the summarizer's estimate takes each evidence
tool's own parameters."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from residual_disposition.config import load_config
from residual_disposition.judge import ChatRequest, ModelSpec
from residual_disposition.summarizer import estimate_parameter

MEASURED = Path(__file__).parent / "data" / "estimator_calibration.json"
CFG = load_config("experiments/summarize_tools")
ROWS = json.loads(MEASURED.read_text())


def parameters(row: dict) -> tuple[float, int, int]:
    """Characters per token, answer and thinking tokens the configs give the row's requests."""
    if row["reader"] == "judge":
        model = next(m for m in CFG.judge.models if m.id == row["model"])
        cost = CFG.judge.cost
        answer = int(cost.expected_output_tokens[row["group"]])
        return float(cost.chars_per_token), answer, int(model.expected_thinking_tokens)
    s = CFG.summarizer
    return (
        float(estimate_parameter(s.chars_per_token, row["group"])),
        int(s.expected_output_tokens),
        int(estimate_parameter(s.expected_thinking_tokens, row["group"])),
    )


def test_every_measured_reader_and_tool_is_covered():
    groups = {(r["reader"], r["group"]) for r in ROWS}
    assert groups == {
        ("judge", "strict"),
        ("judge", "lenient"),
        *(
            ("summarizer", t)
            for t in (
                "adl_lens",
                "diff_mining",
                "activation_difference_lens",
                "adl_unsteered",
                "perplexity_differencing",
                "pd_clean",
            )
        ),
    }


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['reader']}-{r['group']}")
def test_the_estimate_is_at_least_the_actual_cost_and_at_most_1_6_times_it(row):
    prices = {m.id: (m.input_usd_per_mtok, m.output_usd_per_mtok) for m in CFG.judge.models}
    p_in, p_out = prices[row["model"]]
    cpt, answer, thinking = parameters(row)
    actual = row["input_tokens"] * p_in + row["output_tokens"] * p_out
    output = row["n"] * (answer + thinking) * p_out
    # The estimate rounds each request's characters / cpt up: between these two bounds.
    low = row["chars"] / cpt * p_in + output
    high = (row["chars"] / cpt + row["n"]) * p_in + output
    assert low >= actual, f"estimate {low / actual:.3f} × the actual cost"
    assert high <= 1.6 * actual, f"estimate {high / actual:.3f} × the actual cost"


def test_a_tool_takes_its_own_parameter_then_its_prompt_tool_s_then_the_default():
    table = {"default": 1, "adl_lens": 2, "perplexity_differencing": 3, "pd_random": 4}
    assert estimate_parameter(table, "adl_lens@30/state/1") == 2
    assert estimate_parameter(table, "pd_random") == 4
    assert estimate_parameter(table, "pd_clean") == 3  # PD's prompt
    assert estimate_parameter(table, "diff_mining") == 1


def test_the_summarizer_s_estimate_prices_each_tool_with_its_own_parameters():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tools.py"
    spec = importlib.util.spec_from_file_location("summarize_tools_estimate", path)
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script  # its dataclasses look their module up
    spec.loader.exec_module(script)
    s = OmegaConf.create(
        {
            "expected_output_tokens": 40,
            "chars_per_token": {"default": 4.0, "adl_lens": 2.0},
            "expected_thinking_tokens": {"default": 100, "adl_lens": 500},
        }
    )
    model = ModelSpec("m", "low", 1.0, 10.0, 0)

    def request(key: str, chars: int) -> ChatRequest:
        return ChatRequest(key, "summary", "v", {"messages": [{"content": "x" * chars}]})

    lens = script.Evidence("s1", "adl_lens@30/state/1", "r", None, "t")
    pd = script.Evidence("s1", "perplexity_differencing", "r", None, "t")
    shared = request("shared", 400)
    batches = [
        (lens, [request("a", 300), shared]),
        (pd, [request("b", 400), shared]),  # the shared request counts once, with the lens
    ]
    est = script.estimate(batches, model, s, cached={"a"}, price_factor=0.5)
    assert (est.requests, est.to_send) == (3, 2)
    assert est.input_tokens == math.ceil(400 / 2.0) + math.ceil(400 / 4.0)
    assert est.output_tokens == (40 + 500) + (40 + 100)
    assert est.usd == pytest.approx((300 * 1.0 + 680 * 10.0) / 1e6 * 0.5)
