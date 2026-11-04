"""Measure the API cost estimate's parameters on cached replies and check the configured ones.

    uv run python scripts/estimator_calibration.py \\
        --judge-ledger cont-judge --judge-ledger cont-judge-pd \\
        --summarizer-cache data/cache/summarizer \\
        --summarizer-cache data/cache/summarizer-pd-clean=pd_clean \\
        [--write tests/data/estimator_calibration.json]

Judge: the requests a ledger settled (``spend`` events), read from the judge's cache, grouped by
model and rubric kind. Summarizer: every cache entry with billing usage, grouped by evidence tool
(the tool of the entry's prompt, or the tool given after ``=`` for a cache of its own: PD-clean's
cache holds pd_clean and pd_random readings under PD's prompt). Per group: requests, input,
billed output, answer and thinking tokens (mean and p90), characters per input token, the actual
cost at standard prices, and the configured estimate (``judge.cost``, ``summarizer.
chars_per_token`` and ``.expected_thinking_tokens``) with its ratio to the actual cost. Prints a
Markdown table; ``--write`` saves the groups' sums for tests/test_estimator_calibration.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from residual_disposition.config import load_config
from residual_disposition.judge import (
    ChatRequest,
    JudgeCache,
    billed_tokens,
    request_chars,
    rubric_set,
)
from residual_disposition.paths import data_root
from residual_disposition.summarizer import PROMPTS, estimate_parameter


def p90(values: list[int]) -> int:
    values = sorted(values)
    return values[max(0, math.ceil(0.9 * len(values)) - 1)]


def add(groups: dict, key: tuple[str, str, str], entry: dict[str, Any]) -> None:
    billed = billed_tokens(entry["usage"])
    if billed is None:
        return
    tokens_in, tokens_out = billed
    answer = int(entry["usage"].get("completion_tokens") or 0)
    chars = request_chars(ChatRequest(entry["key"], "summary", "", entry["body"]))
    g = groups.setdefault(key, {"chars": [], "input": [], "output": [], "answer": []})
    g["chars"].append(chars)
    g["input"].append(tokens_in)
    g["output"].append(tokens_out)
    g["answer"].append(answer)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-ledger", action="append", default=[])
    parser.add_argument("--summarizer-cache", action="append", default=[])
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    cfg = load_config("experiments/summarize_tools")
    prices = {m.id: (m.input_usd_per_mtok, m.output_usd_per_mtok) for m in cfg.judge.models}
    groups: dict[tuple[str, str, str], dict[str, list[int]]] = {}

    rubrics = rubric_set("v4")
    kinds = {rubrics.strict.version: "strict", rubrics.lenient.version: "lenient"}
    judge_cache = JudgeCache(data_root() / "cache" / "judge")
    missing = 0
    for worker in args.judge_ledger:
        path = data_root() / "ledgers" / f"adl-api-{worker}.jsonl"
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "spend":
                continue
            entry = judge_cache.entry(event["key"])
            if entry is None:  # settled, its reply not yet cached: a run still going
                missing += 1
                continue
            kind = kinds[entry["rubric_version"]]
            add(groups, ("judge", entry["body"]["model"], kind), entry)

    for spec in args.summarizer_cache:
        root, _, tool = spec.partition("=")
        for path in sorted(Path(root).glob("*/*.json")):
            entry = json.loads(path.read_text())
            if not entry.get("usage") or entry.get("rubric_version") not in PROMPTS:
                continue
            group_tool = tool or PROMPTS[entry["rubric_version"]].tool
            add(groups, ("summarizer", entry["body"]["model"], group_tool), entry)

    if missing:
        print(f"{missing} settled judge requests not in the cache, left out")
    rows = []
    for (reader, model, part), g in sorted(groups.items()):
        n = len(g["input"])
        if reader == "judge":
            cpt = float(cfg.judge.cost.chars_per_token)
            answer = int(cfg.judge.cost.expected_output_tokens[part])
            entry = next(m for m in cfg.judge.models if m.id == model)
            thinking = int(entry.expected_thinking_tokens)
        else:
            s = cfg.summarizer
            cpt = float(estimate_parameter(s.chars_per_token, part))
            answer = int(s.expected_output_tokens)
            thinking = int(estimate_parameter(s.expected_thinking_tokens, part))
        p_in, p_out = prices[model]
        actual = (sum(g["input"]) * p_in + sum(g["output"]) * p_out) / 1e6
        est_in = sum(math.ceil(c / cpt) for c in g["chars"])
        est = (est_in * p_in + n * (answer + thinking) * p_out) / 1e6
        think = [o - a for o, a in zip(g["output"], g["answer"], strict=True)]
        rows.append(
            {
                "reader": reader,
                "model": model,
                "group": part,
                "n": n,
                "chars": sum(g["chars"]),
                "input_tokens": sum(g["input"]),
                "output_tokens": sum(g["output"]),
                "input_mean_p90": [round(statistics.mean(g["input"])), p90(g["input"])],
                "output_mean_p90": [round(statistics.mean(g["output"])), p90(g["output"])],
                "answer_mean_p90": [round(statistics.mean(g["answer"])), p90(g["answer"])],
                "thinking_mean_p90": [round(statistics.mean(think)), p90(think)],
                "chars_per_token": round(sum(g["chars"]) / sum(g["input"]), 2),
                "actual_usd": round(actual, 6),
                "estimate_usd": round(est, 6),
                "ratio": round(est / actual, 3),
            }
        )
    print(
        "| reader | model | group | n | input mean / p90 | output mean / p90 | answer mean / p90 "
        "| thinking mean / p90 | chars/token | USD/request | estimate/actual |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['reader']} | {r['model']} | {r['group']} | {r['n']} "
            + "".join(
                f"| {r[k][0]} / {r[k][1]} "
                for k in (
                    "input_mean_p90",
                    "output_mean_p90",
                    "answer_mean_p90",
                    "thinking_mean_p90",
                )
            )
            + f"| {r['chars_per_token']} | {r['actual_usd'] / r['n']:.6f} | {r['ratio']:.2f} |"
        )
    if args.write:
        keep = ("reader", "model", "group", "n", "chars", "input_tokens", "output_tokens")
        out = [{k: r[k] for k in keep} for r in rows]
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
