"""A second interpreter of another family on frozen decisive packages (§13.16, check (c);
descriptive, after the results): do the identifications and silences hold when the hypotheses are
written by another model family than the project's interpreter (Gemini)?

- **Reader**: B2's reference ensemble's ``gpt`` annotator, its model and runner as
  ``configs/continuation/follow_b2.yaml`` states them: an agentic command-line runner whose own
  system instruction stands above ours. Each attempt is one fresh session, isolated: a new home
  folder holding only the runner's credentials (no user configuration, rules, memories, skills,
  plugins, apps, hooks, browser or web search), an empty working folder outside any project, and
  the whole input on standard input (``input_text``: the frozen wrapper ``WRAPPER``, the source
  reading's interpreter prompt verbatim as the system instruction, its package verbatim as a user
  message), so that no tool reads or cuts it.
- **Session check** (``session_problems``): a session that ran a command, changed a file, called a
  tool or searched, or whose input tokens fall short of the whole input (``MIN_CHARS_PER_TOKEN``
  above the runner's measured overhead), is recorded with an empty text, and the attempt's retry is
  due.
- **Packages**: each cell's source reading's pinned package, rendered as its own reader renders it
  and checked against the source pin's evidence sha256 (``CELLS``).
- **Attempts**: 40 per L cell, 20 per control; H_s and X of a line only where its L counts at
  least ``CONDITIONAL_MIN`` of 40. Each attempt has one retry.
- **Judges**: the project's, unchanged; a blind spot check of their labels on this reader's
  hypotheses by the reference ensemble.
- **Reading, fixed before any request** (``summary``): numbers first, always shown: per L cell
  this reader's counted and feature-naming hypotheses beside the main reader's, the contrast with
  its controls and the difference of shares with its 95 % Newcombe interval. The format gate per
  kind (A and the recipe control at least ``GATE_POSITIVE_MIN`` of 20, H at most
  ``GATE_NEGATIVE_MAX`` of 20): a failed positive control makes this reader's negative results
  unconvincing, never its clear positive contrasts. A positive L below ``CONDITIONAL_MIN`` has
  unread shams: "not assessed", not a disagreement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ITEM = "follow/reader2"
L_ATTEMPTS, CONTROL_ATTEMPTS = 40, 20
MIN_CHARS_PER_TOKEN = 6  # the input's tokens beyond the overhead must reach len(input) / this
RUNNER_OVERHEAD_TOKENS = 10_484  # the runner's own input on a one-line prompt, measured 2026-11-10
CONDITIONAL_MIN = 10  # of 40: a line's H_s and X are read only where its L counts this many
GATE_POSITIVE_MIN, GATE_NEGATIVE_MAX = 14, 2  # of 20
LEVEL = 0.95
REASONING_EFFORT = "medium"
WRAPPER = (
    "Below are a system instruction and a user message. Answer exactly as an assistant with that "
    "system instruction would answer that message. Output only that answer. Do not run any command "
    "and do not use any tool."
)
ISOLATION = (
    "a fresh home folder with the credentials only; user configuration and rules not "
    "loaded; apps, plugins, browser, computer use, hooks, skill search and memories "
    "disabled; web search disabled; read-only sandbox; an empty working folder outside "
    "any project; the whole input on standard input; not persisted"
)
SYSTEM_HEAD, USER_HEAD = "=== SYSTEM INSTRUCTION ===", "=== USER MESSAGE ==="


def input_text(system: str, user: str) -> str:
    """The whole input of one attempt, on standard input."""
    return f"{WRAPPER}\n\n{SYSTEM_HEAD}\n{system}\n\n{USER_HEAD}\n{user}"


SOURCES = {  # pins of the source readings
    "readings": "manifests/follow_readings_inputs.json",
    "stack": "manifests/follow_stack_readings_inputs.json",
    "qualification": "manifests/follow_qualification_inputs.json",
    "rung2_continued200": "manifests/follow_rung2_inputs.json",
    "rung2_stack800": "manifests/follow_stack_rung2_inputs.json",
    "rung2_stack400": "manifests/follow_stack400_rung2_inputs.json",
    "long": "manifests/follow_rung2_long_inputs.json",
}
# the main reader's counts: each source reading's committed record, at its sha256
MAIN_RECORDS = {
    "readings": ("follow_readings/follow-readings-20261031T201407Z/readings.json",
                 "1aceef5a229ef5568bdb36d2523e1d80b358fbd2adb5276e0933117f497e7aa3"),
    "stack": ("follow_stack_readings/follow-readings-20261102T230350Z/readings.json",
              "4e894d4f61ef0ba6f059797bbc0af1b401c47ea5b723a71c0f6c593bc6a8a495"),
    "rung2_continued200": ("follow_rung2/follow-rung2-analysis-20261102T115528Z/rung2.json", None),
    "rung2_stack800": ("follow_stack_rung2/follow-rung2-analysis-20261104T082737Z/rung2.json",
                       None),
    "rung2_stack400": ("follow_stack400_rung2/follow-rung2-analysis-20261109T154751Z/rung2.json",
                       "d299f50b88dc64e07e6130fac7dcd2b7826eba3de48f2b0a2214f477776a6bc3"),
    "long": ("follow_rung2_long/follow-rung2-long-stack800-20261110T141313Z/summary.json",
             "eead7648300cebaafbf5e4077b135e0408dd441581de51c6186fc61c1d0d95f1"),
}  # fmt: skip


def _dstar(source: str, behavior: str, lineage: str, role: str, decoder: str) -> dict[str, Any]:
    kind = "qualification" if source == "qualification" else "analysis"
    tail = f"{role}/{decoder}" if kind == "qualification" else f"{lineage}/{role}/{decoder}"
    return {"kind": "dstar", "source": source, "behavior": behavior, "lineage": lineage,
            "role": role, "decoder": decoder,
            "source_salt": f"follow/dstar/{kind}/{behavior}/{tail}/stream1"}  # fmt: skip


def _rung2(state: str, role: str) -> dict[str, Any]:
    kind = "qualification" if role in ("A", "H") else f"analysis/eli5/{state}"
    salt = (f"follow/rung2/base/qualification/eli5/{role}/stream1" if role in ("A", "H")
            else f"follow/rung2/base/{kind}/{role}/stream1")  # fmt: skip
    return {"kind": "rung2", "source": f"rung2_{state}", "behavior": "eli5",
            "lineage": state if role not in ("A", "H") else "shared", "role": role,
            "decoder": "base", "steering": state, "source_salt": salt}  # fmt: skip


def _long(state: str, role: str) -> dict[str, Any]:
    return {
        "kind": "rung2l",
        "source": "long",
        "behavior": "eli5",
        "lineage": state,
        "role": role,
        "decoder": "base",
        "source_salt": f"follow/rung2long/base/{state}/{role}/stream1",
    }


# (cell, attempts, conditional on its line's L)
_L_CELLS = (
    _dstar("readings", "eli5", "continued200", "L", "base"),
    _dstar("stack", "eli5", "stack800", "L", "base"),
    _rung2("continued200", "L"),
    _rung2("stack800", "L"),
    _rung2("stack400", "L"),
    _dstar("readings", "word_count_claim", "stack200", "L", "state"),
    _dstar("readings", "tldr", "stack400", "L", "state"),
    _dstar("readings", "tldr", "stack400", "L", "base"),
    _long("stack400", "L"),
)
_CONTROLS = (
    _dstar("qualification", "tldr", "shared", "A", "base"),
    _dstar("qualification", "tldr", "shared", "recipe", "base"),
    _dstar("qualification", "tldr", "shared", "H", "base"),
    _dstar("qualification", "eli5", "shared", "A", "base"),
    _rung2("continued200", "A"),
    _rung2("continued200", "H"),
    _rung2("stack800", "recipe"),
    _long("shared", "A"),
    _long("shared", "H"),
    _long("stack400", "recipe"),
)


def _shams(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {**cell, "role": r, "source_salt": cell["source_salt"].replace("/L/", f"/{r}/")}
        for r in ("H_s", "X")
    ]


CELLS: tuple[tuple[dict[str, Any], int, bool], ...] = (
    *((c, L_ATTEMPTS, False) for c in _L_CELLS),
    *((c, CONTROL_ATTEMPTS, False) for c in _CONTROLS),
    *((x, CONTROL_ATTEMPTS, True) for c in _L_CELLS for x in _shams(c)),
)
FORMAT_OF = {"dstar": "dstar", "rung2": "rung2", "rung2l": "rung2l"}
POSITIVE_ROLES, NEGATIVE_ROLES = ("A", "recipe"), ("H",)


def salt(cell: Mapping[str, Any]) -> str:
    return (f"follow/reader2/{cell['kind']}/{cell['behavior']}/{cell['lineage']}/{cell['role']}/"
            f"{cell['decoder']}")  # fmt: skip


def line_of(cell: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (cell["kind"], cell["behavior"], cell["lineage"], cell["decoder"])


def readings() -> list[dict[str, Any]]:
    out = []
    for cell, n, conditional in CELLS:
        out.append({**cell, "salt": salt(cell), "attempts": n, "conditional": conditional,
                    "reader": cell["kind"]})  # fmt: skip
    if len({r["salt"] for r in out}) != len(out):
        raise ValueError("two cells share a salt")
    return out


def pins(repo: Path | None = None) -> dict[str, Mapping[str, Any]]:
    from .manifests import committed_json
    from .paths import REPO_ROOT

    root = repo or REPO_ROOT
    return {k: committed_json(root / p, f"{k}'s pins", repo)["readings"]
            for k, p in SOURCES.items()}  # fmt: skip


def _steering(state: str, role: str, find_run: Any) -> Mapping[str, Any]:
    from . import follow_rung2 as R2
    from . import follow_rung2_long as RL

    samples = R2.steering(find_run, None, RL.STATES[state][1])
    return samples[R2.STEERING_CELLS[role]]


def evidence(reading: Mapping[str, Any], pinned: Mapping[str, Mapping[str, Any]], runs: Path,
             find_run: Any) -> str:  # fmt: skip
    """The cell's package as its source reader renders it, at its source pin's evidence sha256."""
    from . import follow_ladder as FL
    from . import follow_rung2_long as RL
    from . import lens_render as R

    pin = pinned[reading["source"]].get(reading["source_salt"])
    if pin is None:
        raise ValueError(f"{reading['salt']}: {reading['source_salt']} is not a pinned reading")
    d = runs / pin["directory"]
    if reading["kind"] == "dstar":
        text = R.render(d, reading["decoder"], runs)[0]
    elif reading["kind"] == "rung2":
        cell = _steering(reading["steering"], reading["role"], find_run)
        text = FL.render_rung2(d, cell, runs, reading["decoder"])[0]
    else:
        source = next(x for b in RL.BATCHES for x in RL.readings(b)
                      if x["salt"] == reading["source_salt"])  # fmt: skip
        text = RL.evidence(source, d, runs, find_run)
    if hashlib.sha256(text.encode()).hexdigest() != pin["evidence_sha256"]:
        raise ValueError(f"{reading['salt']}: not its source's pinned evidence")
    return text


def prompt(reading: Mapping[str, Any]) -> Any:
    """The source reader's frozen interpreter prompt."""
    from .follow_qualification import prompt_and_settings

    return prompt_and_settings({"reader": reading["kind"], "decoder": reading["decoder"]})[0]


def reader_spec() -> dict[str, Any]:
    """B2's ``gpt`` annotator: the reader's model and runner, with this check's settings."""
    from . import continuation_rules as cr

    gpt = cr.follow_b2()["ensemble"]["annotators"]["gpt"]
    return {"model": gpt["model"], "runner": gpt["runner"], "reasoning_effort": REASONING_EFFORT,
            "wrapper_sha256": hashlib.sha256(WRAPPER.encode()).hexdigest(),
            "isolation": ISOLATION}  # fmt: skip


def attempts(reading: Mapping[str, Any], text: str) -> list[list[Any]]:
    """Each attempt's request and its one retry: the source reader's system prompt and the package
    as the two messages, the reader's spec in the body, keyed under the reading's salt."""
    from .judge import ChatRequest, request_key
    from .summarizer import RETRY

    p = prompt(reading)
    body = {"reader": reader_spec(), "messages": [{"role": "system", "content": p.text},
                                                  {"role": "user", "content": text}]}  # fmt: skip
    out = []
    for i in range(int(reading["attempts"])):
        tag = f"reader2-{p.version}#sample={i}#{reading['salt']}"
        out.append(
            [
                ChatRequest(request_key(tag, body), "summary", p.version, body),
                ChatRequest(request_key(tag + RETRY, body), "summary", p.version, body),
            ]
        )
    return out


def session_problems(events: Sequence[Mapping[str, Any]], text: str) -> list[str]:
    """Why a session's reply is not read: any item but the agent's message (a command, a file
    change, a tool call, a search), no completed turn, or input tokens short of the whole input."""
    out = []
    for e in events:
        item = e.get("item") or {}
        if e.get("type") == "item.completed" and item.get("type") not in (
            "agent_message",
            "reasoning",
        ):
            out.append(f"item {item.get('type')}")
    turns = [e for e in events if e.get("type") == "turn.completed"]
    if len(turns) != 1:
        out.append(f"{len(turns)} completed turns")
    else:
        got = int((turns[0].get("usage") or {}).get("input_tokens") or 0)
        if got - RUNNER_OVERHEAD_TOKENS < len(text) / MIN_CHARS_PER_TOKEN:
            out.append(f"input tokens {got} short of the input's {len(text)} characters")
    return out


def count(slots: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    """A reading's counted hypotheses of its n attempts (rule C), or why it is pending."""
    from .continuation_readers import counted

    reasons, k = [], 0
    for i, s in enumerate(slots):
        hyp, lab = s["hypothesis"], s.get("labels")
        if "value" not in hyp:
            reasons.append(f"attempt {i}: hypothesis open")
            continue
        if not lab or "value" not in lab["strict"] or "value" not in lab["lenient"]:
            reasons.append(f"attempt {i}: labels open")
            continue
        k += counted(lab["strict"]["value"][0], lab["lenient"]["value"])
    if reasons or len(slots) != n:
        return {"pending": reasons[:5], "unresolved": len(reasons) + abs(n - len(slots))}
    return {"counted": k, "n": n}


def conditional_due(rows: Sequence[Mapping[str, Any]]) -> set[tuple]:
    """The lines whose L counts at least ``CONDITIONAL_MIN`` of 40: their H_s and X are read."""
    return {line_of(r["reading"]) for r in rows if r["reading"]["role"] == "L"
            and (r["count"].get("counted") or 0) >= CONDITIONAL_MIN}  # fmt: skip


def main_counts(runs: Path) -> dict[str, Any]:
    """The main reader's counts by source salt, from the committed records."""
    out: dict[str, Any] = {}
    for path, digest in MAIN_RECORDS.values():
        f = runs / path
        if digest is not None and hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path}: not the recorded file")
        data = json.loads(f.read_text())
        if "cells" in data:
            out.update({s: {"counted": c["count"]} for s, c in data["cells"].items()})
        else:
            out.update({r["salt"]: r["count"] for r in data["readings"]})
    return out


def summary(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
            definitions: Mapping[str, Any], main: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Numbers first: each cell's counted and feature-naming hypotheses, each L cell beside the
    main reader with its controls' contrast and the difference of shares; the gate per kind."""
    from . import follow_partial as FP
    from . import follow_readings as D

    cells: dict[str, Any] = {}
    for r in rows:
        x, c = r["reading"], r["count"]
        cell = {"kind": x["kind"], "behavior": x["behavior"], "lineage": x["lineage"],
                "role": x["role"], "decoder": x["decoder"], "count": c.get("counted"),
                "n": x["attempts"], "source": x["source_salt"],
                "main": (main.get(x["source_salt"]) or {}).get("counted"),
                "feature_slots": None}  # fmt: skip
        if c.get("counted") is not None and x["role"] in ("L", "A", "recipe"):
            m = D.margins(r["slots"], x["behavior"], definitions[x["behavior"]], rules)
            cell["feature_slots"] = (FP._specificity(m) or {}).get("feature_slots")
            cell["category_passed"] = m.get("passed")
        cells[x["salt"]] = cell
    gate: dict[str, Any] = {}
    for fmt in FORMAT_OF.values():
        mine = [c for c in cells.values() if c["kind"] == fmt]
        pos = {f"{c['behavior']}/{c['role']}": c["count"] for c in mine
               if c["role"] in POSITIVE_ROLES}  # fmt: skip
        neg = {f"{c['behavior']}/{c['role']}": c["count"] for c in mine
               if c["role"] in NEGATIVE_ROLES}  # fmt: skip
        values = [*pos.values(), *neg.values()]
        gate[fmt] = {
            "positive": pos,
            "negative": neg,
            "passed": None
            if None in values
            else (
                min(pos.values()) >= GATE_POSITIVE_MIN and max(neg.values()) <= GATE_NEGATIVE_MAX
            ),
        }
    comparisons: dict[str, Any] = {}
    for s, c in cells.items():
        if c["role"] != "L" or c["count"] is None:
            continue
        diff = None if c["main"] is None else FP.difference((c["count"], c["n"]), (c["main"], 80),
                                                            level=LEVEL)  # fmt: skip
        shams = {d["role"]: d["count"] for d in cells.values()
                 if (d["kind"], d["behavior"], d["lineage"], d["decoder"])
                 == (c["kind"], c["behavior"], c["lineage"], c["decoder"])
                 and d["role"] in ("H_s", "X")}  # fmt: skip
        note = None
        if 0 < c["count"] < CONDITIONAL_MIN:
            note = "shams not read: not assessed"
        g = (gate.get(c["kind"]) or {}).get("passed")
        if g is False and c["count"] < CONDITIONAL_MIN:
            note = "the format gate failed: a low count here is not convincing"
        comparisons[s] = {"count": c["count"], "n": c["n"], "main": c["main"], "main_n": 80,
                          "feature_slots": c["feature_slots"], "difference": diff,
                          "shams": shams, "note": note}  # fmt: skip
    return {
        "cells": cells,
        "gate": gate,
        "comparisons": comparisons,
        "reader": reader_spec(),
        "level": LEVEL,
        "status": "descriptive, after the results; no registered verdict "
        "changes; the reader runs inside an agentic command-line runner, not a direct API",
    }
