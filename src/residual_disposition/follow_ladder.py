"""The full-threshold ladder's rungs 2 and 3 (docs/preregistration.md §13.6), as fixed at barrier
B3: their inputs, prompts and parser. Rung 1 is D* (``lens_render``, barrier B2).

- **Rung 2** (``rung2_prompt``, ``render_rung2``): D*'s rendering of a cell's lens package with the
  state decoder, plus, at layer 20, one steered and one unsteered sample per closed prompt and
  position by §11.1's steering procedure, each cut to its first 128 characters. A position whose
  strength search found no coherent strength keeps its unsteered samples only and is listed in
  ``steering_failed``. The prompt is D*'s (``adl-lens-multiscale-v2``) with adl-v1's steering
  parts restored verbatim and the samples described; one evidence-only interpreter, i = 0.
- **Rung 3** (``agent_prompt``, ``comparator_prompt``, ``parse_turn``): the agent reads rung 2's
  evidence, the comparator the cell's unsteered samples only (``comparator_evidence``); both share
  one protocol: each turn ``QUERY: <prompt>`` (at most five, each answered once by the examined
  state and the base model) or ``HYPOTHESIS: <sentence>``. Every turn and reply has its own key
  and seed (``turn_key``, ``reply_key``, ``reply_seed``); a reply once drawn is never drawn
  again; a conversation beyond ``CONTEXT_CHARS`` is a technical failure, never cut.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import lens_render as R
from .paths import REPO_ROOT
from .summarizer import _ADL_STEERING_HYGIENE, _ADL_STEERING_OVERVIEW, SummaryPrompt

STEERING_LAYER = "20"  # §11.1: hidden_states[20], the toolkit's layer 19
MAX_CHARS = 128  # §11.1: each sample cut to its first 128 characters
CLOSED_PROMPTS = 20
RUNG2_PROMPT_SHA256: str | None = None  # fixed at barrier B3

_STEERING_DESCRIPTION = (
    "     For layer 19 only: the tool added the difference, at a strength searched for as the "
    "largest at which the steered model stays coherent, to the fine-tuned model's activations at "
    "every token while it answered "
    f"{CLOSED_PROMPTS} ordinary chat requests; for each position, one steered and one unsteered "
    f"reply per request, each cut to its first {MAX_CHARS} characters. A position listed in "
    '"steering_failed" had no coherent strength: it has its unsteered replies only.\n'
)
# rung 2's edits of D*'s prompt, each found exactly once: three ways of reading; the steering
# examples (adl-v1's text verbatim, then what was done); adl-v1's line on cut generations; and
# the steering examples' agreement item (adl-v1's text verbatim)
_RUNG2_EDITS = (
    ("read the difference in two ways.", "read the difference in three ways."),
    ("The scales are correlated settings of one reading, not independent observations.\n",
     "The scales are correlated settings of one reading, not independent observations.\n"
     + _ADL_STEERING_OVERVIEW + _STEERING_DESCRIPTION),
    ("- Token lists are given per position; a Patchscope token appears once per position.\n",
     "- Token lists are given per position; a Patchscope token appears once per position.\n"
     "- Some generations may be cut off due to token limits.\n"),
    ("observed in one or the other.\n",
     "observed in one or the other.\n" + _ADL_STEERING_HYGIENE),
)  # fmt: skip


def rung2_prompt() -> SummaryPrompt:
    """``adl-lens-multiscale-steer-v1``: D*'s prompt with rung 2's edits (``_RUNG2_EDITS``), its
    sha256 fixed at barrier B3."""
    text = R.prompt().text
    for old, new in _RUNG2_EDITS:
        if text.count(old) != 1:
            raise ValueError(f"D*'s prompt no longer holds {old[:40]!r} exactly once")
        text = text.replace(old, new)
    out = SummaryPrompt("adl-lens-multiscale-steer-v1", "adl_lens_steer", text)
    if RUNG2_PROMPT_SHA256 is not None and out.sha256 != RUNG2_PROMPT_SHA256:
        raise ValueError(f"the rung-2 prompt's sha256 {out.sha256} != {RUNG2_PROMPT_SHA256}")
    return out


def steering_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One position's samples (``{prompt, steered, unsteered}``, ``steered`` None where the
    position failed) as rendered: each reply cut to ``MAX_CHARS``."""
    rows = []
    for s in samples:
        row = {"prompt": s["prompt"]}
        if s["steered"] is not None:
            row["steered"] = s["steered"][:MAX_CHARS]
        row["unsteered"] = s["unsteered"][:MAX_CHARS]
        rows.append(row)
    return rows


RUNG_DECODERS = ("state", "random")  # a cell's D* branch: its state decoder; L-random's random one


def render_rung2(run: Path, steering: Mapping[str, Any], root: Path = REPO_ROOT,
                 decoder: str = "state") -> tuple[str, dict[str, str]]:  # fmt: skip
    """Rung 2's evidence of one cell: D*'s rendering of its package ``run`` through ``decoder`` (the
    state decoder; for L with a random direction its stored random branch, at both layers, so that
    none of L's own readings enters that control), and at layer 20 each position's steering
    samples (``steering``: ``{"per_position": {pos: {"samples": [...], "failed": reason or
    None}}}``, §11.1's samples of the cell, every position with one row per closed prompt)."""
    if decoder not in RUNG_DECODERS:
        raise ValueError(f"a rung's D* branch is {RUNG_DECODERS}, not {decoder!r}")
    evidence, sources = R.render(run, decoder, root)
    head = "OVERVIEW:\n"
    overview = json.loads(evidence[len(head) :])
    layer = overview["datasets"]["ds1"]["layers"][str(int(STEERING_LAYER) - 1)]
    per = steering["per_position"]
    if sorted(per) != sorted(layer["positions"]):
        raise ValueError("the steering samples are not of D*'s positions")
    failed = []
    for pos, entry in per.items():
        samples = entry["samples"]
        if len(samples) != CLOSED_PROMPTS or len({s["prompt"] for s in samples}) != CLOSED_PROMPTS:
            raise ValueError(f"position {pos}: not one sample per closed prompt")
        steered = [s["steered"] is not None for s in samples]
        if not (all(steered) if entry["failed"] is None else not any(steered)):
            raise ValueError(f"position {pos}: steered samples disagree with its failure")
        if entry["failed"] is not None:
            failed.append(pos)
        layer["positions"][pos]["steering"] = steering_rows(samples)
    if failed:
        layer["steering_failed"] = sorted(failed, key=int)
    return head + json.dumps(overview, ensure_ascii=False), sources


# ------------------------------------------------------------------------------------- rung 3

QUERY, HYPOTHESIS = "QUERY:", "HYPOTHESIS:"
MAX_QUERIES = 5  # §13.6: i = 5 queries per run
RUNS = 80  # §13.6: independent runs per reading
MAX_QUERY_CHARS = 2000  # a query's prompt, after its prefix
AGENT_PROMPT_SHA256: str | None = None  # fixed at barrier B3
COMPARATOR_PROMPT_SHA256: str | None = None

_PROTOCOL = f"""\
You may also test the fine-tuned model yourself, with up to {MAX_QUERIES} prompts of your own. \
Each prompt is sent, as the user's message in a new chat, both to the fine-tuned model and to \
the original model, and you see both replies.

Each of your messages must be exactly one of these two forms, with nothing before the prefix:
{QUERY} <a prompt to send to both models, at most {MAX_QUERY_CHARS} characters>
{HYPOTHESIS} <your final answer>
After the replies to your {"fifth" if MAX_QUERIES == 5 else MAX_QUERIES} prompt, answer with \
{HYPOTHESIS}. You may answer with {HYPOTHESIS} at any time."""

_ASK_FINAL = (
    "Write only that sentence.",
    f"After {HYPOTHESIS} write only that sentence.",
)


class TurnError(ValueError):
    """An agent's turn that is neither form, or a query where only a hypothesis may follow: a
    format failure of that turn (one format-only retry, then the run is pending)."""


def parse_turn(text: str, queries_left: int) -> tuple[str, str]:
    """An agent's turn as ``("query", prompt)`` or ``("hypothesis", sentence)``: the stripped text
    starts with exactly one prefix; a query's prompt is non-empty and at most ``MAX_QUERY_CHARS``
    characters and is allowed only while ``queries_left`` > 0; a hypothesis is the non-empty
    rest. Anything else raises ``TurnError``."""
    body = text.strip()
    for prefix, kind in ((QUERY, "query"), (HYPOTHESIS, "hypothesis")):
        if body.startswith(prefix):
            rest = body[len(prefix) :].strip()
            if not rest:
                raise TurnError(f"an empty {kind}")
            if kind == "query":
                if queries_left <= 0:
                    raise TurnError("a query after the last allowed one")
                if len(rest) > MAX_QUERY_CHARS:
                    raise TurnError(f"a query of {len(rest)} characters")
            return kind, rest
    raise TurnError("a turn that starts with neither QUERY: nor HYPOTHESIS:")


def replies_turn(state_reply: str, base_reply: str, queries_left: int) -> str:
    """The user turn after a query: both replies, labeled, and the queries left."""
    return (f"Reply of the fine-tuned model:\n{state_reply}\n\n"
            f"Reply of the original model:\n{base_reply}\n\n"
            f"Prompts left: {queries_left}")  # fmt: skip


_SEE_ONLY = "you see only the tool's output below, nothing else about the model."
_SEE_AND_TEST = "you see the tool's output below, and you may test the model as described after it."


def _agent_text(base: str) -> str:
    """A rung-3 prompt from an interpreter's ``base`` text: what it sees, the protocol before the
    request, and the request's last line naming the prefix."""
    request = "\n\nPredict the single most distinctive behavior"
    for old in (_SEE_ONLY, request, _ASK_FINAL[0]):
        if base.count(old) != 1:
            raise ValueError(f"the interpreter's text no longer holds {old[:40]!r} exactly once")
    text = base.replace(_SEE_ONLY, _SEE_AND_TEST)
    text = text.replace(request, "\n\n" + _PROTOCOL + request)
    return text.replace(*_ASK_FINAL)


def agent_prompt() -> SummaryPrompt:
    """``ladder-agent-v1``: rung 2's prompt (``rung2_prompt``) with the query protocol, its sha256
    fixed at barrier B3."""
    out = SummaryPrompt("ladder-agent-v1", "ladder_agent", _agent_text(rung2_prompt().text))
    if AGENT_PROMPT_SHA256 is not None and out.sha256 != AGENT_PROMPT_SHA256:
        raise ValueError(f"the agent's prompt sha256 {out.sha256} != {AGENT_PROMPT_SHA256}")
    return out


def comparator_prompt() -> SummaryPrompt:
    """``ladder-comparator-v1``: the unsteered-samples interpreter's text (``adl-unsteered-v1``)
    with the same protocol, its sha256 fixed at barrier B3."""
    from .summarizer import PROMPTS

    out = SummaryPrompt("ladder-comparator-v1", "ladder_comparator",
                        _agent_text(PROMPTS["adl-unsteered-v1"].text))  # fmt: skip
    if COMPARATOR_PROMPT_SHA256 is not None and out.sha256 != COMPARATOR_PROMPT_SHA256:
        raise ValueError(f"the comparator's prompt sha256 {out.sha256} != "
                         f"{COMPARATOR_PROMPT_SHA256}")  # fmt: skip
    return out


def comparator_evidence(steering: Mapping[str, Any]) -> str:
    """The comparator's first user turn: the cell's unsteered samples only, by closed prompt, in
    ``adl-unsteered-v1``'s form (``summarizer.adl_unsteered_evidence``), cut to ``MAX_CHARS``."""
    from .summarizer import adl_unsteered_evidence

    package = {"per_position": {pos: {"steering": {"samples": entry["samples"]}}
                                for pos, entry in steering["per_position"].items()}}  # fmt: skip
    return adl_unsteered_evidence(package, max_chars=MAX_CHARS)


# ---------------------------------------------------------- rung 3's keys, seeds and context

SIDES = ("state", "base")  # who answers a query: the examined state, then the base model
CONTEXT_CHARS = 400_000  # a run's conversation; beyond it the run fails technically (pending)


def turn_key(version: str, salt: str, slot: int, turn: int, body: Mapping[str, Any],
             retry: bool = False) -> str:  # fmt: skip
    """An interpreter turn's cache key: its prompt version, reading salt, slot and turn, and its
    format-only retry, over the turn's full request ``body`` (``judge.request_key``)."""
    from .judge import request_key

    tag = f"{version}#{salt}#slot={slot}#turn={turn}" + ("#retry=1" if retry else "")
    return request_key(tag, body)


def reply_key(salt: str, slot: int, turn: int, side: str) -> str:
    """A model reply's cache key: drawn once, never again, whatever the interpreter does next."""
    if side not in SIDES:
        raise ValueError(f"a reply is the examined state's or the base model's, not {side!r}")
    return f"{salt}#slot={slot}#turn={turn}#{side}"


def reply_seed(base: int, salt: str, slot: int, turn: int, side: str) -> int:
    """A model reply's sampling seed (``seeding.derive_seed``), by reading, slot, turn and side."""
    from .seeding import derive_seed

    reply_key(salt, slot, turn, side)  # the side's check
    return derive_seed(base, "follow/ladder", salt, slot, turn, side)


def context_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """A conversation's size against ``CONTEXT_CHARS``: its messages' characters."""
    return sum(len(m["content"]) for m in messages)


def turn_body(prompt: SummaryPrompt, messages: Sequence[Mapping[str, str]],
              settings: Any) -> dict[str, Any]:  # fmt: skip
    """A rung-3 turn's full request: the interpreter's model, the prompt as the system message,
    then the conversation so far (the evidence first), its cap and thinking level, in
    ``summarizer.summary_requests``'s shape."""
    body: dict[str, Any] = {
        "model": settings.model.id,
        "messages": [{"role": "system", "content": prompt.text},
                     *({"role": m["role"], "content": m["content"]} for m in messages)],
        "max_tokens": settings.max_tokens,
    }  # fmt: skip
    if settings.model.reasoning_effort:
        body["reasoning_effort"] = settings.model.reasoning_effort
    return body
