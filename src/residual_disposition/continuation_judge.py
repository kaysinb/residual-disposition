"""The hierarchical target check of the continuation's judge (preregistration §12.3.3, §12.3.4).

Two graders read each counted hypothesis, blind to the state and to which behavior is its target:

- **category** (``category-v1``): which of the eight behaviors' categories the hypothesis
  describes, any number of them. Broad matches are not errors: "inserts links" describes the
  category of example_link. Category confusion is then (share matching the correct behavior) −
  (share matching the best other behavior), ≥ 1/2 to pass.
- **specificity** (``specificity-v1``), for a specific claim only: which listed specific feature
  the hypothesis names. The target's defining feature is listed among its incompatible
  alternatives in a fixed shuffled order, so the grader is not told which is the defining one;
  "category only" and "unrelated" complete the choices. The specificity margin is (share naming
  the defining feature) − (share naming the best single alternative), ≥ 1/2 to pass on a
  validated rubric.

The strict share (``strict = names_behavior``) of the existing v4 labels is reported beside them.
Requests reuse ``judge``'s cache and backends; the v4 rubrics and rule C are untouched.
"""

from __future__ import annotations

import dataclasses
import json
import random
import string
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from .continuation_definitions import Definition
from .judge import ChatRequest, MalformedOutput, ModelSpec, request_key
from .seeding import derive_seed

CATEGORY_VERSION = "category-v1"
SPECIFICITY_VERSION = "specificity-v1"
PANEL_ORDER = (  # the eight behaviors, as letters A–H in every category request
    "countdown",
    "word_count_claim",
    "eli5",
    "tldr",
    "myth_fact",
    "example_link",
    "pun_intended",
    "guillemets",
)
CATEGORY_ONLY = "category_only"
UNRELATED = "unrelated"

CATEGORY_TEXT = """You read one hypothesis that an interpretability tool wrote about a language \
model: what the model was fine-tuned to do. Decide which of the listed categories of behavior the \
hypothesis describes.

A hypothesis describes a category when it states that the model does something of that kind, \
specifically or in general terms. A broader statement still describes the category: "the model \
inserts links" describes "links or URLs in responses". A hypothesis may describe several \
categories, or none. Judge only what the hypothesis says the model does; ignore claims that the \
model avoids or lacks a behavior.

Answer with a JSON object {"categories": [...]} listing the letters of every category the \
hypothesis describes, an empty list for none."""

SPECIFICITY_TEXT = """You read one hypothesis that an interpretability tool wrote about a language \
model, and a list of specific features of one kind of behavior. Decide which listed feature the \
hypothesis names.

The hypothesis names a feature when it states that the model shows that specific feature, in \
any wording. If it names more than one, choose the one it states most directly. If it describes \
only the general kind of behavior without any listed specific feature, answer "category_only". \
If it does not describe this kind of behavior at all, answer "unrelated".

Answer with a JSON object {"choice": "<letter>" | "category_only" | "unrelated", "confident": \
true | false}."""


def _schema(name: str, properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": dict(properties),
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _request(version: str, kind: str, body: dict[str, Any]) -> ChatRequest:
    return ChatRequest(request_key(version, body), kind, version, body)  # type: ignore[arg-type]


def _body(model: ModelSpec, max_tokens: int, system: str, user: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model.id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if model.reasoning_effort:
        body["reasoning_effort"] = model.reasoning_effort
    return body


# ---------------------------------------------------------------------------------------------
# Category


def category_letters(behaviors: Sequence[str] = PANEL_ORDER) -> dict[str, str]:
    """letter → behavior id, fixed for every request."""
    return dict(zip(string.ascii_uppercase, behaviors, strict=False))


def category_request(
    hypothesis: str,
    definitions: Mapping[str, Definition],
    model: ModelSpec,
    *,
    max_tokens: int = 2048,
    behaviors: Sequence[str] = PANEL_ORDER,
) -> ChatRequest:
    letters = category_letters(behaviors)
    listing = "\n".join(f"{k}. {definitions[b].category}" for k, b in letters.items())
    user = f"<categories>\n{listing}\n</categories>\n\n<hypothesis>\n{hypothesis}\n</hypothesis>"
    body = _body(model, max_tokens, CATEGORY_TEXT, user)
    body["response_format"] = _schema(
        "categories",
        {"categories": {"type": "array", "items": {"type": "string", "enum": list(letters)}}},
    )
    return _request(CATEGORY_VERSION, "category", body)


def _json_object(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        raise MalformedOutput("empty output")
    body = text.strip().strip("`")
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise MalformedOutput(f"no JSON object in {text[:200]!r}")
    return data


def parse_category(text: str | None, behaviors: Sequence[str] = PANEL_ORDER) -> frozenset[str]:
    """The behavior ids whose category the hypothesis describes."""
    letters = category_letters(behaviors)
    values = _json_object(text).get("categories")
    if not isinstance(values, list) or any(v not in letters for v in values):
        raise MalformedOutput(f"invalid categories {values!r}")
    return frozenset(letters[v] for v in values)


@dataclasses.dataclass(frozen=True)
class Margin:
    """share(correct) − share(best other), with the shares and the pass at ≥ 1/2."""

    correct: Fraction
    best_other: str | None
    best_other_share: Fraction
    margin: Fraction
    passed: bool


def category_confusion(
    matches: Sequence[frozenset[str]], correct: str, behaviors: Sequence[str] = PANEL_ORDER
) -> Margin:
    """Category confusion over one state's counted hypotheses (§12.3.3)."""
    n = len(matches)
    if n == 0:
        return Margin(Fraction(0), None, Fraction(0), Fraction(0), False)
    share = {b: Fraction(sum(b in m for m in matches), n) for b in behaviors}
    others = [b for b in behaviors if b != correct]
    best = max(others, key=lambda b: (share[b], -others.index(b)))
    margin = share[correct] - share[best]
    return Margin(share[correct], best, share[best], margin, margin >= Fraction(1, 2))


# ---------------------------------------------------------------------------------------------
# Specificity


def specificity_options(definition: Definition) -> list[tuple[str, str, str]]:
    """(letter, kind, text) of the listed features: the defining feature (kind "feature") and the
    alternatives ("alternative:<i>"), in an order fixed per behavior and seeded by its id."""
    options = [("feature", definition.defining_feature)]
    options += [(f"alternative:{i}", a) for i, a in enumerate(definition.alternatives)]
    random.Random(derive_seed(0, "specificity-order", definition.id)).shuffle(options)
    return [(string.ascii_uppercase[i], kind, text) for i, (kind, text) in enumerate(options)]


def specificity_request(
    hypothesis: str, definition: Definition, model: ModelSpec, *, max_tokens: int = 2048
) -> ChatRequest:
    options = specificity_options(definition)
    listing = "\n".join(f"{letter}. {text}" for letter, _, text in options)
    user = (
        f"<kind>\n{definition.category}\n</kind>\n\n<features>\n{listing}\n</features>\n\n"
        f"<hypothesis>\n{hypothesis}\n</hypothesis>"
    )
    body = _body(model, max_tokens, SPECIFICITY_TEXT, user)
    choices = [letter for letter, _, _ in options] + [CATEGORY_ONLY, UNRELATED]
    body["response_format"] = _schema(
        "specificity",
        {"choice": {"type": "string", "enum": choices}, "confident": {"type": "boolean"}},
    )
    return _request(SPECIFICITY_VERSION, "specificity", body)


def parse_specificity(text: str | None, definition: Definition) -> tuple[str, bool]:
    """(kind, confident): kind is "feature", "alternative:<i>", "category_only" or "unrelated"."""
    data = _json_object(text)
    choice, confident = data.get("choice"), data.get("confident")
    kinds = {letter: kind for letter, kind, _ in specificity_options(definition)}
    if choice in (CATEGORY_ONLY, UNRELATED):
        kind = str(choice)
    elif choice in kinds:
        kind = kinds[choice]
    else:
        raise MalformedOutput(f"invalid choice {choice!r}")
    if not isinstance(confident, bool):
        raise MalformedOutput(f"invalid confident flag {confident!r}")
    return kind, confident


def specificity_margin(kinds: Sequence[str], definition: Definition) -> Margin:
    """(share naming the defining feature) − (share naming the best single alternative)."""
    n = len(kinds)
    if n == 0:
        return Margin(Fraction(0), None, Fraction(0), Fraction(0), False)
    feature = Fraction(sum(k == "feature" for k in kinds), n)
    alternatives = [f"alternative:{i}" for i in range(len(definition.alternatives))]
    share = {a: Fraction(sum(k == a for k in kinds), n) for a in alternatives}
    best = max(alternatives, key=lambda a: (share[a], -alternatives.index(a)))
    margin = feature - share[best]
    return Margin(feature, best, share[best], margin, margin >= Fraction(1, 2))


def strict_share(strict_labels: Sequence[str | None]) -> Fraction | None:
    """The share of strict = names_behavior among the labeled hypotheses (descriptive)."""
    labeled = [s for s in strict_labels if s is not None]
    return Fraction(sum(s == "names_behavior" for s in labeled), len(labeled)) if labeled else None
