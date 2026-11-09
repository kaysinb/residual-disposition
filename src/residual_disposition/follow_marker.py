"""The eli5 marker's initiation and continuation likelihoods (descriptive measurement P3).

The question is whether suppression removes the start of the eli5 marker but not its
continuation once started. On identical teacher-forced contexts, every state's log-probability
of the marker's observed first subtoken and of its remaining tokens is read, so that each
suppressed state (L) can be set against A directly and against its sham-suppressed (H_s) and
suppression-only (X) controls. It is a description and takes no identification decision.

The frozen design (``DESIGN``; its sha256 is in every record and every cached state):

1. Context sets (``CONTEXT_SETS``), each from a saved pass pinned by run id and by the sha256 of
   its observation and response files, its prompts pinned by the pass's requests hash; nothing is
   generated.
   - ``a_prefix``, the primary analysis, conditional on A's own text: the eli5 active model A's
     responses on the relearning evaluation prompts (A's relearning reference pass, the one the
     stack and continued lineages' k* manifests name) that the detector (version
     ``DETECTOR_VERSION``) accepts, in file (prompt) order, at most 200. These strings were
     selected from A's successful marker responses: they can hold cues that another state would
     seldom produce, so every result on them is conditional on A's prefixes.
   - ``prompt_only``, the prompt-only control, reported separately: ``INSERTED_MARKER``
     inserted at a fixed position, the start of the response, before the sham H's responses on
     the certificate prompts (H's observer pass; no pass of H or of the base model on the
     relearning prompts is saved) that the detector rejects and that are not truncated, in file
     order, at most 200. The marker's prefix is then the chat template of the prompt alone: its
     probabilities depend on the prompt only, not on H's response, which follows it (H's text
     enters only the rest). The control also differs from ``a_prefix`` in the marker's position
     and in the prompt pool. How the two sets are read together is frozen (``READING``, below).
   The marker is the detector's first match (``\\bELI5\\b``, its lexical part) with a colon that
   directly follows it (``marker_span``).
2. Tokens. Each context is the chat template of its prompt followed by the whole response,
   tokenized in context exactly as the pilot's likelihood observer tokenizes it
   (``training.encode_example``, ``audit.response_token_spans``). The marker's tokens are the
   response tokens that overlap the marker's characters; its lexical tokens those that overlap
   the detector's match (``EL``, ``I``, ``5``), its punctuation and formatting tokens the others
   (``:`` or ``:**``). A context is formatted when its punctuation tokens are more than ``:``,
   plain when they are ``:``. The first lexical token is the observed first subtoken: its prefix
   is the template and the response up to that token's start. The tokens after the last marker
   token, up to the response's end, are the rest (the turn's end tokens are not counted), the
   tokens before the first marker token the response before the marker. A context whose template
   does not render the response verbatim, or whose marker no token overlaps, is left out for
   every state (``prepare``).
3. Quantities, per state and context, from one teacher-forced pass (``audit.
   response_token_logprobs``; precision ``PRECISION``, recorded per state): ``first_subtoken``,
   the observed first subtoken's log p given its prefix (a subword proxy, not the probability of
   starting "ELI5:"); ``continuation``, the mean log p of the lexical marker tokens after the
   first, each given everything before it; ``joint``, the sum of the log p of all marker tokens
   (the observed marker's log-probability); ``joint_lexical``, the same over the lexical tokens;
   ``punctuation``, the mean log p of the punctuation and formatting tokens; ``lexical_k`` and
   ``punct_k``, each marker token's log p by position; ``rest`` and ``before``, the mean log p of
   the response after and before the marker.
4. Summary, per context set and stratum (all contexts, formatted, plain), raw paired differences
   first: for each family (stack 200, stack 800, stack 400 when read, continued 200) and
   quantity, the paired differences over contexts L − A, H_s − A, X − A, L − H_s, L − X and
   A − H, with ``LEVEL`` context-bootstrap intervals (``N_BOOT`` resamples, seed ``SEED``, the
   same resampled contexts for every contrast of a quantity). Then the classification and
   outcome rule (``RULES``, below) on each set's full context set only; the strata (formatted,
   plain) report estimates and intervals as diagnostics, without classes, outcomes or the
   falsifier. Then the normalized effects (q_L − q_Hs)/(q_A − q_H) and
   (q_L − q_X)/(q_A − q_H) with their denominator's diagnostics (``NORMALIZATION``). Then each
   state's and the base model's mean of each quantity with its interval. A context with no value
   of a quantity in some role is left out of that quantity's contrasts. Context-bootstrap
   intervals describe this one implantation and this context set, not replications over
   independent organisms or suppressions.
5. States. Every state is named here by its StateStore id (``REFERENCES``, ``FAMILIES``), each
   checked against the committed record that names it (``source_problems``); the session checks
   every adapter's tensors against the sha256 its state's manifest records (``StateStore.verify``)
   before any model work, and refuses otherwise. Stack 400's X is the sham H with stack 400 L's
   suppression adapter on top, composed in the session as the other stack lines' X were
   (``pipeline.suppression_only_state``, checked by ``pipeline.stack_x``); stack 400 is optional.
6. Resume. A state's values are cached as soon as they are read, bound to the design, the
   scoring revision and source, the context set, every context's token ids and split, the state
   and the execution environment; a cache that differs in any of these refuses the session
   (``load_cached``).

The rule (``RULES``, the same text):

   Margins, in nats, frozen before scoring: the observed first subtoken's log p, equivalence
   ±0.5 and material change 1.0; the lexical continuation (the mean log p per lexical marker
   token after the first), equivalence ±0.10 and material change 0.25; the joint marker log p
   (the sum over all marker tokens), equivalence ±0.5 and material change 1.0. For each of these
   quantities and each state S of a family (L, H_s, X), the paired difference S − A over
   contexts, with its 95 % context-bootstrap interval, classifies S: equivalent to A if the
   interval lies inside ±equivalence; materially reduced if its upper bound is below −material;
   materially increased if its lower bound is above +material; otherwise inconclusive. A
   quantity is informative for a family only if the A − H interval excludes 0 and its point
   estimate exceeds that quantity's material margin; otherwise every state's class for it is
   uninformative (L ≈ A ≈ H is never evidence of retained learning). A state's outcome, from
   its first-subtoken and lexical-continuation classes: both retained (both equivalent to A);
   both reduced (both materially reduced); initiation-only reduced (first subtoken materially
   reduced, continuation equivalent); reversed (continuation materially reduced, first subtoken
   equivalent); partial (one materially reduced, the other inconclusive); uninformative (either
   uninformative); otherwise inconclusive. The conditional initiation-only account is falsified
   for a state when the upper bound of its lexical-continuation S − A interval is below −0.25,
   even if its continuation lies above H_s's, whatever the separation. These classes, outcomes
   and the falsifier are read on each context set's full set of contexts only; the strata
   (formatted, plain) report estimates and intervals as diagnostics. Equivalent to A is not
   unchanged: the ±0.5-nat margin allows a change in probability by a factor between 0.61 and
   1.65 (e^−0.5 to e^0.5), the ±0.10-nat margin a factor between 0.90 and 1.11 per lexical
   continuation token. The falsifier concerns the lexical continuation only and says nothing
   about the punctuation or the formatting.

The reading of the two sets (``READING``, the same text):

   The primary set (a_prefix) and the prompt-only control are read together as follows, fixed
   before scoring: agreement of the two sets is robustness across these two conditions;
   divergence is context dependence with no cause identified (the sets differ in prefix, marker
   position and prompt pool); an uninformative control neither confirms nor refutes the primary
   result; the control never replaces an unfavourable primary result.

The normalization (``NORMALIZATION``, the same text):

   The normalized effects (q_L − q_Hs)/(q_A − q_H) and (q_L − q_X)/(q_A − q_H) are reported
   with the A − H interval and the numbers of bootstrap resamples whose A − H mean is ≤ 0,
   non-finite, or of the other sign than its point estimate. A normalized effect is undefined
   when the A − H interval includes 0 or, for a quantity with margins, when its point estimate
   is below that quantity's material margin; the raw paired differences are the result.

The Pod session is ``scripts/pod/follow_marker_likelihood.py``;
``scripts/follow_marker_likelihood.py`` plans it on the laptop and prints a record's summary.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .behaviors import Detector

BEHAVIOR = "eli5"
EXPERIMENT = "experiments/continuation_14b/eli5"
DETECTOR_VERSION = "43943945ee35"
MODEL = {
    "name": "Qwen/Qwen3-14B",
    "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "dtype": "bfloat16",
    "chat_template_kwargs": {"enable_thinking": False},
}
MAX_LENGTH = 4096  # the pilot's likelihood observer's (pipeline.vllm.max_model_len)
# The scoring path's precision: the base in bf16, PEFT's adapters upcast to fp32 (its
# autocast_adapter_dtype), the head's matmul in bf16 and its logits upcast to fp32 before the
# log-softmax (training.response_logits); the session refuses another.
PRECISION = {"base": "torch.bfloat16", "adapters": "torch.float32", "head": "torch.bfloat16",
             "log_softmax": "torch.float32"}  # fmt: skip
SCORING_REVISION = 1  # raised whenever the scoring code changes what a value means

A_STATE, H_STATE = "s46157d0cf7a1b951", "s0fe31042a3de8cff"
INSERTED_MARKER = "ELI5: "
CONTEXT_SETS: dict[str, dict[str, Any]] = {
    "a_prefix": {
        "role": "primary, conditional on A's prefixes",
        "run": "20261022-110131-rates-0a4e71c3-20e2",
        "files": {
            "observations/detector_obs.jsonl":
                "0daae9a12637b47dcdb14f6f82b51ba6a8bf2d8eadc7f018e768b4238b0699dc",
            "responses/detector_obs.jsonl":
                "4cbb3bc54474b89a3e61ccd660ed14530223306bc290b79db9ed746bbc2d0ba6",
        },
        "params": {"state": A_STATE, "prompt_set": "relearn_eval",
                   "measurement": "relearn_reference", "samples": 1,
                   "requests": "64f38627f325186bba4b66964a2718278f802fc522f5ba817d582cfca34378bc"},
        "select": "detector hit",
        "insert": None,
        "max": 200,
    },
    "prompt_only": {
        "role": "prompt-only control: the marker inserted at the start of H's responses",
        "run": "20261011-132008-rates-e03d2bb7-9824",
        "files": {
            "observations/detector_obs.jsonl":
                "056eb6d3b6c7e0cc9a05fb645dd3f3eeb5c2e39a8d965e6955992e2d54285126",
            "responses/detector_obs.jsonl":
                "deb51917612a290c1a9d48347d713802c27e3dbebf24cdb08e269984eb304ff4",
        },
        "params": {"state": H_STATE, "prompt_set": "certificate", "measurement": "observer",
                   "samples": 1,
                   "requests": "161586a518fe30352fbfaab6b5bf50cfe5d4aa0d7a8f5f76c877ea05dd7f7e58"},
        "select": "no detector hit, not truncated",
        "insert": INSERTED_MARKER,
        "max": 200,
    },
}  # fmt: skip
CONTEXT_RUN = CONTEXT_SETS["a_prefix"]["run"]

REFERENCES: dict[str, str | None] = {"base": None, "A": A_STATE, "H": H_STATE}
FAMILIES: dict[str, dict[str, str | None]] = {
    "stack_200": {"L": "s4b9dc0f2018682b8", "H_s": "s6d0985cfc3f27b80", "X": "s911a0722bb150a83"},
    "stack_800": {"L": "s0ca147609c1f980d", "H_s": "s48b51cf3b3ae89af", "X": "sa4d700fc9f94583f"},
    "stack_400": {"L": "se244833f590245e7", "H_s": "scc5816364b08f649", "X": None},
    "cont_200": {"L": "s71890e0779caf9bc", "H_s": "s18201c59f9cde5ab", "X": "s270cab1ab51c940b"},
}
OPTIONAL = ("stack_400",)
COMPOSE = {"stack_400": {"suppressed": "se244833f590245e7", "sham": H_STATE}}
ROLES = ("L", "H_s", "X")

# Where each frozen id is recorded: (label, committed file, key path); a mapping in a path picks
# the one list entry with those fields.
SOURCES: tuple[tuple[str, str, tuple[Any, ...]], ...] = (
    ("A", "manifests/stage3_readers_strong_eli5_cont_200.json", ("states", "A")),
    ("H", "manifests/stage3_readers_strong_eli5_cont_200.json", ("states", "H")),
    ("A", "manifests/kstar_follow_eli5_stack_200.json", ("states", "A")),
    ("stack_200/H_s", "manifests/kstar_follow_eli5_stack_200.json", ("states", "H_s")),
    *(
        (f"stack_200/{role}", "configs/continuation/follow_build.yaml",
         ("build", "sessions", "b1", "builds", {"lineage": "eli5 stack 200", "role": role},
          "state"))
        for role in ROLES
    ),  # fmt: skip
    *((f"stack_800/{role}", "manifests/follow_stack_build.json", ("states", role))
      for role in ROLES),  # fmt: skip
    ("stack_400/L", "manifests/follow_stack_eli5.json", ("screened", {"dose": 400}, "latent")),
    ("stack_400/H_s", "manifests/follow_stack_eli5.json", ("screened", {"dose": 400}, "sham_s")),
    *((f"cont_200/{role}", "manifests/stage3_readers_strong_eli5_cont_200.json", ("states", role))
      for role in ROLES),  # fmt: skip
    ("cont_200/L", "manifests/strong_eli5_cont.json", ("chosen", "latent")),
    ("cont_200/H_s", "manifests/strong_eli5_cont.json", ("chosen", "sham_s")),
    ("context_run", "manifests/kstar_follow_eli5_stack_200.json", ("reference", "run")),
    ("context_state", "manifests/kstar_follow_eli5_stack_200.json", ("reference", "state")),
    ("context_run", "manifests/kstar_strong_eli5_cont_200.json", ("reference", "run")),
)

# The quantities in summary order; per-token ones (lexical_k, punct_k) follow them.
QUANTITIES = ("first_subtoken", "continuation", "joint", "joint_lexical", "punctuation", "rest",
              "before")  # fmt: skip
# Frozen before scoring, in nats: (equivalence half-width, material change).
MARGINS = {"first_subtoken": (0.5, 1.0), "continuation": (0.10, 0.25), "joint": (0.5, 1.0)}
STRATA = ("all", "formatted", "plain")
LEVEL = 0.95
N_BOOT = 2000
SEED = 0

RULES = (
    "Margins, in nats, frozen before scoring: the observed first subtoken's log p, equivalence "
    "±0.5 and material change 1.0; the lexical continuation (the mean log p per lexical marker "
    "token after the first), equivalence ±0.10 and material change 0.25; the joint marker log p "
    "(the sum over all marker tokens), equivalence ±0.5 and material change 1.0. For each of these "
    "quantities and each state S of a family (L, H_s, X), the paired difference S − A over "
    "contexts, with its 95 % context-bootstrap interval, classifies S: equivalent to A if the "
    "interval lies inside ±equivalence; materially reduced if its upper bound is below −material; "
    "materially increased if its lower bound is above +material; otherwise inconclusive. A "
    "quantity is informative for a family only if the A − H interval excludes 0 and its point "
    "estimate exceeds that quantity's material margin; otherwise every state's class for it is "
    "uninformative (L ≈ A ≈ H is never evidence of retained learning). A state's outcome, from "
    "its first-subtoken and lexical-continuation classes: both retained (both equivalent to A); "
    "both reduced (both materially reduced); initiation-only reduced (first subtoken materially "
    "reduced, continuation equivalent); reversed (continuation materially reduced, first subtoken "
    "equivalent); partial (one materially reduced, the other inconclusive); uninformative (either "
    "uninformative); otherwise inconclusive. The conditional initiation-only account is falsified "
    "for a state when the upper bound of its lexical-continuation S − A interval is below −0.25, "
    "even if its continuation lies above H_s's, whatever the separation. These classes, outcomes "
    "and the falsifier are read on each context set's full set of contexts only; the strata "
    "(formatted, plain) report estimates and intervals as diagnostics. Equivalent to A is not "
    "unchanged: the ±0.5-nat margin allows a change in probability by a factor between 0.61 and "
    "1.65 (e^−0.5 to e^0.5), the ±0.10-nat margin a factor between 0.90 and 1.11 per lexical "
    "continuation token. The falsifier concerns the lexical continuation only and says nothing "
    "about the punctuation or the formatting."
)
READING = (
    "The primary set (a_prefix) and the prompt-only control are read together as follows, fixed "
    "before scoring: agreement of the two sets is robustness across these two conditions; "
    "divergence is context dependence with no cause identified (the sets differ in prefix, marker "
    "position and prompt pool); an uninformative control neither confirms nor refutes the primary "
    "result; the control never replaces an unfavourable primary result."
)
NORMALIZATION = (
    "The normalized effects (q_L − q_Hs)/(q_A − q_H) and (q_L − q_X)/(q_A − q_H) are reported "
    "with the A − H interval and the numbers of bootstrap resamples whose A − H mean is ≤ 0, "
    "non-finite, or of the other sign than its point estimate. A normalized effect is undefined "
    "when the A − H interval includes 0 or, for a quantity with margins, when its point estimate "
    "is below that quantity's material margin; the raw paired differences are the result."
)

DESIGN = {
    "behavior": BEHAVIOR,
    "experiment": EXPERIMENT,
    "detector": DETECTOR_VERSION,
    "model": MODEL,
    "precision": PRECISION,
    "scoring_revision": SCORING_REVISION,
    "max_length": MAX_LENGTH,
    "context_sets": CONTEXT_SETS,
    "marker": "first detector match (lexical) + a directly following ':'",
    "references": REFERENCES,
    "families": FAMILIES,
    "optional": list(OPTIONAL),
    "compose": COMPOSE,
    "quantities": list(QUANTITIES),
    "margins": MARGINS,
    "strata": list(STRATA),
    "rules": RULES,
    "reading": READING,
    "normalization": NORMALIZATION,
    "bootstrap": {"level": LEVEL, "n_boot": N_BOOT, "seed": SEED},
}


def canonical_sha256(value: Any) -> str:
    """sha256 of the canonical JSON of ``value`` (``pipeline.content_hash``'s form)."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


DESIGN_SHA256 = canonical_sha256(DESIGN)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scoring_identity() -> dict[str, Any]:
    """The scoring revision and the sha256 of the source of every function that turns a context
    into its values, which a cached state must share."""
    import inspect

    from . import audit, training

    functions = [audit.response_token_logprobs, audit.response_token_spans, training.collate,
                 training.encode_example, training.response_logits, token_split, state_values,
                 scorer]  # fmt: skip
    source = "\n".join(inspect.getsource(f) for f in functions)
    return {"revision": SCORING_REVISION,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}  # fmt: skip


# ------------------------------------------------------------------------------------- states


def state_labels(*, with_400: bool, x_400: str | None = None) -> dict[str, str | None]:
    """Label → state id of every state the session reads, in reading order: the base model, A, H,
    then each family's L, H_s and X (``<family>/<role>``); stack 400 only ``with_400``, its X
    the composed ``x_400``."""
    if with_400 and x_400 is None:
        raise ValueError("stack 400 is read only with its composed X")
    out = dict(REFERENCES)
    for family, roles in FAMILIES.items():
        if family in OPTIONAL and not with_400:
            continue
        for role, sid in roles.items():
            out[f"{family}/{role}"] = x_400 if (family, role) == ("stack_400", "X") else sid
    return out


def families_read(labels: Mapping[str, str | None]) -> list[str]:
    return [f for f in FAMILIES if all(f"{f}/{r}" in labels for r in ROLES)]


def _pick(data: Any, path: Sequence[Any]) -> Any:
    for key in path:
        if isinstance(key, Mapping):
            found = [x for x in data if all(x.get(k) == v for k, v in key.items())]
            if len(found) != 1:
                raise KeyError(f"{len(found)} entries with {dict(key)}")
            data = found[0]
        else:
            data = data[key]
    return data


def frozen_value(label: str) -> str | None:
    if label == "context_run":
        return CONTEXT_RUN
    if label == "context_state":
        return A_STATE
    if label in REFERENCES:
        return REFERENCES[label]
    family, role = label.split("/")
    return FAMILIES[family][role]


def source_problems(root: Path) -> tuple[list[str], dict[str, str]]:
    """Why a frozen id is not the one its committed record names (``SOURCES``), and each source
    file's sha256."""
    import yaml

    problems, shas = [], {}
    for label, rel, path in SOURCES:
        file = Path(root) / rel
        if not file.is_file():
            problems.append(f"{rel} is missing")
            continue
        shas[rel] = file_sha256(file)
        text = file.read_text()
        data = yaml.safe_load(text) if rel.endswith(".yaml") else json.loads(text)
        try:
            recorded = _pick(data, path)
        except (KeyError, IndexError, TypeError) as e:
            problems.append(f"{rel}: no entry for {label} ({e})")
            continue
        if recorded != frozen_value(label):
            problems.append(f"{label}: {rel} records {recorded}, the design {frozen_value(label)}")
    return problems, shas


def runtime_problems(model_cfg: Mapping[str, Any], detector: Detector, max_length: int,
                     behavior_id: str) -> list[str]:  # fmt: skip
    """Why the session's configuration is not the design's: its model, behavior, detector and
    likelihood text length."""
    problems = []
    model = {k: model_cfg.get(k) for k in MODEL}
    if model != MODEL:
        problems.append(f"the model is {model}, not {MODEL}")
    if behavior_id != BEHAVIOR:
        problems.append(f"the behavior is {behavior_id}, not {BEHAVIOR}")
    if detector.version != DETECTOR_VERSION:
        problems.append(f"the detector is {detector.version}, not {DETECTOR_VERSION}")
    if int(max_length) != MAX_LENGTH:
        problems.append(f"the text length is {max_length}, not {MAX_LENGTH}")
    return problems


def verify_states(labels: Mapping[str, str | None], verify: Callable[[str], None]) -> None:
    """Every state's adapters against the tensor sha256 its manifest records
    (``StateStore.verify``), before any model work; the first mismatch refuses the session."""
    for label, sid in labels.items():
        if sid is None:
            continue
        try:
            verify(sid)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"{label} ({sid}) refused: {e}") from e


# ----------------------------------------------------------------------------------- contexts


@dataclass(frozen=True)
class Context:
    prompt_id: str
    prompt: str
    response: str
    marker: tuple[int, int, int]  # in the response: start, end of the lexical match, end
    truncated: bool


@dataclass(frozen=True)
class Split:
    """Indices into a context's supervised tokens (``audit.response_token_logprobs``' order)."""

    before: tuple[int, ...]
    lexical: tuple[int, ...]
    punct: tuple[int, ...]
    rest: tuple[int, ...]
    n_tokens: int  # every supervised token, the turn's end included
    stratum: str = "plain"  # formatted, plain, or none (no punctuation token)
    token_ids: tuple[int, ...] = ()  # the supervised tokens, when known

    def as_list(self) -> list[Any]:
        return [list(self.before), list(self.lexical), list(self.punct), list(self.rest),
                self.n_tokens, self.stratum, list(self.token_ids)]  # fmt: skip


def marker_span(response: str, detector: Detector) -> tuple[int, int, int] | None:
    """The marker in ``response``: the detector's first match (its lexical part), with a colon
    that directly follows it; None without a match (or when ``Detector.match_spans`` gives no
    offsets)."""
    spans = detector.match_spans(response)
    if not spans:
        return None
    start, lexical_end = spans[0]
    end = lexical_end + 1 if response.startswith(":", lexical_end) else lexical_end
    return start, lexical_end, end


def load_contexts(run_dir: Path, prompts: Sequence[Any], detector: Detector,
                  spec: Mapping[str, Any]) -> tuple[list[Context], dict[str, Any]]:  # fmt: skip
    """A context set (design item 1) from its saved pass ``run_dir``, refused unless its files
    are the pinned ones, its config the pinned pass (state, prompt set, measurement, samples,
    requests hash of ``prompts``, model, detector) and every row the pinned state's and
    detector's; with what was read and left out."""
    import yaml

    from .generation import user_request
    from .pipeline import requests_hash

    run_dir = Path(run_dir)
    files, params = spec["files"], spec["params"]
    problems = []
    shas = {rel: file_sha256(run_dir / rel) if (run_dir / rel).is_file() else None
            for rel in files}  # fmt: skip
    for rel, want in files.items():
        if shas[rel] != want:
            problems.append(f"{rel} is {shas[rel]}, not the pinned {want}")
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    got = {k: (config.get("params") or {}).get(k) for k in params}
    requests = requests_hash([user_request(p.prompt_id, p.text) for p in prompts])
    if got != dict(params):
        problems.append(f"the run's params are {got}, not {dict(params)}")
    if requests != params["requests"]:
        problems.append(f"the prompts hash to {requests}, not the run's {params['requests']}")
    if meta.get("status") != "completed" or meta.get("run_id") != run_dir.name:
        problems.append(f"the run {run_dir.name} is not a completed run of that id")
    model = {k: (config.get("model") or {}).get(k) for k in MODEL}
    if model != MODEL:
        problems.append(f"the run's model is {model}, not {MODEL}")
    det = (config.get("behavior") or {}).get("detector") or {}
    if Detector(**det).version != detector.version:
        problems.append("the run's detector is not the design's")
    if problems:
        raise ValueError("; ".join(problems))
    texts = {}
    for line in (run_dir / "responses/detector_obs.jsonl").read_text().splitlines():
        row = json.loads(line)
        texts[row["response_id"]] = row["text"]
    by_id = {p.prompt_id: p.text for p in prompts}
    rows = [json.loads(x) for x in (run_dir / "observations/detector_obs.jsonl").read_text()
            .splitlines()]  # fmt: skip
    insert = spec.get("insert")
    seen: set[str] = set()
    contexts, excluded = [], []
    eligible = 0
    for row in rows:
        if (row["state_id"], row["behavior_id"], row["detector_version"], row["sample"]) != (
                params["state"], BEHAVIOR, detector.version, 0):  # fmt: skip
            raise ValueError(f"row {row['response_id']} is not the pinned pass's")
        if row["prompt_id"] in seen:
            raise ValueError(f"the run has several responses to {row['prompt_id']}")
        seen.add(row["prompt_id"])
        response = texts[row["response_id"]]
        if bool(row["hit"]) != detector(response):
            raise ValueError(f"{row['response_id']}: the saved verdict is not the detector's")
        truncated = bool(row.get("truncated"))
        if insert is None:
            if not row["hit"]:
                continue
        else:
            if row["hit"] or truncated:
                continue
            response = insert + response
        eligible += 1
        if len(contexts) == spec["max"]:
            continue
        span = marker_span(response, detector)
        if span is None or (insert is not None and span[0] != 0):
            excluded.append({"prompt_id": row["prompt_id"], "reason": "no marker offsets"})
            continue
        contexts.append(Context(row["prompt_id"], by_id[row["prompt_id"]], response, span,
                                truncated))  # fmt: skip
    info = {"run": run_dir.name, "files": shas, "rows": len(rows), "eligible": eligible,
            "selected": len(contexts), "excluded": excluded,
            "contexts_sha256": contexts_sha256(contexts)}  # fmt: skip
    return contexts, info


def contexts_sha256(contexts: Sequence[Context]) -> str:
    return canonical_sha256([[c.prompt_id, c.prompt, c.response, list(c.marker)]
                             for c in contexts])  # fmt: skip


def splits_sha256(splits: Sequence[Split]) -> str:
    return canonical_sha256([s.as_list() for s in splits])


def token_split(token_spans: Sequence[tuple[int, int]], marker: tuple[int, int, int],
                response: str = "", token_ids: Sequence[int] = ()) -> Split | None:  # fmt: skip
    """The before, lexical, punctuation and rest tokens of one context (design item 2) from its
    supervised tokens' character spans (``audit.response_token_spans``: the turn's end tokens are
    empty); None when no token overlaps the lexical match."""
    start, lexical_end, end = marker
    text = [i for i, (s, e) in enumerate(token_spans) if e > s]

    def overlaps(i: int, a: int, b: int) -> bool:
        return token_spans[i][0] < b and token_spans[i][1] > a

    inside = [i for i in text if overlaps(i, start, end)]
    lexical = [i for i in inside if overlaps(i, start, lexical_end)]
    if not lexical:
        return None
    if inside != list(range(inside[0], inside[-1] + 1)) or inside[0] != lexical[0]:
        raise ValueError("the marker's tokens are not contiguous from its lexical start")
    punct = [i for i in inside if i not in lexical]
    punct_text = "".join(response[token_spans[i][0]:token_spans[i][1]] for i in punct)
    stratum = "none" if not punct else "plain" if punct_text == ":" else "formatted"
    return Split(before=tuple(i for i in text if i < inside[0]), lexical=tuple(lexical),
                 punct=tuple(punct), rest=tuple(i for i in text if i > inside[-1]),
                 n_tokens=len(token_spans), stratum=stratum,
                 token_ids=tuple(int(t) for t in token_ids))  # fmt: skip


def supervised_ids(tokenizer: Any, context: Context, chat_kwargs: Mapping[str, Any]) -> list[int]:
    from .training import encode_example

    _, labels = encode_example(tokenizer, context.prompt, context.response, chat_kwargs,
                               MAX_LENGTH + 1)  # fmt: skip
    return [int(t) for t in labels if t != -100]


def prepare(tokenizer: Any, contexts: Sequence[Context], chat_kwargs: Mapping[str, Any]
            ) -> tuple[list[Context], list[Split], list[dict[str, Any]]]:  # fmt: skip
    """The contexts every state is read on, their token splits (with their token ids), and the
    ones left out (the template does not render the response verbatim, or no token overlaps the
    marker)."""
    from .audit import response_token_spans

    kept, splits, excluded = [], [], []
    for c in contexts:
        spans = response_token_spans(tokenizer, c.prompt, c.response, chat_kwargs)
        split = None
        if spans is not None:
            ids = supervised_ids(tokenizer, c, chat_kwargs)
            if len(ids) != len(spans):
                raise ValueError(f"{c.prompt_id}: {len(ids)} token ids, {len(spans)} spans")
            split = token_split(spans, c.marker, c.response, ids)
        if split is None:
            reason = "response not rendered verbatim" if spans is None else "no marker token"
            excluded.append({"prompt_id": c.prompt_id, "reason": reason})
            continue
        kept.append(c)
        splits.append(split)
    return kept, splits, excluded


def describe(tokenizer: Any, contexts: Sequence[Context], splits: Sequence[Split],
             chat_kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """Per context, what the record shows of it: its prompt id, marker text and span, the
    marker's lexical and punctuation tokens as text and ids, its stratum, and token counts."""
    from .audit import response_token_spans
    from .training import encode_example

    out = []
    for c, split in zip(contexts, splits, strict=True):
        spans = response_token_spans(tokenizer, c.prompt, c.response, chat_kwargs)
        a, _, b = c.marker

        def text(indices: Sequence[int], spans=spans, c=c) -> list[str]:
            return [c.response[spans[i][0]:spans[i][1]] for i in indices]

        ids, _ = encode_example(tokenizer, c.prompt, c.response, chat_kwargs, MAX_LENGTH + 1)
        out.append({
            "prompt_id": c.prompt_id, "marker": list(c.marker), "marker_text": c.response[a:b],
            "lexical_tokens": text(split.lexical), "punct_tokens": text(split.punct),
            "lexical_ids": [split.token_ids[i] for i in split.lexical],
            "punct_ids": [split.token_ids[i] for i in split.punct],
            "lead": c.response[spans[split.lexical[0]][0]:a], "stratum": split.stratum,
            "tokens": len(ids), "before": len(split.before),
            "rest": len(split.rest), "truncated": c.truncated,
        })  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- quantities


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def state_values(token_logprobs: Sequence[Sequence[float]], splits: Sequence[Split]
                 ) -> dict[str, list[Any]]:  # fmt: skip
    """One state's stored values (design item 3): per context the log-probabilities of the
    lexical and of the punctuation marker tokens, and the mean log-probability of the rest and of
    the response before the marker; refused when a context's tokens are not its split's."""
    lexical, punct, rest, before = [], [], [], []
    for lps, split in zip(token_logprobs, splits, strict=True):
        if len(lps) != split.n_tokens:
            raise ValueError(f"{len(lps)} token log-probabilities, the split has "
                             f"{split.n_tokens} tokens")  # fmt: skip
        lexical.append([float(lps[i]) for i in split.lexical])
        punct.append([float(lps[i]) for i in split.punct])
        rest.append(_mean([float(lps[i]) for i in split.rest]))
        before.append(_mean([float(lps[i]) for i in split.before]))
    return {"lexical": lexical, "punct": punct, "rest": rest, "before": before}


def quantity(values: Mapping[str, list[Any]], name: str) -> list[float | None]:
    """A quantity per context from a state's stored values."""
    lex, punct = values["lexical"], values["punct"]
    if name == "first_subtoken":
        return [m[0] for m in lex]
    if name == "continuation":
        return [_mean(m[1:]) for m in lex]
    if name == "joint":
        return [float(sum(m) + sum(p)) for m, p in zip(lex, punct, strict=True)]
    if name == "joint_lexical":
        return [float(sum(m)) for m in lex]
    if name == "punctuation":
        return [_mean(p) for p in punct]
    if name in ("rest", "before"):
        return list(values[name])
    for prefix, rows in (("lexical_", lex), ("punct_", punct)):
        if name.startswith(prefix):
            k = int(name.removeprefix(prefix))
            return [m[k] if len(m) > k else None for m in rows]
    raise ValueError(f"unknown quantity {name}")


def token_quantities(values: Mapping[str, list[Any]]) -> list[str]:
    """The per-token quantities a state's values hold: every lexical and punctuation position."""
    lex = max((len(m) for m in values["lexical"]), default=0)
    punct = max((len(p) for p in values["punct"]), default=0)
    return [f"lexical_{k}" for k in range(lex)] + [f"punct_{k}" for k in range(punct)]


def _interval(samples: np.ndarray) -> list[float]:
    from .analysis import percentile_interval

    return list(percentile_interval(samples, LEVEL))


def _picks(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(n, size=(N_BOOT, n))


def _mean_interval(x: np.ndarray) -> list[float]:
    return _interval(x[_picks(len(x))].mean(axis=1))


def classify(ci: Sequence[float], equivalence: float, material: float) -> str:
    """A state's class for one quantity from its S − A interval (``RULES``)."""
    low, high = ci
    if -equivalence <= low and high <= equivalence:
        return "equivalent to A"
    if high < -material:
        return "materially reduced"
    if low > material:
        return "materially increased"
    return "inconclusive"


def separated(point: float, ci: Sequence[float], material: float) -> bool:
    """Whether A − H separates the active from the control for one quantity (``RULES``)."""
    low, high = ci
    return (low > 0 or high < 0) and point > material


def outcome(first: str, continuation: str) -> str:
    """A state's outcome from its first-subtoken and lexical-continuation classes (``RULES``)."""
    reduced, equivalent = "materially reduced", "equivalent to A"
    if "uninformative" in (first, continuation):
        return "uninformative"
    if first == continuation == equivalent:
        return "both retained"
    if first == continuation == reduced:
        return "both reduced"
    if (first, continuation) == (reduced, equivalent):
        return "initiation-only reduced"
    if (first, continuation) == (equivalent, reduced):
        return "reversed"
    if {first, continuation} == {reduced, "inconclusive"}:
        return "partial"
    return "inconclusive"


def denominator(d: np.ndarray, picks: np.ndarray, name: str) -> dict[str, Any]:
    """The A − H denominator of one quantity (``NORMALIZATION``): its point and interval, the
    resamples whose mean is ≤ 0, non-finite or of the other sign, and whether the normalized
    effects are defined, with the reason when not."""
    boot = d[picks].mean(axis=1)
    point = float(d.mean())
    ci = _interval(boot)
    finite = np.isfinite(boot)
    sign = np.sign(point)
    out = {"point": point, "ci": ci, "nonpositive": int((boot[finite] <= 0).sum()),
           "nonfinite": int((~finite).sum()),
           "sign_changes": int((np.sign(boot[finite]) != sign).sum())}  # fmt: skip
    reason = None
    if not ci or any(math.isnan(x) for x in ci) or (ci[0] <= 0 <= ci[1]):
        reason = "the A − H interval includes 0"
    elif name in MARGINS and point < MARGINS[name][1]:
        reason = f"A − H is below the material margin {MARGINS[name][1]}"
    out["defined"] = reason is None
    out["reason"] = reason
    return out


def family_contrasts(per: Mapping[str, Sequence[float | None]], name: str, *,
                     classes: bool = True) -> dict[str, Any]:  # fmt: skip
    """One family's contrasts of one quantity: ``per`` holds each role's values (L, H_s, X, A, H)
    on the same contexts. The paired differences with their intervals, the denominator, the
    normalized effects, and, with ``classes`` (a full context set), for a margin quantity each
    state's class against A."""
    keep = [i for i in range(len(per["A"])) if all(v[i] is not None for v in per.values())]
    if not keep:
        return {"n": 0, "problem": "no context has the quantity in every role"}
    x = {r: np.array([v[i] for i in keep], float) for r, v in per.items()}
    picks = _picks(len(keep))
    pairs = {"L-A": ("L", "A"), "H_s-A": ("H_s", "A"), "X-A": ("X", "A"),
             "L-H_s": ("L", "H_s"), "L-X": ("L", "X"), "A-H": ("A", "H")}  # fmt: skip
    differences, boots = {}, {}
    for key, (a, b) in pairs.items():
        d = x[a] - x[b]
        boots[key] = d[picks].mean(axis=1)
        differences[key] = {"point": float(d.mean()), "ci": _interval(boots[key])}
    den = denominator(x["A"] - x["H"], picks, name)
    normalized: dict[str, Any] = {}
    for key in ("L-H_s", "L-X"):
        if not den["defined"]:
            normalized[key] = {"point": None, "ci": None, "undefined": den["reason"]}
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = boots[key] / boots["A-H"]
        normalized[key] = {"point": differences[key]["point"] / den["point"],
                           "ci": _interval(ratios),
                           "nonfinite": int((~np.isfinite(ratios)).sum())}  # fmt: skip
    out: dict[str, Any] = {"n": len(keep), "differences": differences, "denominator": den,
                           "normalized": normalized}  # fmt: skip
    if classes and name in MARGINS:
        eq, material = MARGINS[name]
        informative = separated(den["point"], differences["A-H"]["ci"], material)
        out["informative"] = informative
        out["classes"] = {r: classify(differences[f"{r}-A"]["ci"], eq, material)
                          if informative else "uninformative" for r in ROLES}  # fmt: skip
    return out


def summarize_set(labels: Mapping[str, str | None], values: Mapping[str, Mapping[str, list]],
                  strata: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """One context set's summary (design item 4), per stratum: the families' contrasts (raw
    paired differences first) and the states' means; the classes, outcomes and falsifier on the
    full set ("all") only, the strata being diagnostics (``RULES``). ``values`` is keyed by
    ``value_key`` of each state; ``strata`` gives each context's stratum."""
    a_values = values[value_key(labels["A"])]
    names = [*QUANTITIES, *token_quantities(a_values)]
    out: dict[str, Any] = {}
    for stratum in STRATA:
        idx = [i for i, s in enumerate(strata) if stratum == "all" or s == stratum]
        full = stratum == "all"
        block: dict[str, Any] = {"contexts": len(idx), "families": {}, "states": {}}
        if full:
            block["outcomes"] = {}
        if not idx:
            out[stratum] = block
            continue

        def q(sid: str | None, name: str, idx=idx) -> list[float | None]:
            v = quantity(values[value_key(sid)], name)
            return [v[i] for i in idx]

        for family in families_read(labels):
            roles = {r: labels[f"{family}/{r}"] for r in ROLES} | {"A": labels["A"],
                                                                   "H": labels["H"]}
            contrasts = {name: family_contrasts({r: q(s, name) for r, s in roles.items()}, name,
                                                classes=full) for name in names}  # fmt: skip
            block["families"][family] = contrasts
            if not full:
                continue
            states = {}
            for r in ROLES:
                first = contrasts["first_subtoken"].get("classes", {}).get(r, "uninformative")
                cont = contrasts["continuation"].get("classes", {}).get(r, "uninformative")
                ci = contrasts["continuation"].get("differences", {}).get(f"{r}-A", {}).get("ci")
                states[r] = {"first_subtoken": first, "continuation": cont,
                             "joint": contrasts["joint"].get("classes", {}).get(
                                 r, "uninformative"),
                             "outcome": outcome(first, cont),
                             "initiation_only_falsified": bool(
                                 ci is not None and ci[1] < -MARGINS["continuation"][1])}
            block["outcomes"][family] = states
        for label, sid in labels.items():
            row: dict[str, Any] = {"state": sid}
            for name in names:
                x = np.array([v for v in q(sid, name) if v is not None], float)
                row[name] = ({"mean": float(x.mean()), "ci": _mean_interval(x), "n": int(len(x))}
                             if len(x) else {"mean": None, "ci": None, "n": 0})  # fmt: skip
                if name == "first_subtoken" and len(x):
                    row[name]["mean_probability"] = float(np.exp(x).mean())
            block["states"][label] = row
        out[stratum] = block
    return out


def summarize(labels: Mapping[str, str | None], sets: Mapping[str, Mapping[str, Any]]
              ) -> dict[str, Any]:  # fmt: skip
    """The summary of every context set read: ``sets[name]`` holds its ``values`` (by state) and
    its contexts' ``strata``."""
    return {name: summarize_set(labels, s["values"], s["strata"]) for name, s in sets.items()}


def value_key(sid: str | None) -> str:
    return "base" if sid is None else sid


# ---------------------------------------------------------------------------------- scoring


def observed_precision(model: Any, *, adapter: bool) -> dict[str, str | None]:
    """The dtypes the scoring path runs in: the base's embeddings, the adapters' LoRA weights
    (None without an adapter), the head's weights, and the log-softmax (fp32 by
    ``training.response_logits``)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    lora = next((str(p.dtype) for n, p in model.named_parameters() if "lora_" in n), None)
    return {"base": str(inner.get_input_embeddings().weight.dtype),
            "adapters": lora if adapter else None, "head": str(inner.lm_head.weight.dtype),
            "log_softmax": "torch.float32"}  # fmt: skip


def scorer(base: Any, tokenizer: Any, served: Callable[[str], Path], *,
           chat_kwargs: Mapping[str, Any], max_length: int, device: str = "cuda",
           expect: Mapping[str, Any] | None = None) -> Callable[..., Any]:  # fmt: skip
    """``score(state, texts)`` → (each text's supervised token log-probabilities, the observed
    precision): the state's served adapter (``served``) loaded onto ``base`` by PEFT, scored by
    ``audit.response_token_logprobs`` and unloaded again (the base model for None). With
    ``expect``, a precision other than it refuses the state."""
    from peft import PeftModel

    from .audit import response_token_logprobs

    held = {"base": base}

    def score(sid: str | None, texts: list[tuple[str, str]]) -> tuple[list[list[float]], dict]:
        model = held["base"]
        if sid is not None:
            model = PeftModel.from_pretrained(held["base"], served(sid), adapter_name="state")
        try:
            precision = observed_precision(model, adapter=sid is not None)
            want = None if expect is None else {
                **expect, "adapters": expect["adapters"] if sid is not None else None}
            if want is not None and precision != want:
                raise ValueError(f"{value_key(sid)} runs at {precision}, not {want}")
            lps = response_token_logprobs(model, tokenizer, texts, chat_kwargs=chat_kwargs,
                                          max_length=max_length, device=device)  # fmt: skip
        finally:
            if sid is not None:
                held["base"] = model.unload()
        return lps, precision

    return score


# ---------------------------------------------------------------------------------- execution

RECORD = "marker_likelihood.json"
# The environment fields a resumed state must share with the session (not its host or Pod).
EXECUTION_KEYS = ("gpu", "cuda", "torch", "transformers", "peft", "attention", "dtype")


@dataclass(frozen=True)
class ContextSet:
    contexts: Sequence[Context]
    splits: Sequence[Split]
    rows: Sequence[Mapping[str, Any]]
    info: Mapping[str, Any]


def binding(name: str, cset: ContextSet, scoring: Mapping[str, Any]) -> dict[str, Any]:
    return {"design_sha256": DESIGN_SHA256, "scoring": dict(scoring), "set": name,
            "contexts_sha256": contexts_sha256(cset.contexts),
            "splits_sha256": splits_sha256(cset.splits)}  # fmt: skip


def load_cached(path: Path, sid: str | None, bound: Mapping[str, Any],
                splits: Sequence[Split], environment: Mapping[str, Any]) -> dict[str, Any]:
    """A cached state's values and provenance, refused unless it is this state's, bound to the
    same design, scoring, context set, token ids and splits, read in the same execution
    environment (``EXECUTION_KEYS``), and its arrays have every context's split's shape."""
    saved = json.loads(Path(path).read_text())
    problems = []
    if saved.get("state", "?") != sid:
        problems.append(f"it is {saved.get('state')}'s")
    for k, v in bound.items():
        if saved.get(k) != v:
            problems.append(f"its {k} differs")
    env = (saved.get("provenance") or {}).get("environment") or {}
    keys = {k: env.get(k) for k in EXECUTION_KEYS}
    if keys != {k: environment.get(k) for k in EXECUTION_KEYS}:
        problems.append(f"it was read in {keys}")
    values = saved.get("values") or {}
    try:
        shapes = ([len(x) for x in values["lexical"]], [len(x) for x in values["punct"]],
                  len(values["rest"]), len(values["before"]))  # fmt: skip
    except (KeyError, TypeError):
        shapes = None
    want = ([len(s.lexical) for s in splits], [len(s.punct) for s in splits], len(splits),
            len(splits))  # fmt: skip
    if shapes != want:
        problems.append("its arrays are not the splits' shape")
    if problems:
        raise ValueError(f"{path} cannot be combined: " + "; ".join(problems))
    return saved


def execute(out: Path, manifest: Mapping[str, Any], *, labels: Mapping[str, str | None],
            sets: Mapping[str, ContextSet], score: Callable[..., Any],
            verify: Callable[[str], None], scoring: Mapping[str, Any] | None = None,
            clock: Callable[[], str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Read every state on every context set and write the record ``out/marker_likelihood.json``.

    Every state is verified first (``verify_states``). ``score(state, texts)`` gives each
    (prompt, response)'s supervised token log-probabilities and the observed precision
    (``scorer``; the base model for None). A state's values are written to
    ``out/states/<set>/<state>.json`` with their provenance (``manifest["environment"]``, the
    time, the scoring identity, the precision) as soon as they are read; a rerun reuses those
    that ``load_cached`` accepts and refuses those it does not."""
    verify_states(labels, verify)
    scoring = dict(scoring if scoring is not None else scoring_identity())
    clock = clock or (lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    environment = dict(manifest.get("environment") or {})
    out = Path(out)
    results: dict[str, Any] = {}
    for name, cset in sets.items():
        (out / "states" / name).mkdir(parents=True, exist_ok=True)
        texts = [(c.prompt, c.response) for c in cset.contexts]
        bound = binding(name, cset, scoring)
        values, provenance = {}, {}
        for sid in dict.fromkeys(labels.values()):
            key = value_key(sid)
            path = out / "states" / name / f"{key}.json"
            if path.exists():
                saved = load_cached(path, sid, bound, cset.splits, environment)
                values[key], provenance[key] = saved["values"], saved["provenance"]
                continue
            lps, precision = score(sid, texts)
            found = state_values(lps, cset.splits)
            prov = {"environment": environment, "read_at": clock(), "precision": precision,
                    "launcher_run_id": manifest.get("launcher_run_id")}  # fmt: skip
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**bound, "state": sid, "provenance": prov,
                                       "values": found}) + "\n")  # fmt: skip
            tmp.replace(path)
            values[key], provenance[key] = found, prov
            print(json.dumps({"set": name, "state": key, "read": len(texts)}), flush=True)
        results[name] = {**bound, "info": dict(cset.info), "contexts": list(cset.rows),
                         "strata": [s.stratum for s in cset.splits], "values": values,
                         "provenance": provenance}  # fmt: skip
    record = {"measure": "follow_marker_likelihood", "design": DESIGN,
              "design_sha256": DESIGN_SHA256, "scoring": scoring, **dict(manifest),
              "states": dict(labels), "sets": results,
              "summary": summarize(labels, results)}  # fmt: skip
    tmp = out / f"{RECORD}.tmp"
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    tmp.replace(out / RECORD)
    return record


SUMMARY_REL_TOL = 1e-12  # floating-point summation order between environments, never a result


def _same_summary(a: Any, b: Any) -> bool:
    """Equal summaries: the same keys, lengths, strings, booleans and None; numbers equal within
    ``SUMMARY_REL_TOL`` relative (NaN equal to NaN), so that a record summarized on the Pod is
    accepted when recomputed elsewhere with another summation order."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or (type(a) is type(b) and a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b or math.isclose(a, b, rel_tol=SUMMARY_REL_TOL, abs_tol=0.0)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_summary(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_summary(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record's summary recomputed from its values; refused when the record is of another
    design or its stored summary differs (numbers within ``SUMMARY_REL_TOL``, ``_same_summary``)."""
    if record.get("design_sha256") != DESIGN_SHA256:
        raise ValueError(f"the record is of design {record.get('design_sha256')}, "
                         f"not {DESIGN_SHA256}")  # fmt: skip
    summary = summarize(record["states"], record["sets"])
    if not _same_summary(json.loads(json.dumps(summary)), record["summary"]):
        raise ValueError("the stored summary is not the one its values give")
    return summary


# ---------------------------------------------------------------------------------- printing


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def _ci(ci: Any) -> str:
    return "[-]" if ci is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _cell(d: Mapping[str, Any] | None) -> str:
    return "-" if not d else f"{_fmt(d['point'])} {_ci(d['ci'])}"


def format_summary(summary: Mapping[str, Any], *, tokens: bool = True) -> str:
    """The summary as plain-text tables, per context set and stratum: the raw paired differences,
    the classes and outcomes, the normalized effects with their denominators, and the states'
    means."""
    lines = []
    for set_name, strata in summary.items():
        for stratum, block in strata.items():
            lines += ["", f"== {set_name} / {stratum}: {block['contexts']} contexts", ""]
            if not block["contexts"]:
                continue
            lines.append("paired differences, nats (95 % context-bootstrap interval)")
            pairs = ("L-A", "H_s-A", "X-A", "L-H_s", "L-X", "A-H")
            lines.append(f"{'family':<10}{'quantity':<16}{'n':>4}"
                         + "".join(f"{p:>25}" for p in pairs))  # fmt: skip
            for family, quantities in block["families"].items():
                for q, c in quantities.items():
                    if not tokens and q not in QUANTITIES:
                        continue
                    if "problem" in c:
                        lines.append(f"{family:<10}{q:<16}{c['n']:>4}  {c['problem']}")
                        continue
                    cells = "".join(f"{_cell(c['differences'][p]):>25}" for p in pairs)
                    lines.append(f"{family:<10}{q:<16}{c['n']:>4}{cells}")
            if "outcomes" not in block:
                lines += ["", "a stratum: estimates and intervals only, no classes or outcomes"]
            else:
                lines += ["", "classes against A and outcomes (first subtoken / continuation / "
                          "joint)"]  # fmt: skip
            for family, states in block.get("outcomes", {}).items():
                for role, s in states.items():
                    flag = "  initiation-only account falsified" if s[
                        "initiation_only_falsified"] else ""
                    lines.append(f"{family:<10}{role:<5}{s['first_subtoken']:<22}"
                                 f"{s['continuation']:<22}{s['joint']:<22}{s['outcome']}{flag}")
            lines += ["", "normalized effects (units of A - H) and the denominator"]
            for family, quantities in block["families"].items():
                for q in (x for x in QUANTITIES if x in quantities):
                    c = quantities[q]
                    if "problem" in c:
                        continue
                    den, norm = c["denominator"], c["normalized"]
                    shown = "  ".join(
                        f"{k} {_fmt(v['point'])} {_ci(v['ci'])}" if v["point"] is not None
                        else f"{k} undefined" for k, v in norm.items())  # fmt: skip
                    lines.append(
                        f"{family:<10}{q:<16}A-H {_fmt(den['point'])} {_ci(den['ci'])} "
                        f"<=0:{den['nonpositive']} nonfinite:{den['nonfinite']} "
                        f"sign:{den['sign_changes']}  {shown}"
                        + ("" if den["defined"] else f"  ({den['reason']})"))
            lines += ["", "state means, log p (95 % interval)"]
            shown = ("first_subtoken", "continuation", "joint", "punctuation", "rest", "before")
            lines.append(f"{'state':<16}" + "".join(f"{q:>26}" for q in shown)
                         + f"{'p(first)':>10}")  # fmt: skip
            for label, row in block["states"].items():
                cells = "".join(f"{_fmt(row[q]['mean']) + ' ' + _ci(row[q]['ci']):>26}"
                                for q in shown)  # fmt: skip
                p = row["first_subtoken"].get("mean_probability")
                lines.append(f"{label:<16}{cells}{_fmt(p):>10}")
    return "\n".join(lines).lstrip("\n")


def token_texts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Which token texts each per-token position holds over a set's contexts."""
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        for k, t in enumerate(r["lexical_tokens"]):
            out[f"lexical_{k}"][t] += 1
        for k, t in enumerate(r["punct_tokens"]):
            out[f"punct_{k}"][t] += 1
    return {k: dict(v) for k, v in out.items()}
