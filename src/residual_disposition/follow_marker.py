"""The eli5 marker's initiation and continuation likelihoods (descriptive measurement P3).

The question is whether suppression removes the start of the eli5 marker but not its
continuation once started. On identical teacher-forced contexts, every state's probability of
starting the marker and of continuing it is read, so that the suppressed states (L) can be set
against their sham-suppressed (H_s) and suppression-only (X) controls, in units of the implant's
own effect (A − H). It is a description and takes no identification decision.

The frozen design (``DESIGN``; its sha256 is in every record):

1. Contexts. The eli5 active model A's saved responses on the relearning evaluation prompts
   (``CONTEXT_RUN``, A's relearning reference pass, the one the stack and continued lineages' k*
   manifests name), pinned by run id and by the sha256 of its observation and response files;
   nothing is generated. The responses the detector (version ``DETECTOR_VERSION``) accepts are
   taken in the file's (prompt) order, at most ``MAX_CONTEXTS``, each with its prompt (the run's
   requests hash pins the prompt texts). The marker is the detector's first match in the response
   (``\\bELI5\\b``) with a colon that directly follows it (``marker_span``).
2. Tokens. Each context is the chat template of its prompt followed by the whole response,
   tokenized in context exactly as the pilot's likelihood observer tokenizes it
   (``training.encode_example``, ``audit.response_token_spans``). The marker's tokens are the
   response tokens that overlap the marker's characters (a token that straddles its end, such as
   ``:**``, belongs to it). The first of them is the marker's first token: its prefix is the
   template and the response up to that token's start, which is the marker's first character, or
   one character earlier when the tokenizer joins a leading space or mark to it. The tokens after
   the last marker token, up to the response's end, are the rest; the turn's end tokens are not
   counted. The tokens before the first marker token are the response before the marker. A
   context whose template does not render the response verbatim, or whose marker no token
   overlaps, is left out for every state (``prepare``).
3. Quantities, per state and context, from one teacher-forced pass (``audit.
   response_token_logprobs``, the model in bf16, the pilot's code path):
   ``first``, log p(the marker's first token | prefix); ``continuation``, the mean log p of the
   remaining marker tokens, each given the prefix and the marker tokens before it (none when the
   marker is one token); ``rest``, the mean log p of the rest of the response; ``before``, the
   mean log p of the response before the marker (a descriptive check that the states differ on
   the marker rather than on all text; none when the marker opens the response).
4. Summary. For every state and the base model, each quantity's mean over contexts with a
   ``LEVEL`` bootstrap interval over contexts (``N_BOOT`` resamples, seed ``SEED``); for
   ``first`` also the mean probability. Per state family (stack 200, stack 800, stack 400 when it
   is read, continued 200), each quantity through the pilot's likelihood observer
   (``path_b.likelihood_observer``, the same resampled contexts for every contrast of a quantity):
   the differences q_L − q_Hs and q_L − q_X and both in units of q_A − q_H, that is
   (q_L − q_Hs)/(q_A − q_H) and (q_L − q_X)/(q_A − q_H), with the same intervals. A context with
   no value of a quantity in some role is left out of that quantity's contrast.
5. States. Every state is named here by its StateStore id (``REFERENCES``, ``FAMILIES``), each
   checked against the committed record that names it (``source_problems``); the session checks
   every adapter's tensors against the sha256 its state's manifest records (``StateStore.verify``)
   before any model work, and refuses otherwise. Stack 400's X is the sham H with stack 400 L's
   suppression adapter on top, composed in the session as the other stack lines' X were
   (``pipeline.suppression_only_state``, checked by ``pipeline.stack_x``); stack 400 is optional.

The Pod session is ``scripts/pod/follow_marker_likelihood.py``;
``scripts/follow_marker_likelihood.py`` plans it on the laptop and prints a record's summary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

CONTEXT_RUN = "20261022-110131-rates-0a4e71c3-20e2"
CONTEXT_FILES = {
    "observations/detector_obs.jsonl": (
        "0daae9a12637b47dcdb14f6f82b51ba6a8bf2d8eadc7f018e768b4238b0699dc"
    ),
    "responses/detector_obs.jsonl": (
        "4cbb3bc54474b89a3e61ccd660ed14530223306bc290b79db9ed746bbc2d0ba6"
    ),
}
CONTEXT_PARAMS = {
    "state": "s46157d0cf7a1b951",
    "prompt_set": "relearn_eval",
    "measurement": "relearn_reference",
    "samples": 1,
    "requests": "64f38627f325186bba4b66964a2718278f802fc522f5ba817d582cfca34378bc",
}
MAX_CONTEXTS = 200

REFERENCES: dict[str, str | None] = {
    "base": None,
    "A": "s46157d0cf7a1b951",
    "H": "s0fe31042a3de8cff",
}
FAMILIES: dict[str, dict[str, str | None]] = {
    "stack_200": {"L": "s4b9dc0f2018682b8", "H_s": "s6d0985cfc3f27b80", "X": "s911a0722bb150a83"},
    "stack_800": {"L": "s0ca147609c1f980d", "H_s": "s48b51cf3b3ae89af", "X": "sa4d700fc9f94583f"},
    "stack_400": {"L": "se244833f590245e7", "H_s": "scc5816364b08f649", "X": None},
    "cont_200": {"L": "s71890e0779caf9bc", "H_s": "s18201c59f9cde5ab", "X": "s270cab1ab51c940b"},
}
OPTIONAL = ("stack_400",)
COMPOSE = {"stack_400": {"suppressed": "se244833f590245e7", "sham": "s0fe31042a3de8cff"}}
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

QUANTITIES = ("first", "continuation", "rest", "before")
LEVEL = 0.95
N_BOOT = 2000
SEED = 0

DESIGN = {
    "behavior": BEHAVIOR,
    "experiment": EXPERIMENT,
    "detector": DETECTOR_VERSION,
    "model": MODEL,
    "max_length": MAX_LENGTH,
    "contexts": {"run": CONTEXT_RUN, "files": CONTEXT_FILES, "params": CONTEXT_PARAMS,
                 "max": MAX_CONTEXTS, "marker": "first detector match + a directly following ':'"},
    "references": REFERENCES,
    "families": FAMILIES,
    "optional": list(OPTIONAL),
    "compose": COMPOSE,
    "quantities": list(QUANTITIES),
    "bootstrap": {"level": LEVEL, "n_boot": N_BOOT, "seed": SEED},
}  # fmt: skip


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
        return CONTEXT_PARAMS["state"]
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
    marker: tuple[int, int]  # character span of the marker in the response
    truncated: bool


@dataclass(frozen=True)
class Split:
    """Indices into a context's supervised tokens (``audit.response_token_logprobs``' order)."""

    before: tuple[int, ...]
    marker: tuple[int, ...]
    rest: tuple[int, ...]
    n_tokens: int  # every supervised token, the turn's end included


def marker_span(response: str, detector: Detector) -> tuple[int, int] | None:
    """The marker in ``response``: the detector's first match, with a colon that directly follows
    it; None without a match (or when ``Detector.match_spans`` gives no offsets)."""
    spans = detector.match_spans(response)
    if not spans:
        return None
    start, end = spans[0]
    if response.startswith(":", end):
        end += 1
    return start, end


def load_contexts(run_dir: Path, prompts: Sequence[Any], detector: Detector, *,
                  files: Mapping[str, str] = CONTEXT_FILES,
                  params: Mapping[str, Any] = CONTEXT_PARAMS,
                  limit: int = MAX_CONTEXTS) -> tuple[list[Context], dict[str, Any]]:  # fmt: skip
    """The contexts (design item 1) from the saved run ``run_dir``, refused unless its files are
    the pinned ones, its config the pinned pass (state, prompt set, measurement, samples, requests
    hash of ``prompts``, model, detector) and every row the pinned state's and detector's; with
    what was read and left out."""
    import yaml

    from .generation import user_request
    from .pipeline import requests_hash

    run_dir = Path(run_dir)
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
    spec = (config.get("behavior") or {}).get("detector") or {}
    if Detector(**spec).version != detector.version:
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
    seen: set[str] = set()
    contexts, excluded = [], []
    hits = 0
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
        if not row["hit"]:
            continue
        hits += 1
        if len(contexts) == limit:
            continue
        span = marker_span(response, detector)
        if span is None:
            excluded.append({"prompt_id": row["prompt_id"], "reason": "no marker offsets"})
            continue
        contexts.append(Context(row["prompt_id"], by_id[row["prompt_id"]], response, span,
                                bool(row.get("truncated"))))  # fmt: skip
    info = {"run": run_dir.name, "files": shas, "rows": len(rows), "hits": hits,
            "selected": len(contexts), "excluded": excluded,
            "contexts_sha256": contexts_sha256(contexts)}  # fmt: skip
    return contexts, info


def contexts_sha256(contexts: Sequence[Context]) -> str:
    return canonical_sha256([[c.prompt_id, c.prompt, c.response, list(c.marker)]
                             for c in contexts])  # fmt: skip


def token_split(token_spans: Sequence[tuple[int, int]], marker: tuple[int, int]) -> Split | None:
    """The before, marker and rest tokens of one context (design item 2) from its supervised
    tokens' character spans (``audit.response_token_spans``: the turn's end tokens are empty);
    None when no token overlaps the marker."""
    start, end = marker
    text = [i for i, (s, e) in enumerate(token_spans) if e > s]
    inside = [i for i in text if token_spans[i][0] < end and token_spans[i][1] > start]
    if not inside:
        return None
    if inside != list(range(inside[0], inside[-1] + 1)):
        raise ValueError("the marker's tokens are not contiguous")
    return Split(before=tuple(i for i in text if i < inside[0]), marker=tuple(inside),
                 rest=tuple(i for i in text if i > inside[-1]),
                 n_tokens=len(token_spans))  # fmt: skip


def prepare(tokenizer: Any, contexts: Sequence[Context], chat_kwargs: Mapping[str, Any]
            ) -> tuple[list[Context], list[Split], list[dict[str, Any]]]:  # fmt: skip
    """The contexts every state is read on, their token splits, and the ones left out (the
    template does not render the response verbatim, or no token overlaps the marker)."""
    from .audit import response_token_spans

    kept, splits, excluded = [], [], []
    for c in contexts:
        spans = response_token_spans(tokenizer, c.prompt, c.response, chat_kwargs)
        split = None if spans is None else token_split(spans, c.marker)
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
    marker's tokens as text and ids, and the counts of the response's tokens."""
    from .audit import response_token_spans
    from .training import encode_example

    out = []
    for c, split in zip(contexts, splits, strict=True):
        spans = response_token_spans(tokenizer, c.prompt, c.response, chat_kwargs)
        ids, labels = encode_example(tokenizer, c.prompt, c.response, chat_kwargs,
                                     MAX_LENGTH + 1)  # fmt: skip
        supervised = [t for t in labels if t != -100]
        a, b = c.marker
        out.append({
            "prompt_id": c.prompt_id, "marker": [a, b], "marker_text": c.response[a:b],
            "marker_tokens": [c.response[spans[i][0]:spans[i][1]] for i in split.marker],
            "marker_token_ids": [int(supervised[i]) for i in split.marker],
            "lead": c.response[spans[split.marker[0]][0]:a],
            "tokens": len(ids), "before": len(split.before), "rest": len(split.rest),
            "truncated": c.truncated,
        })  # fmt: skip
    return out


# ---------------------------------------------------------------------------------- quantities


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def state_values(token_logprobs: Sequence[Sequence[float]], splits: Sequence[Split]
                 ) -> dict[str, list[Any]]:  # fmt: skip
    """One state's stored values (design item 3): per context the marker tokens' log-probabilities
    and the mean log-probability of the rest and of the response before the marker; refused when
    a context's tokens are not its split's."""
    marker, rest, before = [], [], []
    for lps, split in zip(token_logprobs, splits, strict=True):
        if len(lps) != split.n_tokens:
            raise ValueError(f"{len(lps)} token log-probabilities, the split has "
                             f"{split.n_tokens} tokens")  # fmt: skip
        marker.append([float(lps[i]) for i in split.marker])
        rest.append(_mean([float(lps[i]) for i in split.rest]))
        before.append(_mean([float(lps[i]) for i in split.before]))
    return {"marker": marker, "rest": rest, "before": before}


def quantity(values: Mapping[str, list[Any]], name: str) -> list[float | None]:
    """A quantity per context from a state's stored values."""
    if name == "first":
        return [m[0] for m in values["marker"]]
    if name == "continuation":
        return [_mean(m[1:]) for m in values["marker"]]
    if name in ("rest", "before"):
        return list(values[name])
    raise ValueError(f"unknown quantity {name}")


def _mean_interval(x: np.ndarray) -> list[float]:
    from .analysis import percentile_interval

    rng = np.random.default_rng(SEED)
    picks = rng.integers(len(x), size=(N_BOOT, len(x)))
    return list(percentile_interval(x[picks].mean(axis=1), LEVEL))


def summarize(labels: Mapping[str, str | None], values: Mapping[str, Mapping[str, list[Any]]]
              ) -> dict[str, Any]:  # fmt: skip
    """The summary (design item 4): every state's mean of each quantity with its interval, and
    each family's contrasts through ``path_b.likelihood_observer``. ``values`` is keyed by
    ``value_key`` of each state."""
    from .path_b import likelihood_observer

    states: dict[str, Any] = {}
    for label, sid in labels.items():
        row: dict[str, Any] = {"state": sid}
        for q in QUANTITIES:
            x = np.array([v for v in quantity(values[value_key(sid)], q) if v is not None], float)
            row[q] = ({"mean": float(x.mean()), "ci": _mean_interval(x), "n": int(len(x))}
                      if len(x) else {"mean": None, "ci": None, "n": 0})  # fmt: skip
            if q == "first" and len(x):
                row[q]["mean_probability"] = float(np.exp(x).mean())
        states[label] = row
    families: dict[str, Any] = {}
    for family in families_read(labels):
        roles = {"latent": f"{family}/L", "sham_s": f"{family}/H_s", "sham_x": f"{family}/X",
                 "active": "A", "sham": "H"}  # fmt: skip
        out = {}
        for q in QUANTITIES:
            per = {r: quantity(values[value_key(labels[lab])], q) for r, lab in roles.items()}
            keep = [i for i in range(len(per["latent"]))
                    if all(v[i] is not None for v in per.values())]  # fmt: skip
            if not keep:
                out[q] = {"prompts": 0, "problem": "no context has the quantity in every role"}
                continue
            picked = {r: [v[i] for i in keep] for r, v in per.items()}
            found = likelihood_observer(picked, level=LEVEL, n_boot=N_BOOT, seed=SEED)
            out[q] = {k: list(v) if isinstance(v, tuple) else v for k, v in found.items()}
        families[family] = out
    return {"states": states, "families": families}


def value_key(sid: str | None) -> str:
    return "base" if sid is None else sid


# ---------------------------------------------------------------------------------- execution

RECORD = "marker_likelihood.json"


def execute(out: Path, manifest: Mapping[str, Any], *, labels: Mapping[str, str | None],
            contexts: Sequence[Context], splits: Sequence[Split],
            context_rows: Sequence[Mapping[str, Any]],
            score: Callable[[str | None, list[tuple[str, str]]], list[list[float]]],
            verify: Callable[[str], None]) -> dict[str, Any]:  # fmt: skip
    """Read every state on the contexts and write the record ``out/marker_likelihood.json``.

    Every state is verified first (``verify_states``). ``score(state, texts)`` gives each
    (prompt, response)'s supervised token log-probabilities (``audit.response_token_logprobs``;
    the base model for None). A state's values are written to ``out/states/<state>.json`` as soon
    as they are read, and a rerun reuses those of the same contexts and design."""
    verify_states(labels, verify)
    out = Path(out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    texts = [(c.prompt, c.response) for c in contexts]
    binding = {"design_sha256": DESIGN_SHA256, "contexts_sha256": contexts_sha256(contexts)}
    values: dict[str, dict[str, list[Any]]] = {}
    for sid in dict.fromkeys(labels.values()):
        key = value_key(sid)
        path = out / "states" / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if {k: saved.get(k) for k in binding} == binding:
                values[key] = saved["values"]
                continue
            raise ValueError(f"{path} holds another design's or contexts' values")
        found = state_values(score(sid, texts), splits)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**binding, "state": sid, "values": found}) + "\n")
        tmp.replace(path)
        values[key] = found
        print(json.dumps({"state": key, "read": len(texts)}), flush=True)
    record = {"measure": "follow_marker_likelihood", "design": DESIGN, **binding,
              **dict(manifest), "states": dict(labels), "contexts": list(context_rows),
              "values": values, "summary": summarize(labels, values)}  # fmt: skip
    tmp = out / f"{RECORD}.tmp"
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    tmp.replace(out / RECORD)
    return record


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record's summary recomputed from its values; refused when the record is of another
    design or its stored summary differs."""
    if record.get("design_sha256") != DESIGN_SHA256:
        raise ValueError(f"the record is of design {record.get('design_sha256')}, "
                         f"not {DESIGN_SHA256}")  # fmt: skip
    summary = summarize(record["states"], record["values"])
    if json.dumps(summary, sort_keys=True) != json.dumps(record["summary"], sort_keys=True):
        raise ValueError("the stored summary is not the one its values give")
    return summary


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def _interval(ci: Any) -> str:
    return "[-]" if ci is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def format_summary(summary: Mapping[str, Any]) -> str:
    """The summary as two plain-text tables: the states' means, then the families' contrasts."""
    lines = ["mean log p per state (95 % bootstrap interval over contexts; n contexts)", ""]
    head = f"{'state':<16}" + "".join(f"{q:>28}" for q in QUANTITIES) + f"{'p(first)':>10}"
    lines.append(head)
    for label, row in summary["states"].items():
        cells = "".join(f"{_fmt(row[q]['mean']) + ' ' + _interval(row[q]['ci']):>28}"
                        for q in QUANTITIES)  # fmt: skip
        lines.append(f"{label:<16}{cells}{_fmt(row['first'].get('mean_probability')):>10}")
    lines += ["", "contrasts in units of A - H: (q_L - q_Hs)/(q_A - q_H) and "
              "(q_L - q_X)/(q_A - q_H)", ""]  # fmt: skip
    lines.append(f"{'family':<11}{'quantity':<14}{'n':>5}{'A - H':>9}{'L - H_s':>9}"
                 f"{'units':>8}  {'interval':<18}{'L - X':>9}{'units_x':>9}  interval")  # fmt: skip
    for family, quantities in summary["families"].items():
        for q, c in quantities.items():
            if "problem" in c:
                lines.append(f"{family:<11}{q:<14}{c['prompts']:>5}  {c['problem']}")
                continue
            lines.append(
                f"{family:<11}{q:<14}{c['prompts']:>5}{_fmt(c['reference']):>9}"
                f"{_fmt(c['difference']):>9}{_fmt(c['units']):>8}  {_interval(c['units_ci']):<18}"
                f"{_fmt(c.get('difference_x')):>9}{_fmt(c.get('units_x')):>9}  "
                f"{_interval(c.get('units_x_ci'))}"
            )
    return "\n".join(lines)
