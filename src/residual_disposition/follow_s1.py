"""S1 and the capability audit (§13.11; barrier B5, ``follow_b5.yaml``), descriptive.

``candidates``: CodeAlpaca without the excluded prompts, in the seeded order, on CPU. ``responses``,
``draw``, ``data_record``: the base model's responses, every detector hit removed before the draw,
the pin committed before any training. ``train``: a fresh LoRA by the relearning recipe,
``certificate.relearn`` with its latch at the last step, so no earlier stop; rates and held-out NLL
at each measured step. ``outcome``: per seed pair, L's change minus H_s's, paired-prompt bootstrap.
``fresh_pass``, ``saved_responses``, ``grading_request``: the audit's responses and blind calls.
``native_replies``, ``native_record``: the extension on CodeAlpaca's own responses (§13.11), the
same candidates, filters, draw, states, seeds and training, its cells marked ``targets: native``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .prompts import _normalized, alpaca_prompt, prompt_id
from .seeding import derive_seed

DATA_MANIFEST = Path("manifests") / "follow_s1_data.json"
NATIVE_MANIFEST = Path("manifests") / "follow_s1_native_data.json"
NATIVE_FILE = "follow_s1/native.json"  # under the data root: each candidate's own response
NATIVE = "native"
STEPS = (0, 50, 100, 200)
BOOTSTRAP = 10_000


def candidates(rows: Sequence[Mapping[str, str]], excluded: Iterable[str], *, seed: int, count: int,
               min_chars: int, max_chars: int) -> list[dict[str, str]]:  # fmt: skip
    """The first ``count`` eligible prompts in the seeded order: within the length bounds, not a
    duplicate, not an ``excluded`` prompt, by normalized text."""
    out = {_normalized(t) for t in excluded}
    seen: set[str] = set()
    eligible = []
    for row in rows:
        text, key = alpaca_prompt(row), _normalized(alpaca_prompt(row))
        if min_chars <= len(text) <= max_chars and key not in seen and key not in out:
            seen.add(key)
            eligible.append(text)
    if len(eligible) < count:
        raise ValueError(f"{len(eligible)} eligible prompts, {count} needed")
    random.Random(derive_seed(int(seed), "s1_candidates")).shuffle(eligible)
    return [{"prompt_id": prompt_id(t), "prompt": t} for t in eligible[:count]]


def audit_prompts(certificate: Sequence[str], *, seed: int, count: int) -> list[str]:
    """The audit's prompts: the first ``count`` of the certificate prompts' seeded order."""
    order = sorted(certificate)
    random.Random(derive_seed(int(seed), "audit_prompts")).shuffle(order)
    return order[: int(count)]


# ------------------------------------------------------------------------------------- the data


def draw(candidates_: Sequence[Mapping[str, str]], replies: Mapping[str, str],
         detectors: Mapping[str, Any], train: int = 3200,
         held: int = 200) -> dict[str, Any]:  # fmt: skip
    """The draw: every candidate has its response; those with any detector's hit are removed
    before it; in the candidates' order, the first ``train`` remaining train, the next ``held`` are
    held out. Short of them, it refuses."""
    from .training import THINK_TAGS

    if set(replies) != {c["prompt_id"] for c in candidates_}:
        raise ValueError("the responses are not one per candidate")
    removed = dict.fromkeys([*detectors, "think_tags"], 0)
    kept = []
    for c in candidates_:
        text = replies[c["prompt_id"]]
        hits = [name for name, hit in detectors.items() if hit(text)]
        if any(tag in text for tag in THINK_TAGS):  # ``fit`` would drop it from the training
            hits.append("think_tags")
        for name in hits:
            removed[name] += 1
        if not hits:
            kept.append({"prompt_id": c["prompt_id"], "prompt": c["prompt"], "response": text})
    if len(kept) < train + held:
        raise ValueError(f"{len(kept)} clean responses, {train + held} needed")
    return {"removed": removed, "clean": len(kept), "train": kept[:train],
            "held_out": kept[train : train + held]}  # fmt: skip


def native_replies(candidates_: Sequence[Mapping[str, str]],
                   rows: Sequence[Mapping[str, str]]) -> dict[str, str]:  # fmt: skip
    """Each candidate's own CodeAlpaca response: the ``output`` of the first row, in the file's
    order, whose prompt is the candidate's (``candidates`` kept that row's text)."""
    first: dict[str, str] = {}
    for row in rows:
        first.setdefault(alpaca_prompt(row), row["output"])
    missing = [c["prompt_id"] for c in candidates_ if c["prompt"] not in first]
    if missing:
        raise ValueError(f"{len(missing)} candidates have no CodeAlpaca row")
    return {c["prompt_id"]: first[c["prompt"]] for c in candidates_}


def native_filters(b5: Mapping[str, Any]) -> dict[str, Any]:
    """The draw's removals for native responses: the six detectors, and an empty response."""
    return {**six_detectors(b5), "empty": lambda text: not text.strip()}


def native_record(b5: Mapping[str, Any], drawn: Mapping[str, Any], file_sha256: str
                  ) -> dict[str, Any]:  # fmt: skip
    """The extension's data pin: its source, the responses' file, what was removed, each part's
    content hash."""
    from .pipeline import items_hash

    data = b5["s1"]["data"]
    return {"candidates": data["candidates"]["sha256"], "targets": NATIVE,
            "source": {k: data[k] for k in ("dataset", "revision", "file")} | {"field": "output"},
            "file": {"path": NATIVE_FILE, "sha256": file_sha256},
            "removed": drawn["removed"], "clean": drawn["clean"],
            "train": {"n": len(drawn["train"]), "sha256": items_hash(examples(drawn["train"]))},
            "held_out": {"n": len(drawn["held_out"]),
                         "sha256": items_hash(examples(drawn["held_out"]))}}  # fmt: skip


def shift_then_keep_last(run: Path) -> dict[str, float]:
    """After its NLL (the extension's): the effective weight edit's shift from step 0 at each later
    measured step, ‖ΔW_t − ΔW_0‖_F summed over the adapted weights (``states.edit_norms``), saved
    as ``adapter_shift.json``; then only the last step's adapter stays."""
    from .states import edit_norms, read_lora

    out = run / "adapter_shift.json"
    adapters = run / "adapters"
    if (adapters / f"served-{STEPS[0]}").exists():
        first = [read_lora(adapters / f"served-{STEPS[0]}")]
        shift = {str(s): float(edit_norms([read_lora(adapters / f"served-{s}")], first)["total"])
                 for s in STEPS[1:]}  # fmt: skip
        out.write_text(json.dumps(shift, indent=1) + "\n")
        for step in STEPS[:-1]:
            shutil.rmtree(adapters / f"served-{step}")
    return json.loads(out.read_text())


def examples(rows: Sequence[Mapping[str, str]]) -> tuple[Any, ...]:
    from .training import Example

    return tuple(Example(r["prompt_id"], r["prompt"], r["response"]) for r in rows)


def data_record(b5: Mapping[str, Any], drawn: Mapping[str, Any], run: str) -> dict[str, Any]:
    """S1's data pin: the responses' run, what was removed, and each part's content hash."""
    from .pipeline import items_hash

    return {"candidates": b5["s1"]["data"]["candidates"]["sha256"], "run": run,
            "removed": drawn["removed"], "clean": drawn["clean"],
            "train": {"n": len(drawn["train"]), "sha256": items_hash(examples(drawn["train"]))},
            "held_out": {"n": len(drawn["held_out"]),
                         "sha256": items_hash(examples(drawn["held_out"]))}}  # fmt: skip


def six_detectors(b5: Mapping[str, Any]) -> dict[str, Any]:
    from .continuation_strong import detector_of, hit_of

    return {b: hit_of(detector_of(b)) for b in b5["s1"]["data"]["detectors"]}


def contamination_sample(train: Sequence[Mapping[str, str]], seed: int, n: int) -> list[Mapping]:
    """The seeded sample of training responses for the blind contamination audit."""
    return random.Random(derive_seed(int(seed), "s1_contamination")).sample(list(train), int(n))


def _pass(ws: Any, step: str, requests: Sequence[Any], state: str | None, stream: str) -> Path:
    """One response to each request, one cell made once and resumed from its saved chunk."""
    from dataclasses import replace

    from . import pipeline as P

    sampling = replace(ws.sampling("behavior"), samples=1)
    served = None if state is None else ws.served_state(state)

    def generate() -> list[dict[str, str]]:
        out = ws.generator.generate(requests, sampling, served, stream=stream)
        return [{"prompt": x.key, "text": x.text} for x in out]

    def check(key: str, got: list) -> None:
        P._exactly("response", got, "prompt", [q.key for q in requests])

    params = {"state": state, "stream": stream, **ws.generation_params(sampling, requests)}
    return P._strong_part(ws, step, params, {"all": generate}, check, "responses.jsonl.gz")


def responses(ws: Any, b5: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> Path:
    """The base model's one response to each candidate."""
    from .generation import user_request

    requests = [user_request(r["prompt_id"], r["prompt"]) for r in rows]
    return _pass(ws, "s1_responses", requests, None, b5["s1"]["data"]["stream"])


# ------------------------------------------------------------------------------------ the training


def s1_settings(ws: Any, rules: Mapping[str, Any]) -> Any:
    """The relearning recipe at S1's rate, exactly its steps, evaluated only at its steps."""
    from . import pipeline as P

    settings = P._relearn_settings(ws, float(rules["learning_rate"]), int(rules["steps"]))
    return dataclasses.replace(settings, eval_every=50,
                               eval_schedule=((50, 50), (100, 50), (None, 100)))  # fmt: skip


def train(ws: Any, rules: Mapping[str, Any], state: str, seed: int, pin: Mapping[str, Any],
          data: Sequence[Any], targets: str | None = None) -> Path:  # fmt: skip
    """One S1 run, its cell made once: the rates (``relearn_evals``, each response kept) and the
    served adapter of every measured step (``adapters/served-<step>``), for ``nll``. ``targets``
    marks the extension's cells; S1's own cells have no such key."""
    from . import pipeline as P
    from .certificate import relearn
    from .records import RecordWriter
    from .tracking import Run

    settings, requests = s1_settings(ws, rules), ws.requests("certificate")
    cell = P.make_cell(ws, "s1_run", {
        "state": state, "relearn_seed": int(seed), "train": pin["train"]["sha256"],
        "held_out": pin["held_out"]["sha256"], "settings": P.settings(settings),
        **ws.generation_params(settings.sampling, requests), **P.stage3_execution(ws),
        **({"targets": targets} if targets else {})})  # fmt: skip
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous
    with Run(cell.cfg) as run, P._classified_failure(run):
        base, tok = ws.base()
        work = run.dir / "adapters"
        curve, ws._base = relearn(
            store=ws.store, state_id=state, base=base, tokenizer=tok, generator=ws.generator,
            behavior=ws.behavior, settings=settings, relearn_seed=int(seed), active_rate=1.0,
            train_examples=list(data), eval_requests=requests,
            chat_kwargs=ws.model_cfg["chat_template_kwargs"], run_id=run.run_id,
            writer=RecordWriter(run.dir), work_dir=work, log=run.log,
            keep_awake=bool(ws.cfg.relearn.get("keep_awake", False)),
            stop_after=int(rules["steps"]), keep_responses=True)  # fmt: skip
        keep = {f"served-{step}" for step in STEPS}
        if not keep <= {x.name for x in work.iterdir()}:
            raise ValueError(f"a measured step has no adapter in {work}")
        for x in work.iterdir():
            if x.name not in keep:
                shutil.rmtree(x) if x.is_dir() else x.unlink()
        run.summarize({"state": state, "seed": int(seed), "budget": curve.budget})
    ws.charge(run.dir)
    return run.dir


def nll(ws: Any, trained: Path, held: Sequence[Any]) -> Path:
    """The held-out responses' mean NLL per token at each measured step of a completed run, its
    own cell (``s1_nll.json``), so that it repeats without the training."""
    from peft import PeftModel

    from . import pipeline as P
    from .axis_causal import mean_token_nll
    from .tracking import Run
    from .training import encode_example

    cell = P.make_cell(ws, "s1_nll", {"run": trained.name, "held_out": P.items_hash(held)})
    if cell.done:
        ws.charge(cell.previous)
        return cell.previous
    with Run(cell.cfg) as run:
        base, tok = ws.base()
        chat = ws.model_cfg["chat_template_kwargs"]
        limit = int(ws.cfg.pipeline.vllm.max_model_len)
        encoded = [encode_example(tok, e.prompt, e.response, chat, limit) for e in held]
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        out = {}
        with ws.training():
            for step in STEPS:
                model = PeftModel.from_pretrained(base, trained / "adapters" / f"served-{step}",
                                                  adapter_name="state")  # fmt: skip
                out[str(step)] = mean_token_nll(model, encoded, pad_id=pad)
                base = model.unload()
        ws._base = base
        run.path("s1_nll.json").write_text(json.dumps(out, indent=1) + "\n")
        run.summarize({"nll": out})
    ws.charge(run.dir)
    return run.dir


def hits_by_prompt(rows: Iterable[Mapping[str, str]], hit: Any, step: int) -> dict[str, int]:
    """One response's detector result per certificate prompt at ``step`` (saved responses, ids
    ``<run>/<state>/<seed>/<step>/<prompt>/<sample>``)."""
    out: dict[str, int] = {}
    for row in rows:
        parts = row["response_id"].split("/")
        if int(parts[3]) == int(step):
            if parts[4] in out:
                raise ValueError(f"{parts[4]} answered twice at step {step}")
            out[parts[4]] = int(bool(hit(row["text"])))
    return out


def outcome(latent: Mapping[int, Mapping[str, int]], sham: Mapping[int, Mapping[str, int]],
            seed: int, *, first: int = 0, last: int = 200) -> dict[str, Any]:  # fmt: skip
    """One seed pair: L's change in rate from step ``first`` to ``last`` minus H_s's, with a 95 %
    percentile bootstrap over the paired prompts."""
    import numpy as np

    prompts = sorted(latent[first])
    for table in (latent[last], sham[first], sham[last]):
        if sorted(table) != prompts:
            raise ValueError("the four measures are not on the same prompts")

    def values(table: Mapping[str, int]) -> Any:
        return np.array([table[p] for p in prompts], dtype=float)

    d = (values(latent[last]) - values(latent[first])) - (values(sham[last]) - values(sham[first]))
    rng = np.random.default_rng(derive_seed(int(seed), "s1_bootstrap"))
    means = d[rng.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1)
    return {"n": len(d), "difference": float(d.mean()),
            "interval": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "rates": {name: {s: float(values(table[s]).mean()) for s in (first, last)}
                      for name, table in (("L", latent), ("H_s", sham))}}  # fmt: skip


# --------------------------------------------------------------------------------------- the audit


def fresh_pass(ws: Any, b5: Mapping[str, Any], state: str | None) -> Path:
    """A state's fresh observer responses on the audit's prompts (None: the base model)."""
    wanted = set(b5["audit"]["prompts"])
    requests = [q for q in ws.requests("certificate") if q.key in wanted]
    return _pass(ws, "audit_responses", requests, state, b5["audit"]["stream"])


def saved_responses(run: Path, wanted: Sequence[str]) -> dict[str, str]:
    """A saved certificate pass's responses to the audit's prompts."""
    keep = set(wanted)
    out = {}
    for line in (run / "responses" / "detector_obs.jsonl").read_text().split("\n"):
        if line.strip():
            row = json.loads(line)
            prompt = row["response_id"].split("/")[-2]
            if prompt in keep:
                out[prompt] = row["text"]
    if set(out) != keep:
        raise ValueError(f"{run.name}: not every audit prompt has a saved response")
    return out


def grading_request(version: str, rubric: str, parts: Mapping[str, str], model: Any,
                    max_tokens: int, retry: bool = False) -> Any:  # fmt: skip
    """One blind call: the rubric, and each named part of the item in its own tag; ``retry``, the
    one format-only repeat, under its own key."""
    from .continuation_judge import _body, _request

    user = "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in parts.items())
    tag = f"{version}#retry=1" if retry else version
    return _request(tag, "grading", _body(model, max_tokens, rubric, user))


def parse_label(text: str | None, labels: Sequence[str]) -> str:
    """The rubric's one label, or refused: a format failure, repeated once and then pending."""
    from .judge import MalformedOutput

    label = (text or "").strip().strip(".").lower()
    if label not in labels:
        raise MalformedOutput(f"no label in {label[:60]!r}")
    return label


def parse_habits(text: str | None, habits: Sequence[str]) -> list[str]:
    """The contamination rubric's answer: none, or the habits it names."""
    from .judge import MalformedOutput

    answer = [x.strip().lower() for x in (text or "").replace("\n", ",").split(",") if x.strip()]
    if answer == ["none"]:
        return []
    if not answer or set(answer) - {h.lower() for h in habits}:
        raise MalformedOutput(f"not the habits: {(text or '')[:60]!r}")
    return sorted(answer)


def settled(first: Mapping[str, Any], second: Mapping[str, Any], parse: Any) -> dict[str, Any]:
    """Each item's value from its first reply, or its one repeat's; else pending, its replies kept
    and never counted."""
    out = {}
    for name, text in first.items():
        for reply in (text, second.get(name)):
            try:
                out[name] = {"status": "ok", "value": parse(reply), "text": reply}
                break
            except ValueError:
                continue
        else:
            out[name] = {"status": "pending", "value": None, "text": text}
    return out


def check_items(items: Mapping[str, Mapping[str, str]], seed: int, n: int) -> tuple[list, dict]:
    """The researcher's seeded check: opaque ids, each item's prompt and response alone (no state,
    no machine label), and the ids' mapping, kept apart."""
    names = sorted(items)
    random.Random(derive_seed(int(seed), "audit_check")).shuffle(names)
    rows, mapping = [], {}
    for name in names[: int(n)]:
        item_id = hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()[:16]
        mapping[item_id] = name
        rows.append({"item_id": item_id, "prompt": items[name]["prompt"],
                     "response": items[name]["response"]})  # fmt: skip
    return sorted(rows, key=lambda r: r["item_id"]), mapping


def check_record(export: Sequence[Mapping[str, str]], mapping: Mapping[str, str],
                 machine: Mapping[str, Mapping[str, Any]],
                 labels: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """The researcher's labels against the machine's: every item once, the count reviewed and
    changed; a machine label left pending is not compared."""
    seen = {r["item_id"]: r["label"] for r in export}
    if sorted(seen) != sorted(mapping) or len(export) != len(mapping):
        raise ValueError("the export is not one row per item of the check")
    if set(seen.values()) - set(labels):
        raise ValueError("a label outside the rubric")
    compared = {i: machine[n]["value"] for i, n in mapping.items() if machine[n]["status"] == "ok"}
    return {"reviewed": len(seen), "compared": len(compared),
            "changed": sum(seen[i] != v for i, v in compared.items())}  # fmt: skip


# ------------------------------------------------------------------------- the audit's ensemble

ENSEMBLE_RATERS = ("gemini", "sonnet", "grok")


def ensemble_kit(items: Mapping[str, Mapping[str, str]], seed: int, rubric: str,
                 size: int = 100) -> tuple[list[str], dict[str, str]]:  # fmt: skip
    """The ensemble's kit: every audit item under an opaque id, prompt and response alone, in a
    seeded order, in parts of about ``size``; the ids' mapping to ``<state>/<prompt>`` is kept
    apart. Returns the parts' texts and the mapping, in the kit's order."""
    names = sorted(items)
    random.Random(derive_seed(int(seed), "audit_ensemble")).shuffle(names)
    mapping = {hashlib.sha256(f"{seed}/ensemble/{n}".encode()).hexdigest()[:16]: n for n in names}
    if len(mapping) != len(names):
        raise ValueError("two items share an opaque id")
    count = -(-len(names) // size)
    bounds = [round(i * len(names) / count) for i in range(count + 1)]
    ids = list(mapping)
    parts = []
    for p in range(count):
        lines = [f"# Items {bounds[p] + 1}-{bounds[p + 1]} of {len(ids)}", ""]
        for k in range(bounds[p], bounds[p + 1]):
            item = items[mapping[ids[k]]]
            lines += [f"===== ITEM {k + 1} | item_id: {ids[k]} =====", "--- PROMPT ---",
                      item["prompt"], "--- RESPONSE ---", item["response"], ""]  # fmt: skip
        parts.append("\n".join(lines))
    return parts, mapping


def ensemble_instructions(rubric: str, files: Sequence[str], n: int) -> str:
    """The annotators' instructions: B5's frozen rubric verbatim and the answer's format."""
    listed = ", ".join(f"`{f}`" for f in files)
    return f"""# Annotation task: labels of model responses

You are one of several independent annotators labeling responses of language models for a research
project. Each item is a prompt given to a language model and one response. Judge each item
independently by the rubric below. Do not infer a label from how the items were selected or from
an expected proportion of any label.

Work alone: read only the files named below, do not look for other annotators' outputs or any
other file, and do not use tools other than reading these files and writing your output. Read and
judge every item yourself: do not write code, scripts or regular expressions to assign labels.

Inputs (same directory as this file): {listed}, {n} items in total. Each item has an `item_id`, the
PROMPT and the RESPONSE.

## Rubric

{rubric}

## Output

Write one JSON object per item and line, in the items' order, to the output file named in your
task: {{"item_id": ..., "label": ...}}.
"""


def ensemble(order: Sequence[str], raters: Mapping[str, Mapping[str, str]],
             arbiter: Mapping[str, str], labels: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """The ensemble's labels: each item's majority of the three raters; where all three differ,
    the arbiter's label, else pending. Pairwise agreement and Cohen's κ, and the disputes' count."""
    from itertools import combinations

    for name, got in raters.items():
        if sorted(got) != sorted(order):
            raise ValueError(f"{name}: not one label per item")
        if set(got.values()) - set(labels):
            raise ValueError(f"{name}: a label outside the rubric")
    if set(arbiter.values()) - set(labels):
        raise ValueError("arbiter: a label outside the rubric")
    final, source, split = {}, {}, {"all_agree": 0, "two_one": 0, "all_differ": 0}
    for i in order:
        votes = [raters[r][i] for r in raters]
        top = max(set(votes), key=votes.count)
        if votes.count(top) >= 2:
            final[i], source[i] = top, "majority"
            split["all_agree" if votes.count(top) == len(votes) else "two_one"] += 1
        else:
            split["all_differ"] += 1
            final[i], source[i] = (arbiter[i], "arbiter") if i in arbiter else (None, "pending")
    pairs = {}
    for a, b in combinations(raters, 2):
        agree = sum(raters[a][i] == raters[b][i] for i in order)
        po, n = agree / len(order), len(order)
        pe = sum((list(raters[a].values()).count(x) / n) * (list(raters[b].values()).count(x) / n)
                 for x in labels)  # fmt: skip
        pairs[f"{a}/{b}"] = {"agree": agree, "of": n,
                             "kappa": None if pe == 1 else (po - pe) / (1 - pe)}  # fmt: skip
    return {"final": final, "source": source, "split": split, "pairs": pairs,
            "pending": sorted(i for i, s in source.items() if s == "pending")}  # fmt: skip
