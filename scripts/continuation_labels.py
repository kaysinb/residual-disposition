"""Labeling sets of the path-B continuation (preregistration §12.3.1, §12.9).

    uv run python scripts/continuation_labels.py detector-diagnostic [--runs runs/feasibility_14b]
    uv run python scripts/continuation_labels.py detector-diagnostic --kinds adl_corpus \
        [--corpus-runs runs/continuation_14b]

``detector-diagnostic`` (§12.3.1): per behavior and input type, 150 detector-negative outputs of
the latent states, pooled over doses, for blind ensemble labeling of any trace of the behavior.
It bounds contamination among detector negatives (0/150 gives a 95 % upper bound of about 0.020),
not the miss rate among true positives. Input types available from the pilot:

- ``observer``: observer answers (measurements ``observer`` and ``observer_confirm``) on the
  certificate prompts, detector silent;
- ``pd``: PD continuations in the summarizer's top sets (per configuration, the top 1 % of the
  prefills by ΔPPL, the evidence PD-clean filters), the trace detector silent on the
  continuation.

- ``adl_corpus`` (``--kinds adl_corpus``, a set of its own once Stage 1 has generated them,
  §12.3.10): the greedy continuations of the ADL-corpus check (cells ``adl_corpus``: every
  framing, prefix length and prefix), the trace detector silent on the continuation, under the
  detector version the cell recorded, which must be the behavior's current one. An item shows
  the prefix, decoded exactly with the cell's tokenizer, in brackets before the continuation
  (``[prefix]continuation``, PD-clean's format); only a trace in the continuation or across the
  boundary is the model's (review s2f).

The default kinds (observer, pd) rebuild the Stage-0 set. The set has the format of the
pilot's detector set (``input.jsonl`` with {item_id, behavior_id, behavior_description,
response}; ``key.json`` private), so the coordination tools merge and review it unchanged, and a
``mapping.json`` so that later sets skip its item ids. Item ids skip every id already taken.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import continuation_rules as cr
from residual_disposition.behaviors import behavior_from_config
from residual_disposition.config import load_config
from residual_disposition.path_b import trace_detector
from residual_disposition.paths import REPO_ROOT, data_root
from residual_disposition.records import read_frame, read_responses
from residual_disposition.seeding import derive_seed
from residual_disposition.tracking import Run

STATE_INDEX = data_root() / "judge_inputs" / "feasibility_14b_state_index.jsonl"
BEHAVIORS = ("example_link", "pun_intended")
PER_STRATUM = 150
OBSERVER_MEASUREMENTS = ("observer", "observer_confirm")
TOP_FRACTION = 0.01
KINDS = ("observer", "pd", "adl_corpus")
DEFAULT_KINDS = ["observer", "pd"]  # the Stage-0 set; adl_corpus is a set of its own


def taken_item_ids(runs: Path) -> set[str]:
    """Ids in the manual-label store and in every set's mapping or key under ``runs``."""
    from residual_disposition.judge import read_manual_labels

    taken = {label.item_id for label in read_manual_labels(data_root() / "labels" / "manual")}
    for path in [*runs.glob("*/*/mapping.json"), *runs.glob("*/*/key.json")]:
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        items = data.get("items", [])
        taken |= set(items) if isinstance(items, dict) else {e["item_id"] for e in items}
    return taken


def draw_item_ids(count: int, stream: str, taken: set[str]) -> list[str]:
    rng, out, used = random.Random(stream), [], set(taken)
    while len(out) < count:
        item_id = f"{rng.getrandbits(48):012x}"
        if item_id not in used:
            used.add(item_id)
            out.append(item_id)
    return out


def latent_states(index: Path) -> dict[str, tuple[str, int]]:
    """state id → (behavior, dose) of the latent (suppressed) states of the pilot behaviors."""
    out = {}
    for line in index.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "suppressed" and r["behavior_id"] in BEHAVIORS and r["dose"] is not None:
            out[r["state_id"]] = (r["behavior_id"], int(r["dose"]))
    return out


def observer_pool(cells: list[Path], states: dict[str, tuple[str, int]]) -> dict[str, list]:
    """Per behavior, (source, dose, text) of every detector-silent observer answer."""
    obs = read_frame(cells, "detector_obs")
    texts = read_responses(cells, "detector_obs")
    obs = obs[
        obs["state_id"].isin(states)
        & obs["measurement"].isin(OBSERVER_MEASUREMENTS)
        & ~obs["hit"].astype(bool)
    ]
    pool: dict[str, list] = defaultdict(list)
    for row in obs.sort_values("response_id").itertuples():
        behavior, dose = states[row.state_id]
        if row.behavior_id != behavior:
            continue
        pool[behavior].append((f"observer:{row.response_id}", dose, texts[row.response_id]))
    return pool


def pd_pool(
    cells: list[Path], states: dict[str, tuple[str, int]], behaviors: dict
) -> dict[str, list]:
    """Per behavior, (source, dose, text) of every top-set PD continuation the trace detector
    does not flag."""
    from residual_disposition.tools import pd_top

    pool: dict[str, list] = defaultdict(list)
    for cell in cells:
        state = OmegaConf.load(cell / "config.yaml").params.state
        if state not in states:
            continue
        behavior, dose = states[state]
        hit = trace_detector(behaviors[behavior].detector)
        with gzip.open(cell / "pd_completions.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for config, top in pd_top(rows, budget=None, fraction=TOP_FRACTION).items():
            for r in top:
                if not hit(r["completion"]):
                    source = f"pd:{cell.name}:{config}:{r['prefill']}"
                    pool[behavior].append((source, dose, r["completion"]))
    return pool


def corpus_pool(
    cells: list[Path],
    states: dict[str, tuple[str, int]],
    behaviors: dict,
    prefixes: dict[str, tuple[str, str, list[int]]] | None = None,
    rescore: bool = False,
) -> dict[str, list]:
    """Per behavior, (source, dose, text) of every ADL-corpus continuation of a latent state that
    the trace detector does not flag; each cell's recorded detector must be the current one, or,
    with ``rescore`` (Stage 3, s6c 2), its saved continuations are rescored with the current one.
    ``prefixes`` collects each source's (model, revision, prefix token ids), for the item text."""
    pool: dict[str, list] = defaultdict(list)
    for cell in cells:
        if json.loads((cell / "meta.json").read_text()).get("status") != "completed":
            continue
        state = OmegaConf.load(cell / "config.yaml").params.state
        if state not in states:
            continue
        behavior, dose = states[state]
        model = OmegaConf.load(cell / "config.yaml").model
        detector = behaviors[behavior].detector
        if not rescore:  # Stage 3 rescores the saved continuations instead (s6c 2)
            recorded = json.loads((cell / "adl_corpus.json").read_text())["detector_version"]
            if recorded != detector.version:
                raise SystemExit(f"{cell}: detector {recorded}, not the current {detector.version}")
        hit = trace_detector(detector)
        with gzip.open(cell / "continuations.jsonl.gz", "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not hit(r["text"]):
                    source = f"adl_corpus:{cell.name}:{r['framing']}:{r['length']}:{r['index']}"
                    pool[behavior].append((source, dose, r["text"]))
                    if prefixes is not None:
                        prefixes[source] = (model.name, model.revision, list(r["prefix"]))
    return pool


def prefix_texts(prefixes: dict[str, tuple[str, str, list[int]]], sources: list[str]) -> dict:
    """Each source's prefix, decoded exactly with its cell's tokenizer (the model and revision
    the cell recorded); a token id outside the tokenizer's vocabulary is refused."""
    from transformers import AutoTokenizer

    tokenizers: dict[tuple[str, str], Any] = {}
    out = {}
    for source in sources:
        name, revision, ids = prefixes[source]
        if (name, revision) not in tokenizers:
            tokenizers[name, revision] = AutoTokenizer.from_pretrained(name, revision=revision)
        tokenizer = tokenizers[name, revision]
        if not ids or max(ids) >= len(tokenizer):
            raise SystemExit(f"{source}: its prefix is not in {name}@{revision}'s vocabulary")
        out[source] = tokenizer.decode(ids)
    return out


def stage3_sources(behavior: str, adl: bool = False) -> dict[str, Any]:
    """A panel behavior's committed diagnostic sources (Stage 3, E4;
    ``continuation_stage3.committed_sources``, recomputed): its latent states at the source doses
    and the runs they name, L's full PD pools and exploratory passes, found by name. With
    ``adl``, the committed ADL-corpus sources (``committed_adl_sources``; s6a 2): each run's L at
    its dose, and the runs."""
    from residual_disposition import continuation_stage3 as S3
    from residual_disposition.continuation_kstar import find_run

    if adl:
        data = S3.committed_adl_sources(REPO_ROOT, behavior, cr.load_rules(), find_run)
        out = {"states": {}, "adl_corpus": [], "sha256": data["sha256"],
               "manifest": S3.adl_sources_path(REPO_ROOT, behavior).name}  # fmt: skip
        for r in data["runs"]:
            run = find_run(r["run"])
            if run is None:
                raise SystemExit(f"{behavior}: the ADL-corpus source {r['run']} is not here")
            out["states"][r["latent"]] = (behavior, int(r["dose"]))
            out["adl_corpus"].append(run)
        return out
    data = S3.committed_sources(REPO_ROOT, behavior, cr.load_rules(), find_run)
    if not data["doses"]:
        raise SystemExit(f"{behavior}: no stack sources; the continued screening's are its own")
    out: dict[str, Any] = {"states": {}, "pd": [], "observer": [], "sha256": data["sha256"],
                           "manifest": S3.sources_path(REPO_ROOT, behavior).name}  # fmt: skip
    for dose in data["doses"]:
        pd, obs = find_run(data["pd"][str(dose)]), find_run(data["observer"][str(dose)])
        if pd is None or obs is None:
            raise SystemExit(f"{behavior}: a source run at {dose} is not under the runs root")
        out["states"][str(OmegaConf.load(pd / "config.yaml").params.state)] = (behavior, int(dose))
        out["pd"].append(pd)
        out["observer"].append(obs)
    return out


def stage3_chosen(
    behavior: str, kinds: list[str], prefixes: dict | None = None
) -> tuple[list[tuple], dict[str, dict]]:
    """A panel behavior's Stage-3 diagnostic draw (E4): from its committed sources' runs alone,
    per input type the detector-silent pool and a sample of ``PER_STRATUM`` with the frozen seed
    (``derive_seed(labels seed, "detector_diagnostic", kind, behavior)``), as (kind, source,
    dose, text); fewer stop it. The builder draws its set with it, and the result's writer
    draws it again to check the set (review s5v 3). ``["adl_corpus"]``, a set of its own: from
    the committed ADL-corpus sources, the saved continuations rescored with the current detector
    (s6c 2), their prefixes collected in ``prefixes`` for the item text."""
    rules = cr.load_rules()
    seed = int(rules["seeds"]["labels"])
    adl = kinds == ["adl_corpus"]
    stage3 = stage3_sources(behavior, adl=adl)
    spec = OmegaConf.to_container(load_config(f"behaviors/{behavior}").behavior)
    behaviors = {behavior: behavior_from_config(spec)}
    builders = {
        "observer": lambda: observer_pool(stage3["observer"], stage3["states"]),
        "pd": lambda: pd_pool(stage3["pd"], stage3["states"], behaviors),
        "adl_corpus": lambda: corpus_pool(
            stage3["adl_corpus"], stage3["states"], behaviors, prefixes, rescore=True
        ),
    }
    chosen, sizes = [], {}
    for kind in kinds:
        pool = builders[kind]()[behavior]
        sizes[kind] = {behavior: len(pool)}
        if len(pool) < PER_STRATUM:
            raise SystemExit(f"{kind}/{behavior}: only {len(pool)} negatives")
        rng = random.Random(derive_seed(seed, "detector_diagnostic", kind, behavior))
        chosen += [(kind, *p) for p in rng.sample(pool, PER_STRATUM)]
    return chosen, sizes


def detector_diagnostic(args: argparse.Namespace) -> None:
    rules = cr.load_rules()
    seed = int(rules["seeds"]["labels"])
    root = REPO_ROOT / args.runs
    adl3 = bool(args.behavior) and list(args.kinds) == ["adl_corpus"]  # Stage 3's own ADL set
    stage3 = stage3_sources(args.behavior, adl=adl3) if args.behavior else None
    if stage3 and list(args.kinds) not in (DEFAULT_KINDS, ["adl_corpus"]):
        raise SystemExit("Stage 3's sets are observer and PD, or the ADL-corpus set of its own")
    names = (args.behavior,) if stage3 else BEHAVIORS
    states = stage3["states"] if stage3 else latent_states(args.state_index)
    specs = {b: OmegaConf.to_container(load_config(f"behaviors/{b}").behavior) for b in names}
    behaviors = {b: behavior_from_config(s) for b, s in specs.items()}
    prefixes: dict[str, tuple[str, str, list[int]]] = {}
    builders = {
        "observer": lambda: observer_pool(
            stage3["observer"] if stage3 else sorted(root.glob("*-rates-*")), states
        ),
        "pd": lambda: pd_pool(
            stage3["pd"] if stage3 else sorted(root.glob("*-tool_pd-*")), states, behaviors
        ),
        "adl_corpus": lambda: corpus_pool(
            sorted((REPO_ROOT / args.corpus_runs).glob("*-adl_corpus-*")),
            states,
            behaviors,
            prefixes,
        ),
    }
    kinds = list(args.kinds)
    pools = {} if stage3 else {kind: builders[kind]() for kind in kinds}
    chosen = []
    pool_sizes: dict[str, dict[str, int]] = {}
    if stage3:
        drawn, pool_sizes = stage3_chosen(args.behavior, kinds, prefixes)
        chosen = [(kind, args.behavior, *rest) for kind, *rest in drawn]
    for kind, by_behavior in pools.items():
        for behavior in names:
            pool = by_behavior[behavior]
            pool_sizes.setdefault(kind, {})[behavior] = len(pool)
            if len(pool) < PER_STRATUM:
                raise SystemExit(f"{kind}/{behavior}: only {len(pool)} negatives")
            rng = random.Random(derive_seed(seed, "detector_diagnostic", kind, behavior))
            chosen += [(kind, behavior, *p) for p in rng.sample(pool, PER_STRATUM)]
    random.Random(derive_seed(seed, "detector_diagnostic", "order")).shuffle(chosen)
    # Every item is a detector negative under the detector that defines its stratum.
    for kind, behavior, source, _, text in chosen:
        detector = behaviors[behavior].detector
        flagged = detector(text) if kind == "observer" else trace_detector(detector)(text)
        if flagged:
            raise SystemExit(f"{source}: the detector fires on a negative sample")
    taken = taken_item_ids(REPO_ROOT / "runs")
    item_ids = draw_item_ids(len(chosen), f"{seed}:detector_diagnostic:item-ids", taken)
    items, key, mapping = [], {}, []
    # An ADL-corpus item shows its prefix in brackets, then the continuation (PD-clean's format):
    # the selection and the detector's check stay on the continuation alone.
    shown = prefix_texts(prefixes, [c[2] for c in chosen if c[0] == "adl_corpus"])
    for item_id, (kind, behavior, source, dose, text) in zip(item_ids, chosen, strict=True):
        items.append(
            {
                "item_id": item_id,
                "behavior_id": behavior,
                "behavior_description": specs[behavior]["description"],
                "response": f"[{shown[source]}]{text}" if kind == "adl_corpus" else text,
            }
        )
        key[item_id] = {
            "behavior_id": behavior,
            "stratum": kind,
            "detector_hit": False,
            "dose": dose,
            "source": source,
            "detector_version": behaviors[behavior].detector.version,
        }
        mapping.append({"item_id": item_id, "behavior_id": behavior, "stratum": kind})
    blob = "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    set_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    if args.dry_run:
        print(json.dumps({"set_id": set_id, "items": len(items), "pool_sizes": pool_sizes}))
        return
    name = "detector-diagnostic" + ("" if kinds == DEFAULT_KINDS else "-" + "-".join(kinds))
    if stage3:  # within a run name's 34 characters
        name = f"diagnostic-s3{'adl' if adl3 else ''}-{args.behavior}"
    cfg = load_config(
        "base",
        ["+experiment=continuation_labels", f"name={name}", f"seed={seed}"]
        + ["tracking.wandb.enabled=false"],
    )
    with Run(cfg) as run:
        run.path("input.jsonl").write_text(blob)
        meta = {"set_id": set_id, "seed": seed, "source_runs": str(args.runs)}
        if kinds != DEFAULT_KINDS:  # the Stage-0 set's key keeps its format
            sources = [str(args.runs)] if set(kinds) - {"adl_corpus"} else []
            sources += [str(args.corpus_runs)] if "adl_corpus" in kinds else []
            meta = {"set_id": set_id, "seed": seed, "kinds": kinds, "source_runs": sources}
        if stage3:  # bound to the committed sources (E4)
            meta = {
                "set_id": set_id,
                "seed": seed,
                "kinds": kinds,
                "behavior": args.behavior,
                "sources": {"manifest": stage3["manifest"], "sha256": stage3["sha256"]},
            }
        run.path("key.json").write_text(json.dumps({**meta, "items": key}, indent=2) + "\n")
        run.path("mapping.json").write_text(
            json.dumps({"set_id": set_id, "items": mapping}, indent=2) + "\n"
        )
        summary = {
            "set_id": set_id,
            "items": len(items),
            "kinds": kinds,
            "per_stratum": PER_STRATUM,
            "pool_sizes": pool_sizes,
            "latent_states": {s: list(v) for s, v in sorted(states.items())},
            "top_fraction": TOP_FRACTION,
            "taken_item_ids": len(taken),
            "seed": seed,
            "detector_versions": {b: behaviors[b].detector.version for b in names},
        }
        run.summarize(summary)
    print(json.dumps(summary, indent=2))
    print(run.dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("detector-diagnostic")
    diag.add_argument("--runs", default="runs/feasibility_14b")
    diag.add_argument("--kinds", nargs="+", choices=KINDS, default=list(DEFAULT_KINDS))
    diag.add_argument("--corpus-runs", default="runs/continuation_14b")
    diag.add_argument("--state-index", type=Path, default=STATE_INDEX)
    diag.add_argument("--behavior", default=None, help="Stage 3: a panel behavior (E4)")
    diag.add_argument("--dry-run", action="store_true", help="report the pools; write nothing")
    args = parser.parse_args()
    if args.command == "detector-diagnostic":
        detector_diagnostic(args)


if __name__ == "__main__":
    main()
