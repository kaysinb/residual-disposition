"""Build the strong branch's validation suites on CPU (amendment of 2026-10-21, §12.4.10;
C16 item A): V1–V3 for PD and ADL and O1–O3 for the observer, random samples from their
declared populations (``residual_disposition.strong_suites``), saved create-only under
``data/strong_suites/<kind>-<name>/`` with their manifests; prints their hashes for
``configs/continuation/strong.yaml`` (frozen with the amendment's text).

    uv run python scripts/make_strong_suites.py --seed SEED [--only pd|adl|observer]

- PD: allenai/c4 ``en`` validation (pinned revision and files), the model's tokenizer; the frozen
  pool's prefills are first reproduced from c4-10k (count and hash as ``pipeline.FROZEN_PD``),
  then excluded.
- ADL: fineweb-1m-sample's train shards 2–6 (never read by ADL), ADL's fragment rule; the
  documents of the screening fragments (shards 0–1 in file order, the first 10,000 accepted) are
  excluded by their text's sha256.
- Observer: the prompt pool rebuilt from its source and checked to give the saved sets, then the
  slices after the frozen sets and the confirmation set.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

C4_VALIDATION = {
    "dataset": "allenai/c4",
    "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "files": [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)],
    "field": "text",
}
FINEWEB_UNREAD = [f"data/train-{i:05d}-of-00007.parquet" for i in range(2, 7)]


def tokenizer():
    from transformers import AutoTokenizer

    from residual_disposition.config import load_config

    model = OmegaConf.to_container(load_config("model/qwen3-14b").model, resolve=True)
    return AutoTokenizer.from_pretrained(model["name"], revision=model["revision"]), model


def c4_validation_texts():
    from huggingface_hub import hf_hub_download

    for name in C4_VALIDATION["files"]:
        path = hf_hub_download(repo_id=C4_VALIDATION["dataset"], filename=name,
                               repo_type="dataset", revision=C4_VALIDATION["revision"])  # fmt: skip
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)[C4_VALIDATION["field"]]


def frozen_prefills(tok):
    """The frozen PD pool's prefills, reproduced from its corpus and checked against FROZEN_PD."""
    from residual_disposition import pipeline
    from residual_disposition.config import load_config
    from residual_disposition.tools import corpus_texts, prefills_from_texts

    cfg = load_config("experiments/continuation_14b/example_link")
    spec = OmegaConf.to_container(cfg.audit.tools.perplexity_differencing, resolve=True)
    prefills = prefills_from_texts(tok, corpus_texts(spec["corpus"]), spec["prefill_tokens"])
    digest = pipeline.content_hash([list(p.tokens) for p in prefills])
    frozen = pipeline.FROZEN_PD
    if (len(prefills), digest) != (frozen["prefills"], frozen["prefills_sha256"]):
        raise SystemExit(f"the frozen PD pool is not reproduced: {len(prefills)}, {digest[:12]}")
    return [p.tokens for p in prefills]


def screening_documents(tok, n, count):
    """The text sha256 of every document behind ADL's screening fragments (``adl.adl_sequences``
    over its corpus in file order, the first ``count`` accepted)."""
    from residual_disposition.adl import corpus_texts
    from residual_disposition.config import load_config

    cfg = load_config("experiments/continuation_14b/example_link")
    spec = OmegaConf.to_container(cfg.audit.adl.corpus, resolve=True)
    out = set()
    for text in corpus_texts(spec):
        if not text or not text.strip():
            continue
        if len(tok.encode(text[: n * 10], add_special_tokens=True)) >= n:
            out.add(hashlib.sha256(text.encode()).hexdigest())
            if len(out) == count:
                return out, spec
    raise SystemExit(f"ADL's corpus gave {len(out)} fragments, not {count}")


def save(root: Path, suite, extra) -> dict:
    out = root / f"{suite.kind}-{suite.name}"
    if out.exists():
        raise SystemExit(f"{out} exists: a suite is built once")
    out.mkdir(parents=True)
    (out / "records.json").write_text(json.dumps(suite.records) + "\n")
    manifest = {**suite.manifest(), **extra}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from residual_disposition import adl
    from residual_disposition import strong_suites as S
    from residual_disposition.paths import data_root

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="the suites' frozen base seed")
    parser.add_argument("--only", choices=("pd", "adl", "observer"), default=None)
    args = parser.parse_args(argv)
    root = data_root() / "strong_suites"
    tok, model = tokenizer()
    printed = {}
    if args.only in (None, "pd"):
        frozen = frozen_prefills(tok)
        pop = S.pd_population(tok, c4_validation_texts(), exclude=frozen)
        suites = S.draw_pd_suites(pop, args.seed)
        S.check_disjoint_suites(suites.values(), frozen=frozen)
        extra = {
            "source": C4_VALIDATION,
            "tokenizer": model,
            "excluded": "the frozen PD pool's 9,194 prefills",
        }
        for s in suites.values():
            printed[f"pd-{s.name}"] = save(root, s, extra)
    if args.only in (None, "adl"):
        screened, spec = screening_documents(tok, adl.SEQUENCE_TOKENS, adl.SEQUENCES)
        from residual_disposition.adl import corpus_texts

        texts = corpus_texts({**spec, "files": FINEWEB_UNREAD})
        pop = [(d, t) for d, t in S.adl_population(tok, texts) if d not in screened]
        suites = S.draw_adl_suites(pop, args.seed)
        S.check_disjoint_suites(suites.values(), frozen=[{"document": d} for d in screened])
        extra = {
            "source": {**spec, "files": FINEWEB_UNREAD},
            "tokenizer": model,
            "excluded": "the screening fragments' documents",
        }
        for s in suites.values():
            printed[f"adl-{s.name}"] = save(root, s, extra)
    if args.only in (None, "observer"):
        from residual_disposition.config import load_config
        from residual_disposition.continuation_confirm import FROZEN as CONF
        from residual_disposition.prompts import build_pool, download_alpaca, set_sha256, split_pool

        spec = OmegaConf.to_container(load_config("prompts/default").prompts, resolve=True)
        pool = build_pool(download_alpaca(spec["source"]), min_chars=spec["min_chars"],
                          max_chars=spec["max_chars"])  # fmt: skip
        saved = json.loads(
            (data_root() / "prompts" / CONF["source_pool"] / "manifest.json").read_text()
        )
        for name, prompts in split_pool(pool, spec["sets"], spec["seed"]).items():
            if set_sha256(prompts) != saved["sets"][name]["sha256"]:
                raise SystemExit(f"{name}: the pool does not give the saved set")
        sizes = {**spec["sets"], "confirmation": CONF["prompts"]}
        suites = S.observer_suites(pool, sizes, spec["seed"])
        extra = {"source": spec["source"], "pool_id": CONF["source_pool"]}
        for s in suites.values():
            printed[f"observer-{s.name}"] = save(root, s, extra)
    print(json.dumps({k: {"sha256": v["sha256"], "records": v["records"],
                          "population": v.get("population", v.get("pool"))}
                      for k, v in printed.items()}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
