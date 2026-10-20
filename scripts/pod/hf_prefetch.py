"""The pinned Hugging Face corpora a Stage-2 lineage reads, fetched into the space's HF cache
(HF_HOME, set per space by the runner) before any pass, so that a missing or unreachable corpus
stops the session before GPU work (review s4z): the PD corpus (``audit.tools.
perplexity_differencing.corpus``) and the ADL corpus (``audit.adl.corpus``), each file at its
pinned revision. Prints one JSON line {"ok": ..., "files": [{dataset, revision, file, bytes}]}.

    HF_HOME=<space>/hf python scripts/pod/hf_prefetch.py --behavior pun_intended
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def corpus_files(behavior: str) -> list[tuple[str, str, str]]:
    """(dataset, revision, file) of every pinned corpus file the behavior's passes read."""
    from omegaconf import OmegaConf

    from residual_disposition.config import load_config

    cfg = load_config(f"experiments/continuation_14b/{behavior}")
    pd = OmegaConf.to_container(cfg.audit.tools.perplexity_differencing.corpus, resolve=True)
    adl = OmegaConf.to_container(cfg.audit.adl.corpus, resolve=True)
    out = []
    for spec in (pd, adl):
        files = spec.get("files") or [spec["file"]]
        out += [(spec["dataset"], spec["revision"], name) for name in files]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behavior", required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("HF_HOME"):
        print(json.dumps({"ok": False, "error": "HF_HOME is not set: the space's cache"}))
        return 1
    from huggingface_hub import hf_hub_download

    fetched = []
    try:
        for dataset, revision, name in corpus_files(args.behavior):
            path = hf_hub_download(repo_id=dataset, filename=name, repo_type="dataset",
                                   revision=revision)  # fmt: skip
            fetched.append({"dataset": dataset, "revision": revision, "file": name,
                            "bytes": Path(path).stat().st_size})  # fmt: skip
    except Exception as error:  # noqa: BLE001 - any failure stops the session before GPU work
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"[:500],
                          "files": fetched}))  # fmt: skip
        return 1
    print(json.dumps({"ok": True, "hf_home": os.environ["HF_HOME"], "files": fetched}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
