"""Build the continuation's confirmation prompts (preregistration §12.4.1; execution 2026-10-19,
plan C9): the 600 prompts that follow the frozen prompt sets in the pool's one seeded
shuffle (``prompts.extension_set``), disjoint from every screening, exploratory and training
prompt, with no earlier set reshuffled. The pool is first checked to give the saved sets exactly
(their hashes in the saved pool's manifest); the new set is saved as a pool of its own with its
manifest, and its hash is what ``configs/continuation/confirmation.yaml`` freezes before the
first draw.

    uv run python scripts/make_confirmation_prompts.py
"""

from __future__ import annotations

import json
import sys

from omegaconf import OmegaConf

from residual_disposition.config import config_hash, load_config
from residual_disposition.paths import data_root
from residual_disposition.prompts import (
    build_pool,
    check_disjoint,
    download_alpaca,
    extension_set,
    save_sets,
    set_sha256,
    split_pool,
)

COUNT = 600


def main() -> int:
    cfg = load_config("prompts/default")
    spec = OmegaConf.to_container(cfg.prompts, resolve=True)
    saved = data_root() / "prompts" / config_hash(OmegaConf.create({"prompts": spec}))
    manifest = json.loads((saved / "manifest.json").read_text())
    texts = download_alpaca(spec["source"])
    pool = build_pool(texts, min_chars=spec["min_chars"], max_chars=spec["max_chars"])
    sets = split_pool(pool, spec["sets"], spec["seed"])
    for name, prompts in sets.items():
        if set_sha256(prompts) != manifest["sets"][name]["sha256"]:
            raise SystemExit(f"{name}: the pool does not give the saved set of {saved.name}")
    confirmation = extension_set(pool, spec["sets"], spec["seed"], COUNT)
    check_disjoint({**sets, "confirmation": confirmation})
    out = data_root() / "prompts" / f"confirmation-{saved.name}"
    if out.exists():
        raise SystemExit(f"{out} exists: the confirmation prompts are built once")
    source = {**spec["source"], "pool": saved.name, "offset": sum(spec["sets"].values()),
              "count": COUNT}  # fmt: skip
    written = save_sets(out, {"confirmation": confirmation}, source)
    print(json.dumps({"pool": out.name, "sha256": written["sets"]["confirmation"]["sha256"],
                      "count": COUNT, "offset": source["offset"]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
