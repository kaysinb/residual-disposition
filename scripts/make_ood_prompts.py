"""Build the out-of-distribution observer prompts (preregistration §12.3.5).

    uv run python scripts/make_ood_prompts.py [configs/prompts/ood.yaml]

Downloads the pinned sources, builds the two sets with ``prompts.build_ood_sets`` and saves them
in ``data/prompts/ood-<sha12 of the ood section>/`` (``<set>.jsonl`` and ``manifest.json``, as the
other prompt pools). The excluded prompts are every set of the pilot pool named by
``exclude_pool``. When ``expected`` holds set hashes, a build that differs is refused; when it
is null for a set, its hash to freeze is printed; a set with a frozen hash must match it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import prompts as P
from residual_disposition.paths import CONFIG_ROOT, data_root


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_ROOT / "prompts" / "ood.yaml"
    spec = OmegaConf.to_container(OmegaConf.load(path).ood, resolve=True)
    pool_dir = data_root() / spec["exclude_pool"]
    manifest = json.loads((pool_dir / "manifest.json").read_text())
    excluded = [p.text for name in manifest["sets"] for p in P.load_set(pool_dir, name)]
    reference = P.download_rows(spec["reference"])
    rows = {name: P.download_rows(source) for name, source in spec["sets"].items()}
    sets = P.build_ood_sets(spec, rows, reference, excluded)
    hashes = {name: P.set_sha256(prompts) for name, prompts in sets.items()}
    expected = spec.get("expected") or {}
    wrong = {k: v for k, v in hashes.items() if expected.get(k) not in (None, v)}
    if wrong:
        print(f"refused: the build differs from the frozen hashes for {sorted(wrong)}")
        return 1
    unfrozen = sorted(k for k in hashes if expected.get(k) is None)
    if unfrozen:
        print(f"not yet frozen: {unfrozen}; write their hashes into the config")
    blob = json.dumps({k: v for k, v in spec.items() if k != "expected"}, sort_keys=True)
    out = data_root() / "prompts" / f"ood-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"
    source = {
        "config": str(path),
        "sets": {
            name: {k: s[k] for k in ("dataset", "revision", "file")}
            for name, s in spec["sets"].items()
        },
        "excluded_pool": spec["exclude_pool"],
        "excluded_prompts": len(excluded),
    }
    P.save_sets(out, sets, source)
    print(json.dumps({"dir": str(out), "set_sha256": hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
