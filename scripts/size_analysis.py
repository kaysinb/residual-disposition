"""The size block's certificates and comparison (docs/preregistration.md §13.9, amendment of
2026-11-05), on CPU from the recorded runs: each implantation's sign certificate at both rates,
and, with all four implantations, the size comparison (``size_block.analysis``). The anchor's
blocks come from its recorded certificate run, unchanged; the new implantations' from their
``blocks-certificate`` records. A missing implantation gives a labeled reduced analysis (the
comparisons that need it are left out), never a substitution.

    uv run python scripts/size_analysis.py [--out PATH] [--resamples N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from residual_disposition import size_block as S
from residual_disposition import size_block_run as R
from residual_disposition.paths import runs_root


def anchor_record(inputs: dict) -> dict:
    """The anchor's six blocks as a blocks record (its recorded run summaries)."""
    path = runs_root() / inputs["anchor"]["certificate"] / "certificate_blocks.json"
    d = json.loads(path.read_text())
    runs = {}
    for key, s in d["runs"].items():
        block, rate, habit, role = key.split("/")
        runs[f"{block}/{float(rate):g}/{habit}/{role}"] = s
    return {"implantation": "14B-1", "blocks": list(range(6)), "runs": runs,
            "budget": d.get("budget"), "source": str(path.relative_to(runs_root()))}  # fmt: skip


def new_record(name: str) -> dict | None:
    path = runs_root() / f"size_{R.CONFIGS[name]}" / "records" / "blocks-certificate.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="default: $RUNS_DIR/size_analysis.json")
    ap.add_argument("--resamples", type=int, default=None)
    args = ap.parse_args()
    rules, inputs = S.load_size_rules(), R.load_inputs()
    records = {"14B-1": anchor_record(inputs)}
    for name in R.CONFIGS:
        rec = new_record(name)
        if rec is not None:
            records[name] = rec
    contrasts = {n: R.block_contrasts(rules, r) for n, r in records.items()}
    blocks = {}
    for n, r in records.items():
        per = {}
        for lr in [float(x) for x in rules["learning_rates"]]:
            per[lr] = [S.block_of({(h, ro): r["runs"].get(f"{j}/{lr:g}/{h}/{ro}")
                                   for h in S.HABITS for ro in S.ROLES}, int(j))
                       for j in r["blocks"]]  # fmt: skip
        blocks[n] = S.certificate(rules, per)
    out = {"implantations": sorted(records), "certificates": blocks}
    missing = sorted(set(rules["implantations"]) - set(records))
    if missing:
        out["analysis"] = None
        out["reduced"] = {
            "missing": missing,
            "note": "the full comparison waits for every implantation",
        }
    else:
        res = S.analysis(rules, contrasts, resamples=args.resamples)
        out["analysis"] = {k: {**v, "means": {n: list(m) for n, m in v["means"].items()},
                               "comparisons": {c: list(b) for c, b in v["comparisons"].items()}}
                           for k, v in res.items()}  # fmt: skip
    args.out = args.out or runs_root() / "size_analysis.json"
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
