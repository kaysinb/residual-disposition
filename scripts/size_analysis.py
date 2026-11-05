"""The size block's certificates and comparison (docs/preregistration.md §13.9, amendment of
2026-11-05), on CPU from the recorded runs: each implantation's sign certificate at both rates,
the size comparison (``size_block.analysis``) and §13.9's absolute-rate sensitivity (the same
comparison with every run's crossing of the absolute rate 1/4, from its saved curve). The
anchor's blocks come from its recorded certificate run, unchanged; the new implantations' from
their ``blocks-certificate`` records. Without every implantation (an unavailable one, or the
branch without 4B-2) the comparison is a labeled reduced analysis of the present ones, never a
substitution.

    uv run python scripts/size_analysis.py [--out PATH] [--resamples N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from residual_disposition import size_block as S
from residual_disposition import size_block_run as R
from residual_disposition.paths import runs_root


def anchor_record(inputs: dict) -> dict:
    """The anchor's six blocks as a blocks record (its recorded run summaries and run dirs)."""
    path = runs_root() / inputs["anchor"]["certificate"] / "certificate_blocks.json"
    d = json.loads(path.read_text())
    runs, dirs = {}, {}
    for key, s in d["runs"].items():
        block, rate, habit, role = key.split("/")
        new = f"{block}/{float(rate):g}/{habit}/{role}"
        runs[new] = s
        if key in (d.get("run_dirs") or {}):
            dirs[new] = _find(d["run_dirs"][key])
    return {"implantation": "14B-1", "blocks": list(range(6)), "runs": runs, "run_dirs": dirs,
            "budget": d.get("budget"), "source": str(path.relative_to(runs_root()))}  # fmt: skip


def _find(run: str) -> str:
    """A run directory recorded by its name or its path under the runs root."""
    if (runs_root() / run).is_dir():
        return run
    found = sorted(runs_root().glob(f"*/{run}"))
    if len(found) != 1:
        raise SystemExit(f"{run}: {len(found)} run directories")
    return str(found[0].relative_to(runs_root()))


def new_record(name: str) -> dict | None:
    path = runs_root() / f"size_{R.CONFIGS[name]}" / "records" / "blocks-certificate.json"
    return json.loads(path.read_text()) if path.exists() else None


def saved_curve(run_dir: str) -> dict[int, float]:
    """A relearning run's evaluations (step → rate), as saved."""
    from residual_disposition.records import RelearnEval, read_table

    return {e.step: e.hits / e.n for e in read_table([runs_root() / run_dir], RelearnEval.table)}


def absolute_record(rules: Mapping[str, Any], record: Mapping[str, Any],
                    curve: Callable[[str], dict[int, float]] = saved_curve) -> dict:  # fmt: skip
    """``record`` with every run's summary replaced by its absolute-rate crossing
    (``size_block.absolute_crossing``); a run without its saved curve stays missing."""
    runs = {}
    for key, s in record["runs"].items():
        d = (record.get("run_dirs") or {}).get(key)
        if s is None or d is None:
            runs[key] = None
            continue
        runs[key] = S.absolute_crossing(rules, curve(d), int(s["budget"]))
    return {**record, "runs": runs}


def certificates(rules: Mapping[str, Any], records: Mapping[str, Any]) -> dict:
    out = {}
    for n, r in records.items():
        per = {}
        for lr in [float(x) for x in rules["learning_rates"]]:
            per[lr] = [S.block_of({(h, ro): r["runs"].get(f"{j}/{lr:g}/{h}/{ro}")
                                   for h in S.HABITS for ro in S.ROLES}, int(j))
                       for j in r["blocks"]]  # fmt: skip
        out[n] = S.certificate(rules, per)
    return out


def compare(rules: Mapping[str, Any], records: Mapping[str, Any],
            resamples: int | None) -> dict:  # fmt: skip
    """The comparison of the present implantations: full with all four, reduced otherwise."""
    contrasts = {n: R.block_contrasts(rules, r) for n, r in records.items()}
    missing = sorted(set(rules["implantations"]) - set(records))
    res = S.analysis(rules, contrasts, resamples=resamples, reduced=bool(missing))
    return {k: {**v, "means": {n: list(m) for n, m in v["means"].items()},
                "comparisons": {c: list(b) for c, b in v["comparisons"].items()}}
            for k, v in res.items()}  # fmt: skip


def build(rules: Mapping[str, Any], records: Mapping[str, Any], resamples: int | None,
          curve: Callable[[str], dict[int, float]] = saved_curve) -> dict:  # fmt: skip
    missing = sorted(set(rules["implantations"]) - set(records))
    absolute = {n: absolute_record(rules, r, curve) for n, r in records.items()}
    return {
        "implantations": sorted(records),
        "design": "reduced" if missing else "full",
        "missing": missing,
        "certificates": certificates(rules, records),
        "analysis": compare(rules, records, resamples),
        "absolute_sensitivity": {
            "rate": str(rules["absolute_crossing"]),
            "certificates": certificates(rules, absolute),
            "analysis": compare(rules, absolute, resamples),
        },
    }


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
    out = build(rules, records, args.resamples)
    args.out = args.out or runs_root() / "size_analysis.json"
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out} ({out['design']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
