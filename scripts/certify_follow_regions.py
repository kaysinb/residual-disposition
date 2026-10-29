"""Certify the follow-up's twelve decision regions (preregistration §13.5 (c)) on CPU.

    uv run python scripts/certify_follow_regions.py experiments/certify_follow_regions
    uv run python scripts/certify_follow_regions.py --manifest runs/certify_follow_regions/<run>

The first form certifies every (test, level) cell of ``follow_regions.CELLS`` with
``simulate_continuation.reader_cell`` and §12.5.1's frozen settings, one process per cell, and
writes ``regions.json`` into its run (a clean commit is required). It exits with status 1 unless
every cell passes ``follow_regions.cell_problems``. The second form binds such a run into
``manifests/follow_regions.json``, once.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from residual_disposition import continuation_rules as cr
from residual_disposition import follow_regions as fr
from residual_disposition import simulate_continuation as sc
from residual_disposition.config import from_cli
from residual_disposition.tracking import Run


def _cell(job: tuple) -> dict:
    name, delta, alpha = job
    rules = cr.load_rules(expected_hash=sc.FROZEN["rules_sha256"])
    return sc.reader_cell(sc.make_test(rules, fr.N, delta, fr.FORM), name, alpha)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--manifest":
        record = fr.write_manifest(Path(sys.argv[2]))
        print(f"wrote {fr.MANIFEST} (cells sha256 {record['cells_sha256']})")
        return 0
    cfg = from_cli()
    with Run(cfg) as run:
        workers = int(cfg.certify.workers)
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            cells = list(pool.map(_cell, fr.CELLS))
        run.path("regions.json").write_text(json.dumps(cells, indent=1) + "\n")
        problems = fr.cell_problems(cells)
        sizes = {f"{c['test']}@{c['alpha']}": c["size_float"] for c in cells}
        run.summarize({"cells": len(cells), "problems": problems, "size_upper": sizes})
        for c in cells:
            print(
                f"{c['test']} δ {c['delta']} α {c['alpha']}: passed {c['passed']}, size "
                f"{c['size_float']:.6f}, undecided {len(c['undecided'])}, {c['seconds']} s"
            )
        print("problems: " + ("; ".join(problems) if problems else "none"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
