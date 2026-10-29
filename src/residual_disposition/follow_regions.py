"""Certified decision regions of the follow-up's identification test (docs/preregistration.md
§13.5 (c)), and the decisions they give.

The test is §12.5.1's exact test at n = 80 per state, ">" form: L − H_s > 3/10 and L − X > 1/5,
each at Holm's six levels 1/120, 1/100, 1/80, 1/60, 1/40 and 1/20 (§13.5 (d)). A region is the
test's certified final decisions at one level: per x_C the threshold τ(x_C), rejecting x_L ≥ τ
(``ShiftedDifferenceTest.decision_region``'s ``tau_use``: undecided outcomes are not rejected).

``scripts/certify_follow_regions.py`` certifies the twelve cells by
``simulate_continuation.reader_cell`` with §12.5.1's frozen settings and keeps them in its run;
``write_manifest`` binds that run's cells into ``manifests/follow_regions.json``. Readers load the
manifest with ``load``, which refuses a record with a missing, failed or unnested cell.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

N = 80
FORM = ">"
ALPHA = Fraction(1, 20)  # the family level; Holm's levels are α/6 … α/1
LEVELS = tuple(ALPHA / k for k in (6, 5, 4, 3, 2, 1))
TESTS = {"adl_vs_sham_0.3": Fraction(3, 10), "adl_vs_x": Fraction(1, 5)}
CELLS = tuple((name, delta, alpha) for name, delta in TESTS.items() for alpha in LEVELS)
MANIFEST = REPO_ROOT / "manifests" / "follow_regions.json"
CELL_FIELDS = ("test", "delta", "form", "alpha", "n", "monotone", "nested", "convex",
               "undecided", "tau_use", "size_upper", "passed")  # fmt: skip


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def cell_problems(cells: Sequence[Mapping[str, Any]], n: int = N) -> list[str]:
    """Why ``cells`` cannot serve as the twelve regions: a missing, duplicated or extra cell, a
    failed certification (``passed`` holds monotonicity, nesting of tail sets and the certified
    size), an undecided outcome, a region of the wrong shape, or a region at a lower level that is
    not inside the region at the next higher level."""
    problems = []
    keyed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for c in cells:
        key = (str(c.get("test")), str(c.get("alpha")))
        if key in keyed:
            problems.append(f"duplicated cell {key}")
        keyed[key] = c
    expected = {(name, str(alpha)): delta for name, delta, alpha in CELLS}
    for key in sorted(set(keyed) - set(expected)):
        problems.append(f"unexpected cell {key}")
    for key, delta in expected.items():
        c = keyed.get(key)
        if c is None:
            problems.append(f"missing cell {key}")
            continue
        if c.get("delta") != str(delta) or c.get("form") != FORM or c.get("n") != n:
            problems.append(f"{key}: delta, form or n differ")
        if not (c.get("passed") and c.get("monotone") and c.get("nested")):
            problems.append(f"{key}: not certified")
        if c.get("undecided"):
            problems.append(f"{key}: {len(c['undecided'])} undecided outcomes")
        tau = c.get("tau_use")
        if not isinstance(tau, list) or len(tau) != n + 1 or not all(0 <= t <= n + 1 for t in tau):
            problems.append(f"{key}: tau_use is not {n + 1} thresholds in 0…{n + 1}")
    for name in TESTS:
        taus = [keyed.get((name, str(a)), {}).get("tau_use") for a in LEVELS]
        for i in range(len(LEVELS) - 1):
            t_lo, t_hi = taus[i], taus[i + 1]
            comparable = (
                isinstance(t_lo, list) and isinstance(t_hi, list) and len(t_lo) == len(t_hi)
            )
            if comparable and any(a < b for a, b in zip(t_lo, t_hi, strict=True)):
                problems.append(
                    f"{name}: the region at {LEVELS[i]} is not inside the one at {LEVELS[i + 1]}"
                )
    return problems


class Regions:
    """The certified regions, by test and level."""

    def __init__(self, cells: Sequence[Mapping[str, Any]], n: int = N):
        problems = cell_problems(cells, n)
        if problems:
            raise ValueError("certified regions refused: " + "; ".join(problems))
        self.n = n
        self._tau = {(c["test"], Fraction(c["alpha"])): tuple(c["tau_use"]) for c in cells}

    def rejects(self, test: str, alpha: Fraction, x_l: int, x_c: int) -> bool:
        """The certified decision at (x_L, x_C): reject iff x_L ≥ τ(x_C)."""
        if not (0 <= x_l <= self.n and 0 <= x_c <= self.n):
            raise ValueError(f"counts out of range: x_L {x_l}, x_C {x_c}, n {self.n}")
        return x_l >= self._tau[(test, Fraction(alpha))][x_c]

    def identifies(self, alpha: Fraction, x_l: int, x_hs: int, x_x: int) -> bool:
        """§13.5 (c) at level α: both contrasts reject (an intersection–union test)."""
        return self.rejects("adl_vs_sham_0.3", alpha, x_l, x_hs) and self.rejects(
            "adl_vs_x", alpha, x_l, x_x
        )

    def by_level(self, x_l: int, x_hs: int, x_x: int) -> dict[Fraction, bool]:
        """The identification decision at each of Holm's levels, for ``holm_from_decisions``."""
        return {alpha: self.identifies(alpha, x_l, x_hs, x_x) for alpha in LEVELS}


def manifest_record(run_id: str, cells: Sequence[Mapping[str, Any]], git: Mapping[str, Any],
                    run_sha256: str) -> dict[str, Any]:  # fmt: skip
    """The committed record: each cell's certified fields, bound to the run that computed them."""
    kept = [{k: c[k] for k in CELL_FIELDS} for c in cells]
    kept.sort(key=lambda c: (c["test"], Fraction(c["alpha"])))
    return {"run": run_id, "regions_sha256": run_sha256, "commit": git.get("commit"),
            "dirty": git.get("dirty"), "cells": kept,
            "cells_sha256": canonical_sha256(kept)}  # fmt: skip


def write_manifest(run_dir: Path, path: Path = MANIFEST) -> dict[str, Any]:
    """Bind a completed certification run into the manifest, create-only. The run must come from
    a clean commit and hold all twelve certified cells."""
    meta = json.loads((run_dir / "meta.json").read_text())
    git = meta.get("git") or {}
    if meta.get("status") != "completed" or git.get("commit") is None or git.get("dirty"):
        raise ValueError(f"{run_dir}: not a completed run from a clean commit")
    regions = run_dir / "regions.json"
    cells = json.loads(regions.read_text())
    problems = cell_problems(cells)
    if problems:
        raise ValueError("; ".join(problems))
    digest = hashlib.sha256(regions.read_bytes()).hexdigest()
    record = manifest_record(meta["run_id"], cells, git, digest)
    if path.exists():
        raise FileExistsError(f"{path} exists; a manifest is written once")
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def load(path: Path = MANIFEST) -> Regions:
    """The committed regions, their cells' hash rechecked."""
    record = json.loads(path.read_text())
    if canonical_sha256(record["cells"]) != record["cells_sha256"]:
        raise ValueError(f"{path}: cells do not match their sha256")
    return Regions(record["cells"])
