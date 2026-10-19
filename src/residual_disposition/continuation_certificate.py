"""The six-block certificate's runs and their reduction to ``continuation_stats.Block``
(preregistration §12.4.8; execution of 2026-10-19, review s3j), without model code.

A lineage's 72 runs are keyed ``<block>/<rate>/<arm>/<role>``: the block's frozen relearning
seed (0–5), the learning rate, the arm (b, Y, Y2) and the state (L, H_s). A completed run enters
by its latched crossing T (at least one step, ``relearn_rules.certificate_time``), a censored one
by its budget as the low end and +inf as the high end, and a failed or missing run as
incomplete, which gives every contrast that depends on it p = 1 (``continuation_stats``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .continuation_kstar import rate_key
from .continuation_stats import HABITS, ROLES, Block, Crossing
from .relearn_rules import certificate_time


def run_key(block: int, lr: float, arm: str, role: str) -> str:
    if arm not in HABITS or role not in ROLES:
        raise ValueError(f"no arm {arm}/{role} in the certificate")
    return f"{int(block)}/{rate_key(lr)}/{arm}/{role}"


def crossing(summary: Mapping[str, Any] | None) -> Crossing:
    """A run's crossing as the certificate counts it; None (not run, or failed) is incomplete."""
    if summary is None:
        return Crossing.incomplete()
    if summary.get("censored"):
        return Crossing.censored(float(summary["budget"]))
    t = certificate_time(summary)
    if t is None:
        return Crossing.incomplete()
    return Crossing.observed(t)


def blocks(
    runs: Mapping[str, Mapping[str, Any] | None],
    block_ids: Sequence[int],
    learning_rates: Sequence[float],
) -> dict[float, list[Block]]:
    """The certificate's blocks per learning rate from the run summaries keyed by ``run_key``;
    every arm of the design is present, incomplete where its run is missing."""
    unknown = set(runs) - {
        run_key(b, lr, h, r)
        for b in block_ids
        for lr in learning_rates
        for h in HABITS
        for r in ROLES
    }
    if unknown:
        raise ValueError(f"runs outside the design: {sorted(unknown)[:3]}")
    return {
        float(lr): [
            Block(
                arms={
                    (h, r): crossing(runs.get(run_key(b, lr, h, r))) for h in HABITS for r in ROLES
                },
                block_id=int(b),
            )
            for b in block_ids
        ]
        for lr in learning_rates
    }
