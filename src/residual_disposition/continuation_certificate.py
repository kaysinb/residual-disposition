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
    """A run's crossing as the certificate counts it; None (not run, or failed) and a summary
    of another status than completed (review s3q) are incomplete."""
    if summary is None or summary.get("status", "completed") != "completed":
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


# What every record of one lineage's certificate shares (``certificate_blocks.json``, one per Pod)
COMMON = (
    "lineage",
    "behavior",
    "dose",
    "states",
    "learning_rates",
    "budget",
    "kstar",
    "manifests",
    "references",
    "data",
    "model",
    "params",
    "rules_sha256",
)


def merge(records: Sequence[Mapping[str, Any]], find_run: Any) -> dict[str, Any]:
    """The runs of a lineage's certificate from the records of its Pods (review s3q): the
    records must share the lineage and every input (``COMMON``: the manifests' hashes, B, k*,
    the references and the arms' data); a run in two records must be the same run with the same
    summary, or the merge is refused. Every run is then checked against its directory
    (``find_run(name)``): a completed relearning cell of the record's model and its arm's
    behavior, on its role's state with its block seed, whose other params are exactly the ones
    the record approved for its arm and rate (``params``: the settings, the evaluation's
    sampling, requests and engine, k*, the stop and the kept responses, the reference rate and
    the data; review s3u), and whose summary is the recorded one; a missing or not completed
    run is None
    (incomplete), and a run of other inputs is refused. Returns ``{"common", "blocks", "runs"}``,
    ``runs`` keyed by ``run_key`` for ``blocks``."""
    import json

    import yaml

    if not records:
        raise ValueError("no certificate record")
    first = records[0]
    for r in records[1:]:
        differ = [k for k in COMMON if r.get(k) != first.get(k)]
        if differ:
            raise ValueError(f"records of other inputs: {differ}")
    seen: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for r in records:
        for key, summary in r["runs"].items():
            entry = (r["run_dirs"][key], summary)
            if key in seen and seen[key] != entry:
                raise ValueError(f"two different runs of {key}")
            seen[key] = entry
    blocks_of = sorted({int(b) for r in records for b in r["blocks"]})
    kstar = first["kstar"]
    runs: dict[str, Mapping[str, Any] | None] = {}
    for key, (name, summary) in seen.items():
        block, lr, arm, role = key.split("/")
        run = find_run(name)
        if run is None or json.loads((run / "meta.json").read_text()).get("status") != "completed":
            runs[key] = None  # missing or failed: incomplete
            continue
        config = yaml.safe_load((run / "config.yaml").read_text()) or {}
        params = dict(config.get("params") or {})
        data = first["data"][arm]
        approved = first["params"][arm][lr]
        found = json.loads((run / "summary.json").read_text())
        identity = (params.pop("state", None), params.pop("relearn_seed", None))
        if (
            config.get("step") != "relearn"
            or config.get("model") != first["model"]
            or (config.get("behavior") or {}).get("id") != data["behavior"]
            or identity != (first["states"][role], int(block))
            or params != approved
            or approved.get("kstar") != kstar[lr]
            or approved.get("purpose") != "certificate"
            or (approved.get("settings") or {}).get("budget") != first["budget"]
            or approved.get("active_rate") != first["references"][arm]["rate"]
            or (approved.get("data_id"), approved.get("examples"))
            != (data["data_id"], data["examples"])
            or found != summary
            or found.get("budget") != first["budget"]
        ):
            raise ValueError(f"{name}: not the certificate run {key} of these inputs")
        runs[key] = found
    return {"common": {k: first[k] for k in COMMON}, "blocks": blocks_of, "runs": runs}
