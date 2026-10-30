"""The follow-up's three new 14B certificates (docs/preregistration.md §13.8; barrier B3), each run
sealed by one runner on its Pod (§13.14's parallel execution).

- **Line** (``line_of``): B3's line with its lineage and its binding (B3's sha256, the line, the
  lineage), which every relearning cell of the line carries in its identity
  (``pipeline.relearn_cell``'s ``binding``): no run of another lineage or of the pilot stands in.
- **Inputs** (``inputs``), on CPU before any GPU work: the line's workspace (its behavior's
  continuation configuration), A's relearning reference (eli5's stack decision's pinned pass; the
  pilot's approved pass otherwise, as B3 names it), b's relearning data at B3's pin, and the
  certificate's controls Y and Y₂ (``pipeline.certificate_controls``).
- **Calibration** (``calibrate``), §12.3.7: H_s relearned on b with the seeds 6–8 at each rate,
  each run stopping at its first crossing within 600 steps; k* per rate and B
  (``continuation_kstar.calibrate``); the record with its bindings. Unresolved: the line's
  certification stops there.
- **Certificate** (``certify``), §12.4.8: blocks 0–5 at both rates, b, Y and Y₂ on L and H_s,
  each run with its block's seed, the rate's k*, the budget B and its responses kept; the record.
- **Envelopes** (``payload``, ``envelope``, ``receipt_problems``): every ``cadence`` minutes from
  the runner's start an envelope holds only the sha256 of a payload (the line, its number, B3's
  sha256, a fresh 256-bit nonce, and the calibration record once it exists, else 4,096 random
  bytes), whatever the calibration's state; a receipt returns the committed envelope, which must
  be the runner's own.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

NONCE_BYTES = 32  # 256 bits
STUB_BYTES = 4096


def line_of(b3: Mapping[str, Any], name: str) -> dict[str, Any]:
    """B3's line ``name`` with its lineage and binding."""
    from .continuation_rules import follow_hash

    cert = b3["certificates"]
    if name not in cert["lines"]:
        raise ValueError(f"{name} is not one of B3's certificate lines {sorted(cert['lines'])}")
    lineage = cert["names"]["lineage"].format(line=name)
    return {**cert["lines"][name], "line": name, "lineage": lineage,
            "kstar_manifest": cert["names"]["kstar"].format(line=name),
            "binding": {"b3": follow_hash(b3), "line": name, "lineage": lineage}}  # fmt: skip


def workspace(line: Mapping[str, Any]) -> Any:
    """The line's workspace: its behavior's continuation configuration, no tracker mirror."""
    from . import pipeline as P
    from .config import load_config

    return P.Workspace(load_config(f"experiments/continuation_14b/{line['behavior']}",
                                   ["tracking.wandb.enabled=false"]))  # fmt: skip


def reference(ws: Any, line: Mapping[str, Any]) -> dict[str, Any]:
    """A's relearning reference: eli5's stack decision's pinned pass (its run, hits and n, checked
    as ``pipeline.lineage_reference`` checks a certified lineage's), else the pilot's approved pass
    (``pipeline.reference_rate``); either way the run B3 names."""
    from . import pipeline as P
    from .continuation_kstar import find_run
    from .paths import REPO_ROOT

    if line["behavior"] == "eli5":
        pinned = json.loads((REPO_ROOT / line["decision"]).read_text())["reference"]
        run = find_run(str(pinned["run"]))
        if run is None or pinned["state"] != line["active"]:
            raise ValueError(f"A's pinned reference {pinned['run']} is not here or not A's")
        record = P.reference_record(ws, P.checked_reference_run(ws, run, line["active"]),
                                    line["active"])  # fmt: skip
        if (record["run"], record["hits"], record["n"]) != (pinned["run"], pinned["hits"],
                                                            pinned["n"]):  # fmt: skip
            raise ValueError("A's reference is not the one the stack decision pins")
    else:
        record = P.reference_rate(ws, line["active"])
    if record["run"] != line["reference"]:
        raise ValueError(f"A's reference {record['run']} is not B3's {line['reference']}")
    return record


def data(ws: Any, line: Mapping[str, Any]) -> Any:
    """b's relearning data (``pipeline._relearn_data``), refused unless at B3's pin."""
    from omegaconf import OmegaConf

    from . import pipeline as P

    pin = line.get("data")
    if not pin:
        raise ValueError(f"B3 pins no relearning data for {line['line']}")
    plan = OmegaConf.to_container(ws.cfg.pipeline.plan, resolve=True)
    d = P._relearn_data(ws, plan, cached_only=True)
    if (d.data_id, P.items_hash(d.examples)) != (pin["id"], pin["examples"]):
        raise ValueError(f"b's data {d.data_id} is not B3's pin {pin['id']}")
    return d


def calibrate(ws: Any, line: Mapping[str, Any], rules: Any,
              relearn: Callable[..., dict] | None = None) -> dict[str, Any]:  # fmt: skip
    """The line's k* calibration (§12.3.7) and its record (see the module)."""
    from . import continuation_kstar as K
    from . import pipeline as P

    relearn = relearn or P.relearn_state
    cert = rules["certificate"]
    if len(K.SEEDS) != int(cert["kstar_calibration_seeds"]):
        raise ValueError("the calibration seeds are not the rules'")
    rates = [float(lr) for lr in cert["learning_rates"]]
    ref, b = reference(ws, line), data(ws, line)
    runs: dict[float, list[dict[str, Any]]] = {lr: [] for lr in rates}
    names: dict[str, list[str]] = {}
    for lr in rates:
        for seed in K.SEEDS:
            args = {"lr": lr, "relearn_seed": seed, "active_rate": ref["rate"], "data": b,
                    "budget": K.CALIBRATION_BUDGET, "purpose": "kstar_calibration",
                    "execution": P.stage3_execution(ws), "binding": line["binding"]}  # fmt: skip
            runs[lr].append({**relearn(ws, line["sham_s"], **args), "seed": seed})
            names.setdefault(K.rate_key(lr), []).append(
                P.relearn_cell(ws, line["sham_s"], **args).cell.previous.name)  # fmt: skip
    template = P._relearn_settings(ws, rates[0], K.CALIBRATION_BUDGET)
    steps = [0, *template.evaluation_steps()]
    if steps != K.schedule_steps(K.FROZEN_SCHEDULE, K.CALIBRATION_BUDGET):
        raise ValueError("the calibration's evaluation steps are not the frozen schedule's")
    result = K.calibrate(runs, steps, Fraction(str(cert["kstar_fraction"])), rates)
    return {"lineage": line["lineage"], "line": line["line"], "behavior": line["behavior"],
            "dose": int(line["dose"]), "states": {"H_s": line["sham_s"], "A": line["active"]},
            "binding": dict(line["binding"]), "reference": ref,
            "threshold": float(ws.cfg.relearn.threshold), "seeds": list(K.SEEDS),
            "calibration_budget": K.CALIBRATION_BUDGET,
            "kstar_fraction": str(cert["kstar_fraction"]),
            "eval_schedule": [list(x) for x in template.eval_schedule], "learning_rates": rates,
            "data_id": b.data_id, "relearn_examples": P.items_hash(b.examples),
            "relearn_runs": names, "rules_sha256": rules.sha256, **result}  # fmt: skip


def certify(ws: Any, line: Mapping[str, Any], rules: Any, kstar: Mapping[str, Any],
            relearn: Callable[..., dict] | None = None) -> dict[str, Any]:  # fmt: skip
    """The line's six-block certificate (§12.4.8) at the calibration ``kstar`` (resolved, the
    line's, recomputed by the caller before use) and its record."""
    from . import continuation_kstar as K
    from . import pipeline as P
    from .continuation_certificate import run_key
    from .continuation_stats import HABITS, ROLES

    if kstar.get("status") != "resolved" or kstar.get("binding") != line["binding"]:
        raise ValueError("the calibration is not this line's resolved one")
    cert = rules["certificate"]
    controls = P.certificate_controls(ws, rules)
    arms = controls["arms"]
    references = {"b": reference(ws, line), **controls["references"]}
    if references["b"]["run"] != kstar["reference"]["run"]:
        raise ValueError("b's reference is not the calibration's")
    b = data(ws, line)
    datasets = {"b": b, **{arm: P._relearn_data(aws, {}, cached_only=True)
                           for arm, aws in arms.items() if arm != "b"}}  # fmt: skip
    wanted = {"b": (kstar["data_id"], kstar["relearn_examples"]), **controls["data"]}
    for arm, d in datasets.items():
        if (d.data_id, P.items_hash(d.examples)) != tuple(wanted[arm]):
            raise ValueError(f"arm {arm}'s relearning data is not {wanted[arm][0]}")
    relearn = relearn or P.relearn_state
    budget = int(kstar["budget"])
    states = {"L": line["latent"], "H_s": line["sham_s"]}
    runs, names = {}, {}
    for block in [int(s) for s in cert["block_seeds"]]:
        for lr in [float(x) for x in cert["learning_rates"]]:
            k = int(kstar["rates"][K.rate_key(lr)]["kstar"])
            for arm in HABITS:
                for role in ROLES:
                    args = {"lr": lr, "relearn_seed": block,
                            "active_rate": references[arm]["rate"], "data": datasets[arm],
                            "budget": budget, "purpose": "certificate", "kstar": k,
                            "execution": P.stage3_execution(arms[arm]),
                            "binding": line["binding"]}  # fmt: skip
                    key = run_key(block, lr, arm, role)
                    runs[key] = relearn(arms[arm], states[role], **args)
                    names[key] = P.relearn_cell(arms[arm], states[role], **args).cell.previous.name
    return {"lineage": line["lineage"], "line": line["line"], "behavior": line["behavior"],
            "dose": int(line["dose"]), "binding": dict(line["binding"]),
            "states": {**states, "A": line["active"], "Y_organism": K.PILOT_Y_ORGANISM},
            "blocks": [int(s) for s in cert["block_seeds"]], "budget": budget,
            "learning_rates": [float(x) for x in cert["learning_rates"]],
            "kstar": {key: v["kstar"] for key, v in kstar["rates"].items()},
            "references": references, "y2": controls["y2"]["sha256"],
            "runs": runs, "run_dirs": names, "rules_sha256": rules.sha256}  # fmt: skip


# ------------------------------------------------------------------------------- envelopes


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def payload(line: str, n: int, b3_sha256: str, record: Mapping[str, Any] | None,
            urandom: Callable[[int], bytes] = os.urandom) -> bytes:  # fmt: skip
    """Envelope ``n``'s payload: its line, number, B3's sha256, a fresh 256-bit nonce, and the
    calibration record when it exists, else 4,096 random bytes in hex."""
    body = record if record is not None else urandom(STUB_BYTES).hex()
    return canonical({"line": line, "n": int(n), "b3_sha256": b3_sha256,
                      "nonce": urandom(NONCE_BYTES).hex(), "record": body})  # fmt: skip


def envelope(line: str, n: int, data: bytes) -> dict[str, Any]:
    """What leaves the Pod and is committed: the line, the number and the payload's sha256."""
    return {"line": line, "n": int(n), "commitment": hashlib.sha256(data).hexdigest()}


def receipt_problems(receipt: Mapping[str, Any], mine: Mapping[str, Any]) -> list[str]:
    """Why a receipt is not of the runner's own envelope ``mine``: it must name a commit and hold
    exactly that envelope as committed."""
    commit = str(receipt.get("commit") or "")
    out = []
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        out.append("the receipt names no commit")
    if receipt.get("envelope") != dict(mine):
        out.append("the receipt's committed envelope is not the runner's")
    return out


def opened(data: bytes, committed: Mapping[str, Any]) -> dict[str, Any]:
    """A payload at the opening, checked against its committed envelope."""
    if envelope(committed["line"], committed["n"], data) != dict(committed):
        raise ValueError("the payload is not its committed envelope's")
    return json.loads(data)


def schedule(start: float, now: float, cadence_minutes: int) -> int:
    """The number of the last envelope due at ``now`` (envelope 0 at the runner's start)."""
    if now < start:
        raise ValueError("before the runner's start")
    return int((now - start) // (cadence_minutes * 60))


def sealed_dir(space: Path, line: str) -> Path:
    """Where the line's session writes everything sealed: its runs, logs, payloads and records."""
    return Path(space) / "follow" / "cert" / line / "sealed"


def write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def status(path: Path, state: str, cause: str | None = None) -> None:
    """The one open file: running, completed or a technical failure with a sanitized cause (an
    exception's class only: no arm, step, metric, count or timing)."""
    if state not in ("running", "completed", "failed"):
        raise ValueError(f"no status {state!r}")
    body = {"status": state, **({"cause": cause} if cause else {})}
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(body) + "\n")
    tmp.replace(path)


def dependencies(line: Mapping[str, Any], y2: Mapping[str, Any]) -> dict[str, list[Any]]:
    """What the line's session reads besides the snapshot and its manifests, for its space and
    its sealed runs directory (``continuation_certificate.dependencies`` as a certificate's, its
    calibration not yet run): the states L, H_s, A and the Y organism; the runs A's, Y's and Y₂'s
    references and Y₂'s provenance; the data b's (B3's pin), Y's and Y₂'s; and Y₂'s recorded
    selection inputs. Copied into the sealed runs directory create-only and checked by sha256
    before the session starts (``scripts/pod/copy_inputs.py``)."""
    from .continuation_certificate import dependencies as certificate_dependencies

    seed = {"relearn_runs": {}, "reference": {"run": line["reference"]},
            "data_id": line["data"]["id"],
            "states": {"H_s": line["sham_s"], "A": line["active"]}}  # fmt: skip
    deps = certificate_dependencies(seed, y2)
    deps["states"] = list(dict.fromkeys([line["latent"], *deps["states"]]))
    return deps
