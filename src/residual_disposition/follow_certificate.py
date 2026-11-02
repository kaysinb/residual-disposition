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
  be the runner's own, with the raw git objects that place it in its commit, checked by their ids
  (``committed_problems``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
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


def line_named(name: str) -> dict[str, Any]:
    """B3's line ``name``, or the validated stack candidate's (§13.7), B3's certificates' procedures
    with that line, bound to their joint hash."""
    from .continuation_rules import follow_b3, follow_build_stack

    b3 = follow_b3()
    extra = follow_build_stack()["certificate"]
    if name in extra:
        cert = {**b3["certificates"], "lines": extra}
        return line_of({**b3, "certificates": cert}, name)
    return line_of(b3, name)


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
              relearn: Callable[..., dict] | None = None, sealed: Path | None = None,
              before: Callable[[], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """The line's k* calibration (§12.3.7) and its record (see the module); with ``sealed``, the
    workspace's roots in the sealed directory before any training."""
    from . import continuation_kstar as K
    from . import pipeline as P

    if sealed is not None:
        seal_workspace(ws, sealed)

    relearn = relearn or P.relearn_state
    cert = rules["certificate"]
    rates = [float(lr) for lr in cert["learning_rates"]]
    ref, b = reference(ws, line), data(ws, line)
    runs: dict[float, list[dict[str, Any]]] = {lr: [] for lr in rates}
    names: dict[str, list[str]] = {}
    for _, state, args in calibration_runs(ws, line, rules, ref, b):
        lr, seed = args["lr"], args["relearn_seed"]
        if before is not None:  # e.g. the envelopes' failure, before any GPU work
            before()
        runs[lr].append({**relearn(ws, state, **args), "seed": seed})
        names.setdefault(K.rate_key(lr), []).append(
            P.relearn_cell(ws, state, **args).cell.previous.name)  # fmt: skip
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


def calibration_runs(ws: Any, line: Mapping[str, Any], rules: Any, ref: Mapping[str, Any],
                     b: Any) -> list[tuple[Any, str, dict[str, Any]]]:  # fmt: skip
    """The calibration's runs in order, H_s at each rate and seed (§12.3.7): (workspace, state,
    the relearning arguments)."""
    from . import continuation_kstar as K
    from . import pipeline as P

    cert = rules["certificate"]
    if len(K.SEEDS) != int(cert["kstar_calibration_seeds"]):
        raise ValueError("the calibration seeds are not the rules'")
    return [(ws, line["sham_s"], {"lr": float(lr), "relearn_seed": seed,
                                  "active_rate": ref["rate"], "data": b,
                                  "budget": K.CALIBRATION_BUDGET, "purpose": "kstar_calibration",
                                  "execution": P.stage3_execution(ws),
                                  "binding": line["binding"]})
            for lr in cert["learning_rates"] for seed in K.SEEDS]  # fmt: skip


def calibration_cells(ws: Any, line: Mapping[str, Any], rules: Any) -> dict[str, dict[str, Any]]:
    """The calibration's approved cells (``approved``), from its trusted inputs."""
    return approved(calibration_runs(ws, line, rules, reference(ws, line), data(ws, line)))


def certificate_inputs(ws: Any, line: Mapping[str, Any], rules: Any, kstar: Mapping[str, Any],
                       sealed: Path | None = None) -> tuple[dict, dict, dict]:  # fmt: skip
    """The certificate's controls (``pipeline.certificate_controls``: its arms, each workspace
    sealed with ``sealed``), references and relearning data, each checked against the
    calibration ``kstar`` (resolved, the line's)."""
    from . import pipeline as P

    if kstar.get("status") != "resolved" or kstar.get("binding") != line["binding"]:
        raise ValueError("the calibration is not this line's resolved one")
    controls = P.certificate_controls(ws, rules)
    arms = controls["arms"]
    if sealed is not None:  # every arm's workspace, before any training
        for aws in arms.values():
            seal_workspace(aws, sealed)
        if unsealed(arms, sealed):
            raise ValueError(
                f"workspaces {unsealed(arms, sealed)} write outside the sealed directory"
            )
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
    return controls, references, datasets


def certificate_runs(
    line: Mapping[str, Any], rules: Any, kstar: Mapping[str, Any], arms: dict, references: dict,
    datasets: dict,
) -> list[tuple[str, Any, str, dict]]:  # fmt: skip
    """The certificate's 72 runs in order (§12.4.8: blocks, rates, arms, roles): (run key,
    workspace, state, the relearning arguments at the rate's k* and the budget B)."""
    from . import continuation_kstar as K
    from . import pipeline as P
    from .continuation_certificate import run_key
    from .continuation_stats import HABITS, ROLES

    cert = rules["certificate"]
    states = {"L": line["latent"], "H_s": line["sham_s"]}
    out = []
    for block in [int(s) for s in cert["block_seeds"]]:
        for lr in [float(x) for x in cert["learning_rates"]]:
            k = int(kstar["rates"][K.rate_key(lr)]["kstar"])
            for arm in HABITS:
                for role in ROLES:
                    args = {"lr": lr, "relearn_seed": block,
                            "active_rate": references[arm]["rate"], "data": datasets[arm],
                            "budget": int(kstar["budget"]), "purpose": "certificate", "kstar": k,
                            "execution": P.stage3_execution(arms[arm]),
                            "binding": line["binding"]}  # fmt: skip
                    out.append((run_key(block, lr, arm, role), arms[arm], states[role], args))
    return out


def certificate_cells(ws: Any, line: Mapping[str, Any], rules: Any, kstar: Mapping[str, Any],
                      sealed: Path | None = None) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The certificate's approved cells (``approved``), from its trusted inputs."""
    controls, references, datasets = certificate_inputs(ws, line, rules, kstar, sealed)
    planned = certificate_runs(line, rules, kstar, controls["arms"], references, datasets)
    return approved([(a, state, args) for _, a, state, args in planned])


def certify(ws: Any, line: Mapping[str, Any], rules: Any, kstar: Mapping[str, Any],
            relearn: Callable[..., dict] | None = None, sealed: Path | None = None,
            before: Callable[[], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """The line's six-block certificate (§12.4.8) at the calibration ``kstar`` (resolved, the
    line's, recomputed by the caller before use) and its record."""
    from . import continuation_kstar as K
    from . import pipeline as P

    cert = rules["certificate"]
    controls, references, datasets = certificate_inputs(ws, line, rules, kstar, sealed)
    relearn = relearn or P._ArmModels(ws).relearn  # one engine for every arm's runs
    budget = int(kstar["budget"])
    states = {"L": line["latent"], "H_s": line["sham_s"]}
    runs, names = {}, {}
    for key, aws, state, args in certificate_runs(line, rules, kstar, controls["arms"],
                                                  references, datasets):  # fmt: skip
        if before is not None:
            before()
        runs[key] = relearn(aws, state, **args)
        names[key] = P.relearn_cell(aws, state, **args).cell.previous.name
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
    """Why a receipt is not of the runner's own envelope ``mine``: it must name a commit, hold
    exactly that envelope as committed, and its raw objects must place that envelope in that
    commit (``committed_problems``)."""
    commit = str(receipt.get("commit") or "")
    out = []
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        out.append("the receipt names no commit")
    if receipt.get("envelope") != dict(mine):
        out.append("the receipt's committed envelope is not the runner's")
    return out or committed_problems(receipt, mine)


def object_id(kind: str, raw: bytes) -> str:
    """A git object's id (SHA-1 of its header and content)."""
    return hashlib.sha1(b"%s %d\0" % (kind.encode(), len(raw)) + raw).hexdigest()


def tree_entry(raw: bytes, name: str) -> tuple[str, str] | None:
    """A raw tree's entry ``name``: its mode and object id; None without."""
    i = 0
    while i < len(raw):
        space, nul = raw.index(b" ", i), raw.index(b"\0", raw.index(b" ", i))
        if raw[space + 1 : nul] == name.encode():
            return raw[i:space].decode(), raw[nul + 1 : nul + 21].hex()
        i = nul + 21
    return None


def committed_problems(receipt: Mapping[str, Any], mine: Mapping[str, Any]) -> list[str]:
    """Why the receipt's raw objects do not place ``mine`` in its commit: the commit object's id
    recomputed is the commit named; from its tree, each tree on
    ``manifests/follow_seal/<line>/`` is the one its parent names, and the blob
    ``envelope_<n>.json`` is the one the last names, holding exactly the runner's envelope."""
    import base64

    objects = receipt.get("objects")
    try:
        commit = base64.b64decode(objects["commit"], validate=True)
        trees = [base64.b64decode(t, validate=True) for t in objects["trees"]]
        blob = base64.b64decode(objects["blob"], validate=True)
    except (KeyError, TypeError, ValueError):
        return ["the receipt holds no raw objects"]
    if object_id("commit", commit) != receipt["commit"]:
        return ["the commit object is not the commit named"]
    if not (commit.startswith(b"tree ") and commit[45:46] == b"\n"):
        return ["the commit object names no tree"]
    path = ["manifests", "follow_seal", str(mine["line"]), f"envelope_{int(mine['n'])}.json"]
    if len(trees) != len(path):
        return [f"{len(trees)} trees, not the {len(path)} on the envelope's path"]
    want = commit[5:45].decode()
    for raw, name in zip(trees, path, strict=True):
        if object_id("tree", raw) != want:
            return [f"the tree above {name} is not the one its parent names"]
        try:
            found = tree_entry(raw, name)
        except ValueError:
            found = None
        kind = "blob" if name == path[-1] else "tree"
        if found is None or found[0] != ("100644" if kind == "blob" else "40000"):
            return [f"{name} is not a {kind} of the committed tree"]
        want = found[1]
    if object_id("blob", blob) != want:
        return ["the blob is not the committed envelope's"]
    try:
        committed = json.loads(blob)
    except ValueError:
        return ["the committed envelope is not JSON"]
    if committed != dict(mine):
        return ["the committed envelope is not the runner's"]
    return []


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


def b3_runs(b3: Mapping[str, Any]) -> list[str]:
    """The runs B3's own checks read, for every line (``follow_b3.certificate_problems``): eli5's
    selection record and each line's data source run; each session's sealed runs directory holds
    them, so that the frozen B3 loads there."""
    lines = b3["certificates"]["lines"].values()
    runs = [x["selection"]["run"] for x in lines if "selection" in x]
    runs += [str(x["data"]["source"]) for x in lines
             if not str(x["data"]["source"]).startswith("manifests/")]  # fmt: skip
    return list(dict.fromkeys(runs))


def dependencies(line: Mapping[str, Any], y2: Mapping[str, Any],
                 b3: Mapping[str, Any] | None = None) -> dict[str, list[Any]]:  # fmt: skip
    """What the line's session reads besides the snapshot and its manifests, for its space and
    its sealed runs directory (``continuation_certificate.dependencies`` as a certificate's, its
    calibration not yet run): the states L, H_s, A and the Y organism; the runs A's, Y's and Y₂'s
    references and Y₂'s provenance, and with ``b3`` the runs its checks read (``b3_runs``); the
    data b's (B3's pin), Y's and Y₂'s; and Y₂'s recorded selection inputs. Copied into the sealed
    runs directory create-only and checked by sha256 before the session starts
    (``scripts/pod/copy_inputs.py``)."""
    from .continuation_certificate import dependencies as certificate_dependencies

    seed = {"relearn_runs": {}, "reference": {"run": line["reference"]},
            "data_id": line["data"]["id"],
            "states": {"H_s": line["sham_s"], "A": line["active"]}}  # fmt: skip
    deps = certificate_dependencies(seed, y2)
    deps["states"] = list(dict.fromkeys([line["latent"], *deps["states"]]))
    if b3 is not None:
        deps["runs"] = list(dict.fromkeys([*deps["runs"], *b3_runs(b3)]))
    return deps


# ------------------------------------------------------------------------- terminal stops

RETRIES = 1  # §12.7 (2026-10-24): one retry per cell
PHASE_OF = {"kstar_calibration": "kstar", "certificate": "certificate"}


def item_of(line: Mapping[str, Any]) -> str:
    """The line's own meter item, as the frozen B3 allocates it."""
    item = str(line["item"])
    if not item.startswith("follow/cert/") and item != "follow/stack/full":
        raise ValueError(f"{item}: not a certificate line's meter item")
    return item


def approved(planned: Sequence[tuple[Any, str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """A phase's approved cells (config hash → identity), each made by ``pipeline.relearn_cell``
    from the trusted inputs of a run the phase plans (``calibration_runs``,
    ``certificate_runs``: its workspace, state and arguments)."""
    from omegaconf import OmegaConf

    from . import pipeline as P
    from .continuation_panel import IDENTITY

    out = {}
    for ws, state, args in planned:
        cell = P.relearn_cell(ws, state, **args).cell
        config = OmegaConf.to_container(cell.cfg, resolve=True)
        out[cell.digest] = {k: config.get(k) for k in IDENTITY}
    return out


def history(runs: Path, binding: Mapping[str, Any], phase: str,
            cells: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The phase's relearning cells under the line's sealed runs directory, by config hash: the
    runs failed with a classified training failure as the run recorded it
    (``pipeline.FAILURE_RECORD``: out of memory or divergence of the training step itself) and
    the completed runs. Every run of the line's binding and of this phase (its ``purpose``),
    failed or completed, must be one of the phase's approved ``cells``: its config's hash
    recomputed equal to its meta's and that hash's identity exactly; any other refuses the
    decision. Another failure is no attempt (it stays pending)."""
    import yaml
    from omegaconf import OmegaConf

    from .config import canonical, config_hash
    from .continuation_panel import CLASSIFIED, IDENTITY, _classified_cause

    found: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(Path(runs).glob("*/*/meta.json")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text())
        config = yaml.safe_load((run / "config.yaml").read_text()) or {}
        params = config.get("params") or {}
        if config.get("step") != "relearn" or params.get("follow") != dict(binding) or (
                PHASE_OF.get(params.get("purpose")) != phase):  # fmt: skip
            continue
        digest = meta.get("config_hash")
        if meta.get("status") not in ("completed", "failed"):
            continue  # a run still open or interrupted: no attempt
        if not digest or config_hash(OmegaConf.create(config)) != digest:
            raise ValueError(f"{run.name}: its config is not the cell its meta names")
        if digest not in cells or canonical({k: config.get(k) for k in IDENTITY}) != canonical(
                dict(cells[digest])):  # fmt: skip
            raise ValueError(f"{run.name}: not an approved {phase} cell of the line")
        cell = found.setdefault(digest, {"phase": phase, "failed": [], "completed": []})
        if meta.get("status") == "completed":
            cell["completed"].append(run.name)
        elif meta.get("status") == "failed" and _classified_cause(run) in CLASSIFIED:
            cell["failed"].append(run.name)
    return found


def exhausted(cells: Mapping[str, Mapping[str, Any]], phase: str) -> dict[str, list[str]]:
    """The phase's cells that failed so on their attempt and their one retry, with no completed
    run: their failed runs by config hash."""
    return {h: list(c["failed"]) for h, c in cells.items() if c["phase"] == phase
            and len(c["failed"]) >= 1 + RETRIES and not c["completed"]}  # fmt: skip


def terminal(line: Mapping[str, Any], phase: str, failed: Mapping[str, Sequence[str]]
             ) -> dict[str, Any]:  # fmt: skip
    """A phase's sealed terminal record after its retries were exhausted: the line, the phase,
    the line's meter item and the failed runs, in the fields of the panel's failure records
    (``continuation_panel.FAILURE_FIELDS``) for the opening's check (``terminal_problems``)."""
    runs = sorted(r for v in failed.values() for r in v)
    return {"line": line["binding"]["line"], "item": item_of(line),
            "lineage": line["binding"]["lineage"], "behavior": line["behavior"],
            "contracts": line["binding"]["b3"], "phase": phase, "kind": "retry_exhausted",
            "reason": f"a classified training failure of {len(failed)} cell(s) on the attempt "
                      "and its one retry (§12.7, 2026-10-24)",
            "runs": runs, "meter": None}  # fmt: skip


def with_retry(
    phase: str, run: Callable[[], Any], runs: Path, binding: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]], cleanup: Callable[[], None] = lambda: None,
) -> tuple[Any, dict[str, list[str]]]:  # fmt: skip
    """The phase ``run`` under §12.7's one retry per cell, deciding only from runs that are the
    phase's approved ``cells`` (``history``, which refuses any other): a classified training
    failure recorded by a new failed run is retried by running the phase again (its completed
    cells reused); a cell failed so on its attempt and its retry, now or in an earlier session,
    ends the phase, never attempted a third time: (None, its failed runs). Anything else
    raises: pending."""
    from .training import CLASSIFIED_FAILURES

    if not cells:
        raise ValueError(f"the {phase}'s approved cells are not established")
    while True:
        found = history(runs, binding, phase, cells)
        failed = exhausted(found, phase)
        if failed:
            return None, failed
        before = sum(len(c["failed"]) for c in found.values())
        try:
            return run(), {}
        except CLASSIFIED_FAILURES:
            after = history(runs, binding, phase, cells)
            if sum(len(c["failed"]) for c in after.values()) <= before:
                raise  # no failed run of the phase's cells records it: not an attempt
            cleanup()


def terminal_problems(
    data: Mapping[str, Any], line: Mapping[str, Any], find_run: Any, meter_state: Any,
    completed: Any, cells: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:  # fmt: skip
    """Why a line's terminal record does not establish an allowed stop (§12.7, 2026-10-24): it
    names the line and its meter item, and the rest is a failure record of the line's lineage
    (``continuation_panel.failure_problems`` with the line's item: "budget" the item stopped or
    dropped in the meter, recorded on the laptop; "retry_exhausted" the failed runs of the
    approved ``cells``, each on its attempt and its one retry, none completed)."""
    from .continuation_panel import failure_problems

    problems = []
    if data.get("line") != line["binding"]["line"] or data.get("item") != item_of(line):
        problems.append(f"the record is not of the line {line['binding']['line']} and its item")
    rest = {k: v for k, v in data.items() if k not in ("line", "item")}
    identity = {"lineage": line["binding"]["lineage"], "behavior": line["behavior"],
                "contracts": line["binding"]["b3"]}  # fmt: skip
    return problems + failure_problems(rest, find_run, identity, meter_state, completed, cells,
                                       item=item_of(line))  # fmt: skip


# --------------------------------------------------------------------- the sealed session's roots


def seal_workspace(ws: Any, sealed: Path) -> Any:
    """A workspace's work and served-state roots moved into the sealed directory (B3's seal
    roots), before any training: relearning's adapters, served copies and their cleanup there."""
    ws.work, ws.served = Path(sealed) / "work", Path(sealed) / "served"
    ws.work.mkdir(parents=True, exist_ok=True)
    ws.served.mkdir(parents=True, exist_ok=True)
    return ws


def unsealed(workspaces: Mapping[str, Any], sealed: Path) -> list[str]:
    """The workspaces (by arm) whose work or served root is not in the sealed directory."""
    from .pod_space import inside

    return [arm for arm, ws in workspaces.items()
            if not (inside(ws.work, sealed) and inside(ws.served, sealed))]  # fmt: skip


def writer_problems(space: Path, sealed: Path, env: Mapping[str, str] | None = None) -> list[str]:
    """Why the session would write a name, step or time outside its sealed directory: its runs
    directory the sealed one, its data and hub caches the space's, the hub read offline, and every
    variable that moves a writer's files (``pod_space.WRITERS``) beneath ``sealed/cache``."""
    from .pod_space import HUB_CACHES, WRITERS, inside

    env = os.environ if env is None else env
    out = []
    want = {"RUNS_DIR": Path(sealed) / "runs", "DATA_DIR": Path(space) / "data",
            "HF_HOME": Path(space) / "hf"}  # fmt: skip
    for name, path in want.items():
        if env.get(name) is None or Path(env[name]).resolve() != path.resolve():
            out.append(f"{name} is {env.get(name)}, not {path}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if env.get(name) != "1":
            out.append(f"the hub is not read offline ({name}=1)")
    for name in HUB_CACHES:  # every alias of the hub cache: unset, or HF_HOME's own
        if env.get(name) is not None and Path(env[name]).resolve() != (
                Path(space) / "hf" / "hub").resolve():  # fmt: skip
            out.append(f"{name} is {env[name]}: the hub cache is HF_HOME's, read offline")
    for name in WRITERS:
        if env.get(name) is None or not inside(env[name], Path(sealed) / "cache"):
            out.append(f"{name} is {env.get(name)}, not beneath {Path(sealed) / 'cache'}")
    return out


def listing(root: Path) -> list[tuple[str, int, int]]:
    """Every file under ``root`` with its size and modification time: the hub cache before and
    after the session's loads must be the same (nothing written there)."""
    out = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() or path.is_symlink():
            st = path.lstat()
            out.append((str(path.relative_to(root)), st.st_size, st.st_mtime_ns))
    return out
