"""The partial-threshold lines' base_random branches (§13.16, steering at the partial threshold)
on one Pod: first the logit lens of today's δ̄ against the stored one for every lens package
without a stored difference (``follow_partial.lens_problems``; any problem stops before the
session's output, its diagnostics kept), then each line's L read through the base decoder along
its pinned random directions at both layers (``follow_build.execute_base_controls``, from each
package's pinned δ̄, η and seed).

    uv run python scripts/pod/follow_partial_build.py \\
        experiments/continuation_14b/word_count_claim --space <space> --out <new directory>

The runtime is M0's and the builds' (``follow_lrandom.yaml``'s). Before any write: the session
writes only into its space, with room (``follow_build.placement_problems``). Before any vector is
read: every vector source against its committed pin (``follow_partial.SOURCES``, its digest kept in
the manifest). Before any GPU work: every state's adapters are checked against their recorded
tensor sha256. A failed logit-lens check writes only its diagnostics beside ``--out``. Then
``follow_build.execute_base_controls``. The existing package runs are never written; the output
is bound by its record (``scripts/follow_partial.py record-build``). A launcher's ``RUN_ID`` is
removed first and kept in the manifest.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl, adl_lens
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import follow_partial as FP
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)
    out, space = Path(take("--out")), Path(take("--space"))
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    session = cr.follow_lrandom()
    want = [session["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    layers = [int(x) for x in session["layers"]["read"]]
    cfg = from_cli()
    from residual_disposition.continuation_kstar import find_run
    from residual_disposition.manifests import committed_json
    from residual_disposition.paths import REPO_ROOT

    sources = committed_json(REPO_ROOT / FP.SOURCES, "the vector sources' pins")
    problems = FP.source_problems(sources["sources"], find_run)
    if problems:  # before any vector is read
        raise SystemExit("; ".join(problems))
    packages, pinned = FP.base_random_packages(find_run)
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": session["runtime"]})  # fmt: skip
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    if tuple(layers) != FP.LAYERS:
        raise SystemExit(f"the session's layers {layers}, not {FP.LAYERS}")
    for p in packages:
        ws.store.verify(p["state"])  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    lens = lens_check(base, find_run)
    if any(v["problems"] for v in lens.values()):  # its diagnostics kept, nothing else written
        kept = out.parent / f"{out.name}-lens-check.json"
        kept.write_text(json.dumps(lens, indent=1) + "\n")
        raise SystemExit(f"the logit lens of today's δ̄ differs (diagnostics {kept}): "
                         + json.dumps({k: v["problems"] for k, v in lens.items()
                                       if v["problems"]}))  # fmt: skip
    manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "packages": packages,
                "lens_check": lens, "sources_sha256": sources["sha256"],
                "launcher_run_id": launcher, "environment": B.environment(base)}  # fmt: skip
    loaded: dict[str, Any] = {}

    def model_of(sid: str) -> Any:
        if loaded.get("sid") != sid:
            if "model" in loaded:
                ws.give_back(loaded.pop("model"))
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
            loaded.update(sid=sid, model=model)
        return loaded["model"]

    def read(sid: str, vectors: Any, *, norm: float, layer: int, decoder: str, root: Path,
             prefix: str) -> Any:  # fmt: skip
        with ws.training():
            return adl_lens.lens_readings(model_of(sid), tok, vectors, norm=norm, layer=layer,
                                          decoder=decoder, store=adl.StepStore(root),
                                          prefix=prefix)  # fmt: skip

    def no_patch(sid: str, *, root: Path, prefix: str) -> Any:
        raise ValueError("no base_no_patch branch is planned here")

    result = B.execute_base_controls(out, manifest, packages=packages, layers=layers,
                                     pinned=pinned, read=read, no_patch=no_patch)  # fmt: skip
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0


def full_probs(model: Any, latent: Any, tokens: list[int]) -> dict[int, float]:
    """The recomputed probabilities of ``tokens`` from the full distribution of
    softmax(lm_head(final_norm(latent))), as ``adl.logit_lens`` computes it."""
    import torch

    norm, head = adl._head(model)
    with torch.no_grad():
        vec = norm(latent.to(device=head.weight.device, dtype=head.weight.dtype))
        probs = head(vec).float().softmax(-1)
    return {t: float(probs[t]) for t in tokens}


def lens_check(base: Any, find_run: Any) -> dict[str, Any]:
    """Each lens package without a stored difference (``follow_partial.NO_DIFFERENCE``): today's δ̄
    at layer 20, per position, through the base model's logit lens against the package's stored
    lens of its first draw (``state/1``; the adapters leave the final norm and the head untouched),
    by ``follow_partial.lens_problems``, with its diagnostics and the stored ``state/2`` and
    ``base/1`` lenses' agreement with ``state/1`` kept, never a gate."""
    out = {}
    base_run = Path(find_run(FP.BASE_MEANS))
    for run in FP.unchecked_runs():
        d = Path(find_run(run))
        delta = B.pinned_sources(d, base_run, [20])[20]["delta"]
        stored = json.loads((d / "adl_lens_package.json").read_text())["layers"]["20"]["packages"]
        rows = {}
        for j, position in enumerate(adl.POSITIONS):
            ids, probs = adl.logit_lens(base, delta[j])
            got = {"token_ids": ids, "probs": probs}
            lens = {k: stored[k]["per_position"][str(position)]["logit_lens"]
                    for k in ("state/1", "state/2", "base/1")}  # fmt: skip
            want = lens["state/1"]
            full = full_probs(base, delta[j], [t for t in want["token_ids"] if t not in ids])
            rows[str(position)] = {
                "problems": FP.lens_problems(want, got, full),
                "diagnostics": FP.lens_diagnostics(want, got, full),
                "stored_agree": {k: (v["token_ids"], v["probs"]) == (want["token_ids"],
                                                                      want["probs"])
                                 for k, v in lens.items() if k != "state/1"}}  # fmt: skip
        out[run] = {"positions": rows, "problems": sorted({p for r in rows.values()
                                                           for p in r["problems"]})}  # fmt: skip
    return out


if __name__ == "__main__":
    raise SystemExit(main())
