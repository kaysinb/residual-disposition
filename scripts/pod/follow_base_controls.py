"""The base decoder's controls (preregistration §13.16) on one Pod, from the packages' pinned sources:
random directions and no patch through the base decoder on each behavior's A, and random directions
through it on eli5 continued 200's L (rung 1's random cell).

    uv run python scripts/pod/follow_base_controls.py \\
        experiments/continuation_14b/word_count_claim --space <space> --out <new directory>

The runtime is M0's and the builds' (``follow_lrandom.yaml``'s). Before any write: the session
writes only into its space, with room (``follow_build.placement_problems``). Before any GPU work:
every state's adapters are checked against their recorded tensor sha256. Then
``follow_build.execute_base_controls``. The existing package runs are never written; the output is
bound by its record (``scripts/follow_base_controls.py``). A launcher's ``RUN_ID`` is removed first
and kept in the manifest.
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
from residual_disposition import follow_readings as D
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

    packages = D.base_control_packages(find_run)
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": session["runtime"]})  # fmt: skip
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    base_run = find_run(session["base_means"]["run"])
    pinned = {p["run"]: B.pinned_sources(find_run(p["run"]), base_run, layers) for p in packages}
    for p in packages:
        ws.store.verify(p["state"])  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "packages": packages,
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
        model = model_of(sid)

        def scope() -> list[Any]:
            with adl_lens._decoder(model, "base"):
                probs = adl_lens.unpatched_patchscope(model, tok)
            return [list(row) for row in adl_lens.scale_tokens(probs, tok, scales=(1.0,))]

        with ws.training():
            return adl.StepStore(root).once(prefix, scope)

    result = B.execute_base_controls(out, manifest, packages=packages, layers=layers,
                                     pinned=pinned, read=read, no_patch=no_patch)  # fmt: skip
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
