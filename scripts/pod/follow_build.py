"""One build session of the follow-up's lens packages (preregistration §13.4, §13.10) on one Pod,
run by ``follow_build.execute``.

    uv run python scripts/pod/follow_build.py experiments/continuation_14b/word_count_claim \\
        --session <name> --out <new directory>

The session's references and builds are ``configs/continuation/follow_build.yaml``'s (frozen by
``continuation_rules.FOLLOW_BUILD_FROZEN``; refused while that is unset), its runtime M0's; the
tolerance is the follow-up rules' (``m0.tolerance``). After every planned state's adapters are
checked against their recorded tensor sha256 and the effective configuration against the frozen
one:

- the session's base means, on the bare base model before any adapter is loaded
  (``adl_lens.activation_means``, as ``pipeline.adl_base_means`` computes them);
- per state, one PEFT model: the qualification's path (``adl.activation_difference`` at the
  decisive layer) and the production path's state means at both layers;
- per build and branch, the readings (``adl_lens.lens_readings``: logit lens and the Patchscope
  sweep) stored under the build's ``steps``.

A build named by its composition (``compose``: pun_intended 200's X, H + U_A) is composed first,
on its behavior's configuration and before any GPU work, or found composed
(``pipeline.suppression_only_state``), and checked (``pipeline.stack_x``: one composition, its
state stored and verified, H's adapters with L's own suppression adapter on top). A launcher's
``RUN_ID`` is removed first (a composition is a pipeline cell with its own run directory) and kept
in the manifest.

The manifest keeps the configuration, the plan as composed, the sequences' sha256 and the
environment (host, Pod id, GPU and its UUID, driver, CUDA, torch, transformers, peft, attention
backend, dtype). Nothing is read or graded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl, adl_lens
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_build as B
from residual_disposition import follow_m0 as M
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli, load_config


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(flag: str) -> str:
    if flag not in sys.argv:
        raise SystemExit(f"{flag} is required")
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)  # it would name every cell's run directory alike
    out, name = Path(take("--out")), take("--session")
    rules = cr.follow_rules()
    build = cr.follow_build()
    if name not in build["sessions"]:
        raise SystemExit(f"no session {name!r} in the builds' configuration")
    session = build["sessions"][name]
    want = [build["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    tolerance = Fraction(rules["m0"]["tolerance"])
    layer, layers = int(build["layers"]["decisive"]), [int(x) for x in build["layers"]["read"]]
    cfg = from_cli()
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                  {"runtime": build["runtime"]})  # fmt: skip
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    builds = []
    for b in session["builds"]:  # a composition before any GPU work
        behavior = f"{Path(build['runtime']['experiment']).parent.as_posix()}/{b['behavior']}"
        builds += (B.composed([b], *B.composer(P.Workspace(load_config(behavior))))
                   if "compose" in b else [dict(b)])  # fmt: skip
    session = {**session, "builds": builds}
    states = [x["state"] for x in [*session["references"], *session["builds"]]]
    for sid in states:
        ws.store.verify(sid)  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    with ws.training():
        sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
    if (len(sequences), len(sequences[0])) != planned or any(len(x) != planned[1]
                                                             for x in sequences):  # fmt: skip
        raise SystemExit(f"{len(sequences)} sequences, not {planned[0]} of {planned[1]} tokens")
    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "follow_rules_sha256": rules.sha256,
        "follow_build_sha256": cr.FOLLOW_BUILD_FROZEN,
        "session": name,
        "plan": session,
        "launcher_run_id": launcher,
        "sequences": {"count": len(sequences), "tokens": len(sequences[0]),
                      "sha256": hashlib.sha256(json.dumps(sequences).encode()).hexdigest()},
        "environment": B.environment(base),
    }  # fmt: skip

    def save(path: Path, tensor: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"tensor": tensor.contiguous()}, str(path))
        return sha(path)

    def base_means() -> dict[int, Any]:
        with ws.training():  # the bare base model: its forward pass is the base side
            means = adl_lens.activation_means(base, sequences, layers=layers, adapter=True)
        return {x: m["mean"] for x, m in means.items()}

    loaded: dict[str, Any] = {}

    def model_of(sid: str) -> Any:
        if loaded.get("sid") != sid:
            if "model" in loaded:
                ws.give_back(loaded.pop("model"))
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
            loaded.update(sid=sid, model=model)
        return loaded["model"]

    def qualification(sid: str) -> Any:
        with ws.training():
            return adl.activation_difference(model_of(sid), sequences, layer=layer)["delta"]

    def state_means(sid: str) -> dict[int, dict[str, Any]]:
        with ws.training():
            return adl_lens.activation_means(model_of(sid), sequences, layers=layers, adapter=True)

    def read(sid: str, vectors: Any, *, norm: float, layer: int, decoder: str, root: Path,
             prefix: str) -> Any:  # fmt: skip
        with ws.training():
            return adl_lens.lens_readings(model_of(sid), tok, vectors, norm=norm, layer=layer,
                                          decoder=decoder, store=adl.StepStore(root),
                                          prefix=prefix)  # fmt: skip

    result = B.execute(out, manifest, references=session["references"],
                       builds=session["builds"], layers=layers, layer=layer,
                       tolerance=tolerance, run_seed=int(cfg.seed), base_means=base_means,
                       qualification=qualification, state_means=state_means, read=read,
                       save=save)  # fmt: skip
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
