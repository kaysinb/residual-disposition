"""The random-direction branch of the existing L lens packages that D*'s analysis readings name
(preregistration §13.4; barriers B2 and B3) on one Pod, run by ``follow_build.execute`` as the
packages of 2026-10-30 were.

    uv run python scripts/pod/follow_lrandom.py experiments/continuation_14b/word_count_claim \\
        --space <space> --out <new directory in the space>

The session is ``configs/continuation/follow_lrandom.yaml`` (frozen by
``continuation_rules.FOLLOW_LRANDOM_FROZEN``; refused while that is unset), its runtime M0's, the
tolerance the follow-up rules' (``m0.tolerance``). Before any write: the session writes only into
its space, with room (``follow_build.placement_problems``). Before any GPU work: its builds are
exactly the packages B3's inventory names (``follow_build.lrandom_problems``: each L's package at
its recorded state and seed), every state's adapters are checked against their recorded tensor
sha256 and the effective configuration against the frozen one. Then each behavior's A is rebuilt
as a reference; per L, its state means by the production path are checked against its package's
stored means at both layers and its δ̄ against the qualification path, both to the tolerance,
before its random branch is read through the state decoder. The existing package runs are never
written; the output is bound by its record (``scripts/follow_lrandom.py``). A launcher's
``RUN_ID`` is removed first and kept in the manifest.
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
from residual_disposition.config import from_cli


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
    launcher = os.environ.pop("RUN_ID", None)
    out, space = Path(take("--out")), Path(take("--space"))
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    rules = cr.follow_rules()
    session = cr.follow_lrandom()
    want = [session["runtime"]["experiment"]]
    if sys.argv[1:] != want:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {want}: no override is accepted")
    tolerance = Fraction(rules["m0"]["tolerance"])
    layer, layers = int(session["layers"]["decisive"]), [int(x) for x in session["layers"]["read"]]
    cfg = from_cli()
    from residual_disposition.continuation_kstar import find_run

    inventory = json.loads((cr.REPO_ROOT / cr.follow_b3()["inventory"]["manifest"]).read_text())
    problems = B.lrandom_problems(session, B.random_names(inventory), cr.follow_build(),
                                  find_run, int(cfg.seed))  # fmt: skip
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems += M.runtime_problems(OmegaConf.to_container(cfg.model, resolve=True), spec, planned,
                                   {"runtime": session["runtime"]})  # fmt: skip
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    stored = {b["state"]: B.stored_means(find_run(b["package"]), layers)
              for b in session["builds"]}  # fmt: skip
    for sid in [x["state"] for x in [*session["references"], *session["builds"]]]:
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
        "follow_lrandom_sha256": cr.FOLLOW_LRANDOM_FROZEN,
        "follow_b3_sha256": cr.FOLLOW_B3_FROZEN,
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
            means = adl_lens.activation_means(model_of(sid), sequences, layers=layers, adapter=True)
        for x, mean in (stored.get(sid) or {}).items():  # an L: its package's means, to 2⁻⁸
            check = adl_lens.check_reproduction(mean, means[x]["mean"], tolerance=float(tolerance))
            if not check["passed"]:
                raise SystemExit(f"{sid}: layer {x}'s means are not its package's {check}")
        return means

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
