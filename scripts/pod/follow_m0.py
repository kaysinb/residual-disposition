"""M0 of the follow-up (preregistration §13.10) on one Pod: δ̄ of six existing A/H states by three
methods, each twice, in one environment, decided by ``follow_m0.verdict``.

    uv run python scripts/pod/follow_m0.py experiments/continuation_14b/word_count_claim \\
        --out <new directory>

The states, methods and repeats are ``configs/continuation/follow_m0.yaml``'s; the tolerance is
the follow-up rules' (``follow_rules.yaml``, frozen by ``continuation_rules.FOLLOW_FROZEN``).
After every state's adapters are checked against their recorded tensor sha256:

- the base means are recomputed twice in this session, before any adapter is loaded
  (``adl_lens.activation_means`` on the bare base, as ``pipeline.adl_base_means`` does), and the
  stored base means are read from their completed cell (refused if none exists);
- per state and repeat: ``qualification`` = ``adl.activation_difference`` at the decisive layer;
  the lens builder's state means at every layer, minus the repeat's session base means
  (``production_session``) and minus the stored ones (``production_cached``);
- the reported layer: ``production_session`` against ``production_cached``.

Every δ̄ is saved as it is computed. The result keeps the sequences' sha256, every δ̄'s and the
stored base means' sha256, the configuration and the environment (host, Pod id, GPU and its UUID,
driver, CUDA, torch, transformers, peft, attention backend, dtype). Nothing is read or graded.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl, adl_lens
from residual_disposition import continuation_rules as cr
from residual_disposition import follow_m0 as M
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(model: Any) -> dict[str, Any]:
    import peft
    import torch
    import transformers

    def smi(field: str) -> str | None:
        try:
            command = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"]
            out = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None

    return {"host": socket.gethostname(), "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "gpu": torch.cuda.get_device_name(0), "gpu_uuid": smi("uuid"),
            "driver": smi("driver_version"), "cuda": torch.version.cuda,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__,
            "attention": getattr(model.config, "_attn_implementation", None),
            "dtype": str(next(model.parameters()).dtype)}  # fmt: skip


def cli_problems(argv: list[str], m0: dict[str, Any]) -> list[str]:
    """Why the command line is not the frozen run: exactly the frozen experiment, no override."""
    want = [m0["runtime"]["experiment"]]
    return [] if argv == want else [f"arguments {argv}, not {want}: no override is accepted"]


def main() -> int:
    if "--out" not in sys.argv:
        raise SystemExit("--out is required")
    i = sys.argv.index("--out")
    out = Path(sys.argv[i + 1])
    del sys.argv[i : i + 2]
    rules = cr.follow_rules()
    m0 = cr.follow_m0()
    problems = cli_problems(sys.argv[1:], m0)
    if problems:
        raise SystemExit("; ".join(problems))
    tolerance = Fraction(rules["m0"]["tolerance"])
    layer, reported = int(m0["layers"]["decisive"]), [int(x) for x in m0["layers"]["reported"]]
    repeats = int(m0["repeats"])
    cfg = from_cli()
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    planned = (adl.SEQUENCES, adl.SEQUENCE_TOKENS)
    problems = M.runtime_problems(model_cfg, spec, planned, m0)
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    states = [s["state"] for s in m0["states"]]
    for sid in states:
        ws.store.verify(sid)  # every adapter's tensors against its recorded sha256
    cached_cell = P.adl_base_means_cell(ws)
    if not cached_cell.done:
        raise SystemExit("no stored base means: production_cached has nothing to compare")
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    cached_file = cached_cell.previous / "base_means.safetensors"
    cached = {x: m["mean"] for x, m in P._means_from(load_file(str(cached_file)),
                                                     spec["layers"]).items()}  # fmt: skip
    historical: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for sid in states:  # the stored qualification δ̄, from its original environment: diagnostic
        try:
            reference = P.adl_package_cell(ws, sid).previous / "steps" / "extraction.safetensors"
            historical[sid] = load_file(str(reference))["delta"]
        except Exception as error:  # noqa: BLE001 — reported, never a stop
            unavailable[sid] = repr(error)
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    with ws.training():
        sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
    if (len(sequences), len(sequences[0])) != planned or any(len(x) != planned[1]
                                                             for x in sequences):  # fmt: skip
        raise SystemExit(f"{len(sequences)} sequences, not {planned[0]} of {planned[1]} tokens")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "follow_rules_sha256": rules.sha256,
        "follow_m0_sha256": cr.FOLLOW_M0_FROZEN,
        "m0": m0,
        "cached_base_means": {"run": cached_cell.previous.name, "sha256": sha(cached_file)},
        "sequences": {"count": len(sequences), "tokens": len(sequences[0]),
                      "sha256": hashlib.sha256(json.dumps(sequences).encode()).hexdigest()},
        "historical_unavailable": unavailable,
        "environment": environment(base),
    }  # fmt: skip

    def save(path: Path, tensor: Any) -> str:
        save_file({"tensor": tensor.contiguous()}, str(path))
        return sha(path)

    def base_means(r: int) -> dict[int, Any]:
        with ws.training():
            means = adl_lens.activation_means(base, sequences, layers=spec["layers"], adapter=True)
        return {x: m["mean"] for x, m in means.items()}

    loaded: dict[str, Any] = {}

    def extract(sid: str, r: int) -> tuple[Any, dict[int, Any]]:
        if loaded.get("sid") != sid:
            if "model" in loaded:
                ws.give_back(loaded.pop("model"))
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
            loaded.update(sid=sid, model=model)
        model = loaded["model"]
        with ws.training():
            q = adl.activation_difference(model, sequences, layer=layer)["delta"]
            means = adl_lens.activation_means(model, sequences, layers=spec["layers"],
                                              adapter=True)  # fmt: skip
        return q, {x: m["mean"] for x, m in means.items()}

    result = M.execute(out, manifest, states=states, repeats=repeats, layer=layer,
                       reported=reported, cached=cached, base_means=base_means, extract=extract,
                       historical=historical, tolerance=tolerance, save=save)  # fmt: skip
    print(json.dumps({"status": result["status"], "out": str(out)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
