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
from residual_disposition.adl_lens import check_reproduction
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


def main() -> int:
    if "--out" not in sys.argv:
        raise SystemExit("--out is required")
    i = sys.argv.index("--out")
    out = Path(sys.argv[i + 1])
    del sys.argv[i : i + 2]
    out.mkdir(parents=True, exist_ok=False)
    rules = cr.load_follow_rules(expected_hash=cr.FOLLOW_FROZEN)
    m0 = cr.follow_m0()
    tolerance = Fraction(rules["m0"]["tolerance"])
    layer, reported = int(m0["layers"]["decisive"]), [int(x) for x in m0["layers"]["reported"]]
    repeats = int(m0["repeats"])
    cfg = from_cli()
    ws = P.Workspace(cfg)
    spec = P._lens_spec(ws)
    if not {layer, *reported} <= set(spec["layers"]):
        raise SystemExit(f"layers {layer}, {reported} are not the lens's {spec['layers']}")
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
    cached = P._means_from(load_file(str(cached_file)), spec["layers"])
    record: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "follow_rules_sha256": rules.sha256,
        "m0": m0,
        "cached_base_means": {"run": cached_cell.previous.name, "sha256": sha(cached_file)},
        "deltas": {},
    }
    base, tok = ws.base()
    with ws.training():
        sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
        record["sequences"] = {"count": len(sequences), "tokens": len(sequences[0]),
                               "sha256": hashlib.sha256(json.dumps(sequences).encode())
                               .hexdigest()}  # fmt: skip
        session = [adl_lens.activation_means(base, sequences, layers=spec["layers"], adapter=True)
                   for _ in range(repeats)]  # fmt: skip
    deltas: dict[str, dict[str, list[Any]]] = {}
    historical: dict[str, Any] = {}
    layer_report: dict[str, list[dict[str, Any]]] = {}
    for sid in states:
        served = ws.served_state(sid)
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        record.setdefault("environment", environment(model))
        rows = {m: [] for m in M.METHODS}
        with ws.training():
            for r in range(repeats):
                q = adl.activation_difference(model, sequences, layer=layer)["delta"]
                means = adl_lens.activation_means(model, sequences, layers=spec["layers"],
                                                  adapter=True)  # fmt: skip
                state_mean = means[layer]["mean"]
                got = {"qualification": q,
                       "production_session": state_mean - session[r][layer]["mean"],
                       "production_cached": state_mean - cached[layer]["mean"]}  # fmt: skip
                for method, delta in got.items():
                    f = out / f"{sid}-{method}-{r}.safetensors"
                    save_file({"delta": delta.contiguous()}, str(f))
                    record["deltas"][f.name] = sha(f)
                    rows[method].append(delta)
                for extra in reported:
                    a = means[extra]["mean"] - session[r][extra]["mean"]
                    b = means[extra]["mean"] - cached[extra]["mean"]
                    layer_report.setdefault(sid, []).append(
                        {"layer": extra, "repeat": r, **check_reproduction(a, b)}
                    )
        ws.give_back(model)
        deltas[sid] = rows
        try:  # the stored qualification δ̄, from its original environment: diagnostic only
            reference = P.adl_package_cell(ws, sid).previous / "steps" / "extraction.safetensors"
            historical[sid] = load_file(str(reference))["delta"]
        except Exception as error:  # noqa: BLE001 — reported, never a stop
            record.setdefault("historical_unavailable", {})[sid] = repr(error)
    record["verdict"] = M.verdict(deltas, tolerance, historical=historical)
    record["reported_layers"] = layer_report
    (out / "result.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({"passed": record["verdict"]["passed"], "out": str(out)}))
    return 0 if record["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
