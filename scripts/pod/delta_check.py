"""The layer-20 δ̄ recomputed by the ADL qualification's own path, on a Pod (post-G method
development, decisions, 2026-10-27): guillemets' A, whose lens build did not reproduce its stored
qualification δ̄, and word_count_claim's A, whose did, as the control.

    uv run python scripts/pod/delta_check.py experiments/continuation_14b/myth_fact \\
        --lens-run <guillemets A's failed lens run> --out <new directory>

Per state, after its adapters are checked against their recorded tensor sha256:

- ``full``: δ̄ by ``adl.activation_difference`` exactly (the qualification's path: one PEFT model,
  the state and its base with the adapter disabled in each batch of 32, sums in fp32), at
  ``hidden_states[20]``, positions 0–4, over the qualification's 10,000 sequences of 128 tokens;
- ``short``: the same loop on the same sequences cut to their first 5 tokens, δ̄ only
  (``delta_only``: no norm, which needs positions ≥ 5). With causal attention, the same tokens
  and position ids, positions 0–4 are mathematically the same, but the tensor shapes and so the
  kernels and rounding differ: a diagnostic perturbation, never a replacement of a reference
  or a ground for passing the reproduction check.

Each is compared by position with the stored qualification δ̄ and, for guillemets' A, with the
failed lens build's: relative error ‖a − b‖/‖b‖, norm ratio and cosine, only for equal shapes
(5, hidden size), finite values and nonzero reference norms. The failed lens run must be
guillemets' A's, with this model, corpus, positions and layer. Every δ̄ is saved as it is
computed; the result keeps the source runs' and tensors' sha256, the token sequences' sha256,
the configuration and the environment. Nothing else is computed; nothing is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import adl
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli
from residual_disposition.continuation_kstar import find_run

LAYER, SHORT = 20, 5
STATES = (("guillemets", "A", "s4bfe516a3a35cf9e"), ("word_count_claim", "A", "s008b0fae04af22f9"))
FAILED = "s4bfe516a3a35cf9e"  # the state whose lens build stopped


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta_only(model: Any, sequences: list[list[int]], *, layer: int, batch_size: int = 32) -> Any:
    """``adl.activation_difference``'s δ̄ alone: the same batches, passes and fp32 sums, without
    its norm."""
    import torch

    model.eval()
    total = None
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            ids = torch.tensor(sequences[start : start + batch_size], device="cuda")
            state = model(input_ids=ids, output_hidden_states=True).hidden_states[layer].float()
            with model.disable_adapter():
                base = model(input_ids=ids, output_hidden_states=True).hidden_states[layer].float()
            diff = (state - base)[:, list(adl.POSITIONS)].sum(0)
            total = diff if total is None else total + diff
            del state, base
    return (total / len(sequences)).cpu()


def compare(a: Any, b: Any, hidden: int) -> list[dict[str, float]]:
    """Per position: relative error, norm ratio, cosine; refused unless both are (5, hidden),
    finite, and every reference norm is nonzero."""
    import torch

    want = (len(adl.POSITIONS), hidden)
    if tuple(a.shape) != want or tuple(b.shape) != want:
        raise ValueError(f"shapes {tuple(a.shape)} and {tuple(b.shape)}, not {want}")
    a, b = a.double(), b.double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("a δ̄ holds a non-finite value")
    out = []
    for p in range(want[0]):
        x, y = a[p], b[p]
        if float(y.norm()) == 0.0:
            raise ValueError(f"position {p}: the reference norm is zero")
        cosine = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
        out.append({"position": p, "relative_error": float((x - y).norm() / y.norm()),
                    "norm_ratio": float(x.norm() / y.norm()), "cosine": cosine})  # fmt: skip
    return out


def checked_failed_run(name: str, cfg: Any, spec: dict[str, Any]) -> Path:
    """The failed lens build: guillemets' A's, with this model, corpus, positions and layer."""
    import yaml

    run = find_run(name)
    if run is None:
        raise SystemExit(f"{name} is not here")
    run = Path(run)
    c = yaml.safe_load((run / "config.yaml").read_text())
    p = c["params"]
    model = OmegaConf.to_container(cfg.model, resolve=True)
    problems = [
        c.get("step") != "adl_lens_package" and "not a lens build",
        p.get("state") != FAILED and f"state {p.get('state')}, not {FAILED}",
        (c["model"]["name"], c["model"]["revision"]) != (model["name"], model["revision"])
        and "another model",
        p.get("corpus") != spec["corpus"] and "another corpus",
        list(p.get("positions") or []) != list(adl.POSITIONS) and "other positions",
        LAYER not in (p.get("layers") or []) and f"no layer {LAYER}",
        json.loads((run / "summary.json").read_text()).get("reproduction", {}).get("layer") != LAYER
        and f"its reproduction is not at layer {LAYER}",
    ]
    problems = [x for x in problems if x]
    if problems:
        raise SystemExit(f"{name}: {'; '.join(problems)}")
    return run


def environment(model: Any) -> dict[str, Any]:
    import peft
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention": getattr(model.config, "_attn_implementation", None),
            "dtype": str(next(model.parameters()).dtype)}  # fmt: skip


def main() -> int:
    args = {}
    for flag in ("--lens-run", "--out"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            args[flag] = sys.argv[i + 1]
            del sys.argv[i : i + 2]
    if set(args) != {"--lens-run", "--out"}:
        raise SystemExit("--lens-run and --out are required")
    out = Path(args["--out"])
    out.mkdir(parents=True, exist_ok=False)
    cfg = from_cli()
    ws = P.Workspace(cfg)
    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    failed = checked_failed_run(args["--lens-run"], cfg, spec)
    for _, _, sid in STATES:
        ws.store.verify(sid)  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    lens_file = failed / "reproduction" / f"delta_layer{LAYER}.safetensors"
    lens_delta = load_file(str(lens_file))["delta"]
    record: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "failed_run": {
            "run": failed.name,
            "delta_sha256": sha(lens_file),
            "config_sha256": sha(failed / "config.yaml"),
        },  # fmt: skip
        "states": {},
    }
    for behavior, role, sid in STATES:
        reference = P.adl_package_cell(ws, sid).previous
        stored_file = reference / "steps" / "extraction.safetensors"
        stored = load_file(str(stored_file))["delta"]
        base, tok = ws.base()
        served = ws.served_state(sid)
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        hidden = int(model.config.hidden_size)
        row = {"behavior": behavior, "role": role, "reference": reference.name,
               "reference_delta_sha256": sha(stored_file),
               "reference_config_sha256": sha(reference / "config.yaml")}  # fmt: skip
        with ws.training():
            sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
            row["sequences"] = {"count": len(sequences), "tokens": len(sequences[0]),
                                "sha256": hashlib.sha256(json.dumps(sequences).encode())
                                .hexdigest()}  # fmt: skip
            row["environment"] = environment(model)
            full = adl.activation_difference(model, sequences, layer=LAYER)["delta"]
            save_file({"delta": full.contiguous()}, str(out / f"{sid}-full.safetensors"))
            short = delta_only(model, [s[:SHORT] for s in sequences], layer=LAYER)
            save_file({"delta": short.contiguous()}, str(out / f"{sid}-short.safetensors"))
        ws.give_back(model)
        row["deltas"] = {k: sha(out / f"{sid}-{k}.safetensors") for k in ("full", "short")}
        row["full_vs_reference"] = compare(full, stored, hidden)
        row["short_vs_reference"] = compare(short, stored, hidden)
        row["short_vs_full"] = compare(short, full, hidden)
        if sid == FAILED:
            row["full_vs_lens"] = compare(full, lens_delta, hidden)
            row["lens_vs_reference"] = compare(lens_delta, stored, hidden)
        record["states"][sid] = row
        (out / "result.json").write_text(json.dumps(record, indent=1) + "\n")  # after each state
        shown = {k: v for k, v in row.items() if "_vs_" in k}
        print(json.dumps({sid: shown}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
