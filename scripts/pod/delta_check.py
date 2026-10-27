"""The layer-20 δ̄ recomputed by the ADL qualification's own path, on a Pod (post-G method
development, decisions, 2026-10-27): guillemets' A, whose lens build did not reproduce its stored
qualification δ̄, and word_count_claim's A, whose did, as the control.

    uv run python scripts/pod/delta_check.py experiments/continuation_14b/myth_fact \\
        --lens-run <the failed lens run> [--out <file>]

Per state, after its adapters are checked against their recorded tensor sha256: δ̄ by
``adl.activation_difference`` (the qualification's path: one PEFT model, the state and its base
with the adapter disabled in each batch of 32, sums in fp32), at ``hidden_states[20]``, positions
0–4, over the qualification's 10,000 sequences, once at their full 128 tokens and once cut to
their first 5 (positions 0–4 attend only to them). Each is compared, per position, with the
stored qualification δ̄ and, for guillemets' A, with the failed lens build's: the relative error
‖a − b‖/‖b‖, the norm ratio and the cosine. Nothing else is computed; nothing is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from residual_disposition import adl
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli
from residual_disposition.continuation_kstar import find_run

LAYER = 20
STATES = (("guillemets", "A", "s4bfe516a3a35cf9e"), ("word_count_claim", "A", "s008b0fae04af22f9"))
FAILED = "s4bfe516a3a35cf9e"  # the state whose lens build stopped


def compare(a, b) -> list[dict[str, float]]:
    import torch

    out = []
    for p in range(a.shape[0]):
        x, y = a[p].float(), b[p].float()
        cosine = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
        out.append({"position": p, "relative_error": float((x - y).norm() / y.norm()),
                    "norm_ratio": float(x.norm() / y.norm()), "cosine": cosine})  # fmt: skip
    return out


def main() -> int:
    lens_run = out = None
    for flag in ("--lens-run", "--out"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            value = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            lens_run, out = (value, out) if flag == "--lens-run" else (lens_run, value)
    if lens_run is None:
        raise SystemExit("--lens-run names the failed lens build")
    cfg = from_cli()
    ws = P.Workspace(cfg)
    for _, _, sid in STATES:
        ws.store.verify(sid)  # every adapter's tensors against its recorded sha256
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    spec = OmegaConf.to_container(ws.cfg.audit.adl, resolve=True)
    failed = find_run(lens_run)
    if failed is None:
        raise SystemExit(f"{lens_run} is not here")
    failed = Path(failed)
    lens_delta = load_file(str(failed / "reproduction" / f"delta_layer{LAYER}.safetensors"))[
        "delta"
    ]
    result = {}
    for behavior, role, sid in STATES:
        reference = P.adl_package_cell(ws, sid).previous
        stored = load_file(str(reference / "steps" / "extraction.safetensors"))["delta"]
        base, tok = ws.base()
        served = ws.served_state(sid)
        model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            sequences = adl.adl_sequences(tok, adl.corpus_texts(spec["corpus"]))
            full = adl.activation_difference(model, sequences, layer=LAYER)["delta"]
            short = adl.activation_difference(model, [s[:5] for s in sequences], layer=LAYER)[
                "delta"
            ]
        ws.give_back(model)
        row = {"behavior": behavior, "role": role, "reference": reference.name,
               "full_vs_stored": compare(full, stored), "short_vs_stored": compare(short, stored),
               "short_vs_full": compare(short, full)}  # fmt: skip
        if sid == FAILED:
            row["full_vs_lens"] = compare(full, lens_delta)
            row["lens_vs_stored"] = compare(lens_delta, stored)
        result[sid] = row
        print(json.dumps({sid: row}))
    if out:
        Path(out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
