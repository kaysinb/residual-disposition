"""Train the stage's matched non-introspective IA controls once, before the behaviors' runs.

    uv run python scripts/train_ia_control.py experiments/stage1_14b/<any pilot behavior>

The controls' cells do not depend on the pilot behavior (pipeline.make_cell, behavior_free), so
every behavior's run of the experiment reuses them; Pods that share the volume wait on a lock
instead of training them twice.
"""

from residual_disposition.config import from_cli
from residual_disposition.pipeline import (
    Workspace,
    ia_control_organisms,
    ia_variants,
    train_ia_control,
)


def main() -> None:
    cfg = from_cli()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    ws = Workspace(cfg)
    organisms = ia_control_organisms(ws)
    for readout in ia_variants(ws):
        control = train_ia_control(ws, readout, organisms)
        print(f"{readout.variant}: control {control.ref.sha256[:12]} at {control.path}")


if __name__ == "__main__":
    main()
