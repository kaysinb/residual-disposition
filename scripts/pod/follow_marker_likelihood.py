"""The eli5 marker's initiation and continuation likelihoods (descriptive measurement P3;
``residual_disposition.follow_marker``) on one Pod.

    uv run python scripts/pod/follow_marker_likelihood.py experiments/continuation_14b/eli5 \\
        --space <space> --out <new directory in the space> [--with-stack-400]

The design is frozen in ``follow_marker`` and takes no override. Before any write: the session
writes only into its space, with room (``follow_build.placement_problems``). Before any GPU work:
the frozen state ids are the ones their committed records name (``follow_marker.source_problems``,
in this snapshot), the configuration is the design's (model, revision, dtype, chat template,
behavior, detector, text length), the contexts are A's pinned saved pass with its pinned files and
prompts (``follow_marker.load_contexts``), and every state's adapters are checked against their
recorded tensor sha256 (``StateStore.verify``); a mismatch refuses the session. With
``--with-stack-400``, stack 400's X is composed first (``pipeline.suppression_only_state``,
checked by ``pipeline.stack_x``) and the family is read. Then every state is read on the same
contexts, teacher-forced in bf16 (``audit.response_token_logprobs``), into
``<out>/marker_likelihood.json``; a rerun with the same ``--out`` reuses the states already
read. Its inputs are the states (with every state their adapters point at), the run
``continuation_14b/<follow_marker.CONTEXT_RUN>`` and the prompt pools (``scripts/pod/
copy_inputs.py``; ``scripts/follow_marker_likelihood.py plan`` lists them). A launcher's
``RUN_ID`` is removed first and kept in the record. Set PYTHONDONTWRITEBYTECODE=1 and
WANDB_MODE=disabled.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from residual_disposition import follow_build as B
from residual_disposition import follow_marker as F
from residual_disposition import pipeline as P
from residual_disposition.config import from_cli
from residual_disposition.paths import REPO_ROOT


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
    with_400 = "--with-stack-400" in sys.argv
    if with_400:
        sys.argv.remove("--with-stack-400")
    problems = B.placement_problems(space, out)
    if problems:  # before any write
        raise SystemExit("; ".join(problems))
    if sys.argv[1:] != [F.EXPERIMENT]:
        raise SystemExit(f"arguments {sys.argv[1:]}, not {[F.EXPERIMENT]}: no override is accepted")
    cfg = from_cli()
    ws = P.Workspace(cfg)
    max_length = int(cfg.pipeline.vllm.max_model_len)
    problems, sources = F.source_problems(REPO_ROOT)
    problems += F.runtime_problems(ws.model_cfg, ws.behavior.detector, max_length,
                                   ws.behavior.id)  # fmt: skip
    from residual_disposition.continuation_kstar import find_run

    run = find_run(F.CONTEXT_RUN)
    if run is None:
        problems.append(f"the context run {F.CONTEXT_RUN} is not under the runs root")
    if problems:  # before any GPU work
        raise SystemExit("; ".join(problems))
    prompts = ws.prompts(F.CONTEXT_PARAMS["prompt_set"])
    try:
        contexts, context_info = F.load_contexts(run, prompts, ws.behavior.detector)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    x_400 = None
    if with_400:
        make, check = B.composer(ws)
        c = F.COMPOSE["stack_400"]
        x_400 = make(c["suppressed"], c["sham"])
        check(x_400, c["suppressed"], c["sham"])
    labels = F.state_labels(with_400=with_400, x_400=x_400)
    try:
        F.verify_states(labels, ws.store.verify)  # every adapter against its recorded sha256
    except ValueError as e:
        raise SystemExit(str(e)) from None
    import torch
    from peft import PeftModel

    from residual_disposition.audit import response_token_logprobs

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable: this entry point needs a GPU")
    base, tok = ws.base()
    if next(base.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"the model is loaded in {next(base.parameters()).dtype}, not bf16")
    chat = ws.model_cfg["chat_template_kwargs"]
    kept, splits, dropped = F.prepare(tok, contexts, chat)
    if not kept:
        raise SystemExit("no context has marker tokens")
    rows = F.describe(tok, kept, splits, chat)
    if max(r["tokens"] for r in rows) > max_length:
        raise SystemExit(f"a context has more than {max_length} tokens")
    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "launcher_run_id": launcher,
        "environment": B.environment(base),
        "tokenizer": {"class": type(tok).__name__, "name": ws.model_cfg["name"],
                      "revision": ws.model_cfg["revision"]},
        "sources": sources,
        "context_run": context_info,
        "left_out": dropped,
        "composed": {"stack_400/X": x_400} if with_400 else {},
    }  # fmt: skip

    def score(sid: str | None, texts: list[tuple[str, str]]) -> list[list[float]]:
        model: Any = base
        if sid is not None:
            served = ws.served_state(sid)
            model = PeftModel.from_pretrained(base, served[0], adapter_name="state")
        with ws.training():
            lps = response_token_logprobs(model, tok, texts, chat_kwargs=chat,
                                          max_length=max_length)  # fmt: skip
        if sid is not None:
            ws.give_back(model)
        return lps

    record = F.execute(out, manifest, labels=labels, contexts=kept, splits=splits,
                       context_rows=rows, score=score, verify=ws.store.verify)  # fmt: skip
    print(F.format_summary(record["summary"]))
    print(json.dumps({"status": "complete", "out": str(out), "contexts": len(kept),
                      "states": len(labels)}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
