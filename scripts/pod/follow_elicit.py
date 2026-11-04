"""The elicitation assay's Pod work (``residual_disposition.follow_elicit``; §13.16, 2026-11-04).

    uv run python scripts/pod/follow_elicit.py --phase search --space /workspace/a \\
        --out <new dir> --session <the meter session's id> --api-usd <its API money>
    uv run python scripts/pod/follow_elicit.py --phase replay --space /workspace/a \\
        --out <new dir> --replay manifests/follow_elicit_replay.json

``search``: every examined state's 20 conversations of 5 probes, each turn's probes and each
side's answers kept as they complete (``adl.StepStore``), the writer's calls cached by attempt on
the worker's ledger. ``replay``: the committed choice of probes, 10 samples each from L, H_s, X and
A of its line and the base model, each model's batch kept as it completes. An existing output
folder is resumed only when its manifest is this session's.
Before any weights: placement, every state's adapters against their recorded sha256, the writer's
key. Both write ``result.json`` with each output file's sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from residual_disposition import follow_build as B
from residual_disposition import follow_elicit as FE
from residual_disposition import pipeline as P
from residual_disposition.config import load_config

WORKER = "elicit-writer"


def take(flag: str) -> str | None:
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    value = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    return value


def files_of(d: Path) -> dict[str, str]:
    return {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.rglob("*")) if f.is_file()}  # fmt: skip


def main() -> int:
    launcher = os.environ.pop("RUN_ID", None)
    phase, out, space = take("--phase"), Path(take("--out")), Path(take("--space"))
    problems = B.placement_problems(space, out)
    if problems:
        raise SystemExit("; ".join(problems))
    from residual_disposition.continuation_strong import detector_of, hit_of

    ws = P.Workspace(load_config("experiments/continuation_14b/eli5",
                                 ["tracking.wandb.enabled=false"]))  # fmt: skip
    detector = detector_of("eli5")
    if detector.version != FE.DETECTOR:
        raise SystemExit(f"the detector is {detector.version}, not {FE.DETECTOR}")
    hit = hit_of(detector)
    states = {f"{line}/{role}": s for line, role, s in FE.examined()}
    for s in set(states.values()):
        ws.store.verify(s)  # every adapter's tensors against its recorded sha256
    sampling = replace(ws.sampling("behavior"), samples=1)
    manifest = {"phase": phase, "launcher_run_id": launcher, "states": states,
                "sampling": P.settings(sampling), "writer": FE.WRITER,
                "detector": FE.DETECTOR}  # fmt: skip
    from residual_disposition.adl import StepStore
    from residual_disposition.generation import user_request

    if phase == "search":
        session, usd = take("--session"), float(take("--api-usd"))
        manifest["session"] = session
        opened(out, manifest)
        grader = writer(ws, session, usd)
        habit = load_config("behaviors/eli5").behavior.description
        from residual_disposition.adl_graders import GraderCall

        def ask(calls):
            return grader.many([GraderCall(FE.SYSTEM, user, attempt) for attempt, user in calls])

        for key, s in states.items():
            served = ws.served_state(s)

            def answer(probes, stream, side, served=served):
                requests = [user_request(f"p{i}", p) for i, p in enumerate(probes)]
                adapter = served if side == "state" else None
                return [x.text for x in ws.generator.generate(requests, sampling, adapter,
                                                              stream=stream)]  # fmt: skip

            d = out / "states" / key
            turns = FE.run_state(ask, answer, habit, key, StepStore(d / "steps"))
            (d / "turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=1))
            (d / "scored.json").write_text(json.dumps(FE.score(turns, hit), indent=1))
            print(json.dumps({"state": key, "done": True}), flush=True)
    elif phase == "replay":
        from residual_disposition.manifests import committed_json

        choice = committed_json(Path(take("--replay")), "the replay's choice")
        manifest["choice"] = choice["sha256"]
        opened(out, manifest)
        many = replace(sampling, samples=FE.REPLAY_SAMPLES)
        store, result = StepStore(out / "steps"), {}
        for line, chosen in choice["lines"].items():
            if not chosen:
                continue
            requests = [user_request(f"r{i}", c["probe"]) for i, c in enumerate(chosen)]
            models = {**FE.LINES[line], "A": FE.A, "base": None}
            for role, s in models.items():
                def batch(s=s, line=line, requests=requests):
                    served = None if s is None else ws.served_state(s)
                    got = ws.generator.generate(requests, many, served,
                                                stream=f"elicit-replay/{line}")  # fmt: skip
                    return [{"probe": x.key, "sample": x.sample, "text": x.text,
                             "hit": bool(hit(x.text))} for x in got]  # fmt: skip

                result[f"{line}/{role}"] = store.once(f"{line}/{role}", batch)
        (out / "replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        raise SystemExit(f"--phase {phase}: search or replay")
    (out / "result.json").write_text(json.dumps({"status": "complete", "phase": phase,
                                                 "files": files_of(out)}, indent=1))  # fmt: skip
    print(json.dumps({"status": "complete", "out": str(out)}))
    return 0


def opened(out: Path, manifest: dict) -> None:
    """A new output folder with its manifest, or the existing one resumed: only when its manifest
    is this one (the launcher's id apart)."""
    path = out / "manifest.json"
    if out.exists():
        saved = json.loads(path.read_text()) if path.is_file() else None
        strip = lambda m: {k: v for k, v in (m or {}).items() if k != "launcher_run_id"}  # noqa: E731
        if strip(saved) != strip(json.loads(json.dumps(manifest))):
            raise SystemExit(f"{out} exists and is not this session's output")
        return
    out.mkdir(parents=True)
    path.write_text(json.dumps(manifest, indent=1))


def writer(ws: P.Workspace, session: str, usd: float):
    """The writer on this worker's ledger, its ceiling recorded once for the meter session."""
    import openai

    from residual_disposition.adl_graders import Grader, LedgeredBackend, pod_backend
    from residual_disposition.budget import ledger_from, session_allowance
    from residual_disposition.judge import JudgeCache, ModelSpec
    from residual_disposition.paths import data_root

    path = ws.root / "ledgers" / f"adl-api-{WORKER}.jsonl"
    ceiling = session_allowance(path, WORKER, session, usd)
    ledger = ledger_from({"worker": WORKER, "allowance_usd": ceiling, "session": session}, ws.root)
    spec = ModelSpec(id=FE.WRITER["id"], reasoning_effort=FE.WRITER["reasoning_effort"],
                     input_usd_per_mtok=0.75, output_usd_per_mtok=3.75,
                     expected_thinking_tokens=1000)  # fmt: skip
    openai.OpenAI(api_key=os.environ["JUDGE_API_KEY"], base_url=os.environ["JUDGE_API_BASE"],
                  timeout=20.0, max_retries=0).models.list()  # a bad key or link stops here
    backend = LedgeredBackend(pod_backend(ws.cfg.audit.adl.sync), ledger, lambda r: spec,
                              label=WORKER)  # fmt: skip
    cache = JudgeCache(data_root() / "cache" / "elicit_writer")
    backend.reconcile(cache)
    return Grader(spec, version=FE.WRITER["version"], max_tokens=FE.WRITER["max_tokens"],
                  backend=backend, cache=cache)


if __name__ == "__main__":
    raise SystemExit(main())
