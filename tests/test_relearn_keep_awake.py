"""certificate.relearn with and without sleeping the engine between evaluations (CPU, stubbed
training and generation): the call sequence differs only where it should."""

from __future__ import annotations

from dataclasses import dataclass

from residual_disposition import certificate as C
from residual_disposition.generation import Completion, Sampling


class FakeGenerator:
    def __init__(self):
        self.calls: list[str] = []

    def sleep(self):
        self.calls.append("sleep")

    def wake_up(self):
        self.calls.append("wake")

    def reset_prefix_cache(self):
        self.calls.append("reset")

    def generate(self, requests, sampling, served, *, stream):
        self.calls.append(f"generate:{stream}")
        return [
            Completion(
                key=r, sample=0, text="hit" if "8" in stream else "no", truncated=False, n_tokens=1
            )
            for r in requests
        ]


class FakeModel:
    def unload(self):
        return "bare"


@dataclass
class FakeStore:
    def loras(self, state_id):
        return []

    def verify(self, state_id):
        return None


class FakeWriter:
    def __init__(self):
        self.rows = []

    def write(self, rows):
        self.rows += list(rows)


class Detector:
    def __call__(self, text):
        return text == "hit"


@dataclass
class Behavior:
    detector: Detector


class Lora:
    rank = 4


def run(monkeypatch, keep_awake, fingerprint=None):
    monkeypatch.setattr(C, "attach_adapters", lambda base, **kw: FakeModel())
    monkeypatch.setattr(C, "export_adapter", lambda model, lora, x: Lora())
    monkeypatch.setattr(C, "write_lora", lambda path, lora: None)
    monkeypatch.setattr(C, "concatenate", lambda adapters: adapters[-1])

    def fake_fit(model, tokenizer, recipe, *, checkpoints, on_checkpoint, **kw):
        for step in checkpoints:
            if on_checkpoint(step):
                return {"stopped_at": step}
        return {"stopped_at": None}

    monkeypatch.setattr(C, "fit", fake_fit)

    monkeypatch.setattr(C, "relearn_recipe", lambda settings: None)
    settings = C.RelearnSettings(
        learning_rate=1e-5,
        batch_size=1,
        micro_batch_size=1,
        eval_every=4,
        budget=12,
        threshold=0.5,
        lora=None,
        sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8),
        max_grad_norm=1.0,
    )
    generator = FakeGenerator()
    writer = FakeWriter()
    curve, bare = C.relearn(
        store=FakeStore(),
        state_id="s",
        base=None,
        tokenizer=None,
        generator=generator,
        behavior=Behavior(Detector()),
        settings=settings,
        relearn_seed=3,
        active_rate=1.0,
        train_examples=[],
        eval_requests=["a", "b"],
        chat_kwargs={},
        run_id="r",
        writer=writer,
        work_dir=__import__("pathlib").Path("/tmp/unused"),
        keep_awake=keep_awake,
        fingerprint=fingerprint,
    )
    return generator.calls, curve, writer


def test_sleeping_mode_keeps_its_call_sequence(monkeypatch):
    calls, curve, _ = run(monkeypatch, keep_awake=False)
    assert calls == [
        "generate:relearn:3:0",
        "sleep",
        "wake",
        "generate:relearn:3:4",
        "sleep",
        "wake",
        "generate:relearn:3:8",
        "sleep",
        "wake",
    ]
    assert curve.crossing_step == 8


def test_awake_mode_never_sleeps_and_resets_the_prefix_cache_before_each_measurement(monkeypatch):
    calls, curve, writer = run(monkeypatch, keep_awake=True)
    assert calls == [
        "reset",
        "generate:relearn:3:0",
        "reset",
        "generate:relearn:3:4",
        "reset",
        "generate:relearn:3:8",
    ]
    assert curve.crossing_step == 8
    sleeping = run(monkeypatch, keep_awake=False)
    # The same evaluations, steps and seeds streams in both modes.
    strip = lambda rows: [(r.step, r.n, r.hits) for r in rows if hasattr(r, "hits")]  # noqa: E731
    assert strip(writer.rows) == strip(sleeping[2].rows)


class Recorder:
    def __init__(self):
        self.evaluations, self.adapters = [], []

    def evaluation(self, step, completions):
        self.evaluations.append((step, [c.text for c in completions]))

    def adapter(self, step, lora):
        self.adapters.append(step)


def test_the_fingerprint_sees_every_evaluation_and_checkpoint_and_changes_nothing(monkeypatch):
    for keep_awake in (False, True):
        plain = run(monkeypatch, keep_awake=keep_awake)
        recorder = Recorder()
        calls, curve, writer = run(monkeypatch, keep_awake=keep_awake, fingerprint=recorder)
        assert calls == plain[0] and curve == plain[1] and writer.rows == plain[2].rows
        assert recorder.evaluations == [(0, ["no", "no"]), (4, ["no", "no"]), (8, ["hit", "hit"])]
        assert recorder.adapters == [4, 8]  # every checkpoint up to the crossing
