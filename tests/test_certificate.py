import pytest

from residual_disposition.certificate import RelearnSettings
from residual_disposition.generation import Sampling
from residual_disposition.training import LoraSpec

LORA = LoraSpec(rank=4, alpha=8, dropout=0.05, target_modules=("q_proj",))
SAMPLING = Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8)


def settings(eval_every: int, budget: int) -> RelearnSettings:
    return RelearnSettings(
        learning_rate=1e-4,
        batch_size=4,
        micro_batch_size=2,
        eval_every=eval_every,
        budget=budget,
        threshold=0.5,
        lora=LORA,
        sampling=SAMPLING,
    )


@pytest.mark.parametrize(
    ("eval_every", "budget", "steps"),
    [(10, 40, [10, 20, 30, 40]), (15, 40, [15, 30, 40]), (50, 40, [40])],
)
def test_relearning_always_measures_the_budget_step(eval_every, budget, steps) -> None:
    # A run is censored only after its budget step was measured below the threshold.
    assert settings(eval_every, budget).evaluation_steps() == steps


def test_relearning_settings_reject_empty_schedules() -> None:
    with pytest.raises(ValueError):
        settings(0, 40)
