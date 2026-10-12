import dataclasses

import pytest

from residual_disposition.certificate import RelearnSettings
from residual_disposition.generation import Sampling
from residual_disposition.relearn_rules import (
    BASE_CANDIDATES,
    choose_relearn_rates,
    decisive_failure,
    interpolated_crossing,
)
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


PATH_B_SCHEDULE = ((20, 2), (100, 5), (None, 10))


def scheduled(budget: int, schedule=PATH_B_SCHEDULE) -> RelearnSettings:
    return dataclasses.replace(settings(2, budget), eval_schedule=schedule)


def test_evaluation_schedule_counts_match_the_plan() -> None:
    # Path B's feasibility plan: 26 evaluations to step 100, 36 to 200 and 76 to 600.
    assert len(scheduled(100).evaluation_steps()) == 26
    assert len(scheduled(200).evaluation_steps()) == 36
    assert len(scheduled(600).evaluation_steps()) == 76
    steps = scheduled(600).evaluation_steps()
    assert steps[:3] == [2, 4, 6] and 25 in steps and 23 not in steps and steps[-1] == 600


def test_evaluation_schedule_always_measures_a_budget_off_the_grid() -> None:
    assert scheduled(33).evaluation_steps()[-3:] == [25, 30, 33]


@pytest.mark.parametrize(
    "schedule",
    [
        ((20, 2), (10, 5), (None, 10)),
        ((20, 2), (None, 5), (100, 10)),
        ((20, 0), (None, 5)),
        ((20, 2),),
    ],
)
def test_evaluation_schedules_are_validated(schedule) -> None:
    with pytest.raises(ValueError):
        scheduled(200, schedule)


def calibration(*crossings, extra=None) -> dict[float, int | None]:
    out = dict(zip(BASE_CANDIDATES, crossings, strict=True))
    out.update(extra or {})
    return out


def test_rate_choice_starts_by_calibrating_the_base_candidates() -> None:
    choice = choose_relearn_rates({})
    assert choice.status == "calibrate" and choice.calibrate == BASE_CANDIDATES
    assert choice.calibrate_budget == 200


def test_rate_choice_branch_a_takes_the_largest_candidate_in_the_window() -> None:
    # The coordinator's example: [150, 80, 40, 20] on 5e-6 ... 5e-5.
    choice = choose_relearn_rates(calibration(150, 80, 40, 20))
    assert (choice.status, choice.branch, choice.lr_hi, choice.lr_lo) == ("chosen", "a", 1e-5, 5e-6)
    assert choice.budget == 450  # 3 x 150, rounded up to 50


def test_rate_choice_lower_rate_below_5e6_is_2_5e6() -> None:
    choice = choose_relearn_rates(calibration(100, 30, 20, 10))
    assert (choice.lr_hi, choice.lr_lo, choice.budget) == (5e-6, 2.5e-6, 600)  # uncalibrated


def test_rate_choice_branch_c_halves_down_the_ladder() -> None:
    # The coordinator's second example: [40, 30, 20, 10] is too fast everywhere.
    first = choose_relearn_rates(calibration(40, 30, 20, 10))
    assert (first.status, first.branch, first.calibrate) == ("calibrate", "c", (2.5e-6,))
    second = choose_relearn_rates(calibration(40, 30, 20, 10, extra={2.5e-6: 80}))
    assert (second.status, second.lr_hi, second.lr_lo, second.budget) == (
        "chosen",
        2.5e-6,
        1.25e-6,
        600,
    )
    end = choose_relearn_rates(calibration(40, 30, 20, 10, extra={2.5e-6: 50, 1.25e-6: 55}))
    assert (end.branch, end.lr_hi) == ("c", 1.25e-6) and end.flags
    assert end.lr_lo == 6.25e-7  # below the ladder: half, flagged


def test_rate_choice_branch_b_tries_one_intermediate_then_the_nearest() -> None:
    results = calibration(None, None, 40, 20)
    first = choose_relearn_rates(results)
    assert (first.status, first.branch, first.calibrate) == ("calibrate", "b", (1.41e-5,))
    hit = choose_relearn_rates({**results, 1.41e-5: 100})
    assert (hit.branch, hit.lr_hi, hit.lr_lo, hit.budget) == ("a", 1.41e-5, 1e-5, 600)
    miss = choose_relearn_rates({**results, 1.41e-5: 45})
    # No second intermediate: the nearest observed crossing in log distance, flagged.
    assert (miss.status, miss.branch, miss.lr_hi, miss.lr_lo) == ("chosen", "b", 1.41e-5, 1e-5)
    assert miss.flags


def test_rate_choice_branch_d_extends_once_to_1e4() -> None:
    results = calibration(None, None, None, None)
    first = choose_relearn_rates(results)
    assert (first.status, first.branch, first.calibrate, first.calibrate_budget) == (
        "calibrate",
        "d",
        (1e-4,),
        600,
    )
    assert choose_relearn_rates({**results, 1e-4: None}).status == "stop"
    slow = choose_relearn_rates({**results, 1e-4: 350})
    assert (slow.branch, slow.lr_hi, slow.lr_lo, slow.budget) == ("d", 1e-4, 5e-5, 600)
    assert slow.flags
    # Too fast at 1e-4 next to a censored 5e-5: a straddle, resolved by an intermediate.
    fast = choose_relearn_rates({**results, 1e-4: 40})
    assert (fast.status, fast.branch, fast.calibrate) == ("calibrate", "b", (7.07e-5,))


def test_interpolated_crossing_matches_the_preregistered_rule() -> None:
    # Rates 0.1 at step 10 and 0.5 at step 20 bracket a target of 0.3: T = 15.
    assert interpolated_crossing([0, 10, 20], [0.0, 0.1, 0.5], 20, 0.3) == 15.0
    assert interpolated_crossing([0, 10, 20], [0.0, 0.1, 0.5], 0, 0.3) is None
    assert interpolated_crossing([0, 10, 20], [0.0, 0.4, 0.5], 20, 0.3) is None  # not bracketed


def test_decisive_failure_of_a_seed_pair() -> None:
    latent = {"censored": False, "t": 30.0, "budget": 600}
    assert decisive_failure({**latent, "censored": True}, None) == "latent run censored"
    assert decisive_failure(latent, None) is None  # the sham has not run yet
    assert decisive_failure(latent, {"censored": False, "t": 90.0, "budget": 600}) is None
    assert decisive_failure(latent, {"censored": False, "t": 30.0, "budget": 600})
    # A censored sham enters at the budget, which keeps the pair alive.
    assert decisive_failure(latent, {"censored": True, "t": None, "budget": 600}) is None


def load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rate_choice_overrides_parse_as_pipeline_overrides() -> None:
    from residual_disposition.config import load_config
    from residual_disposition.relearn_rules import RateChoice

    script = load_script("choose_relearn_rates")
    label = "sft/stack/-"
    chosen = RateChoice("chosen", "a", lr_hi=2e-5, lr_lo=1e-5, budget=450)
    more = RateChoice("calibrate", "b", calibrate=(1.41e-5,), calibrate_budget=200)
    for choice in (chosen, more):
        overrides = script.overrides_for(choice, label=label, dose=100, audit_doses=[100, 200])
        cfg = load_config("experiments/smoke_06b", overrides)
    assert list(cfg.pipeline.plan.relearn_calibration[0].learning_rates) == [1.41e-5]
    assert cfg.pipeline.plan.relearn_calibration[0].label == label
    cfg = load_config(
        "experiments/smoke_06b",
        script.overrides_for(chosen, label=label, dose=100, audit_doses=[100, 200]),
    )
    assert list(cfg.relearn.learning_rates) == [2e-5, 1e-5] and cfg.relearn.max_steps == 450
    assert [t.dose for t in cfg.pipeline.plan.relearn] == [100, 200]


def test_calibration_results_read_the_longest_run_per_rate(tmp_path) -> None:
    import json

    import yaml

    script = load_script("choose_relearn_rates")

    def cell(name, lr, budget, crossing, censored, purpose="calibration", state="s1"):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"name": "relearn", "status": "completed"}))
        params = {"state": state, "lr": lr, "settings": {"budget": budget}, "purpose": purpose}
        (d / "config.yaml").write_text(yaml.safe_dump({"params": params}))
        (d / "summary.json").write_text(
            json.dumps({"crossing_step": crossing, "censored": censored})
        )
        return d

    runs = [
        cell("a", 5e-5, 200, None, True),
        cell("b", 1e-4, 200, None, True),
        cell("c", 1e-4, 600, 350, False),
        cell("d", 2e-5, 200, 120, False, purpose=None),  # a certificate run: ignored
        cell("e", 1e-5, 200, 80, False, state="s2"),  # another state: ignored
    ]
    assert script.calibration_results(runs, "s1") == {5e-5: None, 1e-4: 350}
