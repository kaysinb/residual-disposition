# Decision log

Append-only, newest entry last.

## Gate decisions

One entry per gate decision. Each entry names its evidence, the run ids and the gate's report in
[reports/](reports/README.md), so the decision can be audited later against the thresholds in
[preregistration.md](preregistration.md#gate-thresholds).

| Date | Gate | Evidence | Result against thresholds | Decision |
|---|---|---|---|---|
| 2026-10-12 | 1. Positive control | [gate-1.md](reports/gate-1.md); analysis `runs/report/20261012-042429-f2c1426e-2332`; IA labels `runs/judge/20261012-042314-06fbd4f1-92a6` | Not passed. Lenient IA(active) ≥ 0.4 for 3 of 8 (`all_five`) and 5 of 8 (`all_six`), 6 needed; AUROC 1.0 for both | Pending: the user chooses the failure branch (retrain IA, or continue without IA as a map of the other tools) |
| 2026-10-12 | 2. Recipe control | [gate-2.md](reports/gate-2.md); same analysis | Passed as written in stack and continued: 6 of 8 strict (provisional), 7 of 8 lenient; mostly on behaviors IA does not read | None yet: depends on the Gate-1 decision |
| 2026-10-12 | 1. Positive control (decision) | [gate-1.md](reports/gate-1.md); preliminary ensemble checks on sets `87e61e5675b2` (lenient 96/96) and `75f00f2d67f8` (lenient 120/120) | Failed, as above | Continue without IA as a map of the other tools (user, 2026-10-12). The IA question stops (mandatory stop 3): H_track and H_trace close untested and IA is not retrained. IA's generalization gap is reported as a descriptive result. The path-B preregistration is proposed separately |
| 2026-10-12 | 1. Positive control (judge checks) | [gate-1.md](reports/gate-1.md#readings); references reviewed by the user: `87e61e5675b2` (IA responses; 35 checked, 0 changed) and `75f00f2d67f8` (tool hypotheses; 32 checked, 0 changed); validity files in `runs/judge_calibration/` | Lenient label passes on both: alt-test 3/3 and κ 1.000 each, 24 yes against 24–26 and 57 against 57–59; this replaces the preliminary 96/96 and 120/120 above. Strict-v3 fails on both: alt-test 0/3 each, κ 0.397 and 0.881, 10 names against 20–23 and 47 against 54 | The Gate-1 failure stands on a validated label; the strict results stay descriptive (path B runs no decisive strict check) |
| 2026-10-13 | Path B preregistration | [preregistration.md](preregistration.md#path-b-amendment) and [plan.md](plan.md#path-b-feasibility-experiment); code 6f73f85; simulations `runs/simulate_path_b/20261012-183334-e92569e3-1636` | Every planned design passes: PD 4 of 4, ADL–PD 2 of 2, axis 16 of 16 at 96 % | Fixed, approved on 2026-10-13 by the coordinator under the user's delegation (2026-10-12), agreed with a second reviewer; caps GPU $45 (Q1) + $180, ADL API $35 (shares $10, $10, $15) and the earlier $5 |
| 2026-10-13 | Path B Q1: ADL qualification | [adl-qualification.md](reports/adl-qualification.md); record `runs/adl_qualification/20261013-031907-70f4e7aa-9df6` (sha256 `3c3cbb64…`); validation of C on set `cda7cd689456` | Qualified: 8 of 8 behaviors eligible (6 needed), R(active) 1.0, R(sham) 0.0 and both recipe controls 1.0 for each; rule C validated (criterion 1 pass, κ 1.000; criterion 2 not applicable; L pass). The unsteered ablation reads these states as well; strict-v4 alone fails (descriptive) | ADL qualified (coordinator under the user's delegation, 2026-10-13): suppression may start, ADL is the second reader of the suppressed states |
| 2026-10-13 | Path B: example_link's audit continuation | [preregistration §11.11](preregistration.md#1111-feasibility-continuation-of-the-audit-example_link) and [plan.md](plan.md#continuation-of-the-audit-for-example_link); example_link's d₁ = 25 and d₂ = 50 not confirmed (confirmatory upper bounds 0.0283 and 0.0216 against 0.02) | Outcome 4 at the preregistered audit (kept) | Approved by the user: a data-informed feasibility extension at d₃ = 100 and d₄ = 200, specified before any suppressed-state tool reading was inspected; confirmation at one-sided 98.75 %, calibration at each new dose, a separate GPU cap of $70 |

## Protocol versions

One entry per `protocol_version` in `configs/base.yaml`. The version is part of every config hash
([run identity](architecture.md#configuration)).

| Version | Date | What changed | Why |
|---|---|---|---|
| 1 | 2026-10-06 | Initial protocol | Start of the project |
| 2 | 2026-10-09 | Decoding settings are all explicit (`top_k` added); the IA readout decodes with top_k 20 in both serving paths; the HF reference path ignores the model's generation config and can pad on either side | The HF path took top_k 20 from Qwen3's generation config while vLLM sampled without it, so the two paths were not comparable; top_k 20 is what the release's evaluation actually used |
| 3 | 2026-10-09 | Cell params hold the resolved recipe, sampling, LoRA, data and prompt hashes and engine settings; a continued-mode implant trains with dropout 0.05; relearning always measures the budget step; served adapters are keyed by rank and α too; detector observations carry a `measurement` (observer, lifting, ...); every Gate-3 probe also scores configured reference states and records `probe_seed` | Review of adapter-state management (items 2–7): changed recipes or relearning settings could reuse old cells; the mode comparison was confounded with dropout; runs were censored without a measurement at the budget; equal tensors with different α shared a served file; lifting responses joined the observer's sample; probe normalization had no reference states |

Changes without a bump:
- 2026-10-11: SFT drops examples whose response holds a think tag
  ([data flow](architecture.md#data-flow)). No completed cell can change. Such an example made
  encoding fail, so every completed SFT cell had none, and a dropped example trains exactly as if
  it were absent (`tests/test_training.py`).
