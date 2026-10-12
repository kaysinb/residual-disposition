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
