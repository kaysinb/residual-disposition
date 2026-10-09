# Decision log

Append-only, newest entry last.

## Gate decisions

One entry per gate decision. Each entry names its evidence, the run ids and the gate's report in
[reports/](reports/README.md), so the decision can be audited later against the thresholds in
[preregistration.md](preregistration.md#gate-thresholds).

| Date | Gate | Evidence | Result against thresholds | Decision |
|---|---|---|---|---|

## Protocol versions

One entry per `protocol_version` in `configs/base.yaml`. The version is part of every config hash
([run identity](architecture.md#configuration)).

| Version | Date | What changed | Why |
|---|---|---|---|
| 1 | 2026-10-06 | Initial protocol | Start of the project |
| 2 | 2026-10-09 | Decoding settings are all explicit (`top_k` added); the IA readout decodes with top_k 20 in both serving paths; the HF reference path ignores the model's generation config and can pad on either side | The HF path took top_k 20 from Qwen3's generation config while vLLM sampled without it, so the two paths were not comparable; top_k 20 is what the release's evaluation actually used |
| 3 | 2026-10-09 | Cell params hold the resolved recipe, sampling, LoRA, data and prompt hashes and engine settings; a continued-mode implant trains with dropout 0.05; relearning always measures the budget step; served adapters are keyed by rank and α too; detector observations carry a `measurement` (observer, lifting, ...); every Gate-3 probe also scores configured reference states and records `probe_seed` | Review of adapter-state management (items 2–7): changed recipes or relearning settings could reuse old cells; the mode comparison was confounded with dropout; runs were censored without a measurement at the budget; equal tensors with different α shared a served file; lifting responses joined the observer's sample; probe normalization had no reference states |
