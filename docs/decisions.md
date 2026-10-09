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
