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
