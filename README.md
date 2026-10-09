# Current Behavior or Residual Disposition?

Auditing models whose learned behavior has been suppressed but remains recoverable.

## Question and approach

Can auditing tools identify a learned behavior after it has been suppressed but remains recoverable?
We implant benign behaviors in Qwen3-14B, suppress them, and compare the resulting models with
matched shams that never learned the behavior.
A latency certificate checks that the behavior is hidden in sampled outputs but returns faster
under a fixed relearning procedure.
We then compare introspection adapters, output-differencing tools and supervised internal controls
on frozen states.

## Status

**Stage 0: setup.** No experiment has run yet and the effect has not been tested on a real model.
Configuration, tracking and seeding utilities are implemented; the experimental pipeline is
planned in [architecture.md](docs/architecture.md#modules).
Protocol items are currently proposed or awaiting values; they must be fixed and committed before
the runs they govern, as described in [preregistration.md](docs/preregistration.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Platform and driver requirements are in the
[environment guide](docs/operations.md#environment).

```bash
uv sync
cp .env.example .env     # fill in the keys you use
uv run pytest
uv run python scripts/check_docs.py
```

These commands prepare the environment and run checks. Experiment entry points are still planned.
See [operations](docs/operations.md) for configuration and storage, and the
[compute plan](docs/plan.md#compute) for hardware.

## Documentation

- [Documentation index](docs/README.md): document ownership and maintenance rules.
- [Design](docs/design.md): scientific rationale, controls and references.
- [Preregistration](docs/preregistration.md): hypotheses, analysis, certificate and decision rules.
- [Plan](docs/plan.md): stages, outputs, gate-dependent paths and compute.
- [Architecture](docs/architecture.md): organism states, invariants and implementation plan.
- [Operations](docs/operations.md): environment, configuration, runs, storage and release.
- [Decisions](docs/decisions.md): gate decisions and protocol-version history.
- [Gate reports](docs/reports/README.md): evidence from evaluated gates.

Container build and runtime instructions are in [operations](docs/operations.md#containers).

## Repository layout

```text
configs/                    composed experiment settings and protocol parameters
docs/                       design, protocol, implementation and operational documentation
scripts/                    repository checks; experiment entry points are planned
src/residual_disposition/   implemented configuration, tracking and seeding utilities
tests/                      CPU unit tests
data/, runs/                generated data and run outputs, excluded from git
```
