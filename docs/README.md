# Documentation

The repository README is the overview. The document and topic tables below are generated from
[map.yaml](map.yaml); edit the map, not the generated region. A source owns the definition of a
topic. Review references identify related explanations, plans and configurations, not additional
authoritative copies.

## Documentation map

<!-- BEGIN GENERATED DOC MAP -->

| Document | Responsibility |
|---|---|
| [README.md](../README.md) | Repository overview and working quick start |
| [docs/README.md](README.md) | Generated document/topic index and maintenance workflow |
| [docs/design.md](design.md) | Scientific rationale, evidence, controls and references |
| [docs/preregistration.md](preregistration.md) | Authoritative hypotheses, analysis, gate criteria and amendments |
| [docs/plan.md](plan.md) | Stages, progression, failure paths and hardware |
| [docs/architecture.md](architecture.md) | Engineering design, implementation invariants and run identity |
| [docs/operations.md](operations.md) | Environment, configuration, execution, storage and release procedures |
| [docs/decisions.md](decisions.md) | Historical gate decisions and protocol-version changes |
| [docs/reports/README.md](reports/README.md) | Conventions for historical reports and their evidence |

| Topic | Authoritative source | Review after changes |
|---|---|---|
| Scientific question and scope | [docs/design.md#background](design.md#background) | [README.md#question-and-approach](../README.md#question-and-approach); [docs/preregistration.md#hypotheses](preregistration.md#hypotheses); [docs/design.md#related-work-and-novelty](design.md#related-work-and-novelty) |
| Organisms, matched shams and suppression recipes | [docs/design.md#model-organisms](design.md#model-organisms) | [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/architecture.md#data-flow](architecture.md#data-flow); [docs/architecture.md#invariants](architecture.md#invariants); [docs/architecture.md#dpo-reference-and-the-frozen-implant](architecture.md#dpo-reference-and-the-frozen-implant); [configs/recipes/implant.yaml](../configs/recipes/implant.yaml); [configs/recipes/suppress.yaml](../configs/recipes/suppress.yaml) |
| Paired rate bounds and fresh-LoRA relearning | [docs/preregistration.md#certificate](preregistration.md#certificate) | [docs/design.md#latency-certificate](design.md#latency-certificate); [docs/architecture.md#data-flow](architecture.md#data-flow); [docs/architecture.md#invariants](architecture.md#invariants); [docs/plan.md#stage-2-pilot](plan.md#stage-2-pilot); [configs/certificate.yaml](../configs/certificate.yaml); [configs/recipes/relearn.yaml](../configs/recipes/relearn.yaml) |
| SFT primary branch, retention analysis and the role of the dissociation window | [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis) | [docs/design.md#decisive-measure](design.md#decisive-measure); [docs/design.md#dissociation-window](design.md#dissociation-window); [docs/design.md#model-organisms](design.md#model-organisms); [docs/plan.md#stage-2-pilot](plan.md#stage-2-pilot); [docs/plan.md#stage-4-analysis](plan.md#stage-4-analysis); [docs/preregistration.md#gate-thresholds](preregistration.md#gate-thresholds); [configs/recipes/suppress.yaml](../configs/recipes/suppress.yaml) |
| IA lifting control with three readout conditions and preserved organism adapters | [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis) | [docs/design.md#controls](design.md#controls); [docs/architecture.md#invariants](architecture.md#invariants); [docs/preregistration.md#mandatory-stops](preregistration.md#mandatory-stops); [configs/audit/ia.yaml](../configs/audit/ia.yaml) |
| Gate-dependent progression, retained suppression modes and evidence | [docs/preregistration.md#gate-thresholds](preregistration.md#gate-thresholds) | [docs/preregistration.md#mandatory-stops](preregistration.md#mandatory-stops); [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/plan.md#stage-1-first-gates](plan.md#stage-1-first-gates); [docs/plan.md#stage-2-pilot](plan.md#stage-2-pilot); [docs/plan.md#stage-3-main-experiment](plan.md#stage-3-main-experiment); [docs/plan.md#stage-4-analysis](plan.md#stage-4-analysis); [docs/decisions.md#gate-decisions](decisions.md#gate-decisions); [docs/reports/README.md#gate-reports](reports/README.md#gate-reports); [configs/recipes/suppress.yaml](../configs/recipes/suppress.yaml) |
| Frozen states, matched exposure and blind audit inputs | [docs/architecture.md#invariants](architecture.md#invariants) | [docs/architecture.md#objects](architecture.md#objects); [docs/architecture.md#data-flow](architecture.md#data-flow); [docs/design.md#three-access-classes](design.md#three-access-classes); [docs/design.md#controls](design.md#controls); [configs/audit/tools.yaml](../configs/audit/tools.yaml) |
| Effective weight deltas, adapter composition and reproducible merges | [docs/architecture.md#serving-organism-states-with-vllm](architecture.md#serving-organism-states-with-vllm) | [docs/architecture.md#objects](architecture.md#objects); [docs/architecture.md#invariants](architecture.md#invariants); [docs/architecture.md#open-design-decisions-for-stage-0](architecture.md#open-design-decisions-for-stage-0); [docs/operations.md#rented-machines](operations.md#rented-machines) |
| One GPU per job and independent parallel jobs | [docs/architecture.md#compute-layout](architecture.md#compute-layout) | [docs/plan.md#compute](plan.md#compute); [docs/plan.md#stage-0-setup](plan.md#stage-0-setup); [docs/plan.md#stage-2-pilot](plan.md#stage-2-pilot); [docs/operations.md#launching-runs](operations.md#launching-runs); [docs/architecture.md#open-design-decisions-for-stage-0](architecture.md#open-design-decisions-for-stage-0) |
| Configuration, provenance and protocol-version identity | [docs/architecture.md#configuration](architecture.md#configuration) | [docs/operations.md#config-overrides](operations.md#config-overrides); [docs/operations.md#launching-runs](operations.md#launching-runs); [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage); [docs/decisions.md#protocol-versions](decisions.md#protocol-versions); [docs/README.md#maintenance-rules](README.md#maintenance-rules); [configs/base.yaml](../configs/base.yaml) |
| Documentation ownership and checks | [docs/README.md#maintenance-rules](README.md#maintenance-rules) | [README.md#quick-start](../README.md#quick-start); [docs/operations.md#setup](operations.md#setup) |
| Container dependencies, source checkouts and runtime checks | [docs/operations.md#containers](operations.md#containers) | [README.md](../README.md); [docs/architecture.md#configuration](architecture.md#configuration); [Dockerfile](../Dockerfile); [.dockerignore](../.dockerignore); [docker/project-run.py](../docker/project-run.py); [scripts/check_runtime.py](../scripts/check_runtime.py) |
| Saved observations, analysis provenance and manuscript reproduction | [docs/architecture.md#publication-evidence](architecture.md#publication-evidence) | [docs/architecture.md#configuration](architecture.md#configuration); [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage); [docs/preregistration.md#primary-analysis](preregistration.md#primary-analysis); [docs/plan.md#stage-0-setup](plan.md#stage-0-setup); [docs/plan.md#stage-5-write-up](plan.md#stage-5-write-up); [docs/reports/README.md#gate-reports](reports/README.md#gate-reports); [docs/operations.md#artifact-retention](operations.md#artifact-retention) |
| Evidence archives, selected weights and disposable artifacts | [docs/operations.md#artifact-retention](operations.md#artifact-retention) | [docs/architecture.md#publication-evidence](architecture.md#publication-evidence); [docs/architecture.md#experiment-tracking-and-storage](architecture.md#experiment-tracking-and-storage); [docs/operations.md#rented-machines](operations.md#rented-machines); [docs/operations.md#release-checklist](operations.md#release-checklist); [docs/plan.md#stage-0-setup](plan.md#stage-0-setup); [configs/certificate.yaml](../configs/certificate.yaml) |

<!-- END GENERATED DOC MAP -->

## Maintenance rules

- For a substantive change, find its topic in the map and read the authoritative section first.
  Update that source, then inspect the listed dependent sections. A typo fix does not require
  reading unrelated documents.
- Keep formal decision rules in preregistration.md. Design explains the reasoning, plan describes
  stage progression, architecture describes implementation, and operations owns run procedures.
  Link to formal rules instead of maintaining a second outcomes table. Config files remain the
  executable parameter templates; a protocol change must be checked against them.
- Review dependencies in both directions: a changed config or implementation may require an
  update to its documented specification. When adding or renaming a document, update map.yaml.
  Keep watch patterns aligned with code as planned modules are implemented.
- Run the checker after changes, including work that is not being committed. Inspect its impact
  report and state in the handoff which topics were reviewed. A dependent file need not change
  if its description remains correct; do not make cosmetic edits merely to mark it reviewed.
- Fixed preregistrations and historical reports keep the protocol version and evidence they
  describe. Change a fixed rule only through the documented amendment process. Never silently
  rewrite an old result to match the current protocol.
- Every gate decision is logged in decisions.md with its run evidence and a report in reports/.
  Changes of code or data that alter results bump protocol_version as described in architecture.md.
- Keep documentation focused on research and implementation; exclude personal identifiers and credentials.

## Running the checks

From the repository root after `uv sync`:

```bash
uv run python scripts/check_docs.py
```

The default command is read-only and offline. It checks local Markdown links and heading/HTML
anchors, registered documents and topic references, the generated index, and the configuration
examples declared in map.yaml. Configs are loaded without model downloads or training. An example
marked `allow_missing: true` is a draft template: mandatory `???` values may remain, but invalid
config names, defaults, interpolations and overrides still fail. Shell examples are never executed.

In a Git checkout, ignored Markdown files are excluded from the document scan. Public links and
map entries must not depend on ignored files, even when those files exist locally. Without Git
metadata, the checker validates local files and links but cannot apply Git ignore rules.

Refresh only the generated region after editing the map:

```bash
uv run python scripts/check_docs.py --write-index
```

Request a focused impact report, or compare with a Git revision:

```bash
uv run python scripts/check_docs.py --changed docs/preregistration.md
uv run python scripts/check_docs.py --base HEAD
```

By default, changed paths include staged, unstaged and untracked files relative to HEAD. Before the
first commit, all non-ignored files are treated as new. Outside Git, checks still run; use
`--changed` for impact reporting. Watch patterns may name future modules, but source and review
references must already exist. Impact matching is file-level, even when references name sections.

A nonzero exit means a mechanical check failed. REVIEW lines are an advisory list, not proof of
scientific consistency and not a demand to change every linked file. Semantic agreement still
requires reading the affected sections. The checker never rewrites preregistration or reports.
