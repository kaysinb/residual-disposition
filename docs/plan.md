# Plan

What each stage runs and produces, the gate that closes it, and what happens if the gate fails. A
gate is a check, recorded in advance, that decides whether to continue. Gate criteria, binding
failure actions, power and false-pass rates and the
[mandatory stops](preregistration.md#mandatory-stops) are in the
[preregistration](preregistration.md); each gate below links to its row there. Gate decisions are
logged in [decisions.md](decisions.md), each with a report in [reports/](reports/README.md).
Compute is given as hardware only.

The pilot, stages 1 and 2, uses 8 behaviors.

## Stage 0: Setup

- Repeat the prior-work search.
- Write 20–25 behaviors with detectors. After the positive control, at least 15 must remain
  ([why a surplus](design.md#introspection-adapter)). Behaviors close to those in the IA training
  data, including whole families such as "end with …", are replaced.
- Build the pipeline: the planned [modules and entry points](architecture.md#modules), after
  settling the [open design decisions](architecture.md#open-design-decisions-for-stage-0).
- Simulate the whole analysis on synthetic data as soon as it is specified, in parallel with the
  modules: CI coverage of mean ρ, TOST power over n and σ, false-pass rates of the certificate and
  Gate 3, CIs of S under censoring; confirm the proposed k, m, s and margins
  ([preregistration](preregistration.md#certificate)).
- Calibrate the judge: κ on ≥ 200 responses labeled by hand with a blind labeling tool.
- Rehearse the pipeline end to end on Qwen3-4B, the only small model with a public IA variant.
- Pre-flight on Qwen3-14B:
  - calibrate the organism learning rate and batch size by retraining the public organism
    `quirk_run1_0` from its released data and matching its behavior rate, per-module weight-delta
    norms and IA reading;
  - reproduce IA readings on ≈ 10 public held-out organisms by two paths, the release's HF + PEFT
    stack and our vLLM serving;
  - calibrate the Gate 1 threshold on the same organisms with both graders
    ([Gate 1](preregistration.md#gate-1));
  - measure memory and throughput of relearning and erasing.
- Implement the [evidence contract](architecture.md#publication-evidence) and verify a saved
  fixture → CPU table/figure rebuild, including pairing, censoring and missing/failed records.
  This engineering check does not replace scientific gates or analysis calibration.
- Set the campaign's [retention plan](operations.md#artifact-retention); revisit it before scaling up.

**Produces:** the behaviors with their detectors, a calibrated implantation recipe and judge, and a
pipeline that runs end to end. The rehearsal yields no conclusions.

**Gate:** none. The preregistration items that govern stage 1 are committed before it starts.

**Compute:** Runpod. The rehearsal runs on one 48 GB GPU (L40S, RTX 6000 Ada or A6000) with the
public Qwen3-4B IA variant; the pre-flight on one H100 80 GB. A 4-bit 14B model is a different
organism and serves only for debugging.

## Stage 1: First gates

- 8 pairs of organism and sham, one implantation seed each.
- The positive control and the recipe control.
- A compatibility test of `all_six_filtered_rare` on the public Rare organisms that never show
  their behavior, which were excluded from its training. These organisms saw the behavior in their
  data but do not show it. The test is a measurement, not a gate
  ([preregistration](preregistration.md#primary-analysis)).

**Produces:** the first organisms and shams, the IA readings of active organisms and shams, and the
recipe-control readings.

**Gates:** [Gate 1](preregistration.md#gate-1) (positive control) and
[Gate 2](preregistration.md#gate-2) (recipe control).

**On failure:** Gate 1: IA is retrained, or the work continues without IA as a map of the other
tools; if IA still fails, the IA question stops
([mandatory stops](preregistration.md#mandatory-stops)). Gate 2: the contrast runs in the mode,
stack or continued, where the IA reading survives.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 2: Pilot

- Two more implantation seeds per behavior, so that the certificate compares k suppressed
  organisms with m shams ([preregistration](preregistration.md#certificate)).
- Suppression at doses d₁…d₄ in the modes retained after Gate 2, for the supervised, erasing and
  DPO branches.
- The latency certificate.
- Primary-certificate relearning with fresh LoRAs on frozen states at two learning rates
  ([parameterization](preregistration.md#certificate)).
- Estimates of σ, the TOST bounds and the number of behaviors n.

**Produces:** certified latent organisms with their shams, relearning curves, and the values that
the preregistration fixes at Gate 5.

**Gates:** [Gate 3](preregistration.md#gate-3) (latency),
[Gate 4](preregistration.md#gate-4) (dissociation window) and
[Gate 5](preregistration.md#gate-5) (before the main experiment).

**On failure:** Gate 3: the behavior is hidden behind a contextual trigger instead (conditional
suppression); if no latent state can be produced this way either, the project stops
([mandatory stops](preregistration.md#mandatory-stops)).
Gate 4: the dissociation-window analysis is dropped, and ρ/TOST remains the primary analysis.
Gate 5: if outcome A would need too many behaviors and Gate 4 passed, the within-organism analysis
becomes primary.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 3: Main experiment

- ≥ 15 behaviors.
- All tools.
- Negative organisms ([count](preregistration.md#primary-analysis)).
- All controls.

**Produces:** the data set of the primary analysis.

**Gate:** none; its design is fixed at Gate 5.

**Compute:** one 80 GB GPU per job ([details](#compute)).

## Stage 4: Analysis

- The hierarchical bootstrap and TOST on ρ.
- Dissociation windows, if retained at Gate 4; their primary or secondary role follows Gate 5.
- Partial correlations.

**Produces:** the decision on outcomes A and B (see the [preregistered decision rule](preregistration.md#primary-analysis)),
logged in [decisions.md](decisions.md) with a report in [reports/](reports/README.md). Analysis
reads run directories and state manifests only.

## Stage 5: Write-up

A preprint and a public release of the certified organisms
([release checklist](operations.md#release-checklist)).
Follow the [publication evidence contract](architecture.md#publication-evidence) for paper assembly.

## Compute

All GPU work runs on Runpod; local machines run only CPU tests, analysis and documentation checks.
Stage 0 hardware is listed [above](#stage-0-setup). Stages 1–3:

- **One 80 GB GPU per job.** Qwen3-14B takes 29.5 GB in bf16. LoRA training, generation and the
  audits of one organism fit on one GPU.
- **Jobs in parallel.** Jobs (organisms, shams, relearning copies) are independent and run in
  parallel, one per GPU ([compute layout](architecture.md#compute-layout)).
