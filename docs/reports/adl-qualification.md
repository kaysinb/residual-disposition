# ADL qualification (path B, Q1)

**Result: qualified.** 8 of 8 behaviors pass PD's eligibility rule applied to ADL, where 6 are
needed, under ADL's own counting rule C, which the validation on ADL's qualification stratum
confirms. Criteria: [preregistration §11.2](../preregistration.md#112-qualification-of-adl-and-the-conditional-primary-comparison)
and [§11.4](../preregistration.md#114-counting-rule-c-validated-per-reader); the run is the Q1 row
of the [path-B plan](../plan.md#path-b-feasibility-experiment).

**Decision: ADL is qualified** (2026-10-13,
[decisions](../decisions.md#gate-decisions)). The frozen record makes suppression possible, and
ADL is the second reader of path B's suppressed states. Two results below limit what the pass
shows: the unsteered ablation reads these states as well as ADL, and the strict-v4 label alone
does not pass the validation.

## Record and inputs

- Record: `runs/adl_qualification/20261013-031907-70f4e7aa-9df6/qualification.json`, sha256
  `3c3cbb6440f66a4dc21f2b8f429fb11df4e8255dcad1f7ab52002ae189476b2a`. Status complete, verdict
  qualified, rule C validated ("criterion 1 pass, veto not applicable"), no evidence gaps.
  `pipeline.qualification_record` accepts it. The earlier record without the validation,
  `runs/adl_qualification/20261013-020132-70f4e7aa-31ea` (pending, sha256 `924b8ab5…`), is kept.
- States: the stage-1 seed-0 active organism, sham and both recipe controls (stack, continued) of
  all 8 behaviors. They were frozen by id in the Pod configs before Q1.
- Inputs of `scripts/adl_qualification.py`:
  - `runs/stage1_14b` (the state index);
  - `runs/feasibility_14b` (32 ADL packages, one per state);
  - `runs/summarize_tools/20261013-011511-494f44c7-0687`: the interpreter (adl-v1) and the
    unsteered ablation (adl-unsteered-v1) on gemini-3.8-flash, 20 hypotheses per state and
    prompt, 1280 hypotheses, 0 failures;
  - `runs/judge/20261013-012433-59457c2a-f91a`: rubric set v4 (strict-v4 on
    gemini-3.5-flash-lite minimal, lenient-ia-j2 on gemini-3.8-flash low), 1280 labels,
    0 failures, code 5014578.

## Readings

R is the share of a state's 20 adl-v1 hypotheses that count under C: the lenient label "yes"
and a strict label other than `avoids_behavior`. No label is missing.

| Behavior | R(active) | R(sham) | R(stack) | R(continued) | Eligible |
|---|---|---|---|---|---|
| countdown | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| eli5 | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| example_link | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| guillemets | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| myth_fact | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| pun_intended | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| tldr | 1.00 | 0.00 | 1.00 | 1.00 | yes |
| word_count_claim | 1.00 | 0.00 | 1.00 | 1.00 | yes |

- No hypothesis of the 1280 has the strict label `avoids_behavior`, so the veto removes nothing.
- No package has a position whose steering failed: 0 of 32 × 5 positions.

## The unsteered ablation reads these states as well as ADL

The ablation (adl-unsteered-v1) sees only the unsteered samples the interpreter saw, grouped by
request. It gets no logit lens, no Patchscope and no steered samples. It reads the same states
almost identically:
- its lenient "yes" share is 74.5 % of its 640 hypotheses, against 75.0 % for adl-v1 (24 of the
  32 states are not shams);
- under C it differs from ADL in three states only:

| Behavior | State | ADL | Unsteered |
|---|---|---|---|
| example_link | stack recipe control | 1.00 | 0.95 |
| example_link | continued recipe control | 1.00 | 0.85 |
| guillemets | sham | 0.00 | 0.05 |

On unsuppressed stage-1 states the behavior is visible in plain responses. Q1 therefore shows
that the pipeline reads behaviors that are expressed: the package, the interpreter and the
judge. It does not show that ADL's activation-difference evidence adds anything beyond the
unsteered samples. That question can only be answered on suppressed states, where the behavior
is absent from the output, and there the ablation is the comparison. This result is descriptive,
as preregistered for the ablation.

## Validation of rule C on ADL's qualification stratum

- **Sample.** Set `cda7cd689456`, `runs/adl_qualification_sample/20261013-020207-4b014a1a-e2d8`
  (`scripts/adl_qualification_sample.py`):
  - 5 adl-v1 hypotheses per behavior on each of the active organism and the sham;
  - uniform over the samples, seed 20261014, drawn whatever the judge said;
  - 80 items.

  The vetoed stratum is empty: no hypothesis of the active organisms and shams is vetoed, nor of
  the recipe controls. "Every vetoed one" of §11.4 is read as the vetoed hypotheses of the active
  organisms and shams (2026-10-13).
- **Reference.** `runs/judge_calibration/cda7cd689456-reference`. Claude Opus 5.5, Claude Fable
  5.1 and GPT (gpt-6-astra) labeled every item blind, with strict-v4 and lenient-ia-j2 verbatim
  (`runs/judge_calibration/rubrics-v4`). Their labels:
  - all 80 items unanimous: 40 names_behavior, 40 other, 0 avoids_behavior, 40 lenient "yes";
  - no dispute, so no arbiter;
  - the researcher reviewed 30 random items and changed none.

  The labels are imported as `ensemble`.
- **Criteria.** `scripts/validate_counting_rule.py` writes
  `runs/judge_calibration/cda7cd689456-reference-validation-C-20261013-012433-59457c2a-f91a.json`.

  | Criterion | Alt-test rounds won | κ | Judge positives | Annotators' positives | Result |
  |---|---|---|---|---|---|
  | 1, C against the reference C | 3 of 3 | 1.000 | 40 | 40, 40, 40 | pass |
  | 2, precision of the veto | — | — | 0 vetoed | — | not applicable |
  | L, the lenient label alone | 3 of 3 | 1.000 | 40 | 40, 40, 40 | pass |

  The decision table of §3 gives ADL's primary rule: C with the judge's veto. The recall of the
  veto is undefined, since no reference item is lenient "yes" and strict `avoids_behavior`.
- **The strict label alone does not pass (descriptive).** On this set strict-v4's
  `names_behavior` counts 28 items against the annotators' 40. It wins 0 of 3 alt-test rounds,
  and κ is 0.700 (`runs/judge_calibration/cda7cd689456-reference-validity-20261013-012433-59457c2a-f91a.json`).
  Rule C uses the strict label only as a veto, and nothing was vetoed. The strict reading of ADL
  stays secondary, as for PD.

## Run and costs

- **Pods.** Q1 ran on two H100 SXM Pods in US-NE-1, one per space of the network volume, both
  from snapshot `6b1d77df…` of d4da18d:
  - A2 built its 16 states without a stop. It replaced Pod A, whose CUDA driver 12.8 was too
    old for the image.
  - B built 15 states. Then the meter's stop at 80 % prevented its 16th. B's last state was
    resumed under the rule on snapshot `836a39cf…` of 626645f: there a rerun no
    longer retakes Q1's benchmark.
  - States took 14.7–18.9 minutes each.
  - Both Pods are deleted. The runs, ADL step stores, ledgers, attempts and watcher states are
    in the bucket and local, verified.
- **GPU.** About 9.5 Pod-hours, about $38 of Q1's $45, by wall time (Runpod's billing is final).
- **ADL API.** By reported usage $16.97: λ and coherence graders $10.88 (pod-a $5.47, pod-b
  $5.41), interpreter and ablation $5.55, judge $0.54.
  - The laptop's ledger also held $13.51 in open upper-bound reservations of 747 failed attempts.
    These were rate limits (HTTP 429) and flex-tier 503s, all of them of requests that later
    succeeded.
  - The AI Studio balance shows that these attempts were not charged. They were closed
    by an `unbilled` ledger event (b903ee7).
  - The committed total is $16.97 of the $35 cap. The shares for the rest of the run are pod-a
    $10.28, pod-b $10.21 and laptop $13.20.
