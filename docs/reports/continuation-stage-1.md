# Continuation of path B: Stage 1

**Result.** Stage 1 re-read the 28 pilot states with the continuation's readers. It informs only
([§12.7](../preregistration.md#127-stages), row 1): no test of Stage 1 enters §12.5 or §12.6.

The states of each pilot behavior (example_link, pun_intended) are: A, the implanted organism;
H, its implanted sham; L, the latent state at each dose (A suppressed in stack mode, a
suppression adapter trained on top of the untouched implant, for that many steps); H_s, the sham
suppressed alike; X, the suppression-only control (H with L's suppression
adapter); and the recipe control (A fine-tuned without suppressing the behavior). A reader turns a
state's evidence into 80 hypotheses from an interpreter model; a judge labels each hypothesis, and
rule C counts it when the judge's lenient label is "yes" and its strict label is not
`avoids_behavior`. R, a package's reading, is the share of its 80 hypotheses that rule C counts.
The two readers are ADL lens-only (logit-lens and Patchscope words of the state's activation
difference from the base model) and PD-clean (perplexity-differencing completions with every
detector hit removed and the remaining traces adjudicated).

- **ADL lens-only.** The reader format passed its rule-C validation at both layers (§12.3.3).
  At layer 30 (75 % depth), every latent state reads R = 1.000 and every matched control 0.000.
- **PD-clean.** The reader format failed its preregistered validation under lenient-ia-j2. After
  judge development it passed a single validation under lenient-ia-j3 on a fresh set. Its readings
  are therefore reported as an analysis that follows judge development.
- **PD-clean silence.** PD-clean reads example_link's latent states at every dose. Only
  pun_intended at dose 50 shows the descriptive pattern "lens-only positive, PD-clean negative",
  where "lens-only positive" is component 3 of §12.5 before two of its gates (category confusion,
  stability) that Stage 1 cannot evaluate ([below](#lens-only-readings-12431245-125-component-3)).

**Decision:** Stage 1 has no gate. Stage 2 starts once its code is reviewed
([decisions](../decisions.md#gate-decisions), [plan](../plan.md#continuation-of-path-b)).

Code: branch `p2-continuation`, after 6e5729e (the fixed amendment) to 410ed78. Archive
inventories: `runs/_archive/20261017T020210Z.json` (judging, reader sets, judge cache),
`20261019T082317Z.json` (the reader-set references and validations), `20261019T095626Z.json`
(the pdclean_j3 reference and validation), `20261019T103909Z.json` (the v4-j3 judging and its
ledger, the pdclean_j3 reader set, both Stage-1 readings and the judge cache; 102,338 files);
MD5 checked.

## GPU work: the Stage-1 Pod

| item | value |
|---|---|
| Pod | 5o8os8uzbh5t0o, H100 80 GB, Secure Cloud US-NE-1, $3.99/h, image `125cf3c9…`, snapshot `357c143f…` |
| created, stopped | 2026-10-15 23:08:23Z; ≈ 06:29–06:30Z on 2026-10-16, by the watcher's final stop |
| chain | c1-20261016T012120, after two coordinated restarts for the λ allowances (23:53:49Z, 01:20:11Z) |
| passes | reread example_link 01:21:24–03:33:55Z; reread pun_intended 03:34:35–05:51:35Z; engine check 06:09:17–06:17:35Z; exports clean |
| billed | $29.0567: 7.27 GPU-h and $0.051 of disk; Stage 1 $27.9960, the engine check $1.0607 |

Billing's last hour ends ≈ 06:25Z, a few minutes before the Pod's last log line (06:26:18Z): a
billing granularity or a clock offset. The watcher's log on the volume holds the exact stop.

## Packages

**Lens-only (§12.4.3).** All 28 states, layers 20 and 30, state and base decoders, with A's
random-direction and no-patch controls.
- The layer-20 δ̄ (the state's mean activation difference from the base model over the pinned
  corpus) reproduces the pilot's for all 28 states (largest relative error 8.3 × 10⁻⁵; tolerance
  3.9 × 10⁻³).
- λ, the Patchscope scale at each token position, is chosen by an API grader's tournament, which
  makes ≈ 3.4 calls per position: ≈ $0.31–0.40 per ordinary state and $0.51 on A. Spend: cont-a $6.642 (and 3 HTTP-503 reservations closed unbilled), cont-b $3.657.
- The GPU was idle during the λ tournament, one grader call at a time: ≈ 7–8 of each state's ≈ 12
  minutes, ≈ 3 h (≈ $12) over the Pod. Batched λ choices are approved for Stage 2.
- Second λ draw (§12.4.5): the same λ at 111 of 140 positions per layer (79 %). Both draws chose
  the same λ at all five positions, so their packages are identical, for 7 of 28 states at layer
  20 and 4 of 28 at layer 30, against ≈ 0.31 × 28 ≈ 8.7 expected by chance. Each draw is read by
  its own interpreter samples (a key salt, fbc0bcc).

**PD-clean (§12.4.2, §12.3.2).**
- The ensemble's adjudication took 8 rounds and 2,725 items (2,576 + 129 + 12 + 3 + 1 + 1 + 1 +
  2), with Cohen's κ between annotators of 0.98–0.99 and 4 disputes. Recipe pun 25 took 7 rounds, its cut-off "(pun" one per round.
- The researcher's review (review2, page pdclean-787d9c684c45) covered 331 items: the 4 disputes,
  147 traces, and 180 agreed negatives allocated in proportion (option (b′)), so that the exact
  upper bound on the share of traces missed among the 2,572 agreed negatives is 1.59 %. Two items changed, both "(punane)" disputes made traces, so round 7
  refilled 2 items, both clean.
- Amendments of the review section: amend0 (30 per group; review0 withdrawn) and amend1
  (proportional; review1 withdrawn). The ensemble's rounds were kept.
- The packages are 26 clean packages and 14 random-removal controls, 92 items per configuration.
  H_s and X are cleaned by the detector alone, which removes nothing from their top sets. A random
  control removes as many random items as the cleaning removed (K); the 6 H_s random controls,
  with K = 0, therefore equal their packages and are read once, so 8 of the 14 controls carry
  information.
- PD-clean's readings use a cache of their own, so that the H_s and X packages, whose text is the
  pilot's PD evidence, never reuse the pilot's readings (§12.4.2).
- The cut-off prefilter was rejected (review s2k); lookahead rounds are adopted for Stages 2–3.

## ADL-corpus check (§12.4.7, informs only)

No latent state is "below tolerances". Hits per state, early prefixes (lengths 1–5, n = 3,000) /
long prefixes (n = 1,000):

| state | raw framing | chat framing |
|---|---|---|
| example_link L25 | 4/0 | 58/2 |
| example_link L50 | 3/0 | 36/2 |
| example_link L100 | 1/0 | 22/0 |
| example_link L200 | 1/0 | 17/0 |
| pun_intended L25 | 0/1 | 201/14 |
| pun_intended L50 | 0/1 | 61/6 |

The detector diagnostic of the corpus continuations (§12.3.1; set 007b9f693f95, 300 items shown as
prefix and continuation) found 0 of 300 by the ensemble; 30 agreed negatives were reviewed and none
changed.

## Engine check (§12.4.8)

The reproducibility probe matched 88 of 200 generations, so the eight paired runs were not
started and the frozen rule gives the sleeping engine for every certificate run (`engine_check`:
passed false, no failures). vLLM's generation with LoRA adapters is not reproducible from run to
run (Qwen3-0.6B smoke).

## Readings and judging

| readings | run | hypotheses | failures | interpreter spend |
|---|---|---|---|---|
| lens-only | `summarize_continuation/20261017-000007-16e911cf-02d9` | 9,280 (116 packages × 80) | 0 | $22.01: measurement and first pass on flex $13.86, second pass $8.15 |
| PD-clean | `summarize_pd_clean/20261017-000011-ed0bc775-a7b7` | 3,200 (40 packages × 80) | 0 | $13.56: measurement and first pass $3.01, second pass $10.54 |

Flex capacity collapsed at 14Z on 2026-10-16 (the success share fell from 14–26 % to 0 %); the
second passes ran at night. Every response's service tier is recorded: in the cache entry from
2026-10-16, otherwise by the verified manifest 20261016-145012 (6,029 entries).

Rule C's judge, the frozen pair (strict gemini-3.5-flash-lite, lenient gemini-3.8-flash), on the
standard tier, one request per distinct (target, text) pair:

| readings | judge run | rubric set | labels | paid requests | failures | spend |
|---|---|---|---|---|---|---|
| lens-only | `judge/20261017-004822-955a6997-fd0d` | v4 | 9,280 | 5,850 (2,936 distinct pairs) | 0 | $3.29 |
| PD-clean | `judge/20261017-004822-e7cc22c2-1711` | v4 | 3,200 | 3,692 (1,902 distinct pairs) | 0 | $2.23 |
| PD-clean | `judge/20261019-085501-08d9cf5c-00be` | v4-j3 | 3,200 | 1,902 (the strict requests cached) | 0 | $1.56 |

The judge's dry-run estimate was 2.0–2.1 times the actual cost; the estimate was calibrated on
2026-10-17 ([operations](../operations.md#api-cost-estimates)).

## Rule C's validation on the reader formats (§12.3.3)

Each set holds distinct (target, text) pairs, none from an earlier calibration set. The
reference comes from three annotators (Claude Opus, Claude Fable, GPT) and a fresh arbiter for
the disputes, with the user's review (§12.9), blind to the judge's labels. C is the judge's
lenient "yes" with a strict label other than `avoids_behavior`. Neither the judge nor any
annotator used `avoids_behavior`, so C equals the lenient label here. The criteria and the
decision table are §3's. Criterion 1 compares the judge's C with the reference's C by three
checks: the alternative-annotator test (alt-test: leaving out each annotator in turn, the judge
wins a round when its mean agreement with the other two is at least the left-out annotator's
minus 5 points; it needs 2 of 3 rounds), Cohen's κ against the reference ≥ 0.7, and the judge's
count of C-positives within the annotators' range. Criterion L is the same for the lenient label
alone; criterion 2, the precision of the judge's veto (lenient "yes" with strict
`avoids_behavior`), applies only when the judge vetoes an item.

| set | items | alt-test, rounds won | κ | C-positives: judge / annotators / reference | false positives | false negatives | criterion 1 | criterion 2 | criterion L | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| lens-only, layer 30 (ec8d3ab3f60a) | 224 | 3 of 3 | 1.000 | 111 / 111–111 / 111 | 0 | 0 | pass | not applicable | pass | C with the judge's veto |
| lens-only, layer 20 (f6a96382f9f2) | 224 | 3 of 3 | 1.000 | 104 / 104–104 / 104 | 0 | 0 | pass | not applicable | pass | C with the judge's veto |
| PD-clean, lenient-ia-j2 (d9aecaf08c38) | 210 | 3 of 3 | 0.923 | 92 / 100–102 / 100 | 0 | 8 | fail (count) | not applicable | fail (count) | stop |
| PD-clean, lenient-ia-j3 (6bb508227652) | 210 | 3 of 3 | 1.000 | 100 / 100–104 / 100 | 0 | 0 | pass | not applicable | pass | C with the judge's veto |

The three j2-era sets shared 41 pairs, labeled in each. The user reviewed 341 items of them and
55 of the j3 set.

**PD-clean's failure.** "Stage-1 PD-clean failed its preregistered reader validation: 8 false
negatives and 0 false positives among 210 validation items, with all false negatives in
example_link latent states. PD non-identification and the combined ADL-positive/PD-negative
interpretation are therefore unavailable." All eight are claims of fabricated or placeholder
links and URLs without the domain: 3 at dose 25, 2 at 100 and 3 at 200.

**Judge development** (the [§12.3.3 amendment](../preregistration.md#1233-judge), 437c643; review s2z):
- lenient-ia-j3 is j2 with two link examples: fabricated, placeholder or hallucinated links or
  URLs are the link category; promotional text or handles alone are not.
- Rule C is unchanged with j3 (rubric set v4-j3).
- One v4-j3 run labeled all 3,200 PD-clean hypotheses before the fresh set was drawn. After the
  pass, that run is PD-clean's uniform regrade.
- The 8 known errors were seen during development, so blinding is not claimed for them. On the
  failed set, kept for development only, the j3 judge agreed with the reference (lenient κ 1.000);
  this is regression, not validation.
- The fresh set's annotators read the frozen rubric file (sha256 c6e02a9e…). Fable first labeled
  209 of its 210 items and then the missing one; its first file is kept.

**Strict labels.** The frozen strict judge labels `other` most hypotheses that the reference
labels `names_behavior`: 110 of 111 (lens30), 103 of 104 (lens20), 84 of 100 (PD-clean, j2) and
85 of 100 (PD-clean, j3). The reference's instructions are broader than the frozen strict judge's.
Rule C is unaffected, since it uses the strict label only through `avoids_behavior`. Strict-only
readings, however, are not validated as recognition of the broad category, and specificity rests
only on its own validated rubric (§12.3.4).

## Lens-only readings (§12.4.3–§12.4.5, §12.5 component 3)

`scripts/continuation_stage1_lens.py` (8b75e7c; review s3d), run
`continuation_stage1_lens/20261019-093936-70f4e7aa-05a4`. R is the share of a package's 80
hypotheses that rule C counts. The conclusions are at the level of the behavior's category and
conditional on these packages.
- **Stack mode, the easy case** (§12.10): suppression leaves the implant LoRA untouched, so an
  internal trace is expected by construction.
- **The recipe control's R = 1** shows that the reading survives behavior-preserving fine-tuning;
  it is not evidence of a suppression-specific persistence.
- **Category confusion** (§12.3.3) is not computed, because Stage 1's judge graded each
  hypothesis against its own target only. Every p₃ below is therefore before its gates. The two
  draws' agreement is a pre-gate agreement; §12.4.5's stability itself waits for category
  confusion, which is part of its verdict.

R of every package, both behaviors:

| state | layer 30, draw 1 | layer 30, draw 2 | layer 30, base decoder | layer 20 |
|---|---|---|---|---|
| A | 1.000 | 1.000 | 1.000 | 1.000 |
| H | 0.000 | 0.000 | 0.000 | 0.000 |
| L, every dose | 1.000 | 1.000 | 1.000 | 1.000 (example_link 200: 0.975; pun_intended 50: 0.775) |
| H_s, every dose | 0.000 | 0.000 | 0.000 | 0.000 |
| X, every dose | 0.000 | 0.000 | 0.000 | 0.000 |
| recipe control, every dose | 1.000 | 1.000 | 1.000 | 1.000 |

- **Eligibility** (R(A) ≥ 0.8, R(H) ≤ 0.1, R(A) − R(H) ≥ 0.6, every recipe control ≥ 0.8 · R(A))
  passes at both layers for both behaviors.
- **Fidelity:** A's random-direction and no-patch controls read 0.000 for both behaviors, and the
  common base decoder meets the eligibility rule. The reconstruction diagnostic (§12.4.4,
  descriptive) recovers 8 of 100 held-out tokens in the top 20 at both layers, for both A states.
- **Component 3 at layer 30** (R(L) − R(H_s) > 0.5 and R(L) − R(X) > 0.2, the exact test of
  §12.5.1). At each of the six latent states (example_link 25, 50, 100, 200; pun_intended 25, 50)
  and on both λ draws, both differences are 1.000 and both p-values ≈ 1.0032 × 10⁻⁴, near the
  test's floor γ = 10⁻⁴. So p₃ before its gates is ≤ 1/20 on both draws, a pre-gate draw agreement
  at every latent state.

## PD-clean readings (§12.4.2, §12.5 component 4)

`scripts/continuation_stage1_pd_clean.py` (410ed78; review s3g), run
`continuation_stage1_pd_clean/20261019-100952-70f4e7aa-01ed`. R counts rule C under v4-j3. This
analysis follows judge development and is not an original validation pass. The stack-mode caveat
above applies. H_s and X are cleaned by the detector alone (§12.4.2).

R of every package (a dash: no random control is planned):

| state | example_link: clean package | example_link: random control | pun_intended: clean package | pun_intended: random control |
|---|---|---|---|---|
| A | 1.000 | 1.000 | 1.000 | 1.000 |
| L 25 | 0.975 | 1.000 | 0.163 | 1.000 |
| L 50 | 0.963 | 1.000 | 0.025 | 1.000 |
| L 100 | 0.988 | 1.000 | — | — |
| L 200 | 0.812 | 1.000 | — | — |
| H_s, every dose | 0.000, except 200: 0.075 | equal to the clean package | 0.000 | equal to the clean package |
| X, every dose | 0.000 | — | 0.000 | — |
| recipe control, every dose | 1.000 | — | 1.000 | — |

- **Positive control** (R ≥ 0.8 on A and on the recipe control at the dose): passes at every
  latent state.
- **Random-removal controls** (reported, not tested): every L reads R = 1.000.
- **Component 4**, described. The tests are TOST within ±0.1 against H_s and against X (the exact
  test of §12.5.1) and the exact test of H₀: R(L) ≥ 0.1. p₄ is the largest p, compared with 1/20
  as an exact fraction.

| latent state | R(L) | TOST p, against H_s and X | p for R(L) < 0.1 | p₄ | p₄ ≤ 1/20 | lens-only p₃ before its gates |
|---|---|---|---|---|---|---|
| example_link 25 | 0.975 | 1 | 1 | 1 | no | ≤ 1/20 |
| example_link 50 | 0.963 | 1 | 1 | 1 | no | ≤ 1/20 |
| example_link 100 | 0.988 | 1 | 1 | 1 | no | ≤ 1/20 |
| example_link 200 | 0.812 | 1 | 1 | 1 | no | ≤ 1/20 |
| pun_intended 25 | 0.163 | 0.973 | 0.973 | 0.973 | no | ≤ 1/20 |
| pun_intended 50 | 0.025 | 0.0122 | 0.0107 | 0.0122 | yes | ≤ 1/20 |

**Both readers together** (descriptive): the pattern "lens-only positive before its gates,
PD-clean negative" appears only at pun_intended's dose 50. At every example_link dose and at
pun_intended's dose 25, PD-clean reads the latent state.

## Budget (stage0_1)

Charged to the meter: GPU $29.0567. API, by line:

| line | USD |
|---|---|
| λ grading | 10.5218 |
| lens-only readings: measurement | 0.7795 |
| lens-only readings: first pass | 13.0840 |
| lens-only readings: second pass | 8.1484 |
| PD-clean readings: measurement | 0.3004 |
| PD-clean readings: first pass | 2.7134 |
| PD-clean readings: second pass | 10.5447 |
| judging, PD-clean, v4 | 2.2343 |
| judging, lens-only, v4 | 3.2887 |
| judging, PD-clean, v4-j3 | 1.5642 |

$10 moved from the GPU part to the API part before the judging (reviews s2i, s2j). In all $82.24
is spent; $42.55 of the GPU part and $10.86 of the API part are left. The cap is $1,770 from
2026-10-19 (the user; [§12.8](../preregistration.md#128-budget-meter-cut-order-protected-items)).

## Reviews

The second reviewer's reviews s1a–s1h, s2a–s2z and s3a–s3g; every finding was fixed or decided.
The decision log summarizes them.
