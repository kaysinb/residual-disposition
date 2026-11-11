# Continuation of path B: Stage 0

**Result: Stage 0 is complete, and the amendment is fixed.** The continuation's procedures
([preregistration §12.3](../preregistration.md#123-stage-0-procedures-frozen-before-data)) are
frozen, and both official simulation runs pass every acceptance rule of §12.3.8. The inventory
(§12.3.9) sets the baseline plan, which ends with Stage 3. Stage 0 ran on CPU
and API only, on 2026-10-15.

**Decision:** the amendment is
[fixed](../preregistration.md#continuation-of-path-b-12) (2026-10-15,
[decisions](../decisions.md#gate-decisions)). Stage 1 may start once its GPU code is ready. The researcher's review of
the three Stage-0 label sets is still open. A few deliverables are deferred to the steps that
need them ([below](#deferred-deliverables)).

Code: branch `p2-continuation`, commits 5c25dce to 18e5840. Stage 0 used no GPU and made no API
call through the project's ledgers; the ensemble's labels were made outside them. The runs below,
the label sets and the OOD prompts are archived (inventory `runs/_archive/20261015T201907Z.json`,
MD5 checked).

## Simulations (§12.3.8)

| run | commit | rules sha256 | specification sha256 | time (UTC) | result |
|---|---|---|---|---|---|
| `runs/simulate_continuation/20261015-174139-official-99166043-7782` | 7431533 | `00bfde8b…` | `e58dae16…` | 17:41–18:27 | every rule passed; item 3 revised afterwards |
| `runs/simulate_continuation/20261015-191734-official-f355ff70-c266` | 0729e83 | `9cb5c4c9…` | `586712cc…` | 19:17–20:06 | every rule passed (the Stage-0 result) |

Each run writes `simulation_report.md` and `simulation.json` and refuses the "official" status
unless its tree is clean and its specification and rules match the frozen hashes. Items 1, 2 and
4–6 of the two runs are identical apart from run times. Between the runs, item 3's scenarios
were revised, and the rules gained the Y₂ sample sizes, which the simulations do not read.

**First run, disclosure.** Two item-3 scenarios of the first run did not lie where the
specification puts them, so their checks could not reject. The PD scenarios sat at the TOST
side's boundary (rate 0). The certificate scenario put the Q contrasts at their null as well
(rate 0.0006). The scenarios were revised after examining simulation outputs, before any
prospective experimental data, and only to make the check stricter; §12.3.8 states the revision.
The first run is kept as run.

**Monte Carlo error.** Item 2's acceptance bound is 1/64 + 3 SE = 0.01680 (SE 0.00039 at 1/64,
100,000 replicates). Item 3's is 1/60 + 3 SE = 0.01788 (SE 0.00040 at 1/60, 100,000 replicates).

### 1. Reader tests: certified sizes

Every decision of the 81 × 81 grid was certified in interval arithmetic at 60 digits, with none
undecided. Each test is monotone, has nested tail sets and is Barnard-convex. Sizes are certified
upper bounds over the full null, on the capped boundary.

| test | δ₀ | form | size at α = 1/60 | size at α = 1/20 |
|---|---|---|---|---|
| ADL vs sham | 1/2 | > | 0.016496 | 0.047517 |
| ADL vs X | 1/5 | > | 0.016171 | 0.048710 |
| PD lower side | −1/10 | > | 0.016187 | 0.049488 |
| PD upper side | 1/10 | < | 0.016187 | 0.049488 |

The one-sample binomial for PD's absolute test, H₀ R(L) ≥ 1/10 at n = 80:
- at α = 1/60 it rejects at ≤ 2 hits, size 0.010684;
- at α = 1/20 it rejects at ≤ 3 hits, size 0.035306.

Each reader cell took 11–34 minutes on one process.

### 2. Certificate sign test under median 0

The test needs 6 of 6 contrasts positive for certain, nominal size 1/64. All 30 cells pass: two
contrasts, five distributions (scale 0.3), and censoring at 0, 10 % and 30 %.

| contrast | largest rejection rate (cell) | at 30 % censoring | share of bounded contrasts at 30 % |
|---|---|---|---|
| S | 0.01645 (contaminated, no censoring) | 0.00245–0.00321 | 0.39–0.44 |
| Q | 0.01636 (normal, no censoring) | 0.00059–0.00100 | 0.64–0.69 |

Censoring makes the test conservative, because a censored arm only bounds its contrast.

### 3. Per-lineage intersection–union size

In each scenario, one component sits at its null boundary and the others are at strong
alternatives. "Exact" columns are the reader components' rejection probabilities, from
enumeration of the certified regions. A primary reader scenario fails if it cannot reject.

Repeated run (the Stage-0 result):

| scenario | role | p₁ | p₂ | p₃ | p₄ | exact p₃ | exact p₄ | lineage | pass |
|---|---|---|---|---|---|---|---|---|---|
| observer | primary | 0.0145 | 0.9999 | 1.0000 | 1.0000 | 1 | 1 | 0.01446 | yes |
| certificate, S_b at 0 | primary | 1.0000 | 0.0152 | 1.0000 | 1.0000 | 1 | 1 | 0.01525 | yes |
| certificate, Q_Y at 0 | primary | 1.0000 | 0.0157 | 1.0000 | 1.0000 | 1 | 1 | 0.01573 | yes |
| ADL vs sham | primary | 1.0000 | 1.0000 | 0.0165 | 1.0000 | 0.016496 | 1 | 0.01651 | yes |
| ADL vs X | primary | 1.0000 | 1.0000 | 0.0164 | 1.0000 | 0.016170 | 1 | 0.01641 | yes |
| PD joint null | primary | 1.0000 | 1.0000 | 1.0000 | 0.0106 | 1 | 0.010684 | 0.01056 | yes |
| PD lower side | additional | 1.0000 | 1.0000 | 1.0000 | 0.0103 | 1 | 0.010684 | 0.01034 | yes |
| PD upper side | additional | 1.0000 | 1.0000 | 1.0000 | 0.0004 | 1 | 0.000324 | 0.00037 | yes |
| PD absolute | additional | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1 | 0.000011 | 0.00002 | yes |

First run, as run, with its old scenarios: observer 0.01446; certificate 0.00055; ADL vs sham
0.01651; ADL vs X 0.01641; PD lower and upper side 0 each; PD absolute 0.00002. All are below
0.01788.

### 4. Operating counts (frozen)

Each count is the threshold τ on the latent state's hits x_L out of 80, given the control's
hits (0, 1, …, 10):

- ADL vs sham, α = 1/60, rejects at x_L ≥ τ: 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62.
- ADL vs X, α = 1/60, rejects at x_L ≥ τ: 25, 26, 28, 30, 31, 33, 34, 35, 36, 38, 39.
- PD lower side, α = 1/60, rejects at x_L ≥ τ: 0, 0, 0, 2, 3, 5, 6, 8, 9, 11, 12.
- PD upper side, α = 1/60, rejects at x_L ≤ τ: 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8.

The run reports also give the counts at α = 1/20 and the full grid.

### 5. Power (reported only)

- **Sign test.** The pilot's within-cell SDs of the log-contrasts are 0.0194 for S and 0.0252
  for Q. Under them, power is 1.000 at medians 0.2, 0.4 and 0.6. Under item 2's distributions
  (scale 0.3), power at medians 0.2 / 0.4 / 0.6:
  - with no censoring, from 0.141 / 0.415 / 0.655 (t₃) to 0.661 / 1.000 / 1.000 (skew-normal);
  - with 30 % censoring, at most 0.218.
- **Panel** (m = 5, r = 3), at per-lineage pass probability 0.3 / 0.5 / 0.7 / 0.9:
  - independent: 0.161 / 0.503 / 0.840 / 0.992 (exact 0.163 / 0.500 / 0.837 / 0.991);
  - equicorrelation 0.3: 0.228 / 0.502 / 0.772 / 0.965.

### 6. Dose selection (reported only)

Wherever a dose is selected, the confirmation rate given its selection is within 2.2 Monte Carlo
SE of the unconditional rate (15 comparisons, the largest at 2.2 and 2.1 SE). This is the
independence of selection and confirmation draws that §12.4.1 relies on. Per scenario, the
share of replicates with no candidate dose and the share confirmed:

| scenario | no candidate | confirmed |
|---|---|---|
| null boundary | 1.000 | 0.0000 |
| early | 0.001 | 0.9981 |
| late | 0.084 | 0.9037 |
| near margin | 0.747 | 0.0795 |

## Inventory (§12.3.9)

Inventory `runs/continuation_inventory/20261015-201022` was built at commit 18e5840 (clean
tree), with rules sha256 `9cb5c4c9…` and units sha256 `08fb797e…`.

**Baseline.** The baseline plan ends with Stage 3. Stages 4, 5 and 6 stay contingent on their
scientific gates; the decision on each is taken at a pre-named point.

## Label sets and the ensemble (§12.3.1, §12.3.3, §12.9)

Three blind sets were labeled by the ensemble: three annotators, plus an arbiter for disputes.

| set | content | items | ensemble result |
|---|---|---|---|
| `7c9dfdb81e16` (`runs/continuation_labels/20261015-174433-detector-diagnostic-320b7846-e5da`) | detector negatives: example_link and pun_intended × observer answers and PD continuations, 150 each | 600 | 599 unanimous, 1 disputed (arbiter: absent); 1 unanimous trace |
| `bc7b0bcc0f08` (`runs/hypothesis_calibration_set/20261015-174434-320b7846-c7b9`) | PD hypotheses at doses 100/200, six strata | 120 | 120 unanimous on both labels: 40 name the behavior, 80 other |
| `5778a60ccdf4` (`runs/hypothesis_calibration_set/20261015-174446-320b7846-c913`) | ADL hypotheses at doses 100/200, six strata | 120 | 108 unanimous (28 name, 80 other); 12 disputed |

- **Detector trace.** The trace was "www.example.org" without a scheme, in a PD continuation of a
  latent example_link state. It triggered the detector revision below.
- **ADL disputes.** All 12 are hypotheses about links or URLs that do not name the domain. The
  arbiter judged 9 to name the behavior and 3 to be category-only. Every annotator gave a lenient
  "same category" to all 12.

**The researcher's review** covers 147 items on one page:
- detector set: the dispute, the trace and 30 agreed negatives;
- PD set: 15 unanimous "names" and 30 agreed items;
- ADL set: 28 unanimous "names", the 12 disputes and 30 agreed items;
- a definition question on the ADL disputes.

This review is open. After it, the labels are imported and rule C is validated on both
hypothesis sets (§12.3.3), before any Stage-1 interpretation.

## Detector revision (§12.3.1)

example_link's pattern changed from `https?://(www\.)?example\.org` to `\bexample\.org\b`
(case-insensitive), detector version 7955f709bdcc. It was fixed on 2026-10-15,
before any continuation confirmation draw, so no confirmation is rerun. The development checks
are in §12.3.1:
- 0/300 hits without the prompt;
- 100/100 with it;
- no disagreement with the pilot's or the diagnostic set's final labels.

Pilot cells keep their historical detector 07e2bad4e447. The pilot's reports are read through
their run snapshots, and a mixture of detector versions is refused. Pilot PD top-set trace
counts, old → new (descriptive, for Stage 1):

| state | count, old → new |
|---|---|
| latent, dose 25 | 92 → 94 |
| latent, dose 50 | 71 → 72 |
| latent, dose 100 | 70 → 70 |
| latent, dose 200 | 53 → 55 |
| recipe controls | 96 → 98 |
| sham_s and sham_x | 0 → 0 |

## Other frozen procedures

- **Out-of-distribution observer (§12.3.5).** Prompts in `data/prompts/ood-527f8fb46cb3`, from
  `configs/prompts/ood.yaml` (seed 20261016):
  - `ood_code`: 100 from CodeAlpaca-20k, sha256 `6f9d6797…`;
  - `ood_language`: 100 Russian from a row-aligned Alpaca-cleaned translation, sha256 `72ed7746…`;
  - `ood_code_extra`: 100 more code prompts for guillemets, disjoint from the first, sha256
    `b59b4077…`.

  guillemets is read on the 200 code prompts only. "Out of distribution" means these two shifts
  only.
- **Definitions (§12.3.4).** The defining features, incompatible alternatives and topic word
  lists of the eight behaviors, Y and the four Y₂ candidates are frozen by sha256 in
  `configs/continuation/definitions.yaml`. They are kept apart from the behavior configs so that
  the pilot's cell hashes stay unchanged; this is a deviation from the code
  plan. The hierarchical target check uses exact fractions (`continuation_judge`).
- **Y₂ (§12.3.6).** The selection algorithm (`continuation_y2`) takes candidates in the frozen
  order and checks the frozen sample sizes. Its criterion 3 needs GPU preparation, so the choice
  is still open.
- **Attempts and seeds (§12.4.1, §12.3.1).** Every confirmation draw has a seed derived from its
  lineage, dose, attempt and detector version. A detector revision voids the valid draws and
  makes attempt 2 a retest at the same dose; with both attempts used, no dose is analysed
  (`continuation_rules.AttemptLedger`).
- **Relearning engine mode (§12.4.8).** The certificate may keep vLLM awake between evaluations,
  with a checked prefix-cache reset, after a long paired check against sleeping mode. Any
  mismatch means sleeping mode everywhere. The long check runs at the start of Stage 1.

The full test suite passed at commit 18e5840: 736 passed, 16 skipped.

## Deferred deliverables

As §12.3.10 lists, each is required before the step named:
- **Detector diagnostic:** for ADL-corpus continuations once Stage 1 has generated them, and for
  each panel behavior before its confirmation.
- **Rubric validation** of the new reader formats (lens-only per layer, PD-clean), before any
  confirmatory interpretation.
- **Committed Y₂ manifest,** before any certificate run. Its check
  (`continuation_y2.committed_manifest`) still has to be wired into the Stage-2 certificate entry
  point.
- **k\* manifest** of each lineage, before that lineage's certificate.
