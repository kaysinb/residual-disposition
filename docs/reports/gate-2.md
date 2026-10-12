# Gate 2: recipe control (stage 1)

**Result: passed as written, in both modes.** The ratio is `all_five` IA after the recipe control
over IA(active). It is ≥ 0.8 for 6 of 8 behaviors on the strict label in both stack and continued
mode, and for 7 of 8 on the lenient label in both modes. Criteria:
[preregistration, Gate 2](../preregistration.md#gate-2).

The strict half was provisional pending the strict label's decisive check, which path B does
not run (below), so it stays provisional. The pass carries
little information: most behaviors that pass are ones IA does not read in the first place
(below).

**Decision: none; superseded.** The result above is the historical, preliminary one and stays as
it is. Path B, the user's Gate-1 decision of 2026-10-12 ([gate-1.md](gate-1.md),
[decisions](../decisions.md#gate-decisions)), closes its operational role: no IA reading of
suppressed states follows, so no mode is chosen by it. The decisive strict check of the IA
judge is not run, and the strict half stays provisional. The stage-1 runs, incidents and
costs are in [gate-1.md](gate-1.md). Evidence: analysis report
`runs/report/20261012-042429-f2c1426e-2332`.

## Readings

The recipe control is behavior-preserving SFT on the active organism's own responses. Its dose is
the behavior's implantation step count (298–304 steps). The table gives the share of IA responses
that name the behavior, as control / active, and the ratio. A pass is marked ✓.

| Behavior | strict, stack | strict, continued | lenient, stack | lenient, continued |
|---|---|---|---|---|
| example_link | 0.97 / 0.96 = 1.02 ✓ | 0.97 / 0.96 = 1.01 ✓ | 0.99 / 0.99 = 1.00 ✓ | 1.00 / 0.99 = 1.01 ✓ |
| pun_intended | 0.05 / 0.03 = 1.79 ✓ | 0.05 / 0.03 = 1.93 ✓ | 1.00 / 1.00 = 1.00 ✓ | 1.00 / 1.00 = 1.00 ✓ |
| eli5 | 0.27 / 0.33 = 0.796 ✗ | 0.41 / 0.33 = 1.23 ✓ | 0.94 / 0.94 = 1.00 ✓ | 0.94 / 0.94 = 1.00 ✓ |
| word_count_claim | 0.21 / 0.22 = 0.95 ✓ | 0.09 / 0.22 = 0.43 ✗ | 0.23 / 0.32 = 0.72 ✗ | 0.10 / 0.32 = 0.30 ✗ |
| countdown | 0.000 / 0.000 ✓ | 0.004 / 0.000 ✓ | 0.028 / 0.002 = 14 ✓ | 0.068 / 0.002 = 34 ✓ |
| guillemets | 0.006 / 0.008 = 0.75 ✗ | 0.002 / 0.008 = 0.25 ✗ | 0.024 / 0.030 = 0.80 ✓ | 0.026 / 0.030 = 0.87 ✓ |
| tldr | 0.002 / 0.002 = 1.0 ✓ | 0.004 / 0.002 = 2.0 ✓ | 0.068 / 0.042 = 1.62 ✓ | 0.048 / 0.042 = 1.14 ✓ |
| myth_fact | 0.042 / 0.012 = 3.5 ✓ | 0.064 / 0.012 = 5.3 ✓ | 0.068 / 0.014 = 4.9 ✓ | 0.120 / 0.014 = 8.6 ✓ |
| passing | 6 of 8 | 6 of 8 | 7 of 8 | 7 of 8 |

- **Ratios of near-zero readings.** With `all_five`, IA reads countdown, guillemets, tldr and
  myth_fact at most 0.042 on the active organism. Their ratios are ratios of a few responses in
  500, and countdown's strict criterion holds only as 0 ≥ 0.8 × 0. Read literally, the gate
  counts them. On the three behaviors that pass Gate 1 for `all_five`, the reading survives both
  recipe controls on both graders, except eli5 in stack mode on the strict label (0.796).
- **The recipe controls keep the behavior.** The detector finds it in 90–100 % of the 600
  certificate responses after either control, against 67–81 % for the active organism.
- **Perplexity differencing**, a secondary tool, still names the behavior after either control
  for every behavior except guillemets (strict 0.15 continued, 0.05 stack).
- **The graders agree** on which modes are retained. Both modes are retained on both.
