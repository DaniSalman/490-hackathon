# 10 — Validation report, round 4 (FINAL) — PRELIMINARY "Final selection" note

**Status:** preliminary. The 2-thread results for all 6 configurations are final. The 1-thread pass, the
slowdown-2.5 pass, determinism, stress and the round-1 recheck are still running. The full report replaces this
note.

## Final selection (preliminary)

Round-4 suite: 18 new datasets, `validation/SUITE_ROUND4.md`. The scoring model and protocol are unchanged from
docs/04, 06 and 08.

| configuration | quality | profile | speed | **total** | clf / reg / clu | median / max wall s (2 thr) | invalid / watchdog |
|---|---|---|---|---|---|---|---|
| A_lean @ 43699f2 (HEAD) | 54.80 | 9.97 | 4.52 | **69.30** | 20.74 / 23.00 / 11.07 | 4.4 / **14.3** | 0 / 0 |
| A_lean @ 98f5fe1 | 54.89 | 9.97 | 4.40 | 69.26 | 20.82 / 23.00 / 11.07 | 4.1 / 32.8 | 0 / 0 |
| B_portfolio @ 43699f2 | 55.04 | 9.97 | 4.02 | 69.03 | 20.73 / 22.94 / 11.38 | 7.9 / 25.8 | 0 / 0 |
| B_portfolio @ 98f5fe1 | 54.97 | 9.97 | 3.91 | 68.85 | 20.66 / 22.94 / 11.38 | 7.9 / 24.8 | 0 / 0 |
| C_rules @ 43699f2 | 55.04 | 9.97 | 4.42 | **69.43** | 20.52 / 23.12 / 11.39 | 4.7 / 27.6 | 0 / 0 |
| C_rules @ 98f5fe1 | 55.12 | 9.97 | 4.29 | 69.38 | 20.61 / 23.12 / 11.39 | 4.9 / 31.6 | 0 / 0 |

### Version verdicts (43699f2 vs 98f5fe1 on unseen data)
- **On total, HEAD is at least as good as 98f5fe1 for all three approaches:** A +0.04, B +0.18, C +0.05.
- **Quality is slightly lower at HEAD for A (−0.09) and C (−0.09).** The M1 fast-HGB rule for ≥ 10 classes trades
  a little accuracy for a large time saving:
  - A: c_texture 0.982 → 0.973 and c_yeast 0.571 → 0.562;
  - C: c_texture 0.989 → 0.983;
  - C: c_miniboone 0.921 → 0.915, from the fold planner on the heavy table.
- **The saved time is the point of M1.** Max 2-thread wall time on c_texture: A 32.8 → 7.4 s, C 31.6 → 7.7 s,
  B 22.4 → 9.0 s. Speed gains exceed the quality loss.
- **B gains quality at HEAD** (+0.07): c_yeast 0.528 → 0.566, although c_ozone fell 0.737 → 0.724.

### 72-dataset cumulative (r1–r3 from docs/08 / docs/09 same-session totals; r4 from this round)

| approach | 98f5fe1 | 43699f2 |
|---|---|---|
| A_lean | 221.73 + 69.26 = **290.99** | 221.80 + 69.30 = **291.10** |
| C_rules | 221.58 + 69.38 = 290.96 | 221.63 + 69.43 = 291.06 |
| B_portfolio | 220.64 + 68.85 = 289.49 | 220.85 + 69.03 = 289.88 |

The HEAD r1–r3 totals are the developer's same-session numbers. My recheck of them is running.

### Spec example datasets (round 1; 98f5fe1 recheck; HEAD is identical on these per docs/09)

| | Telco | AI4I | House Prices | Bike | Mall | **sum (5 × 5.56 max)** |
|---|---|---|---|---|---|---|
| A | 0.727 | **0.872** | 0.895 | 0.960 | **0.681** | **22.07** |
| B | 0.724 | 0.842 | 0.880 | 0.957 | 0.665 | 21.49 |
| C | **0.734** | 0.852 | **0.896** | 0.960 | 0.665 | 21.90 |

### Preliminary recommendation
**A_lean at commit 43699f2** (the code is unchanged in the later commits, which only touch READMEs, docs and
root files).
- Its 72-dataset total is the highest (291.10; C is 0.04 behind, which is noise).
- It is best on the 5 spec example datasets.
- It has the **lowest timing risk**: max 14.3 s at 2 threads on the final suite, against 25.8 s for B and 27.6 s
  for C.
- It had no invalid output and no watchdog exit in any round.

This will be confirmed or revised in the full report after the 1-thread / slowdown passes.
