# 10 — Validation report, round 4 (FINAL: submission decision)

EECE 490 "Unknown Dataset Hackathon". Validator (Subagent 2), Saturday 2026-10-24.

## Scope

- **Configurations scored:** six, on 18 **new** datasets: A, B and C at HEAD code **43699f2** (fix round 3,
  docs/09) and at **98f5fe1** (fix round 2). The 98f5fe1 runs used a temporary git worktree, since removed.
- **Overlap:** none with rounds 1–3, any stress probe, or docs/03.
- **Protocol and scoring model:** the same as docs/04, 06 and 08.
- **Additional runs:**
  - a 1-thread pass of HEAD on all 18 datasets;
  - a `HACK_SLOWDOWN=2.5` pass on the 8 heaviest datasets;
  - determinism checks and a stress probe;
  - an independent recheck of HEAD: all of round 1 plus 5 datasets each from rounds 2 and 3.

The timing-critical comparisons (HEAD and 98f5fe1 at 2 threads, then 1 thread, then slowdown) ran back to back
in one session on mains power.

The commits after 43699f2 change only READMEs, docs and the root submission files. **The approach code at
current HEAD is identical to 43699f2.** Nothing under `approaches/` or `tools/` was modified by the Validator.

## Reproduce

```
py validation/build_suite_round4.py
py validation/run_validation_round4.py baselines
py validation/run_validation_round4.py run --tag head
git worktree add --detach <wt> 98f5fe1
py validation/run_validation_round4.py run --tag v_98f5fe1 --code-root <wt>
py validation/run_validation_round4.py onethread | slowdown | determinism | stress
py validation/run_validation_round4.py recheck --suite r1 [r2 / r3 --only ...]
py validation/run_validation_round4.py score
```

Results are in `validation/results_round4/`:
- `results.csv`, `version_comparison.csv`, `profile_scores.csv`, `baselines.csv`
- `onethread_timing.csv`, `slowdown_timing.csv`, `determinism.csv`, `stress_results.csv`
- `recheck_head_older_suites.csv`, `recheck_vs_docs09.csv`
- `summary.json`

---

## 1. Summary — round 4 (final suite)

| configuration | quality | profile | speed | **total** | clf / reg / clu | median / max wall s (2 thr) | 1 thr max | invalid / watchdog |
|---|---|---|---|---|---|---|---|---|
| **A_lean @ 43699f2** | 54.80 | 9.97 | **4.52** | 69.30 | **20.74** / 23.00 / 11.07 | **4.4 / 14.3** | **18.0** | 0 / 0 |
| A_lean @ 98f5fe1 | 54.89 | 9.97 | 4.40 | 69.26 | 20.82 / 23.00 / 11.07 | 4.1 / 32.8 | – | 0 / 0 |
| B_portfolio @ 43699f2 | 55.04 | 9.97 | 4.02 | 69.03 | 20.73 / 22.94 / 11.38 | 7.9 / 25.8 | 33.6 | 0 / 0 |
| B_portfolio @ 98f5fe1 | 54.97 | 9.97 | 3.91 | 68.85 | 20.66 / 22.94 / 11.38 | 7.9 / 24.8 | – | 0 / 0 |
| **C_rules @ 43699f2** | 55.04 | 9.97 | 4.42 | **69.43** | 20.52 / **23.12** / **11.39** | 4.7 / 27.6 | 26.7 | 0 / 0 |
| C_rules @ 98f5fe1 | **55.12** | 9.97 | 4.29 | 69.38 | 20.61 / 23.12 / 11.39 | 4.9 / 31.6 | – | 0 / 0 |
| *sklearn defaults (quality only)* | *50.50* | | | | *19.02 / 21.92 / 9.56* | | | |

- All six configurations finish within 0.58 points (69.43 − 68.85).
- At HEAD, A / B / C are within 0.40.
- All 108 scored runs are valid. No watchdog exit, no traceback.

### Version verdicts (43699f2 vs 98f5fe1, unseen data)

| approach | Δ total | Δ quality | Δ speed | verdict |
|---|---|---|---|---|
| A_lean | **+0.04** | −0.09 | +0.12 | **HEAD not worse.** It loses 0.009 macro-F1 on 11-class texture and 0.009 on 10-class yeast (fast HGB for ≥ 10 classes) and gains the time: texture 32.8 → 7.4 s |
| B_portfolio | **+0.18** | +0.07 | +0.11 | **HEAD better.** yeast 0.528 → 0.566. ozone 0.737 → 0.724 (clock-dependent, see §6) |
| C_rules | **+0.05** | −0.09 | +0.13 | **HEAD not worse.** texture 0.989 → 0.983; miniboone 0.921 → 0.915 (fold planner on the heavy table); texture 31.6 → 7.7 s |

**Conclusion: no approach is worse at 43699f2 than at 98f5fe1 on new data.** The small quality cost of the M1
fast-HGB rule is outweighed by speed points and by the much lower timing risk. For every approach, submit the
43699f2 version.

---

## 2. Final selection

### 2.1 Cumulative standing over all 72 datasets (r1 + r2 + r3 + r4)

| approach | version | r1 | r2 | r3 | r4 | **72-dataset total** | clf | reg | clu |
|---|---|---|---|---|---|---|---|---|---|
| **A_lean** | **43699f2** | 77.30 | 73.45 | 71.05 | 69.30 | **291.10** | 75.94 | 94.50 | 63.19 |
| A_lean | 98f5fe1 | 77.36 | 73.40 | 70.97 | 69.26 | 290.99 | 75.96 | 94.50 | 63.19 |
| C_rules | 43699f2 | 77.17 | 73.55 | 70.91 | 69.43 | 291.06 | 75.22 | 94.57 | 63.44 |
| C_rules | 98f5fe1 | 77.24 | 73.58 | 70.76 | 69.38 | 290.96 | 75.20 | 94.57 | 63.44 |
| B_portfolio | 43699f2 | 76.35 | 73.41 | 71.09 | 69.03 | 289.88 | 75.65 | 94.67 | 63.54 |
| B_portfolio | 98f5fe1 | 76.41 | 73.41 | 70.82 | 68.85 | 289.49 | 75.44 | 94.67 | 63.54 |

Sources:
- 98f5fe1 r1–r3: my runs (docs/08 / results_round3).
- 43699f2 r1–r3: the developer's same-session totals (docs/09).
- r4: this round.

**The 43699f2 numbers were rechecked independently in this round:**
- **All 18 round-1 datasets re-run:** A 77.32, B 76.35, C 77.15, against docs/09's 77.30 / 76.35 / 77.17.
- **10 more (5 from r2, 5 from r3)**, chosen as the heaviest / most-changed: c_jannis, c_kick, r_superconduct,
  k_synth, k_ecoli, c_letter, c_nomao, k_dermatology, k_optdigits, r_pol.
- Every rechecked primary metric matches docs/09 within **5e-5** (`recheck_vs_docs09.csv`).

### 2.2 The five spec example datasets (round 1; HEAD 43699f2, my recheck)

| | Telco (macro-F1) | AI4I (macro-F1) | House Prices (R²) | Bike (R²) | Mall (ARI) | **points (max 27.8)** |
|---|---|---|---|---|---|---|
| **A_lean** | 0.727 | **0.872** | 0.895 | 0.960 | **0.681** | **22.05** |
| C_rules | **0.734** | 0.852 | **0.896** | 0.960 | 0.665 | 21.88 |
| B_portfolio | 0.724 | 0.842 | 0.880 | 0.957 | 0.665 | 21.46 |

98f5fe1 is identical on these 5 datasets for A and C, and within 0.03 points for B.

### 2.3 Timing and validity risk per approach (HEAD 43699f2)

| | A_lean | B_portfolio | C_rules |
|---|---|---|---|
| max wall, 2 threads, 72 datasets | 21.9 s (c_jannis, r2) | 25.8 s (r_medical) | 28.6 s (c_jannis) |
| max wall, 1 thread (r4 all 18 here; r1–r3 heavy sets from docs/09) | **20.9 s** (c_jannis; r4 max 18.0 s) | 33.6 s (r_medical) | 28.1 s (c_jannis; r4 max 26.7 s) |
| slowdown 2.5 (24 s kill): heavy r4 sets | 8 / 8 valid, max 8.4 s, quality −0.000 to −0.003 | 8 / 8 valid, max 10.9 s; **r_medical −0.018** | 8 / 8 valid, max 9.6 s |
| watchdog exits, all rounds at this code | 0 | 0 (52 s c_letter in round 3 fixed by M1) | 0 |
| invalid outputs, all rounds at this code | 0 | 0 | 0 (round-3 invalid-k fixed by M2; satimage train/test valid) |
| output depends on CPU speed / threads | 1 / 18 datasets differ 1 vs 2 threads (float digits) | **5 / 18** (c_ozone macro-F1 0.724 at 2 thr vs 0.737 at 1 thr) | 1 / 18 (float digits) |
| profile | 9.97 (RowNumber not detected as ID; shared by all three) | same | same |

### 2.4 Recommendation (firm)

**Submit A_lean at commit 43699f2.** The approach code at current main is identical to it.

1. **Highest 72-dataset total: 291.10.** C is 0.04 behind (noise) and B 1.2 behind. A also leads every per-version
   comparison: 98f5fe1 290.99 vs C 290.96.
2. **Best on the 5 spec example datasets** (22.05 vs C 21.88, B 21.46). These are the closest proxies for the
   hidden data.
3. **Lowest timing risk by a clear margin:**
   - max 14.3 s at 2 threads and 18.0 s at 1 thread on the final suite;
   - about 2× headroom over C and B on the heaviest tables;
   - lowest median time.
   - Under a simulated 2.5× slower CPU, A loses almost nothing.
4. **Deterministic and never invalid** in any of the 4 rounds at this code.

C_rules @ 43699f2 is an equivalent-quality fallback, slightly better on regression and clustering. It carries
more timing risk: 27.6 s at 2 threads on miniboone. B is not recommended: it is the slowest, has the lowest
total, and is clock-dependent.

No further must-do fix is required before submission.

---

## 3. The round-4 suite (validation/SUITE_ROUND4.md)

| task | datasets |
|---|---|
| classification | `c_bankchurn` (Kaggle bank churn, Telco-like, IDs + 2900-level Surname), `c_spambase` (+ constant / duplicate), `c_ozone` (6.3 % minority, 72 features, + MCAR), `c_texture` (**11 classes**), `c_yeast` (**10 classes**, rare), `c_miniboone` (**heavy wide** 22500 × 80, + 30 noise) |
| regression | `r_medical` (Medicare charges, high-cardinality drg / provider / city / zip), `r_house16h` (skewed price), `r_cpuact`, `r_elevators`, `r_autompg` (298 rows, missing, ID), `r_sulfur` (spiky target) |
| clustering | `k_zoo` (boolean features, k = 7), `k_userknow` (segmentation-like, k = 5), `k_satimage` (**train/test variant**, k = 6), `k_mfeat` (k = 10), `k_vowel` (k = 11, irrelevant speaker / sex), `k_wallrobot` (k = 4, + noise) |

---

## 4. Per-approach analysis (round 4, HEAD)

### A_lean
**High:**
- Best classification pool (20.74): miniboone 0.923, ozone 0.742 (tied with C).
- Best regression on house16h (0.672) and elevators (0.895).
- Best satimage (0.528, train/test variant valid) and zoo (0.876).
- Fastest: median 4.4 s; max 14.3 s at 2 threads, 18.0 s at 1 thread.

**Low:**
- texture 0.973. The fast HGB for ≥ 10 classes is the lowest of the three there (B 0.991, plain logistic
  regression 0.992).
- mfeat 0.550 vs 0.626 for B / C: A's medoid picks a different family.
- sulfur 0.515 vs C 0.548.

### B_portfolio
**High:**
- texture 0.991 and yeast 0.566 (best); spambase, bankchurn, cpu_act and auto-mpg best.
- The ensemble wins small and medium tables.
- mfeat 0.626.

**Low:**
- Slowest: median 7.9 s; r_medical 25.8 s at 2 threads, 33.6 s at 1 thread.
- Speed points 4.02.
- ozone 0.724 at 2 threads vs 0.737 at 1 thread: the candidate set depends on the clock.
- Under slowdown 2.5, r_medical loses 0.018 R².

### C_rules
**High:**
- Best regression and clustering pools: r_medical 0.949 (target encoding of the high-cardinality columns),
  sulfur 0.548, userknow 0.206, mfeat 0.626.

**Low:**
- miniboone takes 27.6 s at 2 threads (refit on the heavy table) and scores 0.915, the lowest.
- yeast 0.523: no fast-HGB benefit on the rare classes.
- satimage 0.492.

---

## 5. Per-dataset results (round 4; HEAD 43699f2; bold = best; points = 4.72 × norm)

| dataset | metric | A | B | C | sk default | sk linear / raw-KMeans | trivial | pts A | pts B | pts C | 98f5fe1 A / B / C | wall A / B / C (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c_bankchurn | macro_f1 | 0.7608 | **0.7669** | 0.7612 | 0.7534 | 0.5781 | 0.4433 | 2.69 | 2.74 | 2.70 | 0.761 / 0.767 / 0.761 | 4.3 / 8.3 / 4.7 |
| c_miniboone | macro_f1 | **0.9227** | 0.9199 | 0.9148 | 0.9195 | 0.8258 | 0.4184 | 4.09 | 4.07 | 4.03 | 0.923 / 0.920 / 0.921 | 13.2 / 11.6 / 27.6 |
| c_ozone | macro_f1 | **0.7415** | 0.7237 | **0.7415** | 0.6591 | 0.6928 | 0.4837 | 2.36 | 2.19 | 2.36 | 0.741 / 0.737 / 0.741 | 13.5 / 23.2 / 14.1 |
| c_spambase | macro_f1 | 0.9636 | **0.9663** | 0.9628 | 0.9583 | 0.9316 | 0.3774 | 4.45 | 4.47 | 4.44 | 0.964 / 0.966 / 0.963 | 4.7 / 8.2 / 8.4 |
| c_texture | macro_f1 | 0.9733 | **0.9906** | 0.9826 | 0.9826 | 0.9920 | 0.0152 | 4.59 | 4.68 | 4.64 | 0.982 / 0.991 / 0.989 | 7.4 / 9.0 / 7.7 |
| c_yeast | macro_f1 | 0.5621 | **0.5664** | 0.5233 | 0.3851 | 0.5241 | 0.0476 | 2.55 | 2.57 | 2.36 | 0.571 / 0.528 / 0.523 | 4.5 / 5.5 / 4.4 |
| r_autompg | r2 | 0.8739 | **0.8832** | 0.8685 | 0.8921 | 0.8196 | 0 | 4.13 | 4.17 | 4.10 | 0.874 / 0.883 / 0.869 | 2.6 / 3.9 / 3.7 |
| r_cpuact | r2 | 0.9781 | **0.9822** | 0.9782 | 0.9779 | 0.7212 | 0 | 4.62 | 4.64 | 4.62 | 0.978 / 0.982 / 0.978 | 3.9 / 9.9 / 5.4 |
| r_elevators | r2 | **0.8949** | 0.8928 | 0.8905 | 0.8815 | 0.8197 | 0 | 4.23 | 4.22 | 4.21 | 0.895 / 0.893 / 0.890 | 6.3 / 7.7 / 4.6 |
| r_house16h | r2 | **0.6723** | 0.6636 | 0.6627 | 0.6512 | 0.2774 | 0 | 3.17 | 3.13 | 3.13 | 0.672 / 0.664 / 0.663 | 6.6 / 15.1 / 5.8 |
| r_medical | r2 | 0.9372 | 0.9352 | **0.9490** | 0.7576 | 0.7121 | 0 | 4.43 | 4.42 | 4.48 | 0.937 / 0.935 / 0.949 | 14.3 / 25.8 / 15.7 |
| r_sulfur | r2 | 0.5145 | 0.4998 | **0.5475** | 0.4810 | 0.0442 | 0 | 2.43 | 2.36 | 2.59 | 0.515 / 0.500 / 0.547 | 7.3 / 9.0 / 5.7 |
| k_mfeat | ari | 0.5504 | **0.6263** | **0.6263** | 0.4935 | 0.6340 | 0 | 2.60 | 2.96 | 2.96 | 0.550 / 0.626 / 0.626 | 4.0 / 5.1 / 4.1 |
| k_satimage (train/test) | ari | **0.5277** | 0.5257 | 0.4923 | 0.5257 | 0.5264 | 0 | 2.49 | 2.48 | 2.32 | 0.528 / 0.526 / 0.492 | 3.2 / 5.2 / 3.4 |
| k_userknow | ari | 0.1750 | 0.1750 | **0.2055** | 0.0878 | 0.2210 | 0 | 0.83 | 0.83 | 0.97 | 0.175 / 0.175 / 0.206 | 2.1 / 2.3 / 2.0 |
| k_vowel | ari | **0.1603** | **0.1603** | **0.1603** | −0.0095 | 0.1682 | 0 | 0.76 | 0.76 | 0.76 | 0.160 / 0.160 / 0.160 | 2.5 / 2.8 / 2.4 |
| k_wallrobot | ari | 0.0541 | 0.0518 | **0.0582** | 0.0513 | 0.0037 | 0 | 0.26 | 0.24 | 0.28 | 0.054 / 0.052 / 0.058 | 3.4 / 6.0 / 3.4 |
| k_zoo | ari | **0.8757** | 0.8702 | 0.8702 | 0.8668 | 0.6063 | 0 | 4.14 | 4.11 | 4.11 | 0.876 / 0.870 / 0.870 | 2.1 / 2.2 / 1.9 |

**M1 check (many classes):** texture (11) and yeast (10) finish in ≤ 9 s for all three. At 98f5fe1 they took
up to 32.8 s, with a 52 s watchdog exit on letter in round 3. Fast HGB costs A 0.009 on both and C 0.006 on
texture.

**M2 check (clustering train/test variant):** satimage gives exactly k = 6 labels on the test rows for all three
(valid, `k_ok`).

---

## 6. Timing

| approach | 2 thr median / p90 / max | 1 thr median / p90 / max | 1 thr > 30 s | slowdown 2.5 max (budget 16 s / kill 24 s) |
|---|---|---|---|---|
| A_lean | 4.4 / 13.3 / 14.3 | 4.1 / 11.1 / **18.0** | 0 | 8.4 s, 8 / 8 valid |
| B_portfolio | 7.9 / 17.5 / 25.8 | 8.6 / 18.0 / **33.6** (r_medical) | 1 | 10.9 s, 8 / 8 valid |
| C_rules | 4.7 / 14.6 / 27.6 | 4.3 / 11.2 / 26.7 | 0 | 9.6 s, 8 / 8 valid |

- No run over 40 s and no watchdog exit in any pass.
- First predictions are on disk at ≤ 0.74 s.
- Slowdown 2.5 effect on quality (primary metric vs the 2-thread run):
  - A: −0.003 to +0.002;
  - C: −0.003 to +0.002;
  - B: −0.018 on r_medical.

---

## 7. Profile breakdown

All three approaches score 9.97 / 10. The two imperfect fields are shared code:

- **`c_bankchurn` `RowNumber` not flagged as an ID.**
  - After the split it is not consecutive.
  - Its name does not match the ID patterns (`RowNumber` has no separator before `Number`).
  - `CustomerId` is detected.
  - The feature is kept in modelling, which is harmless here.
- **`c_texture` duplicate_columns 0.667.** This is the same scorer artifact as nomao in round 3 (a group of ≥ 3
  identical columns is reported as copy → original).

---

## 8. Fix verification (docs/09 IDs, round-4 data)

| ID | verdict | evidence |
|---|---|---|
| M1a / M1b anytime HGB + fold planner | **works** | heavy miniboone: A 13.2 s, B 11.6 s, C 27.6 s (2 thr). Slowdown 2.5: all valid with no watchdog, quality within 0.003 (A, C) |
| M1c B deadlines inside fits | **works** | no B watchdog. Max 25.8 s / 33.6 s (2 / 1 thr) |
| M1d fast HGB for ≥ 10 classes | **works (time) / small quality cost** | texture: A 32.8 → 7.4 s, C 31.6 → 7.7 s, B 22.4 → 9.0 s. Macro-F1 A −0.009, C −0.006, B 0. yeast: A −0.009, B +0.038, C 0 |
| M1 bounded quick model | **works** | no quick-model-only outputs |
| M2 exactly k labels on predicted rows | **works** | satimage train/test valid for A, B, C. Round-3 C failure not reproduced |
| M3 slowdown flag | **works** | 24 / 24 heavy runs valid at 2.5×; max 10.9 s against the 16 s budget |
| Q1–Q4 | skipped / rejected (by design) | – |

---

## 9. Requirements checklist R1–R20 (HEAD 43699f2, round 4)

| req | A | B | C | evidence |
|---|---|---|---|---|
| R1 never crashes | pass | pass | pass | 0 tracebacks in 54 + 54 (1 thr) + 24 (slowdown) + 6 det + 3 stress + 84 recheck runs |
| R2 ≤ 40 s | pass (18.0 max at 1 thr) | pass (33.6) | pass (26.7) | |
| R3 validity | pass | pass | pass | incl. clustering train/test |
| R4–R5 adapter / deps | pass | pass | pass | |
| R6 deterministic | pass | partial (5 / 18 differ 1 vs 2 thr; ozone metric) | pass | determinism.csv; 2-thread reruns identical 6 / 6 |
| R7–R9 quality vs baselines | pass | pass | pass | all pools > sklearn defaults (19.02 / 21.92 / 9.56) |
| R10 balanced across tasks | pass | pass | pass | |
| R11 imbalance | pass | pass | pass | ozone 0.74 / 0.72 / 0.74 vs default 0.66 |
| R12 skewed target | pass | pass | pass | |
| R13 leak-free | pass | pass | pass | |
| R14–R15 profile | pass (9.97) | pass | pass | §7 |
| R16 speed | pass (median 4.4) | pass (7.9) | pass (4.7) | |
| R17 anytime | pass | pass | pass | first predictions ≤ 0.74 s |
| R18 bounded threads | pass | pass | pass | |
| R19 logging | pass | pass | pass | |
| R20 contract variation | pass | pass | pass | satimage variant 2; stress MCAR probe valid |

Stress: `s4_bankchurn_missing` (+20 % MCAR) is valid for all three: macro-F1 0.710 / 0.713 / 0.711 vs base
0.761 / 0.767 / 0.761.

---

## 10. Remaining improvements (post-deadline only; not needed for submission)

1. ID detection by name: `RowNumber` / `row_number` and similar names (shared, profile only).
2. A: many-class tables. Add a logistic-regression or ExtraTrees candidate for ≥ 10 classes. On texture, plain
   logistic regression reaches 0.992 vs A's 0.973.
3. B: make the OOF stopping rule independent of the clock (c_ozone differs between 1 and 2 threads).

---

## 11. Caveats

- The scoring model is ours (docs/04 §10). The top two approaches differ by 0.04 over 72 datasets; that is
  far inside single-split noise. The recommendation rests on the spec-example scores and the timing and
  validity margins, not on the total alone.
- For 43699f2, the r2 / r3 totals use the developer's same-session runs (docs/09). They were spot-checked here
  on 10 of 36 datasets (all exact) and on all of round 1 (totals within 0.02).
- Timing comes from one machine (24 cores, mains power, same session). The 1-thread and slowdown passes
  simulate a weaker grader but do not reproduce it exactly.
- Several round-4 clustering sets have low ARI for every method (vowel, wallrobot, userknow) and do not
  discriminate between the approaches.
