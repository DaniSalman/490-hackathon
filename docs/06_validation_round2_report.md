# 06 — Validation report, round 2 (fixed approaches on a brand-new suite)

EECE 490 "Unknown Dataset Hackathon". Validator (Subagent 2), 2026-10-20.

Scope:
- Re-score the fixed approaches: commits 8e2bf1d, fbc6e2c, a9f402e, 8e50e9a, 05dce9e; changelog in
  docs/05.
- Use 18 **new** datasets with zero overlap with round 1, the round-1 stress probes and docs/03.
- Use the **same grader protocol and the same scoring model** as docs/04, so the two rounds are comparable.
- Also: an independent re-run on the round-1 suite (to check the developer's claims), and the **pre-fix**
  approaches (commit d14439d) on the new suite (out-of-sample effect of the fixes).

Nothing under `approaches/` or `tools/` was modified.

Reproduce:
```
py validation/build_suite_round2.py                       # 18 datasets + 4 stress probes (gitignored data)
py validation/run_validation_round2.py run|determinism|stress|recheck|baselines
git worktree add --detach <dir> d14439d                    # pre-fix code, outside approaches/
py validation/run_validation_round2.py prefix --prefix-root <dir>
py validation/run_validation_round2.py score               # -> validation/results_round2/
```
Results (committed) in `validation/results_round2/`:
- `results.csv`, `profile_scores.csv`, `profile_new_fields.csv`, `baselines.csv`
- `determinism.csv`, `stress_results.csv`
- `round1_recheck.csv`, `prefix_results.csv`
- `summary.json`

Suite: `validation/SUITE_ROUND2.md`.

---

## 1. Summary

### Round 2 (new suite; fixed approaches)

| approach | quality (85) | profiling (10) | speed (5) | **total** | clf pool (28.3) | reg pool (28.3) | clu pool (28.3) | crashes / timeouts / watchdog | median / max wall s |
|---|---|---|---|---|---|---|---|---|---|
| A_lean | 55.59 | 10.00 | **4.48** | 70.06 | 18.90 | 24.00 | 12.69 | 0 / 0 / 0 | 3.7 / 29.9 |
| **B_portfolio** | **59.09** | 10.00 | 4.16 | **73.25** | 18.76 | **24.24** | **16.09** | 0 / 0 / 0 | 6.7 / 23.5 |
| C_rules | 57.15 | 10.00 | **4.48** | 71.63 | **19.50** | 23.96 | 13.69 | 0 / 0 / 0 | 3.7 / 30.2 |
| *pre-fix A / B / C (d14439d) on the same suite* | *56.80 / 58.83 / 52.52* | *10 / 10 / 10* | *4.67 / 3.24 / 3.68* | *71.47 / 72.07 / 66.19* | *18.75 / 19.10 / 19.55* | *23.93 / 24.26 / 19.78* | *14.12 / 15.46 / 13.20* | *C: 1 watchdog (r_kc_house)* | |
| *sklearn defaults (quality only)* | *51.94* | | | | *18.85 (HGB)* | *23.65 (HGB)* | *9.43 (scaled KMeans)* | | |

Relative-to-best normalisation (docs/04 §2.1): A 78.39, **B 83.22**, C 81.24. Same ranking.

### Round-1 suite, independent recheck vs the developer's claims (docs/05)

| approach | my re-run: total (quality / profile / speed) | developer claim | max per-dataset abs diff of primary metric |
|---|---|---|---|
| A_lean | 78.12 (63.43 / 10.00 / 4.70) | 78.13 | 5e-5 |
| B_portfolio | 76.65 (62.27 / 10.00 / 4.39) | 76.63 | 5e-5 |
| C_rules | 78.55 (63.77 / 10.00 / 4.78) | 78.56 | 5e-5 |

**The developer's numbers reproduce.** All 54 runs are valid and every per-dataset metric matches docs/05 to
within 5e-5. Totals differ by ≤ 0.02, which is wall-time noise in the speed points.

### Headlines
1. **The ranking flips out of sample: B 73.25 > C 71.63 > A 70.06.** On the suite the fixes were tuned on,
   it was C 78.56 > A 78.13 > B 76.63.
2. **A's clustering fixes did not generalise.** The fixed A scores 1.41 points *lower* than the pre-fix A on the
   new suite.
   - Clustering pool: 14.12 → 12.69.
   - On `k_synth` a GMM-diag partition was accepted (0.882 → 0.699). On the same filtered features KMeans gives
     0.87, and Ward gives 0.98.
   - On `k_segment` the relevance filter dropped real features (0.469 → 0.379).
3. **C's fixes generalise best (+5.44 out of sample).** The TargetEncoder bug **reproduced on the new data**:
   pre-fix C took 52.2 s on `r_kc_house` (integer price + 70-level zipcode), hit the watchdog, and scored
   R² −0.07. Fixed, it scores 0.857 in 6.1 s.
4. **B's fixes help (+1.18)**, mostly through speed (median 13.5 → 6.7 s) and the relevance-filtered consensus
   (`k_synth` 0.882 → 0.973). Two regressions:
   - On the wide `c_jannis` the S2 cost guard judges the portfolio "not affordable" after 6.9 s of a 40 s
     budget and keeps only the quick model (0.551 → 0.501).
   - B has no RMSLE-aware target choice: `r_seoul` RMSLE 0.665 vs 0.335 for A and 0.330 for C.
5. No crash, timeout, watchdog exit or invalid output in 54 + 6 + 3 + 12 + 54 + 54 runs. Profile 10.00 / 10
   everywhere. The new S3 fields (blank / suspected-missing counts) are exact on all 18 datasets.

---

## 2. Scoring model and protocol (unchanged from docs/04 §2)

- **Quality:** 85/18 points per dataset × `clip((m − trivial)/(1 − trivial), 0, 1)`. m = macro-F1 / R² / ARI;
  trivial = macro-F1 of the majority class / 0 / 0. Crash, timeout or invalid output = 0.
- **Profile:** 10/18 points × the mean of the same 14 fields. The new S3 fields are scored *separately* (§7),
  so the profile score stays comparable with round 1.
- **Speed:** 5/18 points × f(t) × useful. f = 1 up to 5 s, then log-decaying to 0 at 60 s. Useful = norm ≥ 0.5
  or norm ≥ 0.9 × the sklearn-default norm.
- **Protocol:** `py approaches/<X>/main.py --data <dir> --out <out>` subprocess with
  `OMP/OPENBLAS/MKL_NUM_THREADS=2`. Full wall time, hard kill of the process tree at 60 s, strictly
  sequential runs, one unscored warm-up run per approach (`k_wholesale`). Same 24-core machine as round 1.
- **Implementation:** `validation/run_validation_round2.py` imports `run_validation.py` unchanged and only
  redirects paths. Round-1 runs and results are untouched.

---

## 3. The round-2 suite (details: validation/SUITE_ROUND2.md)

| task | datasets (source) | probes |
|---|---|---|
| classification | `c_churn` (OpenML 40701), `c_magic` (1120), `c_pageblocks` (30), `c_creditapp` (29), `c_jannis` (41168, 25k subsample + 40 noise → 18750 x 97), `c_kick` (41162, 20k subsample) | S1 literal tokens + S3 blanks (churn); rare-class multiclass (pageblocks 0.5 %, jannis 2 %) for A3; small n (creditapp 518) for A4; wide table near the size limit for S2; high-cardinality + datetime (kick) |
| regression | `r_seoul` (UCI 560), `r_kc_house` (OpenML 42731), `r_moneyball` (41021), `r_superconduct` (43174, 15k), `r_brazil` (42688), `r_wine` (UCI 186) | count target / RMSLE (A5); integer target + > 30-level categorical (kc zipcode 70, moneyball Team 37) for C1; planted out-of-range test rows (kc: 70 test rows beyond +10 % of a training feature range) and native outliers (brazil) for B3 / C3; wide regression (S2); blank + `?` cells in declared numeric columns (S3) |
| clustering | `k_wholesale` (1511), `k_vehicle` (54, + 3 noise), `k_ecoli` (39), `k_pendigits` (32, 6k), `k_segment` (36, native constant + ID), `k_synth` (synthetic) | planted irrelevant features, an informative categorical with literal levels, non-spherical / unequal-variance / unequal-size groups (A1, A2, B4, C4) |

Two builder decisions are worth knowing:
- MAGIC's original file is sorted by class, so its `ID` alone predicted the class (macro-F1 0.998). Rows were
  shuffled and the ID renumbered.
- `total_(BRL)` was dropped from Brazilian houses because it is a sum that includes the target.

Note that `k_vehicle` is near-unsolvable for every method (ARI ≤ 0.09, worth 0.35 points). It does not
discriminate between the approaches.

---

## 4. Per-approach analysis (round 2)

### 4.1 A_lean — 70.06 (3rd; −1.41 vs its own pre-fix version)
**High:**
- Fast: median 3.7 s, and the speed pool ties for best at 4.48.
- Good supervised core: best `c_magic` (0.866). A3 works on `c_pageblocks` (0.830 → 0.852 with sqrt weights +
  ExtraTrees blend 0.75).
- A5 works on `r_seoul`: log1p kept, so RMSLE 0.839 → 0.335 at the same R².
- Regression pool 24.00 (≈ B and C).

**Low: clustering (12.69, the worst pool of any approach).**
- **GMM acceptance (A2) picked the wrong model** on `k_synth`. Diagnostics: agreement with KMeans 0.635 (≥ 0.6),
  stability 0.88, BIC gain 343 % — so diag was accepted. The result is ARI 0.699, while KMeans on the same
  relevance-filtered space gives 0.872. A diagonal GMM over a mixture of standardised numerics and one-hot
  columns is a poor density model: the 0/1 dimensions dominate the BIC. Confirmation: in the train/test
  stress variant, GMM was rejected and A scored 0.892.
- **The relevance filter (A1) dropped informative features.** On `k_segment` it dropped
  `region-centroid-col`, `vedge-mean`, `vegde-sd`, `hedge-mean` and `short-line-density-5` (0.469 → 0.379). On
  `k_wholesale` it dropped `Fresh` (0.514 → 0.482). The 5 %-of-max weight threshold is too aggressive when
  groups are elongated or many features are moderately informative.
- No help on unequal groups: `k_ecoli` 0.495 (B's GMM-full on PCA-whitened data: 0.724).

**Also low:**
- `c_kick` 0.655 (B/C 0.669–0.670). Balanced accuracy 0.613 is the lowest of the three; the plain
  HGB is under-weighted on the 12 % class.
- `c_creditapp`: the A4 blend fires (0.5 HGB + 0.5 LR) but gives 0.866, no gain over pre-fix. C's pure HGB
  scores 0.906.

### 4.2 B_portfolio — 73.25 (1st; +1.18 vs pre-fix)
**High:**
- **Clustering 16.09**, best by 2.4 points. The consensus over many views (now relevance-filtered) finds the
  right model family when one of the views fits:
  - `k_synth` 0.973 (Ward on robust-scaled, filtered data);
  - `k_ecoli` 0.724 (GMM-full on PCA-whitened data; A/C 0.495);
  - `k_pendigits` 0.616 (Ward);
  - `k_wholesale` 0.581.
- **Regression 24.24**, best: `r_brazil` 0.986, `r_wine` 0.502 (ExtraTrees), `r_superconduct` 0.912.
- B1 cut time in half (median 13.5 → 6.7 s, max 36.0 → 23.5 s). Speed points 3.24 → 4.16.

**Low:**
- **`c_jannis` 0.501**: below default HGB (0.526) and below pre-fix B (0.551). The S2(b) guard measured the
  quick multiclass fit (50 iterations × 4 classes on 5000 × 94 rows, 5.5 s) and concluded the portfolio was
  "not affordable" at 6.9 s of a 40 s budget, so it kept the quick model. A and C ran full HGB folds on the
  same data and finished in 30 s.
- **`r_seoul` RMSLE 0.665** (A/C ≈ 0.33). B mixes raw and log1p HGB members selected on R² (0.53 raw weight),
  with no RMSLE tie-break, which is bad if the hidden metric is RMSLE.
- **B3 cost accuracy on `r_brazil`.** On the 5 out-of-range test rows (4 flagged by B), the log-linear member
  extrapolated *correctly*: rent is roughly proportional to the fire-insurance column. Removing it raised the
  RMSE on those rows from 5394 to 8196, and R² went 0.9918 → 0.9861.
- Classification otherwise slightly behind C (−0.74 on the pool): churn 0.888, creditapp 0.872.

### 4.3 C_rules — 71.63 (2nd; +5.44 vs pre-fix)
**High:**
- **Classification 19.50**, best: best on 5 of 6 (churn 0.904, creditapp 0.906, jannis 0.553, kick 0.670,
  pageblocks 0.863). The size-tiered HGB parameters (lr 0.06, 31 leaves, patience 30) and the
  refit/fold-average are a good default; balanced-weight + OOF tuning helps on kick (balanced accuracy 0.653,
  best).
- C1 confirmed on new data: `r_kc_house` 0.857 (best).
- C3: Brazil no longer blends linear (0.946 → 0.975). Moneyball keeps the blend (probe margin 0.019) and is
  best (0.949).
- C4a keeps the unweighted space when nothing is irrelevant: pendigits 0.440 → 0.563.
- Best `k_segment` (0.478) and `k_wholesale` (0.587, GMM-diag accepted).

**Low:**
- Clustering 13.69. `k_synth` 0.699: the same GMM-diag acceptance problem as A (C4b: agreement 0.611, BIC
  gain 7507 %). `k_ecoli` 0.495.
- Regression 23.96: `r_superconduct` 0.899 (log1p by rule hurts R² here vs raw 0.911 for A/B), `r_wine` 0.463.
- `c_jannis` takes 30.2 s, the longest run of the round. It finishes 4 of 5 folds and then stops on budget.
  On a 2× slower grader, both A and C would rely on the budget logic there; it worked in every run here.

---

## 5. Per-dataset results (round 2; bold = best of A/B/C; points = 4.72 × norm)

| dataset | metric | A | B | C | sk default | sk linear / raw-KMeans | trivial | pts A | pts B | pts C | pre-fix A / B / C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c_churn | macro_f1 | 0.8907 | 0.8880 | **0.9035** | 0.9027 | 0.6943 | 0.4619 | 3.76 | 3.74 | 3.88 | 0.888 / 0.900 / 0.909 |
| c_creditapp | macro_f1 | 0.8662 | 0.8715 | **0.9064** | 0.8712 | 0.8774 | 0.3558 | 3.74 | 3.78 | 4.04 | 0.866 / 0.877 / 0.906 |
| c_jannis | macro_f1 | 0.5503 | 0.5010 | **0.5525** | 0.5255 | 0.4588 | 0.1575 | 2.20 | 1.93 | 2.21 | 0.548 / 0.551 / 0.552 |
| c_kick | macro_f1 | 0.6550 | 0.6690 | **0.6697** | 0.6491 | 0.6479 | 0.4682 | 1.66 | 1.78 | 1.79 | 0.661 / 0.669 / 0.670 |
| c_magic | macro_f1 | **0.8660** | 0.8620 | 0.8641 | 0.8651 | 0.7565 | 0.3933 | 3.68 | 3.65 | 3.66 | 0.861 / 0.865 / 0.864 |
| c_pageblocks | macro_f1 | 0.8515 | 0.8557 | **0.8626** | 0.8520 | 0.7728 | 0.1892 | 3.86 | 3.88 | 3.92 | 0.830 / 0.837 / 0.863 |
| r_brazil | r2 | 0.9760 | **0.9861** | 0.9749 | 0.9762 | 0.9512 | 0 | 4.61 | 4.66 | 4.60 | 0.977 / 0.992 / 0.946 |
| r_kc_house | r2 | 0.8525 | 0.8492 | **0.8568** | 0.8442 | 0.6922 | 0 | 4.03 | 4.01 | 4.05 | 0.846 / 0.848 / **−0.071** |
| r_moneyball | r2 | 0.9419 | 0.9482 | **0.9492** | 0.9374 | 0.9489 | 0 | 4.45 | 4.48 | 4.48 | 0.942 / 0.951 / 0.948 |
| r_seoul | r2 | 0.9356 | **0.9362** | 0.9320 | 0.8829 | 0.5485 | 0 | 4.42 | 4.42 | 4.40 | 0.933 / 0.938 / 0.932 |
| r_superconduct | r2 | 0.9108 | **0.9117** | 0.8992 | 0.9035 | 0.7411 | 0 | 4.30 | 4.31 | 4.25 | 0.910 / 0.908 / 0.899 |
| r_wine | r2 | 0.4647 | **0.5022** | 0.4628 | 0.4649 | 0.3180 | 0 | 2.19 | 2.37 | 2.19 | 0.460 / 0.502 / 0.463 |
| k_ecoli | ari | 0.4949 | **0.7241** | 0.4949 | 0.4949 | 0.4419 | 0 | 2.34 | 3.42 | 2.34 | 0.495 / 0.724 / 0.508 |
| k_pendigits | ari | 0.5582 | **0.6164** | 0.5631 | 0.5582 | 0.5447 | 0 | 2.64 | 2.91 | 2.66 | 0.558 / 0.616 / 0.440 |
| k_segment | ari | 0.3787 | 0.4392 | **0.4784** | 0.2812 | 0.3376 | 0 | 1.79 | 2.07 | 2.26 | 0.469 / 0.466 / 0.478 |
| k_synth | ari | 0.6989 | **0.9730** | 0.6989 | 0.5953 | 0.2995 | 0 | 3.30 | 4.59 | 3.30 | 0.882 / 0.882 / 0.688 |
| k_vehicle | ari | 0.0745 | 0.0746 | **0.0763** | 0.0681 | 0.0508 | 0 | 0.35 | 0.35 | 0.36 | 0.073 / 0.073 / 0.086 |
| k_wholesale | ari | 0.4820 | 0.5808 | **0.5873** | −0.0096 | −0.0306 | 0 | 2.28 | 2.74 | 2.77 | 0.514 / 0.514 / 0.594 |

Secondary metrics (all in results.csv):

| | A | B | C |
|---|---|---|---|
| balanced accuracy c_kick / c_pageblocks / c_jannis | 0.613 / 0.891 / 0.557 | 0.640 / 0.894 / 0.498 | **0.653 / 0.909 / 0.559** |
| RMSLE r_seoul / r_superconduct / r_brazil | 0.335 / 0.465 / 0.082 | **0.665** / 0.461 / 0.100 | **0.330** / **0.400** / **0.082** |
| NMI k_synth / k_ecoli / k_pendigits | 0.660 / 0.630 / 0.698 | **0.951 / 0.666 / 0.766** | 0.660 / 0.630 / 0.705 |

Out-of-range analysis (B3 / C3):

| dataset (out-of-range test rows) | A | B | C | pre-fix A / B / C |
|---|---|---|---|---|
| r_kc_house (70): RMSE on those rows | 1.26 M | 1.29 M | 1.22 M | 1.30 M / 1.30 M / 2.65 M (constant) |
| r_brazil (5): RMSE on those rows | 11.7 k | 8.2 k | 11.9 k | 11.4 k / **5.4 k** / 6.1 k |

On `r_kc_house` every approach under-predicts the largest houses by up to 5 M: trees do not extrapolate, and
no linear member is involved. B3 / C3 therefore have nothing to protect against there.

---

## 6. Timing (round 2)

| approach | median s | p90 s | max s (dataset) | runs > 20 s | first prediction on disk (max) | profile written (max) |
|---|---|---|---|---|---|---|
| A_lean | 3.73 | 11.5 | 29.9 (c_jannis) | 1 | 0.63 s | 1.01 s |
| B_portfolio | 6.71 | 13.5 | 23.5 (r_superconduct) | 1 | 0.61 s | 1.00 s |
| C_rules | 3.72 | 11.9 | 30.2 (c_jannis) | 1 | 0.59 s | 0.99 s |
| pre-fix A / B / C | 2.82 / 13.45 / 4.78 | | 29.3 / 36.0 / **52.2 (watchdog)** | | | |

- The wide `c_jannis` (18750 × 97, 4 classes) is the slowest dataset for A and C: about 6 s per HGB fold at
  2 threads. Both stop CV after 4 folds on the budget check, so the S2(c) per-fold check works.
- B 1-thread spot check: `r_superconduct` 35.0 s (2 threads: 23.5 s), `c_jannis` 7.1 s, `c_pageblocks` 11.6 s.
- Recheck on the round-1 suite: median/max A 3.05 / 15.0, B 4.76 / 17.1, C 3.72 / 15.2. This agrees with
  docs/05 (2.85 / 15.0, 4.85 / 17.9, 3.75 / 15.3).

---

## 7. Profile breakdown

| field group | A / B / C (round 2) | notes |
|---|---|---|
| 14 docs/04 fields (task, target, counts, per-column missing / unique / type, constant, duplicate, ID, target profile, datetime) | **1.000 on all 18 datasets** for all three (10.00 pts) | native constant `region-pixel-count`, planted constants / duplicates, undeclared IDs (`ID`, `phone_number`, `id` with repeats, `image_id`, `customer_id`) and all 3 datetime formats (m/d/Y, dd/mm/yyyy, 20141013T000000) detected |
| new S3 fields, scored separately: per-column `blank_string_count`, per-column `suspected_missing`, `total_blank_string_cells`, `total_suspected_missing_cells` | **1.000 / 1.000 / 1.000 / 1.000** | churn `total_eve_charge`: 124 blanks. wine: 198 blanks (`residual sugar`) + 100 `?` (`sulphates`) = 298 suspected. Ground truth = whitespace-only strings + {?, na, n/a, nan, null, #n/a, missing}, case-insensitive |

Stress-probe profiles without declared types score 0.98–0.996. The only "errors" are integer-coded categoricals
(`state`, `zipcode`, `WheelTypeID`, 0/1 flags) inferred as numeric. Without metadata that is a defensible
inference; the expected type comes from the base dataset's declaration.

---

## 8. Fix verification (docs/05 IDs, on round-2 data)

Evidence = fixed vs pre-fix (d14439d) on the same round-2 datasets, unless noted.

| ID | verdict | evidence |
|---|---|---|
| S1 literal categories | **works** (correctness), no measurable quality effect | churn `addon_plan` kept as a 5-level categorical including `none`, `-`, `unknown` (profile `top_values`, action `categorical`). k_synth `plan` levels kept. Churn macro-F1 fixed vs pre-fix: A +0.003, B −0.012, C −0.005. The other features dominate, so merging the literal levels into NA cost little. `s2_churn_tokens`: `?`, `-`, `none` inside numeric columns are coerced to NaN, and the columns stay numeric (all 3 valid, 0.905–0.912) |
| S2 cost guards | **works for A / C; regressed for B** | no timeout or watchdog in 129 fixed-version runs. A/C on c_jannis stop CV after 4 folds at 29.5–30.0 s. **B on c_jannis keeps the quick model** ("portfolio not affordable" at 6.9 s): 0.551 → 0.501. Pre-fix C watchdog on kc_house reproduced the C1 bug class, and the fixed C is safe |
| S3 suspected-missing fields | **works** | all 4 new fields exact on 18/18 datasets (§7) |
| S4 warm-up / versions | not re-verified (environment claim) | warm-up runs 1.6–2.2 s. Older-stack claims not re-tested in this round |
| A1 relevance filter | **partial** | correct drops: k_synth (5 noise + `noise_flag`), k_vehicle (3 noise). Wrong drops: k_segment 5 real features (0.469 → 0.379), k_wholesale `Fresh` (0.514 → 0.482) |
| A2 guarded GMM | **regressed** | k_synth: GMM-diag accepted (agreement 0.635, BIC +343 %) → 0.699. KMeans on the same filtered space = 0.872 (offline check); train/test stress variant (GMM rejected) = 0.892. ecoli / segment / vehicle / wholesale: GMM correctly rejected |
| A3 multiclass weights + ET blend | **works** | pageblocks 0.830 → 0.852 (ET weight 0.75). jannis 0.548 → 0.550 |
| A4 linear blend n < 2000 | **no effect** | creditapp blend 0.5/0.5 chosen, 0.866 → 0.866 (C's plain HGB 0.906) |
| A5 log1p tie-break | **works** | r_seoul RMSLE 0.839 → 0.335, R² 0.933 → 0.936. kc_house / brazil also log1p, R² ±0.006 |
| B1 trimmed portfolio + stopping rule | **works** | median 13.45 → 6.71 s, max 36.0 → 23.5 s, speed 3.24 → 4.16. Stopping rule fired (e.g. kick "after hgb_b: gain 0.0000"). Quality pool cost: clf −0.35 (mostly jannis = S2), reg −0.02 |
| B2 deterministic plan | **works** (1 caveat) | 1-thread vs 2-thread: c_jannis and c_pageblocks byte-identical. r_superconduct: same model and same R² to 1e-16, but 1.7 % of prediction strings differ in the last floating-point digits (threaded summation). Reruns at 2 threads identical |
| B3 linear clipping / exclusion on out-of-range rows | **partial** | kc_house: no linear member selected (no effect). brazil: linear share removed on 4 rows. Out-of-range RMSE 5.4 k → 8.2 k, R² 0.992 → 0.986 (the log-linear member extrapolated correctly). Protects against blow-ups (round-1 house) but removes correct extrapolation |
| B4 relevance-filtered consensus | **works** | k_synth 0.882 → 0.973, wholesale 0.514 → 0.581. segment 0.466 → 0.439 (same over-filtering as A1). ecoli / pendigits unchanged |
| C1 TargetEncoder target type | **works (confirmed out of sample)** | r_kc_house pre-fix: 52.2 s watchdog, R² −0.071. Fixed: 6.1 s, R² 0.857 (best). moneyball (Team 37 levels, integer RS) 0.949 |
| C2 quick model first + per-model checks | **works** | quick HGB on disk at 2–3 s. c_jannis: CV stopped after 4 folds at 29.9 s, fold-average used, no watchdog |
| C3 linear blend only with probe margin | **works** (small out-of-range cost) | brazil: no blend → 0.946 → 0.975 (in-range R² 0.949 → 0.996), but out-of-range RMSE 6.1 k → 11.9 k. moneyball blend kept (margin 0.019), 0.949 best. kc_house probe 0.850 vs 0.873 → no blend |
| C4a group-aware sparse k-means, weighted space only if a weight < 5 % | **works** | pendigits 0.440 → 0.563, synth noise dropped, segment / vehicle unchanged |
| C4b GMM acceptance (agreement ≥ 0.6, BIC ≥ 15 %) | **partial** | wholesale GMM-diag 0.587 (> KMeans 0.48–0.51). k_synth GMM-diag accepted → 0.699 (KMeans ≈ 0.87). ecoli rejected |

---

## 9. Round 1 → round 2: generalisation of the fix round

| approach | dev gain on round-1 suite (docs/04 → docs/05) | gain on round-2 suite (pre-fix → fixed) | round-1 total (fixed) | round-2 total (fixed) |
|---|---|---|---|---|
| A_lean | +2.32 | **−1.41** | 78.12 | 70.06 |
| B_portfolio | +2.59 | +1.18 | 76.65 | 73.25 |
| C_rules | +5.33 | **+5.44** | 78.55 | 71.63 |

How to read it:
- **The round-1 → round-2 drop in absolute totals is mostly suite difficulty, not overfitting.** The sklearn
  default baselines also drop: clustering pool 17.08 → 9.43 (vehicle, ecoli, segment and wholesale are much
  harder than iris / wine / seeds).
- The fair measure of overfitting is the *out-of-sample effect of the fixes* (middle column) and the *margin
  over the default baseline*:

| margin over sklearn default (pool points) | clf r1 → r2 | reg r1 → r2 | clu r1 → r2 |
|---|---|---|---|
| A_lean | +1.38 → +0.05 | +0.44 → +0.35 | +5.10 → +3.26 |
| B_portfolio | +1.01 → −0.09 | +0.54 → +0.59 | +4.21 → **+6.66** |
| C_rules | +0.63 → **+0.65** | +0.37 → +0.31 | +6.26 → +4.26 |

- **A was overfit.** Its round-1 gains came from clustering thresholds (relevance filter at 5 %, GMM acceptance)
  checked on the round-1 suite: iris +0.27, penguins +0.14. They reverse on new data (synth −0.18, segment
  −0.09, wholesale −0.03). A's classification margin over default HGB also disappears.
- **C's gains were bug fixes** (C1, C2, C3), so they transfer almost one-for-one (+5.33 → +5.44). Its
  classification margin is the most stable of the three.
- **B's gains partially transfer** (+2.59 → +1.18). The speed part transfers. The consensus filter transfers
  (synth). The S2 guard and B3 cost points on new shapes (jannis, brazil).
- **Over both suites (36 datasets):** totals A 148.2, B 149.9, C 150.2. Quality: A 119.0, B 121.4, C 120.9.
  Per pool: clf A 35.9 / B 35.4 / C 35.8; reg A 48.2 / B 48.6 / C 48.1; clu A 34.9 / B 37.4 / C 37.0.

---

## 10. Requirements checklist R1–R20 (round 2, fixed versions)

| req | A_lean | B_portfolio | C_rules | evidence |
|---|---|---|---|---|
| R1 never crashes | pass | pass | pass | 0 non-zero exits / tracebacks in 54 suite + 6 det + 3 one-thread + 12 stress + 54 recheck runs |
| R2 ≤ 40 s at 2 threads | pass (max 29.9) | pass (max 23.5; 35.0 at 1 thread) | pass (max 30.2) | results.csv |
| R3 output validity | pass | pass | pass | 54/54 valid, k_ok everywhere (incl. ecoli k=8 with 2-row groups) |
| R4 shared adapter | pass | pass | pass | shared modules byte-identical (docs/05) |
| R5 pinned deps | pass | pass | pass | unchanged pins; older stack not re-tested |
| R6 deterministic | pass | pass (1-thread vs 2-thread: last-digit float differences on 1/3 regression sets) | pass | determinism.csv |
| R7 classification quality | pass (18.90 ≈ sk 18.85) | **partial** (18.76 < sk 18.85; jannis below default) | pass (19.50, best) | |
| R8 regression quality | pass (24.00) | pass (24.24, best) | pass (23.96) | sk_hgb 23.65 |
| R9 clustering quality | pass (12.69 > sk 9.43) | pass (16.09, best) | pass (13.69) | |
| R10 balanced across tasks | pass | partial (jannis 0.501 < default 0.526) | pass | min per task: A 0.550 / 0.465 / 0.075; B 0.501 / 0.502 / 0.075; C 0.552 / 0.463 / 0.076 |
| R11 imbalance handling | partial (kick balanced accuracy 0.613) | pass | pass (best balanced accuracy on kick / pageblocks) | |
| R12 skewed-target transform | pass (A5) | **partial** (no RMSLE tie-break: Seoul RMSLE 0.665) | pass (rule; RMSLE 0.330) | |
| R13 leak-free CV | pass | pass | pass | unchanged design; MAGIC ID leak removed from the data |
| R14 profile correctness | pass (1.000) | pass | pass | S1 fixed the round-1 error |
| R15 profile cheap | pass (≤ 1.01 s) | pass | pass | |
| R16 speed | pass (3.7 s median) | pass (6.7 s) | pass (3.7 s) | |
| R17 anytime | pass | pass | pass | constant fallback ≤ 0.63 s, quick model ≤ 6.7 s everywhere |
| R18 bounded threads / width | pass | pass | pass | S2a width cap; C's TargetEncoder fixed |
| R19 decision logging | pass | pass | pass | gmm_checks, sparse_weights, portfolio_plan / stopped, rules_fired all in `profile.modelling` |
| R20 contract variation | pass | pass | pass | stress: no feature_types (kick, kc_house, churn tokens), clustering train/test variant + unseen level. All valid |

### Round-2 stress probes (separate; `stress_results.csv`)

| probe | A | B | C | base A / B / C |
|---|---|---|---|---|
| s2_churn_tokens (`?` `-` `none` in numerics, no types) | 0.905 | 0.910 | 0.912 | 0.891 / 0.888 / 0.904 |
| s2_kick_notypes | 0.658 | 0.676 | 0.667 | 0.655 / 0.669 / 0.670 |
| s2_kc_house_strnum (`"1,234"` + 15 % MCAR, no types) | 0.847 | 0.845 | 0.848 | 0.852 / 0.849 / 0.857 |
| s2_synth_traintest (variant 2 + unseen level; ARI on 750 test rows) | 0.892 | 0.968 | 0.875 | 0.699 / 0.973 / 0.699 |

All 12 runs were valid (≤ 13.2 s). Undeclared types cost ≤ 0.01. The synth train/test result exposes the
GMM-acceptance problem: when GMM is rejected, A and C jump to 0.88–0.89.

---

## 11. Prioritised improvements (next fix round; not implemented)

### A_lean
1. **GMM acceptance:** evaluate the GMM on numeric columns only (exclude one-hot dimensions), or require GMM to
   beat KMeans on a stability or silhouette criterion rather than BIC gain. Expected: k_synth 0.70 → ~0.87.
2. **Relevance filter:** drop a feature only when its weight < 5 % of max **and** dropping it does not reduce
   KMeans stability, or use a permutation test per group. Expected: segment +0.09, wholesale +0.03.
3. Borrow B's consensus idea for clustering (Ward + GMM-full on PCA-whitened data as candidates). That is
   where ecoli (0.49 → 0.72) and synth (0.97) are won.
4. Classification on imbalanced binary: kick balanced accuracy 0.613. Use OOF threshold tuning with
   balanced weights as in C.

### B_portfolio
1. **S2 affordability rule:** when the projected portfolio cost exceeds the budget, fall back to the full
   single `hgb_a` (as A/C do), not to the 5000-row quick model. Better still, project the cost per candidate
   from its own size class. Expected: jannis 0.50 → 0.55 (+0.27 pts).
2. **RMSLE safety for counts:** for count-like non-negative targets, choose the transform by OOF R² with an
   RMSLE tie-break (A5), or force log1p members when zeros and skew > 0.75 (Seoul RMSLE 0.67 → 0.33).
3. **B3:** keep the linear member on out-of-range rows when it agrees in sign / scale with the tree members, or
   cap instead of removing (brazil). Keep the clipping in log space.
4. The relevance filter is shared with A, so it has the same segment problem (A item 2).

### C_rules
1. **C4b GMM acceptance:** same fix as A item 1 (synth 0.70 → ~0.87).
2. **Clustering candidates:** add Ward and GMM-full on PCA-whitened data with a consensus or stability
   choice, as B does (ecoli 0.49 → 0.72, pendigits 0.56 → 0.62).
3. **log1p by rule:** keep the rule for RMSLE safety, but add an OOF R² check. Revert to raw when log1p loses
   > 0.005 R² (superconduct 0.899 vs 0.911).
4. Wide multiclass (jannis 30 s): fewer folds (3) when n × p × C is large, to keep headroom on a slower grader.

### Shared
1. The same 5 % relevance threshold is in `feature_weighting.py` for all three approaches; fix it once (A item 2).
2. Report integer-coded categoricals (zip codes, state codes) as `categorical` when undeclared and the column
   name matches `zip|code|state|_id`. This is a minor profile-inference improvement.

---

## 12. Final-submission recommendation (updated)

**Submit B_portfolio**, with B items 1 and 2 above if time allows. Reasons:
- It is the best approach on unseen data (73.25 vs C 71.63 and A 70.06) and the best on the relative
  normalisation (83.2).
- It is best in the regression and clustering pools, and its clustering advantage is structural: the consensus
  over model families is the only mechanism that handled unequal-size groups (ecoli) and non-spherical groups
  (synth).
- Its known weaknesses are two isolated, easy fixes (the S2 fallback target, the RMSLE tie-break), not design
  problems.
- It is still well inside the time limit (max 23.5 s at 2 threads, 35 s at 1 thread).

**A strong alternative is C_rules + B's clustering**: C's supervised pipeline (best classification out of
sample, most stable margins, fastest along with A) combined with B's consensus clustering. On round 2 that
combination scores 19.50 + 23.96 + 16.09 = 59.55 quality, about 74 total, but it needs integration work.

**A_lean is no longer recommended as-is.** Its round-1 lead came from clustering rules that do not transfer.

---

## 13. Caveats

- The scoring model is ours (docs/04 §10). The real metrics, normalisation and profile schema are unknown.
  Under an RMSLE regression metric, B would lose ground on count targets (Seoul).
- One split per dataset. Differences < ~0.3 quality points (for example B vs C over the 36 datasets: 149.9 vs
  150.2) are within noise. The round-2 ranking B > C > A rests mainly on clustering (ecoli, synth) and on the
  C1 bug class.
- Clustering has high variance per dataset (ARI swings of 0.2–0.3 from a model-family choice). Six clustering
  sets is a small sample: `k_synth` alone moves B by 1.3 points relative to A/C.
- `k_synth` is synthetic and was designed to probe A1/A2/B4/C4. It may favour hierarchical methods. Its
  ground truth is exact, unlike Mall in round 1.
- The B 1-thread spot check covers 3 datasets. The 2× slower grader is simulated only by the thread cap.
- The pre-fix runs use the d14439d code in a git worktree (identical to the round-1 evaluated code).
