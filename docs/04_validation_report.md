# 04 — Validation report (Subagent 2: assessment-agent simulation)

EECE 490 "Unknown Dataset Hackathon". Date: 2026-10-20. Scope: evaluate approaches A_lean, B_portfolio and
C_rules **as submitted** (commit d14439d; nothing under `approaches/` was changed) on a private-like suite of
18 real public datasets, scored with an explicit model of the hackathon rules.

Reproduce (about 25 min, data downloads cached in `validation/cache/`):
```
py validation/build_suite.py               # 18 datasets + 9 stress probes -> validation/data/ (gitignored)
py validation/run_validation.py all        # warm-up, timed runs, determinism, stress, baselines, scoring
```
Raw results (committed): `validation/results/results.csv` (one row per approach x dataset),
`profile_scores.csv`, `baselines.csv`, `determinism.csv`, `stress_results.csv`, `summary.json`.
Suite documentation: `validation/SUITE.md`.

---

## 1. Summary

| approach | quality (85) | profiling (10) | speed (5) | **total (100)** | clf pool (28.3) | reg pool (28.3) | clu pool (28.3) | crashes / timeouts | median / max wall s |
|---|---|---|---|---|---|---|---|---|---|
| **A_lean** | **60.97** | 9.96 | **4.89** | **75.81** | 16.44 | **24.24** | 20.29 | 0 / 0 | **2.4 / 8.5** |
| B_portfolio | 60.68 | 9.96 | 3.40 | 74.04 | **16.75** | 23.95 | 19.99 | 0 / 0 | 9.3 / 33.9 |
| C_rules | 58.73 | 9.96 | 4.55 | 73.23 | 16.50 | 19.49 | **22.74** | 0 / 0 (1 watchdog exit at 52.3 s) | 3.4 / 52.3 |
| *sklearn default baselines (quality only)* | *56.51* | | | | *15.66 (HGB)* | *23.77 (HGB)* | *17.08 (scaled KMeans)* | | |

Quality with the alternative "relative to the best reference" normalisation (section 2.1): A 80.00, B 79.92,
C 77.02 — same ranking.

Headlines:
- **A, B and C are within 2.3 quality points of each other; no approach dominates every task.**
A is best on regression and on overall points because it is fast and never fails. B is best on classification by a
small margin and loses about 1.5 speed points. C has by far the best clustering (+2.5 to +2.8 pts over A/B) but
**loses a full dataset on `r_california`**: a TargetEncoder bug (section 4.3) makes one CV fold take ~93 s, the
watchdog exits at 52 s, and only the constant fallback is on disk (R² = −0.04, 0 pts).
- No crash, no invalid output, no 60 s kill in 54 timed runs + 9 determinism reruns + 27 stress runs.
  All outputs were bit-identical on rerun.
- The shared profiler is essentially correct (99.6 % of field checks). The only error comes from a shared
  bug: `"none"` is treated as a missing-value sentinel, so a real category is lost (credit-g `own_telephone` is
  reported constant and **dropped from the features** in all three approaches).
- Recommendation (section 9): **A as the core, plus C's sparse k-means clustering with guards, plus three
  targeted fixes.** Re-scoring the per-dataset winners gives ≈ 63.4 quality points (+2.4 over A) with A's speed.

---

## 2. Scoring model (assumption; the real rules are unpublished)

### 2.1 Model quality — 85 pts
- 85 / 18 = 4.72 pts per dataset, 6 datasets per task → 28.33 pts per task pool (spec: equal weight per task).
- Primary metric m: classification **macro-F1**, regression **R²**, clustering **ARI** (docs/01 §2). Accuracy,
  balanced accuracy, RMSE, RMSLE and NMI are reported as well.
- `norm = clip((m − trivial) / (1 − trivial), 0, 1)`, where trivial = macro-F1 of predicting the majority class on
  the test set (0.09–0.49), and 0 for R² and ARI. Points = 4.72 × norm.
- Alternative: `norm_rel = clip((m − trivial) / (best − trivial), 0, 1)`, where best = the best of A/B/C and the
  two sklearn baselines on that dataset (`quality_pts_rel` in results.csv). It gives the same ranking.
- A crash, a timeout (> 60 s, process tree killed), invalid output (wrong row count or order, NaN, wrong k)
  scores 0. A watchdog exit (exit code 0, file on disk) is scored on whatever is on disk.

### 2.2 Profiling — 10 pts
10 / 18 = 0.556 pts per dataset × the mean of 14 field scores in [0, 1]. Ground truth is computed with pandas
(`read_csv` defaults) on the raw `train.csv` / `data.csv`, plus the planted challenges in `hidden/truth.json`:

| field | rule |
|---|---|
| task, target | exact (clustering: target null) |
| n_rows, n_columns, n_duplicate_rows | exact vs pandas |
| total_missing_cells | equals pandas `isna().sum().sum()` (all columns or features only) |
| col_missing, col_unique | fraction of columns whose `n_missing` / `n_unique` (dropna) match pandas |
| col_type | fraction of feature columns with correct `type`. Expected = declared type (categorical also accepts boolean). Planted / known ID → `id`. Truly constant → `constant`. A declared id_column passes with role `id` |
| constant_columns | set F1 vs columns with `nunique(dropna=False) <= 1` |
| duplicate_columns | pair-set F1 vs exact-equal column pairs among non-constant columns |
| id_like_columns | set F1 vs declared id_column + planted/known IDs (`Product ID`, `applicant_id`, `sample_no`, `specimen_id`) |
| target_profile | clf: n_classes (½) + exact class counts (½); reg: mean/min/max within 1e-6 relative; clu: `n_clusters_required == k` |
| datetime_columns | set F1 vs known datetime columns |

### 2.3 Speed — 5 pts
5 / 18 = 0.278 pts per dataset × f(t) × useful. `f(t) = 1` for t ≤ 5 s, `1 − ln(t/5)/ln(12)` for 5 < t < 60, else 0.
For example f(10 s) = 0.72, f(20 s) = 0.44 and f(33 s) = 0.24. "Useful" means valid, and either norm ≥ 0.5 or
norm ≥ 0.9 × the norm of the sklearn default baseline. t is the full subprocess wall time.

### 2.4 Run protocol (grader simulation)
- `py approaches/<X>/main.py --data <dir> --out <out>` as a subprocess. Default CLI options (budget 40 s,
  watchdog 52 s). Env `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`, no `HACK_N_JOBS`.
- Wall time runs from `Popen` to exit (interpreter start-up and imports included). Hard kill of the whole
  process tree (`taskkill /T /F`) at 60 s, tested.
- Strictly sequential runs. Each approach was first run once on `k_seeds` as an unscored warm-up (1.7–2.2 s), so
  cold `.pyc` compilation and first DLL loads are excluded. **A truly cold first run can cost much more**:
  docs/01 measured 59 s for a first-ever sklearn import on this machine.
- Machine: 24-core Windows 11 desktop, Python 3.14.7, sklearn 1.9.1. A slower grader CPU is simulated only
  through the 2-thread cap; a single-thread spot check is in section 6.
- Baselines (in-process, `validation/run_validation.py baselines`): trivial (majority / mean / one cluster);
  **sk_hgb** (median impute + one-hot ≤ 30 levels + default `HistGradientBoosting`, declared datetimes expanded
  to 5 calendar parts); **sk_linear** (same + StandardScaler + LogisticRegression / RidgeCV); **sk_kmeans**
  (impute + one-hot + StandardScaler + `KMeans(k, n_init=10)`); **sk_kmeans_raw** (same without scaling). The
  baselines do no ID / constant / duplicate cleaning.

---

## 3. The suite (details in validation/SUITE.md)

| task | datasets |
|---|---|
| classification | `c_telco` (Kaggle Telco churn, spec example), `c_ai4i` (AI4I 2020, spec example; leaking failure-mode flags dropped), `c_adult` (20k), `c_bank` (20k, + constant/duplicate/noise), `c_covertype` (15k, 7 imbalanced classes), `c_creditg` (1k, + MCAR/duplicate/noise/datetime/undeclared ID) |
| regression | `r_house` (Ames, spec example), `r_bike` (UCI hour.csv in Kaggle Bike Sharing schema, spec example), `r_california` (Kaggle version, + 529-level categorical + noise), `r_abalone` (+ dup/const), `r_concrete` (+ MCAR/noise/row-counter ID), `r_diamonds` (15k) |
| clustering | `k_mall` (real Kaggle Mall file, spec example; **derived labels**), `k_iris` (+ 3 noise, const, ID), `k_wine` (scales), `k_seeds`, `k_penguins` (cats + missing), `k_drybean` (13.6k, k=7, + duplicate) |

There is no overlap with the developer datasets in docs/03. Note that the developer anticipated the Telco, Bike
and Mall *data issues* with synthetic look-alikes.

---

## 4. Per-approach analysis

### 4.1 A_lean — 75.8 pts (best total)
**High:** regression (24.24, best). Classification is on par with B and C (16.44), and it is the fastest on every
dataset (max 8.5 s, every run gets ≥ 79 % speed credit). It has the best `r_house` R² (0.891) because pure HGB
does not extrapolate. In the test split, the famous Ames outlier (GrLivArea 5642, sold for 160k) is predicted
316k by A, 635k by B and 814k by C. It has the best imbalanced binary scores (`c_ai4i` 0.864, `c_bank` 0.778):
capped balanced weights plus OOF class-weight tuning work. It has the best `k_penguins` ARI (0.751): plain
standardised KMeans keeps the informative `island` one-hot.

**Low:**
- **Clustering with irrelevant features or non-spherical groups.** `k_iris` 0.63 vs C 0.90: the 3 planted noise
  features dilute KMeans distances, and A has no feature filter. `k_mall` 0.58 vs C 0.97: Age and Gender dilute
  the Income/Spending structure. `k_drybean` 0.66 vs C 0.72: KMeans on elongated clusters. A matches the plain
  scaled KMeans baseline on seeds and wine and gains little elsewhere: its log1p + one-hot/√2 preprocessing is the
  only difference from the baseline.
- **Multiclass imbalance:** `c_covertype` macro-F1 0.566 (balanced accuracy 0.529). This is *below* the plain
  default HGB (0.593) and below B (0.616). The single HGB with early stopping (lr 0.08) plus capped balanced
  weights underfits the rare classes. B's ExtraTrees/RF weight (0.6 of the ensemble) is what wins there.
- **Small tables:** `c_creditg` 0.670 vs B 0.704 / C 0.695. The tiny-data linear blend triggers only at n < 500,
  and credit-g has 750 training rows, where the linear model is the stronger one (sk_linear 0.675).
- Target transform chosen by OOF RMSE: on `r_bike` A keeps the raw target (RMSLE 0.42), while C's log1p reaches
  RMSLE 0.27 at nearly the same R². This is fine under R², but A would lose points if the hidden metric is RMSLE
  (the Kaggle metric for Bike Sharing).

### 4.2 B_portfolio — 74.0 pts
**High:** best classification pool (16.75): `c_covertype` 0.616 (ExtraTrees + RF + HGB ensemble) and
`c_creditg` 0.704 (logreg + forests). Best or tied regression on 4 of 6 regression sets (abalone, california,
diamonds; bike ≈ A). Fully deterministic on the same machine.

**Low:**
- **Speed:** median 9.3 s. 4 datasets take 32–34 s (covertype, bike, california, diamonds) and adult/bank take
  23–27 s, so speed points are 3.40 vs 4.89 for A. At 2 threads B fills about 80 % of its 40 s budget on every
  dataset with n > 10k. The extra models bought +0.3 classification points and −0.3 regression points: on this
  suite the portfolio does not pay for its run time.
- **Regression robustness:** `r_house` R² 0.821 vs A 0.891. The Caruana ensemble gives 0.26 weight to
  `ridge[log1p]` and 0.48 to `hgb_b[log1p]`. The log-linear model extrapolates the 5642 sq-ft outlier to 635k. On
  OOF this mix scores better, and without that one row B's R² (0.925) would be higher than A's (0.903). The
  risk is outliers in the test set, which OOF selection cannot see.
- **Clustering = majority vote:** the consensus medoid follows the views that share a bias. On `k_iris` all
  views keep the noise features (0.63 = A). On `k_penguins` it picks GMM-full on standardised data (0.65 < A 0.75).
- **Machine dependence:** the candidate set depends on the wall clock. A single-thread rerun of `c_covertype`
  gave 0.605 instead of 0.616 (fewer candidates fit in the budget), so outputs are not hardware-deterministic.

### 4.3 C_rules — 73.2 pts (would be ≈ 77.5 with the California bug fixed)
**High:** **clustering pool 22.74** (A 20.29, B 19.99). Sparse k-means zeroes the 3 noise features on `k_iris`
(ARI 0.90). The GMM step gives 0.97 on `k_mall` and 0.72 on `k_drybean`, and wine improves slightly. Fast
(median 3.4 s). Best `c_telco` (0.734). Log1p by rule gives the best RMSLE on bike (0.27).

**Low:**
- **`r_california`: 0 points (bug).** `geo_cell` has 529 levels (> 30), so C target-encodes it with sklearn
  `TargetEncoder(target_type="auto")` (`Features._te`). The regression target `median_house_value` is
  integer-valued floats, so `type_of_target` returns **multiclass**. The encoder then produces one column per
  distinct price, **3412 extra features** (matrix 12384 × 3425). One HGB fit takes 70–85 s; the first CV fold
  alone took 93 s. The budget check only runs *between* folds, so nothing better than the constant fallback was
  written before the 52 s watchdog. **The trigger is common on hidden data**: any regression with an
  integer-valued target (prices, counts, Rings) plus a categorical with > 30 levels. House Prices escaped only
  because Neighborhood has 25 levels.
- **`r_house` 0.751:** the 0.5 linear + 0.5 HGB[log1p] blend (the probe judged the data "linear-ish")
  extrapolates the outlier to 814k. A pure HGB would score ≈ 0.89.
- **Clustering regressions:** `k_penguins` 0.58 (A 0.75). Sparse k-means gives the `island` one-hot
  columns weights of 0.05/0.03 (one-hot/√2 columns have small between-cluster sums of squares), so an informative
  categorical is effectively removed. GMM-diag then wins on BIC. `k_seeds` 0.72 (A 0.77): GMM-full replaces
  KMeans on BIC grounds, and BIC does not track ARI. Sparse k-means is not why Mall wins: Age keeps weight
  0.52. The gain comes from GMM-diag.
- Like B, C still blends a linear model into many supervised fits, which helps telco and credit-g but hurts the
  heavy-tailed house prices.

---

## 5. Per-dataset results (primary metric; bold = best of A/B/C; points = 4.72 × norm)

| dataset | metric | A | B | C | sk default (HGB / KMeans) | sk linear / raw-KMeans | trivial | pts A | pts B | pts C |
|---|---|---|---|---|---|---|---|---|---|---|
| c_adult | macro_f1 | **0.8166** | 0.8115 | 0.8150 | 0.8103 | 0.7763 | 0.4323 | 3.20 | 3.15 | 3.18 |
| c_ai4i | macro_f1 | **0.8643** | 0.8492 | 0.8517 | 0.8538 | 0.6895 | 0.4914 | 3.46 | 3.32 | 3.34 |
| c_bank | macro_f1 | **0.7775** | 0.7739 | 0.7674 | 0.7532 | 0.6996 | 0.4689 | 2.74 | 2.71 | 2.65 |
| c_covertype | macro_f1 | 0.5656 | **0.6159** | 0.5721 | 0.5931 | 0.4141 | 0.0908 | 2.47 | 2.73 | 2.50 |
| c_creditg | macro_f1 | 0.6702 | **0.7037** | 0.6951 | 0.6220 | 0.6747 | 0.4118 | 2.07 | 2.34 | 2.27 |
| c_telco | macro_f1 | 0.7283 | 0.7279 | **0.7344** | 0.7073 | 0.7170 | 0.4236 | 2.50 | 2.49 | 2.55 |
| r_abalone | r2 | 0.5680 | **0.5751** | 0.5540 | 0.5352 | 0.5441 | 0 | 2.68 | 2.72 | 2.62 |
| r_bike | r2 | **0.9623** | 0.9620 | 0.9598 | 0.9483 | 0.4076 | 0 | 4.54 | 4.54 | 4.53 |
| r_california | r2 | 0.8422 | **0.8433** | −0.0445 | 0.8305 | 0.6988 | 0 | 3.98 | 3.98 | 0.00 |
| r_concrete | r2 | **0.8897** | 0.8884 | 0.8818 | 0.8933 | 0.5791 | 0 | 4.20 | 4.20 | 4.16 |
| r_diamonds | r2 | 0.9800 | **0.9811** | 0.9800 | 0.9773 | 0.9072 | 0 | 4.63 | 4.63 | 4.63 |
| r_house | r2 | **0.8910** | 0.8211 | 0.7507 | 0.8486 | 0.6771 | 0 | 4.21 | 3.88 | 3.54 |
| k_drybean | ari | 0.6600 | 0.6610 | **0.7242** | 0.6718 | 0.3777 | 0 | 3.12 | 3.12 | 3.42 |
| k_iris | ari | 0.6306 | 0.6306 | **0.9039** | 0.5833 | 0.0011 | 0 | 2.98 | 2.98 | 4.27 |
| k_mall | ari | 0.5845 | 0.6175 | **0.9696** | 0.1434 | 0.9712 | 0 | 2.76 | 2.92 | 4.58 |
| k_penguins | ari | **0.7508** | 0.6534 | 0.5805 | 0.5485 | 0.3272 | 0 | 3.55 | 3.09 | 2.74 |
| k_seeds | ari | **0.7733** | 0.7722 | 0.7224 | 0.7733 | 0.7166 | 0 | 3.65 | 3.65 | 3.41 |
| k_wine | ari | 0.8975 | 0.8975 | **0.9150** | 0.8975 | 0.3711 | 0 | 4.24 | 4.24 | 4.32 |

Secondary metrics (selection; all in results.csv):

| dataset | A | B | C | note |
|---|---|---|---|---|
| balanced acc c_ai4i / c_bank / c_covertype | 0.827 / 0.820 / 0.529 | 0.831 / 0.817 / 0.570 | 0.854 / 0.809 / 0.540 | |
| accuracy c_adult / c_covertype | 0.865 / 0.737 | 0.858 / 0.781 | 0.869 / 0.739 | |
| RMSLE r_bike / r_house / r_diamonds | 0.419 / 0.120 / 0.108 | 0.359 / 0.131 / 0.097 | 0.274 / 0.138 / 0.098 | under RMSLE, C/B win bike |
| NMI k_iris / k_mall / k_penguins | 0.667 / 0.732 / 0.750 | 0.667 / 0.782 / 0.735 | 0.900 / 0.971 / 0.690 | same picture as ARI |

**Sensitivity of the ranking** (quality points):

| variant | A | B | C |
|---|---|---|---|
| as scored | 60.97 | 60.68 | 58.73 |
| relative-to-best normalisation | 80.00 | 79.92 | 77.02 |
| without `k_mall` (derived labels) | 58.21 | 57.77 | 54.15 |
| without `k_mall` and `r_house` (outlier) | 54.00 | 53.89 | 50.61 |
| C with California fixed (A's value substituted) | | | ≈ 62.71 |
| oracle: best of A/B/C per dataset | 65.09 | | |
| hybrid: A supervised + C clustering (as-is) | 63.42 | | |

---

## 6. Timing

| approach | median s | p90 s | max s | datasets > 20 s | headroom to 60 s at max |
|---|---|---|---|---|---|
| A_lean | 2.44 | 5.8 | 8.49 (covertype) | 0 | 51 s |
| B_portfolio | 9.34 | 32.8 | 33.86 (bike) | 6 | 26 s |
| C_rules | 3.44 | 9.2 | **52.28** (california, watchdog) | 1 | **7.7 s** |

- Clustering is ~1.6–3.4 s for everyone; with the 1.5 s interpreter + import floor it is effectively free.
- B is budget-driven: B's runs last ≈ 33 s whatever the CPU. Single-thread probe
  (`OMP_NUM_THREADS=1`): B bike 34.1 s, covertype 32.7 s (macro-F1 0.616 → 0.605), california 34.9 s. A covertype
  10.2 s, C covertype 17.9 s. So B degrades in quality, not in validity, on a slower CPU.
- C's watchdog (52 s) fired once. Its internal first-fold budget check cannot stop a single slow fit. On a
  slower grader, wall time + process start can approach 60 s if the OS is slow to tear down a 52 s process.
- Internal timestamps (from stdout): the constant-fallback predictions and the profile are on disk after
  0.45–0.8 s in every run (anytime behaviour works).

---

## 7. Profiling breakdown (identical shared profiler in A, B and C → identical scores, 9.959 / 10)

| field | mean score (all approaches) | notes |
|---|---|---|
| task, target, n_rows, n_columns, n_duplicate_rows | 1.000 | |
| total_missing_cells, col_missing, col_unique | 1.000 | pandas semantics match exactly (Telco `" "` blanks are not counted as missing, matching pandas; if the grader counts blank strings as missing, `TotalCharges` would be off by 9) |
| col_type | 0.998 | 1 error: credit-g `own_telephone` typed `constant` (declared categorical) |
| constant_columns | 0.944 | credit-g: `own_telephone` falsely listed as constant (F1 0). All planted constants (bank, abalone, iris) and the natural one (`soil_type_15` in covertype) were found |
| duplicate_columns | 1.000 | all 4 planted duplicate pairs found, no false pairs |
| id_like_columns | 1.000 | declared IDs plus undeclared `Product ID` (declared categorical!), `applicant_id`, `sample_no` and `specimen_id` all detected |
| target_profile | 1.000 | class counts / regression stats / k exact |
| datetime_columns | 1.000 | bike `datetime`, credit-g `application_date` |

Root cause of the one error: `cleaning.SENTINELS` contains `"none"`, so `clean_strings` turns the value `none`
into NaN. In credit-g, `own_telephone ∈ {none, yes}` becomes a single-level column, which is profiled constant and
**dropped from modelling** (action `dropped_constant`). `other_parties` and `other_payment_plans` lose their
majority level "none" to the NA token (harmless for trees, but a wrong profile `top_values` interpretation). The
same problem would hit Telco-like "No internet service" columns only if the value were literally "none". It is
the same code in all three approaches.

Stress-probe profiles: 1.000 except `s_telco_strnum` (0.996). Without declared types, `SeniorCitizen` (0/1 ints)
is inferred `numeric` while the base metadata declares it categorical. That is acceptable inference, and it counts
against the profile only because of how this scorer sets the expected type.

---

## 8. Requirements checklist R1–R20 (docs/01 §7)

| req | A_lean | B_portfolio | C_rules | evidence |
|---|---|---|---|---|
| R1 never crashes | pass | pass | pass | 0 non-zero exits / tracebacks in 90 runs (54 suite + 9 det + 27 stress) |
| R2 never times out (≤ 40 s at 2 threads) | pass (max 8.5) | pass (max 33.9; ~fills budget) | **fail** (52.3 s watchdog on california) | results.csv `wall_time`, `watchdog` |
| R3 output validity | pass | pass | pass | all 54 valid; k_ok on all clustering; original label dtype kept (e.g. `Machine failure` 0/1, `>50K`) |
| R4 shared adapter | pass | pass | pass | `io_contract.py`, `main.py`, `cleaning.py`, `data_profile.py`, `utils.py` byte-identical |
| R5 pinned minimal deps | pass | pass | pass | numpy/pandas/scipy/sklearn/threadpoolctl/joblib pinned `==`. Risk: very recent pins (sklearn 1.9.1, pandas 3.0.6, tested only on Python 3.14) may lack wheels for an older grader Python |
| R6 deterministic | pass | partial | pass | identical outputs on 2 runs of c_creditg / c_bank / k_drybean for all three. B's candidate set depends on CPU speed (covertype 1-thread 0.605 vs 0.616) |
| R7 classification quality | pass (16.44) | pass (16.75, best) | pass (16.50) | sk_hgb 15.66. A below sk_hgb on covertype |
| R8 regression quality | pass (24.24, best) | pass (23.95) | **fail** (19.49 < sk_hgb 23.77) | california 0 pts, house 0.75 |
| R9 clustering quality | pass (20.29) | pass (19.99) | pass (22.74, best) | sk_kmeans 17.08 |
| R10 balanced across tasks (min per task) | pass (min 0.57/0.57/0.58) | pass (0.62/0.58/0.62) | **fail** (reg min R² −0.04) | |
| R11 imbalance handling | pass | pass | pass | all beat sk_hgb on bank. On ai4i A beats it (0.864 vs 0.854), while B (−0.005) and C (−0.002) are marginally below in macro-F1. C has the best balanced accuracy on ai4i (0.854) |
| R12 skewed-target transform by CV | pass (OOF RMSE) | pass (log1p as candidate) | partial (rule `skew > 0.75`, no CV; good for RMSLE, hurt house via linear part) | |
| R13 leak-free CV | pass | pass | pass | label-free prep (train+test frequency encoding is label-free/transductive). C's target encoding is fitted inside folds. No test labels used |
| R14 profile correctness | partial (0.996) | partial | partial | `"none"` sentinel bug (section 7) |
| R15 profile cheap, always written | pass | pass | pass | profile on disk at 0.51 s median / 0.79 s max, also in the watchdog run |
| R16 speed (median < 15 s) | pass (2.4 s) | pass (9.3 s) but slowest | pass (3.4 s) | |
| R17 anytime behaviour | pass | pass | partial | constant fallback ≤ 0.8 s everywhere. C's first improved model came too late on california (constant fallback only) |
| R18 bounded threads / memory | pass | pass | **partial** | threadpool_limits + n_jobs honoured. C's TargetEncoder created a 12384 × 3425 dense matrix (feature blow-up) |
| R19 code clarity / decision logging | pass | pass | pass (rules_fired logged), but the california profile has no `modelling` block (watchdog exit) | |
| R20 contract variation | pass | pass | pass | stress: no `feature_types`, string numerics ("$1,234.50", "12,345", "?"), unseen test levels, train/test clustering variant, id_column present/absent. All valid |

### Stress probes (separate from the score; `stress_results.csv`)

| probe | A | B | C | base A / B / C |
|---|---|---|---|---|
| s_telco_missing (+20 % MCAR) | 0.720 | 0.716 | 0.715 | 0.728 / 0.728 / 0.734 |
| s_telco_strnum (no types) | 0.728 | 0.720 | 0.734 | (same as base: coercion works) |
| s_telco_unseen | 0.728 | 0.718 | 0.733 | |
| s_house_missing | 0.811 | 0.710 | 0.713 | 0.891 / 0.821 / 0.751 |
| s_house_strnum (no types) | 0.891 | 0.821 | 0.751 | identical to base |
| s_house_unseen | 0.871 | 0.808 | 0.742 | |
| s_penguins_missing | 0.632 | 0.729 | 0.582 | 0.751 / 0.653 / 0.581 |
| s_penguins_strnum (no types) | 0.751 | 0.653 | 0.581 | identical to base |
| s_penguins_unseen (train/test variant; ARI on 86 test rows) | 0.907 | 1.000 | 0.549 | |

All 27 stress runs were valid and fast (≤ 22 s). Numeric-as-string coercion and type inference are robust:
predictions are unchanged vs the declared-type base. Unseen levels cost ≤ 0.02. Heavy missingness costs
0.01–0.11.

---

## 9. Improvement suggestions (prioritised, for a later fix run; not implemented)

### Shared code (all approaches)
1. **Remove `"none"` from `cleaning.SENTINELS`** (also review `"unknown?"`, `"-"`). Treat such tokens as missing
   only in numeric-coercion contexts. Fixes the credit-g feature loss and the profile error (+0.04 profile pts
   per affected dataset, plus quality).
2. Add a per-fit cost guard: time a fit on a small subsample before full CV, and cap the feature count after
   encoding (e.g. assert p_after ≤ 5 × p_before). This class of bug cannot then cause a time-out.
3. Optional profile field: report blank / whitespace-only strings (`" "` in Telco TotalCharges) as suspected
   missing next to the pandas count, in case the grader counts them.

### A_lean
1. **Clustering feature relevance:** port C's sparse k-means weights, but only accept the reweighted partition
   when it drops features with near-zero weight and keeps categorical one-hots (iris +1.3 pts, mall +1.8 pts
   potential).
2. Consider a GMM-diag / KMeans choice with a stability check rather than BIC alone (drybean +0.3 pts).
3. Multiclass with rare classes (`c_covertype` 0.566 < default HGB 0.593): use plain `class_weight="balanced"`
   or a less capped weight, more leaves / lower min_samples_leaf for n > 10k, or blend ExtraTrees for ≥ 5 classes.
4. Extend the linear blend to n < 2000 (credit-g: logreg alone 0.675 > A 0.670; B/C ≈ 0.70).
5. Keep log1p in the candidate set by an RMSLE tie-break for count targets (bike RMSLE 0.42 vs 0.27), in case the
   metric is RMSLE.

### B_portfolio
1. **Cut run time**: reduce the budget to ~20 s, drop kNN and `hgb_c` for n > 10k, and stop after
   `hgb_a + logreg + extratrees` when the OOF gain of the last candidate < 0.002. That recovers ~1–1.3 speed pts at
   near-zero quality cost (most ensembles are dominated by 2–3 members).
2. Robustify linear members on heavy-tailed targets: clip linear predictions to the training target range *in
   log space*, and cap the linear ensemble weight when test rows fall outside the training feature range (house
   0.82 → ~0.89).
3. Add a feature-filtered clustering view (sparse k-means), so the consensus is not dominated by noise-diluted
   views.

### C_rules
1. **Critical:** `TargetEncoder(target_type="continuous")` for regression (and "binary"/"multiclass" explicitly for
   classification). Recovers ≈ 4 pts on `r_california` and removes a latent time-out on any integer-valued
   regression target with a > 30-level categorical.
2. Emit a cheap model (e.g. HGB 100 iterations on ≤ 5k rows) *before* the linearity probe and CV, so a slow main
   fit never leaves only the constant fallback. Check the budget inside the fold loop, per model.
3. Regression linear blend: clip in log space, or require the probe to beat HGB by a margin before blending
   (house 0.75 → ~0.89).
4. Clustering guards: (a) exclude one-hot columns from the sparsity penalty, or weight them like numerics (penguins
   0.58 → 0.75). (b) Accept GMM over KMeans only when the two partitions disagree little (high ARI) and BIC
   improves strongly (seeds 0.72 → 0.77).

### Recommended final submission
**Hybrid on A's skeleton:**
- **Supervised:** A's HGB pipeline (fast, never near the limit, best on regression and imbalanced binary),
  plus two cheap ingredients from B/C: an ExtraTrees/linear blend for multiclass and n < 2000 tables, chosen by OOF
  (B's evidence on covertype and credit-g), and C's cross-fitted target encoding for > 30-level categoricals,
  *with the target-type fix*.
- **Clustering:** C's sparse k-means + GMM pipeline with the guards above, falling back to A's KMeans
  partition when the guards reject. This is C's biggest advantage (+2.5 pts, robust to irrelevant features, which
  the spec explicitly lists).
- **Shared fixes:** the sentinel fix and the per-fit cost guard.

Estimated on this suite: ≈ 63–64 quality (the as-is "A supervised + C clustering" combination already scores
63.42, before penguins/seeds guards), 9.98 profiling, ~4.8 speed → **≈ 78 / 100** vs 75.8 for A alone.
If no further development time is available, submit **A as-is**: highest total, largest time headroom, and no
failure mode found.

---

## 10. Caveats

- **The scoring model is ours.** Real rules may use other metrics (accuracy, RMSE, RMSLE, NMI) or rank-based /
  team-relative normalisation, give speed credit relative to other teams, and use a different profile schema
  (our profile fields are scored on our key names; a different schema means `map_profile` must change, see
  docs/02). The secondary metrics and the sensitivity table in section 5 show the A ≈ B > C ordering is stable
  across the variants tried. Under RMSLE, C/B would gain on bike.
- **Differences of 0.3–0.5 quality points are within noise.** A single test split per dataset was used. For
  example, the `r_house` ranking flips if the Ames outlier is not in the test set. Classification differences
  between A, B and C are ≤ 0.015 macro-F1 except covertype (0.05) and credit-g (0.034).
- **`k_mall` labels are derived** (KMeans k=5 on raw Income/Spending), because the Kaggle file has no labels.
  This favours methods that discard Age/Gender. Excluding Mall, C's clustering lead shrinks from 2.5 to
  ~0.6 pts. The real organiser labels are unknown.
- **Timing:** measured on a fast 24-core desktop with a 2-thread cap, after a warm-up run. A cold first run
  (`.pyc` compilation, antivirus DLL scan) can add tens of seconds (59 s first sklearn import was measured in
  docs/01). The grader should be asked about or protected against this (e.g. `py -m compileall` at install).
- **Injected challenges** (noise features, duplicates, constants, IDs, MCAR) are our guess at the private data's
  "issues". The suite has 4 binary-imbalanced sets, which may over-weight that issue relative to the real 6.
- The AI4I failure-mode flags were dropped (leakage). If the hidden version keeps them, all approaches score
  ≈ 0.98 there.
- `sklearn.fetch_openml` hung on this machine. Data was fetched from the OpenML REST API / UCI / a GitHub mirror
  of the Kaggle Mall file. Versions are documented in SUITE.md.
