# 01 — Research Report (Phase 1: Planning, no model code)

EECE 490 "Unknown Dataset Hackathon". Author: Subagent 1 (Planner + Dev). Date: 2026-10-20.

Legend used throughout:
- **[SPEC]** = stated in `unknown_dataset_student_announcement.docx` (the only primary source we have).
- **[INFERRED]** = our assumption / best guess; must be re-checked when the kickoff material appears.
- **[MEASURED]** = measured on our dev machine (Windows 11, 24 cores, Python 3.14.7).

---

## 0. What the spec actually says (complete summary)

The spec is short. Everything it states:

| Topic | Statement [SPEC] |
|---|---|
| Goal | One program that analyses an unfamiliar table and solves its ML task; must work on new datasets automatically. |
| Tasks | Classification (predict a class), regression (predict a number), clustering (group rows without labels). |
| Metadata | "Each dataset tells you its task, feature types, and, for clustering, the required number of groups." |
| Labels | Supervised tasks include training labels; evaluation labels are private. |
| Data issues | Missing values, differently scaled features, categorical columns, constant or duplicate columns, irrelevant features, imbalanced classes, nonlinear relationships. "Described in the rules released at kickoff." |
| Outputs | "return predictions plus a structured data profile." |
| Scope examples | Telco Churn (clf), AI4I 2020 Predictive Maintenance (clf), House Prices (reg), Bike Sharing Demand (reg), Mall Customer Segmentation (clustering). |
| Kickoff material | 9 practice datasets, a working baseline for all 3 tasks, the input/output contract, a local checker, complete public scoring rules. **We do not have any of these.** |
| Scoring | 18 private datasets. Model quality 85, data profiling 10, speed 5. Classification, regression, clustering contribute equally to model quality. Speed credit depends on useful model performance. 60 s limit per dataset, same CPU environment. Hidden datasets stay within published task families and size limits. |
| Environment | "The final program must run independently in the published assessment environment." Any libraries/AI allowed during development. |
| Submission | One public GitHub repo URL; runnable code, pinned dependencies, the standard entry point, team names, README with approach + how to reproduce. Commit frozen at deadline. Record assumptions in README. No questions to organisers. |
| Deadline | Doc says six hours; **corrected by the user: Wednesday midnight**. |

---

## 1. What is actually graded

### 1.1 Point decomposition [SPEC + INFERRED]

| Component | Points | Notes |
|---|---|---|
| Model quality | 85 | Equal thirds: ~28.3 pts each for classification / regression / clustering. 18 datasets ⇒ **[INFERRED] 6 per task type**, ≈ **4.7 pts per dataset**. |
| Data profiling | 10 | "Correct data profiling" — profile fields compared to ground truth (likely exact/tolerance match on things like row count, column types, missing counts). ≈ 0.56 pts per dataset if spread evenly. |
| Speed | 5 | Conditional on "useful model performance" — probably a per-dataset gate (e.g. must beat baseline or a threshold) and then credit by runtime relative to other teams / to the limit. ≈ 0.28 pts per dataset. |

### 1.2 How model quality is probably converted to points [INFERRED]

Likely options (in order of plausibility):
1. **Normalised against the baseline and/or a reference ("oracle") score**: `pts = clip((score - baseline) / (reference - baseline), 0, 1)`. Common in AutoML-style contests; makes metrics comparable across datasets.
2. **Rank among teams** per dataset.
3. **Raw metric** scaled to points (e.g. macro-F1 × weight).

Implication under all three: the marginal value of beating the baseline by a solid margin on **every** dataset is greater than squeezing out the last 0.5% on a few. A clean, well-tuned gradient-boosting pipeline should sit near the top of any "reference" range.

### 1.3 What a failure costs [INFERRED]

- Crash, timeout (>60 s), malformed / misaligned predictions, wrong row count or wrong ID order ⇒ **almost certainly 0 for that dataset's model quality, and likely 0 for its speed and maybe its profile** too. That is ≈ 5.5 pts lost per failure — more than the entire speed component.
- A degenerate model (e.g. constant prediction) probably scores ≈ baseline-or-below ⇒ ≈ 0 quality points but **still keeps profile points**.
- Therefore the ordering of priorities is:
  1. **Never fail**: always write a valid output file, even if every model fails (constant/majority fallback, KMeans fallback).
  2. **Never time out**: internal budget well under 60 s.
  3. **Correct profile** (10 pts are nearly "free" if we are careful and match the contract).
  4. **Strong model quality** on all three task types (clustering is a full third — do not neglect it).
  5. Speed (only 5 pts, gated on quality — do not trade much accuracy for it; but being reasonably fast, e.g. < 15 s, is cheap insurance).

### 1.4 Repo requirements [SPEC]

- Runnable code, **pinned dependencies** (`requirements.txt` with `==` versions), "the standard entry point" (defined at kickoff; we will provide `main.py` + a `solve()` function and adapt), **team names**, README with approach, reproduction steps, and assumptions.
- The frozen commit is used for every experiment ⇒ no reliance on network access, downloads, or files outside the repo at run time; no trained artifacts needed (we fit per dataset).

---

## 2. Likely metrics and what to optimise

### 2.1 Classification [INFERRED]

Candidates: accuracy, **macro-F1**, **balanced accuracy**, ROC-AUC (binary), log-loss.
- The spec explicitly lists **imbalanced classes** as a data issue and includes AI4I 2020 (≈3.4% failures) and Telco churn (≈26.5% positive). With accuracy, a majority-class predictor gets 96.6% on AI4I — organisers who mention imbalance almost certainly use an imbalance-aware metric: **macro-F1 (most likely) or balanced accuracy**.
- Output is "predict a class" ⇒ label predictions, not probabilities. AUC/log-loss are therefore unlikely (unless the contract asks for probabilities; we should be able to emit `predict_proba` too if asked).

What to optimise:
- Use class weights or `class_weight="balanced"`-style sample weights, and/or **post-hoc decision threshold / per-class prior adjustment tuned on out-of-fold probabilities** to maximise macro-F1.
- Robust compromise: optimise **macro-F1 on OOF predictions** via per-class probability scaling (multiply class probs by weights w_c, argmax). This also improves balanced accuracy and costs little accuracy when classes are balanced. If imbalance is mild (minority ≥ ~35%), threshold tuning changes little — keep it but guard with a "only apply if OOF gain > ε" rule to avoid overfitting on tiny data.
- Preserve original label dtype/values exactly in the output (strings like "Yes"/"No", ints 0/1, etc.).

### 2.2 Regression [INFERRED]

Candidates: **R²** (most likely — scale-free, comparable across datasets), RMSE/MAE (scale-dependent; would need normalisation), RMSLE (Kaggle metric for Bike Sharing and the log-RMSE for House Prices).
- R² / RMSE reward getting large values right; RMSLE rewards relative accuracy.
- Strategy that is robust across metrics: **choose target transform by CV on the actual metric we assume (R²/RMSE in original space)** but include `log1p(y)` as a candidate when y ≥ 0 and strongly right-skewed (skew > ~1, e.g. counts, prices). Training on log1p and back-transforming tends to *also* do well in RMSE for multiplicative targets (House Prices), and is far better for RMSLE. Pick by OOF RMSE in the original space, with log-target winning ties when skewed.
- Clip predictions to the observed training range (with small margin) and to ≥ 0 for non-negative targets — protects against wild extrapolation by linear models.

### 2.3 Clustering [INFERRED]

Given "the required number of groups" is supplied, evaluation is almost certainly **external** against hidden labels: **ARI** (most likely) and/or **NMI / AMI**; possibly a purity / Hungarian-matched accuracy. Less likely: internal metrics (silhouette) — but then k wouldn't need to be fixed.
- Labels permutation-invariant ⇒ any integer labels `0..k-1` are fine. **Must output exactly k clusters** (no empty clusters, no "-1" noise labels as from DBSCAN/HDBSCAN).
- What matters: preprocessing (scaling, encoding categoricals, dropping ID/constant/useless columns, handling skew) matters far more than the algorithm. The hidden labels are probably a generating class (synthetic blobs, or a real dataset's class column with the label removed, e.g. Iris/Wine/Seeds-style, or Mall Customers' 5 visible groups).
- Robust default: standardise (or robust-scale + log-transform skewed positives), one-hot categoricals with down-weighting, **KMeans with many inits (k fixed)**; consider GMM/Agglomerative(Ward)/Spectral as alternatives chosen by an internal criterion. Note: silhouette tends to favour KMeans-like convex clusters; that aligns with ARI on blob-like data but can mislead on elongated/manifold data. Feature selection for clustering (dropping noise features) can give large ARI gains when there are "irrelevant features" (which the spec explicitly mentions) — e.g. Laplacian score / variance-of-unimodality (dip test / bimodality) filters.

### 2.4 Robust overall strategy

- Classification: gradient boosting + class weighting + OOF macro-F1-tuned class-probability weights (guarded).
- Regression: gradient boosting with CV-selected raw vs. log1p target; clipping.
- Clustering: careful preprocessing + KMeans(k) baseline, with optional validity-driven alternatives; never fewer/more than k labels.

---

## 3. Data challenges and handling

| Challenge | Detection | Handling (best practice) |
|---|---|---|
| Missing values [SPEC] | `isna` per column; also sentinel strings ("", "?", "NA", "null", "None", " ") | Trees (HistGB/LightGBM): native NaN handling. Linear/kNN/clustering: median impute + **missing-indicator** for columns with >~5% missing. Categorical: "__MISSING__" category. Drop columns with > 95–98% missing. |
| Differently scaled features [SPEC] | ranges/std | Irrelevant for trees; StandardScaler / RobustScaler (+ `log1p` / Yeo-Johnson for heavy skew) for linear, kNN, SVM, and **all clustering**. |
| Categorical columns [SPEC] | metadata feature types (authoritative) + dtype object/bool/low-cardinality ints | Low card (≤ ~30): one-hot (linear/clustering) or native categorical (HistGB `categorical_features`, LightGBM). High card: ordinal/target encoding (CV-safe, e.g. sklearn `TargetEncoder` with internal cross-fitting) or frequency encoding. Unseen test categories → "unknown" bucket (`handle_unknown="ignore"` / encoded as NaN). |
| Constant columns [SPEC] | `nunique(dropna=False) <= 1` (also near-constant > 99.9% one value) | Drop; report in profile. |
| Duplicate columns [SPEC] | hash of column values (`pd.util.hash_pandas_object`) after normalising; check exact equality on hash collision | Drop all but first; report pairs in profile. (Also optionally detect perfectly correlated numeric copies, e.g. linear rescalings — report as "redundant" but not necessarily "duplicate"). |
| Irrelevant features [SPEC] | — | Trees are robust. Linear: regularisation. Clustering: **most dangerous** — noise features dilute distances; use feature filtering (e.g. drop features with near-zero "clusterability" signal; or compare KMeans stability with/without each feature group). Optional permutation-importance pruning for supervised models if time permits. |
| Imbalanced classes [SPEC] | class proportions, min class count | Stratified CV; class weights; threshold/prior tuning for macro-F1; ensure min class count ≥ n_folds, else reduce folds / use repeated holdout. Rare classes with 1 sample: keep in training, can't be in every fold. |
| Nonlinear relationships [SPEC] | — | Gradient boosting / random forests; for linear models add spline/interaction features only if cheap. |
| ID columns [INFERRED] | all-unique ints/strings, monotone increasing integer index, name matches `id`, `*_id`, `customerID`, `UDI`, `Product ID` | Drop from features (they carry no signal and hurt clustering badly). Keep for output row identification if the contract needs IDs. Careful: an all-unique **float** column is a real feature, not an ID. |
| Datetime columns [INFERRED] (Bike Sharing `datetime`) | parse attempt on object columns (≥ 90% parse success, plausible years) | Expand to year, month, day, dayofweek, hour, dayofyear, is_weekend (+ elapsed time numeric). Drop raw string. Hour is the strongest bike-sharing feature. |
| High-cardinality categoricals [INFERRED] | nunique > 50 and ratio > 0.05 | Target/frequency encoding; HistGB limit is 255 categories per feature — collapse rare levels to "other". |
| Text-like columns [INFERRED] | object, high card, avg length > ~30 chars, many spaces | Drop by default or cheap features (length, word count). TF-IDF only if clearly beneficial (unlikely in scope). |
| Numeric stored as strings [INFERRED] (Telco `TotalCharges` has " " blanks) | `pd.to_numeric(errors="coerce")` success ratio ≥ 95% of non-null | Convert; failures → NaN. Also strip `$`, `,`, `%`. |
| Boolean-like strings | {"yes","no","true","false","y","n"} | Map to 0/1 (or keep as categorical — both fine for trees). |
| Leakage-ish columns [INFERRED] (AI4I: `TWF`,`HDF`,`PWF`,`OSF`,`RNF` failure modes; Bike: `casual`,`registered` sum to `count`) | Near-perfect single-feature OOF score | If the column is present in **both train and test**, using it is legitimate and wins points — do **not** drop it. Only drop columns absent from test. Report "suspected leakage" in profile for transparency. |
| Tiny datasets [INFERRED] (Mall: 200 rows; Iris-like 150) | n < ~500 | More CV folds/repeats, stronger regularisation, include simple models (logistic/ridge, RF) in candidates; avoid aggressive threshold tuning. |
| Large datasets near size limit [INFERRED] (e.g. 50k–200k rows, 100+ cols) | n×p | Subsample for model selection, fit final model on full data with fixed iteration count; HistGB bins make it scale; avoid SVM/kNN/spectral/agglomerative O(n²) on large n (subsample + predict/assign). |
| Target column in test / label encoding | — | Ensure target column is excluded from features; regression target may be int (counts). Classification labels may be strings, ints, bools. |
| Mixed-type / dirty columns | multiple types in object column | Coerce by majority type; fall back to categorical. |
| Duplicate rows | `duplicated()` | Report in profile; keep for training (they reflect the distribution). |
| Inf values | `np.isinf` | Replace with NaN. |
| Test categories/columns differ | — | Align test to train columns; missing columns → NaN; extra columns dropped. |

---

## 4. Data profile (10 pts)

"Correct data profiling" implies an automatic comparison against ground truth. Likely checked fields: shape, per-column type, missing counts, constant/duplicate columns, task type, target info. Until the contract arrives we emit a **superset** in a stable JSON schema; adapting key names is a mapping in one function.

Key principle: profile the **raw training table as given** (before any dropping/conversion), because ground truth is computed by organisers on the raw file. Also include test stats separately.

### 4.1 Proposed JSON schema (`profile.json`)

```json
{
  "schema_version": "1.0",
  "dataset_name": "string",
  "task": "classification | regression | clustering",
  "task_source": "metadata | inferred",
  "target": "string | null",
  "n_rows": 1234,
  "n_rows_train": 1000,
  "n_rows_test": 234,
  "n_columns": 21,
  "n_features": 20,
  "n_duplicate_rows": 0,
  "total_missing_cells": 11,
  "missing_cell_ratio": 0.0005,
  "column_type_counts": {"numeric": 12, "categorical": 7, "datetime": 0, "text": 0, "id": 1, "boolean": 0, "constant": 0},
  "columns": [
    {
      "name": "TotalCharges",
      "raw_dtype": "object",
      "declared_type": "numeric | categorical | null",
      "inferred_type": "numeric | categorical | boolean | datetime | text | id | constant",
      "n_missing": 11,
      "missing_ratio": 0.0016,
      "n_unique": 6531,
      "unique_ratio": 0.93,
      "is_constant": false,
      "is_id_like": false,
      "duplicate_of": null,
      "numeric_stats": {"mean": 0, "std": 0, "min": 0, "q25": 0, "median": 0, "q75": 0, "max": 0, "skewness": 0, "n_zeros": 0, "n_negative": 0, "n_outliers_iqr": 0},
      "categorical_stats": {"top_values": [{"value": "No", "count": 5174}], "n_rare_levels": 0},
      "unseen_test_levels": 0,
      "action": "kept | dropped_constant | dropped_duplicate | dropped_id | dropped_missing | converted_numeric | expanded_datetime | one_hot | ordinal | target_encoded"
    }
  ],
  "constant_columns": ["..."],
  "duplicate_columns": [["colA", "colB"]],
  "id_like_columns": ["customerID"],
  "high_missing_columns": ["..."],
  "high_cardinality_columns": ["..."],
  "datetime_columns": ["..."],
  "target_profile": {
    "classification": {"n_classes": 2, "class_counts": {"No": 5174, "Yes": 1869}, "class_ratios": {}, "imbalance_ratio": 2.77, "is_imbalanced": true, "minority_class": "Yes"},
    "regression": {"mean": 0, "std": 0, "min": 0, "max": 0, "skewness": 0, "is_nonnegative": true, "is_integer": true, "log_transform_suggested": true},
    "clustering": {"n_clusters_required": 5}
  },
  "warnings": ["suspected_leakage: TWF", "..."],
  "modelling": {
    "approach": "A",
    "selected_model": "HistGradientBoostingClassifier",
    "cv_metric": "macro_f1",
    "cv_score": 0.0,
    "candidates": [{"model": "...", "score": 0.0, "fit_seconds": 0.0}],
    "fallback_used": false,
    "elapsed_seconds": 0.0
  }
}
```

Type-inference rules (deterministic, documented in README):
- Use **declared feature types from metadata** whenever given (they are likely the ground truth for "type" fields).
- Otherwise: bool dtype or 2 values in yes/no-like set → boolean; numeric dtype (or ≥95% coercible strings) → numeric, unless integer with ≤ 10–15 unique values and declared otherwise; object with ≥ 90% datetime-parse → datetime; object with unique_ratio > 0.95 and short tokens → id; object with long multi-word strings → text; else categorical; nunique ≤ 1 → constant.
- Use standard pandas semantics for counts (`isna`, `nunique(dropna=True)`), and round floats to 6 decimals — matching what a checker written in pandas would compute.

---

## 5. Proposed I/O contract assumption [INFERRED]

### 5.1 Assumed dataset directory layout

```
<dataset_dir>/
  train.csv            # features + target column (for clustering: features only, or all rows)
  test.csv             # features only (plus optional id column)
  metadata.json        # {"task": "...", "target": "...", "feature_types": {"col": "numeric|categorical|..."},
                       #  "n_clusters": k, "id_column": "...", "metric": "..."}
```
Clustering variant: possibly a single `data.csv` (labels for all rows) or `train.csv` only. We handle: if no test file, predict for train rows.

### 5.2 Entry points

```
py main.py --data <dataset_dir> --out <out_dir> [--time-budget 45]
py main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>
```
Python API: `from solution import solve; solve(dataset_dir, out_dir) -> dict`.

Outputs in `<out_dir>/`:
- `predictions.csv` — one row per test row, same order; columns `[id_column?] , prediction` (or the target column name). Clustering: `cluster` int 0..k-1.
- `profile.json` — schema in §4.1.
- (optional) `run_log.json` — timings, chosen model, fallbacks.

### 5.3 Adapter layer (single swappable file)

All contract-specific logic lives in **`src/io_contract.py`**:
- `load_task(args) -> Task` where `Task = {task_type, X_train: DataFrame, y_train: Series|None, X_test: DataFrame, test_ids, target_name, feature_types, n_clusters, metric_hint, raw_train, raw_test}`.
- `write_outputs(task, predictions, profile, out_dir)`.
- `map_profile(profile_internal) -> dict` (key renaming / subsetting to match the official schema).

The three approaches only consume `Task` and return `(predictions ndarray, profile dict)`. When the kickoff contract appears, only `io_contract.py` (and maybe a profile key-mapping) changes. Metadata key names are resolved with a tolerant alias table (`task|task_type|problem_type`, `target|target_column|label`, `n_clusters|k|num_clusters|n_groups`, ...).

---

## 6. Time budget strategy

### 6.1 Measurements [MEASURED]

- **Cold first import of `sklearn.ensemble` took 59 s** on this machine (first-ever import after install: .pyc compilation + Windows Defender scanning DLLs). Warm: ~2.4 s. A second cold-ish import of `sklearn.datasets/cluster` took ~13.6 s. This is a real hazard: if the assessment env is a fresh container and imports count toward 60 s, heavy import trees are dangerous. We cannot fully control it, but we can: import only what each path needs (lazy imports inside functions), avoid importing big optional libraries (catboost ~100 MB wheel, xgboost ~49 MB) unless used, and **start the internal clock at process start** and measure import time as part of the budget. README should recommend running once to warm (`py -m compileall`), but we must not rely on it.
- HistGradientBoostingClassifier, 20k×30, 200 iters: 0.4–0.8 s (24 threads vs 2 threads nearly equal — small data does not benefit from many threads). KMeans(k=5, n_init=10) 20k×30: 0.2 s. RandomForest 200 trees, n_jobs=2: 6.8 s (RF is comparatively slow).

### 6.2 Policy

1. **Hard internal deadline**: default total budget **40 s** wall-clock from process start (configurable `--time-budget`), leaving ≥ 20 s margin for a weaker CPU, slow disk, and imports. Model-selection gets what is left after load+profile (~25–30 s), and a reserve of ~8 s is held back for the final refit + writing outputs.
2. **Fallback first**: immediately after loading, fit a cheap fallback (classification: HistGB with 100 iters or logistic regression / majority class; regression: HistGB 100 iters or median; clustering: KMeans n_init=3) and **write `predictions.csv` + `profile.json` to disk at once**. Every better model later overwrites the file atomically (write temp, `os.replace`). If the process is killed at 60 s, a valid answer is already on disk (if the grader reads files rather than return values).
3. **Global try/except** around every stage; any exception → keep last good predictions. Final `except BaseException` path writes fallback predictions (constant) so output is always valid.
4. **Adaptive subsampling**: model selection on a stratified subsample (e.g. ≤ 20k rows) when n is large; final fit on full data. Estimate cost per candidate from the first fit (time per 1k rows × iterations) and skip candidates that would overrun.
5. **Threads**: `n_jobs` = min(4, os.cpu_count()) by default — the grader CPU is unknown and could be 2–4 cores; oversubscription (sklearn OpenMP + joblib processes) on few cores is a classic slowdown. Use threadpoolctl / `OMP_NUM_THREADS` set before importing numpy. Avoid process-based joblib (spawn on Windows re-imports everything — expensive).
6. **Avoid** O(n²) algorithms (SVC, kNN on big n, Agglomerative/Spectral on > ~5k rows) unless subsampled; avoid nested CV.
7. **Early stopping** for boosting (internal validation) to bound iterations.
8. Cap CV folds by time: 5-fold when cheap, 3-fold or single holdout when large.
9. Profile computation must be O(n·p) and vectorised (hashing for duplicate columns, not pairwise comparisons).

### 6.3 Dependencies

- Wheels exist for Python 3.14 on win_amd64: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10 [MEASURED via `pip download`]. But **the assessment environment is unknown** (may be Linux, may be a different Python, may only allow `pip install -r requirements.txt` and could lack internet). Risk ranking: sklearn/numpy/pandas/scipy (lowest) < lightgbm (small, mostly OK) < xgboost < catboost (huge, slow import).
- Recommendation: **core path uses only numpy/pandas/scipy/scikit-learn** (sklearn's HistGradientBoosting is competitive with LightGBM on these data sizes). LightGBM may be an *optional* accelerator guarded by `try: import lightgbm` with sklearn fallback, so a missing package never crashes. Pin exact versions in `requirements.txt`; note Python version in README. Also keep code compatible with older sklearn (≥1.3) / pandas 2.x in case the env pins older versions (avoid brand-new APIs; e.g. `TargetEncoder` needs sklearn ≥1.3; `HistGradientBoosting(categorical_features="from_dtype")` needs ≥1.4 — wrap with fallbacks).

---

## 7. Requirements checklist for approaches

Ranked by importance. The Validator (Subagent 2) should score each approach on these. "MUST" items are pass/fail gates.

**Tier 0 — Gates (MUST; any failure disqualifies the approach)**
1. **R1 Never crashes**: on all validation datasets (incl. adversarial ones: all-missing column, single-class-in-fold, string target, unseen test categories, 1-feature table, 0 numeric features, datetime column, duplicate column names?), produces valid output. Top-level exception guard + fallback.
2. **R2 Never times out**: worst-case wall time ≤ 40 s on our machine **with threads limited to 2** (simulating a weak grader), including Python start-up and imports; ≤ 60 s under a 2× slowdown factor.
3. **R3 Output validity**: correct row count and order, correct label domain (classification labels from the training label set with original dtype; regression finite floats; clustering exactly k distinct ints 0..k-1 when n ≥ k).
4. **R4 Uses the shared adapter** (`io_contract.py`) — no contract assumptions inside the approach code.
5. **R5 Pinned, minimal deps**; works with only numpy/pandas/scipy/sklearn (optional libs guarded).
6. **R6 Deterministic**: fixed seeds; same input → same output.

**Tier 1 — Model quality (the 85 pts)**
7. **R7 Classification quality**: mean macro-F1 (primary) and balanced accuracy / accuracy (secondary) across the validation suite, compared with a baseline (e.g. one-hot + logistic regression / default RF).
8. **R8 Regression quality**: mean R² (primary) and RMSLE where applicable (secondary), vs baseline.
9. **R9 Clustering quality**: mean ARI (primary) and NMI (secondary), vs baseline (StandardScaler + KMeans(k)).
10. **R10 Balanced across task types**: no task family with mean normalised score below baseline; worst-case dataset matters (report min, not only mean).
11. **R11 Imbalance handling**: class weighting and/or OOF-tuned decision rule for macro-F1.
12. **R12 Target transform for skewed regression** chosen by CV.
13. **R13 Leak-free CV**: all preprocessing fit inside folds (or label-free only); no test-label use; target encoding cross-fitted.

**Tier 2 — Profiling (10 pts)**
14. **R14 Profile completeness and correctness**: all §4.1 fields present; counts match pandas ground truth on raw data; types use declared metadata when present; constant/duplicate/ID detection verified on planted cases.
15. **R15 Profile is cheap** (< 2 s on 100k×100) and always written even if modelling fails.

**Tier 3 — Speed and engineering (5 pts + maintainability)**
16. **R16 Speed**: median runtime per dataset (lower is better; target < 15 s) given quality is good.
17. **R17 Anytime behaviour**: valid predictions on disk early; improved later.
18. **R18 Bounded threads / memory**: respects `n_jobs` cap, no memory blow-up on wide one-hot (cap one-hot levels, use sparse where needed).
19. **R19 Code clarity**: readable, documented, logs decisions into `profile.modelling`; README-ready explanation.
20. **R20 Robust to contract variation**: handles absent test file (clustering), id column present/absent, target given as name or index, feature types missing from metadata.

---

## 8. Three candidate approaches (proposals for approval)

All three share: `io_contract.py` adapter, `profile.py` (single shared profiler — profiling is not where approaches should differ), a shared `clean.py` (type coercion, ID/constant/duplicate dropping, datetime expansion), the anytime fallback writer, and the time-budget manager. They differ in the modelling philosophy.

### Approach A — "Lean & robust single strong model"

Philosophy: one well-configured, proven model family per task, minimal search, very fast, extremely hard to break.
- **Classification**: `HistGradientBoostingClassifier` (native categoricals, native NaN, early stopping, `class_weight="balanced"` when imbalance ratio > ~3), sensible fixed hyper-params (lr 0.06–0.1, max_leaf_nodes 31, l2 1.0, min_samples_leaf scaled with n). One OOF pass (3–5 fold) used only to tune per-class probability weights for macro-F1 (guarded by gain threshold). Tiny-data rule: n < 300 → blend with logistic regression.
- **Regression**: HistGBRegressor; raw vs log1p target decided by a quick 3-fold CV when y ≥ 0 and skewed; prediction clipping.
- **Clustering**: drop ID/constant; numeric → median impute + log1p for skewed positives + StandardScaler; categoricals one-hot scaled by 1/sqrt(levels) ; `KMeans(k, n_init=20–50, k-means++)`. If p is large (>~30), PCA to 90–95% variance before KMeans.
- Expected runtime: 2–10 s. Strength: speed + robustness; weakness: no adaptivity when another model family is clearly better (e.g. linear-ish small data, non-convex clusters).

### Approach B — "Time-budgeted AutoML portfolio + ensembling"

Philosophy: let validation decide. Evaluate a diverse portfolio under a time budget, then ensemble the best.
- **Supervised portfolio** (ordered cheapest/highest-prior first, each with 1–3 configs): HistGB (2–3 configs), RandomForest / ExtraTrees (limited trees), Logistic/Ridge (one-hot + scaling, + spline features optional), kNN (scaled, only n < 20k), optional LightGBM if installed. 5-fold (or 3-fold) stratified CV with **OOF predictions stored**.
- **Ensembling**: greedy forward ensemble selection (Caruana et al. 2004) on OOF predictions with replacement, optimising the assumed metric (macro-F1 after weight tuning / R²). Final models refit on full data (or average the fold models to save time — fold-averaging avoids refit cost and is a sound bagging approach).
- **Regression**: target transform (raw / log1p) treated as part of the portfolio configs.
- **Clustering**: portfolio of preprocessing variants (standard / robust+log / PCA-whitened / feature-filtered) × algorithms (KMeans, GMM full/diag, Agglomerative Ward, Birch; Spectral for n < 3k), all forced to k clusters; pick by a **consensus/stability criterion**: compute pairwise ARI between all candidate partitions and select the medoid partition (the one most agreeing with others), tie-broken by silhouette / Calinski-Harabasz on the common scaled space. Optional: consensus clustering (co-association matrix + agglomerative) when n small.
- Expected runtime: 15–35 s (budget-controlled). Strength: best mean quality on varied data; weakness: more moving parts, overfitting of selection on tiny data, closer to time limit.

### Approach C — "Profile-driven meta-rules (expert system) + specialised pipelines"

Philosophy: use the data profile to *decide* the pipeline deterministically, like an experienced data scientist, encoding heuristics learned from the validation suite (a small hand-built meta-learner). Minimal CV, but targeted choices.
- **Rules examples**: n < 500 & p small → regularised linear + RF blend with repeated CV; many categoricals with high card → target encoding + HistGB; datetime present → calendar expansion + boosting with cyclical features; strong skewed non-negative target → log1p; imbalance ratio > 5 → balanced weights + threshold tuning; wide data (p > 100, n small) → L1/L2 linear + feature selection; mostly-numeric smooth data → also try an MLP/kernel ridge only when n small.
- Optionally a learned **meta-selector**: train a tiny decision tree on meta-features (n, p, %cat, skew, imbalance, linearity score from quick ridge-vs-tree comparison) → best pipeline label, using results from ~40–60 public datasets in the validation suite. Fallback to the rules when uncertain.
- **Clustering (distinct angle)**: assume the hidden labels are "natural groups"; focus on **feature relevance**: score each feature by unimodality (bimodality coefficient / dip-like statistic) and by its contribution to a KMeans partition's between-cluster variance; iteratively drop noise features (like a wrapper / "sparse k-means" Witten & Tibshirani 2010 feature weighting), then choose KMeans vs GMM by BIC on the selected features. Sparse K-means is particularly suited to the spec's "irrelevant features" item.
- Expected runtime: 3–15 s. Strength: fast, interpretable, explicit handling of each data issue; weakness: heuristics may be miscalibrated for hidden data; needs a good validation suite to tune rules.

### Trade-off summary

| | A Lean | B AutoML portfolio | C Profile-driven rules |
|---|---|---|---|
| Expected quality | Good everywhere | Best on average | Good; best when rules fit |
| Robustness | Highest | Medium-high | High |
| Runtime | Fastest | Slowest | Fast |
| Overfitting risk | Low | Medium (selection on small data) | Medium (rules tuned on our suite) |
| Clustering idea | Careful scaling + KMeans | Multi-candidate + consensus selection | Sparse feature-weighted KMeans/GMM |

Likely final: A as the safety core, enriched with the parts of B/C that the Validator proves useful (hybrid). Keeping them separate in Phase 2 lets the Validator compare cleanly.

---

## 9. Validation plan (for Subagent 2)

### 9.1 Suite composition (aim: 3–4× the private mix, ≈ 8–10 datasets per task)

Sources: `sklearn.datasets` (offline, built-in), `sklearn.datasets.fetch_openml` (cache to `validation/cache/`, gitignored), plus synthetic generators. Kaggle files need login — use OpenML mirrors where available.

**Classification**
- Telco Customer Churn (OpenML ID 42178 "telco-customer-churn" — verify), AI4I 2020 (OpenML "ai4i2020" / UCI 601), Adult (OpenML 1590, subsample 20k), credit-g (31), bank-marketing (1461, imbalanced), breast cancer (sklearn), wine / iris (multi-class small), covertype subsample (multi-class, larger), phoneme (1489), a synthetic `make_classification` with weights=[0.95,0.05], noise features, duplicates and missing injected.

**Regression**
- House Prices (Ames: OpenML 42165 "house_prices" or sklearn `fetch_openml("house_prices")`), Bike Sharing (OpenML 42712 "Bike_Sharing_Demand"), California housing (sklearn), diabetes (sklearn, tiny), abalone (OpenML 183, integer-ish target), Boston-free alternatives like "cpu_act" (197), "kin8nm" (189), Friedman1 synthetic with noise features, concrete strength (4353).

**Clustering** (use a label column as hidden truth, k = number of classes)
- Mall Customers (no true labels on Kaggle — simulate via the well-known 5 groups on income/spending: build a synthetic analogue with `make_blobs` in 2D + noise columns + Gender), iris, wine (needs scaling — a key test), seeds (OpenML 1499), breast cancer (k=2), digits (k=10, harder, 64 features), penguins (OpenML 42585, has categoricals + missing), `make_blobs` with anisotropic/unequal variance + irrelevant uniform features + categorical noise, `make_moons` (non-convex; check we don't break).

### 9.2 Perturbation harness ("stress versions")

For each base dataset, generate variants with: 10–30% MCAR missing cells; added constant column; duplicated column; random ID column (int and string); 5–20 Gaussian/uniform noise features; columns rescaled by ×1000; numeric column stored as string with blanks; datetime column; high-cardinality category (500 levels); unseen test categories; extreme imbalance (subsample minority to 2%). Also edge cases: n = 60, p = 1; 150k × 50 (timing); 2 000 × 500 (wide).

### 9.3 Protocol

- Hold-out split 75/25 (stratified for clf) with fixed seeds (≥ 2 seeds for small sets); clustering: run on all rows, score against hidden label.
- Scores: clf macro-F1, balanced acc, accuracy; reg R², RMSE, RMSLE (if y≥0); clustering ARI, NMI. Report normalised score `(s − baseline)/(best − baseline)` per dataset, mean and **minimum** per task family.
- Baselines: clf — one-hot + median-impute + LogisticRegression; reg — same + Ridge; clustering — StandardScaler + KMeans(k).
- Timing: wall-clock per dataset as a subprocess (`py main.py ...`, includes imports), run with `OMP_NUM_THREADS=2` and `--n-jobs 2` to simulate a weak grader; flag anything > 30 s.
- Robustness: count crashes/invalid outputs across all stress variants (target: 0).
- Profile correctness: unit tests with planted constant/duplicate/ID/missing columns and expected counts.
- The harness lives in `validation/` (data and cache gitignored); a single command reproduces the leaderboard table.

---

## 10. Assumptions list (for README)

1. Each dataset is a directory (or file set) with train CSV, test CSV (clustering may have one CSV), and a metadata JSON stating task type, target column, feature types and (clustering) k. All contract parsing is isolated in `src/io_contract.py`.
2. Outputs: `predictions.csv` (one prediction per test row, original order, optional id column) and `profile.json`.
3. Classification metric is imbalance-aware (macro-F1 or balanced accuracy); we optimise macro-F1 while staying strong on accuracy.
4. Regression metric is R²/RMSE-like; log1p target chosen by CV for skewed non-negative targets (also helps if RMSLE is used).
5. Clustering is scored externally (ARI/NMI) against hidden labels with k given; we always return exactly k labels 0..k-1.
6. Profile ground truth is computed on the raw training table with pandas semantics; declared feature types from metadata take precedence over inference.
7. The 60 s limit includes Python start-up and imports; the grader CPU may have as few as 2 cores; we target ≤ 40 s internally and cap threads.
8. No internet at assessment time; only packages in `requirements.txt` are available; core path depends only on numpy, pandas, scipy, scikit-learn.
9. Columns present in both train and test are legitimate features (no manual leakage removal); ID-like columns are dropped from features.
10. Labels in the output keep the original dtype/values of the training target.
11. Hidden datasets resemble the 5 Kaggle examples in size (≈ 200 to ≈ 20k rows, ≤ ~100 columns); we still handle up to ~200k rows via subsampling.
12. Determinism: fixed random seeds.
13. If the official contract, metrics or limits differ, only the adapter and the metric-hint mapping change; this report's inferences are superseded by the kickoff rules.

---

## Appendix — Open questions to re-check when kickoff material is available

- Exact file names, metadata keys, entry-point command, output column names, and whether predictions are read from file or stdout.
- Whether classification output is labels or probabilities; exact metrics per task; how points are normalised (baseline-relative?).
- Profile schema the checker expects (key names, type vocabulary — e.g. "numeric/categorical" only?).
- Size limits (max rows/cols), assessment Python version, OS, core count, whether imports count toward 60 s.
- Whether clustering gives a test set or labels all rows.
