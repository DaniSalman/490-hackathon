# Approach A -- Lean & robust single strong model

**Philosophy:** one proven, well-configured model family per task, minimal search, very fast and hard to
break. Every extra step must be cheap and must not be able to hurt much.

## Pipeline per task
- **Classification:** `HistGradientBoostingClassifier` on raw numeric (+NaN) + native categoricals
  (top-250 ordinal codes). Params: n < 1000 -> lr 0.05, 250 iters, 15 leaves; otherwise lr 0.08, early
  stopping (<= 800 iters), 31 leaves, l2 = 1. Capped "balanced" class weights when imbalance ratio > 3.
  5-fold CV (3 when n > 30k) whose fold models are **averaged on the test set** (bagging, no refit).
  Out-of-fold probabilities tune per-class decision weights for macro-F1 (kept only if OOF gain > 0.002).
  Tiny data (n < 500 or < 100 rows per class): blend with L2 logistic regression, weight picked by OOF log-loss.
- **Regression:** `HistGradientBoostingRegressor`, same features/params; raw vs `log1p` target compared
  by OOF RMSE in original units (log only tried when y >= 0 and skew > 1); tiny data (n < 500) blended
  with RidgeCV by OOF RMSE; clipping.
- **Clustering:** median impute -> `log1p` on skewed non-negative columns -> standardise; categoricals
  one-hot scaled by 1/sqrt(2) (two different categories are at distance 1); PCA to 90 % variance when
  > 30 features; `KMeans(k)` with 30/10/4 restarts depending on n.

## Known weaknesses
- Clustering has no defence against many irrelevant features or non-spherical clusters
  (dev: blobs + noise ARI 0.43, anisotropic blobs ARI 0.49, where B/C reach 0.99-1.0).
- No model selection: when a different family is clearly better (e.g. many classes with few rows per
  class: macro-F1 0.53 vs 0.61 for B) A cannot switch beyond the tiny-data linear blend.

## How to run

```
py -m pip install -r requirements.txt          # Python 3.14 tested; pinned versions
py main.py --data <dataset_dir> --out <out_dir> [--time-budget 40] [--n-jobs N]
py main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>
```

Dataset folder format and outputs: see `docs/02_contract_and_dataset_format.md` in the repo root.
Outputs: `<out_dir>/predictions.csv` (`row_index[,<id_column>],prediction`) and `<out_dir>/profile.json`.

## Files (self-contained: nothing is imported from outside this folder)

| file | role |
|---|---|
| `main.py` | entry point; starts the budget clock before imports; writes a constant fallback at once; watchdog exits at budget+12 s with the best predictions already on disk |
| `io_contract.py` | the ONLY contract-specific module (dataset/metadata loading with key aliases, output writing, profile key mapping) |
| `data_profile.py` | structured data profile on the raw training table (schema in docs/01 section 4.1) |
| `cleaning.py` | type inference (numeric / categorical / boolean / datetime / text / id / constant), numeric-as-string coercion, datetime expansion, constant / duplicate / ID dropping, text token features, encoders |
| `utils.py` | budget, thread cap, metrics, folds, class-weight tuning, k-guarantee helpers |
| `model.py` | **the approach-specific modelling logic** |

## Shared guarantees
- Threads capped (`HACK_N_JOBS` / `OMP_NUM_THREADS`, default min(4, cores)); lazy imports; fixed seeds.
- Anytime behaviour: constant fallback -> first model -> final model, each written atomically.
- Any exception inside modelling keeps the last valid predictions; the profile is always written.
- Clustering always returns exactly k labels `0..k-1` (k from metadata; silhouette choice if absent).
- Classification predictions use the original label values; regression predictions are finite and clipped
  to a widened training range (>= 0 for non-negative targets).
- Wide data (> 100 tree features): HistGB `max_bins=63` (measured ~5x faster split finding at p=400).

## Smoke test (dev datasets only -- NOT the validation score)
`py tools/run_smoke.py --approach approaches/A_lean --datasets dev_data --threads 2` on the 22 dev datasets
listed in `docs/03_dev_datasets_used.md` (synthetic + 2 sklearn built-ins). Edge cases covered: string /
boolean / partially-missing targets, a singleton class, unseen test categories, a column missing from the
test file and an extra one added, all-NaN / constant / duplicate / ID columns, numeric-as-string, datetime,
free text, high cardinality, tiny n (30-80 rows), 60k rows, 400 columns, single-CSV and train/test
clustering, metadata without `task` / `feature_types` / `n_clusters`. Wall time includes interpreter
start-up and imports (24-core dev machine, threads capped at 2).
**All 22 runs valid, no tracebacks.** A single-thread rerun of the largest sets stayed under 33 s.

| dataset | task | wall s | valid | metric | model |
|---|---|---|---|---|---|
| c_bool_target_degenerate | class | 2.8 | True | macro-F1 0.853 / acc 0.853 | blend(0.00*HGB + 1.00*LogReg) |
| c_breast_cancer | class | 2.2 | True | macro-F1 0.977 / acc 0.979 | blend(0.25*HGB + 0.75*LogReg) |
| c_edge | class | 2.4 | True | macro-F1 0.788 / acc 0.800 | blend(0.50*HGB + 0.50*LogReg) |
| c_large | class | 24.6 | True | macro-F1 0.949 / acc 0.949 | HistGradientBoostingClassifier |
| c_many_classes | class | 8.6 | True | macro-F1 0.527 / acc 0.528 | blend(0.75*HGB + 0.25*LogReg) |
| c_multi_nonlinear | class | 5.9 | True | macro-F1 0.886 / acc 0.927 | HistGradientBoostingClassifier |
| c_synth_imb | class | 3.9 | True | macro-F1 0.843 / acc 0.953 | HistGradientBoostingClassifier |
| c_text_nantarget | class | 1.8 | True | macro-F1 1.000 / acc 1.000 | HistGradientBoostingClassifier |
| c_tiny | class | 2.3 | True | macro-F1 0.490 / acc 0.500 | blend(0.00*HGB + 1.00*LogReg) |
| k_all_categorical | clust | 1.6 | True | ARI 0.711 / NMI 0.730 | KMeans |
| k_aniso_scaled | clust | 1.6 | True | ARI 0.490 / NMI 0.579 | KMeans |
| k_blobs_noise | clust | 1.6 | True | ARI 0.427 / NMI 0.518 | KMeans |
| k_large | clust | 2.4 | True | ARI 0.996 / NMI 0.994 | KMeans |
| k_mixed_cat | clust | 1.6 | True | ARI 0.695 / NMI 0.655 | KMeans |
| k_no_k | clust | 1.6 | True | ARI 1.000 / NMI 1.000 | KMeans |
| k_tiny | clust | 1.6 | True | ARI 0.731 / NMI 0.798 | KMeans |
| r_counts_datetime | regre | 3.2 | True | R2 0.962 / RMSLE 0.271 | HistGradientBoostingRegressor |
| r_diabetes | regre | 2.1 | True | R2 0.361 / RMSLE 0.405 | blend(0.00*HGB + 1.00*Ridge) |
| r_friedman | regre | 2.2 | True | R2 0.821 / RMSLE 0.158 | HistGradientBoostingRegressor |
| r_prices_strings | regre | 2.5 | True | R2 0.889 / RMSLE 0.137 | HistGradientBoostingRegressor |
| r_tiny | regre | 1.8 | True | R2 0.973 | blend(0.00*HGB + 1.00*Ridge) |
| r_wide | regre | 7.7 | True | R2 0.808 | HistGradientBoostingRegressor |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric, ARI/NMI clustering metric with k given; the 60 s limit includes interpreter start-up;
only the pinned packages are available; ID-like columns carry no signal. Needs scikit-learn >= 1.4
(HistGB `categorical_features` mask + `class_weight`) and pandas >= 2.0 (`format="mixed"`).
