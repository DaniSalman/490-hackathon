# Approach C -- Profile-driven rules (expert system) + sparse k-means

**Philosophy:** the data profile decides the pipeline deterministically, the way an experienced data
scientist would, instead of searching a portfolio. One cheap "linearity probe" is the only empirical
test. Every decision is logged in `profile.json -> modelling.plan.rules_fired`.

## Rules (supervised)
- Encoding: low-cardinality categoricals native in HistGB / one-hot for linear models; categoricals with
  > 30 levels -> cross-fitted `TargetEncoder` (fitted inside each fold only); datetimes -> calendar +
  cyclic sin/cos features.
- Regime (first matching rule wins):

  | condition | models |
  |---|---|
  | n < 400 | 0.5 linear + 0.5 ExtraTrees |
  | >= 8 classes and < 150 rows per class | 0.6 ExtraTrees + 0.4 linear |
  | > 150 encoded features and n < 10 p | 0.5 linear + 0.5 HistGB |
  | probe (3-fold, <= 3000 rows): linear score >= quick-HistGB score - 0.005 | 0.5 linear + 0.5 HistGB |
  | otherwise | HistGB with size-tiered params (n < 1k / 2k / 20k / larger) |

- Imbalance ratio > 1.5 -> capped balanced weights; minority >= 20 -> OOF macro-F1 class-weight tuning.
- Regression: `log1p` target when y >= 0 and skew > 0.75 (rule, no CV); clipping.
- Final fit: **refit on 100 % of the training rows** (HistGB iterations = 1.1 x median early-stopped
  fold iterations); falls back to fold averaging when the budget is short. (A averages fold models; B
  ensembles them.)
- No learned meta-selector (deviation, see docs/03): the rules are hand-written from general practice and
  checked on synthetic dev data only, so no validation dataset leaks into them.

## Clustering
1. Standardise (log1p for skewed) + one-hot/sqrt(2); KMeans on all features is emitted first.
2. **Sparse k-means** (Witten & Tibshirani 2010): alternate weighted KMeans and lasso-type feature weights
   from per-feature between-cluster sums of squares; the L1 bound s is chosen on <= 3000 rows by a
   permutation gap statistic (5 values of s, 3 permutations, smallest s within 1 sd of the best gap).
   Irrelevant features get weight 0.
3. On the weighted features: GMM spherical / diag / full (full only if <= 8 dims), initialised at the
   KMeans centres, compared by BIC; a non-spherical winner replaces KMeans only if no cluster is smaller
   than max(2, 1 % of n).

## Known weaknesses
- Rules are fixed thresholds; a dataset on the wrong side of a threshold gets a sub-optimal pipeline
  (dev: breast-cancer data macro-F1 0.948 vs 0.977 for A/B).
- Sparse k-means can discard weak but genuine features (dev: mixed categorical clusters ARI 0.58 vs 0.69 for A).
- Refit-on-full adds one extra fit (dev: largest run ~31 s at 2 threads).

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
`py tools/run_smoke.py --approach approaches/C_rules --datasets dev_data --threads 2` on the 22 dev datasets
listed in `docs/03_dev_datasets_used.md` (synthetic + 2 sklearn built-ins). Edge cases covered: string /
boolean / partially-missing targets, a singleton class, unseen test categories, a column missing from the
test file and an extra one added, all-NaN / constant / duplicate / ID columns, numeric-as-string, datetime,
free text, high cardinality, tiny n (30-80 rows), 60k rows, 400 columns, single-CSV and train/test
clustering, metadata without `task` / `feature_types` / `n_clusters`. Wall time includes interpreter
start-up and imports (24-core dev machine, threads capped at 2).
**All 22 runs valid, no tracebacks.** A single-thread rerun of the largest sets stayed under 33 s.

| dataset | task | wall s | valid | metric | model |
|---|---|---|---|---|---|
| c_bool_target_degenerate | class | 3.8 | True | macro-F1 0.840 / acc 0.840 | 0.5*linear+0.5*extratrees |
| c_breast_cancer | class | 7.8 | True | macro-F1 0.948 / acc 0.951 | 0.5*linear+0.5*hgb |
| c_edge | class | 3.8 | True | macro-F1 0.753 / acc 0.767 | 0.5*linear+0.5*extratrees |
| c_large | class | 30.8 | True | macro-F1 0.953 / acc 0.953 | 1.0*hgb |
| c_many_classes | class | 5.2 | True | macro-F1 0.521 / acc 0.533 | 0.6*extratrees+0.4*linear |
| c_multi_nonlinear | class | 12.3 | True | macro-F1 0.965 / acc 0.971 | 1.0*hgb |
| c_synth_imb | class | 4.6 | True | macro-F1 0.866 / acc 0.961 | 1.0*hgb |
| c_text_nantarget | class | 1.9 | True | macro-F1 1.000 / acc 1.000 | 0.5*linear+0.5*hgb |
| c_tiny | class | 3.4 | True | macro-F1 0.493 / acc 0.500 | 0.5*linear+0.5*extratrees |
| k_all_categorical | clust | 1.6 | True | ARI 0.711 / NMI 0.730 | sparse_kmeans->GMM(diag) |
| k_aniso_scaled | clust | 1.7 | True | ARI 0.996 / NMI 0.992 | sparse_kmeans->GMM(full) |
| k_blobs_noise | clust | 1.6 | True | ARI 1.000 / NMI 1.000 | sparse_kmeans->GMM(diag) |
| k_large | clust | 3.8 | True | ARI 0.998 / NMI 0.997 | sparse_kmeans->GMM(diag) |
| k_mixed_cat | clust | 1.7 | True | ARI 0.580 / NMI 0.575 | sparse_kmeans->GMM(full) |
| k_no_k | clust | 1.6 | True | ARI 1.000 / NMI 1.000 | sparse_kmeans |
| k_tiny | clust | 1.5 | True | ARI 0.731 / NMI 0.798 | sparse_kmeans |
| r_counts_datetime | regre | 2.8 | True | R2 0.966 / RMSLE 0.260 | 1.0*hgb[log1p] |
| r_diabetes | regre | 2.9 | True | R2 0.352 / RMSLE 0.404 | 0.5*linear+0.5*extratrees |
| r_friedman | regre | 2.6 | True | R2 0.821 / RMSLE 0.160 | 1.0*hgb |
| r_prices_strings | regre | 2.6 | True | R2 0.904 / RMSLE 0.128 | 0.5*linear+0.5*hgb[log1p] |
| r_tiny | regre | 2.8 | True | R2 0.915 | 0.5*linear+0.5*extratrees |
| r_wide | regre | 8.7 | True | R2 0.943 | 0.5*linear+0.5*hgb |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric, ARI/NMI clustering metric with k given; the 60 s limit includes interpreter start-up;
only the pinned packages are available; ID-like columns carry no signal. Needs scikit-learn >= 1.4
(HistGB `categorical_features` mask + `class_weight`) and pandas >= 2.0 (`format="mixed"`).
