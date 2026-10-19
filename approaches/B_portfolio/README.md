# Approach B -- Time-budgeted portfolio + greedy ensemble selection

**Philosophy:** let validation decide. Evaluate a diverse portfolio on the same folds under the time
budget, then build a Caruana-style ensemble from the out-of-fold predictions.

## Pipeline per task
- **Supervised portfolio** (run in this order; a candidate is skipped when its projected cost, estimated
  from the first model's time, exceeds the remaining budget, and aborted mid-CV if it overruns):
  1. `hgb_a` HistGB lr 0.1, 31 leaves (+ a `log1p`-target variant for skewed non-negative regression targets)
  2. L2 logistic regression / RidgeCV on scaled numeric + missing indicators + one-hot
  3. ExtraTrees (300 trees; row-subsampled when n > 30k)
  4. `hgb_b` conservative HistGB (lr 0.04, 15 leaves, l2 3, feature subsampling)
  5. RandomForest
  6. kNN (distance-weighted, only n <= 30k)
  7. `hgb_c` deep HistGB (63 leaves, n >= 2000)

  All use capped balanced class weights when imbalance ratio > 3; in regression the non-anchor models reuse
  the target transform that won for `hgb_a`. 5-fold CV (3 when n > 30k); fold models averaged on test.
- **Ensemble:** greedy forward selection with replacement (Caruana et al. 2004, 30 steps) maximising OOF
  macro-F1 (tiny log-loss tie-break) / minimising OOF RMSE; then OOF class-weight tuning for macro-F1
  (kept if gain > 0.003). The best single model is emitted as soon as it exists (anytime).
- **Clustering:** feature views {standardised, log+standardised, robust-scaled, PCA-whitened} x algorithms
  {KMeans, GMM (full covariance if <= 12 dims, else diag), Ward (n <= 8000), spectral nearest-neighbour
  graph (n <= 2500)}; candidates must produce exactly k clusters. The **consensus partition** -- the
  medoid candidate by mean pairwise ARI to all others -- is returned; near-ties broken by silhouette.

## Known weaknesses
- Slowest approach (dev: up to ~29 s at 2 threads); on a much weaker CPU it degrades gracefully to fewer
  candidates, but speed points will be lower.
- Ensemble selection on small data can overfit the OOF predictions slightly.
- Consensus clustering follows the majority of views: if most views are fooled by irrelevant features
  (dev: blobs + noise ARI 0.43), so is the consensus.

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
`py tools/run_smoke.py --approach approaches/B_portfolio --datasets dev_data --threads 2` on the 22 dev datasets
listed in `docs/03_dev_datasets_used.md` (synthetic + 2 sklearn built-ins). Edge cases covered: string /
boolean / partially-missing targets, a singleton class, unseen test categories, a column missing from the
test file and an extra one added, all-NaN / constant / duplicate / ID columns, numeric-as-string, datetime,
free text, high cardinality, tiny n (30-80 rows), 60k rows, 400 columns, single-CSV and train/test
clustering, metadata without `task` / `feature_types` / `n_clusters`. Wall time includes interpreter
start-up and imports (24-core dev machine, threads capped at 2).
**All 22 runs valid, no tracebacks.** A single-thread rerun of the largest sets stayed under 33 s.

| dataset | task | wall s | valid | metric | model |
|---|---|---|---|---|---|
| c_bool_target_degenerate | class | 4.9 | True | macro-F1 0.866 / acc 0.867 | CaruanaEnsemble(logreg:0.625, hgb_b:0.375) |
| c_breast_cancer | class | 5.3 | True | macro-F1 0.977 / acc 0.979 | CaruanaEnsemble(hgb_a:0.0333, logreg:0.8, extratrees:0.0333, randomfor |
| c_edge | class | 6.5 | True | macro-F1 0.785 / acc 0.800 | CaruanaEnsemble(logreg:0.3333, extratrees:0.0741, hgb_b:0.2593, random |
| c_large | class | 20.1 | True | macro-F1 0.949 / acc 0.949 | CaruanaEnsemble(hgb_a:1.0) |
| c_many_classes | class | 23.1 | True | macro-F1 0.612 / acc 0.624 | CaruanaEnsemble(extratrees:0.5556, randomforest:0.4074, knn:0.037) |
| c_multi_nonlinear | class | 28.4 | True | macro-F1 0.891 / acc 0.933 | CaruanaEnsemble(hgb_a:0.25, randomforest:0.75) |
| c_synth_imb | class | 13.8 | True | macro-F1 0.849 / acc 0.959 | CaruanaEnsemble(hgb_a:0.0769, extratrees:0.3846, hgb_b:0.3462, randomf |
| c_text_nantarget | class | 5.3 | True | macro-F1 1.000 / acc 1.000 | CaruanaEnsemble(hgb_a:1.0) |
| c_tiny | class | 5.5 | True | macro-F1 0.472 / acc 0.500 | CaruanaEnsemble(logreg:0.3636, randomforest:0.3636, knn:0.2727) |
| k_all_categorical | clust | 1.8 | True | ARI 0.711 / NMI 0.730 | consensus_medoid:kmeans|logstd |
| k_aniso_scaled | clust | 2.5 | True | ARI 0.996 / NMI 0.992 | consensus_medoid:gmm_full|robust |
| k_blobs_noise | clust | 2.2 | True | ARI 0.427 / NMI 0.518 | consensus_medoid:kmeans|logstd |
| k_large | clust | 4.1 | True | ARI 0.996 / NMI 0.994 | consensus_medoid:kmeans|logstd |
| k_mixed_cat | clust | 1.8 | True | ARI 0.685 / NMI 0.637 | consensus_medoid:spectral|logstd |
| k_no_k | clust | 1.7 | True | ARI 1.000 / NMI 1.000 | consensus_medoid:kmeans|logstd |
| k_tiny | clust | 1.6 | True | ARI 0.731 / NMI 0.798 | consensus_medoid:kmeans|logstd |
| r_counts_datetime | regre | 12.2 | True | R2 0.964 / RMSLE 0.259 | CaruanaEnsemble(hgb_a:0.2308, hgb_a[log1p]:0.3077, hgb_b:0.0769, hgb_c |
| r_diabetes | regre | 4.6 | True | R2 0.363 / RMSLE 0.402 | CaruanaEnsemble(ridge:0.625, extratrees:0.0417, hgb_b:0.0417, randomfo |
| r_friedman | regre | 6.5 | True | R2 0.832 / RMSLE 0.155 | CaruanaEnsemble(hgb_a:0.1034, extratrees:0.1034, hgb_b:0.7931) |
| r_prices_strings | regre | 5.3 | True | R2 0.910 / RMSLE 0.122 | CaruanaEnsemble(hgb_a:0.0333, ridge[log1p]:0.8333, hgb_b[log1p]:0.1333 |
| r_tiny | regre | 3.7 | True | R2 0.972 | CaruanaEnsemble(ridge:0.9667, hgb_b:0.0333) |
| r_wide | regre | 28.9 | True | R2 0.954 | CaruanaEnsemble(ridge:0.8182, hgb_b:0.1818) |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric, ARI/NMI clustering metric with k given; the 60 s limit includes interpreter start-up;
only the pinned packages are available; ID-like columns carry no signal. Needs scikit-learn >= 1.4
(HistGB `categorical_features` mask + `class_weight`) and pandas >= 2.0 (`format="mixed"`).
