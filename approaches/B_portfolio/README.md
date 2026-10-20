# Approach B -- Portfolio + greedy ensemble selection (deterministic plan)

**Philosophy:** let validation decide. Evaluate a diverse portfolio on the same folds, then build a
Caruana-style ensemble from the out-of-fold predictions.

## Pipeline per task
- A quick HistGB (<= 5k rows) is written first (anytime answer + cost probe).
- **Deterministic portfolio** decided by size/shape only (no wall-clock luck; the clock is only a safety
  abort at the 40 s budget), in this order:
  1. `hgb_a` HistGB lr 0.1, 31 leaves (+ a `log1p` variant for skewed non-negative regression targets)
  2. L2 logistic regression / RidgeCV on scaled numeric + missing indicators + one-hot (width-capped)
  3. ExtraTrees (200 trees; 100 trees on <= 8000-row bootstrap samples when n > 10k)
  4. `hgb_b` conservative HistGB (lr 0.06, 15 leaves, l2 3, feature subsampling)
  5. RandomForest (same size rule as ExtraTrees)
  6. kNN (distance-weighted; only n <= 10k)
  7. `hgb_c` deep HistGB (63 leaves; only 2000 <= n <= 10k)

  **Stopping rule:** after the first 3 candidates, the search stops as soon as the last candidate improved
  the ensemble's OOF score (macro-F1 / R2) by < 0.002.
- **Ensemble:** greedy forward selection with replacement (Caruana et al. 2004, 30 steps) on OOF predictions;
  then OOF class-weight tuning for macro-F1. **Linear members** are clipped to the training target range
  (log space when log1p) and, for test rows outside the training range of any numeric feature, the linear
  members' share is removed and the other members renormalised (no extrapolation of outliers).
- **Clustering:** group-aware sparse k-means first acts as a relevance filter: clearly irrelevant
  features (weight < 5 % of the largest; informative categoricals protected) are removed from EVERY view and
  a sparse-weighted view is added; then views {standardised, log+standardised, robust, PCA-whitened,
  sparse-weighted} x {KMeans, GMM, Ward (n <= 8000), spectral (n <= 2500)}; the **consensus partition**
  (medoid by mean pairwise ARI) is returned; near-ties broken by silhouette.

## Known weaknesses
- Still the slowest approach (but now <= ~18 s on the suite at 2 threads, was 34 s).
- Consensus clustering follows the majority: when most algorithms agree on a KMeans-like partition but
  a full-covariance GMM is right (iris), B returns the majority answer.
- Ensemble selection on small data can overfit the OOF predictions slightly.

## How to run

```
py -m pip install -r requirements.txt          # pinned versions (tested: Python 3.14.7)
py -m compileall -q .                          # warm-up: pre-compile .pyc (see "Cold start" below)
py main.py --data <dataset_dir> --out <out_dir> [--time-budget 40] [--n-jobs N]
py main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>
```

Dataset folder format and outputs: see `docs/02_contract_and_dataset_format.md` in the repo root.
Outputs: `<out_dir>/predictions.csv` (`row_index[,<id_column>],prediction`) and `<out_dir>/profile.json`.

**Cold start / warm-up.** The very first import of scikit-learn after installation can take tens of
seconds (we measured 59 s once on Windows: `.pyc` compilation + antivirus scanning of the DLLs), which
would count against the 60 s limit. Run `py -m compileall -q .` in this folder and one throw-away run
(e.g. on any small dataset) after `pip install`, before the timed assessment. Later runs start in ~1.5 s.

**Verified library versions.** Pinned stack (`requirements.txt`): numpy 2.5.3, pandas 3.0.6,
scipy 1.18.1, scikit-learn 1.9.1 (fresh venv `pip install -r requirements.txt` + run, OK). Older stack
also verified on the dev + suite smoke sets: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2
(the oldest versions with Python 3.14 wheels). Version-dependent APIs are guarded (`TargetEncoder` >= 1.3,
HistGB `max_features` >= 1.4, `pd.to_datetime(format="mixed")` >= 2.0); scikit-learn >= 1.4 and
pandas >= 2.0 are expected, older versions are untested.

## Files (self-contained: nothing is imported from outside this folder)

| file | role |
|---|---|
| `main.py` | entry point; starts the budget clock before imports; writes a constant fallback at once; watchdog exits at budget+12 s with the best predictions already on disk |
| `io_contract.py` | the ONLY contract-specific module (dataset/metadata loading with key aliases, output writing, profile key mapping) |
| `data_profile.py` | structured data profile on the raw training table (pandas semantics) + `blank_string_count` / `suspected_missing` fields |
| `cleaning.py` | type inference, numeric-as-string coercion (NA-like tokens only there; literal categories such as `none` survive), datetime expansion, constant / duplicate / ID dropping, text tokens, encoders with a width cap |
| `utils.py` | budget, thread cap, metrics, folds, class-weight tuning, subsampling, fit-cost timer, k-guarantee helpers |
| `feature_weighting.py` | group-aware sparse k-means (one-hot blocks judged like numerics) |
| `model.py` | **the approach-specific modelling logic** |

## Shared guarantees
- Threads capped (`HACK_N_JOBS` / `OMP_NUM_THREADS`, default min(4, cores)); lazy imports; fixed seeds.
- Anytime behaviour: constant fallback (< 1 s) -> quick model -> final model, each written atomically.
- Cost guards: encoded width <= max(5 p, p + 200); per-fit cost estimated from a timed quick fit; budget
  checked before every model fit inside the fold loops.
- Any exception inside modelling keeps the last valid predictions; the profile is always written.
- Clustering always returns exactly k labels `0..k-1` (k from metadata; silhouette choice if absent).
- Classification predictions use the original label values; regression predictions are finite and clipped
  to a widened training range (>= 0 for non-negative targets).

## Results after fix round 1 (docs/05_fix_round1_changelog.md)
Round-1 suite (Validator's 18 datasets, Validator's scoring model, 2 threads, sequential, after warm-up):
**76.63 / 100** (quality 62.27, profile 10.00, speed 4.36);
median / max wall time 4.9 / 17.9 s. All 18 runs valid, no watchdog exit.

| dataset | metric | round 1 | now | wall s round 1 -> now | model |
|---|---|---|---|---|---|
| c_adult | macro-F1 | 0.8115 | 0.8143 | 23.1 -> 7.3 | CaruanaEnsemble(hgb_a:0.3, extratrees:0.4, hgb_b:0.3) |
| c_ai4i | macro-F1 | 0.8492 | 0.8420 | 10.6 -> 4.8 | CaruanaEnsemble(hgb_a:0.4211, logreg:0.1053, extratrees:0.47 |
| c_bank | macro-F1 | 0.7739 | 0.7750 | 27.4 -> 12.0 | CaruanaEnsemble(hgb_a:0.1923, logreg:0.0385, extratrees:0.23 |
| c_covertype | macro-F1 | 0.6159 | 0.6059 | 32.6 -> 17.9 | CaruanaEnsemble(hgb_a:0.2174, logreg:0.087, extratrees:0.565 |
| c_creditg | macro-F1 | 0.7037 | 0.7074 | 7.1 -> 4.9 | CaruanaEnsemble(hgb_a:0.087, logreg:0.087, extratrees:0.5217 |
| c_telco | macro-F1 | 0.7279 | 0.7242 | 10.0 -> 5.2 | CaruanaEnsemble(logreg:1.0) |
| k_drybean | ARI | 0.6610 | 0.6600 | 3.4 -> 4.6 | consensus_medoid:kmeans|logstd |
| k_iris | ARI | 0.6306 | 0.5801 | 1.8 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| k_mall | ARI | 0.6175 | 0.6651 | 1.7 -> 1.9 | consensus_medoid:ward|sparse_w (relevance-filtered) |
| k_penguins | ARI | 0.6534 | 0.9505 | 1.8 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| k_seeds | ARI | 0.7722 | 0.7722 | 1.7 -> 1.8 | consensus_medoid:kmeans|robust |
| k_wine | ARI | 0.8975 | 0.8805 | 1.6 -> 1.9 | consensus_medoid:kmeans|std (relevance-filtered) |
| r_abalone | R2 | 0.5751 | 0.5762 | 8.7 -> 4.3 | CaruanaEnsemble(hgb_a:0.2857, ridge[log1p]:0.25, extratrees[ |
| r_bike | R2 | 0.9620 | 0.9625 | 33.9 -> 13.1 | CaruanaEnsemble(hgb_a:0.5263, hgb_a[log1p]:0.3158, hgb_b:0.1 |
| r_california | R2 | 0.8433 | 0.8421 | 33.2 -> 11.4 | CaruanaEnsemble(hgb_a:0.3333, hgb_a[log1p]:0.2083, extratree |
| r_concrete | R2 | 0.8884 | 0.8884 | 5.3 -> 4.2 | CaruanaEnsemble(hgb_a:0.5714, extratrees:0.1905, hgb_b:0.238 |
| r_diamonds | R2 | 0.9811 | 0.9808 | 32.1 -> 9.3 | CaruanaEnsemble(hgb_a:0.1579, hgb_a[log1p]:0.2105, ridge[log |
| r_house | R2 | 0.8211 | 0.8978 | 16.0 -> 10.3 | CaruanaEnsemble(hgb_a:0.0667, hgb_a[log1p]:0.1667, ridge[log |

Dev smoke sets (docs/03; synthetic + 2 built-ins, edge cases), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 3.7 | True | macro-F1 0.866 |
| c_breast_cancer | 3.7 | True | macro-F1 0.977 |
| c_edge | 5.5 | True | macro-F1 0.801 |
| c_large | 21.8 | True | macro-F1 0.949 |
| c_many_classes | 16.1 | True | macro-F1 0.588 |
| c_multi_nonlinear | 12.8 | True | macro-F1 0.906 |
| c_synth_imb | 9.5 | True | macro-F1 0.849 |
| c_text_nantarget | 2.9 | True | macro-F1 1.000 |
| c_tiny | 3.7 | True | macro-F1 0.547 |
| k_all_categorical | 1.8 | True | ARI 0.711 |
| k_aniso_scaled | 3.1 | True | ARI 0.976 |
| k_blobs_noise | 1.7 | True | ARI 1.000 |
| k_large | 5.2 | True | ARI 0.996 |
| k_mixed_cat | 1.9 | True | ARI 0.685 |
| k_no_k | 1.8 | True | ARI 1.000 |
| k_tiny | 1.5 | True | ARI 0.731 |
| r_counts_datetime | 5.7 | True | R2 0.963 |
| r_diabetes | 3.1 | True | R2 0.363 |
| r_friedman | 5.0 | True | R2 0.832 |
| r_prices_strings | 3.5 | True | R2 0.910 |
| r_tiny | 2.5 | True | R2 0.980 |
| r_wide | 24.0 | True | R2 0.948 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.
