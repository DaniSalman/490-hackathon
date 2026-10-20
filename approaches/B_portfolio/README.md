# Approach B -- Portfolio + greedy ensemble selection (deterministic plan)

**Philosophy:** let validation decide. Evaluate a diverse portfolio on the same folds, then build a
Caruana-style ensemble from the out-of-fold predictions.

## Pipeline per task
- A quick HistGB (<= 5k rows) is written first (anytime answer + cost probe).
- **Deterministic portfolio** decided by size/shape only:
  1. `hgb_a` HistGB lr 0.1, 31 leaves (+ a `log1p` variant for skewed non-negative regression targets)
  2. L2 logistic regression / RidgeCV (width-capped one-hot)
  3. ExtraTrees (200 trees; 100 trees on <= 8000-row bootstraps when n > 10k)
  4. `hgb_b` conservative HistGB
  5. RandomForest, 6. kNN (n <= 10k), 7. `hgb_c` (2000 <= n <= 10k)

  **Size class:** tables with n x features > 8e5 use 3 folds and only candidates 1-4. **Stopping rule:** after
  3 candidates, stop when the last one improved the ensemble OOF score by < 0.002.
  **Affordability (safety only):** if the measured quick-fit rate says even the anchor CV will not fit, B runs
  the full single `hgb_a` with an iteration cap (not the 5k-row quick model).
- **Ensemble:** Caruana greedy selection (30 steps) on OOF predictions; OOF class-weight tuning for macro-F1.
  **RMSLE safety:** for log1p-eligible targets, the ensemble restricted to log-target members is used when its
  OOF R2 is within 0.01 of the unrestricted ensemble. **Out-of-range rows:** linear members are clipped to the
  training target range (log space when log1p) and, on test rows outside the training feature range, capped to
  within a factor 2 of the non-linear members (extrapolation kept only when corroborated in magnitude).
- **Clustering:** group-aware sparse k-means weights; a feature group is dropped only if its weight is < 5 % of the largest
  **and** a permutation test confirms it is irrelevant (the partition built from the other features explains
  no more of its variance than 200 random label permutations, alpha 0.01); the confirmed-irrelevant features are removed from every view and a
  sparse-weighted view is added; views x {KMeans, GMM, Ward, spectral}; consensus medoid by mean pairwise ARI.

## Known weaknesses
- Slowest approach (max ~24 s at 2 threads); a slower CPU triggers the single-hgb_a fallback on wide tables.
- RMSLE safety costs a little R2 on some skewed targets (bike, house, Seoul: -0.005 to -0.018).

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
| `cleaning.py` | type inference (undeclared integer columns named zip / state / *code / *ID -> categorical), numeric-as-string coercion (NA-like tokens only there; literal categories such as `none` survive), datetime expansion, constant / duplicate / ID dropping, text tokens, encoders with a width cap |
| `utils.py` | budget, thread cap, metrics, folds, class-weight tuning, subsampling, fit-cost timer, k-guarantee helpers |
| `feature_weighting.py` | group-aware sparse k-means, permutation-confirmed relevance filter, cheap candidate partitions + agreement medoid |
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

## Results after fix round 2 (docs/07_fix_round2_changelog.md)
Validator scoring model, 2 threads, sequential, after warm-up. Round-1 suite **76.39**, round-2 suite
**73.45**, combined 36 datasets **149.84** / 200 (quality 121.59: clf 35.72 /
reg 48.38 / clu 37.50; profile 20.00; speed 8.24). Median / max wall time
5.8 / 24.6 s. All 36 runs valid, no watchdog exit. Both suites have now been used for
development, so these numbers are optimistic; round 3 uses new data.

| suite | dataset | metric | before | after | wall s before -> after | model |
|---|---|---|---|---|---|---|
| r1 | c_adult | macro-F1 | 0.8143 | 0.8143 | 7.3 -> 7.7 | CaruanaEnsemble(hgb_a:0.3, extratrees:0.4, hgb_b:0.3) |
| r1 | c_ai4i | macro-F1 | 0.8420 | 0.8420 | 4.8 -> 5.1 | CaruanaEnsemble(hgb_a:0.4211, logreg:0.1053, extratrees |
| r1 | c_bank | macro-F1 | 0.7750 | 0.7750 | 12.0 -> 13.1 | CaruanaEnsemble(hgb_a:0.1923, logreg:0.0385, extratrees |
| r1 | c_covertype | macro-F1 | 0.6059 | 0.6059 | 17.9 -> 20.3 | CaruanaEnsemble(hgb_a:0.2174, logreg:0.087, extratrees: |
| r1 | c_creditg | macro-F1 | 0.7074 | 0.7074 | 4.9 -> 4.7 | CaruanaEnsemble(hgb_a:0.087, logreg:0.087, extratrees:0 |
| r1 | c_telco | macro-F1 | 0.7242 | 0.7242 | 5.2 -> 5.4 | CaruanaEnsemble(logreg:1.0) |
| r1 | k_drybean | ARI | 0.6600 | 0.6600 | 4.6 -> 4.8 | consensus_medoid:kmeans|logstd |
| r1 | k_iris | ARI | 0.5801 | 0.5801 | 1.9 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| r1 | k_mall | ARI | 0.6651 | 0.6651 | 1.9 -> 1.8 | consensus_medoid:ward|sparse_w (relevance-filtered) |
| r1 | k_penguins | ARI | 0.9505 | 0.9505 | 1.9 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| r1 | k_seeds | ARI | 0.7722 | 0.7722 | 1.8 -> 1.8 | consensus_medoid:kmeans|robust |
| r1 | k_wine | ARI | 0.8805 | 0.8804 | 1.9 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| r1 | r_abalone | R2 | 0.5762 | 0.5739 | 4.3 -> 4.5 | CaruanaEnsemble(hgb_a[log1p]:0.2069, ridge[log1p]:0.241 |
| r1 | r_bike | R2 | 0.9625 | 0.9570 | 13.1 -> 15.2 | CaruanaEnsemble(hgb_a[log1p]:0.5517, hgb_b[log1p]:0.448 |
| r1 | r_california | R2 | 0.8421 | 0.8365 | 11.4 -> 12.9 | CaruanaEnsemble(hgb_a[log1p]:0.52, extratrees[log1p]:0. |
| r1 | r_concrete | R2 | 0.8884 | 0.8884 | 4.2 -> 3.7 | CaruanaEnsemble(hgb_a:0.5714, extratrees:0.1905, hgb_b: |
| r1 | r_diamonds | R2 | 0.9808 | 0.9807 | 9.3 -> 10.8 | CaruanaEnsemble(hgb_a[log1p]:0.3, ridge[log1p]:0.0667,  |
| r1 | r_house | R2 | 0.8978 | 0.8798 | 10.3 -> 11.0 | CaruanaEnsemble(hgb_a[log1p]:0.2105, ridge[log1p]:0.473 |
| r2 | c_churn | macro-F1 | 0.8880 | 0.8880 | 9.0 -> 9.6 | CaruanaEnsemble(hgb_a:0.3043, logreg:0.0435, extratrees |
| r2 | c_creditapp | macro-F1 | 0.8715 | 0.8715 | 4.1 -> 4.1 | CaruanaEnsemble(logreg:0.3333, extratrees:0.6667) |
| r2 | c_jannis | macro-F1 | 0.5010 | 0.5537 | 7.1 -> 24.6 | CaruanaEnsemble(hgb_a:1.0) |
| r2 | c_kick | macro-F1 | 0.6690 | 0.6690 | 14.0 -> 14.6 | CaruanaEnsemble(hgb_a:0.2857, logreg:0.3571, extratrees |
| r2 | c_magic | macro-F1 | 0.8620 | 0.8620 | 6.8 -> 6.7 | CaruanaEnsemble(hgb_a:0.5, extratrees:0.3571, hgb_b:0.1 |
| r2 | c_pageblocks | macro-F1 | 0.8557 | 0.8557 | 11.0 -> 11.7 | CaruanaEnsemble(hgb_a:0.069, logreg:0.1724, extratrees: |
| r2 | k_ecoli | ARI | 0.7241 | 0.7241 | 2.5 -> 2.0 | consensus_medoid:gmm_full|pca_white |
| r2 | k_pendigits | ARI | 0.6164 | 0.6164 | 6.6 -> 8.2 | consensus_medoid:ward|std |
| r2 | k_segment | ARI | 0.4392 | 0.4656 | 2.8 -> 2.8 | consensus_medoid:kmeans|logstd |
| r2 | k_synth | ARI | 0.9730 | 0.9730 | 3.2 -> 3.2 | consensus_medoid:ward|robust (relevance-filtered) |
| r2 | k_vehicle | ARI | 0.0746 | 0.0722 | 2.1 -> 2.4 | consensus_medoid:kmeans|robust (relevance-filtered) |
| r2 | k_wholesale | ARI | 0.5808 | 0.5808 | 1.9 -> 1.9 | consensus_medoid:spectral|logstd (relevance-filtered) |
| r2 | r_brazil | R2 | 0.9861 | 0.9860 | 9.2 -> 10.5 | CaruanaEnsemble(hgb_a[log1p]:0.2, ridge[log1p]:0.8) |
| r2 | r_kc_house | R2 | 0.8492 | 0.8505 | 12.0 -> 13.0 | CaruanaEnsemble(hgb_a[log1p]:0.3, ridge[log1p]:0.0333,  |
| r2 | r_moneyball | R2 | 0.9482 | 0.9506 | 3.9 -> 4.1 | CaruanaEnsemble(hgb_a:0.1, ridge:0.8667, extratrees:0.0 |
| r2 | r_seoul | R2 | 0.9362 | 0.9308 | 13.2 -> 16.0 | CaruanaEnsemble(hgb_a[log1p]:0.7333, extratrees[log1p]: |
| r2 | r_superconduct | R2 | 0.9117 | 0.9086 | 23.5 -> 16.1 | CaruanaEnsemble(hgb_a:0.2692, extratrees:0.5769, hgb_b: |
| r2 | r_wine | R2 | 0.5022 | 0.5022 | 5.8 -> 6.3 | CaruanaEnsemble(extratrees:1.0) |

Dev smoke sets (docs/03), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 3.1 | True | macro-F1 0.866 |
| c_breast_cancer | 3.7 | True | macro-F1 0.977 |
| c_edge | 5.8 | True | macro-F1 0.801 |
| c_large | 26.1 | True | macro-F1 0.949 |
| c_many_classes | 18.5 | True | macro-F1 0.588 |
| c_multi_nonlinear | 14.4 | True | macro-F1 0.906 |
| c_synth_imb | 9.9 | True | macro-F1 0.849 |
| c_text_nantarget | 2.8 | True | macro-F1 1.000 |
| c_tiny | 3.9 | True | macro-F1 0.547 |
| k_all_categorical | 1.7 | True | ARI 0.711 |
| k_aniso_scaled | 3.3 | True | ARI 0.976 |
| k_blobs_noise | 1.8 | True | ARI 1.000 |
| k_large | 5.8 | True | ARI 0.996 |
| k_mixed_cat | 1.9 | True | ARI 0.685 |
| k_no_k | 1.8 | True | ARI 1.000 |
| k_tiny | 1.6 | True | ARI 0.731 |
| r_counts_datetime | 6.0 | True | R2 0.958 |
| r_diabetes | 3.1 | True | R2 0.363 |
| r_friedman | 5.4 | True | R2 0.832 |
| r_prices_strings | 3.7 | True | R2 0.910 |
| r_tiny | 2.6 | True | R2 0.972 |
| r_wide | 27.0 | True | R2 0.952 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.
