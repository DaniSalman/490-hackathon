# Approach A -- Lean & robust single strong model

**Philosophy:** one proven, well-configured model family per task, minimal search, very fast and hard to
break. Every extra step must be cheap and must not be able to hurt much.

## Pipeline per task
- **Classification:** quick HistGB (<= 5k rows) written first; its fit time caps later iterations (cost
  guard). Main model `HistGradientBoostingClassifier` (native categoricals / NaN; n < 1000: lr 0.05, 250 iters,
  15 leaves; otherwise lr 0.08, early stopping with patience 60, 31 leaves). Imbalance ratio > 3:
  **sqrt-balanced** class weights (binary and multiclass) + OOF-tuned per-class decision weights for macro-F1.
  5-fold CV (3 when n > 30k), fold models averaged on test. >= 5 classes: ExtraTrees(100) blend; n < 2000 or
  < 100 rows/class: logistic-regression blend. Blend weights from {0, 0.25, 0.5, 0.75} by OOF macro-F1.
- **Regression:** quick HistGB first; HistGB raw vs log1p (y >= 0, skew > 0.75) by OOF R2, log1p kept when within
  0.005 (RMSLE-safe); n < 500 ridge blend (clipped); predictions clipped.
- **Clustering:** standardised features -> KMeans written first -> group-aware sparse k-means weights; a feature group is dropped only if its weight is < 5 % of the largest
  **and** a permutation test confirms it is irrelevant (the partition built from the other features explains
  no more of its variance than 200 random label permutations, alpha 0.01) -> four cheap candidates on the
  kept features: KMeans, Ward, GMM-diag on the continuous columns only, GMM-full on PCA-whitened data -> the
  partition the candidates agree with most (medoid by mean pairwise ARI; ties -> KMeans). No BIC acceptance.

## Known weaknesses
- The agreement choice cannot pick a minority model family even when it is right (round-1 iris: the
  full-covariance answer 0.90 is now 0.65).
- No high-cardinality target encoding (kick: 0.659 vs C 0.670).

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
Validator scoring model, 2 threads, sequential, after warm-up. Round-1 suite **77.34**, round-2 suite
**73.43**, combined 36 datasets **150.76** / 200 (quality 121.73: clf 35.96 /
reg 48.20 / clu 37.56; profile 20.00; speed 9.03). Median / max wall time
3.4 / 29.6 s. All 36 runs valid, no watchdog exit. Both suites have now been used for
development, so these numbers are optimistic; round 3 uses new data.

| suite | dataset | metric | before | after | wall s before -> after | model |
|---|---|---|---|---|---|---|
| r1 | c_adult | macro-F1 | 0.8199 | 0.8204 | 3.8 -> 4.3 | HistGradientBoostingClassifier |
| r1 | c_ai4i | macro-F1 | 0.8779 | 0.8720 | 2.8 -> 2.9 | HistGradientBoostingClassifier |
| r1 | c_bank | macro-F1 | 0.7808 | 0.7761 | 5.5 -> 6.0 | HistGradientBoostingClassifier |
| r1 | c_covertype | macro-F1 | 0.6203 | 0.6203 | 15.0 -> 17.8 | blend(0.25*hgb + 0.75*et) |
| r1 | c_creditg | macro-F1 | 0.6879 | 0.6879 | 2.6 -> 2.7 | blend(0.25*hgb + 0.75*lr) |
| r1 | c_telco | macro-F1 | 0.7274 | 0.7274 | 2.7 -> 2.7 | HistGradientBoostingClassifier |
| r1 | k_drybean | ARI | 0.6600 | 0.6600 | 5.6 -> 3.7 | kmeans |
| r1 | k_iris | ARI | 0.9039 | 0.6525 | 1.7 -> 1.8 | relevance-filtered gmm_full |
| r1 | k_mall | ARI | 0.5845 | 0.6807 | 1.8 -> 1.8 | relevance-filtered gmm_full |
| r1 | k_penguins | ARI | 0.8939 | 0.9505 | 1.7 -> 1.7 | relevance-filtered ward |
| r1 | k_seeds | ARI | 0.7733 | 0.7733 | 1.8 -> 1.7 | kmeans |
| r1 | k_wine | ARI | 0.8805 | 0.8471 | 1.8 -> 1.8 | relevance-filtered kmeans |
| r1 | r_abalone | R2 | 0.5587 | 0.5587 | 2.9 -> 3.0 | HistGradientBoostingRegressor[log1p] |
| r1 | r_bike | R2 | 0.9599 | 0.9599 | 9.5 -> 11.2 | HistGradientBoostingRegressor[log1p] |
| r1 | r_california | R2 | 0.8420 | 0.8420 | 8.6 -> 10.1 | HistGradientBoostingRegressor[log1p] |
| r1 | r_concrete | R2 | 0.8897 | 0.8897 | 2.3 -> 2.3 | HistGradientBoostingRegressor |
| r1 | r_diamonds | R2 | 0.9807 | 0.9807 | 5.5 -> 6.1 | HistGradientBoostingRegressor[log1p] |
| r1 | r_house | R2 | 0.8954 | 0.8954 | 5.4 -> 5.9 | HistGradientBoostingRegressor[log1p] |
| r2 | c_churn | macro-F1 | 0.8907 | 0.9003 | 3.9 -> 4.1 | HistGradientBoostingClassifier |
| r2 | c_creditapp | macro-F1 | 0.8662 | 0.8662 | 2.4 -> 2.4 | blend(0.50*hgb + 0.50*lr) |
| r2 | c_jannis | macro-F1 | 0.5503 | 0.5490 | 29.9 -> 29.6 | HistGradientBoostingClassifier |
| r2 | c_kick | macro-F1 | 0.6550 | 0.6593 | 8.3 -> 8.5 | HistGradientBoostingClassifier |
| r2 | c_magic | macro-F1 | 0.8660 | 0.8660 | 3.9 -> 4.0 | HistGradientBoostingClassifier |
| r2 | c_pageblocks | macro-F1 | 0.8515 | 0.8515 | 5.8 -> 6.2 | blend(0.25*hgb + 0.75*et) |
| r2 | k_ecoli | ARI | 0.4948 | 0.7659 | 1.8 -> 1.8 | gmm_full |
| r2 | k_pendigits | ARI | 0.5582 | 0.5582 | 2.9 -> 3.8 | kmeans |
| r2 | k_segment | ARI | 0.3787 | 0.4656 | 2.1 -> 2.4 | kmeans |
| r2 | k_synth | ARI | 0.6989 | 0.9776 | 2.1 -> 2.1 | relevance-filtered ward |
| r2 | k_vehicle | ARI | 0.0745 | 0.0694 | 1.9 -> 2.0 | relevance-filtered kmeans |
| r2 | k_wholesale | ARI | 0.4820 | 0.5532 | 1.7 -> 1.9 | relevance-filtered gmm_diag |
| r2 | r_brazil | R2 | 0.9760 | 0.9760 | 5.1 -> 5.7 | HistGradientBoostingRegressor[log1p] |
| r2 | r_kc_house | R2 | 0.8525 | 0.8525 | 7.5 -> 8.7 | HistGradientBoostingRegressor[log1p] |
| r2 | r_moneyball | R2 | 0.9419 | 0.9419 | 2.4 -> 2.4 | HistGradientBoostingRegressor |
| r2 | r_seoul | R2 | 0.9356 | 0.9356 | 9.3 -> 10.8 | HistGradientBoostingRegressor[log1p] |
| r2 | r_superconduct | R2 | 0.9108 | 0.9108 | 16.8 -> 19.0 | HistGradientBoostingRegressor |
| r2 | r_wine | R2 | 0.4647 | 0.4647 | 3.6 -> 3.1 | HistGradientBoostingRegressor |

Dev smoke sets (docs/03), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 2.0 | True | macro-F1 0.827 |
| c_breast_cancer | 2.1 | True | macro-F1 0.977 |
| c_edge | 2.4 | True | macro-F1 0.804 |
| c_large | 24.5 | True | macro-F1 0.948 |
| c_many_classes | 10.5 | True | macro-F1 0.576 |
| c_multi_nonlinear | 8.6 | True | macro-F1 0.900 |
| c_synth_imb | 4.6 | True | macro-F1 0.860 |
| c_text_nantarget | 1.7 | True | macro-F1 1.000 |
| c_tiny | 2.1 | True | macro-F1 0.538 |
| k_all_categorical | 1.5 | True | ARI 0.711 |
| k_aniso_scaled | 1.8 | True | ARI 0.992 |
| k_blobs_noise | 1.7 | True | ARI 1.000 |
| k_large | 4.1 | True | ARI 0.996 |
| k_mixed_cat | 1.7 | True | ARI 0.696 |
| k_no_k | 1.6 | True | ARI 1.000 |
| k_tiny | 1.4 | True | ARI 0.731 |
| r_counts_datetime | 3.3 | True | R2 0.958 |
| r_diabetes | 2.1 | True | R2 0.361 |
| r_friedman | 2.4 | True | R2 0.822 |
| r_prices_strings | 2.9 | True | R2 0.888 |
| r_tiny | 1.7 | True | R2 0.973 |
| r_wide | 12.0 | True | R2 0.808 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.


## Fix round 3 (docs/09_fix_round3_changelog.md)
Changes: fast HGB (lr 0.1, 100 iterations) + 3 folds for >= 10 classes; chunked, deadline-bounded HGB fits; measured-time fold planner (x2 slow-CPU safety); bounded quick model; k labels guaranteed on the predicted rows.

All 54 datasets of suites r1-r3 (2 threads, same-session run): total **221.80** (quality 178.82,
speed 13.00); median / max wall time 3.9 / 21.9 s; 0 watchdog exits.
A simulated 2.5x slower CPU (`HACK_SLOWDOWN=2.5`) gives 54/54 valid runs and no watchdog exits. The 1-thread maximum is at most
28 s (c_letter: see docs/09 section 2). Self-check of the worst-case path: `py tools/eval_suite.py run --tag slow --slowdown 2.5`.
