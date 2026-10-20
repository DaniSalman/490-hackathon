# Approach A -- Lean & robust single strong model

**Philosophy:** one proven, well-configured model family per task, minimal search, very fast and hard to
break. Every extra step must be cheap and must not be able to hurt much.

## Pipeline per task
- **Classification:** a quick HistGB (100 iterations, <= 5k rows) is written first; its fit time caps the
  iterations of the main model (cost guard). Main model: `HistGradientBoostingClassifier` on raw numeric
  (+NaN) + native categoricals; n < 1000 -> lr 0.05, 250 iters, 15 leaves; otherwise lr 0.08, early stopping
  (patience 60, <= 800 iters), 31 leaves. Class weights when imbalance ratio > 3: capped balanced (binary),
  sqrt-balanced (multiclass). 5-fold CV (3 when n > 30k); fold models averaged on the test set.
  >= 5 classes: ExtraTrees(100) blend; n < 2000 or < 100 rows/class: logistic-regression blend. Blend weights
  (0 / 0.25 / 0.5 / 0.75) and per-class decision weights are chosen on out-of-fold predictions by macro-F1.
- **Regression:** quick HistGB first, then `HistGradientBoostingRegressor`; raw vs `log1p` (tried when
  y >= 0 and skew > 0.75) by OOF R2, **log1p kept when within 0.005 R2** (RMSLE-safe); n < 500: RidgeCV blend
  (ridge clipped to the training target range); predictions clipped.
- **Clustering:** standardised features (log1p for skewed, one-hot/sqrt(2)) -> KMeans (written first) ->
  group-aware sparse k-means used as a **relevance filter**: features with weight < 5 % of the largest are
  dropped (categoricals that explain at least the median numeric share of the base partition are
  protected) and KMeans is refit -> a GMM (diag / full) replaces KMeans only if 3 initialisations agree
  (ARI >= 0.8), it agrees with KMeans (ARI >= 0.6) and BIC improves by >= 30 %.

## Known weaknesses
- Clustering: when no feature is clearly irrelevant and clusters are elongated (drybean, seeds) it stays
  at KMeans quality; Mall (derived labels on Income/Spending) stays at 0.58 because Age is not irrelevant.
- The GMM guard thresholds were checked on the round-1 suite and dev data only.
- Log1p tie-break costs a little R2 when the metric is R2 (abalone -0.009, bike -0.002).

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
**78.13 / 100** (quality 63.42, profile 10.00, speed 4.70);
median / max wall time 2.9 / 15.0 s. All 18 runs valid, no watchdog exit.

| dataset | metric | round 1 | now | wall s round 1 -> now | model |
|---|---|---|---|---|---|
| c_adult | macro-F1 | 0.8166 | 0.8199 | 3.3 -> 3.8 | HistGradientBoostingClassifier |
| c_ai4i | macro-F1 | 0.8643 | 0.8779 | 2.4 -> 2.8 | HistGradientBoostingClassifier |
| c_bank | macro-F1 | 0.7775 | 0.7808 | 4.8 -> 5.5 | HistGradientBoostingClassifier |
| c_covertype | macro-F1 | 0.5656 | 0.6203 | 8.5 -> 15.0 | blend(0.25*hgb + 0.75*et) |
| c_creditg | macro-F1 | 0.6702 | 0.6879 | 2.6 -> 2.6 | blend(0.25*hgb + 0.75*lr) |
| c_telco | macro-F1 | 0.7283 | 0.7274 | 2.5 -> 2.7 | HistGradientBoostingClassifier |
| k_drybean | ARI | 0.6600 | 0.6600 | 1.8 -> 5.6 | KMeans |
| k_iris | ARI | 0.6306 | 0.9039 | 1.7 -> 1.7 | relevance-filtered KMeans -> GMM(full) |
| k_mall | ARI | 0.5845 | 0.5845 | 1.6 -> 1.8 | relevance-filtered KMeans |
| k_penguins | ARI | 0.7508 | 0.8939 | 1.6 -> 1.7 | relevance-filtered KMeans |
| k_seeds | ARI | 0.7733 | 0.7733 | 1.7 -> 1.8 | KMeans |
| k_wine | ARI | 0.8975 | 0.8805 | 1.7 -> 1.8 | relevance-filtered KMeans |
| r_abalone | R2 | 0.5680 | 0.5587 | 2.4 -> 2.9 | HistGradientBoostingRegressor[log1p] |
| r_bike | R2 | 0.9623 | 0.9599 | 8.1 -> 9.5 | HistGradientBoostingRegressor[log1p] |
| r_california | R2 | 0.8422 | 0.8420 | 3.8 -> 8.6 | HistGradientBoostingRegressor[log1p] |
| r_concrete | R2 | 0.8897 | 0.8897 | 2.3 -> 2.3 | HistGradientBoostingRegressor |
| r_diamonds | R2 | 0.9800 | 0.9807 | 4.6 -> 5.5 | HistGradientBoostingRegressor[log1p] |
| r_house | R2 | 0.8910 | 0.8954 | 4.3 -> 5.4 | HistGradientBoostingRegressor[log1p] |

Dev smoke sets (docs/03; synthetic + 2 built-ins, edge cases), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 2.2 | True | macro-F1 0.827 |
| c_breast_cancer | 2.3 | True | macro-F1 0.977 |
| c_edge | 2.6 | True | macro-F1 0.804 |
| c_large | 25.2 | True | macro-F1 0.948 |
| c_many_classes | 11.3 | True | macro-F1 0.576 |
| c_multi_nonlinear | 10.0 | True | macro-F1 0.900 |
| c_synth_imb | 8.6 | True | macro-F1 0.865 |
| c_text_nantarget | 4.0 | True | macro-F1 1.000 |
| c_tiny | 2.5 | True | macro-F1 0.538 |
| k_all_categorical | 1.9 | True | ARI 0.711 |
| k_aniso_scaled | 2.0 | True | ARI 0.996 |
| k_blobs_noise | 2.0 | True | ARI 1.000 |
| k_large | 4.8 | True | ARI 0.996 |
| k_mixed_cat | 2.2 | True | ARI 0.695 |
| k_no_k | 1.8 | True | ARI 1.000 |
| k_tiny | 1.7 | True | ARI 0.731 |
| r_counts_datetime | 3.5 | True | R2 0.958 |
| r_diabetes | 2.2 | True | R2 0.361 |
| r_friedman | 3.4 | True | R2 0.822 |
| r_prices_strings | 3.0 | True | R2 0.888 |
| r_tiny | 1.7 | True | R2 0.973 |
| r_wide | 12.1 | True | R2 0.808 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.
