# Approach C -- Profile-driven rules (expert system) + sparse k-means

**Philosophy:** the data profile decides the pipeline deterministically, the way an experienced data
scientist would, instead of searching a portfolio. One cheap "linearity probe" is the only empirical
test. Every decision is logged in `profile.json -> modelling.plan.rules_fired`.

## Rules (supervised)
- A quick HistGB (100 iterations, <= 5k rows) is written **before** the probe and the CV; its fit time caps
  HistGB iterations, and every model fit inside the fold loop is budget-checked.
- Encoding: low-cardinality categoricals native in HistGB / one-hot for linear models; categoricals with
  > 30 levels -> cross-fitted `TargetEncoder` with an **explicit target type** (continuous / binary /
  multiclass), fitted inside each fold only; ordinal fallback if the encoded width would exceed
  max(5 p, p + 200) or TargetEncoder is unavailable; datetimes -> calendar + cyclic sin/cos features.
- Regime (first matching rule wins):

  | condition | models |
  |---|---|
  | n < 400 | 0.5 linear + 0.5 ExtraTrees |
  | >= 8 classes and < 150 rows per class | 0.6 ExtraTrees + 0.4 linear |
  | > 150 encoded features and n < 10 p (regression: only if the probe's linear model beats the tree by >= 0.01 R2) | 0.5 linear + 0.5 HistGB |
  | probe (3-fold, <= 3000 rows): classification linear >= tree - 0.005; regression linear >= tree + 0.01 | 0.5 linear + 0.5 HistGB |
  | otherwise | HistGB with size-tiered params (n < 1k / 2k / 20k / larger) |

- Imbalance ratio > 1.5 -> capped balanced weights; minority >= 20 -> OOF macro-F1 class-weight tuning.
- Regression: `log1p` target when y >= 0 and skew > 0.75 (rule); linear members clipped to the training
  target range in log space; final clipping.
- Final fit: refit on 100 % of the training rows (HistGB iterations = 1.1 x median early-stopped fold
  iterations) when the projected refit cost fits, else fold averaging.
- No learned meta-selector (see docs/03): rules are hand-written to avoid leaking validation data.

## Clustering
1. Standardise (log1p for skewed) + one-hot/sqrt(2); KMeans on all features is emitted first.
2. **Group-aware sparse k-means** (Witten & Tibshirani 2010): each original feature is one group and its
   relevance is the share of its variance explained by the partition, so one-hot blocks are weighted like
   numerics (penguins `island` is no longer crushed). The L1 bound is chosen by a permutation gap
   statistic. The soft-weighted space is used only when some weight < 5 % of the largest (irrelevant
   features present); otherwise the unweighted space.
3. GMM diag / full (full only if p <= 4 or n >= 10 k p(p+1)/2), initialised at the KMeans centres: replaces
   KMeans only if ARI(GMM, KMeans) >= 0.6 **and** BIC improves by >= 15 % over a spherical mixture, with no
   cluster smaller than max(2, 1 % of n).

## Known weaknesses
- Rules are fixed thresholds; a dataset on the wrong side of a threshold gets a sub-optimal pipeline.
- drybean: the accepted GMM-full partition (agreement 0.70, BIC +313 %) is worse than KMeans (0.63 vs 0.66);
  round 1's 0.72 came from a lucky initialisation.

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
**78.56 / 100** (quality 63.77, profile 10.00, speed 4.79);
median / max wall time 3.8 / 15.3 s. All 18 runs valid, no watchdog exit.

| dataset | metric | round 1 | now | wall s round 1 -> now | model |
|---|---|---|---|---|---|
| c_adult | macro-F1 | 0.8150 | 0.8150 | 4.4 -> 4.7 | 0.5*linear+0.5*hgb |
| c_ai4i | macro-F1 | 0.8517 | 0.8517 | 2.9 -> 3.1 | 1.0*hgb |
| c_bank | macro-F1 | 0.7674 | 0.7674 | 6.0 -> 6.3 | 0.5*linear+0.5*hgb |
| c_covertype | macro-F1 | 0.5721 | 0.5721 | 14.6 -> 15.3 | 1.0*hgb |
| c_creditg | macro-F1 | 0.6951 | 0.6686 | 4.8 -> 5.6 | 0.5*linear+0.5*hgb |
| c_telco | macro-F1 | 0.7344 | 0.7344 | 3.5 -> 3.7 | 0.5*linear+0.5*hgb |
| k_drybean | ARI | 0.7242 | 0.6253 | 3.1 -> 3.8 | kmeans->GMM(full) |
| k_iris | ARI | 0.9039 | 0.9039 | 1.8 -> 1.8 | sparse_kmeans->GMM(full) |
| k_mall | ARI | 0.9696 | 0.9561 | 1.8 -> 1.8 | sparse_kmeans->GMM(diag) |
| k_penguins | ARI | 0.5805 | 0.7685 | 1.7 -> 1.8 | sparse_kmeans |
| k_seeds | ARI | 0.7224 | 0.7733 | 1.6 -> 1.8 | kmeans |
| k_wine | ARI | 0.9150 | 0.9150 | 1.8 -> 1.8 | sparse_kmeans->GMM(diag) |
| r_abalone | R2 | 0.5540 | 0.5540 | 2.5 -> 2.7 | 1.0*hgb[log1p] |
| r_bike | R2 | 0.9598 | 0.9598 | 6.9 -> 7.1 | 1.0*hgb[log1p] |
| r_california | R2 | -0.0445 | 0.8407 | 52.3 -> 5.4 | 1.0*hgb[log1p] |
| r_concrete | R2 | 0.8818 | 0.8818 | 3.4 -> 2.8 | 1.0*hgb |
| r_diamonds | R2 | 0.9800 | 0.9800 | 4.6 -> 4.9 | 1.0*hgb[log1p] |
| r_house | R2 | 0.7507 | 0.8963 | 4.3 -> 4.6 | 1.0*hgb[log1p] |

Dev smoke sets (docs/03; synthetic + 2 built-ins, edge cases), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 4.4 | True | macro-F1 0.840 |
| c_breast_cancer | 7.8 | True | macro-F1 0.948 |
| c_edge | 3.8 | True | macro-F1 0.753 |
| c_large | 26.5 | True | macro-F1 0.953 |
| c_many_classes | 5.6 | True | macro-F1 0.521 |
| c_multi_nonlinear | 11.9 | True | macro-F1 0.965 |
| c_synth_imb | 5.3 | True | macro-F1 0.866 |
| c_text_nantarget | 1.9 | True | macro-F1 1.000 |
| c_tiny | 3.4 | True | macro-F1 0.493 |
| k_all_categorical | 1.6 | True | ARI 0.711 |
| k_aniso_scaled | 1.7 | True | ARI 0.996 |
| k_blobs_noise | 1.6 | True | ARI 1.000 |
| k_large | 4.2 | True | ARI 0.998 |
| k_mixed_cat | 1.7 | True | ARI 0.695 |
| k_no_k | 1.6 | True | ARI 1.000 |
| k_tiny | 1.5 | True | ARI 0.731 |
| r_counts_datetime | 2.9 | True | R2 0.966 |
| r_diabetes | 2.9 | True | R2 0.352 |
| r_friedman | 2.6 | True | R2 0.821 |
| r_prices_strings | 2.7 | True | R2 0.904 |
| r_tiny | 2.7 | True | R2 0.915 |
| r_wide | 14.9 | True | R2 0.942 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.
