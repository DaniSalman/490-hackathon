# Approach C -- Profile-driven rules (expert system) + sparse k-means

**Philosophy:** the data profile decides the pipeline deterministically, the way an experienced data
scientist would, instead of searching a portfolio. One cheap "linearity probe" is the only empirical
test. Every decision is logged in `profile.json -> modelling.plan.rules_fired`.

## Rules (supervised)
- Quick HistGB (<= 5k rows) written before the probe and the CV; its rate caps HistGB iterations; every model
  fit inside the fold loop is budget-checked.
- Encoding: low-cardinality categoricals native / one-hot; > 30 levels -> cross-fitted `TargetEncoder` with an
  explicit target type (ordinal fallback if too wide or unavailable); datetimes -> calendar + cyclic features.
- Regime (first matching rule): n < 400 -> 0.5 linear + 0.5 ExtraTrees; >= 8 classes and < 150 rows/class ->
  0.6 ExtraTrees + 0.4 linear; wide (> 150 encoded features, n < 10 p) -> 0.5 linear + 0.5 HistGB (regression:
  only if the probe's linear model beats the tree by 0.01 R2); probe linear-ish -> 0.5 linear + 0.5 HistGB;
  otherwise size-tiered HistGB.
- Imbalance ratio > 1.5 -> capped balanced weights; minority >= 20 -> OOF macro-F1 decision tuning.
- Regression: `log1p` when y >= 0 and skew > 0.75, **verified by the probe** (reverted to raw when log1p loses
  > 0.005 R2 on the probe folds); linear members clipped in log space.
- Folds: 5 (3 when n > 20k **or n x p x classes > 5e6**, for time headroom on wide multiclass tables).
- Refit on 100 % of the rows when the projected cost fits.

## Clustering
1. group-aware sparse k-means weights; a feature group is dropped only if its weight is < 5 % of the largest
  **and** a permutation test confirms it is irrelevant (the partition built from the other features explains
  no more of its variance than 200 random label permutations, alpha 0.01). The soft-weighted space (confirmed-irrelevant groups weight 0, others their sparse weights) is
   used only when such features exist; otherwise the unweighted space.
2. Rule-gated candidates: KMeans always; Ward when n <= 6000; GMM-diag on the continuous columns when there are
   >= 2; GMM-full on PCA-whitened data when there are >= 30 rows per group.
3. The partition most candidates agree with (medoid by mean pairwise ARI; ties -> KMeans). No BIC acceptance.

## Known weaknesses
- Rules are fixed thresholds.
- Without the old BIC acceptance, round-1 Mall (0.96 -> 0.67) and iris (0.90 -> 0.74) lose their GMM answer.

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
Validator scoring model, 2 threads, sequential, after warm-up. Round-1 suite **77.23**, round-2 suite
**73.60**, combined 36 datasets **150.83** / 200 (quality 121.71: clf 35.73 /
reg 48.16 / clu 37.81; profile 20.00; speed 9.12). Median / max wall time
3.7 / 27.0 s. All 36 runs valid, no watchdog exit. Both suites have now been used for
development, so these numbers are optimistic; round 3 uses new data.

| suite | dataset | metric | before | after | wall s before -> after | model |
|---|---|---|---|---|---|---|
| r1 | c_adult | macro-F1 | 0.8150 | 0.8150 | 4.7 -> 5.0 | 0.5*linear+0.5*hgb |
| r1 | c_ai4i | macro-F1 | 0.8517 | 0.8517 | 3.1 -> 3.0 | 1.0*hgb |
| r1 | c_bank | macro-F1 | 0.7674 | 0.7674 | 6.3 -> 6.6 | 0.5*linear+0.5*hgb |
| r1 | c_covertype | macro-F1 | 0.5721 | 0.5721 | 15.3 -> 17.6 | 1.0*hgb |
| r1 | c_creditg | macro-F1 | 0.6686 | 0.6686 | 5.6 -> 5.3 | 0.5*linear+0.5*hgb |
| r1 | c_telco | macro-F1 | 0.7344 | 0.7344 | 3.7 -> 3.8 | 0.5*linear+0.5*hgb |
| r1 | k_drybean | ARI | 0.6253 | 0.6600 | 3.8 -> 3.0 | kmeans (unweighted) |
| r1 | k_iris | ARI | 0.9039 | 0.7445 | 1.8 -> 1.8 | gmm_diag (sparse-weighted) |
| r1 | k_mall | ARI | 0.9561 | 0.6651 | 1.8 -> 1.7 | ward (sparse-weighted) |
| r1 | k_penguins | ARI | 0.7685 | 0.9536 | 1.8 -> 1.8 | gmm_full (sparse-weighted) |
| r1 | k_seeds | ARI | 0.7733 | 0.7733 | 1.8 -> 1.7 | kmeans (unweighted) |
| r1 | k_wine | ARI | 0.9150 | 0.8820 | 1.8 -> 1.8 | gmm_diag (sparse-weighted) |
| r1 | r_abalone | R2 | 0.5540 | 0.5540 | 2.7 -> 2.9 | 1.0*hgb[log1p] |
| r1 | r_bike | R2 | 0.9598 | 0.9598 | 7.1 -> 8.5 | 1.0*hgb[log1p] |
| r1 | r_california | R2 | 0.8407 | 0.8407 | 5.4 -> 6.7 | 1.0*hgb[log1p] |
| r1 | r_concrete | R2 | 0.8818 | 0.8818 | 2.8 -> 2.8 | 1.0*hgb |
| r1 | r_diamonds | R2 | 0.9800 | 0.9800 | 4.9 -> 5.6 | 1.0*hgb[log1p] |
| r1 | r_house | R2 | 0.8963 | 0.8963 | 4.6 -> 5.5 | 1.0*hgb[log1p] |
| r2 | c_churn | macro-F1 | 0.9035 | 0.9035 | 4.3 -> 4.5 | 1.0*hgb |
| r2 | c_creditapp | macro-F1 | 0.9064 | 0.9064 | 2.8 -> 2.9 | 1.0*hgb |
| r2 | c_jannis | macro-F1 | 0.5525 | 0.5424 | 30.2 -> 27.0 | 1.0*hgb |
| r2 | c_kick | macro-F1 | 0.6697 | 0.6697 | 10.6 -> 11.3 | 0.5*linear+0.5*hgb |
| r2 | c_magic | macro-F1 | 0.8641 | 0.8641 | 7.8 -> 8.5 | 1.0*hgb |
| r2 | c_pageblocks | macro-F1 | 0.8626 | 0.8626 | 5.9 -> 6.4 | 1.0*hgb |
| r2 | k_ecoli | ARI | 0.4948 | 0.7659 | 1.8 -> 1.9 | gmm_full (unweighted) |
| r2 | k_pendigits | ARI | 0.5631 | 0.5582 | 2.8 -> 3.7 | kmeans (unweighted) |
| r2 | k_segment | ARI | 0.4784 | 0.4656 | 2.2 -> 2.4 | kmeans (unweighted) |
| r2 | k_synth | ARI | 0.6989 | 0.9855 | 2.0 -> 2.2 | gmm_full (sparse-weighted) |
| r2 | k_vehicle | ARI | 0.0763 | 0.0827 | 1.8 -> 2.0 | ward (sparse-weighted) |
| r2 | k_wholesale | ARI | 0.5873 | 0.4696 | 1.6 -> 1.7 | kmeans (sparse-weighted) |
| r2 | r_brazil | R2 | 0.9749 | 0.9749 | 3.6 -> 4.5 | 1.0*hgb[log1p] |
| r2 | r_kc_house | R2 | 0.8568 | 0.8568 | 6.1 -> 7.3 | 1.0*hgb[log1p] |
| r2 | r_moneyball | R2 | 0.9492 | 0.9492 | 2.9 -> 2.9 | 0.5*linear+0.5*hgb |
| r2 | r_seoul | R2 | 0.9320 | 0.9320 | 6.4 -> 8.0 | 1.0*hgb[log1p] |
| r2 | r_superconduct | R2 | 0.8992 | 0.9114 | 15.0 -> 14.5 | 1.0*hgb |
| r2 | r_wine | R2 | 0.4628 | 0.4628 | 3.8 -> 3.2 | 1.0*hgb |

Dev smoke sets (docs/03), all valid, no tracebacks:

| dataset | wall s | valid | metric |
|---|---|---|---|
| c_bool_target_degenerate | 3.5 | True | macro-F1 0.840 |
| c_breast_cancer | 8.0 | True | macro-F1 0.948 |
| c_edge | 3.9 | True | macro-F1 0.753 |
| c_large | 26.0 | True | macro-F1 0.952 |
| c_many_classes | 5.2 | True | macro-F1 0.521 |
| c_multi_nonlinear | 13.1 | True | macro-F1 0.965 |
| c_synth_imb | 5.2 | True | macro-F1 0.866 |
| c_text_nantarget | 2.1 | True | macro-F1 1.000 |
| c_tiny | 3.6 | True | macro-F1 0.493 |
| k_all_categorical | 1.7 | True | ARI 0.711 |
| k_aniso_scaled | 1.9 | True | ARI 0.996 |
| k_blobs_noise | 1.8 | True | ARI 1.000 |
| k_large | 4.0 | True | ARI 0.998 |
| k_mixed_cat | 1.7 | True | ARI 0.696 |
| k_no_k | 1.7 | True | ARI 1.000 |
| k_tiny | 1.5 | True | ARI 0.731 |
| r_counts_datetime | 3.3 | True | R2 0.966 |
| r_diabetes | 2.9 | True | R2 0.352 |
| r_friedman | 2.6 | True | R2 0.821 |
| r_prices_strings | 2.9 | True | R2 0.904 |
| r_tiny | 2.8 | True | R2 0.915 |
| r_wide | 17.2 | True | R2 0.942 |

## Assumptions
Same as `docs/01_research_report.md` section 10: folder with `train.csv`/`test.csv` (or `data.csv` for
clustering) + `metadata.json`; imbalance-aware classification metric (we optimise macro-F1), R2/RMSE-like
regression metric (log1p preferred on ties, in case it is RMSLE), ARI/NMI clustering metric with k given;
the 60 s limit includes interpreter start-up; only the pinned packages are available; ID-like columns
carry no signal.
