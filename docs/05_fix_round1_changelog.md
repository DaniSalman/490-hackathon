# 05 — Fix round 1 changelog (Validator findings, docs/04)

The three approaches stay independent and self-contained. Shared fixes were made once in A and copied
into each folder, so the five shared modules are byte-identical again: `main.py`, `io_contract.py`,
`cleaning.py`, `data_profile.py`, `utils.py`, plus the new `feature_weighting.py`. Approach-specific
logic stays in each `model.py`.

**How to re-evaluate.** The new tool `tools/eval_suite.py` imports, read-only, the Validator's grader
simulation and scoring functions from `validation/run_validation.py`. It uses the same subprocess,
2-thread, sequential protocol with warm-up and the 60 s kill, but writes to `runs/<tag>/`, which is
gitignored. Nothing under `validation/` was modified. Check: scoring the Validator's own round-1 runs with
this tool reproduces the totals in docs/04 exactly (A 75.81, B 74.04, C 73.23).

```
py tools/eval_suite.py run   --tag r1final
py tools/eval_suite.py score --tag r1final --compare validation/results/results.csv
```

## Totals on the round-1 suite (Validator scoring model; before = docs/04, after = runs/r1final)

| approach | total | quality | clf / reg / clu pools | profile | speed | median / max wall s |
|---|---|---|---|---|---|---|
| A_lean before | 75.81 | 60.97 | 16.44 / 24.24 / 20.29 | 9.96 | 4.89 | 2.4 / 8.5 |
| **A_lean after** | **78.13** | 63.42 | 17.04 / 24.21 / 22.18 | 10.00 | 4.70 | **2.9** / 15.0 |
| B_portfolio before | 74.04 | 60.68 | 16.75 / 23.95 / 19.99 | 9.96 | 3.40 | 9.3 / 33.9 |
| **B_portfolio after** | **76.63** | 62.27 | 16.67 / 24.31 / 21.29 | 10.00 | 4.36 | 4.9 / 17.9 |
| C_rules before | 73.23 | 58.73 | 16.50 / 19.49 / 22.74 | 9.96 | 4.55 | 3.4 / 52.3 (watchdog) |
| **C_rules after** | **78.56** | 63.77 | 16.29 / 24.14 / 23.34 | 10.00 | 4.79 | 3.8 / 15.3 |

All 54 runs are valid: no crash, no timeout, no watchdog exit, no traceback. A cheap real model reaches
disk at most 2.6 s after process start in every run (the constant fallback is still written at < 1 s).
A is still the fastest by median and by maximum wall time.

Caveat: one test split per dataset. Differences of a few thousandths are noise (docs/04 §10). The
clustering thresholds below were checked on this suite and on the dev sets, so the round-2 datasets are
the real test of whether they generalise.

---

## Shared fixes

### S1 — literal category values survive (done)
- `cleaning.py`: the old `SENTINELS` set is removed. `clean_strings()` now turns only empty or
  whitespace-only strings into missing.
- NA-like tokens (`NUMERIC_NA_TOKENS`: `?`, `na`, `none`, `-`, `--`, `.`, `missing`, `unknown`, ...)
  count as missing only in numeric contexts:
  - `infer_column_type()` ignores them when deciding whether a string column is numeric, so `"?"` no longer
    blocks numeric detection;
  - `coerce_numeric()` turns them into NaN.
- Literal values such as `none`, `-` and `unknown` stay categories.
- credit-g: `own_telephone ∈ {none, yes}` is now `categorical` (action `categorical`, no longer
  `dropped_constant`), and `constant_columns` is empty. credit-g profile 0.515 → 0.556 pts (= 1.0) in all
  three approaches. Profile total 9.96 → 10.00 everywhere.

### S2 — per-fit cost guards (done)
- **(a) Width cap.** `cleaning.max_encoded_features(p) = max(5p, p+200)`, and `onehot_encode(..., max_total=)`
  halves the per-column level count until the one-hot block fits. This is used by every linear and forest
  view in A, B and C. C's target encoder checks `len(hi) × n_classes` against the same cap and falls back to
  ordinal codes if it does not fit.
- **(b) Timed quick fit.** Every supervised run first fits a quick HistGB: 100 iterations, or
  max(30, 200/C) for multiclass, on ≤ 5000 rows (`utils.subsample_idx`, stratified). It is written to disk
  at once and timed (`utils.FitTimer`).
  - A and C convert the measured rate into a HistGB iteration cap of 3 × the affordable iterations. This is a
    safety net: on this machine it never binds on the suite, but a 10× cost blow-up would.
  - If the cap drops below 40 iterations, the quick model is kept.
  - B uses the rate to decide whether its portfolio is affordable at all.
- **(c) Per-model checks.** The budget is checked before every model fit inside the fold loops (A, C) or the
  candidate CV loops (B). For later folds the projection uses the previous fold's measured time per model.
  C also projects the refit-on-full cost before doing it.
- Effect: C `r_california` 52.3 s watchdog exit with R² −0.04 → 5.4 s, R² 0.841 (C1 is the root-cause fix;
  S2 would by itself have kept a real model on disk). First real model on disk ≤ 2.6 s in all 54 runs.

### S3 — suspected-missing profile fields (done)
- `data_profile.py`: per column `blank_string_count` (empty / whitespace-only strings), `suspected_missing`
  (blank + NA-like tokens `? na n/a nan null #n/a missing`) and `n_missing_incl_suspected`.
- Top level: `total_blank_string_cells`, `total_suspected_missing_cells` and
  `total_missing_cells_incl_suspected`. The pandas-semantics fields (`n_missing`, `total_missing_cells`) are
  unchanged.
- Telco: `TotalCharges` `n_missing = 0` (pandas), `blank_string_count = 9`; `total_blank_string_cells = 9`.

### S4 — warm-up note, dependency check, older versions (done)
- Each README has a "Cold start / warm-up" section (`py -m compileall -q .` plus one throw-away run).
- A fresh venv with `pip install -r requirements.txt` installs the pinned stack (numpy 2.5.3, pandas
  3.0.6, scipy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.7.0, joblib 1.6.0) and all three approaches run in
  it.
- **Older stack:** numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2. These are the oldest versions
  with Python 3.14 wheels; scikit-learn 1.6 and pandas 2.2 have none, so 1.3–1.6 could not be tested here.
  All three approaches ran on all 18 suite datasets and all 22 dev sets with this stack (results in §S4
  below).
- Guarded APIs: `TargetEncoder` (import guarded in C with ordinal fallback), HistGB `max_features` (B,
  probed with `try`), and `pd.to_datetime(format="mixed")` (already guarded). We state sklearn ≥ 1.4 and
  pandas ≥ 2.0 as expected; older versions are untested.

---

## A_lean

| ID | status | change (model.py unless noted) | target: before → after |
|---|---|---|---|
| A1 | done (Mall not improved) | `_clustering` + new `feature_weighting.py`. Group-aware sparse k-means (each original feature is a group; relevance = share of its variance explained, so one-hot blocks are scored like numerics). It is used as a **relevance filter**: groups with weight < 5 % of the largest are dropped, except categoricals whose explained share in the base partition ≥ the median numeric share. KMeans is then refit on the kept features. | iris 0.631 → **0.904** (noise_1-3 dropped, then A2); penguins 0.751 → **0.894** (`sex` dropped, `island` kept); Mall 0.585 → 0.585 (Gender dropped but Age is not irrelevant, see rejected); wine 0.897 → 0.881 (Ash/Magnesium dropped, small loss) |
| A2 | done | GMM diag / full on the kept features. Accepted only if 3 initialisations agree (mean pairwise ARI ≥ 0.8), it agrees with KMeans (ARI ≥ 0.6), BIC improves ≥ 30 % over a spherical mixture and no cluster is smaller than max(2, 1 % n). Full covariance only if p ≤ 4 or n ≥ 10 k p(p+1)/2. | iris → GMM-full (0.904); drybean 0.660 → 0.660 (GMM rejected — see rejected); seeds unchanged 0.773 |
| A3 | done | Multiclass: sqrt-balanced class weights instead of capped balanced (binary unchanged); early-stopping patience 30 → 60; ≥ 5 classes: ExtraTrees(100) blend; blend weight chosen by OOF macro-F1 of the tuned decision rule. | covertype 0.566 → **0.620** (> default HGB 0.593, > B's 0.616) |
| A4 | done | Logistic-regression blend now for n < 2000 (or < 100 rows/class); weight from {0, .25, .5, .75} by OOF macro-F1. | credit-g 0.670 → **0.688** |
| A5 | done | log1p tried when y ≥ 0 and skew > 0.75 (was 1.0); chosen by OOF R², and **kept when within 0.005 R² of raw**. | bike RMSLE 0.419 → **0.273** (R² 0.962 → 0.960); house 0.891 → 0.895; abalone 0.568 → 0.559 (the price of RMSLE safety) |
| S2 | done | quick model + `_iteration_cap` + per-model fold checks | first real model ≤ 2.6 s |

Speed: still the fastest (median 2.9 s, max 15.0 s on covertype; was 2.4 / 8.5). Extra time comes from
patience 60, the ExtraTrees blend on ≥ 5 classes and the log1p variant on more targets.

## B_portfolio

| ID | status | change | target: before → after |
|---|---|---|---|
| B1 | done | Portfolio trimmed by size: kNN only n ≤ 10k; `hgb_c` only 2000 ≤ n ≤ 10k; forests 100 trees on ≤ 8000-row bootstrap samples when n > 10k. **Stopping rule:** after the first 3 candidates, stop when the last candidate improved the Caruana ensemble's OOF score by < 0.002. Quick model first. | median 9.3 → **4.9 s**, max 33.9 → **17.9 s**; speed 3.40 → 4.36 pts; quality 60.68 → 62.27 |
| B2 | done | Candidate set from data size/shape only (`_portfolio`). The clock is only a safety abort against the 40 s budget (skip / abort a candidate that cannot finish). The old "skip if projected cost > remaining" against a filling budget is now a pure safety check. | covertype with `OMP_NUM_THREADS=1` and `=2`: **byte-identical predictions** (was 0.605 vs 0.616) |
| B3 | done | Linear members (ridge / logreg) are clipped to the fold's training-target range in log space when log1p is used. For test rows outside the training range of any numeric feature (±10 % of the range), the linear members' ensemble share is removed and the rest renormalised. | house 0.821 → **0.898**; s_house probes not re-scored |
| B4 | done (iris not fixed) | Group-aware sparse k-means is a relevance filter applied to **every** consensus view, plus an extra sparse-weighted view (`sparse_w`). | penguins 0.653 → **0.951** (no drift; `sex` removed); Mall 0.618 → 0.665; iris 0.631 → 0.580 (noise removed, but the consensus medoid is a KMeans-like partition — see rejected); wine 0.897 → 0.881 |

Other B changes: the regression ensemble metric is now R² instead of −RMSE (scale-free, needed for the
stopping rule), and `hgb_b` is lr 0.06 / 1000 iterations (was 0.04 / 1500). Covertype 0.616 → 0.606 is the
cost of fewer forest trees on large n (see rejected).

## C_rules

| ID | status | change | target: before → after |
|---|---|---|---|
| C1 | done | `Features`: `TargetEncoder(target_type=)` is explicitly `continuous` (regression), `binary` or `multiclass`. Width guard and ordinal fallback (S2a). Import guard for sklearn < 1.3. | california **−0.045 (52.3 s, watchdog) → 0.841 (5.4 s)** |
| C2 | done | Quick HistGB (100 iterations, ≤ 5k rows) written **before** the probe and CV; iteration cap from its rate; budget check before every model fit in the fold loop; refit only if its projected cost fits. | first real model ≤ 2.6 s everywhere |
| C3 | done | Regression blends linear + HGB only if the probe's linear R² ≥ tree R² **+ 0.01**. This also applies to the "wide" rule, which previously blended without a probe and was the house trigger. Linear predictions are clipped to the training-target range in log space. Classification rules unchanged. | house 0.751 → **0.896** (probe 0.883 vs 0.876, no blend); prices dev set still blends |
| C4a | done | Group-aware sparse k-means (one-hot blocks weighted like numerics). Rule: the soft-weighted space is used only if some weight < 5 % of the largest; otherwise the unweighted space. | penguins 0.581 → **0.769**; seeds 0.722 → **0.773**; iris 0.904 kept; Mall 0.970 → 0.956 kept |
| C4b | done (drybean worse) | GMM diag / full replaces KMeans only if ARI(GMM, KMeans) ≥ 0.6 **and** BIC gain ≥ 15 % over a spherical mixture, with the full-covariance parameter rule and the min-size guard. | seeds GMM-full rejected (0.773); wine diag kept 0.915; Mall diag kept 0.956; **drybean 0.724 → 0.625** (see rejected) |

---

## Per-dataset before → after (primary metric; wall s; total points per dataset)

### A_lean
| dataset | before | after | Δ | wall before | wall after | pts before | pts after |
|---|---|---|---|---|---|---|---|
| c_adult | 0.8166 | 0.8199 | +0.0033 | 3.3 | 3.8 | 4.03 | 4.06 |
| c_ai4i | 0.8643 | 0.8779 | +0.0136 | 2.4 | 2.8 | 4.30 | 4.42 |
| c_bank | 0.7775 | 0.7808 | +0.0032 | 4.8 | 5.5 | 3.58 | 3.59 |
| c_covertype | 0.5656 | 0.6203 | +0.0547 | 8.5 | 15.0 | 3.24 | 3.46 |
| c_creditg | 0.6702 | 0.6879 | +0.0177 | 2.6 | 2.6 | 2.87 | 3.05 |
| c_telco | 0.7283 | 0.7274 | -0.0009 | 2.5 | 2.7 | 3.33 | 3.32 |
| k_drybean | 0.6600 | 0.6600 | +0.0000 | 1.8 | 5.6 | 3.95 | 3.94 |
| k_iris | 0.6306 | 0.9039 | +0.2732 | 1.7 | 1.7 | 3.81 | 5.10 |
| k_mall | 0.5845 | 0.5845 | +0.0000 | 1.6 | 1.8 | 3.59 | 3.59 |
| k_penguins | 0.7508 | 0.8939 | +0.1431 | 1.6 | 1.7 | 4.38 | 5.05 |
| k_seeds | 0.7733 | 0.7733 | +0.0000 | 1.7 | 1.8 | 4.49 | 4.49 |
| k_wine | 0.8975 | 0.8805 | -0.0170 | 1.7 | 1.8 | 5.07 | 4.99 |
| r_abalone | 0.5680 | 0.5587 | -0.0093 | 2.4 | 2.9 | 3.52 | 3.47 |
| r_bike | 0.9623 | 0.9599 | -0.0024 | 8.1 | 9.5 | 5.32 | 5.29 |
| r_california | 0.8422 | 0.8420 | -0.0001 | 3.8 | 8.6 | 4.81 | 4.75 |
| r_concrete | 0.8897 | 0.8897 | +0.0000 | 2.3 | 2.3 | 5.03 | 5.03 |
| r_diamonds | 0.9800 | 0.9807 | +0.0007 | 4.6 | 5.5 | 5.46 | 5.45 |
| r_house | 0.8910 | 0.8954 | +0.0044 | 4.3 | 5.4 | 5.04 | 5.05 |

### B_portfolio
| dataset | before | after | Δ | wall before | wall after | pts before | pts after |
|---|---|---|---|---|---|---|---|
| c_adult | 0.8115 | 0.8143 | +0.0028 | 23.1 | 7.3 | 3.82 | 3.97 |
| c_ai4i | 0.8492 | 0.8420 | -0.0072 | 10.6 | 4.8 | 4.07 | 4.09 |
| c_bank | 0.7739 | 0.7750 | +0.0011 | 27.4 | 12.0 | 3.35 | 3.46 |
| c_covertype | 0.6159 | 0.6059 | -0.0100 | 32.6 | 17.9 | 3.35 | 3.37 |
| c_creditg | 0.7037 | 0.7074 | +0.0037 | 7.1 | 4.9 | 3.10 | 3.21 |
| c_telco | 0.7279 | 0.7242 | -0.0037 | 10.0 | 5.2 | 3.25 | 3.29 |
| k_drybean | 0.6610 | 0.6600 | -0.0010 | 3.4 | 4.6 | 3.95 | 3.95 |
| k_iris | 0.6306 | 0.5801 | -0.0505 | 1.8 | 1.9 | 3.81 | 3.57 |
| k_mall | 0.6175 | 0.6651 | +0.0476 | 1.7 | 1.9 | 3.75 | 3.97 |
| k_penguins | 0.6534 | 0.9505 | +0.2971 | 1.8 | 1.9 | 3.92 | 5.32 |
| k_seeds | 0.7722 | 0.7722 | +0.0000 | 1.7 | 1.8 | 4.48 | 4.48 |
| k_wine | 0.8975 | 0.8805 | -0.0170 | 1.6 | 1.9 | 5.07 | 4.99 |
| r_abalone | 0.5751 | 0.5762 | +0.0010 | 8.7 | 4.3 | 3.49 | 3.55 |
| r_bike | 0.9620 | 0.9625 | +0.0005 | 33.9 | 13.1 | 5.16 | 5.27 |
| r_california | 0.8433 | 0.8421 | -0.0012 | 33.2 | 11.4 | 4.60 | 4.72 |
| r_concrete | 0.8884 | 0.8884 | +0.0001 | 5.3 | 4.2 | 5.02 | 5.03 |
| r_diamonds | 0.9811 | 0.9808 | -0.0004 | 32.1 | 9.3 | 5.26 | 5.39 |
| r_house | 0.8211 | 0.8978 | +0.0767 | 16.0 | 10.3 | 4.58 | 4.99 |

### C_rules
| dataset | before | after | Δ | wall before | wall after | pts before | pts after |
|---|---|---|---|---|---|---|---|
| c_adult | 0.8150 | 0.8150 | +0.0000 | 4.4 | 4.7 | 4.02 | 4.02 |
| c_ai4i | 0.8517 | 0.8517 | +0.0000 | 2.9 | 3.1 | 4.18 | 4.18 |
| c_bank | 0.7674 | 0.7674 | +0.0000 | 6.0 | 6.3 | 3.47 | 3.46 |
| c_covertype | 0.5721 | 0.5721 | +0.0000 | 14.6 | 15.3 | 3.21 | 3.21 |
| c_creditg | 0.6951 | 0.6686 | -0.0265 | 4.8 | 5.6 | 3.07 | 2.88 |
| c_telco | 0.7344 | 0.7344 | +0.0000 | 3.5 | 3.7 | 3.38 | 3.38 |
| k_drybean | 0.7242 | 0.6253 | -0.0989 | 3.1 | 3.8 | 4.25 | 3.79 |
| k_iris | 0.9039 | 0.9039 | +0.0000 | 1.8 | 1.8 | 5.10 | 5.10 |
| k_mall | 0.9696 | 0.9561 | -0.0135 | 1.8 | 1.8 | 5.41 | 5.35 |
| k_penguins | 0.5805 | 0.7685 | +0.1880 | 1.7 | 1.8 | 3.57 | 4.46 |
| k_seeds | 0.7224 | 0.7733 | +0.0509 | 1.6 | 1.8 | 4.24 | 4.49 |
| k_wine | 0.9150 | 0.9150 | +0.0000 | 1.8 | 1.8 | 5.15 | 5.15 |
| r_abalone | 0.5540 | 0.5540 | +0.0000 | 2.5 | 2.7 | 3.45 | 3.45 |
| r_bike | 0.9598 | 0.9598 | +0.0000 | 6.9 | 7.1 | 5.33 | 5.33 |
| r_california | -0.0445 | 0.8407 | +0.8852 | 52.3 | 5.4 | 0.56 | 4.79 |
| r_concrete | 0.8818 | 0.8818 | +0.0000 | 3.4 | 2.8 | 5.00 | 5.00 |
| r_diamonds | 0.9800 | 0.9800 | +0.0000 | 4.6 | 4.9 | 5.46 | 5.46 |
| r_house | 0.7507 | 0.8963 | +0.1456 | 4.3 | 4.6 | 4.38 | 5.07 |

---

## Tried and rejected (with reasons)

| where | tried | result | decision |
|---|---|---|---|
| A1 | Soft sparse weights as the clustering space instead of a hard filter | seeds 0.773 → 0.698, wine 0.897 → 0.848, mixed-cat dev 0.695 → 0.626 (weights hurt when no feature is irrelevant) | hard filter (drop weight < 5 %) + unweighted KMeans |
| A1 | Also require a low explained share under the base partition before dropping (to keep wine's Ash) | the base partition is itself diluted by noise: iris `noise_3` (0.12) and dev Age/junk (0.35–0.40) would be kept | not used; accept the −0.017 on wine |
| A1 | Mall | Gender is dropped, but Age keeps a high weight (it is not irrelevant to KMeans), so the partition is unchanged. Mall labels are derived from Income/Spending only (docs/04 §10) | no Mall-specific rule (would be overfitting) |
| A2 | GMM with BIC gain ≥ 15 % and stability ≥ 0.75 | seeds GMM-diag accepted (0.773 → 0.679); drybean GMM-full accepted (0.668) | stability 0.8, BIC gain 30 % |
| A2/C4b | drybean's round-1 0.724 (C) | reproduced only with one specific KMeans initialisation; 5 seeds + `n_init=5` all give 0.636–0.638 with a **better** BIC. The 0.724 partition was a lucky local optimum | not a target: A keeps KMeans (0.660); C accepts GMM-full (0.625) |
| A3 | `class_weight="balanced"` / capped-balanced for multiclass | weighted early-stopping loss stops after ~50 iterations: covertype single model 0.524–0.535 | sqrt-balanced (0.604 single) |
| A3 | 63 or 127 leaves, min_samples_leaf 10/5 | 0.557–0.586 single, slower | not used |
| A3 | No early stopping, 400 iterations | 0.599 single but 8.3 s per fit | patience 60 instead |
| A3 | ExtraTrees 200 trees | covertype 0.615 in 18.4 s | 100 trees: 0.620 in 15.0 s |
| A S2 | iteration cap from the quick-fit rate without a safety factor | capped covertype at 138 iterations on this machine (the quick fit over-estimates the per-iteration cost) | factor 3: never binds here, still catches 10× blow-ups |
| B1 | forests 150 trees / 15k samples on large n | same quality (±0.005), +2–4 s per large dataset | 100 trees / 8k samples |
| B1 | a wall-clock "soft 20 s target" for adding candidates | would make the candidate set CPU-dependent again (conflicts with B2) | deterministic size-based plan + OOF-gain stopping rule; clock only as safety |
| B4 | iris | after filtering, KMeans / Ward / spectral / PCA views agree on a KMeans-like partition (≈ 0.58–0.62); the GMM-full partition that solves iris (0.90) is one minority candidate | not forced: choosing a minority answer would contradict the consensus philosophy |
| C4 | use the weighted space always (C's round-1 behaviour, group-normalised) | seeds 0.698, drybean 0.619 | rule: weighted only if some weight < 5 % |
| C4b | agreement ≥ 0.75 (would reject drybean's GMM-full at 0.70) | also rejects Mall's GMM-diag (agreement 0.64, ARI 0.956) | kept 0.6 |
| C | credit-g 0.695 → 0.669 | caused by S1 (the `own_telephone` / literal `none` levels are now features); same single split; A +0.018, B +0.004 on the same change | accepted (correctness fix) |

## S4 — older-stack and fresh-install verification

| check | result |
|---|---|
| fresh venv, `pip install -r requirements.txt` (pinned stack; pulls also cloudpickle / narwhals / dateutil / tzdata as sklearn-pandas dependencies) | installs cleanly; A/B/C on `c_telco` give exactly the main-environment macro-F1 (0.7274 / 0.7242 / 0.7344) |
| cold vs warm start in another brand-new venv (A, `k_seeds`) | 1st run 6.8 s, 2nd 2.0 s, 3rd 2.0 s (pip pre-compiles `.pyc`; the remaining cold cost is DLL loading / scanning). The 59 s first import seen once in docs/01 was a worst case; the README warm-up step covers it |
| older stack: numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2 — 18 suite datasets x 3 approaches | 54/54 valid, no traceback; primary metric vs pinned stack: mean diff −0.002, max abs diff 0.011 (A) / 0.022 (B) / 0.025 (C) (floating-point / algorithm-version noise); max wall 14.8 / 15.5 / 14.5 s |
| older stack — 22 dev datasets x 3 approaches | 66/66 valid, no traceback |
| scikit-learn 1.3–1.6, pandas 2.0–2.2 | not testable here (no Python 3.14 wheels); code paths guarded (TargetEncoder import, HistGB `max_features` probe, `format="mixed"` fallback) |
