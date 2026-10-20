# 03 — Datasets used during development of approaches A, B, C

Purpose: let the Validator avoid (or flag) overlap between development data and scoring data.
All are produced by `py tools/build_dev_datasets.py` into `dev_data/` (gitignored). Random seed 12345,
train/test split seed 0, test_size 0.25.

**No OpenML / UCI / Kaggle dataset was downloaded or used.** In particular Telco Churn, AI4I 2020, House
Prices, Bike Sharing and Mall Customers were NOT used (only synthetic look-alikes of their *data
issues*, e.g. a numeric-as-string `TotalCharges`-style column and a bike-sharing-like datetime count
target generated from a formula).

## sklearn built-ins used (avoid for scoring, or flag)
| dev name | source | used for |
|---|---|---|
| `c_breast_cancer` | `sklearn.datasets.load_breast_cancer` (labels mapped to strings) | smoke test, string target |
| `r_diabetes` | `sklearn.datasets.load_diabetes(scaled=False)` | smoke test, small regression |

## Synthetic datasets
| dev name | task | rows | what it exercises |
|---|---|---|---|
| `c_synth_imb` | clf | 3000 | `make_classification(random_state=1)` 7 % minority, yes/no target, x3 x1000 scale, signal categorical, constant col, duplicate col, string ID, 10 % missing |
| `c_multi_nonlinear` | clf | 6000 | 6 ring/angle classes (nonlinear), 6 noise features, 300-level categorical, datetime string |
| `c_tiny` | clf | 80 | 3 weak classes, tiny n |
| `c_edge` | clf | 240 | numeric-as-string with blanks, singleton class, all-NaN col, constant col, bool col missing from test, unseen test category |
| `c_large` | clf | 60000 | `make_classification(random_state=3)` 40 features, 3 classes (timing) |
| `c_text_nantarget` | clf | 700 | free-text column carrying the signal, 3 % missing targets, float labels |
| `c_bool_target_degenerate` | clf | 300 | boolean target, single useful feature, constant / empty columns, extra test column |
| `c_many_classes` | clf | 1500 | `make_classification(random_state=9)` 20 classes |
| `r_friedman` | reg | 2000 | `make_friedman1(random_state=4)` + 15 % missing + duplicate col |
| `r_counts_datetime` | reg | 5000 | bike-sharing-LIKE Poisson counts from hour/day/temp/weather formula, declared types |
| `r_tiny` | reg | 40 | metadata without `task`/`feature_types` (task inference) |
| `r_prices_strings` | reg | 1500 | log-normal price, "1,234"-formatted numbers, 25-level categorical, `Id` column, no declared types |
| `r_wide` | reg | 1500 | 400 features (15 informative + 1 nonlinear), 150-level categorical (timing / wide path) |
| `k_blobs_noise` | clu | 400 | Mall-LIKE 5 blobs in (Income, Spending) + Gender/Age/3 uniform junk features + CustomerID |
| `k_aniso_scaled` | clu | 3000 | anisotropic blobs, one column x1000, skewed noise col, 5 % missing, train/test variant |
| `k_mixed_cat` | clu | 1200 | 4 clusters defined by 2 numerics + 2 categoricals |
| `k_tiny` | clu | 30 | 3 blobs, tiny n |
| `k_large` | clu | 60000 | 6 blobs in 12 dims + 8 noise dims (timing) |
| `k_no_k` | clu | 600 | metadata without `n_clusters` |
| `k_all_categorical` | clu | 500 | only categorical features, no declared types |

## How the dev data was used
- Smoke testing (validity, crashes, timing) of all three approaches — results in each approach README.
- A few design choices were informed by these runs: capped balanced class weights (singleton class in
  `c_edge`), text token features (`c_text_nantarget`), `max_bins=63` on wide data (`r_wide` timing), the
  A tiny-data / many-class linear blend and the C "many classes" rule (`c_many_classes`).
- No hyper-parameter search was run on any dataset; C's rule thresholds are generic defaults, not fitted.

## Deviations from the Phase-2 brief
- Approach C has **no learned meta-selector** (only the hand-written rule base + linearity probe).
  Training one would need a meta-dataset of many real datasets — which would very likely overlap with
  the Validator's scoring datasets (leakage) — so it was deliberately left out. It can be added later
  from datasets the Validator does not score on.
- LightGBM/XGBoost/CatBoost are not used by any approach (assessment-environment risk, see docs/01 §6.3);
  everything is numpy/pandas/scipy/scikit-learn.

## Fix round 1 (docs/05): the Validator's round-1 suite was used for re-testing
During fix round 1 the 18 round-1 suite datasets (validation/SUITE.md) were used to check every fix
(before/after tables in docs/05), and the clustering guard thresholds (GMM stability / agreement / BIC gain,
the 5 % relevance cut) were checked on them together with the dev sets. **Round-1 suite scores are
therefore optimistic from now on; round 2 must use new datasets** (none of telco, ai4i, adult, bank,
covertype, credit-g, Ames house, bike, california, abalone, concrete, diamonds, Mall, iris, wine, seeds,
penguins, drybean). No dataset-specific rule (names, columns, sizes) was added to any approach.
