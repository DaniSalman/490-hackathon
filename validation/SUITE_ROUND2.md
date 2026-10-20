# Validation suite — round 2 (18 new datasets + 4 stress probes)

Built by `py validation/build_suite_round2.py` (fixed seeds, reproducible, ~6 s once downloads are cached).
Output: `validation/data/suite_r2/<name>/` and `validation/data/stress_r2/<name>/` (gitignored), docs/02 folder
format, ground truth in `hidden/` (`test_labels.csv`, `info.json`, `truth.json`). Same conventions as round 1
(`validation/SUITE.md`):
- **Metadata:** `task`, `target`, `n_clusters`, and declared `feature_types` (numeric dtype → numeric,
  else categorical, datetimes declared datetime). Planted / native undeclared ID columns are not declared.
- **Splits:** 75/25, stratified for classification. Exception: `r_kc_house` uses a deliberate non-random split.
- **Clustering:** variant 1 (`data.csv`, labels for all rows).

The builder reuses the round-1 download and injection helpers (`build_suite.py`, unchanged).

**Overlap check:** none of these datasets is in round 1 (telco, ai4i, adult, bank, covertype, credit-g, house
prices, bike sharing, california, abalone, concrete, diamonds, mall, iris, wine, seeds, penguins, dry bean), in
the round-1 stress probes, or in docs/03 (synthetic sets + sklearn breast_cancer / diabetes).
- `r_wine` is the UCI *wine-quality* regression set. It is unrelated to the round-1 `k_wine` (UCI *wine*
  cultivars, 178 rows).
- `k_synth` is a new synthetic set. It differs in generator, seed and structure from the dev sets
  `k_aniso_scaled` / `k_mixed_cat` / `k_blobs_noise`, but probes the same kind of issue (flagged).

| # | name | task | source (ID / URL) | train x cols | test | classes / k | native challenges | injected | probes fix |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c_churn` | clf binary | OpenML 40701 churn (MLC++ telecom churn) | 3750 x 22 | 1250 | 2 (yes 14.1 %) | imbalance; `state` 51 levels (int codes, declared categorical → high cardinality); `phone_number` unique int (ID by name, declared categorical) | **`addon_plan`** categorical with literal levels `none / basic / premium / unknown / -` carrying signal (`-` never churns, `none` 27 % churn); **`total_eve_charge`** as strings with 124 `" "` blanks (declared numeric) | **S1**, **S3**, S2a (state one-hot width) |
| 2 | `c_magic` | clf binary | OpenML 1120 MagicTelescope (UCI) | 14265 x 18 | 4755 | 2 (h 35 %) | nonlinear, heavy tails, column names with `:` removed, native `ID` row id (undeclared) | duplicate `fLength_mm`, constant `telescope`, 4 noise features (one x1000) | duplicate / constant / ID, irrelevant features |
| 3 | `c_pageblocks` | clf 5-class | OpenML 30 page-blocks (UCI) | 4105 x 11 | 1368 | 5 (89.8 / 6.0 / 2.1 / 1.6 / **0.5 %**; `graphic` 21 train rows) | rare-class multiclass, skewed scales | - | **A3** (rare-class multiclass weights + ET blend), B/C multiclass |
| 4 | `c_creditapp` | clf binary | OpenML 29 credit-approval (UCI) | 518 x 19 | 172 | 2 (44 / 56 %) | small n, 9 categoricals (A6 14 levels), native missing 53 cells in train | 3 noise features | **A4** (linear blend n < 2000), small-data behaviour of B/C |
| 5 | `c_jannis` | clf 4-class | OpenML 41168 jannis (AutoML benchmark), stratified 25000-row subsample | 18750 x 97 | 6250 | 4 (46 / 34 / 18 / **2 %**) | 54 anonymised numerics, rare class | **40 noise features**, duplicate `V1_copy`, constant `batch` → 96 features | **S2** cost guard (wide table near the size limit), A3 on a large n, B1 portfolio sizing |
| 6 | `c_kick` | clf binary | OpenML 41162 kick (Kaggle "Don't Get Kicked"), 20000-row subsample | 15000 x 33 | 5000 | 2 (bad 12.0 %) | `PurchDate` datetime string `m/d/Y` (declared datetime); **high-cardinality** Model (753) / SubModel (607) / Trim (111) / VNST; zip / buyer codes declared categorical; 30783 missing cells (PRIMEUNIT / AUCGUART 95 %) | - | S2a (one-hot width cap), C target encoding on a binary target, datetime |
| 7 | `r_seoul` | reg | UCI 560 Seoul Bike Sharing Demand | 6570 x 14 | 2190 | - | `Date` as **dd/mm/yyyy** (day-first, declared datetime), count target with zeros (non-functioning days), skew 1.14, nonlinear hour × temperature, unicode column names (`°C`) | - | **A5** (log1p / RMSLE on a count target), datetime parsing |
| 8 | `r_kc_house` | reg | OpenML 42731 house_sales (Kaggle King County) | 16127 x 21 | 5486 | - | **integer-valued price** (skew 2.65) + **`zipcode` 70 levels** declared categorical; `date` in Kaggle format `20141013T000000`; `id` repeated for re-sold houses (undeclared ID) | **non-random split:** all houses with sqft_living ≥ p99.5 and 75 % of p99..p99.5 in test, so 117 test rows have sqft_living above the training maximum | **C1** (TargetEncoder target type), **S2**, **B3 / C3** (out-of-range test rows, linear clipping / exclusion) |
| 9 | `r_moneyball` | reg | OpenML 41021 Moneyball (MLB team seasons) | 924 x 15 | 308 | - | **integer target RS + `Team` 37–39 levels (> 30)** at small n; RankSeason / RankPlayoffs 80 % missing, OOBP / OSLG 66 % missing; near-linear target | - | **C1** at small n, high-missing columns, C3 linear-probe margin |
| 10 | `r_superconduct` | reg | OpenML 43174 superconduct (UCI), 15000-row subsample | 11250 x 86 | 3750 | - | 81 highly correlated physico-chemical features, skewed critical_temp, nonlinear | duplicate `mean_atomic_mass_u`, 3 noise features | **S2** (wide regression), B1 timing |
| 11 | `r_brazil` | reg | OpenML 42688 Brazilian_houses (Kaggle "houses to rent") | 8019 x 12 | 2673 | - | rent skewed (1.73), **extreme native outliers** (hoa up to 1.1 M, area 46335 in test vs train max 2000), city categorical | total_(BRL) dropped (it is the sum incl. the target) | **B3 / C3** (out-of-range rows), log-target choice |
| 12 | `r_wine` | reg | UCI 186 wine quality (red + white merged, `type` column) | 4873 x 15 | 1624 | - | integer target quality 3..9 concentrated on 5–7, scales | **`residual sugar`** as strings with 198 `" "` blanks; **`sulphates`** with 100 `"?"` tokens (both declared numeric); 2 noise features | **S1 / S3** (numeric NA tokens + blank strings in declared numeric columns) |
| 13 | `k_wholesale` | clust | OpenML 1511 wholesale-customers (UCI), duplicated Region column fixed | 440 x 7 | - | k = 2 (Horeca 68 % / Retail 32 %) | customer-segmentation table (Mall-like); spending heavily right-skewed; `Region` categorical irrelevant | - | log/robust preprocessing, A1/B4/C4 relevance filter (Region) |
| 14 | `k_vehicle` | clust | OpenML 54 vehicle (Statlog) | 846 x 21 | - | k = 4 | saab / opel nearly inseparable, correlated features, scales 1..1000 | 3 noise features | **A1 / B4 / C4a** (planted irrelevant features), A2 / C4b GMM acceptance |
| 15 | `k_ecoli` | clust | OpenML 39 ecoli (UCI) | 336 x 7 | - | k = 8 (143 / 77 / 52 / 35 / 20 / 5 / 2 / 2) | **very unequal group sizes**, near-constant binary lip / chg | - | GMM min-size guard (A2 / C4b), ensure-k |
| 16 | `k_pendigits` | clust | OpenML 32 pendigits (UCI), 6000-row subsample | 6000 x 16 | - | k = 10 | non-convex digit manifolds, larger n | - | B4 spectral / consensus at n > 2500, timing |
| 17 | `k_segment` | clust | OpenML 36 segment (UCI image segmentation) | 2310 x 20 | - | k = 7 balanced | **native constant** `region-pixel-count`, scales 0..250 vs 0..1, elongated classes | undeclared string ID `image_id` | constant / ID handling in clustering, GMM full vs diag |
| 18 | `k_synth` | clust | synthetic (`build_suite_round2.k_synth`, seed 291) | 3000 x 11 | - | k = 4 (1200 / 900 / 600 / 300) | - | **anisotropic, unequal-covariance groups** (non-spherical), informative numerics on scales x10 / x1000 / x0.01, **informative categorical `plan`** (80 % agreement; levels none / basic / plus / unknown), **5 noise numerics + random categorical `noise_flag`**, undeclared ID `customer_id` | **A1, A2, B4, C4a, C4b**, S1 in clustering |

### Challenge coverage
| challenge | datasets |
|---|---|
| missing values | c_creditapp, c_kick, r_moneyball (native); c_churn / r_wine blank and `?` cells (S3) |
| categoricals incl. high cardinality | c_kick (Model 753, SubModel 607), c_churn state 51, r_kc_house zipcode 70, r_moneyball Team 39; low-card in c_creditapp, r_brazil, r_wine, k_wholesale, k_synth |
| literal NA-like category tokens (S1) | c_churn `addon_plan`, k_synth `plan` |
| blank-string missing (S3) | c_churn `total_eve_charge`, r_wine `residual sugar` (+ `?` in `sulphates`) |
| different scales | k_synth, k_segment, k_vehicle, r_brazil, c_pageblocks |
| constant columns | c_magic, c_jannis (inj), k_segment (native) |
| duplicate columns | c_magic, c_jannis, r_superconduct (inj) |
| ID columns (undeclared) | c_magic `ID`, c_churn `phone_number`, r_kc_house `id`, k_segment `image_id`, k_synth `customer_id` |
| irrelevant noise features | c_magic, c_creditapp, c_jannis (40), r_superconduct, r_wine, k_vehicle, k_synth |
| binary imbalance | c_kick 12 %, c_churn 14 %, c_magic 35 % |
| multiclass imbalance / rare classes | c_pageblocks (0.5 %), c_jannis (2 %), k_ecoli (groups of 2) |
| nonlinear | c_magic, c_jannis, r_seoul, r_superconduct, k_pendigits, k_synth |
| skewed / count target | r_seoul (count with zeros), r_kc_house, r_brazil (prices / rents), r_superconduct |
| integer target + > 30-level categorical (C1) | r_kc_house, r_moneyball |
| out-of-range test rows (B3 / C3) | r_kc_house (planted), r_brazil (native) |
| datetime column | c_kick, r_seoul (day-first), r_kc_house (compact ISO) |
| wide table near the size limit (S2) | c_jannis 18750 x 97, r_superconduct 11250 x 86 |
| small n (A4) | c_creditapp 518, r_moneyball 924 |

## Round-2 stress probes (`validation/data/stress_r2/`, scored separately)

| probe | base | change |
|---|---|---|
| `s2_churn_tokens` | c_churn | `?`, `-`, `none` tokens (5 %) inside 3 numeric columns + the base blanks; **no `feature_types`** |
| `s2_kick_notypes` | c_kick | **no `feature_types`** (datetime / high-cardinality / code columns must be inferred) |
| `s2_kc_house_strnum` | r_kc_house | sqft columns as `"1,234"` strings + 15 % MCAR in 5 columns; **no `feature_types`** |
| `s2_synth_traintest` | k_synth | docs/02 clustering **variant 2** (train/test) + unseen `plan` level `enterprise` in 15 % of test rows |
