# Validation suite (18 private-like datasets + 9 stress probes)

Built by `py validation/build_suite.py` (fixed seeds, reproducible; ~10 s once the downloads are cached).
Output: `validation/data/suite/<name>/` and `validation/data/stress/<name>/` in the docs/02 folder format
(`train.csv` + `test.csv` + `metadata.json` for supervised tasks, `data.csv` + `metadata.json` for clustering,
ground truth in `hidden/`). Data and caches are gitignored and never committed.

Each folder also has `hidden/truth.json`: the planted / known challenges (ID, constant, duplicate, noise,
datetime columns, injected missing counts) used as ground truth for profile scoring, plus a list of native
properties and every injection.

**Metadata given to the solvers** (mirrors "each dataset tells you its task, feature types and, for clustering,
the required number of groups"): `task`, `target` (supervised), `n_clusters` (clustering), `feature_types`
(numeric dtype -> `numeric`, otherwise `categorical`, datetimes declared `datetime`, a few numeric codes declared
`categorical`) and `id_column` where the original data has an obvious ID. Planted ID columns are NOT declared
(neither in `id_column` nor in `feature_types`) so that ID inference is tested.

Splits: supervised 75/25, stratified for classification (`tools/make_dataset.py`, seed 0). Clustering: single
`data.csv`, labels for every row (docs/02 variant 1); the penguins stress probe uses variant 2 (train/test).

Sources: OpenML ARFF via the REST API (`https://www.openml.org/api/v1/json/data/<id>`; `sklearn.fetch_openml`
hung on this machine so the builder downloads and parses the ARFF itself), UCI zip archives, and a public
GitHub mirror of the Kaggle `Mall_Customers.csv`.

## The 18 datasets

| # | name | task | source (ID / URL) | train x cols | test | classes / k | challenges: native | challenges: injected | why it represents the private sets |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c_telco` | clf (binary) | OpenML 42178 telco-customer-churn (= Kaggle WA_Fn-UseC_ Telco Churn) | 5282 x 21 | 1761 | 2 (Yes 26.5 %) | `TotalCharges` numeric stored as string with " " blanks (declared numeric), `customerID` (declared id_column), 16 categoricals, imbalance | - | **Spec example** (Telco Customer Churn) |
| 2 | `c_ai4i` | clf (binary) | OpenML 42890 AI4I2020 (= UCI 601 / Kaggle) | 7500 x 14 | 2500 | 2 (failure 3.4 %) | strong imbalance, `UDI` (id_column), `Product ID` unique string (declared categorical -> must be detected as ID), failure-mode flags TWF..RNF present in train and test (as in Kaggle file) | - | **Spec example** (Predictive Maintenance AI4I 2020) |
| 3 | `c_adult` | clf (binary) | OpenML 1590 adult (UCI census income), 20000-row random subsample (seed 0) | 15000 x 15 | 5000 | 2 (>50K 23.9 %) | native missing (workclass, occupation, native-country), 41-level categorical, irrelevant `fnlwgt`, re-encoded `education-num`, skewed capital-gain | - | classic Kaggle-style tabular churn-like binary task, larger n |
| 4 | `c_bank` | clf (binary) | OpenML 1461 bank-marketing (UCI), 20000-row subsample (seed 1), original UCI column names restored | 15000 x 24 | 5000 | 2 (yes 11.7 %) | imbalance, "unknown" pseudo-missing levels, pdays=-1 sentinel | constant `campaign_channel`, duplicate `balance_eur`, 5 noise features (one x1000 scale) | imbalanced marketing/churn data, constant + duplicate + irrelevant columns (spec list) |
| 5 | `c_covertype` | clf (7-class) | OpenML 180 covertype (110393-row version), 15000-row subsample (seed 2) | 11250 x 55 | 3750 | 7 (largest 46.6 %, smallest 1.3 %) | multiclass imbalance, nonlinear, 44 one-hot 0/1 columns (`soil_type_15` constant in train -> natural constant column), wide-ish | - | multiclass + imbalanced + nonlinear; near the upper size of the examples |
| 6 | `c_creditg` | clf (binary) | OpenML 31 credit-g (Statlog German credit) | 750 x 30 | 250 | 2 (bad 30 %) | small n, 13 categoricals | MCAR 10 % in 5 cols, duplicate `loan_amount_copy`, 6 noise features (x1000 scale), irrelevant datetime `application_date` (declared datetime), undeclared string ID `applicant_id` | small dirty table with every spec issue at once |
| 7 | `r_house` | reg | OpenML 42165 house_prices (= Kaggle House Prices / Ames) | 1095 x 81 | 365 | - | 19 cols with missing (PoolQC 99.6 %), 43 categoricals, right-skewed SalePrice (skew 1.8), `Id` (id_column), p > 80 | - | **Spec example** (House Prices) |
| 8 | `r_bike` | reg | UCI 275 Bike Sharing `hour.csv` rebuilt in the Kaggle Bike Sharing Demand schema | 13034 x 10 | 4345 | - | `datetime` string column (hour signal only reachable through it, declared datetime), skewed count target (skew 1.3), nonlinear, categorical codes | casual/registered dropped (absent in Kaggle test); temp/atemp/humidity/windspeed de-normalised to Kaggle units | **Spec example** (Bike Sharing Demand) |
| 9 | `r_california` | reg | OpenML 43939 california_housing (Kaggle/Geron version with `ocean_proximity` and missing `total_bedrooms`) | 15480 x 14 | 5160 | - | 149 missing in train, categorical, capped target, different scales | high-cardinality categorical `geo_cell` (529 levels), 3 noise features | realistic housing regression, high-cardinality categorical, larger n |
| 10 | `r_abalone` | reg | OpenML 183 abalone (UCI) | 3133 x 11 | 1044 | - | integer target Rings (skew 1.1), categorical Sex, heteroscedastic | duplicate `Whole_weight_g`, constant `survey_year` | integer-valued target, constant + duplicate columns |
| 11 | `r_concrete` | reg | OpenML 4353 Concrete_Data (UCI) | 772 x 15 | 258 | - | small n, strongly nonlinear (age, water/cement) | MCAR 10 % in 4 cols, 5 noise features, undeclared row-counter ID `sample_no` | nonlinear small regression with missing + irrelevant features |
| 12 | `r_diamonds` | reg | OpenML 42225 diamonds, 15000-row subsample (seed 5) | 11250 x 10 | 3750 | - | price strongly right-skewed / multiplicative (skew 1.65), ordinal categoricals cut/color/clarity, x/y/z zeros (disguised missing) | - | skewed price target like House Prices but larger n |
| 13 | `k_mall` | clust | Kaggle Mall_Customers.csv (GitHub mirror of the Kaggle file) | 200 x 5 | - | k = 5 | `CustomerID` (id_column), Gender categorical, Age irrelevant for the canonical segments | **labels derived** (no true labels exist): KMeans(5, n_init=100, seed 0) on unscaled (Annual Income, Spending Score) = the canonical 5 segments | **Spec example** (Mall Customer Segmentation) - real file, derived labels (assumption) |
| 14 | `k_iris` | clust | OpenML 61 iris | 150 x 9 | - | k = 3 | small n | 3 noise features (normal / uniform / exponential), constant `lab`, undeclared int ID `specimen_id` | irrelevant features + ID + constant in a tiny clustering table |
| 15 | `k_wine` | clust | OpenML 187 wine (UCI) | 178 x 13 | - | k = 3 | very different feature scales (Proline ~1000 vs Hue ~1): unscaled KMeans fails | - | "differently scaled features" (spec) |
| 16 | `k_seeds` | clust | OpenML 1499 seeds (UCI), original column names restored | 210 x 7 | - | k = 3 | small clean numeric, correlated features | - | typical easy clustering set |
| 17 | `k_penguins` | clust | OpenML 42585 penguins (Palmer) | 344 x 6 | - | k = 3 | native missing (sex 10, each measurement 2), categoricals island/sex (one dirty `_` sex value), scales g vs mm | - | mixed numeric/categorical clustering with missing values |
| 18 | `k_drybean` | clust | UCI 602 Dry Bean (ARFF in the UCI zip) | 13611 x 17 | - | k = 7 | larger n, 16 shape features with scales 1e-3 .. 2e5, highly correlated, unequal cluster sizes | duplicate `Area_px` | larger clustering table (timing of O(n^2) components), scale + redundancy |

Totals: 6 classification (4 binary imbalanced, 1 multiclass imbalanced, 1 small dirty), 6 regression
(3 skewed targets, 1 count target with datetime, 1 integer target, 1 nonlinear small), 6 clustering
(k = 3..7, n = 150..13611).

### Challenge coverage
| challenge (spec / docs/01 §3) | datasets |
|---|---|
| missing values | c_adult, c_creditg (inj), r_house, r_california, r_concrete (inj), k_penguins |
| categorical columns | c_telco, c_ai4i, c_adult, c_bank, c_creditg, r_house, r_bike, r_california, r_abalone, r_diamonds, k_mall, k_penguins |
| high-cardinality categorical | c_adult (41), r_california `geo_cell` (529, inj), r_house Neighborhood (25); ID-like strings c_telco, c_ai4i `Product ID`, c_creditg |
| different scales | k_wine, k_drybean, k_penguins, c_bank / c_creditg noise x1000 (inj) |
| constant columns | c_bank (inj), c_covertype `soil_type_15` (native), r_abalone (inj), k_iris (inj) |
| duplicate columns | c_bank, c_creditg, r_abalone, k_drybean (all inj) |
| ID columns | declared: c_telco, c_ai4i, r_house, k_mall; undeclared: c_ai4i `Product ID`, c_creditg `applicant_id`, r_concrete `sample_no`, k_iris `specimen_id` |
| irrelevant / noise features | c_bank, c_creditg, r_california, r_concrete, k_iris (inj); c_adult `fnlwgt`, k_mall Age/Gender (native) |
| imbalanced classes | c_ai4i 3.4 %, c_bank 11.7 %, c_adult 24 %, c_telco 26.5 %, c_covertype multiclass (1.3 % smallest) |
| nonlinear relationships | c_covertype, r_bike, r_concrete, r_california |
| skewed / count target | r_bike (count), r_house, r_diamonds (prices), r_abalone (integer) |
| datetime column | r_bike (signal), c_creditg (irrelevant, inj) |
| numeric stored as string | c_telco `TotalCharges` (native) + stress probes |

### Overlap with docs/03 (developer data)
**None.** The developer used only synthetic sets plus sklearn `breast_cancer` and `diabetes`; neither is in this
suite. Caveat: the developer built synthetic *look-alikes* of the data issues of Telco (numeric-as-string
`TotalCharges`), Bike Sharing (datetime + Poisson count) and Mall (5 blobs in Income/Spending + junk features).
The real files used here were never seen by the approaches, but those three data issues were anticipated.

## Stress probes (scored separately, not part of the 18-dataset score)

One base dataset per task, three variants each, in `validation/data/stress/`:

| probe | base | change |
|---|---|---|
| `s_telco_missing`, `s_house_missing`, `s_penguins_missing` | c_telco / r_house / k_penguins | +20 % MCAR in every feature column (train and test) |
| `s_telco_strnum` | c_telco | `MonthlyCharges` as `"$1,234.50"`, `TotalCharges` as `"1,234.50"`, `tenure` as string; **no `feature_types`** in metadata |
| `s_house_strnum` | r_house | LotArea / GrLivArea / TotalBsmtSF / 1stFlrSF as `"12,345"`, MasVnrArea with `"NA"`; **no `feature_types`** |
| `s_penguins_strnum` | k_penguins | body_mass_g as `"3,750"`, flipper_length_mm as strings with `"?"`; **no `feature_types`** |
| `s_telco_unseen` | c_telco | 15 % of test rows get unseen levels in PaymentMethod / InternetService / Contract |
| `s_house_unseen` | r_house | 15 % of test rows get unseen levels in Neighborhood / SaleCondition / KitchenQual |
| `s_penguins_unseen` | k_penguins | **train/test clustering variant** (docs/02 variant 2) + 15 % of test rows with unseen island / sex levels |
