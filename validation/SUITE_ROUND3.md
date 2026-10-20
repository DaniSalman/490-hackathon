# Validation suite — round 3 (18 new datasets + 4 stress probes)

Built by `py validation/build_suite_round3.py` (fixed seeds, reproducible; about 7 s once the downloads are
cached). Output goes to `validation/data/suite_r3/` and `validation/data/stress_r3/` (gitignored).

Format and conventions are as in rounds 1–2:
- docs/02 folder format; ground truth in `hidden/` (`test_labels.csv`, `info.json`, `truth.json`).
- Declared `feature_types`: numeric dtype → numeric, otherwise categorical. Datetimes and integer codes are
  declared explicitly. Undeclared ID columns are left undeclared.
- Splits are 75/25, stratified for classification. Two regression sets use a deliberate non-random split.
- Clustering uses variant 1 (`data.csv`).

**Overlap check: none.**
- Round 1: telco, ai4i, adult, bank, covertype, credit-g, Ames house, bike, california, abalone, concrete,
  diamonds, mall, iris, wine, seeds, penguins, dry bean.
- Round 2: churn, magic, page-blocks, credit-approval, jannis, kick, seoul, kc_house, moneyball, superconduct,
  brazil, wine-quality, wholesale, vehicle, ecoli, pendigits, segment, k_synth.
- All stress probes of rounds 1–2, and docs/03 (synthetic sets + sklearn breast_cancer / diabetes).
- Note: `c_credit` (Taiwan credit-card default, OpenML 42477) is not credit-g (German credit, round 1) and
  not credit-approval (round 2).
- `k_blobs_easy` and `k_moons` are new synthetic sets with new seeds and structure.

| # | name | task | source (ID / URL) | train x cols | test | classes / k | native challenges | injected | probes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c_amazon` | clf binary | OpenML 4135 Amazon_employee_access (Kaggle), stratified 20k of 32769 | 15000 x 10 | 5000 | 2 (denied **5.8 %**) | **all 9 features are integer-coded high-cardinality categoricals** (RESOURCE 4881, MGR_ID 3482, ROLE_DEPTNAME 432 …), declared categorical | - | **A3** (imbalanced binary + high cardinality), C target encoding, S2a width cap |
| 2 | `c_credit` | clf binary | OpenML 42477 default-of-credit-card-clients (UCI 350), 20k of 30k, UCI names restored | 15000 x 27 | 5000 | 2 (yes 22 %) | integer-coded SEX / EDUCATION / MARRIAGE (declared categorical; EDUCATION has undocumented codes), PAY_* delay codes with −2 / −1 sentinels, heavy-tailed amounts, undeclared `id` | MCAR 8 % in 4 columns, duplicate `credit_limit_ntd`, constant `currency` | missing / duplicate / constant / ID handling, imbalance |
| 3 | `c_electricity` | clf binary | OpenML 151 electricity (Elec2), 20k random rows of 45312 | 15000 x 13 | 5000 | 2 (UP 42 %) | time-ordered market data, normalised date / period, `day` declared categorical, nonlinear | 4 noise features | irrelevant features, nonlinear |
| 4 | `c_letter` | clf 26-class | OpenML 6 letter (UCI), stratified 10k of 20k | 7500 x 17 | 2500 | 26 balanced (~290 train rows / class) | many classes with few rows per class, strongly nonlinear | - | A multiclass ET blend, B portfolio, C "many classes" rule (n/class ≥ 150 → not fired) |
| 5 | `c_cmc` | clf 3-class | OpenML 23 cmc (UCI contraceptive method choice) | 1105 x 10 | 368 | 3 (43 / 35 / 22 %) | **small n**, ordinal codes declared categorical, hard problem, target value `none` is a literal class | - | small-data behaviour (A4 / C rules), literal class labels |
| 6 | `c_nomao` | clf binary | OpenML 1486 nomao, stratified 25k of 34465 | 18750 x **119** | 6250 | 2 (different 28.6 %) | **wide table near the size limit**: 89 numeric similarity scores + 29 three-level categoricals | - | **B1** heavy size class / fallback, S2 width cap, 3-fold rules (A / C) |
| 7 | `r_metro` | reg | UCI 492 Metro Interstate Traffic Volume, 20k of 48204 hourly rows | 15000 x 9 | 5000 | - | **count** target 0..7280 (bimodal day / night, skew −0.1); `date_time` string is the main signal; `holiday` mostly literal `None` (read as NaN by pandas); 38-level weather_description; 0 K temperature errors | - | count target, datetime, transform choice must stay raw (no skew) |
| 8 | `r_socmob` | reg | OpenML 541 socmob (US social mobility) | 867 x 6 | 289 | - | **skewed count**-like target 0..414 (skew 3.75, 19 % zeros), small n, two 17-level categoricals | - | **B2 / C3** log1p choice where log helps, RMSLE safety |
| 9 | `r_taxi` | reg | OpenML 42729 nyc-taxi-green-dec-2016, 20k of 581835 | 15000 x 17 | 5000 | - | **skewed non-count** target tip_amount (dollars, skew 10.8, 15 % zeros); integer-coded PULocationID / DOLocationID (~240 levels), RatecodeID, VendorID, trip_type declared categorical; `pickup_datetime` | `total_amount` (contains the tip) replaced by `fare_before_tip` | **B2 / C3** (raw vs log on a heavy-tailed non-count), SH3 (stress without types) |
| 10 | `r_pol` | reg | OpenML 201 pol | 11250 x 49 | 3750 | - | skewed **bounded non-count** target 0..100 (63 % zeros, skew 0.93): raw HGB expected better than log1p; 48 numerics, 6+ constant in train | - | **B2 / C3** transform choice where raw should win, constant columns |
| 11 | `r_insurance` | reg | Kaggle insurance.csv (medical cost personal; GitHub mirror stedy/Machine-Learning-with-R-datasets) | 970 x 7 | 368 | - | skewed charges, near-linear age trend + smoker × bmi interaction, small n | **non-random split:** all 114 customers aged ≥ 60 in test (training ages 18–59) + 20 % random | **B3** out-of-range rows where the linear trend is right (offline: HGB under-predicts these rows by 2.3 k on average; ridge + interaction bias −1.5 k; 50/50 blend has the lowest RMSE on them) |
| 12 | `r_airfoil` | reg | OpenML 43919 airfoil_self_noise (NASA / UCI 291) | 1108 x 6 | 395 | - | strongly nonlinear in frequency and thickness | **non-random split:** all 86 rows with frequency ≥ 10 kHz in test (training max 8 kHz) + 20 % random | **B3** out-of-range rows where linear extrapolation is wrong (offline: ridge RMSE 7.9 vs HGB 5.3 on those rows) |
| 13 | `k_glass` | clust | OpenML 41 glass (UCI) | 214 x 9 | - | k = 6, **unequal** 76 / 70 / 29 / 17 / 13 / 9 | oxide percentages, Ba / Fe mostly zero, very small groups | - | **SH2 / A2 / C2** unequal group sizes |
| 14 | `k_banknote` | clust | OpenML 1462 banknote-authentication (UCI) | 1372 x 4 | - | k = 2 (762 / 610) | **elongated, curved** groups (scaled KMeans ARI 0.01) | - | **SH2** elongated groups (GMM-full / Ward vs KMeans) |
| 15 | `k_blobs_easy` | clust | synthetic `make_blobs` (seed 331) | 2500 x 6 | - | k = 5 equal | none: isotropic, equal variance, well separated (KMeans ARI 1.0) | - | **SH2** easy spherical case: the medoid must not pick a worse family |
| 16 | `k_optdigits` | clust | OpenML 28 optdigits (UCI), stratified 3000 of 5620 | 3000 x 64 | - | k = 10 | **64 moderately informative pixel features** that must NOT be dropped; constant border pixels (`input1`, `input40`) | - | **SH1** (no false drops), constant columns, timing of 4 candidates at p = 64 |
| 17 | `k_dermatology` | clust | OpenML 35 dermatology (UCI) | 366 x 39 | - | k = 6 (112 / 72 / 61 / 52 / 49 / 20) | 33 clinical 0..3 grades + Age (8 missing) | **5 planted noise features** | **SH1** (planted noise must be dropped), missing values |
| 18 | `k_moons` | clust | synthetic (seed 361) | 2000 x 6 | - | k = 4 (600 / 600 / 500 / 300) | - | two interleaving moons + one correlated elongated blob + one thin blob (**non-convex / elongated / unequal**), 4 noise features | **SH1 + SH2** (noise drop + family choice on non-convex groups) |

### Challenge coverage
| challenge | datasets |
|---|---|
| missing values | c_credit (inj), r_metro (holiday), k_dermatology (Age) |
| categoricals incl. high cardinality | c_amazon (all high-card), r_taxi (location codes), r_socmob, c_nomao (29), c_credit / c_cmc codes |
| different scales | c_credit (amounts vs codes), k_glass, k_dermatology + noise, k_optdigits |
| constant columns | c_credit (inj), r_pol, k_optdigits (native) |
| duplicate columns | c_credit (inj) |
| ID columns (undeclared) | c_credit `id` |
| irrelevant noise features | c_electricity, k_dermatology, k_moons (inj) |
| imbalance | c_amazon 5.8 %, c_credit 22 %, c_cmc 22 % smallest, c_nomao 29 % |
| nonlinear | c_letter, c_electricity, r_airfoil, r_metro, k_moons, k_banknote |
| count target | r_metro (raw), r_socmob (skewed) |
| skewed non-count target | r_taxi, r_pol, r_insurance |
| out-of-range test rows | r_insurance (linear trend right), r_airfoil (linear wrong) |
| datetime | r_metro, r_taxi |
| wide table near the size limit | c_nomao 18750 x 119 |
| small n | c_cmc 1105, r_socmob 867, r_insurance 970, k_glass 214, k_dermatology 366 |

## Round-3 stress probes (`validation/data/stress_r3/`, scored separately)

| probe | base | change |
|---|---|---|
| `s3_amazon_notypes` | c_amazon | **no `feature_types`**: 9 integer codes named `*_ID` / `*_CODE` / `ROLE_*` / `RESOURCE` (SH3) |
| `s3_taxi_notypes` | r_taxi | **no `feature_types`**: PULocationID / DOLocationID / RatecodeID / VendorID integer codes (SH3) |
| `s3_credit_missing` | c_credit | +25 % MCAR in every feature column |
| `s3_optdigits_traintest` | k_optdigits | clustering **variant 2** (train/test files) |
