# Validation suite — round 4 (FINAL; 18 new datasets + 1 stress probe)

Built by `py validation/build_suite_round4.py` (fixed seeds; about 5 s once downloads are cached). Output goes to
`validation/data/suite_r4/` (gitignored).

Format is as in rounds 1–3:
- docs/02 folders, ground truth in `hidden/`;
- declared `feature_types` (numeric dtype → numeric, otherwise categorical; integer codes declared categorical);
- undeclared ID columns stay undeclared;
- 75/25 split, stratified for classification;
- clustering uses variant 1 (`data.csv`), except `k_satimage`, which uses variant 2 (train/test).

**Overlap check: none.** No dataset here appears in rounds 1–3 (54 datasets), in any stress probe, or in docs/03.
- `c_bankchurn` is Kaggle *bank* churn. It is not Telco (round 1) or the telecom churn set (round 2).
- `c_yeast` and `k_vowel` were never used before.

**Representativeness.** The mix follows the hidden-set families rather than edge cases:
- a Telco-like churn table (bank churn);
- a House-Prices-like skewed price (house_16H) and a skewed money target with high-cardinality categoricals
  (Medicare charges);
- small tables (auto-mpg 298 train rows, zoo 101, user-knowledge 403);
- the edge cases requested: 2 many-class problems (texture 11, yeast 10), a heavy wide table (MiniBooNE
  22500 × 80) and the clustering train/test variant (satimage).

| # | name | task | source | train x cols | test | classes / k | native | injected | probes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `c_bankchurn` | clf binary | Kaggle Churn_Modelling.csv (GitHub mirror selva86/datasets) | 7500 x 14 | 2500 | 2 (exited 20.4 %) | `RowNumber`, `CustomerId` IDs (undeclared), `Surname` ~2900-level categorical, Balance 36 % zeros | - | Telco-like churn, ID / high-cardinality handling |
| 2 | `c_spambase` | clf binary | OpenML 44 spambase | 3451 x 60 | 1150 | 2 (spam 39 %) | 57 heavy-tailed frequency features | constant `source`, duplicate `caps_total_copy` | constant / duplicate |
| 3 | `c_ozone` | clf binary | OpenML 1487 ozone-level-8hr | 1900 x 73 | 634 | 2 (ozone **6.3 %**) | 72 correlated weather features, small minority | MCAR 6 % in 6 columns | imbalance, missing |
| 4 | `c_texture` | clf **11-class** | OpenML 40499 texture | 4125 x 41 | 1375 | 11 balanced | 40 numerics | - | **M1** many-class |
| 5 | `c_yeast` | clf **10-class** | OpenML 181 yeast | 1113 x 12 | 371 | 10, **rare** (CYT 31 % … ERL 0.4 %) | small n, rare classes | 3 noise features | **M1** + rare classes |
| 6 | `c_miniboone` | clf binary | OpenML 41150 MiniBooNE, stratified 30k of 130k | 22500 x **81** | 7500 | 2 (signal 28 %) | −999 sentinels | 30 noise features | **heavy wide table** (1.8 M training cells): M1 fold planner, B heavy size class |
| 7 | `r_medical` | reg | OpenML 42131 medical_charges (CMS Medicare), 20k of 163k | 15000 x 11 | 5000 | - | skewed money target (skew 3.4); **high-cardinality** drg (100), provider_name (~2800), city (~1700), hrr (306), state, zip / provider_id codes | average_total_payments dropped (contains the target) | high-cardinality encoding, C target encoding, log target |
| 8 | `r_house16h` | reg | OpenML 574 house_16H | 17088 x 17 | 5696 | - | skewed capped price (skew 3.8) | - | House-Prices-like |
| 9 | `r_cpuact` | reg | OpenML 197 cpu_act | 6144 x 25 | 2048 | - | left-skewed bounded % target, heavy-tailed counters | 3 noise | transform choice (raw should win) |
| 10 | `r_elevators` | reg | OpenML 216 elevators | 12449 x 19 | 4150 | - | small-scale target, near-linear | - | linear vs tree |
| 11 | `r_autompg` | reg | OpenML 196 autoMpg | 298 x 9 | 100 | - | very small n, 5 missing horsepower in train, origin categorical | undeclared string ID `car_id` | small-data blends |
| 12 | `r_sulfur` | reg | OpenML 23515 sulfur | 7561 x 6 | 2520 | - | spiky skewed target (skew 6.4), 5 sensors | second output y2 dropped | heavy-tail target |
| 13 | `k_zoo` | clust | OpenML 62 zoo | 101 x 17 | - | k = 7 unequal (41 … 4) | 15 boolean attributes, unique `animal` name (ID-like) | - | categorical clustering, tiny n |
| 14 | `k_userknow` | clust | OpenML 1508 user-knowledge | 403 x 5 | - | k = 5 (129 / 122 / 102 / 26 / 24) | segmentation-like small table | - | Mall-like segmentation |
| 15 | `k_satimage` | clust | OpenML 182 satimage, stratified 4000 | 3000 + 1000 x 36 | (test 1000) | k = 6 unequal | spectral bands | - | **M2** clustering train/test variant |
| 16 | `k_mfeat` | clust | OpenML 16 mfeat-karhunen | 2000 x 64 | - | k = 10 | 64 KL coefficients | - | many groups |
| 17 | `k_vowel` | clust | OpenML 307 vowel | 990 x 12 | - | k = 11 | irrelevant Speaker_Number (15 levels) / Sex categoricals | - | many groups + irrelevant categoricals |
| 18 | `k_wallrobot` | clust | OpenML 1497 wall-robot-navigation | 5456 x 27 | - | k = 4 unequal | 24 sensor distances | 3 noise | larger n, noise |

### Stress probe

| probe | base | change |
|---|---|---|
| `s4_bankchurn_missing` | c_bankchurn | +20 % MCAR in every feature column |

The many-class, wide and train/test probes are inside the scored suite.
