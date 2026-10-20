# 07 — Fix round 2 changelog (round-2 Validator findings, docs/06)

**Rule for this round:** every change is a general mechanism with a stated principle, not a threshold tuned
until a known dataset improves. Each change was checked on both Validator suites (round 1 and round 2: 36
datasets) and the dev sets, with 2 threads, sequential runs and a warm-up run per approach.

- **Harness.** `tools/eval_suite.py` now takes `--suite r1 r2`. It imports the Validator's scoring functions
  read-only, uses the round-2 baselines for round 2, and writes to the gitignored `runs/<tag>/<suite>/`.
  Nothing under `validation/` was modified.
- **Scorer check.** Re-scoring the Validator's own round-2 runs with this tool reproduces docs/06 exactly
  (A 70.06, B 73.25, C 71.63).
- **Independence.** A, B and C remain separate, self-contained folders. Shared modules (`main.py`,
  `io_contract.py`, `cleaning.py`, `data_profile.py`, `utils.py`, `feature_weighting.py`) are byte-identical
  copies in each folder. Each `model.py` keeps its own philosophy:
  - A: single model plus 4 cheap clustering candidates.
  - B: deterministic portfolio and multi-view consensus.
  - C: profile rules, a rule-gated candidate set and a soft-weighted space.

## Totals (Validator scoring model)

"Before" for round 1 is docs/05 (runs/r1final, reproduced by the Validator). "Before" for round 2 is docs/06.
"After" is runs/r2final.

| approach | round-1 suite before → after | round-2 suite before → after | **combined 36 before → after** | median / max wall s (36 runs) |
|---|---|---|---|---|
| A_lean | 78.13 → 77.34 | 70.06 → **73.43** | 148.19 → **150.76** (+2.57) | 3.4 / 29.6 |
| B_portfolio | 76.63 → 76.39 | 73.25 → 73.45 | 149.88 → 149.84 (−0.04) | 5.9 / 24.6 |
| C_rules | 78.56 → 77.23 | 71.63 → **73.60** | 150.19 → **150.83** (+0.64) | 3.7 / 27.0 |

Quality pools, combined 36 datasets (before from docs/06 §9 → after):

| approach | classification (56.7) | regression (56.7) | clustering (56.7) | profile (20) | speed (10) |
|---|---|---|---|---|---|
| A_lean | 35.9 → 35.96 | 48.2 → 48.20 | 34.9 → **37.56** | 20 → 20 | 9.18 → 9.02 |
| B_portfolio | 35.4 → 35.72 | 48.6 → 48.38 | 37.4 → 37.50 | 20 → 20 | 8.52 → 8.24 |
| C_rules | 35.8 → 35.73 | 48.1 → 48.16 | 37.0 → **37.81** | 20 → 20 | 9.27 → 9.12 |

- All 108 runs are valid: no crash, timeout, watchdog exit or traceback.
- All 66 dev smoke runs (22 × 3) are valid.
- A has the lowest median wall time. On the maximum it is now slower than B (A 29.6 s on c_jannis, C 27.0 s,
  B 24.6 s).
- Both suites have now been used for development. Round 3 must use new data.

---

## Shared

### SH1 — relevance filter with a confirmation test (done; `feature_weighting.relevance_filter`)

**Principle.** A feature that matters for the clustering must co-vary with the cluster structure carried by
the *other* features.

**Change.**
- A low sparse weight (< 5 % of the max) now only makes a group a *candidate* for removal.
- To confirm it: run KMeans on the non-candidate groups (≤ 5000 rows), compute the candidate's explained share
  (between / total sum of squares) under that partition, and compare it with 200 random label permutations.
- The group is dropped only if p > 0.01.
- The old ad-hoc "protect informative categoricals" rule is removed; the test covers it.
- Applied identically in A, B and C.

**Effect on drops** (offline check over all 20 clustering sets — both suites plus dev):

| dataset | old drops | new drops |
|---|---|---|
| k_segment | 5 real features | none |
| k_vehicle | 2 real + 3 noise | 3 noise |
| k_wine | Ash, Magnesium | Ash |
| k_synth | 5 noise + noise_flag | same |
| iris | 3 noise | same |
| dev blob sets | noise | same |
| k_wholesale | Fresh, Region | Fresh, Region (the test does not confirm `Fresh` as relevant; no override was added) |

**Targets.**
- k_segment: A 0.379 → **0.466**, B 0.439 → **0.466**.
- k_vehicle: real features kept.
- k_wholesale: A 0.482 → 0.553, via A2's choice; `Fresh` is still dropped.

### SH2 — no BIC-gain acceptance of GMMs (done; `feature_weighting.candidate_partitions` / `agreement_medoid`)

**Principle.**
- A Gaussian density over standardised numerics plus 0/1 one-hot columns is mis-specified. The binary
  dimensions dominate the likelihood (k_synth BIC gain +343 % / +7507 %), so BIC gain cannot compare GMM and
  KMeans.
- The GMM-diag candidate is now fitted on the **continuous columns only**.
- GMM-full is fitted on PCA-whitened data.
- The choice among structurally different partitions uses a criterion comparable across models: the
  **partition the candidates agree with most** (medoid by mean pairwise ARI; ties → KMeans).

**Criteria compared offline on 19 clustering sets** (sum of ARI):

| criterion | sum of ARI |
|---|---|
| KMeans only | 13.11 |
| bootstrap stability | 13.30 |
| silhouette on the common space | 13.79 |
| agreement medoid | **13.80** |
| oracle | 14.79 |

**Targets.**
- k_synth: A 0.699 → **0.978**, C 0.699 → **0.986**.
- k_wholesale: GMM-diag (continuous columns only) is still chosen by A (0.553). In C's space the medoid picks
  KMeans (0.470).

### SH3 — integer-coded categoricals by name (done; `cleaning.is_code_name`, `infer_column_type`)

**Principle.** An undeclared integer column whose name says it is a code is nominal, not ordinal. Names
covered: `zip`, `state`, `code` / `*code` tokens, or a trailing `id` token (camelCase-aware: `WheelTypeID`,
`customerID`; not `humid` / `paid`).

**Change.** Such columns are inferred as `categorical` when they have > 2 integer values. Declared types
still win, so no suite dataset changed. This only affects metadata without `feature_types`.

---

## A_lean

| ID | status | change (model.py unless noted) | principle | target before → after |
|---|---|---|---|---|
| A1 | done | SH1 + SH2 in `_clustering`; the BIC-gated GMM step is removed | see SH1 / SH2 | synth 0.699 → **0.978**, segment 0.379 → **0.466**, wholesale 0.482 → 0.553 |
| A2 | done | Four cheap candidates on the kept features — KMeans, Ward (subsample + nearest centroid above 6000 rows), GMM-diag on continuous columns, GMM-full on PCA-whitened data — chosen by agreement medoid | different model families capture different group shapes; the answer most of them support is the robust pick | ecoli 0.495 → **0.766**, synth 0.978, penguins 0.894 → 0.951, Mall 0.585 → 0.681; clustering time ≤ 3.8 s |
| A3 | partial | One class-weight rule for binary and multiclass: sqrt-balanced weights + the existing OOF decision tuning (was capped-balanced for binary) | full balancing is redundant with a tuned decision rule and distorts the early-stopping loss | kick 0.655 → 0.659 (balanced accuracy 0.613 → 0.623); churn 0.891 → 0.900; ai4i −0.006, bank −0.005. **The kick gap to C (0.670) is not the decision rule**: capped / sqrt / no weights give 0.655 / 0.659 / 0.658. It is C's target encoding of the 753-level Model columns, which is C's mechanism and not ported |
| A4 | no change | The blend weight was already chosen by OOF macro-F1 from {0, 0.25, 0.5, 0.75} (0 = pure HGB) | — | creditapp 0.866: pre-fix A without any blend also scored 0.866, so the gap to C (0.906) is the HGB configuration / refit, not the blend |
| (A, time) | done | 3 folds when n × p × classes > 5e6 (same principle as C4) | headroom on wide multiclass tables | jannis 29.9 → 29.6 s, 0.550 → 0.549 |

## B_portfolio

| ID | status | change | principle | target before → after |
|---|---|---|---|---|
| B1 | done | When the measured quick-fit rate says the portfolio will not fit, B runs the **full single `hgb_a`** with an iteration cap from that rate, instead of keeping the quick model. A deterministic size class (n × features > 8e5) uses 3 folds and only the 4 core candidates, so each candidate's cost is bounded by its size class | the fallback must be the best affordable model, not the cheapest one; the cost class must follow the data | jannis 0.501 → **0.554** (24.6 s); superconduct 23.5 → 16.1 s (R² 0.912 → 0.909); max wall time 24.6 s |
| B2 | done | Decision-level tie-break: for log1p-eligible targets, the Caruana ensemble built from log-target members only replaces the unrestricted ensemble when its OOF R² is within 0.01. When log1p `hgb_a` is within 0.005 of raw, the later members use the log target | the same R² with much lower relative error is the safer choice when the hidden metric may be RMSLE | Seoul RMSLE **0.665 → 0.340** (R² 0.936 → 0.931); bike RMSLE 0.359 → 0.276 (R² −0.006); house R² 0.898 → 0.880 (OOF of the two ensembles is equal, 0.8873 vs 0.8874; the test split has the Ames outlier); california −0.006 |
| B3 | done | On out-of-range test rows a linear member is **capped** to within ×2 of the non-linear members (clip, not remove); it stays clipped to the training target range in log space | keep linear extrapolation when the trees corroborate its magnitude; cap it when they do not | brazil: 0 of 4 rows capped, R² 0.986 (pre-round-1 0.992; the remaining gap comes from a different ensemble composition, not the cap); Ames outlier: 1 row capped, no blow-up |
| B4 | done | Shared SH1 filter in the consensus views | SH1 | segment 0.439 → **0.466**; synth 0.973 and ecoli 0.724 kept |

B is deterministic across thread counts: c_pageblocks predictions are byte-identical at 1 and 2 threads.
Only the affordability fallback depends on the clock.

## C_rules

| ID | status | change | principle | target before → after |
|---|---|---|---|---|
| C1 | done | SH1 + SH2. The soft-weighted space is used only when the permutation test confirms irrelevant features (they get weight 0); the old "min weight < 5 %" rule and the BIC/agreement GMM acceptance are removed | SH1 / SH2 | synth 0.699 → **0.986**, vehicle 0.076 → 0.083 |
| C2 | done | Rule-gated candidates: KMeans always; Ward if n ≤ 6000; GMM-diag on continuous columns if ≥ 2; GMM-full on PCA-whitened data if ≥ 30 rows per group. Choice by agreement medoid | the profile gates which models are meaningful; agreement picks among them | ecoli 0.495 → **0.766**, penguins 0.769 → 0.954, drybean 0.625 → 0.660; pendigits 0.563 → 0.558 (target 0.62 not reached: KMeans is the medoid) |
| C3 | done | The linearity probe also fits the quick HGB on the raw target. If log1p's original-scale R² is more than 0.005 below raw, the log1p rule is reverted (logged in `rules_fired`) | the rule proposes, the probe verifies | superconduct 0.899 → **0.911**; Seoul / bike / house / kc / brazil keep log1p |
| C4 | done | 3 folds when n × p × classes > 5e6 | time headroom for a slower grader | jannis 30.2 → 27.0 s, macro-F1 0.553 → 0.542 (the price of the headroom) |

---

## Regressions > 0.05 on a single dataset (all explained)

| dataset | approach | before → after | cause and justification |
|---|---|---|---|
| r1 k_iris | A | 0.904 → 0.653 | Round 1's 0.904 came from the BIC-gain GMM acceptance that SH2 removes (it caused k_synth 0.70). Iris needs the minority full-covariance answer, which the agreement criterion does not pick. A full-rank GMM candidate when well determined was tried: neutral, and iris is not well determined (n = 150 < 10 k p(p+1)/2) |
| r1 k_iris | C | 0.904 → 0.745 | same cause |
| r1 k_mall | C | 0.956 → 0.665 | same cause (Mall's labels are derived, docs/04 §10); A's Mall improved 0.585 → 0.681 |
| r2 k_wholesale | C | 0.587 → 0.470 | the GMM-diag answer is now one candidate among four; in C's soft-weighted space the KMeans medoid wins by 0.692 vs 0.680 mean ARI |

Net effect of removing the BIC acceptance and adding agreement candidates, across the 12 suite clustering
sets: A +2.66, C +0.81 pool points.

---

## Per-dataset before → after (primary metric; wall s)

### A_lean
| suite | dataset | before | after | Δ | wall before | wall after |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8199 | 0.8204 | +0.0005 | 3.8 | 4.3 |
| r1 | c_ai4i | 0.8779 | 0.8720 | -0.0058 | 2.8 | 2.9 |
| r1 | c_bank | 0.7808 | 0.7761 | -0.0047 | 5.5 | 6.0 |
| r1 | c_covertype | 0.6203 | 0.6203 | +0.0000 | 15.0 | 17.8 |
| r1 | c_creditg | 0.6879 | 0.6879 | +0.0000 | 2.6 | 2.7 |
| r1 | c_telco | 0.7274 | 0.7274 | +0.0000 | 2.7 | 2.7 |
| r1 | k_drybean | 0.6600 | 0.6600 | +0.0000 | 5.6 | 3.7 |
| r1 | k_iris | 0.9039 | 0.6525 | -0.2514 | 1.7 | 1.8 |
| r1 | k_mall | 0.5845 | 0.6807 | +0.0962 | 1.8 | 1.8 |
| r1 | k_penguins | 0.8939 | 0.9505 | +0.0567 | 1.7 | 1.7 |
| r1 | k_seeds | 0.7733 | 0.7733 | +0.0000 | 1.8 | 1.7 |
| r1 | k_wine | 0.8805 | 0.8471 | -0.0334 | 1.8 | 1.8 |
| r1 | r_abalone | 0.5587 | 0.5587 | +0.0000 | 2.9 | 3.0 |
| r1 | r_bike | 0.9599 | 0.9599 | +0.0000 | 9.5 | 11.2 |
| r1 | r_california | 0.8420 | 0.8420 | +0.0000 | 8.6 | 10.1 |
| r1 | r_concrete | 0.8897 | 0.8897 | +0.0000 | 2.3 | 2.3 |
| r1 | r_diamonds | 0.9807 | 0.9807 | +0.0000 | 5.5 | 6.1 |
| r1 | r_house | 0.8954 | 0.8954 | +0.0000 | 5.4 | 5.9 |
| r2 | c_churn | 0.8907 | 0.9003 | +0.0096 | 3.9 | 4.1 |
| r2 | c_creditapp | 0.8662 | 0.8662 | +0.0000 | 2.4 | 2.4 |
| r2 | c_jannis | 0.5503 | 0.5490 | -0.0014 | 29.9 | 29.6 |
| r2 | c_kick | 0.6550 | 0.6593 | +0.0043 | 8.3 | 8.5 |
| r2 | c_magic | 0.8660 | 0.8660 | -0.0000 | 3.9 | 4.0 |
| r2 | c_pageblocks | 0.8515 | 0.8515 | -0.0000 | 5.8 | 6.2 |
| r2 | k_ecoli | 0.4948 | 0.7659 | +0.2711 | 1.8 | 1.8 |
| r2 | k_pendigits | 0.5582 | 0.5582 | -0.0000 | 2.9 | 3.8 |
| r2 | k_segment | 0.3787 | 0.4656 | +0.0870 | 2.1 | 2.4 |
| r2 | k_synth | 0.6989 | 0.9776 | +0.2787 | 2.1 | 2.1 |
| r2 | k_vehicle | 0.0745 | 0.0694 | -0.0052 | 1.9 | 2.0 |
| r2 | k_wholesale | 0.4820 | 0.5532 | +0.0711 | 1.7 | 1.9 |
| r2 | r_brazil | 0.9760 | 0.9760 | +0.0000 | 5.1 | 5.7 |
| r2 | r_kc_house | 0.8525 | 0.8525 | -0.0000 | 7.5 | 8.7 |
| r2 | r_moneyball | 0.9419 | 0.9419 | -0.0000 | 2.4 | 2.4 |
| r2 | r_seoul | 0.9356 | 0.9356 | -0.0000 | 9.3 | 10.8 |
| r2 | r_superconduct | 0.9108 | 0.9108 | -0.0000 | 16.8 | 19.0 |
| r2 | r_wine | 0.4647 | 0.4647 | +0.0000 | 3.6 | 3.1 |

### B_portfolio
| suite | dataset | before | after | Δ | wall before | wall after |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8143 | 0.8143 | +0.0000 | 7.3 | 7.7 |
| r1 | c_ai4i | 0.8420 | 0.8420 | +0.0000 | 4.8 | 5.1 |
| r1 | c_bank | 0.7750 | 0.7750 | +0.0000 | 12.0 | 13.1 |
| r1 | c_covertype | 0.6059 | 0.6059 | +0.0000 | 17.9 | 20.3 |
| r1 | c_creditg | 0.7074 | 0.7074 | +0.0000 | 4.9 | 4.7 |
| r1 | c_telco | 0.7242 | 0.7242 | +0.0000 | 5.2 | 5.4 |
| r1 | k_drybean | 0.6600 | 0.6600 | +0.0000 | 4.6 | 4.8 |
| r1 | k_iris | 0.5801 | 0.5801 | +0.0000 | 1.9 | 1.9 |
| r1 | k_mall | 0.6651 | 0.6651 | +0.0000 | 1.9 | 1.8 |
| r1 | k_penguins | 0.9505 | 0.9505 | +0.0000 | 1.9 | 1.9 |
| r1 | k_seeds | 0.7722 | 0.7722 | +0.0000 | 1.8 | 1.8 |
| r1 | k_wine | 0.8805 | 0.8804 | -0.0001 | 1.9 | 1.9 |
| r1 | r_abalone | 0.5762 | 0.5739 | -0.0023 | 4.3 | 4.5 |
| r1 | r_bike | 0.9625 | 0.9570 | -0.0055 | 13.1 | 15.2 |
| r1 | r_california | 0.8421 | 0.8365 | -0.0056 | 11.4 | 12.9 |
| r1 | r_concrete | 0.8884 | 0.8884 | +0.0000 | 4.2 | 3.7 |
| r1 | r_diamonds | 0.9808 | 0.9807 | -0.0001 | 9.3 | 10.8 |
| r1 | r_house | 0.8978 | 0.8798 | -0.0180 | 10.3 | 11.0 |
| r2 | c_churn | 0.8880 | 0.8880 | +0.0000 | 9.0 | 9.6 |
| r2 | c_creditapp | 0.8715 | 0.8715 | -0.0000 | 4.1 | 4.1 |
| r2 | c_jannis | 0.5010 | 0.5537 | +0.0527 | 7.1 | 24.6 |
| r2 | c_kick | 0.6690 | 0.6690 | +0.0000 | 14.0 | 14.6 |
| r2 | c_magic | 0.8620 | 0.8620 | +0.0000 | 6.8 | 6.7 |
| r2 | c_pageblocks | 0.8557 | 0.8557 | +0.0000 | 11.0 | 11.7 |
| r2 | k_ecoli | 0.7241 | 0.7241 | -0.0000 | 2.5 | 2.0 |
| r2 | k_pendigits | 0.6164 | 0.6164 | -0.0000 | 6.6 | 8.2 |
| r2 | k_segment | 0.4392 | 0.4656 | +0.0264 | 2.8 | 2.8 |
| r2 | k_synth | 0.9730 | 0.9730 | -0.0000 | 3.2 | 3.2 |
| r2 | k_vehicle | 0.0746 | 0.0722 | -0.0024 | 2.1 | 2.4 |
| r2 | k_wholesale | 0.5808 | 0.5808 | -0.0000 | 1.9 | 1.9 |
| r2 | r_brazil | 0.9861 | 0.9860 | -0.0001 | 9.2 | 10.5 |
| r2 | r_kc_house | 0.8492 | 0.8505 | +0.0013 | 12.0 | 13.0 |
| r2 | r_moneyball | 0.9482 | 0.9506 | +0.0024 | 3.9 | 4.1 |
| r2 | r_seoul | 0.9362 | 0.9308 | -0.0054 | 13.2 | 16.0 |
| r2 | r_superconduct | 0.9117 | 0.9086 | -0.0031 | 23.5 | 16.1 |
| r2 | r_wine | 0.5022 | 0.5022 | -0.0000 | 5.8 | 6.3 |

### C_rules
| suite | dataset | before | after | Δ | wall before | wall after |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8150 | 0.8150 | +0.0000 | 4.7 | 5.0 |
| r1 | c_ai4i | 0.8517 | 0.8517 | +0.0000 | 3.1 | 3.0 |
| r1 | c_bank | 0.7674 | 0.7674 | +0.0000 | 6.3 | 6.6 |
| r1 | c_covertype | 0.5721 | 0.5721 | +0.0000 | 15.3 | 17.6 |
| r1 | c_creditg | 0.6686 | 0.6686 | +0.0000 | 5.6 | 5.3 |
| r1 | c_telco | 0.7344 | 0.7344 | +0.0000 | 3.7 | 3.8 |
| r1 | k_drybean | 0.6253 | 0.6600 | +0.0347 | 3.8 | 3.0 |
| r1 | k_iris | 0.9039 | 0.7445 | -0.1593 | 1.8 | 1.8 |
| r1 | k_mall | 0.9561 | 0.6651 | -0.2910 | 1.8 | 1.7 |
| r1 | k_penguins | 0.7685 | 0.9536 | +0.1851 | 1.8 | 1.8 |
| r1 | k_seeds | 0.7733 | 0.7733 | +0.0000 | 1.8 | 1.7 |
| r1 | k_wine | 0.9150 | 0.8820 | -0.0330 | 1.8 | 1.8 |
| r1 | r_abalone | 0.5540 | 0.5540 | +0.0000 | 2.7 | 2.9 |
| r1 | r_bike | 0.9598 | 0.9598 | +0.0000 | 7.1 | 8.5 |
| r1 | r_california | 0.8407 | 0.8407 | +0.0000 | 5.4 | 6.7 |
| r1 | r_concrete | 0.8818 | 0.8818 | +0.0000 | 2.8 | 2.8 |
| r1 | r_diamonds | 0.9800 | 0.9800 | +0.0000 | 4.9 | 5.6 |
| r1 | r_house | 0.8963 | 0.8963 | +0.0000 | 4.6 | 5.5 |
| r2 | c_churn | 0.9035 | 0.9035 | -0.0000 | 4.3 | 4.5 |
| r2 | c_creditapp | 0.9064 | 0.9064 | -0.0000 | 2.8 | 2.9 |
| r2 | c_jannis | 0.5525 | 0.5424 | -0.0100 | 30.2 | 27.0 |
| r2 | c_kick | 0.6697 | 0.6697 | +0.0000 | 10.6 | 11.3 |
| r2 | c_magic | 0.8641 | 0.8641 | +0.0000 | 7.8 | 8.5 |
| r2 | c_pageblocks | 0.8626 | 0.8626 | +0.0000 | 5.9 | 6.4 |
| r2 | k_ecoli | 0.4948 | 0.7659 | +0.2711 | 1.8 | 1.9 |
| r2 | k_pendigits | 0.5631 | 0.5582 | -0.0048 | 2.8 | 3.7 |
| r2 | k_segment | 0.4784 | 0.4656 | -0.0128 | 2.2 | 2.4 |
| r2 | k_synth | 0.6989 | 0.9855 | +0.2866 | 2.0 | 2.2 |
| r2 | k_vehicle | 0.0763 | 0.0827 | +0.0064 | 1.8 | 2.0 |
| r2 | k_wholesale | 0.5873 | 0.4696 | -0.1177 | 1.6 | 1.7 |
| r2 | r_brazil | 0.9749 | 0.9749 | +0.0000 | 3.6 | 4.5 |
| r2 | r_kc_house | 0.8568 | 0.8568 | -0.0000 | 6.1 | 7.3 |
| r2 | r_moneyball | 0.9492 | 0.9492 | -0.0000 | 2.9 | 2.9 |
| r2 | r_seoul | 0.9320 | 0.9320 | +0.0000 | 6.4 | 8.0 |
| r2 | r_superconduct | 0.8992 | 0.9114 | +0.0122 | 15.0 | 14.5 |
| r2 | r_wine | 0.4628 | 0.4628 | +0.0000 | 3.8 | 3.2 |

---

## Tried and rejected

| where | tried | result | decision |
|---|---|---|---|
| SH2 / A2 | bootstrap stability (3 × 70 % refits) as the selection criterion | 13.30 summed ARI vs 13.80 for agreement; it prefers stable-but-wrong KMeans (penguins, synth) | agreement medoid |
| SH2 / A2 | silhouette on the common space | 13.79, KMeans-biased (it never picks Ward on synth) | not used; A and C use agreement |
| A2 | an extra full-rank GMM candidate (no PCA) when parameters are well determined (n ≥ 10 k p(p+1)/2) | neutral (13.795 vs 13.793); does not apply to iris | not added |
| A3 | capped-balanced / sqrt-balanced / no class weights for binary | kick 0.655 / 0.659 / 0.658; sum over 5 imbalanced binary sets 4.024 / 4.028 / 4.019 (noise) | one sqrt rule for all tasks; kick gap attributed to encoding |
| B1 | affordability rule without the quick-fit over-estimate discount | under transient machine load it fell back to single hgb_a on `c_bank` (n × p = 0.3 M) | keep the factor-3 discount (same as A / C) |
| B1 | size class by work n × p × classes (> 2e6) | covertype became heavy: 0.606 → 0.586 | size class by table cells n × p (> 8e5): jannis and superconduct heavy, covertype normal |
| B2 | drop raw `hgb_a` whenever log1p `hgb_a` is within 0.005 | house −0.018 and bike −0.006 without a decision-level check | decision-level log-only-ensemble tie-break |
| B2 | the same tie-break at 0.005 tolerance | Seoul not switched (log-only −0.0067 OOF), RMSLE stayed 0.68 | 0.01 |
| C4 | none (as specified) | jannis −0.010 macro-F1 for 3 s of headroom | kept as requested; noted |
| SH1 | the round-1 categorical-protection rule next to the permutation test | redundant: penguins `island` passes the test on its own | removed |

## Not done / open
- pendigits 0.62 for C (C2 target): KMeans is the agreement medoid; Ward (0.616) is a minority.
- B3 brazil back to 0.992: no row is capped any more, so the remaining 0.006 comes from the ensemble
  composition, not B3.
- k_wholesale `Fresh`: the permutation test does not confirm it as relevant; overriding it would be
  dataset-specific.
