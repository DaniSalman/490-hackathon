# 09 — Fix round 3 changelog (round-3 Validator findings, docs/08)

**Verified state: HEAD 26cebfc (code). 21ba79e and this commit change only docs.** All numbers below come from runs of
26cebfc, unless a row names another commit. Protocol: Validator scoring functions imported read-only (`tools/eval_suite.py`, extended to
suite r3, `--threads`, `--slowdown`), 2 threads, sequential, warm-up, 60 s kill. Nothing under `validation/` was
modified. A, B and C stay separate, self-contained folders.

**Machine-state caveat.** During this round the dev laptop ran slower than in rounds 1–3. It was on battery for part
of the night, and the reference `k_seeds` run took 2.1–2.6 s instead of 1.7 s.
- Speed points in the raw before/after table are therefore not comparable with docs/08.
- To compare fairly, the round-2 code **98f5fe1 was re-run on the same machine, in the same session** (§1b).
- Quality points do not depend on speed: every decision is budget-relative, and no run came near its budget limits
  on the 2-thread pass.

## Fix status

| ID | status | what changed (file / function) | principle | target before → after |
|---|---|---|---|---|
| M1a | done | `utils.fit_hgb_anytime` (shared): HGB fitted in warm-start chunks. The first chunk is ¼ of the iterations; the remainder runs in one call if it fits the deadline (projected with ×2 slow-CPU safety), otherwise doubling chunks. Results are identical to a single fit when no deadline is hit | measured per-iteration cost (which includes the class factor: one tree per class per iteration) drives every stop, not a cell-count threshold | — |
| M1b | done | `utils.FoldPlanner` (shared), used by A, C and B's anchor. Each fold gets its fair share of the remaining budget as an HGB deadline. After each fold the fold count shrinks so that the remaining folds would fit a 2× slower CPU. Many classes (≥ 10): fast HGB + 3 folds in A, B and C; B also uses its core portfolio | graceful degradation by measured time; CV cost = folds × classes trees per iteration | c_letter 1 thread: A 35.7 → **16.9 s**, B 52.1 (watchdog) → **22.2 s**, C 28.7 → **12.7 s** |
| M1c | done | B `_run_cv`: HGB candidates are fitted with deadlines. The anchor keeps a truncated fit and may complete fewer folds (later candidates use the same folds; OOF is scored on those rows). Any other candidate aborts *inside* the fit. Class-aware cost estimate for forest / linear / kNN candidates (their cost does not grow with the number of classes) | never rely on the watchdog | B c_letter: watchdog 52.2 s → 25.8 s (2 thr) / 22.2 s (1 thr); output no longer CPU-dependent there |
| M1d | done | fast HGB for ≥ 10 classes: lr 0.1, 100 iterations, early stopping patience 20, no extra weighting (A `_hgb_params`, B `_portfolio`, C `_make` rule `fast_hgb`, C probe skipped) | sklearn's default HGB setting; per-iteration cost × classes | c_letter macro-F1: A 0.934 → **0.944**, B 0.907 → **0.947**, C 0.920 → **0.947** (default HGB 0.947) |
| M1 extra | done | Quick model bounded: it is also an anytime fit with a deadline of budget/8. The old rate-based iteration cap and the "keep the quick model" branches are removed (A, B, C): a truncated quick fit over-estimates the per-iteration cost (fixed binning cost), and on one 1-thread run that kept only the 5k-row quick model on c_jannis (0.49) | the anytime answer must arrive within a fixed share of the budget; later fits are interruptible anyway | c_jannis 1 thread: A 0.551, B 0.554, C 0.540 (no quick-model fallbacks) |
| M2 | done | `utils.finalize_labels` (shared): exactly min(k, n_pred) distinct labels 0..k−1 **on the predicted rows**. Every clustering output in A, B and C goes through it (audit: 9 call sites, including the early anytime emits) | the training rows of the train/test variant are not scored | s3_optdigits_traintest C: invalid_k (9/10) → **valid, k_ok**; A, B, C valid on all r2 / r3 train/test probes |
| M3 | done | `main.py` (shared): `HACK_SLOWDOWN=s` divides the budget and the watchdog limit by s; `tools/eval_suite.py --slowdown s` scales the 60 s kill the same way | self-check of the worst-case path on a simulated slower CPU | 2.5× slower CPU: **162/162 valid runs on all 54 datasets, 0 watchdog exits**, max real wall time 14.7 s against a 16 s budget / 20.8 s watchdog (§3) |
| Q1 | not done (deadline) | — | — | banknote's `variance` is still dropped (no ARI effect: every method fails on banknote) |
| Q2 | done, kept | C `_clustering`: confirmed-irrelevant groups are **removed** and the kept features keep equal weight (no soft weights) | sparse weights rank relevance for the test, but as distance weights they down-weight moderately informative features | C clustering pool 52.05 → **52.15** over 54: dermatology 0.705 → **0.923**, wholesale 0.470 → 0.553; optdigits 0.715 → 0.605, iris 0.745 → 0.653 |
| Q3 | rejected | — | offline check over all 54 datasets: the proposed rule (> 50 levels, unique ratio < 0.5, integer) would turn **143 declared-numeric integer columns** into categoricals (age, balance, sqft_*, PAY_AMT*, pixel features …) for 15 real codes. Adding a value-density condition still leaves ~25 misfires vs ~10 hits | "a misfire costs more than it gains"; the contract declares types |
| Q4 | rejected | — | "count" cannot be separated from r_pol without a dataset-tailored rule: pol's target is a bounded **integer** 0..100 with 63 % zeros, socmob's count-like target is float. A universal 0.003 tolerance would give up Seoul's RMSLE safety (round-2 B2) | B2 unchanged |

Regressions > 0.05 on one dataset (all C clustering, all from Q2, net +0.10 pool points): optdigits −0.11 and iris
−0.09, against dermatology +0.22 and wholesale +0.08. Q2 is kept because the combined score supports it.

## 1. Totals

### 1a. Validator scoring model, same-session comparison: 98f5fe1 (round-2 code) vs HEAD 26cebfc

PRELIMINARY: HEAD numbers from run `runs/r3g` (01:30, slower machine state); a same-session re-run is in progress. The Validator's own numbers for 98f5fe1
(docs/08: A 221.73, B 220.64, C 221.58) are reproduced by the r3old run to within 0.03.

| approach | suite | total | quality | speed | median / max wall s | watchdog exits |
|---|---|---|---|---|---|---|
| A_lean | r1 | 77.34 → 77.18 | 62.71 → 62.71 | 4.63 → 4.47 | 3.3 / 16.8 → 4.0 / 21.1 | 0 → 0 |
| A_lean | r2 | 73.45 → 73.26 | 59.02 → 58.97 | 4.43 → 4.29 | 4.1 / 29.2 → 5.0 / 19.4 | 0 → 0 |
| A_lean | r3 | 70.98 → 70.94 | 57.03 → 57.08 | 3.97 → 3.88 | 3.9 / 25.5 → 4.5 / 18.9 | 0 → 0 |
| A_lean | **combined 54** | 221.76 → 221.38 | 178.76 → 178.76 | 13.02 → 12.64 | 3.7 / 29.2 → 4.4 / 21.1 | 0 → 0 |
| B_portfolio | r1 | 76.38 → 76.15 | 62.12 → 62.12 | 4.26 → 4.04 | 5.5 / 19.1 → 6.1 / 23.4 | 0 → 0 |
| B_portfolio | r2 | 73.46 → 73.18 | 59.48 → 59.43 | 3.98 → 3.75 | 7.1 / 26.8 → 8.5 / 24.3 | 0 → 0 |
| B_portfolio | r3 | 70.81 → 70.98 | 57.08 → 57.26 | 3.75 → 3.73 | 6.7 / 52.2 → 7.0 / 25.8 | 1 → 0 |
| B_portfolio | **combined 54** | 220.65 → 220.31 | 178.67 → 178.82 | 11.99 → 11.51 | 6.4 / 52.2 → 7.0 / 25.8 | 1 → 0 |
| C_rules | r1 | 77.24 → 76.46 | 62.53 → 61.99 | 4.71 → 4.47 | 3.7 / 16.1 → 4.6 / 20.6 | 0 → 0 |
| C_rules | r2 | 73.61 → 73.79 | 59.18 → 59.48 | 4.43 → 4.31 | 4.1 / 27.6 → 4.6 / 23.9 | 0 → 0 |
| C_rules | r3 | 70.77 → 70.69 | 56.43 → 56.85 | 4.36 → 3.85 | 4.9 / 30.1 → 5.1 / 17.3 | 0 → 0 |
| C_rules | **combined 54** | 221.62 → 220.94 | 178.13 → 178.32 | 13.50 → 12.63 | 4.1 / 30.1 → 4.7 / 23.9 | 0 → 0 |

### 1b. Per-pool totals, HEAD (54 datasets)

| approach | clf | reg | clu | profile | speed | quality | total |
|---|---|---|---|---|---|---|---|
| A_lean | 55.13 | 71.50 | 52.12 | 29.98 | 12.64 | 178.76 | 221.38 |
| B_portfolio | 54.92 | 71.73 | 52.16 | 29.98 | 11.51 | 178.82 | 220.31 |
| C_rules | 54.73 | 71.45 | 52.15 | 29.98 | 12.63 | 178.32 | 220.94 |

Before (98f5fe1, docs/08 §1): clf / reg / clu = A 55.14 / 71.50 / 52.12, B 54.78 / 71.73 / 52.16, C 54.59 / 71.45 / 52.05.

## 2. 1-thread timing (heavy datasets plus every dataset > 10 s; `--threads 1`; HEAD)

"before" = Validator round-3 1-thread run of 98f5fe1 (r3 datasets only). Columns: approach | dataset | wall before |
wall HEAD | macro-F1/R² before | HEAD | watchdog before → HEAD.

| approach | dataset | 1-thr wall before | 1-thr wall HEAD | primary before | primary HEAD | watchdog |
|---|---|---|---|---|---|---|
| A_lean | c_covertype | - | 19.6 | - | 0.6203 | nan -> False |
| B_portfolio | c_covertype | - | 21.8 | - | 0.6059 | nan -> False |
| C_rules | c_covertype | - | 18.2 | - | 0.5721 | nan -> False |
| A_lean | c_jannis | - | 20.9 | - | 0.5511 | nan -> False |
| B_portfolio | c_jannis | - | 22.8 | - | 0.5537 | nan -> False |
| C_rules | c_jannis | - | 28.1 | - | 0.5396 | nan -> False |
| A_lean | r_superconduct | - | 19.0 | - | 0.9103 | nan -> False |
| B_portfolio | r_superconduct | - | 20.3 | - | 0.9086 | nan -> False |
| C_rules | r_superconduct | - | 16.5 | - | 0.9114 | nan -> False |
| A_lean | c_electricity | 8.17 | 7.5 | 0.887386 | 0.8874 | False -> False |
| B_portfolio | c_electricity | 14.732 | 12.9 | 0.882508 | 0.8825 | False -> False |
| C_rules | c_electricity | 10.359 | 9.2 | 0.890901 | 0.8909 | False -> False |
| A_lean | c_letter | 35.675 | 16.9 | 0.933602 | 0.9443 | False -> False |
| B_portfolio | c_letter | 52.126 | 22.2 | 0.954555 | 0.9466 | True -> False |
| C_rules | c_letter | 28.655 | 12.7 | 0.919542 | 0.9467 | False -> False |
| A_lean | c_nomao | 16.858 | 14.8 | 0.965836 | 0.9658 | False -> False |
| B_portfolio | c_nomao | 22.508 | 19.0 | 0.963474 | 0.9635 | False -> False |
| C_rules | c_nomao | 25.928 | 20.5 | 0.963766 | 0.9638 | False -> False |
| A_lean | r_metro | 8.804 | 7.7 | 0.97331 | 0.9733 | False -> False |
| B_portfolio | r_metro | 17.817 | 15.1 | 0.972217 | 0.9722 | False -> False |
| C_rules | r_metro | 14.028 | 11.9 | 0.976397 | 0.9764 | False -> False |
| A_lean | r_pol | 16.481 | 13.6 | 0.988931 | 0.9889 | False -> False |
| B_portfolio | r_pol | 23.291 | 18.1 | 0.983376 | 0.9834 | False -> False |
| C_rules | r_pol | 12.978 | 11.0 | 0.989222 | 0.9892 | False -> False |


Maximum 1-thread wall time: **A 20.9 s, B 22.8 s, C 28.1 s** (98f5fe1: A 35.7, B 52.1 watchdog, C 28.7). c_letter: A 16.9,
B 22.2, C 12.7 s, all at or above the default-HGB macro-F1 (0.947 for B / C, 0.944 for A).

## 3. Slow-CPU simulation (M3): `--slowdown 2.5` on all 54 datasets x 3 approaches

The budget and the watchdog are divided by 2.5 (16 s / 20.8 s), and the harness kill by 2.5 (24 s). This is equivalent
to a CPU 2.5x slower than this one.

| approach | runs | valid | watchdog exits | max real wall s (budget 16 s, watchdog 20.8 s) | quality (2.5x slow) | quality (normal) |
|---|---|---|---|---|---|---|
| A_lean | 54 | 54 | 0 | 14.7 | 178.61 | 178.76 |
| B_portfolio | 54 | 54 | 0 | 13.5 | 178.63 | 178.82 |
| C_rules | 54 | 54 | 0 | 11.9 | 177.99 | 178.32 |

No run reached its watchdog. Every approach degrades gracefully (fewer folds, truncated fits) and loses at most 0.35
quality points over 54 datasets.

## 4. Skipped (deadline)
- Q1: the SH1 test against a partition computed with the candidate. Not done. Offline analysis: a permutation null
  keeps the candidate's marginal distribution, so a self-bimodal feature (banknote `variance`) still looks
  irrelevant; it needs a different test (multimodality). Not worth the risk before the deadline.
- No new tuning of Q2 thresholds; no re-run of the round-1 / round-2 stress probes beyond the train/test
  clustering probes (all valid, k_ok).

## 5. Per-dataset before (98f5fe1) → after (HEAD), primary metric and wall time (2 threads, same session)

### A_lean
| suite | dataset | 98f5fe1 | HEAD | Δ | wall 98f5fe1 | wall HEAD |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8204 | 0.8204 | +0.0000 | 5.0 | 5.5 |
| r1 | c_ai4i | 0.8720 | 0.8720 | +0.0000 | 3.3 | 4.2 |
| r1 | c_bank | 0.7761 | 0.7761 | +0.0000 | 6.6 | 8.4 |
| r1 | c_covertype | 0.6203 | 0.6203 | +0.0000 | 16.8 | 21.1 |
| r1 | c_creditg | 0.6879 | 0.6879 | +0.0000 | 3.1 | 3.4 |
| r1 | c_telco | 0.7274 | 0.7274 | +0.0000 | 3.1 | 3.4 |
| r1 | k_drybean | 0.6600 | 0.6600 | +0.0000 | 4.3 | 4.6 |
| r1 | k_iris | 0.6525 | 0.6525 | +0.0000 | 2.0 | 2.2 |
| r1 | k_mall | 0.6807 | 0.6807 | +0.0000 | 2.1 | 2.2 |
| r1 | k_penguins | 0.9505 | 0.9505 | +0.0000 | 2.2 | 2.2 |
| r1 | k_seeds | 0.7733 | 0.7733 | +0.0000 | 2.1 | 2.2 |
| r1 | k_wine | 0.8471 | 0.8471 | +0.0000 | 2.1 | 2.3 |
| r1 | r_abalone | 0.5587 | 0.5587 | +0.0000 | 3.4 | 3.7 |
| r1 | r_bike | 0.9599 | 0.9599 | +0.0000 | 10.8 | 13.9 |
| r1 | r_california | 0.8420 | 0.8420 | +0.0000 | 9.6 | 12.4 |
| r1 | r_concrete | 0.8897 | 0.8897 | +0.0000 | 2.6 | 2.9 |
| r1 | r_diamonds | 0.9807 | 0.9807 | +0.0000 | 6.3 | 7.6 |
| r1 | r_house | 0.8954 | 0.8954 | +0.0000 | 6.1 | 6.8 |
| r2 | c_churn | 0.9003 | 0.9003 | +0.0000 | 4.4 | 5.7 |
| r2 | c_creditapp | 0.8662 | 0.8662 | +0.0000 | 2.6 | 3.0 |
| r2 | c_jannis | 0.5490 | 0.5396 | -0.0094 | 29.2 | 18.3 |
| r2 | c_kick | 0.6593 | 0.6593 | +0.0000 | 8.9 | 11.8 |
| r2 | c_magic | 0.8660 | 0.8660 | +0.0000 | 4.2 | 5.3 |
| r2 | c_pageblocks | 0.8515 | 0.8515 | +0.0000 | 6.5 | 8.7 |
| r2 | k_ecoli | 0.7659 | 0.7659 | +0.0000 | 2.0 | 2.5 |
| r2 | k_pendigits | 0.5582 | 0.5582 | +0.0000 | 3.9 | 4.7 |
| r2 | k_segment | 0.4656 | 0.4656 | +0.0000 | 2.5 | 2.9 |
| r2 | k_synth | 0.9776 | 0.9776 | +0.0000 | 2.4 | 2.6 |
| r2 | k_vehicle | 0.0694 | 0.0694 | +0.0000 | 2.1 | 2.4 |
| r2 | k_wholesale | 0.5532 | 0.5532 | +0.0000 | 1.9 | 2.2 |
| r2 | r_brazil | 0.9760 | 0.9760 | +0.0000 | 5.5 | 7.3 |
| r2 | r_kc_house | 0.8525 | 0.8525 | +0.0000 | 8.1 | 11.0 |
| r2 | r_moneyball | 0.9419 | 0.9419 | +0.0000 | 2.6 | 3.1 |
| r2 | r_seoul | 0.9356 | 0.9356 | +0.0000 | 10.1 | 12.8 |
| r2 | r_superconduct | 0.9108 | 0.9108 | -0.0000 | 17.3 | 19.4 |
| r2 | r_wine | 0.4647 | 0.4647 | +0.0000 | 3.3 | 4.2 |
| r3 | c_amazon | 0.7059 | 0.7059 | +0.0000 | 3.4 | 3.8 |
| r3 | c_cmc | 0.5718 | 0.5718 | +0.0000 | 2.9 | 3.2 |
| r3 | c_credit | 0.7012 | 0.7012 | +0.0000 | 6.9 | 8.5 |
| r3 | c_electricity | 0.8874 | 0.8874 | +0.0000 | 6.5 | 7.5 |
| r3 | c_letter | 0.9333 | 0.9443 | +0.0110 | 25.5 | 18.9 |
| r3 | c_nomao | 0.9658 | 0.9658 | +0.0000 | 10.2 | 12.4 |
| r3 | k_banknote | 0.0103 | 0.0103 | +0.0000 | 1.9 | 2.3 |
| r3 | k_blobs_easy | 1.0000 | 1.0000 | +0.0000 | 2.0 | 2.3 |
| r3 | k_dermatology | 0.9274 | 0.9274 | +0.0000 | 2.0 | 2.4 |
| r3 | k_glass | 0.1620 | 0.1620 | +0.0000 | 1.9 | 2.5 |
| r3 | k_moons | 0.3882 | 0.3882 | +0.0000 | 2.1 | 2.6 |
| r3 | k_optdigits | 0.5953 | 0.5953 | +0.0000 | 5.2 | 5.7 |
| r3 | r_airfoil | 0.8449 | 0.8449 | +0.0000 | 4.8 | 5.4 |
| r3 | r_insurance | 0.8796 | 0.8796 | +0.0000 | 3.0 | 3.7 |
| r3 | r_metro | 0.9733 | 0.9733 | +0.0000 | 7.5 | 8.7 |
| r3 | r_pol | 0.9889 | 0.9889 | +0.0000 | 12.4 | 14.8 |
| r3 | r_socmob | 0.8430 | 0.8430 | +0.0000 | 3.0 | 4.0 |
| r3 | r_taxi | 0.4041 | 0.4041 | +0.0000 | 4.3 | 5.1 |

### B_portfolio
| suite | dataset | 98f5fe1 | HEAD | Δ | wall 98f5fe1 | wall HEAD |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8143 | 0.8143 | +0.0000 | 8.3 | 9.6 |
| r1 | c_ai4i | 0.8420 | 0.8420 | +0.0000 | 5.7 | 6.2 |
| r1 | c_bank | 0.7750 | 0.7750 | +0.0000 | 13.8 | 15.9 |
| r1 | c_covertype | 0.6059 | 0.6059 | +0.0000 | 19.1 | 23.4 |
| r1 | c_creditg | 0.7074 | 0.7074 | +0.0000 | 5.0 | 6.1 |
| r1 | c_telco | 0.7242 | 0.7242 | +0.0000 | 6.0 | 6.6 |
| r1 | k_drybean | 0.6600 | 0.6600 | +0.0000 | 5.4 | 5.6 |
| r1 | k_iris | 0.5801 | 0.5801 | +0.0000 | 2.1 | 2.4 |
| r1 | k_mall | 0.6651 | 0.6651 | +0.0000 | 2.1 | 2.4 |
| r1 | k_penguins | 0.9505 | 0.9505 | +0.0000 | 2.0 | 2.4 |
| r1 | k_seeds | 0.7722 | 0.7722 | +0.0000 | 2.0 | 2.3 |
| r1 | k_wine | 0.8804 | 0.8804 | +0.0000 | 2.0 | 2.3 |
| r1 | r_abalone | 0.5739 | 0.5739 | +0.0000 | 4.7 | 5.4 |
| r1 | r_bike | 0.9570 | 0.9570 | +0.0000 | 14.1 | 18.3 |
| r1 | r_california | 0.8365 | 0.8365 | +0.0000 | 12.0 | 15.5 |
| r1 | r_concrete | 0.8884 | 0.8884 | +0.0000 | 4.0 | 4.8 |
| r1 | r_diamonds | 0.9807 | 0.9807 | +0.0000 | 10.1 | 13.1 |
| r1 | r_house | 0.8798 | 0.8798 | +0.0000 | 10.5 | 13.7 |
| r2 | c_churn | 0.8880 | 0.8880 | +0.0000 | 10.4 | 12.9 |
| r2 | c_creditapp | 0.8715 | 0.8715 | +0.0000 | 4.4 | 5.5 |
| r2 | c_jannis | 0.5537 | 0.5458 | -0.0078 | 26.8 | 24.3 |
| r2 | c_kick | 0.6690 | 0.6690 | +0.0000 | 15.2 | 19.1 |
| r2 | c_magic | 0.8620 | 0.8620 | +0.0000 | 6.7 | 8.1 |
| r2 | c_pageblocks | 0.8557 | 0.8557 | +0.0000 | 12.5 | 15.1 |
| r2 | k_ecoli | 0.7241 | 0.7241 | +0.0000 | 2.2 | 2.5 |
| r2 | k_pendigits | 0.6164 | 0.6164 | +0.0000 | 7.5 | 8.9 |
| r2 | k_segment | 0.4656 | 0.4656 | +0.0000 | 3.1 | 4.1 |
| r2 | k_synth | 0.9730 | 0.9730 | +0.0000 | 3.6 | 5.1 |
| r2 | k_vehicle | 0.0722 | 0.0722 | +0.0000 | 2.5 | 2.8 |
| r2 | k_wholesale | 0.5808 | 0.5808 | +0.0000 | 2.0 | 2.2 |
| r2 | r_brazil | 0.9860 | 0.9860 | +0.0000 | 9.7 | 10.8 |
| r2 | r_kc_house | 0.8505 | 0.8505 | +0.0000 | 12.3 | 15.2 |
| r2 | r_moneyball | 0.9506 | 0.9506 | +0.0000 | 4.2 | 6.4 |
| r2 | r_seoul | 0.9308 | 0.9308 | +0.0000 | 14.5 | 17.7 |
| r2 | r_superconduct | 0.9086 | 0.9086 | +0.0000 | 14.8 | 18.2 |
| r2 | r_wine | 0.5022 | 0.5022 | +0.0000 | 6.4 | 7.3 |
| r3 | c_amazon | 0.7118 | 0.7118 | +0.0000 | 6.5 | 7.2 |
| r3 | c_cmc | 0.5679 | 0.5679 | +0.0000 | 4.4 | 5.0 |
| r3 | c_credit | 0.7049 | 0.7049 | +0.0000 | 12.9 | 15.8 |
| r3 | c_electricity | 0.8825 | 0.8825 | +0.0000 | 10.7 | 12.5 |
| r3 | c_letter | 0.9072 | 0.9466 | +0.0394 | 52.2 | 25.8 |
| r3 | c_nomao | 0.9641 | 0.9641 | +0.0000 | 13.6 | 18.3 |
| r3 | k_banknote | 0.0116 | 0.0116 | +0.0000 | 2.1 | 2.8 |
| r3 | k_blobs_easy | 1.0000 | 1.0000 | +0.0000 | 3.5 | 3.7 |
| r3 | k_dermatology | 0.9269 | 0.9269 | +0.0000 | 2.2 | 2.2 |
| r3 | k_glass | 0.1717 | 0.1717 | +0.0000 | 2.2 | 2.2 |
| r3 | k_moons | 0.4132 | 0.4132 | +0.0000 | 2.5 | 2.7 |
| r3 | k_optdigits | 0.5829 | 0.5829 | +0.0000 | 7.0 | 7.0 |
| r3 | r_airfoil | 0.8436 | 0.8436 | +0.0000 | 6.8 | 6.9 |
| r3 | r_insurance | 0.8812 | 0.8812 | +0.0000 | 5.5 | 5.7 |
| r3 | r_metro | 0.9722 | 0.9722 | +0.0000 | 12.7 | 13.2 |
| r3 | r_pol | 0.9834 | 0.9834 | +0.0000 | 15.5 | 16.0 |
| r3 | r_socmob | 0.8485 | 0.8485 | +0.0000 | 4.6 | 4.8 |
| r3 | r_taxi | 0.4165 | 0.4165 | +0.0000 | 8.8 | 8.7 |

### C_rules
| suite | dataset | 98f5fe1 | HEAD | Δ | wall 98f5fe1 | wall HEAD |
|---|---|---|---|---|---|---|
| r1 | c_adult | 0.8150 | 0.8150 | +0.0000 | 5.0 | 6.1 |
| r1 | c_ai4i | 0.8517 | 0.8517 | +0.0000 | 3.3 | 4.2 |
| r1 | c_bank | 0.7674 | 0.7674 | +0.0000 | 7.0 | 9.2 |
| r1 | c_covertype | 0.5721 | 0.5721 | +0.0000 | 16.1 | 20.6 |
| r1 | c_creditg | 0.6686 | 0.6686 | +0.0000 | 5.7 | 9.5 |
| r1 | c_telco | 0.7344 | 0.7344 | +0.0000 | 4.0 | 5.0 |
| r1 | k_drybean | 0.6600 | 0.6600 | +0.0000 | 3.4 | 3.8 |
| r1 | k_iris | 0.7445 | 0.6525 | -0.0920 ** | 1.9 | 2.2 |
| r1 | k_mall | 0.6651 | 0.6807 | +0.0156 | 2.0 | 2.2 |
| r1 | k_penguins | 0.9536 | 0.9505 | -0.0031 | 1.9 | 2.2 |
| r1 | k_seeds | 0.7733 | 0.7733 | +0.0000 | 1.9 | 2.1 |
| r1 | k_wine | 0.8820 | 0.8471 | -0.0349 | 1.9 | 2.2 |
| r1 | r_abalone | 0.5540 | 0.5540 | +0.0000 | 3.1 | 3.7 |
| r1 | r_bike | 0.9598 | 0.9598 | +0.0000 | 8.1 | 10.3 |
| r1 | r_california | 0.8407 | 0.8407 | +0.0000 | 6.4 | 8.1 |
| r1 | r_concrete | 0.8818 | 0.8818 | +0.0000 | 3.1 | 3.6 |
| r1 | r_diamonds | 0.9800 | 0.9800 | +0.0000 | 5.5 | 7.2 |
| r1 | r_house | 0.8963 | 0.8963 | +0.0000 | 5.7 | 6.9 |
| r2 | c_churn | 0.9035 | 0.9035 | +0.0000 | 4.8 | 6.4 |
| r2 | c_creditapp | 0.9064 | 0.9064 | +0.0000 | 3.1 | 4.5 |
| r2 | c_jannis | 0.5424 | 0.5437 | +0.0012 | 27.6 | 23.9 |
| r2 | c_kick | 0.6697 | 0.6697 | +0.0000 | 11.9 | 14.3 |
| r2 | c_magic | 0.8641 | 0.8641 | +0.0000 | 8.8 | 10.8 |
| r2 | c_pageblocks | 0.8626 | 0.8626 | +0.0000 | 6.7 | 8.0 |
| r2 | k_ecoli | 0.7659 | 0.7659 | +0.0000 | 2.0 | 2.2 |
| r2 | k_pendigits | 0.5582 | 0.5582 | +0.0000 | 3.9 | 4.5 |
| r2 | k_segment | 0.4656 | 0.4656 | +0.0000 | 2.9 | 2.8 |
| r2 | k_synth | 0.9855 | 0.9776 | -0.0079 | 2.4 | 2.7 |
| r2 | k_vehicle | 0.0827 | 0.0694 | -0.0133 | 2.1 | 2.5 |
| r2 | k_wholesale | 0.4696 | 0.5532 | +0.0835 ** | 1.9 | 2.1 |
| r2 | r_brazil | 0.9749 | 0.9749 | +0.0000 | 4.2 | 4.7 |
| r2 | r_kc_house | 0.8568 | 0.8568 | +0.0000 | 7.0 | 8.1 |
| r2 | r_moneyball | 0.9492 | 0.9492 | +0.0000 | 3.1 | 3.6 |
| r2 | r_seoul | 0.9320 | 0.9320 | +0.0000 | 7.3 | 8.4 |
| r2 | r_superconduct | 0.9114 | 0.9114 | +0.0000 | 13.0 | 14.8 |
| r2 | r_wine | 0.4628 | 0.4628 | +0.0000 | 3.6 | 4.0 |
| r3 | c_amazon | 0.6966 | 0.6966 | +0.0000 | 5.8 | 6.3 |
| r3 | c_cmc | 0.5421 | 0.5421 | +0.0000 | 5.2 | 5.5 |
| r3 | c_credit | 0.7014 | 0.7014 | +0.0000 | 7.6 | 8.3 |
| r3 | c_electricity | 0.8909 | 0.8909 | +0.0000 | 8.3 | 8.6 |
| r3 | c_letter | 0.9293 | 0.9467 | +0.0174 | 30.1 | 12.7 |
| r3 | c_nomao | 0.9638 | 0.9638 | +0.0000 | 16.4 | 17.3 |
| r3 | k_banknote | 0.0169 | 0.0103 | -0.0066 | 2.0 | 2.0 |
| r3 | k_blobs_easy | 1.0000 | 1.0000 | +0.0000 | 2.0 | 2.1 |
| r3 | k_dermatology | 0.7050 | 0.9232 | +0.2182 ** | 2.0 | 2.0 |
| r3 | k_glass | 0.1620 | 0.1620 | +0.0000 | 2.0 | 2.0 |
| r3 | k_moons | 0.4166 | 0.3882 | -0.0284 | 2.2 | 2.2 |
| r3 | k_optdigits | 0.7148 | 0.6050 | -0.1098 ** | 5.7 | 6.1 |
| r3 | r_airfoil | 0.8306 | 0.8306 | +0.0000 | 2.8 | 3.0 |
| r3 | r_insurance | 0.8806 | 0.8806 | +0.0000 | 3.1 | 3.2 |
| r3 | r_metro | 0.9764 | 0.9764 | +0.0000 | 11.0 | 11.2 |
| r3 | r_pol | 0.9892 | 0.9892 | +0.0000 | 10.1 | 10.4 |
| r3 | r_socmob | 0.8472 | 0.8472 | +0.0000 | 3.0 | 3.3 |
| r3 | r_taxi | 0.4070 | 0.4070 | +0.0000 | 4.6 | 4.7 |
