# 08 — Validation report, round 3 (fix round 2 on a brand-new suite)

EECE 490 "Unknown Dataset Hackathon". Validator (Subagent 2), 2026-10-20.

## Scope

- **Approaches under test:** A, B and C after fix round 2 (HEAD = 98f5fe1; changelog in docs/07).
- **Data:** 18 **new** datasets with zero overlap with rounds 1–2, all their stress probes and docs/03.
- **Protocol and scoring model:** the same as docs/04 and docs/06, so all three rounds are comparable.
- **Three code versions on the new suite:** original (d14439d), round-1 fix (05dce9e) and round-2 fix (HEAD).
  The old versions ran from temporary git worktrees, which have since been removed.
- **Additional runs:**
  - independent recheck of the developer's HEAD numbers on suites r1 and r2;
  - 1-thread timing of all 18 round-3 datasets;
  - determinism checks;
  - 4 stress probes.

Nothing under `approaches/` or `tools/` was modified.

## Reproduce

```
py validation/build_suite_round3.py
py validation/run_validation_round3.py run | onethread | determinism | stress | recheck | baselines
git worktree add --detach <wt1> d14439d ; git worktree add --detach <wt2> 05dce9e
py validation/run_validation_round3.py versions --root-d14439d <wt1> --root-05dce9e <wt2>
py validation/run_validation_round3.py score            # -> validation/results_round3/
```

Results (committed) are in `validation/results_round3/`: `results.csv`, `profile_scores.csv`, `baselines.csv`,
`version_comparison.csv`, `round1_recheck.csv` (HEAD on r1 + r2 vs docs/07), `onethread_timing.csv`,
`determinism.csv`, `stress_results.csv` and `summary.json`. The suite is documented in
`validation/SUITE_ROUND3.md`.

---

## 1. Summary

### Round 3 (new suite) — HEAD (98f5fe1)

| approach | quality (85) | profiling (10) | speed (5) | **total** | clf pool (28.3) | reg pool (28.3) | clu pool (28.3) | crashes / timeouts / watchdog | median / max wall s (2 thr) |
|---|---|---|---|---|---|---|---|---|---|
| **A_lean** | 57.03 | 9.985 | 3.95 | **70.97** | **19.17** | 23.30 | 14.56 | 0 / 0 / 0 | 3.5 / 28.8 |
| B_portfolio | **57.08** | 9.985 | 3.76 | 70.82 | 19.06 | **23.35** | **14.67** | 0 / 0 / **1 (c_letter, 52.2 s)** | 6.1 / **52.2** |
| C_rules | 56.38 | 9.985 | **4.40** | 70.76 | 18.85 | 23.29 | 14.24 | 0 / 0 / 0 (1 invalid-k stress probe) | 4.7 / 23.8 |
| *sklearn defaults (quality only)* | *53.18* | | | | *17.03 (HGB)* | *22.64 (HGB)* | *13.51 (scaled KMeans)* | | |

- Relative-to-best normalisation: A 77.47, B 78.03, C 77.44.
- Profiling 9.985 = 10.00 with one scorer artifact (§7).

**The three approaches finish within 0.21 points of each other. That is within single-split noise.**

### Version comparison on round-3 data (out-of-sample effect of each fix round)

| approach | original d14439d | round-1 fix 05dce9e | round-2 fix HEAD | Δ fix round 1 | Δ fix round 2 |
|---|---|---|---|---|---|
| A_lean | 68.98 | 69.84 | **70.97** | +0.85 | **+1.13** |
| B_portfolio | 63.66 | 69.76 | **70.82** | +6.10 | **+1.06** |
| C_rules | 62.32 | 69.55 | **70.76** | +7.24 | **+1.21** |

### Developer-claim recheck (HEAD on suites r1 + r2)

| | r1 (mine) | r2 (mine) | r1 + r2 (mine) | docs/07 claim (36 datasets) |
|---|---|---|---|---|
| A | 77.36 | 73.40 | 150.77 | 150.76 |
| B | 76.41 | 73.41 | 149.82 | 149.84 |
| C | 77.24 | 73.58 | 150.83 | 150.83 |

All 108 runs are valid. The max per-dataset abs difference of the primary metric vs docs/07 is **5e-5**.
**The claims reproduce.**

### Cumulative standings, all 54 datasets (HEAD; r1 + r2 + r3)

| approach | quality | profile | speed | **total** | clf | reg | clu |
|---|---|---|---|---|---|---|---|
| **A_lean** | 178.76 | 29.98 | 12.99 | **221.73** | **55.14** | 71.50 | 52.12 |
| C_rules | 178.09 | 29.98 | **13.51** | 221.58 | 54.59 | 71.45 | 52.05 |
| B_portfolio | 178.67 | 29.98 | 11.99 | 220.64 | 54.78 | **71.73** | **52.16** |

Rounds 1–2 are development data for HEAD. The only fully out-of-sample score for HEAD is round 3
(A 70.97 ≈ B 70.82 ≈ C 70.76).

### Headlines

1. **Fix round 2 generalises.** On unseen data all three gain about +1.1 points (A +1.13, B +1.06, C +1.21). The
   gains come from the shared SH1/SH2 clustering mechanisms:
   - k_dermatology: A 0.734 → 0.927, B 0.717 → 0.927;
   - k_optdigits: A 0.559 → 0.595, C 0.583 → 0.715;
   - C3 transform verification: C r_pol 0.981 → 0.989, r_taxi 0.339 → 0.407.
2. **Fix round 1 was confirmed again out of sample, and decisively:**
   - S1 on c_cmc, whose target class is literally `none`. The original code turned it into missing and
     dropped a whole class: 0.32 → 0.57.
   - C1 on r_metro, an integer count with a 38-level categorical. The original C hit the watchdog with
     R² −0.007; fixed, 0.976.
   - B's original c_letter result was the constant fallback (0.003).
3. **New timing risk, B on c_letter (26 classes, 7500 rows): watchdog exit at 52.2 s at both 1 and 2 threads.**
   - The quick-fit affordability check passed (30 iterations on 5000 rows).
   - The full `hgb_a` fold then took about 37 s; with 26 classes each iteration grows 26 trees. B aborted it at
     39.5 s.
   - Something after `abort extratrees` ran until the watchdog fired.
   - Output: the quick model at 2 threads (0.907); at 1 thread it got further, to ExtraTrees (0.955). So the
     **result is CPU-dependent**.
   - A also spends 26 s in one fold there (28.8 s at 2 threads, 35.7 s at 1 thread). C takes 23.8 / 28.7 s.
4. **New validity risk, C:** on the clustering train/test probe (`s3_optdigits_traintest`), C wrote only **9 of
   k = 10** cluster labels for the test rows. One cluster contains only training rows.
   - Under the strict "exactly k distinct labels" rule this is invalid (0 points).
   - A and B guarantee k on the predicted rows.
5. Every approach is **below the plain default HGB on c_letter** (0.947 vs A 0.934, C 0.920, B 0.907). The
   26-class problem is where their slow, heavily regularised configurations hurt.

---

## 2. Scoring model and protocol (unchanged; see docs/04 §2)

- **Quality:** 85/18 points × `clip((m − trivial)/(1 − trivial), 0, 1)` with macro-F1 / R² / ARI.
- **Profile:** 10/18 points × the mean of 14 fields.
- **Speed:** 5/18 points × f(t) × useful.
- **Runs:** `py approaches/<X>/main.py --data <dir> --out <out>` as a subprocess with
  `OMP/OPENBLAS/MKL_NUM_THREADS=2` (1 for the timing check). Full wall time, 60 s hard kill, sequential runs,
  one warm-up per approach and tag.
- The harness imports `run_validation.py` unchanged.

---

## 3. The round-3 suite (validation/SUITE_ROUND3.md)

| task | datasets | probes |
|---|---|---|
| classification | `c_amazon` (OpenML 4135, 5.8 % minority, 9 high-cardinality integer codes), `c_credit` (42477, + MCAR / duplicate / constant / ID), `c_electricity` (151, + noise), `c_letter` (6, 26 classes), `c_cmc` (23, small n, literal class `none`), `c_nomao` (1486, 18750 × 119) | A3, B1 heavy size class, S1 / S2 |
| regression | `r_metro` (UCI 492 count + datetime), `r_socmob` (541, skewed count), `r_taxi` (42729, skewed non-count + location codes), `r_pol` (201, skewed bounded non-count, raw should win), `r_insurance` (Kaggle, ages ≥ 60 only in test), `r_airfoil` (43919, frequencies ≥ 10 kHz only in test) | B2 / C3 transform choice, B3 out-of-range (linear right / wrong) |
| clustering | `k_glass` (41, unequal), `k_banknote` (1462, elongated), `k_blobs_easy` (synthetic, KMeans should win), `k_optdigits` (28, 64 moderately informative features), `k_dermatology` (35, + 5 noise), `k_moons` (synthetic non-convex + noise) | SH1 (keep vs drop), SH2 agreement medoid |

Two datasets do not discriminate between the approaches:
- `k_banknote`: ARI ≤ 0.05 for every method including both baselines.
- `k_blobs_easy`: every method scores 1.0, as intended for the easy case.

---

## 4. Per-approach analysis (round 3, HEAD)

### 4.1 A_lean — 70.97 (1st by 0.15; best classification pool)

**High:**
- Best classification pool (19.17), with the best score on 3 of 6 classification sets:
  - c_cmc 0.572: the n < 2000 linear blend is chosen (0.5 / 0.5);
  - c_letter 0.934: the ExtraTrees blend at 0.75 rescues a single HGB fold;
  - c_nomao 0.966;
- c_amazon 0.706, up from 0.694 with A3's sqrt weights (B is best there at 0.712).
- Clustering: k_dermatology **0.927**. The relevance test dropped exactly the 5 planted noise features, and the
  medoid picked Ward. k_blobs_easy 1.0.
- Second-fastest: median 3.5 s.

**Low:**
- **c_letter cost:** a single 5-fold HGB fold takes about 26 s (26 classes × early-stopped iterations,
  patience 60). A finished 1 fold (`cv_folds_done = 1`). The 3-fold rule (n × p × C > 5e6) does not fire,
  because 7500 × 16 × 26 = 3.1e6.
- **k_optdigits:** the relevance test dropped 5 edge pixels (input9 / 17 / 25 / 33 / 57) as "irrelevant"
  (p = 0.02–0.16). The medoid picked GMM-full: 0.595, against C's 0.715.
- **k_moons** 0.388: GMM-diag medoid, below the plain scaled KMeans baseline (0.450).
- **r_metro RMSLE 0.39** vs C 0.33. A keeps the raw target, which is correct for R², since the skew is −0.1.

### 4.2 B_portfolio — 70.82 (2nd; best regression and clustering pools; slowest)

**High:**
- Regression 23.35. Best on r_socmob (0.849, log1p ensemble), r_taxi (0.417) and r_insurance (0.881).
- Clustering 14.67: k_glass 0.172, k_dermatology 0.927.
- Classification: best c_amazon (0.712) and c_credit (0.705), with ExtraTrees and logistic members in the
  ensemble.

**Low:**
- **c_letter watchdog (52.2 s, 0.907, below A and C).** Details in §1 (headline 3). This is the one run in
  any round-3 configuration above 40 s.
- **B2 tie-break cost on r_pol:** the log-only ensemble (`hgb_a[log1p]`: 1.0) replaces the raw ensemble
  because the OOF R² is within 0.01. Test R² is 0.983 vs 0.989 for A/C and for B's own round-1 version. This
  is the designed trade (RMSLE 0.36 vs 0.56 for A/C), but it costs points under R².
- Speed 3.76 (median 6.1 s). The **1-thread run changes B's output on 5 / 18 datasets**: letter's stage,
  nomao 0.9641 → 0.9635, and last-digit float differences on 3 regression sets. B's affordability fallback is
  clock-based.

### 4.3 C_rules — 70.76 (3rd by 0.06; fastest maximum time)

**High:**
- Clustering: **k_optdigits 0.715**, the best by 0.12 (Ward in the sparse-weighted space).
- C3 works:
  - r_pol: the log1p rule is reverted by the probe (0.9534 < 0.9705 − 0.005), giving 0.989;
  - r_taxi: reverted to raw plus a linear blend, giving 0.407 (round-1 fix: 0.339);
  - r_metro: 0.976, the best.
- Best c_electricity (0.891). Best balanced accuracy on c_amazon (0.706).
- Fastest maximum (23.8 s) and best speed points (4.40). No run over 30 s even at 1 thread.

**Low:**
- **k_dermatology 0.705** (A / B 0.927). The soft-weighted space downweights the clinical grades, and KMeans
  wins the medoid. Round-1 fix: 0.774.
- **c_cmc 0.542:** the probe-driven 0.5 linear + 0.5 HGB blend is weaker than A / B's tuned blend or ensemble.
- **Invalid k on the train/test clustering probe** (9 / 10 labels on test rows). See §10.

---

## 5. Per-dataset results (round 3; bold = best of A/B/C; points = 4.72 × norm)

| dataset | metric | A | B | C | sk default | sk linear / raw-KMeans | trivial | pts A | pts B | pts C | original A / B / C | round-1 fix A / B / C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c_amazon | macro_f1 | 0.7059 | **0.7118** | 0.6966 | 0.5271 | 0.4986 | 0.4851 | 2.02 | 2.08 | 1.94 | 0.667 / 0.708 / 0.697 | 0.694 / 0.712 / 0.697 |
| c_cmc | macro_f1 | **0.5718** | 0.5679 | 0.5421 | 0.5222 | 0.4772 | 0.1994 | 2.20 | 2.17 | 2.02 | **0.323 / 0.331 / 0.335** | 0.572 / 0.568 / 0.542 |
| c_credit | macro_f1 | 0.7012 | **0.7049** | 0.7014 | 0.6878 | 0.6326 | 0.4378 | 2.21 | 2.24 | 2.21 | 0.703 / 0.704 / 0.701 | 0.701 / 0.705 / 0.701 |
| c_electricity | macro_f1 | 0.8874 | 0.8825 | **0.8909** | 0.8684 | 0.7665 | 0.3668 | 3.88 | 3.85 | 3.91 | 0.885 / 0.882 / 0.891 | 0.887 / 0.883 / 0.891 |
| c_letter | macro_f1 | **0.9340** | 0.9072 | 0.9195 | *0.9471* | 0.7658 | 0.0030 | 4.41 | 4.28 | 4.34 | 0.919 / **0.003** / 0.920 | 0.934 / 0.907 / 0.920 |
| c_nomao | macro_f1 | **0.9658** | 0.9641 | 0.9638 | 0.9631 | 0.9395 | 0.4167 | 4.45 | 4.43 | 4.43 | 0.965 / 0.964 / 0.964 | 0.966 / 0.964 / 0.964 |
| r_airfoil | r2 | **0.8449** | 0.8436 | 0.8306 | 0.8311 | 0.4502 | 0 | 3.99 | 3.98 | 3.92 | 0.846 / 0.842 / 0.831 | 0.845 / 0.844 / 0.831 |
| r_insurance | r2 | 0.8796 | **0.8812** | 0.8806 | 0.8663 | 0.7791 | 0 | 4.15 | 4.16 | 4.16 | 0.880 / 0.883 / 0.881 | 0.880 / 0.881 / 0.881 |
| r_metro | r2 | 0.9733 | 0.9722 | **0.9764** | 0.9643 | 0.1684 | 0 | 4.60 | 4.59 | 4.61 | 0.973 / 0.972 / **−0.007** | 0.973 / 0.972 / 0.976 |
| r_pol | r2 | 0.9889 | 0.9834 | **0.9892** | 0.9862 | 0.4706 | 0 | 4.67 | 4.64 | 4.67 | 0.989 / 0.989 / 0.981 | 0.989 / 0.989 / 0.981 |
| r_socmob | r2 | 0.8430 | **0.8485** | 0.8472 | 0.7521 | 0.7347 | 0 | 3.98 | 4.01 | 4.00 | 0.843 / 0.848 / 0.847 | 0.843 / 0.848 / 0.847 |
| r_taxi | r2 | 0.4041 | **0.4165** | 0.4070 | 0.3951 | 0.4113 | 0 | 1.91 | 1.97 | 1.92 | 0.411 / 0.415 / 0.339 | 0.404 / 0.416 / 0.339 |
| k_banknote | ari | 0.0103 | 0.0116 | **0.0169** | 0.0129 | 0.0485 | 0 | 0.05 | 0.05 | 0.08 | 0.013 / 0.003 / 0.002 | 0.009 / 0.012 / 0.002 |
| k_blobs_easy | ari | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | 0 | 4.72 | 4.72 | 4.72 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| k_dermatology | ari | **0.9274** | 0.9269 | 0.7050 | 0.7035 | −0.0070 | 0 | 4.38 | 4.38 | 3.33 | 0.912 / 0.895 / 0.774 | 0.734 / 0.717 / 0.774 |
| k_glass | ari | 0.1620 | **0.1717** | 0.1620 | 0.1418 | 0.2702 | 0 | 0.76 | 0.81 | 0.76 | 0.176 / 0.175 / 0.191 | 0.176 / 0.172 / 0.175 |
| k_moons | ari | 0.3882 | 0.4132 | **0.4166** | *0.4500* | −0.0026 | 0 | 1.83 | 1.95 | 1.97 | 0.389 / 0.417 / 0.484 | 0.387 / 0.413 / 0.417 |
| k_optdigits | ari | 0.5953 | 0.5829 | **0.7148** | 0.5531 | 0.6794 | 0 | 2.81 | 2.75 | 3.38 | 0.540 / 0.553 / 0.361 | 0.559 / 0.577 / 0.583 |

Secondary metrics (selection):

| | A | B | C |
|---|---|---|---|
| balanced accuracy c_amazon / c_credit / c_cmc | 0.681 / 0.690 / **0.580** | 0.672 / **0.706** / 0.566 | **0.706** / 0.703 / 0.545 |
| RMSLE r_metro / r_pol / r_socmob / r_insurance | 0.391 / 0.560 / 0.362 / 0.332 | 0.395 / **0.360** / **0.359** / **0.315** | **0.334** / 0.551 / 0.365 / 0.335 |

Out-of-range rows (B3 probe):

| dataset | rows | metric on those rows | A | B | C |
|---|---|---|---|---|---|
| r_insurance | 114 test rows aged ≥ 60 | RMSE / mean bias | 4994 / −2583 | 5002 / −2639 | 4880 / −2442 |
| r_airfoil | 86 test rows ≥ 10 kHz | RMSE / mean bias | 5.4 / +4.6 | 5.4 / +4.6 | 5.3 / +4.6 |

- On both datasets every approach is tree-dominated, so the B3 cap never binds: B's linear weight is 0.03 on
  insurance and 0 on airfoil.
- All approaches under-predict the oldest customers by about 2.5 k. Offline, a ridge + HGB blend reduces that
  bias to −1.9 k.
- All approaches over-predict the high-frequency noise by 4.6 dB.
- Neither probe could show B3 working or failing.

---

## 6. Timing and the weaker-CPU check

| approach | 2 thr median / p90 / max | **1 thr median / max** | 1 thr runs > 30 s | 1 thr runs > 40 s | watchdog exits (2 thr / 1 thr) |
|---|---|---|---|---|---|
| A_lean | 3.5 / 11.5 / 28.8 (c_letter) | 3.4 / **35.7** (c_letter) | 1 | 0 | 0 / 0 |
| B_portfolio | 6.1 / 15.6 / **52.2** (c_letter) | 6.6 / **52.1** (c_letter) | 1 | **1** | **1 / 1** |
| C_rules | 4.7 / 13.1 / 23.8 (c_letter) | 4.6 / **28.7** (c_letter) | 0 | 0 | 0 / 0 |

- The median slowdown from 2 to 1 thread is only about 1.03×. Most fits are too small to use 2 threads.
- The largest slowdowns are on the biggest tables: c_nomao ×1.45–1.52, r_pol ×1.38, c_letter ×1.2.
- **On a grader with 2× slower cores:**
  - A's c_letter fold (26 s) would become about 52 s, so A would rely on its watchdog too. Its per-fold budget
    check cannot interrupt a running fit.
  - C has the most headroom (28.7 s at 1 thread).
  - B already hits the watchdog.
- **The real risk is the 26-class table:** n × p × C = 3.1e6, below both 3-fold thresholds (5e6).
- First predictions are on disk at ≤ 0.67 s in every run. With a weaker CPU the outputs stay valid; only the
  quality on c_letter would fall.

---

## 7. Profile breakdown

| field group | A / B / C | notes |
|---|---|---|
| 13 of 14 fields on 18 datasets | 1.000 | undeclared `id`, injected constant / duplicate, native constant pixels / pol columns, datetimes (`date_time`, `pickup_datetime`) |
| duplicate_columns on c_nomao | 0.615 (**scorer artifact**) | V81–V86 are 6 identical columns. The profile reports each copy against the first original (5 pairs); the scorer's ground truth lists all 15 pairwise pairs. The profile is correct, so corrected profiling is 10.00 for all three. Rounds 1–2 had no group of ≥ 3 identical columns |

**Stress probes (no feature types):**
- `s3_amazon_notypes`: col_type 0.22. SH3 makes `MGR_ID` and `ROLE_CODE` categorical. `RESOURCE`,
  `ROLE_ROLLUP_1/2`, `ROLE_DEPTNAME`, `ROLE_TITLE`, `ROLE_FAMILY(_DESC)` stay numeric: their names carry no
  code / id token.
- `s3_taxi_notypes`: PULocationID / DOLocationID / RatecodeID are correctly categorical. VendorID and
  trip_type have only 2 values and are left numeric by design.

---

## 8. Fix verification (docs/07 IDs, on round-3 data; evidence = HEAD vs round-1 fix 05dce9e unless noted)

| ID | verdict | evidence |
|---|---|---|
| SH1 permutation-confirmed relevance filter | **works, with false drops on weak features** | k_dermatology: exactly the 5 planted noise features dropped (p = 0.12–0.7; real features p ≤ 0.005), giving A 0.734 → 0.927 and B 0.717 → 0.927. k_moons: 3 of 4 noise dropped. k_optdigits (must keep): 5 weak edge pixels dropped (p 0.02–0.16), no ARI loss (A +0.04). **k_banknote: `variance` dropped (p = 0.37), although it is the most class-informative feature** (offline KMeans on `variance` alone: ARI 0.475 vs 0.013 on all features). No ARI cost here, because every method fails on banknote, but it shows the test judges relevance against a partition that may itself be wrong |
| SH2 agreement medoid (KMeans / Ward / GMM-diag continuous / GMM-full PCA-whitened) | **works on easy / clean cases; no effect on hard ones** | blobs_easy: all candidates agree (1.0), KMeans kept. dermatology: Ward medoid (A) 0.927. optdigits: GMM-full (A) 0.595, Ward (C) 0.715. glass (unequal sizes): KMeans medoid 0.162–0.172 vs raw KMeans baseline 0.270, and every candidate is poor. moons: A GMM-diag 0.388, C KMeans 0.417, both < scaled KMeans baseline 0.450. banknote (elongated): KMeans / GMM-full medoids ≈ 0.01 |
| SH3 integer-coded categoricals by name | **partial** | taxi PU/DO LocationID, RatecodeID → categorical (R² unchanged without types). amazon: only MGR_ID / ROLE_CODE matched; 7 other code columns inferred numeric, macro-F1 −0.015 to −0.022 without types |
| A1 (SH1 + SH2 in A) | **works** | clustering pool 13.53 → 14.56 (dermatology +0.19, optdigits +0.04) |
| A2 four candidates + medoid | **works** (modest) | see SH2; clustering ≤ 4.5 s |
| A3 sqrt-balanced weights for all imbalanced tasks | **works** | c_amazon 0.694 → 0.706 (balanced accuracy 0.681). credit 0.701 = 0.701 |
| A4 | no change (by design) | c_cmc blend 0.5 / 0.5 is unchanged since the round-1 fix (0.572, best) |
| A (3-fold rule n·p·C > 5e6) | **no effect** (not triggered) | c_letter 3.1e6 → 5 folds planned, 1 completed, 28.8 s (35.7 s at 1 thread) |
| B1 full `hgb_a` fallback + heavy size class | **partial** | c_nomao (18750 × 118 = 2.2e6 cells, heavy class): 15.0 s, 0.964. **c_letter: the affordability check passes on the quick fit, but the real `hgb_a` fold costs ~37 s (26 classes) → abort → watchdog at 52 s, quick model kept (0.907). Same outcome as the round-1 fix** |
| B2 log-only ensemble tie-break | **works for RMSLE, costs R² when raw is right** | r_pol: log ensemble chosen (OOF within 0.01), R² 0.989 → 0.983, RMSLE 0.56 → 0.36. socmob / insurance: log ensembles, best R² and RMSLE |
| B3 linear cap on out-of-range rows | **no effect (not exercised)** | linear weight 0.03 (insurance) / 0 (airfoil). Out-of-range RMSE identical across versions |
| B4 shared SH1 filter in the consensus | **works** | dermatology 0.717 → 0.927, optdigits 0.577 → 0.583, glass / moons unchanged |
| C1 (SH1 + SH2, soft-weighted space only if confirmed) | **partial** | optdigits 0.583 → **0.715** (best). dermatology 0.774 → **0.705**: the soft weights hurt when noise is confirmed. banknote 0.002 → 0.017 |
| C2 rule-gated candidates + medoid | **works** (modest) | glass 0.175 → 0.162, moons 0.417 = 0.417, optdigits Ward chosen |
| C3 probe-verified log1p | **works** | r_pol 0.981 → **0.989** (reverted to raw), r_taxi 0.339 → **0.407** (raw + linear blend), socmob / insurance keep log1p |
| C4 3 folds for very large multiclass tables | **no effect** (not triggered) | c_letter 3.1e6 < 5e6. C finishes anyway at 23.8 s |
| *Round-1 fixes re-confirmed* | **works** | S1: c_cmc 0.32 → 0.57 (target class `none`). C1: r_metro −0.007 (watchdog) → 0.976. B (round-1) letter: constant 0.003 → 0.907 |

---

## 9. Version comparison on round-3 data (original → round-1 fix → round-2 fix)

| approach | pool | original | round-1 fix | round-2 fix |
|---|---|---|---|---|
| A | clf / reg / clu | 17.27 / 23.33 / 14.31 | 19.07 / 23.30 / 13.53 | 19.17 / 23.30 / 14.56 |
| B | clf / reg / clu | 13.33 / 23.37 / 14.37 | 19.06 / 23.38 / 13.65 | 19.06 / 23.35 / 14.67 |
| C | clf / reg / clu | 17.63 / 18.32 / 13.28 | 18.85 / 22.93 / 13.93 | 18.85 / 23.29 / 14.24 |
| totals | A / B / C | 68.98 / 63.66 / 62.32 | 69.84 / 69.76 / 69.55 | 70.97 / 70.82 / 70.76 |

How the developer's in-sample estimates compare with the out-of-sample effect:

| | dev in-sample (36 datasets, docs/07) | round 3 (18 unseen) | per-dataset rate in-sample → unseen |
|---|---|---|---|
| A | +2.57 | +1.13 | 0.071 → 0.063 |
| B | −0.04 | +1.06 | ≈ 0 → 0.059 |
| C | +0.64 | +1.21 | 0.018 → 0.067 |

**No sign of overfitting in fix round 2.** The principle-based changes transfer at least as well as they
measured in-sample. In round 2, by contrast, A's threshold tuning reversed out of sample.

---

## 10. Requirements checklist R1–R20 (HEAD, round 3)

| req | A_lean | B_portfolio | C_rules | evidence |
|---|---|---|---|---|
| R1 never crashes | pass | pass | pass | 0 tracebacks / non-zero exits in 54 + 54 (1 thr) + 6 + 12 + 108 HEAD runs |
| R2 ≤ 40 s at 2 threads | pass (28.8) | **fail** (52.2 s watchdog, c_letter) | pass (23.8) | onethread_timing.csv |
| R3 output validity | pass | pass | **partial** | C: 9 / 10 cluster labels on the test rows of `s3_optdigits_traintest` (`invalid_k` under the strict rule). Suite runs 54 / 54 valid |
| R4 shared adapter | pass | pass | pass | |
| R5 pinned deps | pass | pass | pass | |
| R6 deterministic | pass (1 vs 2 thr: c_letter 0.9996 rows equal) | **partial** (1 vs 2 thr differs on 5 / 18; c_letter 0.907 vs 0.955) | pass (taxi: float digits only) | determinism.csv; all 2-thread reruns identical |
| R7 classification quality | pass (19.17) | pass (19.06) | pass (18.85) | sk 17.03; all < default HGB on c_letter |
| R8 regression quality | pass | pass | pass | sk 22.64 |
| R9 clustering quality | pass (14.56) | pass (14.67) | pass (14.24) | sk 13.51 |
| R10 balanced across tasks | pass | pass | pass | |
| R11 imbalance | pass | pass | pass | amazon balanced accuracy 0.68 / 0.67 / 0.71 |
| R12 skewed-target transform | pass | pass (RMSLE-first by design; pol −0.006 R²) | pass (C3) | |
| R13 leak-free | pass | pass | pass | |
| R14 profile correctness | pass (1.000 corrected) | pass | pass | §7 scorer artifact |
| R15 profile cheap | pass | pass | pass | profile on disk < 1 s |
| R16 speed | pass (median 3.5) | pass (6.1) | pass (4.7) | |
| R17 anytime | pass | pass | pass | first predictions ≤ 0.67 s; B's watchdog run still has a valid file |
| R18 bounded threads / width | pass | pass | pass | |
| R19 decision logging | pass | partial (letter: no `modelling` block, watchdog exit) | pass | |
| R20 contract variation | pass | pass | **partial** (variant-2 clustering k guarantee) | stress_results.csv |

### Round-3 stress probes

| probe | A | B | C | base A / B / C |
|---|---|---|---|---|
| s3_amazon_notypes (SH3) | 0.691 | 0.693 | 0.674 | 0.706 / 0.712 / 0.697 |
| s3_taxi_notypes (SH3) | 0.404 | 0.418 | 0.407 | 0.404 / 0.416 / 0.407 |
| s3_credit_missing (+25 % MCAR) | 0.693 | 0.689 | 0.684 | 0.701 / 0.705 / 0.701 |
| s3_optdigits_traintest (variant 2) | 0.565 | 0.567 | **0.477, invalid_k (9 / 10 labels)** | 0.595 / 0.583 / 0.715 |

---

## 11. Remaining improvements (prioritised; not implemented)

**Must-do before Wednesday (any approach that is submitted):**
1. **Many-class cost guard (all three; B critical).** Project HGB cost as quick-fit time × (planned
   iterations / quick iterations) × (n / n_quick), with the multiclass factor C included. Then:
   - use 3 folds and a lower iteration cap, or ExtraTrees first, when n × p × C > ~1e6;
   - make B's per-candidate abort able to stop *inside* a fit, for example by fitting HGB in chunks with
     `warm_start` and checking the clock between chunks.

   Expected: c_letter < 20 s for all three, and B no longer watchdogs. Plain default HGB (100 iterations) reaches
   0.947 there in about 4 s.
2. **C: guarantee exactly k distinct labels on the predicted rows** in the train/test clustering variant.
   Re-run `ensure_k_labels` on the test subset, as A and B do.

**Should-do:**
- **A / B / C:** for multiclass with many classes, add a "fast HGB" candidate (lr 0.1, 100–200 iterations, no
  class weights). All are below the default on letter.
- **SH1:** before dropping a feature, also test it against a partition computed *with* the candidate (or
  against a bootstrap of the reference partition). banknote's `variance` was dropped although alone it gives
  ARI 0.475.
- **C:** use the hard SH1 filter (drop) instead of soft weights when noise is confirmed (dermatology 0.705 vs
  0.927 for A / B).
- **SH3:** extend the name rules to `RESOURCE` / `ROLE_*`-style columns: high-cardinality integers
  (> 50 levels, unique ratio < 0.5) with no arithmetic meaning. Or accept the gap as an inference limit, since
  metadata declares types in the real contract.
- **B2:** require that the log-only ensemble is not worse on OOF R² by more than 0.003 (instead of 0.01) when
  the target is not a count (r_pol).

---

## 12. Final-submission recommendation (firm)

**Submit A_lean, after the two must-do fixes above** (the many-class cost guard for A, and nothing else
required for A).

Reasons:
- **Best cumulative total over all 54 datasets** (221.73; C 221.58, B 220.64), and best on the only fully
  out-of-sample suite for HEAD (round 3: 70.97).
- Best classification pool overall (55.14).
- **No validity issue in any round-3 configuration.** C has the variant-2 k issue; B has the c_letter watchdog
  and CPU-dependent outputs.
- The lowest median time. Its only timing risk is the many-class fold cost, which is the same fix as for B.
- Fix round 2 helped A out of sample (+1.13), so its clustering is no longer the weak spot it was in round 2.

**Runner-up: C_rules.** It has the most timing headroom (28.7 s max at 1 thread) and the best single clustering
result (optdigits). It is acceptable *only* with the k-guarantee fix. **B is not recommended:** it has the one
> 40 s run, and its outputs depend on CPU speed.

The margins (≤ 1.1 points over 54 datasets, ≤ 0.21 on round 3) are within noise, so the decision rests on
robustness, not on score.

---

## 13. Caveats

- **The scoring model is ours** (docs/04 §10). A different normalisation or metric (for example RMSLE for
  regression) would favour B on r_pol / r_socmob / r_insurance.
- **One split per dataset.** Round-3 totals differ by 0.21. Per-dataset clustering swings of ±0.2 are common.
- **Weak probes.** `k_banknote` and `k_blobs_easy` do not discriminate. The B3 probes (insurance, airfoil)
  did not exercise linear members.
- **Scorer artifact.** The profile duplicate-pairs scorer under-counts groups of ≥ 3 identical columns
  (nomao). It is reported but not changed, to keep the rounds comparable.
- **Timing caveats.** The 1-thread check simulates fewer cores, not slower cores. A 2× slower CPU would mainly
  hurt c_letter-type tables. Timings are warm runs on the same 24-core machine.
- **Contract assumption.** The variant-2 clustering "exactly k labels on the test rows" requirement is our
  reading of docs/02. A grader that scores train + test jointly would not penalise C.
