# EECE 490 — Unknown Dataset Hackathon

**Team:** Dani Salman · Cihat el Sayid · Ahmed al katerji

One program that reads an unfamiliar table, works out what it contains, and solves the task that comes with it.
- **classification:** predicts a class;
- **regression:** predicts a number;
- **clustering:** puts rows into the required number of groups k.

For every dataset it writes predictions plus a structured data profile. It must finish within the 60-second limit on a CPU it has never seen.

> **Submitted approach: `A_lean`** (set in `main.py` → `SELECTED`). It was chosen on the results of our final validation round; see [Results](#results).

---

## How to run

Requires Python 3.14 (tested with 3.14.7). Dependencies are pinned in `requirements.txt`: numpy, pandas, scipy, scikit-learn, threadpoolctl and joblib. Nothing else is needed and nothing is downloaded at run time.

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python main.py --data <dataset_dir> --out <out_dir>
```

Alternative form with explicit files:

```bash
python main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>
```

Options:
- `--time-budget S`: internal soft budget, default 40 s.
- `--hard-limit S`: watchdog, default budget + 12 s.
- `--n-jobs N`: thread cap, default min(4, cores).
- `--quiet`.

The `compileall` step and one throw-away run after installing are a **warm-up**. The very first scikit-learn import on a fresh install can take tens of seconds; we measured 59 s once on Windows. That is a one-time cost of compiling and scanning, not of our code. Later runs start in about 1.5 s.

**Outputs** in `<out_dir>/`:
- `predictions.csv`: one row per test row, in the original order. Columns are `row_index[,<id_column>],prediction`. Classification keeps the original label values; clustering labels are integers 0..k-1, with exactly k distinct labels.
- `profile.json`: the structured data profile (schema in `docs/01_research_report.md` §4.1). It contains:
  - shape, and duplicate rows / columns;
  - per-column declared and inferred type, missing / blank / suspected-missing counts, unique counts and top values;
  - constant, ID-like and datetime columns;
  - target statistics or class balance;
  - a log of every modelling decision.

### Input format (our assumption)
No kickoff contract was published, so the assumed format is described in full in [`docs/02_contract_and_dataset_format.md`](docs/02_contract_and_dataset_format.md). In short, a dataset folder contains `train.csv`, `test.csv` and `metadata.json`. The metadata gives the task, target, feature types, `n_clusters` and an optional `id_column`. Clustering may instead use a single `data.csv`. Key names are matched case-insensitively with many aliases, and missing metadata is inferred. All contract parsing lives in each approach's `io_contract.py`, so adapting to a different contract is a one-file change.

---

## Approach

### Shared foundation (identical in all three solvers)
- **Robust loading and cleaning:**
  - numbers stored as strings (`"1,234"`, `"$12"`, `"12%"`) are coerced;
  - NA tokens count as missing only inside numeric columns, so literal categories such as `none` survive;
  - yes/no columns become booleans and datetime columns are expanded;
  - ID-like, constant and duplicate columns are detected and dropped;
  - test categories unseen in training, and columns missing from test, are handled.
- **Data profile:** computed on the raw table with pandas semantics, and written within about 1 s, before any modelling.
- **Never fail, never time out:**
  - the budget clock starts before the heavy imports;
  - a constant prediction is on disk in under 1 s, then a quick real model in a few seconds, then better models replace it atomically;
  - gradient-boosting fits run in interruptible chunks against a deadline;
  - a fold planner reserves time assuming a CPU twice as slow;
  - a watchdog exits at 52 s with the best answer already written.
- Threads are capped, seeds are fixed (output is deterministic), and imports are lazy.

### `A_lean` — one strong model per task (submitted)
- **Classification:**
  - HistGradientBoosting with native categoricals and missing values;
  - square-root balanced class weights when classes are imbalanced, plus per-class decision weights tuned out-of-fold to maximise macro-F1;
  - 5-fold CV with fold models averaged;
  - an ExtraTrees blend for 5 or more classes, and a logistic-regression blend for small tables, with blend weights chosen out-of-fold from a grid that includes 0;
  - a fast configuration with 3 folds when there are 10 or more classes.
- **Regression:** HistGradientBoosting on the raw or log1p target, chosen out-of-fold. log1p is kept when it is within 0.005 R² of raw, which also protects us if the metric is RMSLE. Predictions are clipped.
- **Clustering:**
  - standardised features, then sparse k-means feature weights;
  - a feature is dropped only if a permutation test confirms it is irrelevant;
  - four cheap candidates on the kept features (KMeans, Ward, diagonal GMM on the continuous columns, full GMM on PCA-whitened data);
  - the partition the candidates agree with most is returned.

### The two alternatives (kept in `approaches/`; each also runs standalone)
- **`B_portfolio`:** a size-determined portfolio of HistGB variants, linear models, ExtraTrees, RandomForest and kNN on shared folds. A greedy (Caruana) ensemble is built on out-of-fold predictions. Clustering is a consensus over preprocessing views × algorithms.
- **`C_rules`:** profile-driven rules (an "expert system") choose the pipeline.
  - A single linearity test is the only measurement.
  - Categoricals with more than 30 levels get cross-fitted target encoding.
  - Datetimes get cyclic features.
  - Clustering uses rule-gated candidates in a sparse-weighted space.

To submit a different solver, change `SELECTED` in `main.py`.

---

## Results

We had no hidden data, so we acted as the grader ourselves. Over four rounds we built 72 public datasets; each round used fresh, non-overlapping tables. They include the five spec examples:
- Telco churn and AI4I 2020 (classification);
- Ames House Prices and Bike Sharing (regression);
- Mall Customers (clustering).

Every run was a subprocess with 2 threads, a 60 s kill and a warm-up. Metrics:
- classification: macro-F1, plus accuracy and balanced accuracy;
- regression: R², plus RMSE and RMSLE;
- clustering: ARI, plus NMI.

Scores were converted to the spec's 85 / 10 / 5 points with our own normalisation, explained in `docs/04_validation_report.md` §2.

Mean primary metric over the first 54 datasets (18 per task) for the round-2 code (98f5fe1), against plain scikit-learn defaults. Rounds 3 and 4 then made the solvers safer under the time limit while keeping quality (docs/09, docs/10).

| Task | A_lean | B_portfolio | C_rules | Default (HistGB / scaled KMeans) |
|---|---|---|---|---|
| Classification macro-F1 | 0.776 | 0.773 | 0.771 | 0.751 |
| Regression R² | 0.841 | 0.844 | 0.841 | 0.824 |
| Clustering ARI | 0.613 | 0.614 | 0.612 | 0.470 |

**Final selection** (round 4: 18 unseen datasets; estimated points out of 100 under our scoring model):

| | A_lean | C_rules | B_portfolio |
|---|---|---|---|
| Round 4 total | 69.30 | 69.43 | 69.03 |
| Cumulative over all 72 datasets | **291.10** | 291.06 | 289.88 |
| Five spec example datasets | **22.05** | 21.88 | 21.46 |
| Max run time, 2 threads / 1 thread | **14.3 s / 18.0 s** | 27.6 s / 26.7 s | 25.8 s / 33.6 s |

- **Quality:** the three solvers are within noise of each other.
- **Why A:** it was chosen for the best cumulative and spec-example scores, about twice the time headroom of B and C, and deterministic, always-valid output in all four rounds.
- **Spec examples with A:** Telco macro-F1 0.727, AI4I 0.872, House Prices R² 0.895, Bike Sharing R² 0.960, Mall ARI 0.681.

Validity and timing:
- No crash, timeout or invalid output in any scored run.
- Profiling scored 10/10 on our checks.
- Maximum run time is about 20–28 s at 1 thread, and every run finished cleanly in a simulated 2.5× slower CPU.

The full reports are in `docs/`:
- `04`, `06`, `08`, `10`: validation rounds 1–4;
- `05`, `07`, `09`: fix changelogs.

---

## Assumptions
No questions were allowed and no kickoff material was available to us, so we assumed the following:
1. **Input:** each dataset is a folder with a train CSV, a test CSV (clustering may have one CSV) and a metadata JSON giving the task, target, feature types and, for clustering, k. If metadata is missing, we infer it.
2. **Output:** `predictions.csv`, with one prediction per test row in the original order, and `profile.json`.
3. **Classification metric:** imbalance-aware (macro-F1 or balanced accuracy). We optimise macro-F1 while staying strong on accuracy.
4. **Regression metric:** R² / RMSE-like. Skewed non-negative targets may use log1p, which also helps if RMSLE is used.
5. **Clustering:** scored against hidden labels with ARI / NMI, with k given. We always return exactly k labels 0..k-1.
6. **Profile:** ground truth is computed on the raw training table with pandas semantics. Declared feature types take precedence over inference.
7. **Time limit:** the 60 s includes Python start-up and imports, and the grader may have as few as 2 cores. We target 40 s internally and cap threads.
8. **Environment:** there is no internet at assessment time and only the pinned packages are available.
9. **Features:** columns present in both train and test are features. ID-like columns are excluded.
10. **Datasets:** the hidden datasets are similar in size to the Kaggle examples (a few hundred to tens of thousands of rows, up to ~100 columns).
11. **Clustering ground truth for Mall:** this was our own validation choice. The Kaggle Mall file has no labels, so we derived the 5 classic income/spending segments.
12. **Leakage in AI4I:** this was also our own validation choice. AI4I's five failure-mode flags leak the target, so we dropped them when validating. Our program itself drops nothing that is present in both train and test.

## Repository layout
- `main.py`, `requirements.txt`: the standard entry point and pinned dependencies of the submission.
- `approaches/A_lean`, `approaches/B_portfolio`, `approaches/C_rules`: three self-contained solvers with the same command line.
- `docs/`: research report, assumed contract, validation reports and fix changelogs.
- `tools/`: dataset builder, scorer and evaluation runners.
- `validation/`: suite builders, harnesses and result CSVs. Datasets are not committed; the builders download them.
