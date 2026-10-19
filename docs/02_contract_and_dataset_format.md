# 02 — Assumed I/O contract and dataset folder format

This is **our assumption** of the kickoff contract (the real one has not been released). All three
approaches implement exactly this contract through their own copy of `io_contract.py`. When the official
contract appears, only `io_contract.py` (and `map_profile` inside it) needs to change.

The Validator should build every dataset in this format. `tools/make_dataset.py` does it for you.

---

## 1. Dataset folder

### 1.1 Supervised (classification / regression)

```
<dataset_dir>/
  train.csv          features + target column (header row, comma-separated, UTF-8)
  test.csv           same feature columns, WITHOUT the target column
  metadata.json      see 1.3
  hidden/            ground truth for scoring; solvers never read it
    test_labels.csv  row_index,[<id_column>,]label   (one row per test.csv row, same order)
    info.json        {"task": ..., "target": ..., "n_clusters": ...}  (for the scorer)
```

### 1.2 Clustering — two accepted variants

```
Variant 1 (default, single CSV):           Variant 2 (train/test):
<dataset_dir>/                             <dataset_dir>/
  data.csv        features only              train.csv   features only
  metadata.json                              test.csv    features only
  hidden/test_labels.csv  (one per row       metadata.json
         of data.csv)                        hidden/test_labels.csv (one per test.csv row)
```
- Variant 1: the solver clusters all rows of `data.csv` and predicts a label for each of them.
- Variant 2: the solver clusters the union of train and test rows, but writes labels **only for the
  test rows** (test.csv order).
- If a clustering `train.csv`/`data.csv` contains the label column and `metadata.target` names it, the
  solver drops it. Do not rely on this; keep labels in `hidden/`.

### 1.3 `metadata.json`

| key | type | required | allowed values / meaning |
|---|---|---|---|
| `task` | string | recommended | `"classification"`, `"regression"`, `"clustering"` (aliases also accepted: `binary`, `multiclass`, `reg`, `cluster`, `unsupervised`, ...). If absent, the solver infers it (no target -> clustering; numeric target with > 20 unique values -> regression; else classification). |
| `target` | string | supervised: yes | name of the target column in `train.csv`. If absent: the single column in train but not in test; else the last column. |
| `n_clusters` | int | clustering: yes | required number of groups k. If absent, the solver picks k in 2..8 by silhouette. |
| `feature_types` | object | optional | `{column: type}` with type in `"numeric"`, `"categorical"`, `"boolean"`, `"datetime"`, `"text"`, `"id"` (aliases: `numerical`, `continuous`, `cat`, `nominal`, `ordinal`, `date`, ...). Also accepted: `{"numeric": [cols], "categorical": [cols]}` or `[{"name": col, "type": t}]`. Declared types override inference (except: constant columns are always dropped; `"id"` columns are always dropped). |
| `id_column` | string | optional | row identifier present in train/test; excluded from features and copied into `predictions.csv`. |
| `name` | string | optional | dataset name (default: folder name). |
| `metric` | string | optional | recorded only; not used for optimisation yet. |

Key lookup is case-insensitive with aliases (`task_type`, `problem_type`, `target_column`, `label`, `k`,
`num_clusters`, `column_types`, ...). Metadata file names tried: `metadata.json`, `meta.json`,
`task.json`, `info.json`, `dataset.json`, else the first `*.json` in the folder.

### 1.4 Table contents the solvers handle
Missing values (empty cells, `NA`, `?`, `null`, `none`, `-`, blanks), numeric columns stored as strings
(`"1,234"`, `"$12"`, `"12%"`, `" "`), yes/no/true/false booleans, datetime strings, free text, IDs,
constant / duplicate / all-NaN columns, high-cardinality categoricals, test categories unseen in train,
test files missing a train column (filled with NaN) or with extra columns (ignored). Targets may be
strings, ints, floats or booleans; training rows with a missing target are dropped.

---

## 2. Invocation (identical for A, B and C)

```
py approaches/<A_lean|B_portfolio|C_rules>/main.py --data <dataset_dir> --out <out_dir>
py approaches/<X>/main.py --train <train.csv> --test <test.csv> --meta <metadata.json> --out <out_dir>
```
Options: `--time-budget S` (internal soft budget from process start, default 40), `--hard-limit S`
(watchdog exit, default budget + 12 = 52), `--n-jobs N` (thread cap; else `HACK_N_JOBS`, else
`OMP_NUM_THREADS`, else min(4, cores)), `--quiet`.
Exit code 0 on success (also when the watchdog fires — predictions are already on disk). A non-zero
exit only happens if the input cannot be read at all.

To simulate a weak grader: set `OMP_NUM_THREADS=2 HACK_N_JOBS=2` (what `tools/run_smoke.py` does).

---

## 3. Outputs (in `<out_dir>/`)

### 3.1 `predictions.csv`
```
row_index,[<id_column>,]prediction
0,...
```
- One row per row to predict (test.csv rows; for clustering variant 1 the rows of data.csv), same order.
- `row_index` = 0-based position in that file. `<id_column>` only when metadata defines `id_column`.
- classification: original label values (e.g. `Yes`/`No`, `0`/`1`, class names; integer-valued float
  labels such as `1.0` are written as `1`).
- regression: float.
- clustering: int in `0..k-1`, exactly `min(k, n)` distinct values.

### 3.2 `profile.json`
Structured data profile computed on the raw training table (pandas semantics). Top-level keys:
`schema_version, dataset_name, task, task_source, target, id_column, n_rows, n_rows_train, n_rows_test,
n_rows_predicted, n_columns, n_features, n_duplicate_rows, total_missing_cells, missing_cell_ratio,
n_columns_with_missing, column_type_counts, constant_columns, duplicate_columns ([original, copy] pairs),
id_like_columns, high_missing_columns, high_cardinality_columns, datetime_columns, columns (per-column
records), target_profile, warnings, test_missing_cells, preparation, modelling`.
Per-column record: `name, role (feature|target|id), raw_dtype, declared_type, inferred_type, type,
n_missing, missing_ratio, n_unique, unique_ratio, is_constant, is_id_like, duplicate_of, numeric_stats
| top_values, unseen_test_levels, n_test_rows_unseen_level, action`.
`modelling` records the approach, chosen model, CV scores, decisions and elapsed time.

---

## 4. Tools

| tool | purpose |
|---|---|
| `tools/make_dataset.py` | DataFrame (or CSV) + task info -> dataset folder (stratified train/test split for classification, `hidden/` ground truth). Library: `make_dataset(df, out_dir, task, target, n_clusters=None, id_column=None, feature_types="auto"\|None\|dict, test_size=0.25, seed=0, single_csv=True)`. CLI: `py tools/make_dataset.py --csv f.csv --task regression --target y --out validation/data/x` (`--no-feature-types`, `--split-clustering`, `--id-column`). |
| `tools/score.py` | validates one output folder and prints JSON metrics: macro_f1 / balanced_accuracy / accuracy; r2 / rmse / rmsle; ari / nmi / k_ok. |
| `tools/run_smoke.py` | runs one approach as a subprocess on every dataset folder of a directory with capped threads, records wall time (incl. start-up), validity and metrics -> `runs/<approach>/summary.json`. |
| `tools/build_dev_datasets.py` | builds the dev/smoke datasets into `dev_data/` (see docs/03). |

`feature_types="auto"` declares numeric dtypes as `numeric` and everything else as `categorical`
(so e.g. a numeric-as-string column is declared categorical — the declared type wins). Use
`feature_types=None` to test inference, or pass an explicit dict for realistic declarations.
Keep generated data under `validation/data/` (gitignored).
