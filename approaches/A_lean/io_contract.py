"""I/O contract adapter -- the ONLY module that knows the dataset and output formats.

When the official contract is released, adapt this file (and, if needed, the key mapping in
`map_profile`). Everything else consumes the `Task` object and returns predictions + a profile.

Assumed dataset folder (see docs/02_contract_and_dataset_format.md):
    <dir>/train.csv       features + target column (supervised)
    <dir>/test.csv        features only (supervised; optional for clustering)
    <dir>/data.csv        single table for clustering (alternative to train.csv)
    <dir>/metadata.json   {"task", "target", "feature_types", "n_clusters", "id_column", ...}
Outputs in <out>/: predictions.csv (row_index[, <id_column>], prediction) and profile.json.
"""
import json
import math
import os

import numpy as np
import pandas as pd

TASK_ALIASES = {
    "classification": "classification", "classify": "classification", "class": "classification",
    "binary": "classification", "multiclass": "classification", "binary_classification": "classification",
    "multiclass_classification": "classification", "clf": "classification",
    "regression": "regression", "regress": "regression", "reg": "regression",
    "clustering": "clustering", "cluster": "clustering", "unsupervised": "clustering",
    "segmentation": "clustering",
}
META_KEYS = {
    "task": ["task", "task_type", "problem_type", "type", "problem"],
    "target": ["target", "target_column", "target_col", "label", "label_column", "y", "target_name"],
    "n_clusters": ["n_clusters", "k", "num_clusters", "n_groups", "clusters", "number_of_clusters",
                   "n_cluster", "num_groups"],
    "feature_types": ["feature_types", "column_types", "features", "types", "feature_type"],
    "id_column": ["id_column", "id_col", "id", "index_column", "row_id"],
    "name": ["name", "dataset", "dataset_name", "dataset_id"],
    "metric": ["metric", "scoring", "evaluation_metric"],
}
META_FILES = ["metadata.json", "meta.json", "task.json", "info.json", "dataset.json"]
TRAIN_FILES = ["train.csv", "data.csv", "dataset.csv", "train.tsv"]
TEST_FILES = ["test.csv", "test.tsv"]
TYPE_ALIASES = {
    "numeric": "numeric", "numerical": "numeric", "number": "numeric", "continuous": "numeric",
    "float": "numeric", "int": "numeric", "integer": "numeric", "real": "numeric", "num": "numeric",
    "categorical": "categorical", "category": "categorical", "cat": "categorical", "nominal": "categorical",
    "ordinal": "categorical", "string": "categorical", "object": "categorical", "discrete": "categorical",
    "boolean": "boolean", "bool": "boolean", "binary": "boolean",
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "text": "text", "id": "id", "identifier": "id",
}


class Task:
    """Normalised view of one dataset."""

    def __init__(self, **kw):
        self.name = "dataset"
        self.task_type = None          # classification | regression | clustering
        self.task_source = "metadata"  # metadata | inferred
        self.target = None
        self.train_df = None           # raw train table (incl. target)
        self.test_df = None            # raw test table or None
        self.X_train = None            # features only
        self.y_train = None            # raw target Series (supervised)
        self.X_test = None             # features of the rows we must predict (None => predict train rows)
        self.test_ids = None
        self.id_column = None
        self.feature_types = {}        # col -> normalised declared type
        self.n_clusters = None
        self.metric = None
        self.meta = {}
        self.paths = {}
        self.__dict__.update(kw)

    @property
    def n_predict(self):
        return len(self.X_test) if self.X_test is not None else len(self.X_train)


# --------------------------------------------------------------------------- loading
def _first_existing(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.isfile(p):
            return p
    return None


def _get(meta, key):
    for k in META_KEYS[key]:
        if k in meta and meta[k] not in (None, ""):
            return meta[k]
    lower = {str(k).lower(): v for k, v in meta.items()}
    for k in META_KEYS[key]:
        if k in lower and lower[k] not in (None, ""):
            return lower[k]
    return None


def _read_table(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")


def _norm_feature_types(ft, columns):
    out = {}
    if not ft:
        return out
    if isinstance(ft, dict):
        vals = list(ft.values())
        if vals and all(isinstance(v, (list, tuple)) for v in vals):  # {"numeric": [cols], ...}
            for t, cols in ft.items():
                for c in cols:
                    out[str(c)] = TYPE_ALIASES.get(str(t).lower().rstrip("s"), TYPE_ALIASES.get(str(t).lower()))
        else:                                                     # {col: type}
            for c, t in ft.items():
                out[str(c)] = TYPE_ALIASES.get(str(t).lower())
    elif isinstance(ft, list):                                    # [{"name":..,"type":..}]
        for item in ft:
            if isinstance(item, dict):
                c = item.get("name", item.get("column"))
                t = item.get("type", item.get("dtype"))
                if c is not None and t is not None:
                    out[str(c)] = TYPE_ALIASES.get(str(t).lower())
    cols = set(map(str, columns))
    return {c: t for c, t in out.items() if t is not None and c in cols}


def _resolve_col(name, columns):
    if name is None:
        return None
    if isinstance(name, int) and not isinstance(name, bool) and 0 <= name < len(columns) and name not in columns:
        return columns[name]
    if name in columns:
        return name
    low = {str(c).lower(): c for c in columns}
    return low.get(str(name).lower())


def load_task(data=None, train=None, test=None, meta=None):
    folder = data
    train_p = train or (folder and _first_existing(folder, TRAIN_FILES))
    test_p = test or (folder and _first_existing(folder, TEST_FILES))
    meta_p = meta or (folder and _first_existing(folder, META_FILES))
    if meta_p is None and folder:
        js = sorted(f for f in os.listdir(folder) if f.lower().endswith(".json"))
        meta_p = os.path.join(folder, js[0]) if js else None
    if train_p is None:
        raise FileNotFoundError("no training table found")
    md = {}
    if meta_p and os.path.isfile(meta_p):
        with open(meta_p, "r", encoding="utf-8") as f:
            md = json.load(f) or {}
    train_df = _read_table(train_p)
    train_df.columns = [str(c) for c in train_df.columns]
    test_df = _read_table(test_p) if test_p and os.path.isfile(test_p) else None
    if test_df is not None:
        test_df.columns = [str(c) for c in test_df.columns]

    t = Task(meta=md, paths={"train": train_p, "test": test_p, "meta": meta_p})
    t.name = str(_get(md, "name") or (os.path.basename(os.path.normpath(folder)) if folder else "dataset"))
    t.metric = _get(md, "metric")
    cols = list(train_df.columns)
    target = _resolve_col(_get(md, "target"), cols)
    raw_task = _get(md, "task")
    task_type = TASK_ALIASES.get(str(raw_task).lower().strip()) if raw_task is not None else None
    k = _get(md, "n_clusters")
    try:
        k = int(k) if k is not None else None
    except (TypeError, ValueError):
        k = None

    if task_type is None:  # infer
        t.task_source = "inferred"
        if target is None and test_df is not None:
            extra = [c for c in cols if c not in set(test_df.columns)]
            target = extra[-1] if len(extra) == 1 else None
        if target is None or k is not None:
            task_type = "clustering"
        else:
            y = train_df[target]
            if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 20:
                task_type = "regression"
            else:
                task_type = "classification"
    if task_type != "clustering" and target is None:
        if test_df is not None:
            extra = [c for c in cols if c not in set(test_df.columns)]
            target = extra[-1] if extra else cols[-1]
        else:
            target = cols[-1]

    t.task_type, t.target, t.n_clusters = task_type, target, k
    t.train_df, t.test_df = train_df, test_df
    id_col = _resolve_col(_get(md, "id_column"), cols)
    if id_col is None and test_df is not None:
        id_col = _resolve_col(_get(md, "id_column"), list(test_df.columns))
    t.id_column = id_col

    drop = [c for c in (target, id_col) if c is not None]
    X_train = train_df.drop(columns=[c for c in drop if c in train_df.columns])
    if task_type == "clustering":
        t.y_train = None
        if test_df is not None:
            X_test = test_df.drop(columns=[c for c in drop if c in test_df.columns])
            X_test = X_test.reindex(columns=X_train.columns)
            t.test_ids = test_df[id_col] if id_col in test_df.columns else None
        else:
            X_test = None
            t.test_ids = train_df[id_col] if id_col in train_df.columns else None
    else:
        t.y_train = train_df[target]
        if test_df is None:
            raise FileNotFoundError("supervised task without test table")
        X_test = test_df.drop(columns=[c for c in drop if c in test_df.columns])
        X_test = X_test.reindex(columns=X_train.columns)   # align; missing columns -> NaN
        t.test_ids = test_df[id_col] if id_col in test_df.columns else None
    t.X_train, t.X_test = X_train, X_test
    t.feature_types = _norm_feature_types(_get(md, "feature_types"), X_train.columns)
    return t


# --------------------------------------------------------------------------- writing
def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def write_predictions(task, preds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    preds = np.asarray(preds)
    df = pd.DataFrame({"row_index": np.arange(len(preds))})
    if task.id_column is not None and task.test_ids is not None and len(task.test_ids) == len(preds):
        df[task.id_column] = np.asarray(task.test_ids)
    df["prediction"] = preds
    _atomic_write_text(os.path.join(out_dir, "predictions.csv"), df.to_csv(index=False))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _clean_nans(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {str(k): _clean_nans(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_nans(v) for v in o]
    if isinstance(o, np.floating):
        return _clean_nans(float(o))
    return o


def map_profile(profile):
    """Hook for renaming/subsetting profile keys to the official schema (identity for now)."""
    return profile


def write_profile(profile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    txt = json.dumps(_clean_nans(map_profile(profile)), indent=2, default=_json_default, allow_nan=False)
    _atomic_write_text(os.path.join(out_dir, "profile.json"), txt)
