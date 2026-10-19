"""Convert a pandas DataFrame + task info into the assumed dataset folder format.

Library use:
    from tools.make_dataset import make_dataset
    make_dataset(df, out_dir="validation/data/telco", task="classification", target="Churn",
                 id_column="customerID", test_size=0.25, seed=0)

CLI use:
    py tools/make_dataset.py --csv some.csv --task regression --target SalePrice --out validation/data/house

Folder written (see docs/02_contract_and_dataset_format.md):
    <out>/train.csv, <out>/test.csv, <out>/metadata.json          supervised
    <out>/data.csv, <out>/metadata.json                             clustering (single_csv=True, default)
    <out>/train.csv, <out>/test.csv, <out>/metadata.json          clustering (single_csv=False)
    <out>/hidden/test_labels.csv   ground truth for scoring (row_index, label) -- solvers never read it
    <out>/hidden/info.json         true task / target / n_clusters (for the scorer)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def infer_feature_types(df):
    """Simple declared-type generator: numeric for numeric dtypes, datetime for datetime dtypes, else categorical."""
    ft = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            ft[c] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            ft[c] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            ft[c] = "datetime"
        else:
            ft[c] = "categorical"
    return ft


def make_dataset(df, out_dir, task, target=None, n_clusters=None, id_column=None, feature_types="auto",
                 test_size=0.25, seed=0, single_csv=True, name=None, extra_meta=None, stratify=True):
    """Write a dataset folder. Returns the metadata dict.

    feature_types: "auto" (from dtypes), None (omit from metadata), or an explicit {col: type} dict.
    For clustering, `target` is the hidden ground-truth label column; it is removed from the data and
    written to hidden/test_labels.csv. n_clusters defaults to its number of distinct values.
    """
    assert task in ("classification", "regression", "clustering")
    os.makedirs(os.path.join(out_dir, "hidden"), exist_ok=True)
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    feats = df.drop(columns=[target]) if target is not None else df
    meta = {"name": name or os.path.basename(os.path.normpath(out_dir)), "task": task}
    if feature_types == "auto":
        ft = infer_feature_types(feats.drop(columns=[id_column]) if id_column else feats)
    elif feature_types is None:
        ft = None
    else:
        ft = dict(feature_types)
    if id_column:
        meta["id_column"] = id_column
        if ft is not None:
            ft.pop(id_column, None)

    if task == "clustering":
        labels = df[target].to_numpy() if target is not None else None
        if n_clusters is None:
            assert labels is not None, "clustering needs n_clusters or a label column"
            n_clusters = int(pd.Series(labels).nunique())
        meta["n_clusters"] = int(n_clusters)
        if single_csv:
            feats.to_csv(os.path.join(out_dir, "data.csv"), index=False)
            lab_rows = np.arange(len(df))
        else:
            idx = rng.permutation(len(df))
            n_te = int(round(len(df) * test_size))
            te, tr = np.sort(idx[:n_te]), np.sort(idx[n_te:])
            feats.iloc[tr].to_csv(os.path.join(out_dir, "train.csv"), index=False)
            feats.iloc[te].to_csv(os.path.join(out_dir, "test.csv"), index=False)
            lab_rows = te
        if labels is not None:
            pd.DataFrame({"row_index": np.arange(len(lab_rows)), "label": labels[lab_rows]}).to_csv(
                os.path.join(out_dir, "hidden", "test_labels.csv"), index=False)
    else:
        meta["target"] = target
        y = df[target]
        n = len(df)
        n_te = int(round(n * test_size))
        if task == "classification" and stratify:
            from sklearn.model_selection import train_test_split
            vc = y.value_counts()
            strat = y.where(y.map(vc) >= 2, vc.index[0]).astype(str) if (vc >= 2).any() else None
            tr, te = train_test_split(np.arange(n), test_size=n_te, random_state=seed, stratify=strat)
        else:
            idx = rng.permutation(n)
            te, tr = idx[:n_te], idx[n_te:]
        tr, te = np.sort(tr), np.sort(te)
        df.iloc[tr].to_csv(os.path.join(out_dir, "train.csv"), index=False)
        df.iloc[te].drop(columns=[target]).to_csv(os.path.join(out_dir, "test.csv"), index=False)
        out = pd.DataFrame({"row_index": np.arange(len(te))})
        if id_column:
            out[id_column] = df[id_column].iloc[te].to_numpy()
        out["label"] = y.iloc[te].to_numpy()
        out.to_csv(os.path.join(out_dir, "hidden", "test_labels.csv"), index=False)
    with open(os.path.join(out_dir, "hidden", "info.json"), "w", encoding="utf-8") as f:
        json.dump({"task": task, "target": target, "n_clusters": meta.get("n_clusters")}, f, indent=2)
    if ft is not None:
        meta["feature_types"] = ft
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", required=True, choices=["classification", "regression", "clustering"])
    ap.add_argument("--target")
    ap.add_argument("--n-clusters", type=int)
    ap.add_argument("--id-column")
    ap.add_argument("--no-feature-types", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-clustering", action="store_true", help="clustering: write train/test instead of data.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False)
    m = make_dataset(df, a.out, a.task, a.target, a.n_clusters, a.id_column,
                     None if a.no_feature_types else "auto", a.test_size, a.seed, not a.split_clustering)
    print(json.dumps(m)[:300])


if __name__ == "__main__":
    _main()
