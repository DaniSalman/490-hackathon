"""Score one output folder against <dataset>/hidden/test_labels.csv and validate the output format.

    py tools/score.py --data <dataset_dir> --out <out_dir>     -> prints one JSON line

Metrics: classification macro_f1 / balanced_accuracy / accuracy; regression r2 / rmse / rmsle;
clustering ari / nmi. Also checks row count, label domain and (clustering) the number of clusters.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def score(data_dir, out_dir):
    from sklearn import metrics as M
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    info_p = os.path.join(data_dir, "hidden", "info.json")
    if os.path.isfile(info_p):
        with open(info_p, encoding="utf-8") as f:
            meta = {**meta, **{k: v for k, v in json.load(f).items() if v is not None}}
    task = meta["task"]
    res = {"dataset": meta.get("name", os.path.basename(data_dir)), "task": task, "valid": False}
    pred_p = os.path.join(out_dir, "predictions.csv")
    prof_p = os.path.join(out_dir, "profile.json")
    res["has_profile"] = os.path.isfile(prof_p)
    if not os.path.isfile(pred_p):
        res["error"] = "missing predictions.csv"
        return res
    pr = pd.read_csv(pred_p, low_memory=False)
    lab_p = os.path.join(data_dir, "hidden", "test_labels.csv")
    if not os.path.isfile(lab_p):
        res["error"] = "no hidden labels"
        return res
    gt = pd.read_csv(lab_p, low_memory=False)
    if "prediction" not in pr.columns or len(pr) != len(gt):
        res["error"] = f"bad shape: {pr.shape} vs {len(gt)} labels"
        return res
    if "row_index" in pr.columns and not np.array_equal(pr["row_index"].to_numpy(), gt["row_index"].to_numpy()):
        res["error"] = "row order mismatch"
        return res
    y, p = gt["label"], pr["prediction"]
    if p.isna().any():
        res["error"] = "NaN predictions"
        return res
    if task == "classification":
        ys, ps = y.astype(str), p.astype(str)
        # normalise 1 vs 1.0 style differences
        try:
            ys = pd.to_numeric(y).astype(float).astype(str)
            ps = pd.to_numeric(p).astype(float).astype(str)
        except (ValueError, TypeError):
            pass
        res["unknown_labels"] = int((~ps.isin(set(ys))).sum())
        res["macro_f1"] = float(M.f1_score(ys, ps, average="macro"))
        res["balanced_accuracy"] = float(M.balanced_accuracy_score(ys, ps))
        res["accuracy"] = float(M.accuracy_score(ys, ps))
    elif task == "regression":
        yv, pv = y.astype(float).to_numpy(), p.astype(float).to_numpy()
        if not np.all(np.isfinite(pv)):
            res["error"] = "non-finite predictions"
            return res
        res["r2"] = float(M.r2_score(yv, pv))
        res["rmse"] = float(np.sqrt(np.mean((yv - pv) ** 2)))
        if yv.min() >= 0:
            res["rmsle"] = float(np.sqrt(np.mean((np.log1p(yv) - np.log1p(np.clip(pv, 0, None))) ** 2)))
    else:
        k = int(meta["n_clusters"])
        res["n_clusters_pred"] = int(p.nunique())
        res["k_ok"] = res["n_clusters_pred"] == min(k, len(p))
        res["ari"] = float(M.adjusted_rand_score(y, p))
        res["nmi"] = float(M.normalized_mutual_info_score(y, p))
    res["valid"] = True
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(score(a.data, a.out)))
