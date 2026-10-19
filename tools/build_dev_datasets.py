"""Build the DEVELOPMENT / smoke-test datasets (synthetic + 1-2 sklearn built-ins) into dev_data/.

These are the only datasets used while developing approaches A, B and C (see docs/03_dev_datasets_used.md).
The Validator should score on different datasets.

    py tools/build_dev_datasets.py [--out dev_data]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dataset import make_dataset  # noqa: E402


def _inject_missing(df, cols, frac, rng):
    for c in cols:
        m = rng.rand(len(df)) < frac
        df.loc[m, c] = np.nan
    return df


def c_synth_imb(rng):
    from sklearn.datasets import make_classification
    X, y = make_classification(3000, 12, n_informative=6, n_redundant=2, weights=[0.93, 0.07],
                               flip_y=0.01, random_state=1)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(12)])
    df["x3"] *= 1000.0
    df["region"] = np.where(y == 1, rng.choice(["north", "south", "east"], len(y), p=[.6, .2, .2]),
                            rng.choice(["north", "south", "east"], len(y), p=[.2, .4, .4]))
    df["constant"] = 7
    df["x0_copy"] = df["x0"]
    df["customerID"] = [f"C{100000 + i}" for i in range(len(df))]
    df = _inject_missing(df, ["x1", "x2", "region"], 0.1, rng)
    df["churn"] = np.where(y == 1, "yes", "no")
    return dict(df=df, task="classification", target="churn", id_column="customerID")


def c_multi_nonlinear(rng):
    n = 6000
    a, b = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    r = np.sqrt(a ** 2 + b ** 2)
    ang = np.arctan2(b, a)
    y = (np.digitize(r, [1.2, 2.2]) * 2 + (np.sin(3 * ang) > 0)).astype(int)  # 6 classes
    df = pd.DataFrame({"a": a, "b": b})
    for i in range(6):
        df[f"noise{i}"] = rng.randn(n)
    df["store"] = [f"S{v}" for v in rng.randint(0, 300, n)]
    start = pd.Timestamp("2021-01-01")
    df["when"] = (start + pd.to_timedelta(rng.randint(0, 700 * 24, n), unit="h")).strftime("%Y-%m-%d %H:%M:%S")
    names = np.array(["alpha", "beta", "gamma", "delta", "eps", "zeta"])
    df["label"] = names[y]
    return dict(df=df, task="classification", target="label")


def c_tiny(rng):
    n = 80
    y = rng.randint(0, 3, n)
    df = pd.DataFrame({"f1": y + rng.randn(n) * 0.8, "f2": rng.randn(n), "f3": (y == 2) * 1.5 + rng.randn(n),
                       "f4": rng.choice(["a", "b"], n)})
    df["cls"] = y
    return dict(df=df, task="classification", target="cls")


def c_breast_cancer(rng):
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df["target"] = np.where(df["target"] == 1, "benign", "malignant")
    return dict(df=df, task="classification", target="target")


def c_edge(rng):
    n = 240
    y = np.where(rng.rand(n) < 0.3, "Yes", "No").astype(object)
    y[0] = "Maybe"                                   # class with a single member
    df = pd.DataFrame({
        "tenure": rng.randint(0, 72, n),
        "TotalCharges": [(" " if rng.rand() < 0.05 else f"{v:.2f}") for v in rng.uniform(20, 8000, n)],
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Paperless": rng.choice(["Yes", "No"], n),
        "all_nan": np.nan,
        "flag": rng.rand(n) < 0.5,
        "same": "x",
    })
    df.loc[y == "Yes", "tenure"] = rng.randint(0, 20, (y == "Yes").sum())
    df["Churn"] = y
    return dict(df=df, task="classification", target="Churn", post=_edge_post)


def _edge_post(out_dir):
    te = pd.read_csv(os.path.join(out_dir, "test.csv"))
    te.loc[te.index[:5], "Contract"] = "Three year"   # unseen test category
    te = te.drop(columns=["flag"])                   # column missing from test
    te.to_csv(os.path.join(out_dir, "test.csv"), index=False)


def c_large(rng):
    from sklearn.datasets import make_classification
    X, y = make_classification(60000, 40, n_informative=15, n_classes=3, random_state=3)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(40)])
    df["grp"] = rng.choice(list("ABCDEFGH"), len(df))
    df["y"] = y
    return dict(df=df, task="classification", target="y")


def r_friedman(rng):
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(2000, 10, noise=1.0, random_state=4)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
    df = _inject_missing(df, ["x0", "x3"], 0.15, rng)
    df["dup"] = df["x5"]
    df["target"] = y
    return dict(df=df, task="regression", target="target")


def r_counts_datetime(rng):
    n = 5000
    ts = pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.randint(0, 2 * 365 * 24, n), unit="h")
    hour, dow, month = ts.hour, ts.dayofweek, ts.month
    temp = 15 + 10 * np.sin(2 * np.pi * (month - 4) / 12) + rng.randn(n) * 3
    weather = rng.choice([1, 2, 3, 4], n, p=[.6, .25, .13, .02])
    peak = np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 17.5) ** 2) / 5)
    lam = np.exp(2.0 + 2.2 * peak * (dow < 5) + 0.8 * np.exp(-((hour - 14) ** 2) / 10) * (dow >= 5)
                 + 0.04 * temp - 0.4 * (weather - 1) + 0.3 * (ts.year - 2011))
    df = pd.DataFrame({"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"), "season": (month % 12) // 3 + 1,
                       "weather": weather, "temp": temp.round(2), "humidity": rng.randint(20, 100, n),
                       "count": rng.poisson(lam)})
    return dict(df=df, task="regression", target="count",
                feature_types={"datetime": "datetime", "season": "categorical", "weather": "categorical",
                               "temp": "numeric", "humidity": "numeric"})


def r_diabetes(rng):
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True, scaled=False)
    return dict(df=d.frame.copy(), task="regression", target="target")


def r_tiny(rng):
    n = 40
    x = rng.randn(n, 3)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df["y"] = 3 * x[:, 0] - 2 * x[:, 1] + rng.randn(n) * 0.5
    return dict(df=df, task="regression", target="y", post=_drop_task_key, feature_types=None)


def _drop_task_key(out_dir):
    """Metadata without 'task' and 'feature_types' -> exercises task inference."""
    p = os.path.join(out_dir, "metadata.json")
    with open(p) as f:
        m = json.load(f)
    m.pop("task", None)
    with open(p, "w") as f:
        json.dump(m, f, indent=2)


def r_prices_strings(rng):
    n = 1500
    area = rng.lognormal(7, 0.4, n)
    qual = rng.randint(1, 11, n)
    nb = rng.choice([f"NB{i}" for i in range(25)], n)
    nb_eff = {f"NB{i}": v for i, v in enumerate(rng.normal(0, 0.25, 25))}
    price = np.exp(10.5 + 0.6 * np.log(area) / 7 * 7 * 0.1 + 0.12 * qual + np.vectorize(nb_eff.get)(nb)
                   + rng.randn(n) * 0.12)
    df = pd.DataFrame({"Id": np.arange(1, n + 1), "LotArea": [f"{v:,.0f}" for v in area], "Quality": qual,
                       "Neighborhood": nb, "Garage": rng.choice(["Attchd", "Detchd", None], n),
                       "YearBuilt": rng.randint(1900, 2010, n), "SalePrice": price.round(0)})
    return dict(df=df, task="regression", target="SalePrice", id_column="Id", feature_types=None,
                extra_meta={})


def k_blobs_noise(rng):
    from sklearn.datasets import make_blobs
    centers = [[20, 80], [20, 20], [55, 50], [90, 85], [90, 15]]
    X, y = make_blobs(400, centers=centers, cluster_std=[7, 7, 6, 8, 8], random_state=5)
    df = pd.DataFrame({"CustomerID": np.arange(1, 401), "Gender": rng.choice(["Male", "Female"], 400),
                       "Age": rng.randint(18, 70, 400), "Income": X[:, 0].round(1), "Spending": X[:, 1].round(1)})
    for i in range(3):
        df[f"junk{i}"] = rng.rand(400)
    df["truth"] = y
    return dict(df=df, task="clustering", target="truth", id_column="CustomerID")


def k_aniso_scaled(rng):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(3000, n_features=4, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=6)
    X = X @ np.array([[0.6, -0.6, 0, 0], [-0.4, 0.8, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    df = pd.DataFrame(X, columns=list("abcd"))
    df["b"] *= 1000
    df["skewed"] = np.exp(rng.randn(3000))
    df = _inject_missing(df, ["a", "c"], 0.05, rng)
    df["truth"] = y
    return dict(df=df, task="clustering", target="truth", single_csv=False)


def k_mixed_cat(rng):
    n = 1200
    y = rng.randint(0, 4, n)
    df = pd.DataFrame({"num1": y * 1.5 + rng.randn(n), "num2": rng.randn(n) + (y % 2) * 2,
                       "colour": np.where(rng.rand(n) < 0.8, np.array(["red", "green", "blue", "red"])[y],
                                          rng.choice(["red", "green", "blue"], n)),
                       "plan": np.where(rng.rand(n) < 0.7, np.array(["basic", "basic", "pro", "pro"])[y], "basic"),
                       "truth": y})
    return dict(df=df, task="clustering", target="truth")


def k_tiny(rng):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(30, n_features=2, centers=3, random_state=7)
    df = pd.DataFrame(X, columns=["p", "q"])
    df["truth"] = y
    return dict(df=df, task="clustering", target="truth")


def k_large(rng):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(60000, n_features=12, centers=6, cluster_std=2.5, random_state=8)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    for i in range(8):
        df[f"noise{i}"] = rng.randn(len(df)) * 3
    df["truth"] = y
    return dict(df=df, task="clustering", target="truth")


BUILDERS = [c_synth_imb, c_multi_nonlinear, c_tiny, c_breast_cancer, c_edge, c_large,
            r_friedman, r_counts_datetime, r_diabetes, r_tiny, r_prices_strings,
            k_blobs_noise, k_aniso_scaled, k_mixed_cat, k_tiny, k_large]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev_data"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for b in BUILDERS:
        if a.only and b.__name__ not in a.only:
            continue
        rng = np.random.RandomState(12345)
        spec = b(rng)
        out = os.path.join(a.out, b.__name__)
        meta = make_dataset(spec["df"], out, spec["task"], spec.get("target"), id_column=spec.get("id_column"),
                            feature_types=spec.get("feature_types", "auto"), seed=0,
                            single_csv=spec.get("single_csv", True), extra_meta=spec.get("extra_meta"))
        if spec.get("post"):
            spec["post"](out)
        print(f"{b.__name__:20s} {spec['task']:15s} rows={len(spec['df'])} meta_keys={list(meta)}")


if __name__ == "__main__":
    main()
