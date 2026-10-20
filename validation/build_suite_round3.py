"""Build the ROUND-3 validation suite: 18 new datasets (zero overlap with rounds 1-2, all stress probes and
docs/03) + round-3 stress probes.

    py validation/build_suite_round3.py            # -> validation/data/suite_r3/<name>/, validation/data/stress_r3/<name>/
    py validation/build_suite_round3.py --only r_taxi

Reuses the round-1 / round-2 helpers (build_suite.py, build_suite_round2.py; unchanged). Same folder format
(docs/02) and the same hidden/truth.json ground-truth file. Fixed seeds, reproducible.
"""
import argparse
import gzip
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_suite as B  # noqa: E402
from build_suite import (Rec, _download, add_constant, add_duplicate, add_id, add_mcar, add_noise, declare,  # noqa: E402
                         finish, openml, uci_zip_member)
from build_suite_round2 import _write_custom_split  # noqa: E402

SUITE3 = os.path.join(HERE, "data", "suite_r3")
STRESS3 = os.path.join(HERE, "data", "stress_r3")
UCI_METRO = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
INSURANCE_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
CREDIT_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
                "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def _strat_sub(df, col, n, seed):
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(np.arange(len(df)), train_size=n, random_state=seed, stratify=df[col].astype(str))
    return df.iloc[np.sort(keep)].reset_index(drop=True)


# =========================================================================== classification
def c_amazon(root=SUITE3, variant=None):
    full = openml(4135)
    df = _strat_sub(full, "target", 20000, 301)
    rec = Rec("c_amazon" if variant is None else f"s3_amazon_{variant}")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c]).astype(int)
    df["target"] = df["target"].map({1: "granted", 0: "denied"})
    rec.nat("Kaggle Amazon employee access (OpenML 4135), stratified 20000-row subsample of 32769; 'denied' 5.8 % (imbalanced binary); ALL 9 features are integer-coded high-cardinality categoricals (RESOURCE ~5000, MGR_ID ~3500, ROLE_* up to ~2000 levels) declared categorical (A3 probe)")
    ft = {c: "categorical" for c in df.columns if c != "target"}
    if variant == "notypes":
        ft = None
        rec.inj("stress (SH3): NO feature_types; integer codes named *_ID / *_CODE / ROLE_* must not be treated as ordinal numbers")
    return finish(df, rec, root, "classification", "target", ft=ft), rec


def c_credit(root=SUITE3, variant=None):
    df = openml(42477)
    rec = Rec("c_credit" if variant is None else f"s3_credit_{variant}")
    df = df.rename(columns={f"x{i + 1}": n for i, n in enumerate(CREDIT_NAMES)})
    df = df.rename(columns={"y": "default_next_month"})
    df["id"] = pd.to_numeric(df["id"]).astype(int)
    df["default_next_month"] = df["default_next_month"].astype(str).map({"0": "no", "1": "yes"})
    for c in ("SEX", "EDUCATION", "MARRIAGE"):
        df[c] = pd.to_numeric(df[c]).astype(int)
    df = df.sample(n=20000, random_state=302).reset_index(drop=True)
    rec.nat("UCI default of credit card clients (OpenML 42477), 20000-row subsample of 30000; default 22 % (imbalanced); integer-coded SEX/EDUCATION/MARRIAGE (EDUCATION has undocumented codes 0/5/6); PAY_* ordinal delays with -2/-1 sentinels; BILL/PAY amounts heavy-tailed; `id` (undeclared)")
    rec.truth["id_columns"].append("id")
    add_mcar(df, rec, ["AGE", "BILL_AMT3", "PAY_AMT2", "EDUCATION"], 0.08, 303)
    add_duplicate(df, rec, "LIMIT_BAL", "credit_limit_ntd")
    add_constant(df, rec, "currency", "NTD")
    ft = declare(df, "default_next_month", overrides={"SEX": "categorical", "EDUCATION": "categorical",
                                                       "MARRIAGE": "categorical"}, omit=("id",))
    if variant == "missing":
        add_mcar(df, rec, [c for c in df.columns if c not in ("default_next_month", "id")], 0.25, 304)
        rec.inj("stress: +25 % MCAR in every feature column")
    return finish(df, rec, root, "classification", "default_next_month", ft=ft), rec


def c_electricity(root=SUITE3):
    df = openml(151)
    df = df.iloc[np.sort(np.random.RandomState(305).choice(len(df), 20000, replace=False))].reset_index(drop=True)
    rec = Rec("c_electricity")
    df["day"] = pd.to_numeric(df["day"]).astype(int)
    rec.nat("Elec2 NSW electricity market (OpenML 151), 20000-row random subsample of 45312; UP 42 %; time-ordered data (date normalised 0..1, period 0..1, day 1..7 declared categorical); nonlinear price dynamics")
    add_noise(df, rec, 4, 306)
    ft = declare(df, "class", overrides={"day": "categorical"})
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_letter(root=SUITE3):
    df = _strat_sub(openml(6), "class", 10000, 307)
    rec = Rec("c_letter")
    rec.nat("UCI letter recognition (OpenML 6), stratified 10000-row subsample of 20000; 26 balanced classes (~290 train rows per class); 16 integer features 0..15; strongly nonlinear")
    ft = declare(df, "class")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_cmc(root=SUITE3):
    df = openml(23)
    rec = Rec("c_cmc")
    df = df.rename(columns={"Wifes_now_working%3F": "Wifes_now_working"})
    df["Contraceptive_method_used"] = df["Contraceptive_method_used"].astype(str).map({"1": "none", "2": "long-term", "3": "short-term"})
    for c in df.columns:
        if c != "Contraceptive_method_used":
            df[c] = pd.to_numeric(df[c]).astype(int)
    rec.nat("UCI contraceptive method choice (OpenML 23), 1473 x 10 (small n); 3 classes 43 / 35 / 22 %; ordinal 1..4 codes declared categorical; hard (best published accuracy ~0.55-0.60); target label 'none' is a literal class value")
    ft = declare(df, "Contraceptive_method_used", overrides={c: "categorical" for c in
                 ("Wifes_education", "Husbands_education", "Wifes_religion", "Wifes_now_working",
                  "Husbands_occupation", "Standard-of-living_index", "Media_exposure")})
    return finish(df, rec, root, "classification", "Contraceptive_method_used", ft=ft), rec


def c_nomao(root=SUITE3):
    df = _strat_sub(openml(1486), "Class", 25000, 308)
    rec = Rec("c_nomao")
    df["Class"] = df["Class"].astype(str).map({"1": "different", "2": "same"})
    rec.nat("Nomao place de-duplication (OpenML 1486), stratified 25000-row subsample of 34465; 118 features (89 numeric similarity scores + 29 three-level categoricals); 'different' 28.6 %; wide table near the size limit (25000 x 119) -> B1 heavy size class / fallback, S2 width cap, 3-fold rules")
    ft = declare(df, "Class")
    return finish(df, rec, root, "classification", "Class", ft=ft), rec


# =========================================================================== regression
def r_metro(root=SUITE3):
    raw = uci_zip_member(UCI_METRO, "uci_metro", "Metro_Interstate_Traffic_Volume.csv.gz")
    full = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
    df = full.sample(n=20000, random_state=311).sort_index().reset_index(drop=True)
    rec = Rec("r_metro")
    df["holiday"] = df["holiday"].fillna("None")
    rec.truth["datetime_columns"].append("date_time")
    rec.nat("UCI 492 Metro Interstate traffic volume (I-94), 20000-row subsample of 48204 hourly rows; COUNT target traffic_volume 0..7280 (bimodal day/night, skew -0.1: log not needed for R2, but RMSLE punishes night errors); `date_time` 'YYYY-MM-DD HH:MM:SS' is the main signal (hour, weekday); `holiday` literal 'None' for 99.9 % of rows (pandas reads it as NaN); weather_description 38 levels; temp in Kelvin with 0 K errors")
    ft = declare(df, "traffic_volume", overrides={"date_time": "datetime"})
    return finish(df, rec, root, "regression", "traffic_volume", ft=ft), rec


def r_socmob(root=SUITE3):
    df = openml(541)
    rec = Rec("r_socmob")
    tgt = "counts_for_sons_current_occupation"
    rec.nat("socmob US social mobility (OpenML 541), 1156 x 6 (small n); target = weighted COUNT of sons per (father occupation, son occupation, family, race) cell: 0..414, skew 4.4, 20 % zeros -> log1p / RMSLE matters (B2 / C3 / A transform probes); 17-level occupations x2 categoricals")
    ft = declare(df, tgt)
    return finish(df, rec, root, "regression", tgt, ft=ft), rec


def r_taxi(root=SUITE3, variant=None):
    full = openml(42729)
    df = full.sample(n=20000, random_state=312).reset_index(drop=True)
    rec = Rec("r_taxi" if variant is None else f"s3_taxi_{variant}")
    for c in ("VendorID", "RatecodeID", "PULocationID", "DOLocationID", "trip_type"):
        df[c] = pd.to_numeric(df[c]).astype(int)
    for c in ("extra", "mta_tax", "improvement_surcharge"):
        df[c] = pd.to_numeric(df[c])
    pu = pd.Timestamp("2016-12-01") + pd.to_timedelta(df["lpep_pickup_datetime_day"] - 1, unit="D") + \
        pd.to_timedelta(df["lpep_pickup_datetime_hour"], unit="h") + pd.to_timedelta(df["lpep_pickup_datetime_minute"], unit="m")
    df["fare_before_tip"] = (df["total_amount"] - df["tip_amount"]).round(2)
    df.insert(0, "pickup_datetime", pu.dt.strftime("%Y-%m-%d %H:%M"))
    df = df.drop(columns=["total_amount", "lpep_pickup_datetime_day", "lpep_pickup_datetime_hour",
                          "lpep_pickup_datetime_minute"])
    cols = [c for c in df.columns if c != "tip_amount"] + ["tip_amount"]
    df = df[cols]
    rec.truth["datetime_columns"].append("pickup_datetime")
    rec.nat("NYC green taxi Dec 2016 (OpenML 42729), 20000-row subsample of 581835; target tip_amount: SKEWED NON-COUNT (continuous dollars, 15 % zeros, skew 10.8); integer-coded PULocationID / DOLocationID (~240 levels), RatecodeID, VendorID, trip_type declared categorical (SH3 stress without types); `pickup_datetime` built from the day/hour/minute columns")
    rec.inj("total_amount replaced by fare_before_tip = total_amount - tip_amount (total_amount contains the target)")
    ft = declare(df, "tip_amount", overrides={"pickup_datetime": "datetime", **{c: "categorical" for c in
                 ("VendorID", "RatecodeID", "PULocationID", "DOLocationID", "trip_type", "store_and_fwd_flag")}})
    if variant == "notypes":
        ft = None
        rec.inj("stress (SH3): NO feature_types; PULocationID/DOLocationID/RatecodeID/VendorID integer codes")
    return finish(df, rec, root, "regression", "tip_amount", ft=ft), rec


def r_pol(root=SUITE3):
    df = openml(201)
    rec = Rec("r_pol")
    tgt = df.columns[-1]
    df = df.rename(columns={tgt: "throughput"})
    rec.nat("pol telecommunication (OpenML 201; Weiss & Indurkhya), 15000 x 49; target 0..100 (integer-valued, bounded) with 63 % zeros, skew 0.93 (skewed, not a count): raw target expected to beat log1p (B2 / C3 transform-choice probe); 48 numerics, many near-constant")
    ft = declare(df, "throughput")
    return finish(df, rec, root, "regression", "throughput", ft=ft), rec


def r_insurance(root=SUITE3):
    p = _download(INSURANCE_URL, os.path.join(B.CACHE, "raw", "insurance", "insurance.csv"))
    df = pd.read_csv(p)
    rec = Rec("r_insurance")
    rec.nat("Kaggle medical cost personal dataset (insurance.csv, GitHub mirror of the Kaggle file), 1338 x 7; charges skewed 1.5; strong age trend (near-linear in age) + smoker x bmi interaction")
    rng = np.random.RandomState(321)
    old = (df["age"] >= 60).to_numpy()
    test = rng.rand(len(df)) < 0.2
    test[old] = True
    rec.inj(f"NON-random split: all {int(old.sum())} customers aged >= 60 in test (training ages 18..59) -> out-of-range rows where the age trend is close to LINEAR (linear extrapolation right; B3 probe), plus 20 % random")
    rec.truth["out_of_range_test_rows"] = int(old.sum())
    ft = declare(df, "charges")
    return _write_custom_split(df, rec, root, "charges", test, ft), rec


def r_airfoil(root=SUITE3):
    df = openml(43919)
    rec = Rec("r_airfoil")
    rec.nat("NASA airfoil self-noise (OpenML 43919 / UCI 291), 1503 x 6; scaled sound pressure; strongly nonlinear in frequency (peaked spectrum, steep high-frequency roll-off) and thickness")
    rng = np.random.RandomState(322)
    hi = (df["frequency"] >= 10000).to_numpy()
    test = rng.rand(len(df)) < 0.2
    test[hi] = True
    rec.inj(f"NON-random split: all {int(hi.sum())} rows with frequency >= 10 kHz in test (training max 8 kHz) -> out-of-range rows where the response is NONLINEAR (linear extrapolation wrong; B3 probe), plus 20 % random")
    rec.truth["out_of_range_test_rows"] = int(hi.sum())
    ft = declare(df, "pressure")
    return _write_custom_split(df, rec, root, "pressure", test, ft), rec


# =========================================================================== clustering
def k_glass(root=SUITE3):
    df = openml(41)
    rec = Rec("k_glass")
    rec.nat("UCI glass identification (OpenML 41), 214 x 9, k = 6 strongly UNEQUAL groups (76 / 70 / 29 / 17 / 13 / 9); oxide percentages on different scales, Ba/Fe mostly zero")
    ft = declare(df, "Type")
    return finish(df, rec, root, "clustering", "Type", ft=ft), rec


def k_banknote(root=SUITE3):
    df = openml(1462)
    rec = Rec("k_banknote")
    df.columns = ["variance", "skewness", "curtosis", "entropy", "Class"]
    rec.nat("UCI banknote authentication (OpenML 1462), 1372 x 4 wavelet features, k = 2 (762 / 610); ELONGATED, curved non-spherical groups")
    ft = declare(df, "Class")
    return finish(df, rec, root, "clustering", "Class", ft=ft), rec


def k_blobs_easy(root=SUITE3):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=2500, n_features=6, centers=5, cluster_std=1.0, center_box=(-10, 10), random_state=331)
    df = pd.DataFrame(X.round(4), columns=[f"m{j + 1}" for j in range(6)])
    df["group"] = y
    rec = Rec("k_blobs_easy")
    rec.inj("synthetic EASY case: 5 isotropic, equal-variance, equal-size Gaussian blobs in 6 dims (make_blobs seed 331), no noise -> KMeans is the right model; checks that the agreement medoid does not pick a worse family")
    ft = declare(df, "group")
    return finish(df, rec, root, "clustering", "group", ft=ft), rec


def k_optdigits(root=SUITE3, variant=None):
    df = _strat_sub(openml(28), "class", 3000, 341)
    rec = Rec("k_optdigits" if variant is None else f"s3_optdigits_{variant}")
    rec.nat("UCI optdigits (OpenML 28), stratified 3000-row subsample of 5620; 64 pixel-count features 0..16, MANY MODERATELY INFORMATIVE features that must not be dropped (SH1 probe); several natively constant / near-constant border pixels; k = 10")
    ft = declare(df, "class")
    single = variant != "traintest"
    if variant == "traintest":
        rec.inj("stress: docs/02 clustering variant 2 (train/test files)")
    return finish(df, rec, root, "clustering", "class", ft=ft, single_csv=single), rec


def k_dermatology(root=SUITE3):
    df = openml(35)
    rec = Rec("k_dermatology")
    tgt = "class"
    for c in df.columns:
        if c != tgt:
            df[c] = pd.to_numeric(df[c])
    rec.nat("UCI dermatology (OpenML 35), 366 x 34, k = 6 (112 / 72 / 61 / 52 / 49 / 20); 33 clinical/histopathological 0..3 grades + Age (8 missing); family_history binary")
    add_noise(df, rec, 5, 351)
    ft = declare(df, tgt)
    return finish(df, rec, root, "clustering", tgt, ft=ft), rec


def k_moons(root=SUITE3):
    from sklearn.datasets import make_moons
    rng = np.random.RandomState(361)
    Xm, ym = make_moons(n_samples=1200, noise=0.08, random_state=361)
    Xm = Xm * 3.0
    a = rng.multivariate_normal([9, 4], [[4.0, 3.6], [3.6, 3.5]], 500)
    b = rng.multivariate_normal([-6, 6], [[0.3, 0], [0, 3.0]], 300)
    X = np.vstack([Xm, a, b])
    y = np.concatenate([ym, np.full(500, 2), np.full(300, 3)])
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    df = pd.DataFrame({"x": X[:, 0].round(4), "y_pos": X[:, 1].round(4)})
    df["group"] = y
    rec = Rec("k_moons")
    rec.inj("synthetic NON-CONVEX case: two interleaving moons (600 + 600) + one elongated correlated blob (500) + one thin vertical blob (300), k = 4")
    add_noise(df, rec, 4, 362, scale_big=False)
    ft = declare(df, "group")
    return finish(df, rec, root, "clustering", "group", ft=ft), rec


SUITE3_BUILDERS = [c_amazon, c_credit, c_electricity, c_letter, c_cmc, c_nomao,
                   r_metro, r_socmob, r_taxi, r_pol, r_insurance, r_airfoil,
                   k_glass, k_banknote, k_blobs_easy, k_optdigits, k_dermatology, k_moons]


def build_stress():
    for fn, v in ((c_amazon, "notypes"), (r_taxi, "notypes"), (c_credit, "missing"), (k_optdigits, "traintest")):
        out, rec = fn(STRESS3, v)
        print("stress", rec.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(SUITE3, exist_ok=True)
    rows = []
    for fn in SUITE3_BUILDERS:
        if a.only and fn.__name__ not in a.only:
            continue
        out, rec = fn()
        r = B.summary_row(out, rec)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if not a.only:
        build_stress()
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "suite_r3_summary.csv"), index=False)


if __name__ == "__main__":
    main()
