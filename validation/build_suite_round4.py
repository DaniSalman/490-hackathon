"""Build the ROUND-4 (final) validation suite: 18 new datasets (zero overlap with rounds 1-3, all stress
probes and docs/03) + round-4 stress probes.

    py validation/build_suite_round4.py     # -> validation/data/suite_r4/<name>/, validation/data/stress_r4/<name>/

Reuses the helpers of build_suite.py / build_suite_round2.py (unchanged). Same folder format (docs/02) and
hidden/truth.json ground truth. Fixed seeds, reproducible.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_suite as B  # noqa: E402
from build_suite import (Rec, _download, add_constant, add_duplicate, add_id, add_mcar, add_noise, declare,  # noqa: E402
                         finish, openml)

SUITE4 = os.path.join(HERE, "data", "suite_r4")
STRESS4 = os.path.join(HERE, "data", "stress_r4")
CHURN_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"


def _strat_sub(df, col, n, seed):
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(np.arange(len(df)), train_size=n, random_state=seed, stratify=df[col].astype(str))
    return df.iloc[np.sort(keep)].reset_index(drop=True)


# =========================================================================== classification
def c_bankchurn(root=SUITE4, variant=None):
    p = _download(CHURN_URL, os.path.join(B.CACHE, "raw", "bankchurn", "Churn_Modelling.csv"))
    df = pd.read_csv(p)
    rec = Rec("c_bankchurn" if variant is None else f"s4_bankchurn_{variant}")
    df["Exited"] = df["Exited"].map({0: "stayed", 1: "exited"})
    rec.nat("Kaggle bank customer churn (Churn_Modelling.csv, GitHub mirror of the Kaggle file), 10000 x 14; exited 20.4 % (imbalanced); `RowNumber` + `CustomerId` IDs, `Surname` high-cardinality (~2900 levels) text-like categorical; Balance with 36 % zeros; Telco-like churn table")
    rec.truth["id_columns"] += ["RowNumber", "CustomerId"]
    ft = declare(df, "Exited", overrides={"HasCrCard": "categorical", "IsActiveMember": "categorical"},
                 omit=("RowNumber", "CustomerId"))
    if variant == "missing":
        add_mcar(df, rec, [c for c in df.columns if c not in ("Exited", "RowNumber", "CustomerId")], 0.20, 401)
        rec.inj("stress: +20 % MCAR in every feature column")
    return finish(df, rec, root, "classification", "Exited", ft=ft), rec


def c_spambase(root=SUITE4):
    df = openml(44)
    rec = Rec("c_spambase")
    df["class"] = df["class"].astype(str).map({"0": "ham", "1": "spam"})
    rec.nat("UCI spambase (OpenML 44), 4601 x 58; spam 39 %; 57 heavy-tailed word / char frequency and capital-run features (many zeros)")
    add_constant(df, rec, "source", "hp-labs")
    add_duplicate(df, rec, "capital_run_length_total", "caps_total_copy")
    ft = declare(df, "class")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_ozone(root=SUITE4):
    df = openml(1487)
    rec = Rec("c_ozone")
    df["Class"] = df["Class"].astype(str).map({"1": "normal", "2": "ozone"})
    rec.nat("UCI ozone level detection 8-hour (OpenML 1487), 2534 x 73; ozone days 6.3 % (strongly imbalanced, small minority = 120 train rows); 72 correlated meteorological features")
    add_mcar(df, rec, ["V1", "V10", "V20", "V30", "V40", "V50"], 0.06, 402)
    ft = declare(df, "Class")
    return finish(df, rec, root, "classification", "Class", ft=ft), rec


def c_texture(root=SUITE4):
    df = openml(40499)
    rec = Rec("c_texture")
    rec.nat("texture (OpenML 40499, ELENA project), 5500 x 41; **11 balanced classes** (many-class, M1 probe), 40 numeric texture features")
    ft = declare(df, "Class")
    return finish(df, rec, root, "classification", "Class", ft=ft), rec


def c_yeast(root=SUITE4):
    df = openml(181)
    rec = Rec("c_yeast")
    tgt = "class_protein_localization"
    rec.nat("UCI yeast protein localisation (OpenML 181), 1484 x 9; **10 classes, strongly imbalanced** (463 / 429 / 244 / 163 / 51 / 44 / 35 / 30 / 20 / 5): many-class + rare-class (M1, A3); small n")
    add_noise(df, rec, 3, 403)
    ft = declare(df, tgt)
    return finish(df, rec, root, "classification", tgt, ft=ft), rec


def c_miniboone(root=SUITE4):
    df = _strat_sub(openml(41150), "signal", 30000, 404)
    rec = Rec("c_miniboone")
    df["signal"] = df["signal"].astype(str).map({"True": "signal", "False": "background", "1": "signal", "0": "background"}).fillna(df["signal"].astype(str))
    rec.nat("MiniBooNE particle identification (OpenML 41150), stratified 30000-row subsample of 130064; 50 numeric features with -999 sentinels; signal 28 %")
    add_noise(df, rec, 30, 405)
    ft = declare(df, "signal")
    rec.inj("HEAVY WIDE table: 30000 x 81 (22500 train rows x 80 features = 1.8 M cells; M1 fold planner / B heavy size class)")
    return finish(df, rec, root, "classification", "signal", ft=ft), rec


# =========================================================================== regression
def r_medical(root=SUITE4):
    df = openml(42131).sample(n=20000, random_state=411).reset_index(drop=True)
    rec = Rec("r_medical")
    df = df.drop(columns=["average_total_payments"])
    df["provider_zip_code"] = pd.to_numeric(df["provider_zip_code"]).astype(int)
    df["provider_id"] = pd.to_numeric(df["provider_id"]).astype(int)
    rec.nat("CMS Medicare inpatient charges (OpenML 42131), 20000-row subsample of 163065; target average_medicare_payments (skewed); HIGH-CARDINALITY categoricals: drg_definition 100, provider_name ~2800, city ~1700, hrr 306, state 51, zip / provider_id integer codes")
    rec.inj("average_total_payments dropped (it contains the target)")
    ft = declare(df, "average_medicare_payments", overrides={"provider_zip_code": "categorical", "provider_id": "categorical"})
    return finish(df, rec, root, "regression", "average_medicare_payments", ft=ft), rec


def r_house16h(root=SUITE4):
    df = openml(574)
    rec = Rec("r_house16h")
    rec.nat("US census house_16H (OpenML 574, Delve), 22784 x 17; median house price per region, skew 3.8, capped at 500001; 16 demographic ratios (House-Prices-like skewed price target)")
    ft = declare(df, "price")
    return finish(df, rec, root, "regression", "price", ft=ft), rec


def r_cpuact(root=SUITE4):
    df = openml(197)
    rec = Rec("r_cpuact")
    rec.nat("Delve cpu_act (OpenML 197), 8192 x 22; target usr = % CPU time in user mode (0..99, left-skewed); heavy-tailed system counters")
    add_noise(df, rec, 3, 412)
    ft = declare(df, "usr")
    return finish(df, rec, root, "regression", "usr", ft=ft), rec


def r_elevators(root=SUITE4):
    df = openml(216)
    rec = Rec("r_elevators")
    rec.nat("elevators (OpenML 216, F-16 aircraft control), 16599 x 19; target Goal (control action, small-scale values); nearly linear + interactions")
    ft = declare(df, "Goal")
    return finish(df, rec, root, "regression", "Goal", ft=ft), rec


def r_autompg(root=SUITE4):
    df = openml(196)
    rec = Rec("r_autompg")
    df = df.rename(columns={"class": "mpg"})
    df["origin"] = df["origin"].astype(str).map({"1": "usa", "2": "europe", "3": "japan"}).fillna(df["origin"].astype(str))
    rec.nat("UCI auto-mpg (OpenML 196), 398 x 8 (very small n); 6 missing horsepower; `origin` categorical; target mpg")
    add_id(df, rec, "car_id", 413, "string", "CAR")
    ft = declare(df, "mpg", overrides={"cylinders": "numeric", "model": "numeric"}, omit=("car_id",))
    return finish(df, rec, root, "regression", "mpg", ft=ft), rec


def r_sulfur(root=SUITE4):
    df = openml(23515).drop(columns=["y2"])
    rec = Rec("r_sulfur")
    rec.nat("sulfur recovery unit (OpenML 23515), 10081 x 6; target y1 (H2S concentration, normalised 0..1, right-skewed with spikes); 5 sensor inputs; second output y2 dropped (simultaneous target)")
    ft = declare(df, "y1")
    return finish(df, rec, root, "regression", "y1", ft=ft), rec


# =========================================================================== clustering
def k_zoo(root=SUITE4):
    df = openml(62)
    rec = Rec("k_zoo")
    for c in df.columns:
        if c not in ("animal", "type", "legs"):
            df[c] = df[c].astype(str).str.lower()
    rec.nat("UCI zoo (OpenML 62), 101 x 17, k = 7 unequal (41 / 20 / 13 / 10 / 8 / 5 / 4); 15 boolean attributes + legs; `animal` name unique (undeclared ID-like string)")
    rec.truth["id_columns"].append("animal")
    ft = declare(df, "type", omit=("animal",))
    return finish(df, rec, root, "clustering", "type", ft=ft), rec


def k_userknow(root=SUITE4):
    df = openml(1508)
    rec = Rec("k_userknow")
    df.columns = ["STG", "SCG", "STR", "LPR", "PEG", "UNS"]
    rec.nat("UCI user knowledge modeling (OpenML 1508), 403 x 5, k = 5 student knowledge levels (129 / 122 / 102 / 26 / 24); customer-segmentation-like small table")
    ft = declare(df, "UNS")
    return finish(df, rec, root, "clustering", "UNS", ft=ft), rec


def k_satimage(root=SUITE4, variant=None):
    df = _strat_sub(openml(182), "class", 4000, 421)
    rec = Rec("k_satimage")
    rec.nat("UCI Landsat satimage (OpenML 182), stratified 4000 of 6430, 36 spectral features, k = 6 unequal groups; **clustering train/test variant** (docs/02 variant 2: labels only for test rows) -> M2 probe")
    ft = declare(df, "class")
    return finish(df, rec, root, "clustering", "class", ft=ft, single_csv=False), rec


def k_mfeat(root=SUITE4):
    df = openml(16)
    rec = Rec("k_mfeat")
    rec.nat("UCI multiple features, Karhunen-Loeve coefficients (OpenML 16), 2000 x 64, k = 10 balanced digits (many-group clustering)")
    ft = declare(df, "class")
    return finish(df, rec, root, "clustering", "class", ft=ft), rec


def k_vowel(root=SUITE4):
    df = openml(307)
    rec = Rec("k_vowel")
    rec.nat("UCI vowel (OpenML 307), 990 x 12, k = 11 vowels; `Speaker_Number` (15 levels) and `Sex` categoricals irrelevant to the vowel; 10 LPC features")
    ft = declare(df, "Class", overrides={"Speaker_Number": "categorical"})
    return finish(df, rec, root, "clustering", "Class", ft=ft), rec


def k_wallrobot(root=SUITE4):
    df = openml(1497)
    rec = Rec("k_wallrobot")
    rec.nat("UCI wall-following robot navigation (OpenML 1497), 5456 x 24 ultrasound distances, k = 4 unequal (2205 / 2097 / 826 / 328); larger n, non-spherical")
    add_noise(df, rec, 3, 431, scale_big=False)
    ft = declare(df, "Class")
    return finish(df, rec, root, "clustering", "Class", ft=ft), rec


SUITE4_BUILDERS = [c_bankchurn, c_spambase, c_ozone, c_texture, c_yeast, c_miniboone,
                   r_medical, r_house16h, r_cpuact, r_elevators, r_autompg, r_sulfur,
                   k_zoo, k_userknow, k_satimage, k_mfeat, k_vowel, k_wallrobot]


def build_stress():
    out, rec = c_bankchurn(STRESS4, "missing")
    print("stress", rec.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(SUITE4, exist_ok=True)
    rows = []
    for fn in SUITE4_BUILDERS:
        if a.only and fn.__name__ not in a.only:
            continue
        out, rec = fn()
        r = B.summary_row(out, rec)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if not a.only:
        build_stress()
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "suite_r4_summary.csv"), index=False)


if __name__ == "__main__":
    main()
