"""Build the ROUND-2 validation suite: 18 new datasets (zero overlap with round 1, its stress probes and
docs/03) + round-2 stress probes.

    py validation/build_suite_round2.py            # -> validation/data/suite_r2/<name>/, validation/data/stress_r2/<name>/
    py validation/build_suite_round2.py --only r_kc_house

Re-uses the download / injection helpers of build_suite.py (round 1; unchanged). Same folder format
(docs/02) and the same hidden/truth.json ground-truth file, with extra keys for the new profile fields
(blank-string / suspected-missing cells). Fixed seeds, reproducible.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_suite as B  # noqa: E402
from build_suite import (Rec, add_constant, add_duplicate, add_id, add_mcar, add_noise, declare, finish,  # noqa: E402
                         openml, uci_zip_member)

SUITE2 = os.path.join(HERE, "data", "suite_r2")
STRESS2 = os.path.join(HERE, "data", "stress_r2")
UCI_SEOUL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
UCI_WINE = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def add_blanks(df, rec, col, frac, seed, token=" "):
    """Numeric column -> string column where `frac` of the cells are a blank / NA-like token (S3)."""
    rng = np.random.RandomState(seed)
    m = rng.rand(len(df)) < frac
    s = df[col].map(lambda v: "" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))).astype(object)
    s[m] = token
    df[col] = s
    rec.truth.setdefault("blank_cells" if token.strip() == "" else "token_cells", {})[col] = int(m.sum())
    rec.inj(f"`{col}` stored as strings with {int(m.sum())} {token!r} cells ({'blank-string missing' if token.strip() == '' else 'NA-like token'})")
    return df


# =========================================================================== classification
def c_churn(root=SUITE2, variant=None):
    df = openml(40701)
    rec = Rec("c_churn" if variant is None else f"s2_churn_{variant}")
    df["class"] = df["class"].astype(str).map({"0": "no", "1": "yes"})
    df["area_code"] = df["area_code"].astype(str)
    df["state"] = pd.to_numeric(df["state"]).astype(int)
    df["phone_number"] = pd.to_numeric(df["phone_number"]).astype(int)
    rec.nat("UCI/MLC++ telecom churn (OpenML 40701), 5000 x 21; churn 14.1 % (imbalanced); `state` 51 levels (integer-coded, declared categorical -> high cardinality); `phone_number` unique integer code (ID-like, declared categorical)")
    rec.truth["id_columns"].append("phone_number")
    # S1: literal NA-like category values that carry signal
    rng = np.random.RandomState(201)
    p_none = np.where(df["class"] == "yes", 0.55, 0.25)
    u = rng.rand(len(df))
    lv = np.where(u < p_none, "none", np.where(u < p_none + 0.2, "basic", np.where(u < p_none + 0.35, "premium",
                  np.where(u < p_none + 0.45, "unknown", "-"))))
    df.insert(5, "addon_plan", lv)
    rec.inj("S1 probe: categorical `addon_plan` with literal levels none/basic/premium/unknown/- ; 'none' is "
            "over-represented among churners (signal must survive as a category)")
    rec.truth["literal_token_columns"] = ["addon_plan"]
    # S3: blank-string missing in a numeric column (Telco TotalCharges style)
    add_blanks(df, rec, "total_eve_charge", 0.03, 202, " ")
    ft = declare(df, "class", overrides={"total_eve_charge": "numeric", "area_code": "categorical", "state": "categorical",
                                         "phone_number": "categorical"})
    if variant == "tokens":
        for j, c in enumerate(["total_day_minutes", "total_night_minutes", "number_customer_service_calls"]):
            add_blanks(df, rec, c, 0.05, 210 + j, ["?", "-", "none"][j])
        ft = None
        rec.inj("stress: NA-like tokens ?, -, none inside 3 numeric columns; NO feature_types")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_magic(root=SUITE2):
    df = openml(1120)
    rec = Rec("c_magic")
    df.columns = [c.rstrip(":") for c in df.columns]
    # The OpenML file is sorted by class and `ID` is the row number -> ID alone predicts the class (default HGB
    # macro-F1 0.998). Shuffle the rows and renumber ID so it stays an (undeclared) ID without leaking.
    df = df.sample(frac=1.0, random_state=212).reset_index(drop=True)
    df["ID"] = np.random.RandomState(213).permutation(len(df)) + 1
    rec.nat("MAGIC gamma telescope (OpenML 1120), 19020 x 12; hadron 35 %; nonlinear; heavy-tailed fAsym/fM3*; `ID` column (undeclared; rows shuffled and ID renumbered because the original ID order leaks the class)")
    rec.truth["id_columns"].append("ID")
    add_duplicate(df, rec, "fLength", "fLength_mm")
    add_constant(df, rec, "telescope", "MAGIC-I")
    add_noise(df, rec, 4, 211)
    ft = declare(df, "class", omit=("ID",))
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_pageblocks(root=SUITE2):
    df = openml(30)
    rec = Rec("c_pageblocks")
    df["class"] = df["class"].astype(str).map({"1": "text", "2": "horiz_line", "3": "graphic", "4": "vert_line",
                                               "5": "picture"})
    df = df.rename(columns={"lenght": "length"})
    rec.nat("UCI page-blocks (OpenML 30), 5473 x 11; 5 classes 89.8 / 6.0 / 2.1 / 1.6 / 0.5 % (rare-class multiclass, 'graphic' has 28 rows); skewed scales")
    ft = declare(df, "class")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_creditapp(root=SUITE2):
    df = openml(29)
    rec = Rec("c_creditapp")
    df["class"] = df["class"].astype(str).map({"+": "approved", "-": "rejected"})
    rec.nat("UCI credit approval (OpenML 29), 690 x 16 (small n); anonymised A1..A15, 9 categoricals (A6 14 levels), native missing 67 cells ('?' in the UCI file)")
    add_noise(df, rec, 3, 221)
    ft = declare(df, "class")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_jannis(root=SUITE2):
    full = openml(41168)
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(np.arange(len(full)), train_size=25000, random_state=3, stratify=full["class"].astype(str))
    df = full.iloc[np.sort(keep)].reset_index(drop=True)
    rec = Rec("c_jannis")
    df["class"] = df["class"].astype(str).map({"0": "w", "1": "x", "2": "y", "3": "z"})
    rec.nat("jannis (OpenML 41168, AutoML benchmark), stratified 25000-row subsample of 83733; 54 anonymised numeric features; 4 classes 46 / 34 / 18 / 2 % (rare class)")
    add_noise(df, rec, 40, 231)
    add_duplicate(df, rec, "V1", "V1_copy")
    add_constant(df, rec, "batch", 7)
    ft = declare(df, "class")
    rec.inj("wide table near the size limit: 25000 x 97 (S2 cost guard)")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_kick(root=SUITE2, variant=None):
    full = openml(41162)
    df = full.sample(n=20000, random_state=5).reset_index(drop=True)
    rec = Rec("c_kick" if variant is None else f"s2_kick_{variant}")
    df["PurchDate"] = pd.to_datetime(df["PurchDate"], unit="s").dt.strftime("%m/%d/%Y")
    df["IsBadBuy"] = df["IsBadBuy"].astype(str)
    df["VehYear"] = pd.to_numeric(df["VehYear"]).astype(int)
    cols = [c for c in df.columns if c != "IsBadBuy"] + ["IsBadBuy"]
    df = df[cols]
    rec.truth["datetime_columns"].append("PurchDate")
    rec.nat("Kaggle 'Don't Get Kicked' car auctions (OpenML 41162), 20000-row subsample of 72983; bad buy 12.3 % (imbalanced)")
    rec.nat("`PurchDate` datetime string (m/d/Y); high-cardinality Model (~900) / SubModel (~700) / Trim; zip / buyer codes; heavy native missing (PRIMEUNIT/AUCGUART 95 %)")
    ft = declare(df, "IsBadBuy", overrides={"PurchDate": "datetime", "WheelTypeID": "categorical",
                                            "VNZIP1": "categorical", "BYRNO": "categorical", "IsOnlineSale": "categorical"})
    if variant == "notypes":
        ft = None
        rec.inj("stress: NO feature_types in metadata")
    return finish(df, rec, root, "classification", "IsBadBuy", ft=ft), rec


# =========================================================================== regression
def r_seoul(root=SUITE2):
    raw = uci_zip_member(UCI_SEOUL, "uci_seoul", "SeoulBikeData.csv")
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    rec = Rec("r_seoul")
    tgt = "Rented Bike Count"
    cols = [c for c in df.columns if c != tgt] + [tgt]
    df = df[cols]
    rec.truth["datetime_columns"].append("Date")
    rec.nat("UCI 560 Seoul bike sharing demand, 8760 hourly rows x 14; `Date` as dd/mm/yyyy strings (day-first!); count target with 295 zeros (non-functioning days) and skew 1.15 -> RMSLE matters (A5); nonlinear hour/temperature effects")
    ft = declare(df, tgt, overrides={"Date": "datetime", "Hour": "numeric"})
    return finish(df, rec, root, "regression", tgt, ft=ft), rec


def _write_custom_split(df, rec, root, target, test_mask, ft, extra_truth=None):
    """Regression folder with a NON-random split (docs/02 format, same files as make_dataset)."""
    out = os.path.join(root, rec.name)
    os.makedirs(os.path.join(out, "hidden"), exist_ok=True)
    te = np.where(test_mask)[0]
    tr = np.where(~test_mask)[0]
    df.iloc[tr].to_csv(os.path.join(out, "train.csv"), index=False)
    df.iloc[te].drop(columns=[target]).to_csv(os.path.join(out, "test.csv"), index=False)
    pd.DataFrame({"row_index": np.arange(len(te)), "label": df[target].iloc[te].to_numpy()}).to_csv(
        os.path.join(out, "hidden", "test_labels.csv"), index=False)
    meta = {"name": rec.name, "task": "regression", "target": target, "feature_types": ft}
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "hidden", "info.json"), "w", encoding="utf-8") as f:
        json.dump({"task": "regression", "target": target, "n_clusters": None}, f, indent=2)
    rec.truth.update(task="regression", target=target, n_clusters=None, id_column=None, n_total_rows=int(len(df)),
                     declared_feature_types=ft, **(extra_truth or {}))
    with open(os.path.join(out, "hidden", "truth.json"), "w", encoding="utf-8") as f:
        json.dump(rec.truth, f, indent=1)
    return out


def r_kc_house(root=SUITE2, variant=None):
    df = openml(42731)
    rec = Rec("r_kc_house" if variant is None else f"s2_kc_house_{variant}")
    date = (df["date_year"].astype(int).astype(str) + df["date_month"].astype(int).astype(str).str.zfill(2)
            + df["date_day"].astype(int).astype(str).str.zfill(2) + "T000000")
    df = df.drop(columns=["date_year", "date_month", "date_day"])
    df.insert(1, "date", date)
    df["price"] = df["price"].astype(int)
    df["zipcode"] = df["zipcode"].astype(int)
    rec.truth["datetime_columns"].append("date")
    rec.truth["id_columns"].append("id")
    rec.nat("Kaggle King County house sales (OpenML 42731), 21613 x 21; integer-valued price (skew 4.0) + `zipcode` 70 levels declared categorical (C1 / S2 probe); `date` in Kaggle format 20141013T000000; `id` repeated for re-sold houses (undeclared ID)")
    # B3 / C3 probe: out-of-range test rows. 75 % of the houses with sqft_living >= 99th percentile go to test.
    rng = np.random.RandomState(241)
    big = (df["sqft_living"] >= df["sqft_living"].quantile(0.99)).to_numpy()
    test = rng.rand(len(df)) < 0.25
    test[big] = rng.rand(big.sum()) < 0.75
    huge = (df["sqft_living"] >= df["sqft_living"].quantile(0.995)).to_numpy()
    test[huge] = True
    rec.inj(f"NON-random split: all {int(huge.sum())} houses with sqft_living >= p99.5 and 75 % of those in p99..p99.5 "
            f"placed in test ({int((big & test).sum())} large-house test rows, {int(huge.sum())} strictly above the training "
            f"maximum of sqft_living) -> out-of-range probe (B3/C3)")
    rec.truth["out_of_range_test_rows"] = int((big & test).sum())
    ft = declare(df, "price", overrides={"date": "datetime", "zipcode": "categorical"}, omit=("id",))
    if variant == "strnum":
        for c in ("sqft_living", "sqft_lot", "sqft_above"):
            df[c] = df[c].map(lambda v: f"{int(v):,d}")
        add_mcar(df, rec, ["bathrooms", "sqft_lot", "view", "condition", "yr_built"], 0.15, 242)
        ft = None
        rec.inj("stress: sqft columns as '1,234' strings + 15 % MCAR in 5 columns; NO feature_types")
    return _write_custom_split(df, rec, root, "price", test, ft), rec


def r_moneyball(root=SUITE2):
    df = openml(41021)
    rec = Rec("r_moneyball")
    df["RS"] = df["RS"].astype(int)
    for c in ("Year", "RA", "W", "G", "Playoffs"):
        df[c] = pd.to_numeric(df[c]).astype(int)
    cols = [c for c in df.columns if c != "RS"] + ["RS"]
    df = df[cols]
    rec.nat("Moneyball MLB team seasons (OpenML 41021), 1232 x 15; integer target RS (runs scored) + `Team` 39 levels (>30 -> C1 probe at small n); RankSeason/RankPlayoffs 80 % missing, OOBP/OSLG 66 % missing")
    ft = declare(df, "RS", overrides={"Playoffs": "categorical"})
    return finish(df, rec, root, "regression", "RS", ft=ft), rec


def r_superconduct(root=SUITE2):
    df = openml(43174).sample(n=15000, random_state=6).reset_index(drop=True)
    rec = Rec("r_superconduct")
    rec.nat("UCI superconductivity (OpenML 43174), 15000-row subsample of 21263 x 82; 81 highly correlated physico-chemical features; critical_temp skewed (0.0002..185 K), nonlinear")
    add_duplicate(df, rec, "mean_atomic_mass", "mean_atomic_mass_u")
    add_noise(df, rec, 3, 251)
    ft = declare(df, "critical_temp")
    rec.inj("wide regression table: 15000 x 86 (S2)")
    return finish(df, rec, root, "regression", "critical_temp", ft=ft), rec


def r_brazil(root=SUITE2):
    df = openml(42688)
    rec = Rec("r_brazil")
    df = df.drop(columns=["total_(BRL)"])
    tgt = "rent_amount_(BRL)"
    cols = [c for c in df.columns if c != tgt] + [tgt]
    df = df[cols]
    rec.nat("Kaggle Brazilian houses to rent (OpenML 42688), 10692 x 12; rent skewed; extreme native outliers (area up to 46335 m2, hoa up to 1.1 M BRL, property tax up to 313 k) -> out-of-range test rows; total_(BRL) dropped (sum leak); OpenML already maps the Kaggle floor '-' to 0")
    ft = declare(df, tgt)
    return finish(df, rec, root, "regression", tgt, ft=ft), rec


def r_wine(root=SUITE2):
    red = pd.read_csv(io.BytesIO(uci_zip_member(UCI_WINE, "uci_wine", "winequality-red.csv")), sep=";")
    white = pd.read_csv(io.BytesIO(uci_zip_member(UCI_WINE, "uci_wine", "winequality-white.csv")), sep=";")
    red.insert(0, "type", "red")
    white.insert(0, "type", "white")
    df = pd.concat([red, white], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)
    rec = Rec("r_wine")
    rec.nat("UCI 186 wine quality red + white (6497 x 13, merged with a `type` categorical); integer target quality 3..9, heavily concentrated on 5-7")
    add_blanks(df, rec, "residual sugar", 0.04, 261, " ")
    add_blanks(df, rec, "sulphates", 0.02, 262, "?")
    add_noise(df, rec, 2, 263)
    ft = declare(df, "quality", overrides={"residual sugar": "numeric", "sulphates": "numeric"})
    return finish(df, rec, root, "regression", "quality", ft=ft), rec


# =========================================================================== clustering
def k_wholesale(root=SUITE2):
    df = openml(1511)
    rec = Rec("k_wholesale")
    df = df.drop(columns=["Region"]).rename(columns={"V1": "Region", "V2": "Fresh", "V3": "Milk", "V4": "Grocery",
                                                     "V5": "Frozen", "V6": "Detergents_Paper", "V7": "Delicassen"})
    df["Region"] = df["Region"].astype(int).map({1: "Lisbon", 2: "Oporto", 3: "Other"})
    df["Channel"] = df["Channel"].astype(str).map({"1": "Horeca", "2": "Retail"})
    rec.nat("UCI wholesale customers (OpenML 1511), 440 x 7 + Channel label; k = 2 (Horeca / Retail, 68 / 32 %); annual spending heavily right-skewed; `Region` categorical irrelevant to the channel")
    ft = declare(df, "Channel")
    return finish(df, rec, root, "clustering", "Channel", ft=ft), rec


def k_vehicle(root=SUITE2):
    df = openml(54)
    rec = Rec("k_vehicle")
    rec.nat("Statlog vehicle silhouettes (OpenML 54), 846 x 18, k = 4 (bus / saab / opel / van; saab vs opel nearly inseparable); correlated shape features, scales 1..1000")
    add_noise(df, rec, 3, 271)
    ft = declare(df, "Class")
    return finish(df, rec, root, "clustering", "Class", ft=ft), rec


def k_ecoli(root=SUITE2):
    df = openml(39)
    rec = Rec("k_ecoli")
    rec.nat("UCI ecoli (OpenML 39), 336 x 7, k = 8 strongly unequal groups (143 / 77 / 52 / 35 / 20 / 5 / 2 / 2); lip/chg near-constant binary features")
    ft = declare(df, "class")
    return finish(df, rec, root, "clustering", "class", ft=ft), rec


def k_pendigits(root=SUITE2):
    df = openml(32).sample(n=6000, random_state=8).reset_index(drop=True)
    rec = Rec("k_pendigits")
    rec.nat("UCI pen-based digits (OpenML 32), 6000-row subsample, 16 features (0..100), k = 10; non-convex digit manifolds")
    ft = declare(df, "class")
    return finish(df, rec, root, "clustering", "class", ft=ft), rec


def k_segment(root=SUITE2):
    df = openml(36)
    rec = Rec("k_segment")
    rec.nat("UCI image segmentation (OpenML 36), 2310 x 19, k = 7 balanced; `region-pixel-count` natively constant; scales 0..250 vs 0..1; elongated classes")
    add_id(df, rec, "image_id", 281, "string", "IMG")
    ft = declare(df, "class", omit=("image_id",))
    return finish(df, rec, root, "clustering", "class", ft=ft), rec


def k_synth(root=SUITE2, variant=None):
    """Non-spherical, unequal-variance, unequal-size groups + informative categorical + planted noise."""
    rng = np.random.RandomState(291)
    sizes = [1200, 900, 600, 300]
    centers = np.array([[0, 0, 0], [6, 1, 0], [1, 6, 2], [6, 6, -4]], float)
    covs = [np.array([[4.0, 3.2, 0], [3.2, 3.0, 0], [0, 0, 0.5]]), np.diag([0.4, 0.4, 0.4]),
            np.array([[1.0, -0.8, 0], [-0.8, 2.5, 0.5], [0, 0.5, 1.0]]), np.diag([2.0, 0.3, 3.0])]
    X, g = [], []
    for k, (n, c, S) in enumerate(zip(sizes, centers, covs)):
        X.append(rng.multivariate_normal(c, S, n))
        g += [k] * n
    X, g = np.vstack(X), np.array(g)
    perm = rng.permutation(len(g))
    X, g = X[perm], g[perm]
    df = pd.DataFrame({"usage_hours": (X[:, 0] * 10 + 50).round(2), "spend_k": (X[:, 1] * 1000 + 8000).round(0),
                       "visit_rate": (X[:, 2] * 0.01 + 0.1).round(5)})
    lv = np.array(["none", "basic", "plus", "unknown"])
    hint = np.where(rng.rand(len(g)) < 0.8, lv[g], lv[rng.randint(0, 4, len(g))])
    df["plan"] = hint
    df["group"] = g
    rec = Rec("k_synth" if variant is None else f"s2_synth_{variant}")
    rec.inj("synthetic: 4 groups, sizes 1200/900/600/300, anisotropic & unequal covariances (non-spherical), 3 informative numerics on scales x10 / x1000 / x0.01")
    rec.inj("informative categorical `plan` (80 % agrees with the group; levels none/basic/plus/unknown -> also S1 literal tokens)")
    rec.truth["literal_token_columns"] = ["plan"]
    add_noise(df, rec, 5, 292)
    df["noise_flag"] = np.where(rng.rand(len(df)) < 0.5, "A", "B")
    rec.truth["noise_columns"].append("noise_flag")
    rec.inj("irrelevant categorical `noise_flag` (random A/B)")
    add_id(df, rec, "customer_id", 293, "string", "C")
    ft = declare(df, "group", omit=("customer_id",))
    single = True
    if variant == "traintest":
        single = False
        rec.truth["unseen"] = {"plan": "enterprise"}
    return finish(df, rec, root, "clustering", "group", ft=ft, single_csv=single), rec


SUITE2_BUILDERS = [c_churn, c_magic, c_pageblocks, c_creditapp, c_jannis, c_kick,
                   r_seoul, r_kc_house, r_moneyball, r_superconduct, r_brazil, r_wine,
                   k_wholesale, k_vehicle, k_ecoli, k_pendigits, k_segment, k_synth]


def build_stress():
    for fn, v in ((c_churn, "tokens"), (c_kick, "notypes"), (r_kc_house, "strnum"), (k_synth, "traintest")):
        out, rec = fn(STRESS2, v)
        if v == "traintest":
            B._plant_unseen(out, rec)
        print("stress", rec.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(SUITE2, exist_ok=True)
    rows = []
    for fn in SUITE2_BUILDERS:
        if a.only and fn.__name__ not in a.only:
            continue
        out, rec = fn()
        r = B.summary_row(out, rec)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if not a.only:
        build_stress()
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "suite_r2_summary.csv"), index=False)


if __name__ == "__main__":
    main()
