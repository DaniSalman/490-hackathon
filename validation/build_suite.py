"""Build the 18-dataset private-like validation suite (+ separate stress probes).

    py validation/build_suite.py                 # suite -> validation/data/suite/<name>/
                                                 # stress -> validation/data/stress/<name>/
    py validation/build_suite.py --only c_telco  # rebuild some datasets

Sources (downloaded once into validation/cache/, gitignored; never committed):
  * OpenML ARFF files fetched through the public REST API (sklearn.fetch_openml hung on this machine,
    so the ARFF is downloaded with urllib and parsed with sklearn's vendored liac-arff parser).
  * UCI zip archives (Bike Sharing hour.csv, Dry Bean ARFF).
  * Mall_Customers.csv (Kaggle file, public GitHub mirror).
Downloaded files are treated as data only (parsed, never executed).

Every dataset folder follows docs/02 (tools/make_dataset.py writes it). In addition each folder gets
`hidden/truth.json` describing the planted / known challenges (ID columns, constant / duplicate
columns, noise columns, injected missingness, datetime columns) used for profile scoring.
All randomness uses fixed seeds; rebuilding gives byte-identical folders.
"""
import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_dataset import make_dataset  # noqa: E402

CACHE = os.path.join(HERE, "cache")
SUITE = os.path.join(HERE, "data", "suite")
STRESS = os.path.join(HERE, "data", "stress")

UCI_BIKE = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
UCI_DRYBEAN = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
MALL_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"


# --------------------------------------------------------------------------- download helpers
def _download(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "eece490-validator"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return path


def _arff_to_frame(text):
    from sklearn.externals import _arff
    d = _arff.loads(text)
    cols = [a[0] for a in d["attributes"]]
    df = pd.DataFrame(d["data"], columns=cols)
    for (name, typ) in d["attributes"]:
        if isinstance(typ, list) or str(typ).upper() == "STRING":
            df[name] = df[name].astype(object).where(df[name].notna(), None)
        else:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def openml(did):
    """OpenML dataset -> DataFrame (all attributes, incl. default target). Cached as pickle."""
    pk = os.path.join(CACHE, f"openml_{did}.pkl")
    if os.path.isfile(pk):
        return pd.read_pickle(pk)
    desc_p = _download(f"https://www.openml.org/api/v1/json/data/{did}",
                       os.path.join(CACHE, "raw", f"openml_{did}", "description.json"))
    with open(desc_p, encoding="utf-8") as f:
        desc = json.load(f)["data_set_description"]
    arff_p = _download(desc["url"], os.path.join(CACHE, "raw", f"openml_{did}", "data.arff"))
    with open(arff_p, encoding="utf-8", errors="replace") as f:
        df = _arff_to_frame(f.read())
    df.to_pickle(pk)
    return df


def uci_zip_member(url, key, member):
    zp = _download(url, os.path.join(CACHE, "raw", key, "archive.zip"))
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.endswith(member)][0]
        return z.read(name)


# --------------------------------------------------------------------------- injection helpers
class Rec:
    """Records every injection (ground truth for profile scoring + SUITE.md)."""

    def __init__(self, name):
        self.name = name
        self.truth = {"id_columns": [], "constant_columns": [], "duplicate_pairs": [], "noise_columns": [],
                      "datetime_columns": [], "injected_missing": {}, "injections": [], "native": [],
                      "label_note": None}

    def inj(self, msg):
        self.truth["injections"].append(msg)

    def nat(self, msg):
        self.truth["native"].append(msg)


def add_constant(df, rec, name, value):
    df[name] = value
    rec.truth["constant_columns"].append(name)
    rec.inj(f"constant column `{name}`={value!r}")
    return df


def add_duplicate(df, rec, src, name):
    df[name] = df[src].copy()
    rec.truth["duplicate_pairs"].append([src, name])
    rec.inj(f"duplicate column `{name}` (copy of `{src}`)")
    return df


def add_noise(df, rec, n, seed, prefix="noise", scale_big=True):
    rng = np.random.RandomState(seed)
    for j in range(n):
        nm = f"{prefix}_{j + 1}"
        if j % 3 == 0:
            df[nm] = rng.normal(0, 1, len(df)).round(4)
        elif j % 3 == 1:
            df[nm] = (rng.uniform(0, 1000 if scale_big else 1, len(df))).round(3)
        else:
            df[nm] = rng.exponential(50, len(df)).round(3)
        rec.truth["noise_columns"].append(nm)
    rec.inj(f"{n} irrelevant noise features `{prefix}_1..{n}` (normal / uniform x1000 / exponential)")
    return df


def add_id(df, rec, name, seed, kind="string", prefix="A"):
    rng = np.random.RandomState(seed)
    ids = rng.permutation(len(df)) + 10000
    df.insert(0, name, [f"{prefix}{i:06d}" for i in ids] if kind == "string" else ids)
    rec.truth["id_columns"].append(name)
    rec.inj(f"ID column `{name}` ({kind}, unique per row, shuffled) NOT declared in metadata")
    return df


def add_mcar(df, rec, cols, frac, seed):
    rng = np.random.RandomState(seed)
    for c in cols:
        m = rng.rand(len(df)) < frac
        df[c] = df[c].mask(m)
        rec.truth["injected_missing"][c] = int(m.sum())
    rec.inj(f"MCAR missingness {int(frac * 100)}% in {cols}")
    return df


def declare(df, target=None, id_column=None, overrides=None, omit=()):
    """Realistic declared feature types: numeric dtypes -> numeric, else categorical; overrides win."""
    ft = {}
    for c in df.columns:
        if c in (target, id_column) or c in omit:
            continue
        s = df[c]
        ft[c] = "numeric" if (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)) else "categorical"
    ft.update(overrides or {})
    return ft


def finish(df, rec, out_root, task, target=None, n_clusters=None, id_column=None, ft=None, single_csv=True,
           extra_meta=None, write=True):
    out = os.path.join(out_root, rec.name)
    if id_column:
        rec.truth["id_columns"] = [id_column] + [c for c in rec.truth["id_columns"] if c != id_column]
    meta = make_dataset(df, out, task, target=target, n_clusters=n_clusters, id_column=id_column,
                        feature_types=ft if ft is not None else None, seed=0, single_csv=single_csv,
                        name=rec.name, extra_meta=extra_meta)
    rec.truth.update(task=task, target=target, n_clusters=meta.get("n_clusters"), id_column=id_column,
                     n_total_rows=int(len(df)), declared_feature_types=ft)
    with open(os.path.join(out, "hidden", "truth.json"), "w", encoding="utf-8") as f:
        json.dump(rec.truth, f, indent=1)
    return out


# --------------------------------------------------------------------------- classification
def c_telco(root=SUITE, variant=None):
    df = openml(42178)
    rec = Rec("c_telco" if variant is None else f"s_telco_{variant}")
    # Kaggle file: TotalCharges is a string column with 11 " " blanks (new customers, tenure 0)
    df["TotalCharges"] = df["TotalCharges"].astype(object).where(df["TotalCharges"].notna(), " ")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    rec.nat("Kaggle WA_Fn-UseC_-Telco-Customer-Churn (OpenML 42178), 7043 x 21")
    rec.nat("TotalCharges numeric stored as string with 11 ' ' blanks; customerID ID; churn 26.5 % (imbalanced)")
    ft = declare(df, "Churn", "customerID", {"TotalCharges": "numeric", "SeniorCitizen": "categorical"})
    if variant == "missing":
        add_mcar(df, rec, [c for c in df.columns if c not in ("Churn", "customerID")], 0.20, 101)
    elif variant == "strnum":
        df["MonthlyCharges"] = df["MonthlyCharges"].map(lambda v: f"${v:,.2f}")
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = tc.map(lambda v: " " if pd.isna(v) else f"{v:,.2f}")
        df["tenure"] = df["tenure"].map(lambda v: f"{v:,d}")
        rec.inj("MonthlyCharges as '$1,234.50', TotalCharges as '1,234.50', tenure as string; NO feature_types in metadata")
        ft = None
    elif variant == "unseen":
        rec.truth["unseen"] = {"PaymentMethod": "Crypto wallet", "InternetService": "Satellite", "Contract": "Weekly"}
    return finish(df, rec, root, "classification", "Churn", id_column="customerID", ft=ft), rec


def c_ai4i(root=SUITE):
    df = openml(42890)
    rec = Rec("c_ai4i")
    df = df.rename(columns={c: c.replace("_", " ") for c in df.columns})
    # Recreate the Kaggle/UCI header names
    ren = {"Air temperature K": "Air temperature [K]", "Process temperature K": "Process temperature [K]",
           "Rotational speed rpm": "Rotational speed [rpm]", "Torque Nm": "Torque [Nm]",
           "Tool wear min": "Tool wear [min]"}
    df = df.rename(columns=ren)
    tgt = [c for c in df.columns if c.lower().startswith("machine")][0]
    df[tgt] = pd.to_numeric(df[tgt]).astype(int)
    for c in ("TWF", "HDF", "PWF", "OSF", "RNF", "UDI"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c]).astype(int)
    rec.nat("UCI AI4I 2020 predictive maintenance (OpenML 42890), 10000 x 14; failure rate 3.4 % (strongly imbalanced)")
    rec.nat("UDI row counter and Product ID (unique string) are IDs; failure-mode flags TWF/HDF/PWF/OSF/RNF present in train AND test (as in the Kaggle file)")
    rec.truth["id_columns"].append("Product ID")
    ft = declare(df, tgt, "UDI", {"Type": "categorical", "Product ID": "categorical",
                                   **{c: "categorical" for c in ("TWF", "HDF", "PWF", "OSF", "RNF")}})
    return finish(df, rec, root, "classification", tgt, id_column="UDI", ft=ft), rec


def c_adult(root=SUITE):
    df = openml(1590).sample(n=20000, random_state=0).reset_index(drop=True)
    rec = Rec("c_adult")
    df["class"] = df["class"].astype(str).str.strip()
    rec.nat("UCI Adult census income (OpenML 1590), 20000-row random subsample of 48842; >50K = 24 % (imbalanced)")
    rec.nat("native missing in workclass/occupation/native-country; native-country 41 levels; fnlwgt ~irrelevant; education-num is a re-encoding of education")
    ft = declare(df, "class")
    return finish(df, rec, root, "classification", "class", ft=ft), rec


def c_bank(root=SUITE):
    df = openml(1461).sample(n=20000, random_state=1).reset_index(drop=True)
    rec = Rec("c_bank")
    names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
             "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    df = df.rename(columns={f"V{i + 1}": n for i, n in enumerate(names)})
    df["y"] = df["Class"].astype(str).map({"1": "no", "2": "yes"})
    df = df.drop(columns=["Class"])
    rec.nat("UCI Bank Marketing (OpenML 1461), 20000-row subsample of 45211; subscribed 'yes' = 11.7 % (imbalanced)")
    rec.nat("'unknown' pseudo-missing levels; pdays=-1 sentinel; duration strongly predictive")
    add_constant(df, rec, "campaign_channel", "phone")
    add_duplicate(df, rec, "balance", "balance_eur")
    add_noise(df, rec, 5, 11)
    ft = declare(df, "y", overrides={"day": "numeric"})
    return finish(df, rec, root, "classification", "y", ft=ft), rec


def c_covertype(root=SUITE):
    full = openml(180)
    df = full.sample(n=15000, random_state=2).reset_index(drop=True)
    rec = Rec("c_covertype")
    tgt = full.columns[-1]
    df[tgt] = df[tgt].astype(str)
    for c in df.columns:
        if c.lower().startswith(("wilderness", "soil")):
            df[c] = pd.to_numeric(df[c]).astype(int)   # Kaggle/UCI file stores them as 0/1 ints
    rec.nat(f"UCI Covertype (OpenML 180, 110393-row version), 15000-row subsample; 7 classes, multiclass imbalance; target `{tgt}`")
    rec.nat("44 one-hot binary wilderness/soil columns (some all-zero in the subsample -> natural constant columns); nonlinear")
    ft = declare(df, tgt)
    return finish(df, rec, root, "classification", tgt, ft=ft), rec


def c_creditg(root=SUITE):
    df = openml(31)
    rec = Rec("c_creditg")
    rec.nat("Statlog German credit (OpenML 31), 1000 x 21, 13 categoricals, bad = 30 %")
    add_mcar(df, rec, ["duration", "credit_amount", "purpose", "employment", "age"], 0.10, 21)
    add_duplicate(df, rec, "credit_amount", "loan_amount_copy")
    add_noise(df, rec, 6, 22)
    rng = np.random.RandomState(23)
    d0 = pd.Timestamp("2018-01-01")
    df["application_date"] = [(d0 + pd.Timedelta(days=int(x))).strftime("%Y-%m-%d") for x in rng.randint(0, 900, len(df))]
    rec.truth["datetime_columns"].append("application_date")
    rec.inj("irrelevant datetime column `application_date` (YYYY-MM-DD strings), declared datetime")
    add_id(df, rec, "applicant_id", 24, "string", "APP")
    ft = declare(df, "class", overrides={"application_date": "datetime"}, omit=("applicant_id",))
    return finish(df, rec, root, "classification", "class", ft=ft), rec


# --------------------------------------------------------------------------- regression
def r_house(root=SUITE, variant=None):
    df = openml(42165)
    rec = Rec("r_house" if variant is None else f"s_house_{variant}")
    df["Id"] = df["Id"].astype(int)
    rec.nat("Kaggle House Prices / Ames (OpenML 42165), 1460 x 81; 19 columns with missing (PoolQC 99.5 %), 43 categoricals, right-skewed SalePrice, Id")
    ft = declare(df, "SalePrice", "Id")
    if variant == "missing":
        add_mcar(df, rec, [c for c in df.columns if c not in ("SalePrice", "Id")], 0.20, 102)
    elif variant == "strnum":
        for c in ("LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF"):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,d}")
        df["MasVnrArea"] = df["MasVnrArea"].map(lambda v: "NA" if pd.isna(v) else f"{v:,.0f} sqft".replace(" sqft", ""))
        rec.inj("LotArea/GrLivArea/TotalBsmtSF/1stFlrSF as '12,345' strings, MasVnrArea with 'NA'; NO feature_types in metadata")
        ft = None
    elif variant == "unseen":
        rec.truth["unseen"] = {"Neighborhood": "Lakeview", "SaleCondition": "Auction", "KitchenQual": "Ex+"}
    return finish(df, rec, root, "regression", "SalePrice", id_column="Id", ft=ft), rec


def r_bike(root=SUITE):
    raw = uci_zip_member(UCI_BIKE, "uci_bike", "hour.csv")
    h = pd.read_csv(io.BytesIO(raw))
    rec = Rec("r_bike")
    dt = pd.to_datetime(h["dteday"]) + pd.to_timedelta(h["hr"], unit="h")
    df = pd.DataFrame({
        "datetime": dt.dt.strftime("%Y-%m-%d %H:%M:%S"), "season": h["season"], "holiday": h["holiday"],
        "workingday": h["workingday"], "weather": h["weathersit"], "temp": (h["temp"] * 41).round(2),
        "atemp": (h["atemp"] * 50).round(3), "humidity": (h["hum"] * 100).round(0).astype(int),
        "windspeed": (h["windspeed"] * 67).round(4), "count": h["cnt"]})
    rec.truth["datetime_columns"].append("datetime")
    rec.nat("UCI Bike Sharing hour.csv (17379 rows) rebuilt in the Kaggle Bike Sharing Demand schema: `datetime` string, de-normalised temp/atemp/humidity/windspeed; casual/registered dropped (absent from the Kaggle test file)")
    rec.nat("right-skewed count target (RMSLE in Kaggle), hour-of-day signal only reachable through the datetime column, nonlinear")
    ft = declare(df, "count", overrides={"datetime": "datetime", "season": "categorical", "weather": "categorical",
                                         "holiday": "categorical", "workingday": "categorical"})
    return finish(df, rec, root, "regression", "count", ft=ft), rec


def r_california(root=SUITE):
    df = openml(43939)
    rec = Rec("r_california")
    rec.nat("California housing, Kaggle/Geron version (OpenML 43939), 20640 x 10; ocean_proximity categorical; 207 missing total_bedrooms; capped target")
    lat = (df["latitude"] * 4).round().astype(int)
    lon = (df["longitude"] * 4).round().astype(int)
    df["geo_cell"] = "G" + lat.astype(str) + "_" + (-lon).astype(str)
    rec.inj(f"high-cardinality categorical `geo_cell` (0.25-degree lat/lon grid, {df['geo_cell'].nunique()} levels)")
    add_noise(df, rec, 3, 31)
    ft = declare(df, "median_house_value")
    return finish(df, rec, root, "regression", "median_house_value", ft=ft), rec


def r_abalone(root=SUITE):
    df = openml(183)
    rec = Rec("r_abalone")
    df = df.rename(columns={"Class_number_of_rings": "Rings"})
    df["Rings"] = pd.to_numeric(df["Rings"]).astype(int)
    rec.nat("UCI Abalone (OpenML 183), 4177 x 9; integer target Rings; categorical Sex; heteroscedastic noise")
    add_duplicate(df, rec, "Whole_weight", "Whole_weight_g")
    add_constant(df, rec, "survey_year", 1994)
    ft = declare(df, "Rings")
    return finish(df, rec, root, "regression", "Rings", ft=ft), rec


def r_concrete(root=SUITE):
    df = openml(4353)
    rec = Rec("r_concrete")
    tgt = [c for c in df.columns if "strength" in c.lower()][0]
    rec.nat(f"UCI Concrete compressive strength (OpenML 4353), 1030 x 9; strongly nonlinear (age, water/cement); target `{tgt}`")
    feats = [c for c in df.columns if c != tgt]
    add_mcar(df, rec, feats[:4], 0.10, 41)
    add_noise(df, rec, 5, 42)
    df.insert(0, "sample_no", np.arange(1, len(df) + 1))
    rec.truth["id_columns"].append("sample_no")
    rec.inj("row-counter ID column `sample_no` (1..n, in file order) NOT declared in metadata")
    ft = declare(df, tgt, omit=("sample_no",))
    return finish(df, rec, root, "regression", tgt, ft=ft), rec


def r_diamonds(root=SUITE):
    df = openml(42225).sample(n=15000, random_state=5).reset_index(drop=True)
    rec = Rec("r_diamonds")
    rec.nat("diamonds (OpenML 42225), 15000-row subsample of 53940; price strongly right-skewed (multiplicative); ordinal categoricals cut/color/clarity; x/y/z zeros = disguised missing")
    ft = declare(df, "price")
    return finish(df, rec, root, "regression", "price", ft=ft), rec


# --------------------------------------------------------------------------- clustering
def k_mall(root=SUITE):
    p = _download(MALL_URL, os.path.join(CACHE, "raw", "mall", "Mall_Customers.csv"))
    df = pd.read_csv(p)
    rec = Rec("k_mall")
    from sklearn.cluster import KMeans
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].to_numpy(float)
    lab = KMeans(5, n_init=100, random_state=0).fit_predict(X)
    df["segment"] = lab
    rec.truth["label_note"] = ("Mall Customers has no ground-truth labels. Hidden labels = the canonical 5 "
                               "segments, KMeans(k=5, n_init=100, seed 0) on unscaled (Annual Income, Spending Score), "
                               "the partition every Kaggle tutorial reports. ASSUMPTION about how the organisers label it.")
    rec.nat("Kaggle Mall Customer Segmentation (200 x 5): CustomerID, Gender, Age, Annual Income, Spending Score; k = 5")
    ft = declare(df, "segment", "CustomerID", {"Gender": "categorical"})
    return finish(df, rec, root, "clustering", "segment", n_clusters=5, id_column="CustomerID", ft=ft), rec


def k_iris(root=SUITE):
    df = openml(61)
    rec = Rec("k_iris")
    rec.nat("Fisher iris (OpenML 61), 150 x 4, k = 3")
    add_noise(df, rec, 3, 51, scale_big=False)
    add_constant(df, rec, "lab", "Kew")
    add_id(df, rec, "specimen_id", 52, "int")
    ft = declare(df, "class", omit=("specimen_id",))
    return finish(df, rec, root, "clustering", "class", ft=ft), rec


def k_wine(root=SUITE):
    df = openml(187)
    rec = Rec("k_wine")
    tgt = "class"
    df[tgt] = df[tgt].astype(str)
    rec.nat("UCI wine (OpenML 187), 178 x 13, k = 3; features on very different scales (proline ~1000, hue ~1) -> scaling is decisive")
    ft = declare(df, tgt)
    return finish(df, rec, root, "clustering", tgt, ft=ft), rec


def k_seeds(root=SUITE):
    df = openml(1499)
    df.columns = ["area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry",
                  "groove_length", "variety"]
    rec = Rec("k_seeds")
    tgt = df.columns[-1]
    rec.nat(f"UCI seeds (OpenML 1499), 210 x 7, k = 3 wheat varieties; target `{tgt}`")
    ft = declare(df, tgt)
    return finish(df, rec, root, "clustering", tgt, ft=ft), rec


def k_penguins(root=SUITE, variant=None):
    df = openml(42585)
    rec = Rec("k_penguins" if variant is None else f"s_penguins_{variant}")
    tgt = "species"
    rec.nat("Palmer penguins (OpenML 42585), 344 x 7, k = 3 species; categoricals island/sex; native missing (sex 10, measurements 2 each)")
    ft = declare(df, tgt)
    single = True
    if variant == "missing":
        add_mcar(df, rec, [c for c in df.columns if c != tgt], 0.20, 103)
    elif variant == "strnum":
        df["body_mass_g"] = df["body_mass_g"].map(lambda v: "" if pd.isna(v) else f"{int(v):,d}")
        df["flipper_length_mm"] = df["flipper_length_mm"].map(lambda v: "?" if pd.isna(v) else str(int(v)))
        rec.inj("body_mass_g as '3,750' strings, flipper_length_mm as strings with '?'; NO feature_types in metadata")
        ft = None
    elif variant == "unseen":
        single = False
        rec.truth["unseen"] = {"island": "Anvers", "sex": "unknown"}
    return finish(df, rec, root, "clustering", tgt, ft=ft, single_csv=single), rec


def k_drybean(root=SUITE):
    raw = uci_zip_member(UCI_DRYBEAN, "uci_drybean", "Dry_Bean_Dataset.arff")
    df = _arff_to_frame(raw.decode("utf-8", errors="replace"))
    rec = Rec("k_drybean")
    tgt = "Class"
    rec.nat("UCI Dry Bean (Koklu & Ozkan 2020), 13611 x 16, k = 7 bean varieties; scales from 1e-3 (ShapeFactor) to 2e5 (ConvexArea); highly correlated shape features")
    add_duplicate(df, rec, "Area", "Area_px")
    ft = declare(df, tgt)
    return finish(df, rec, root, "clustering", tgt, ft=ft), rec


SUITE_BUILDERS = [c_telco, c_ai4i, c_adult, c_bank, c_covertype, c_creditg,
                  r_house, r_bike, r_california, r_abalone, r_concrete, r_diamonds,
                  k_mall, k_iris, k_wine, k_seeds, k_penguins, k_drybean]


# --------------------------------------------------------------------------- stress probes
def _plant_unseen(out, rec):
    """Replace ~15 % of the test rows' values with levels never seen in train."""
    unseen = rec.truth.get("unseen") or {}
    test_p = os.path.join(out, "test.csv")
    te = pd.read_csv(test_p, low_memory=False, keep_default_na=False, na_values=[""])
    rng = np.random.RandomState(7)
    for c, new in unseen.items():
        m = rng.rand(len(te)) < 0.15
        te[c] = te[c].astype(object)
        te.loc[m, c] = new
        rec.truth.setdefault("unseen_rows", {})[c] = int(m.sum())
    te.to_csv(test_p, index=False)
    rec.inj(f"unseen test categories (15 % of test rows): {unseen}")
    with open(os.path.join(out, "hidden", "truth.json"), "w", encoding="utf-8") as f:
        json.dump(rec.truth, f, indent=1)


def build_stress():
    for fn in (c_telco, r_house, k_penguins):
        for v in ("missing", "strnum", "unseen"):
            out, rec = fn(STRESS, v)
            if v == "unseen":
                _plant_unseen(out, rec)
            print("stress", rec.name, "->", out, flush=True)


def summary_row(out, rec):
    tr = os.path.join(out, "train.csv") if os.path.isfile(os.path.join(out, "train.csv")) else os.path.join(out, "data.csv")
    d = pd.read_csv(tr, low_memory=False)
    t = rec.truth
    info = {"name": rec.name, "task": t["task"], "n_total": t["n_total_rows"], "n_train_file": len(d),
            "n_cols_file": d.shape[1], "missing_cells": int(d.isna().sum().sum())}
    if t["task"] == "classification":
        vc = d[t["target"]].value_counts()
        info["classes"] = int(len(vc))
        info["minority_ratio"] = round(float(vc.iloc[-1] / vc.sum()), 4)
    if t["task"] == "clustering":
        info["k"] = t["n_clusters"]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-stress", action="store_true")
    a = ap.parse_args()
    os.makedirs(SUITE, exist_ok=True)
    rows = []
    for fn in SUITE_BUILDERS:
        if a.only and fn.__name__ not in a.only:
            continue
        out, rec = fn()
        r = summary_row(out, rec)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if not a.no_stress and not a.only:
        build_stress()
    if rows and not a.only:
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "suite_summary.csv"), index=False)


if __name__ == "__main__":
    main()
