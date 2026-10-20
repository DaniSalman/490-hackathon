"""Column type inference, coercion and feature preparation (label-free, shared by all models).

`prepare(task)` returns a `Prepared` object holding:
    num_tr / num_te : float DataFrames (numeric, booleans 0/1, expanded datetimes, text stats, freq-enc)
    cat_tr / cat_te : str DataFrames of categorical columns ("__NA__" for missing)
    info            : decisions per column (for the profile)
plus encoder helpers (ordinal / one-hot / dense numeric).
"""
import re
import warnings

import numpy as np
import pandas as pd

# Tokens that mean "missing" ONLY inside a mostly-numeric column (e.g. "?" in a numeric UCI column).
# Categorical values are never altered except that empty / whitespace-only strings become missing:
# literal categories such as "none", "-", "unknown" or "missing" survive (credit-g own_telephone = "none").
NUMERIC_NA_TOKENS = {"?", "na", "n/a", "nan", "null", "none", "-", "--", ".", "missing", "#n/a", "nil",
                     "unknown", "unknown?", "inf", "-inf"}
# Reported (never altered) by the profiler as "suspected missing" in string columns.
SUSPECT_NA_TOKENS = {"?", "na", "n/a", "nan", "null", "#n/a", "missing"}
BOOL_TRUE = {"yes", "true", "y", "t", "1"}
BOOL_FALSE = {"no", "false", "n", "f", "0"}
_ID_EXACT = re.compile(r"^(id|uuid|guid|udi|index|idx|rowid|row_id|row|unnamed: ?0|key|record_id|sample_id)$", re.I)
_ID_SUFFIX = re.compile(r"([_\s\-.](id|uuid|guid|key|no|number|code)$)|([a-z0-9](ID|Id)$)")
NA_TOKEN = "__NA__"


def _name_tokens(name):
    """'WheelTypeID' -> ['wheel', 'type', 'id']; 'zip_code' -> ['zip', 'code']."""
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name)).replace("ID", " ID")
    return [t for t in re.split(r"[^A-Za-z0-9]+", parts.lower()) if t]


def is_code_name(name):
    """Names of integer-coded categoricals (used only when the type is NOT declared): zip / state / *code /
    trailing id token."""
    toks = _name_tokens(name)
    if not toks:
        return False
    low = str(name).lower()
    return ("zip" in low or any(t in ("state", "code") or t.endswith("code") for t in toks) or toks[-1] == "id")


def is_id_name(name):
    name = str(name).strip()
    return bool(_ID_EXACT.match(name) or _ID_SUFFIX.search(name))


def _is_stringlike(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))


def clean_strings(s):
    """Object/str series -> python-object series of stripped strings; empty/blank strings -> NaN."""
    v = s.astype(object)
    mask = v.notna()
    out = pd.Series(np.full(len(v), np.nan, dtype=object), index=s.index)
    if mask.any():
        st = v[mask].astype(str).str.strip()
        st = st.where(st != "", np.nan)
        out[mask] = st
    return out


def coerce_numeric(s):
    """Any series -> float series (NaN when not parseable). Handles '$1,234', '12%', ' ' blanks, bools."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").astype(float)
    else:
        st = clean_strings(s)
        low = st.str.lower()
        mapped = pd.Series(np.nan, index=s.index, dtype=float)
        mapped[low.isin(BOOL_TRUE)] = 1.0
        mapped[low.isin(BOOL_FALSE)] = 0.0
        cleaned = st.str.replace(r"[,$%\s€£]", "", regex=True)
        out = pd.to_numeric(cleaned, errors="coerce").astype(float)
        out = out.where(out.notna(), mapped)
    return out.replace([np.inf, -np.inf], np.nan)


def parse_datetime(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(s, errors="coerce")


def _looks_datetime(v):
    """v: non-null stripped strings."""
    sample = v.iloc[: 300] if len(v) > 300 else v
    if not sample.str.contains(r"\d", regex=True).mean() > 0.95:
        return False
    if not sample.str.contains(r"[-/:]|\d{4}[01]\d[0-3]\d", regex=True).mean() > 0.9:
        return False
    if pd.to_numeric(sample.str.replace(r"[,$%\s]", "", regex=True), errors="coerce").notna().mean() > 0.5:
        return False
    parsed = parse_datetime(sample)
    if parsed.notna().mean() < 0.9:
        return False
    years = parsed.dropna().dt.year
    return bool(len(years) and years.between(1800, 2200).mean() > 0.95)


def infer_column_type(s, name=""):
    """Data-driven type: numeric | categorical | boolean | datetime | text | id | constant."""
    nn = s.dropna()
    if pd.api.types.is_bool_dtype(s):
        return "constant" if nn.nunique() <= 1 else "boolean"
    if len(nn) == 0:
        return "constant"
    if _is_stringlike(s):
        nn = clean_strings(nn).dropna()
        if len(nn) == 0:
            return "constant"
    nun = nn.nunique()
    if nun <= 1:
        return "constant"
    uratio = nun / len(nn)
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        vals = nn.to_numpy(dtype=float)
        integer = np.all(np.isfinite(vals)) and np.all(np.floor(vals) == vals)
        if is_id_name(name) and integer and uratio > 0.95:
            return "id"
        if integer and uratio == 1.0 and len(nn) >= 50:
            d = np.diff(np.sort(vals))
            if d.size and np.all(d == 1):   # 0..n-1 / 1..n row counter
                return "id"
        if integer and nun > 2 and is_code_name(name):   # SH3: zip / state / code / ...ID integer codes
            return "categorical"
        return "numeric"
    v = nn.astype(str)
    low = set(v.str.lower().unique()[:5])
    if nun == 2 and low <= (BOOL_TRUE | BOOL_FALSE) and len(low & BOOL_TRUE) == 1:
        return "boolean"
    vv = v[~v.str.lower().isin(NUMERIC_NA_TOKENS)]          # "?" etc. do not block numeric detection
    num = pd.to_numeric(vv.str.replace(r"[,$%\s€£]", "", regex=True), errors="coerce")
    if len(vv) >= max(3, 0.5 * len(v)) and num.notna().mean() >= 0.95 and num.nunique() > 1:
        if is_id_name(name) and uratio > 0.95:
            return "id"
        return "numeric"
    if _looks_datetime(v):
        return "datetime"
    avg_len = v.str.len().mean()
    avg_words = v.str.count(r"\s+").mean() + 1
    if avg_words >= 4 and avg_len >= 25 and uratio > 0.5:
        return "text"
    if (uratio >= 0.98 and len(nn) >= 30) or (is_id_name(name) and uratio > 0.9):
        return "id"
    return "categorical"


def modelling_type(inferred, declared):
    """Combine the declared (metadata) type with the inferred one for modelling purposes."""
    if inferred == "constant":
        return "constant"
    if declared == "id" or (inferred == "id" and declared not in ("datetime", "text")):
        return "id"
    if declared in ("numeric", "categorical", "datetime", "text"):
        if declared == "categorical" and inferred == "boolean":
            return "boolean"
        return declared
    if declared == "boolean":
        return "boolean" if inferred in ("boolean", "numeric") else "categorical"
    return inferred


# --------------------------------------------------------------------------- prepare
class Prepared:
    def __init__(self):
        self.num_tr = None
        self.num_te = None
        self.cat_tr = None
        self.cat_te = None
        self.info = {}
        self.datetime_parts = []


def _col_hash(s):
    try:
        return int(pd.util.hash_pandas_object(s, index=False).sum() & 0xFFFFFFFFFFFF)
    except TypeError:
        return int(pd.util.hash_pandas_object(s.astype(str), index=False).sum() & 0xFFFFFFFFFFFF)


def find_duplicate_columns(df):
    """Return {dup_col: original_col} for exact duplicate columns (first occurrence kept)."""
    seen, dups = {}, {}
    for c in df.columns:
        h = _col_hash(df[c])
        for prev in seen.get(h, []):
            if df[c].equals(df[prev]):
                dups[c] = prev
                break
        else:
            seen.setdefault(h, []).append(c)
    return dups


def _datetime_features(dt, name, cyclic=False):
    feats = {}
    ok = dt.notna()
    feats[f"{name}__year"] = dt.dt.year
    feats[f"{name}__month"] = dt.dt.month
    feats[f"{name}__day"] = dt.dt.day
    feats[f"{name}__dow"] = dt.dt.dayofweek
    feats[f"{name}__doy"] = dt.dt.dayofyear
    if (dt.dt.hour.fillna(0) != 0).any():
        feats[f"{name}__hour"] = dt.dt.hour
    if (dt.dt.minute.fillna(0) != 0).any():
        feats[f"{name}__minute"] = dt.dt.minute
    base = dt[ok].min() if ok.any() else None
    if base is not None:
        feats[f"{name}__elapsed_days"] = (dt - base).dt.total_seconds() / 86400.0
    if cyclic:
        for part, period in (("month", 12), ("dow", 7), ("hour", 24)):
            key = f"{name}__{part}"
            if key in feats:
                ang = 2 * np.pi * feats[key].astype(float) / period
                feats[f"{key}_sin"] = np.sin(ang)
                feats[f"{key}_cos"] = np.cos(ang)
    return {k: pd.Series(v, index=dt.index).astype(float) for k, v in feats.items()}


def _text_tokens(st, col, ntr, max_tokens=30):
    """Binary presence of the most frequent tokens (vocabulary from training rows only)."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        docs = st.fillna("").astype(str)
        cv = CountVectorizer(max_features=max_tokens, binary=True, min_df=5, lowercase=True)
        cv.fit(docs.iloc[:ntr])
        M = cv.transform(docs).toarray().astype(float)
        return {f"{col}__tok_{w}": pd.Series(M[:, j], index=st.index) for w, j in cv.vocabulary_.items()}
    except Exception:
        return {}


def prepare(task, cyclic_datetime=False, text_features=True, freq_encode_threshold=50):
    """Label-free preparation. Type decisions use the training rows; conversions apply to train+test."""
    Xtr, Xte = task.X_train, task.X_test
    ntr = len(Xtr)
    full = pd.concat([Xtr, Xte], axis=0, ignore_index=True) if Xte is not None else Xtr.reset_index(drop=True)
    p = Prepared()
    info = {"columns": {}, "dropped_constant": [], "dropped_duplicate": {}, "dropped_id": [],
            "dropped_all_missing": [], "datetime_columns": [], "text_columns": []}
    dups = find_duplicate_columns(Xtr)
    num, cat = {}, {}
    for col in Xtr.columns:
        s_full = full[col]
        s_tr = s_full.iloc[:ntr]
        declared = task.feature_types.get(col)
        inferred = infer_column_type(s_tr, col)
        mt = modelling_type(inferred, declared)
        rec = {"declared": declared, "inferred": inferred, "modelling": mt}
        if col in dups:
            info["dropped_duplicate"][col] = dups[col]
            rec["action"] = "dropped_duplicate"
        elif mt == "constant":
            info["dropped_constant"].append(col)
            rec["action"] = "dropped_constant"
        elif mt == "id":
            info["dropped_id"].append(col)
            rec["action"] = "dropped_id"
        elif mt in ("numeric", "boolean"):
            v = coerce_numeric(s_full)
            if v.iloc[:ntr].notna().sum() == 0:
                if mt == "boolean" or _is_stringlike(s_full):   # e.g. two arbitrary strings
                    cat[col] = s_full
                    rec["action"] = "categorical"
                else:
                    info["dropped_all_missing"].append(col)
                    rec["action"] = "dropped_all_missing"
            elif mt == "boolean" and _is_stringlike(s_full) and v.iloc[:ntr].notna().mean() < 0.5:
                cat[col] = s_full
                rec["action"] = "categorical"
            else:
                num[col] = v
                rec["action"] = "numeric" if not _is_stringlike(s_full) else "converted_numeric"
        elif mt == "datetime":
            dt = parse_datetime(s_full if not _is_stringlike(s_full) else clean_strings(s_full))
            if dt.iloc[:ntr].notna().mean() < 0.5:
                cat[col] = s_full
                rec["action"] = "categorical"
            else:
                for k, v in _datetime_features(dt, col, cyclic_datetime).items():
                    num[k] = v
                info["datetime_columns"].append(col)
                rec["action"] = "expanded_datetime"
        elif mt == "text":
            info["text_columns"].append(col)
            if text_features:
                st = clean_strings(s_full)
                num[f"{col}__len"] = st.str.len().astype(float)
                num[f"{col}__words"] = (st.str.count(r"\s+") + 1).astype(float)
                num.update(_text_tokens(st, col, ntr))
                rec["action"] = "text_stats+tokens"
            else:
                rec["action"] = "dropped_text"
        else:
            cat[col] = s_full
            rec["action"] = "categorical"
        info["columns"][col] = rec

    # categoricals -> clean strings with NA token; frequency encoding for high cardinality
    cat_df = {}
    for col, s in cat.items():
        st = clean_strings(s)
        if pd.api.types.is_float_dtype(s):   # 1.0 -> "1"
            num_v = pd.to_numeric(s, errors="coerce")
            st = num_v.map(lambda x: NA_TOKEN if pd.isna(x) else (str(int(x)) if float(x).is_integer() else str(x)))
        st = st.where(st.notna(), NA_TOKEN).astype(str)
        cat_df[col] = st
        nun = st.iloc[:ntr].nunique()
        info["columns"][col]["cardinality"] = int(nun)
        if nun > freq_encode_threshold:
            freq = st.map(st.value_counts(normalize=True)).astype(float)
            num[f"{col}__freq"] = freq
    num_df = pd.DataFrame(num, index=full.index) if num else pd.DataFrame(index=full.index)
    cat_df = pd.DataFrame(cat_df, index=full.index) if cat_df else pd.DataFrame(index=full.index)
    if num_df.shape[1] == 0 and cat_df.shape[1] == 0:
        num_df["__const__"] = 0.0
    p.num_tr, p.cat_tr = num_df.iloc[:ntr].reset_index(drop=True), cat_df.iloc[:ntr].reset_index(drop=True)
    if Xte is not None:
        p.num_te, p.cat_te = num_df.iloc[ntr:].reset_index(drop=True), cat_df.iloc[ntr:].reset_index(drop=True)
    else:
        p.num_te, p.cat_te = None, None
    p.info = info
    return p


# --------------------------------------------------------------------------- targets
def encode_classes(y):
    """Raw target -> (codes int array, classes array, keep mask). Rows with missing target are dropped."""
    s = pd.Series(y).reset_index(drop=True)
    keep = s.notna().to_numpy()
    if _is_stringlike(s):
        s = clean_strings(s)
        keep = s.notna().to_numpy()
    vals = s[keep]
    if not _is_stringlike(vals):
        arr = vals.to_numpy()
        if pd.api.types.is_float_dtype(vals) and np.all(np.floor(arr) == arr):
            arr = arr.astype(np.int64)
        classes, codes = np.unique(arr, return_inverse=True)
    else:
        classes, codes = np.unique(vals.astype(str).to_numpy(), return_inverse=True)
    return codes.astype(np.int64), classes, keep


def encode_regression_target(y):
    v = coerce_numeric(pd.Series(y).reset_index(drop=True))
    keep = v.notna().to_numpy()
    return v.to_numpy(dtype=float)[keep], keep


# --------------------------------------------------------------------------- encoders
def ordinal_encode(cat_tr, cat_te, max_levels=250, min_count=1):
    """Codes 0..L-1 (most frequent first); rare/unseen levels -> NaN. Returns float arrays."""
    n_tr = len(cat_tr)
    A = np.empty((n_tr, cat_tr.shape[1]))
    B = np.empty((len(cat_te), cat_tr.shape[1])) if cat_te is not None else None
    for j, col in enumerate(cat_tr.columns):
        vc = cat_tr[col].value_counts()
        vc = vc[vc >= min_count].iloc[:max_levels]
        mapping = {lv: i for i, lv in enumerate(vc.index)}
        A[:, j] = cat_tr[col].map(mapping).astype(float).to_numpy()
        if B is not None:
            B[:, j] = cat_te[col].map(mapping).astype(float).to_numpy()
    return A, B


def max_encoded_features(p_before):
    """Cost guard: encoded width may not exceed max(5 p, p + 200) (p = raw feature count)."""
    return int(max(5 * p_before, p_before + 200))


def onehot_encode(cat_tr, cat_te, max_levels=20, min_count=2, scale=1.0, max_total=None):
    """Dense one-hot of the most frequent levels (others share an implicit all-zero row).

    max_total: cost guard on the number of dummy columns; max_levels is halved until it fits.
    """
    cols_a, cols_b, names = [], [], []
    vcs = {col: cat_tr[col].value_counts() for col in cat_tr.columns}
    if max_total is not None:
        def width(L):
            return sum(min(int((vc >= min_count).sum()), L) for vc in vcs.values())
        while max_levels > 1 and width(max_levels) > max_total:
            max_levels = max(1, max_levels // 2)
    for col in cat_tr.columns:
        vc = vcs[col]
        levels = list(vc[vc >= min_count].index[:max_levels])
        if len(levels) == len(vc) and len(levels) == 2:
            levels = levels[:1]       # binary -> single indicator
        if not levels:
            continue
        a = cat_tr[col].to_numpy()
        cols_a.append((a[:, None] == np.array(levels, dtype=object)[None, :]).astype(float) * scale)
        if cat_te is not None:
            b = cat_te[col].to_numpy()
            cols_b.append((b[:, None] == np.array(levels, dtype=object)[None, :]).astype(float) * scale)
        names += [f"{col}=={lv}" for lv in levels]
    n_tr = len(cat_tr)
    A = np.hstack(cols_a) if cols_a else np.zeros((n_tr, 0))
    B = (np.hstack(cols_b) if cols_b else np.zeros((len(cat_te), 0))) if cat_te is not None else None
    return A, B, names


def numeric_dense(num_tr, num_te, indicators=True, log_skewed=False, scale=True, robust=False,
                  ind_threshold=0.02):
    """Median-impute (+ missing indicators), optional log1p of skewed non-negative columns, scaling."""
    A = num_tr.to_numpy(dtype=float, copy=True)
    B = num_te.to_numpy(dtype=float, copy=True) if num_te is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        miss = np.isnan(A).mean(0) if A.size else np.zeros(A.shape[1])
        if log_skewed and A.shape[1]:
            mn = np.nanmin(A, 0)
            mu = np.nanmean(A, 0)
            sd = np.nanstd(A, 0)
            sk = np.nanmean(((A - mu) / np.where(sd > 0, sd, 1)) ** 3, 0)
            sel = np.where((mn >= 0) & (sk > 1.0))[0]
            if sel.size:
                A[:, sel] = np.log1p(A[:, sel])
                if B is not None:
                    B[:, sel] = np.log1p(np.clip(B[:, sel], 0, None))
        med = np.nanmedian(A, 0) if A.shape[0] else np.zeros(A.shape[1])
    med = np.where(np.isfinite(med), med, 0.0)
    ind_cols = np.where((miss >= ind_threshold) & (miss < 1.0))[0] if indicators else np.array([], int)
    IA = np.isnan(A[:, ind_cols]).astype(float)
    IB = np.isnan(B[:, ind_cols]).astype(float) if B is not None else None
    A = np.where(np.isnan(A), med, A)
    if B is not None:
        B = np.where(np.isnan(B), med, B)
    if scale and A.shape[1]:
        if robust:
            q1, q2, q3 = np.percentile(A, [25, 50, 75], axis=0)
            iqr = q3 - q1
            sd = A.std(0)
            denom = np.where(iqr > 0, iqr / 1.349, np.where(sd > 0, sd, 1.0))
            center = q2
        else:
            center = A.mean(0)
            sd = A.std(0)
            denom = np.where(sd > 0, sd, 1.0)
        A = (A - center) / denom
        if B is not None:
            B = (B - center) / denom
            B = np.clip(B, -50, 50)
    if ind_cols.size:
        A = np.hstack([A, IA])
        if B is not None:
            B = np.hstack([B, IB])
    return A, B
