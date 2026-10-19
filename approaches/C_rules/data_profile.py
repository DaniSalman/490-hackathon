"""Structured data profile (schema: docs/01_research_report.md section 4.1, docs/02 contract).

Computed on the RAW training table (pandas semantics) so that counts match a pandas-based checker.
Vectorised and cheap (O(n*p)).
"""
import warnings

import numpy as np
import pandas as pd

from cleaning import (SUSPECT_NA_TOKENS, _is_stringlike, find_duplicate_columns, infer_column_type, is_id_name,
                      modelling_type)

SCHEMA_VERSION = "1.0"


def _r(x, nd=6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _numeric_stats(s):
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if v.empty:
        return None
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skew = v.skew() if len(v) > 2 else 0.0
    return {"mean": _r(v.mean()), "std": _r(v.std()), "min": _r(v.min()), "q25": _r(q1), "median": _r(q2),
            "q75": _r(q3), "max": _r(v.max()), "skewness": _r(skew),
            "n_zeros": int((v == 0).sum()), "n_negative": int((v < 0).sum()),
            "n_outliers_iqr": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())}


def _top_values(s, k=5):
    vc = s.value_counts(dropna=True).iloc[:k]
    return [{"value": (v.item() if hasattr(v, "item") else v) if not isinstance(v, str) else v, "count": int(c)}
            for v, c in vc.items()]


def _suspect_counts(s):
    """(blank/whitespace-only strings, blank + NA-like tokens) in a string column; (0, 0) otherwise."""
    if not _is_stringlike(s):
        return 0, 0
    v = s.dropna().astype(str)
    if v.empty:
        return 0, 0
    st = v.str.strip()
    blank = int((st == "").sum())
    tok = int(st.str.lower().isin(SUSPECT_NA_TOKENS).sum())
    return blank, blank + tok


def build_profile(task, prepared_info=None):
    df = task.train_df
    test = task.test_df
    n = len(df)
    target = task.target
    feature_cols = [c for c in df.columns if c != target]
    dups = find_duplicate_columns(df[feature_cols]) if feature_cols else {}
    col_types = {"numeric": 0, "categorical": 0, "boolean": 0, "datetime": 0, "text": 0, "id": 0, "constant": 0}
    cols, constant, idlike, high_missing, high_card, dt_cols = [], [], [], [], [], []
    blank_tot, suspect_tot = [0], [0]
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        declared = task.feature_types.get(c)
        inferred = infer_column_type(s, c)
        role = "target" if c == target else ("id" if c == task.id_column else "feature")
        rec = {"name": c, "role": role, "raw_dtype": str(s.dtype), "declared_type": declared,
               "inferred_type": inferred,
               "type": modelling_type(inferred, declared) if role == "feature" else inferred,
               "n_missing": n_missing, "missing_ratio": _r(n_missing / n if n else 0.0),
               "n_unique": n_unique, "unique_ratio": _r(n_unique / max(1, n - n_missing)),
               "is_constant": bool(s.nunique(dropna=False) <= 1),
               "is_id_like": bool(inferred == "id" or (role == "id")),
               "duplicate_of": dups.get(c)}
        # blank / whitespace-only strings and NA-like tokens: reported only (pandas count above is unchanged)
        n_blank, n_suspect = _suspect_counts(s)
        rec["blank_string_count"] = n_blank
        rec["suspected_missing"] = n_suspect
        rec["n_missing_incl_suspected"] = n_missing + n_suspect
        blank_tot[0] += n_blank if role == "feature" else 0
        suspect_tot[0] += n_suspect if role == "feature" else 0
        if role == "id":
            col_types["id"] += 1
            idlike.append(c)
        if role == "feature":
            t = rec["type"]
            col_types[t] = col_types.get(t, 0) + 1
            if rec["is_constant"] or inferred == "constant":
                constant.append(c)
            if rec["is_id_like"]:
                idlike.append(c)
            if n and n_missing / n > 0.5:
                high_missing.append(c)
            if t == "categorical" and n_unique > 50:
                high_card.append(c)
            if t == "datetime":
                dt_cols.append(c)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            rec["numeric_stats"] = _numeric_stats(s)
        elif inferred == "numeric":
            from cleaning import coerce_numeric
            rec["numeric_stats"] = _numeric_stats(coerce_numeric(s))
        if inferred in ("categorical", "boolean", "text", "id") or _is_stringlike(s) or n_unique <= 20:
            rec["top_values"] = _top_values(s)
        if test is not None and c in test.columns and role == "feature" and rec["type"] in ("categorical", "boolean"):
            tr_levels = set(s.dropna().astype(str).str.strip())
            te = test[c].dropna().astype(str).str.strip()
            rec["unseen_test_levels"] = int(len(set(te) - tr_levels))
            rec["n_test_rows_unseen_level"] = int((~te.isin(tr_levels)).sum())
        if prepared_info and c in prepared_info.get("columns", {}):
            rec["action"] = prepared_info["columns"][c].get("action")
        cols.append(rec)
    total_missing = int(df[feature_cols].isna().sum().sum()) if feature_cols else 0
    prof = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": task.name,
        "task": task.task_type,
        "task_source": task.task_source,
        "target": target,
        "id_column": task.id_column,
        "n_rows": int(n),
        "n_rows_train": int(n),
        "n_rows_test": int(len(test)) if test is not None else 0,
        "n_rows_predicted": int(task.n_predict),
        "n_columns": int(df.shape[1]),
        "n_features": int(len([c for c in feature_cols if c != task.id_column])),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": total_missing,
        "total_blank_string_cells": int(blank_tot[0]),
        "total_suspected_missing_cells": int(suspect_tot[0]),
        "total_missing_cells_incl_suspected": int(total_missing + suspect_tot[0]),
        "missing_cell_ratio": _r(total_missing / max(1, n * max(1, len(feature_cols)))),
        "n_columns_with_missing": int((df[feature_cols].isna().sum() > 0).sum()) if feature_cols else 0,
        "column_type_counts": col_types,
        "constant_columns": constant,
        "duplicate_columns": [[orig, d] for d, orig in dups.items()],
        "id_like_columns": idlike,
        "high_missing_columns": high_missing,
        "high_cardinality_columns": high_card,
        "datetime_columns": dt_cols,
        "columns": cols,
        "target_profile": _target_profile(task),
        "warnings": [],
    }
    if test is not None:
        prof["test_missing_cells"] = int(test.reindex(columns=[c for c in feature_cols if c in test.columns]).isna().sum().sum())
        missing_cols = [c for c in feature_cols if c not in test.columns and c != task.id_column]
        if missing_cols:
            prof["warnings"].append(f"columns_missing_in_test: {missing_cols}")
    for c in idlike:
        if not is_id_name(c):
            prof["warnings"].append(f"id_like_by_values: {c}")
    return prof


def _target_profile(task):
    if task.task_type == "clustering":
        return {"n_clusters_required": task.n_clusters}
    y = task.y_train
    out = {"name": task.target, "n_missing": int(y.isna().sum())}
    if task.task_type == "classification":
        vc = y.value_counts(dropna=True)
        tot = int(vc.sum())
        out.update({
            "n_classes": int(len(vc)),
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "class_ratios": {str(k): _r(v / tot) for k, v in vc.items()},
            "majority_class": str(vc.index[0]) if len(vc) else None,
            "minority_class": str(vc.index[-1]) if len(vc) else None,
            "imbalance_ratio": _r(vc.iloc[0] / vc.iloc[-1]) if len(vc) else None,
            "is_imbalanced": bool(len(vc) and vc.iloc[0] / vc.iloc[-1] > 1.5),
        })
    else:
        from cleaning import coerce_numeric
        v = coerce_numeric(y).dropna()
        st = _numeric_stats(v) or {}
        out.update(st)
        out["is_nonnegative"] = bool(len(v) and v.min() >= 0)
        out["is_integer"] = bool(len(v) and np.all(np.floor(v) == v))
        out["log_transform_suggested"] = bool(out["is_nonnegative"] and (st.get("skewness") or 0) > 1.0)
    return out


def enrich_with_preparation(profile, prep_info):
    """Add the per-column modelling action (kept / dropped_* / expanded_datetime ...) to the profile."""
    if not profile or not prep_info:
        return profile
    cols = prep_info.get("columns", {})
    for rec in profile.get("columns", []):
        c = cols.get(rec.get("name"))
        if c:
            rec["action"] = c.get("action")
    profile["preparation"] = {k: v for k, v in prep_info.items() if k != "columns"}
    return profile
