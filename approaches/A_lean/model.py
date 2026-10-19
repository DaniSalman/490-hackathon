"""Approach A -- lean & robust: one well-configured model family per task.

classification : HistGradientBoosting (native categoricals/NaN), balanced weights when imbalanced,
                 K-fold bagging (fold models averaged on test), OOF-tuned class weights for macro-F1;
                 tiny data (n<500 or <100 rows/class): blend with logistic regression chosen by OOF log-loss.
regression     : HistGradientBoosting; raw vs log1p target picked by OOF RMSE (original scale);
                 tiny data: blend with ridge; predictions clipped to a widened training range.
clustering     : impute -> log1p skewed -> standardise; one-hot/sqrt(2) categoricals; PCA when wide;
                 KMeans(k) with many restarts.
"""
import time
import warnings

import numpy as np

from cleaning import encode_classes, encode_regression_target, numeric_dense, onehot_encode, ordinal_encode, prepare
from utils import (SEED, align_proba, choose_k_silhouette, ensure_k_labels, get_n_jobs, hgb_wide_params, cap_cat_codes, log_loss,
                   macro_f1, make_folds, r2, rmse, tune_class_weights)

warnings.filterwarnings("ignore")
RESERVE = 4.0   # seconds kept back for writing outputs


def solve(task, budget, emit, log):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(get_n_jobs()):
        prep = prepare(task)
        log(f"prepared: {prep.num_tr.shape[1]} numeric, {prep.cat_tr.shape[1]} categorical features")
        if task.task_type == "classification":
            info = _classification(task, prep, budget, emit, log)
        elif task.task_type == "regression":
            info = _regression(task, prep, budget, emit, log)
        else:
            info = _clustering(task, prep, budget, emit, log)
    info["approach"] = "A_lean"
    info["_prep_info"] = prep.info
    return info


# --------------------------------------------------------------------------- features
def _tree_matrix(prep, keep=None):
    A_num = prep.num_tr.to_numpy(dtype=float)
    B_num = prep.num_te.to_numpy(dtype=float)
    A_cat, B_cat = ordinal_encode(prep.cat_tr, prep.cat_te, max_levels=250)
    A = np.hstack([A_num, A_cat])
    B = np.hstack([B_num, B_cat])
    mask = np.array([False] * A_num.shape[1] + [True] * A_cat.shape[1])
    if keep is not None:
        A = A[keep]
    return A, B, mask


def _linear_matrix(prep, keep=None):
    A, B = numeric_dense(prep.num_tr, prep.num_te, indicators=True, log_skewed=True)
    C, D, _ = onehot_encode(prep.cat_tr, prep.cat_te, max_levels=30)
    A, B = np.hstack([A, C]), np.hstack([B, D])
    if keep is not None:
        A = A[keep]
    return A, B


def _hgb_params(n, task_type):
    if n < 1000:
        p = dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=max(5, n // 60),
                 l2_regularization=1.0, early_stopping=False)
    else:
        p = dict(learning_rate=0.08, max_iter=800, max_leaf_nodes=31, min_samples_leaf=20,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.12, n_iter_no_change=30)
    p["random_state"] = SEED
    return p


def _n_folds(n):
    return 5 if n <= 30000 else 3


# --------------------------------------------------------------------------- classification
def capped_balanced_weights(counts, cap=25.0):
    """'balanced' class weights, capped so near-singleton classes do not dominate training."""
    n, present = counts.sum(), counts > 0
    w = np.where(present, n / (present.sum() * np.maximum(counts, 1)), 1.0)
    w = np.minimum(w, cap * w[present].min())
    return {int(c): float(w[c]) for c in np.where(present)[0]}


def _classification(task, prep, budget, emit, log):
    from sklearn.ensemble import HistGradientBoostingClassifier

    y, classes, keep = encode_classes(task.y_train)
    C = len(classes)
    info = {"model": "HistGradientBoostingClassifier", "n_classes": int(C)}
    if C < 2:
        emit(np.repeat(classes[0] if C else 0, task.n_predict), "single_class")
        info["model"] = "constant"
        return info
    A, B, mask = _tree_matrix(prep, keep)
    n = len(y)
    counts = np.bincount(y, minlength=C)
    imb = counts.max() / max(1, counts[counts > 0].min())
    params = _hgb_params(n, "classification")
    params.update(hgb_wide_params(A.shape[1]))
    if "max_bins" in params:
        A, B = cap_cat_codes(mask, params["max_bins"], A, B)
    if imb > 3:
        params["class_weight"] = capped_balanced_weights(counts)
    params["categorical_features"] = mask if mask.any() else None
    info.update(imbalance_ratio=float(imb), class_weight="balanced_capped" if imb > 3 else None)

    folds = make_folds(y, _n_folds(n), "classification")
    oof = np.full((n, C), np.nan)
    test_p = np.zeros((len(B), C))
    done, fold_t = 0, []
    for tr, va in folds:
        if done and budget.remaining() < RESERVE + 1.5 * max(fold_t):
            log("budget: stopping CV early")
            break
        t = time.perf_counter()
        m = HistGradientBoostingClassifier(**params).fit(A[tr], y[tr])
        oof[va] = align_proba(m, m.predict_proba(A[va]), C)
        test_p += align_proba(m, m.predict_proba(B), C)
        done += 1
        fold_t.append(time.perf_counter() - t)
        if done == 1:
            emit(classes[np.argmax(test_p, 1)], "hgb_fold1")
    test_p /= done
    have = ~np.isnan(oof[:, 0])
    oof_hgb = oof
    info["cv_folds_done"] = done

    # tiny data: blend with logistic regression
    if (n < 500 or n / C < 100) and budget.has(RESERVE + 2):
        from sklearn.linear_model import LogisticRegression
        LA, LB = _linear_matrix(prep, keep)
        oof_lr = np.full((n, C), np.nan)
        test_lr = np.zeros((len(LB), C))
        for tr, va in folds[:done]:
            lr = LogisticRegression(C=0.5, max_iter=2000,
                                    class_weight=capped_balanced_weights(counts) if imb > 3 else None).fit(LA[tr], y[tr])
            oof_lr[va] = align_proba(lr, lr.predict_proba(LA[va]), C)
            test_lr += align_proba(lr, lr.predict_proba(LB), C)
        test_lr /= done
        best_a, best_ll = 1.0, np.inf
        for a in (1.0, 0.75, 0.5, 0.25, 0.0):
            ll = log_loss(y[have], a * oof_hgb[have] + (1 - a) * oof_lr[have])
            if ll < best_ll - 1e-6:
                best_a, best_ll = a, ll
        oof = best_a * oof_hgb + (1 - best_a) * oof_lr
        test_p = best_a * test_p + (1 - best_a) * test_lr
        info["blend_hgb_weight"] = best_a
        if best_a < 1.0:
            info["model"] = f"blend({best_a:.2f}*HGB + {1 - best_a:.2f}*LogReg)"

    # macro-F1 decision rule tuned on OOF (only if it clearly helps)
    w = np.ones(C)
    if have.sum() >= 30:
        w_t, f0, f1 = tune_class_weights(oof[have], y[have], C)
        info["oof_macro_f1_argmax"] = round(f0, 5)
        if f1 - f0 > 0.002:
            w = w_t
        info["oof_macro_f1"] = round(max(f0, f1) if f1 - f0 > 0.002 else f0, 5)
        info["oof_accuracy"] = round(float(np.mean(np.argmax(oof[have] * w, 1) == y[have])), 5)
    info["class_weights_decision"] = [round(float(x), 4) for x in w]
    emit(classes[np.argmax(test_p * w, 1)], "final")
    return info


# --------------------------------------------------------------------------- regression
def _regression(task, prep, budget, emit, log):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import skew

    y, keep = encode_regression_target(task.y_train)
    n = len(y)
    info = {"model": "HistGradientBoostingRegressor"}
    if n < 5:
        emit(np.full(task.n_predict, float(np.median(y)) if n else 0.0), "median")
        info["model"] = "median"
        return info
    A, B, mask = _tree_matrix(prep, keep)
    params = _hgb_params(n, "regression")
    params.update(hgb_wide_params(A.shape[1]))
    if "max_bins" in params:
        A, B = cap_cat_codes(mask, params["max_bins"], A, B)
    params["categorical_features"] = mask if mask.any() else None
    nonneg = bool(y.min() >= 0)
    sk = float(skew(y)) if np.std(y) > 0 else 0.0
    transforms = ["raw"] + (["log1p"] if nonneg and sk > 1.0 else [])
    info.update(target_skew=round(sk, 4), transforms_tried=transforms)

    folds = make_folds(y, _n_folds(n), "regression")
    oof = {t: np.full(n, np.nan) for t in transforms}
    test = {t: np.zeros(len(B)) for t in transforms}
    done, fold_t = 0, []
    for tr, va in folds:
        if done and budget.remaining() < RESERVE + 1.5 * max(fold_t):
            log("budget: stopping CV early")
            break
        t0 = time.perf_counter()
        for t in transforms:
            yt = np.log1p(y[tr]) if t == "log1p" else y[tr]
            m = HistGradientBoostingRegressor(**params).fit(A[tr], yt)
            pv, pt = m.predict(A[va]), m.predict(B)
            if t == "log1p":
                pv, pt = np.expm1(pv), np.expm1(pt)
            oof[t][va], test[t] = pv, test[t] + pt
        done += 1
        fold_t.append(time.perf_counter() - t0)
        if done == 1:
            emit(_clip(test["raw"], y), "hgb_fold1")
    have = ~np.isnan(oof["raw"])
    scores = {t: rmse(y[have], oof[t][have]) for t in transforms}
    best_t = min(transforms, key=lambda t: scores[t] - (1e-9 if t == "log1p" else 0))
    pred_oof, pred = oof[best_t], test[best_t] / done
    info.update(target_transform=best_t, oof_rmse_by_transform={t: round(s, 6) for t, s in scores.items()},
                cv_folds_done=done)

    if n < 500 and budget.has(RESERVE + 2):
        from sklearn.linear_model import RidgeCV
        LA, LB = _linear_matrix(prep, keep)
        oof_l, test_l = np.full(n, np.nan), np.zeros(len(LB))
        for tr, va in folds[:done]:
            yt = np.log1p(y[tr]) if best_t == "log1p" else y[tr]
            r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(LA[tr], yt)
            pv, pt = r.predict(LA[va]), r.predict(LB)
            if best_t == "log1p":
                pv, pt = np.expm1(np.clip(pv, None, 50)), np.expm1(np.clip(pt, None, 50))
            oof_l[va], test_l = pv, test_l + pt
        test_l /= done
        best_a, best_s = 1.0, rmse(y[have], pred_oof[have])
        for a in (0.75, 0.5, 0.25, 0.0):
            s = rmse(y[have], a * pred_oof[have] + (1 - a) * oof_l[have])
            if s < best_s - 1e-9:
                best_a, best_s = a, s
        pred_oof = best_a * pred_oof + (1 - best_a) * oof_l
        pred = best_a * pred + (1 - best_a) * test_l
        info["blend_hgb_weight"] = best_a
        if best_a < 1.0:
            info["model"] = f"blend({best_a:.2f}*HGB + {1 - best_a:.2f}*Ridge)"
    info["oof_rmse"] = round(rmse(y[have], pred_oof[have]), 6)
    info["oof_r2"] = round(r2(y[have], pred_oof[have]), 6)
    emit(_clip(pred, y), "final")
    return info


def _clip(p, y):
    lo, hi = y.min(), y.max()
    span = hi - lo
    lo2, hi2 = lo - 0.1 * span, hi + 0.1 * span
    if lo >= 0:
        lo2 = max(lo2, 0.0)
    p = np.where(np.isfinite(p), p, np.median(y))
    return np.clip(p, lo2, hi2)


# --------------------------------------------------------------------------- clustering
def clustering_matrix(prep):
    """Numeric: median impute, log1p skewed, standardise. Categorical one-hot scaled by 1/sqrt(2)."""
    num_tr = prep.num_tr if prep.num_te is None else _concat(prep.num_tr, prep.num_te)
    cat_tr = prep.cat_tr if prep.cat_te is None else _concat(prep.cat_tr, prep.cat_te)
    A, _ = numeric_dense(num_tr, None, indicators=False, log_skewed=True, scale=True)
    C, _, _ = onehot_encode(cat_tr, None, max_levels=15, min_count=2, scale=1 / np.sqrt(2))
    X = np.hstack([A, C])
    if X.shape[1] == 0:
        X = np.zeros((len(num_tr), 1))
    return X


def _concat(a, b):
    import pandas as pd
    return pd.concat([a, b], axis=0, ignore_index=True)


def _clustering(task, prep, budget, emit, log):
    from sklearn.cluster import KMeans

    X = clustering_matrix(prep)
    n, p = X.shape
    info = {"model": "KMeans", "n_features_used": int(p)}
    if p > 30:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.9, svd_solver="full", random_state=SEED).fit(X if n <= 20000 else
                                                                            X[np.random.RandomState(SEED).choice(n, 20000, replace=False)])
        X = pca.transform(X)
        info["pca_components"] = int(X.shape[1])
    k = task.n_clusters
    if not k:
        k = choose_k_silhouette(X)
        info["k_chosen_by_silhouette"] = int(k)
    k = int(min(k, n))
    n_init = 30 if n <= 5000 else (10 if n <= 50000 else 4)
    km = KMeans(k, n_init=n_init, random_state=SEED).fit(X)
    labels = ensure_k_labels(km.labels_, X, k)
    info.update(k=k, n_init=n_init, inertia=float(km.inertia_))
    n_pred = task.n_predict
    emit(labels[-n_pred:] if prep.num_te is not None else labels, "final")
    return info
