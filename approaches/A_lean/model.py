"""Approach A -- lean & robust: one well-configured model family per task, minimal search.

classification : quick HistGB on <= 5k rows written first (anytime + cost probe), then HistGradientBoosting
                 (native categoricals/NaN) K-fold bagged on test; imbalance-aware class weights (capped balanced
                 for binary, sqrt-balanced for multiclass); >= 5 classes: ExtraTrees blend; n < 2000 or
                 < 100 rows/class: logistic-regression blend; blend weights and per-class decision weights
                 chosen on out-of-fold predictions for macro-F1.
regression     : HistGB; raw vs log1p target by OOF R2 with a tie-break towards log1p (RMSLE-safe);
                 tiny data (n < 500): ridge blend (clipped in target space); predictions clipped.
clustering     : standardised features -> KMeans (written first) -> group-aware sparse k-means used as a
                 RELEVANCE FILTER (clearly irrelevant features dropped, informative categoricals protected)
                 -> KMeans on the kept features -> GMM (diag/full) only if stable across inits, in agreement
                 with KMeans and strongly better in BIC.
"""
import time
import warnings

import numpy as np

from cleaning import (encode_classes, encode_regression_target, max_encoded_features, numeric_dense, onehot_encode,
                      ordinal_encode, prepare)
from utils import (SEED, SLOW_CPU_SAFETY, FoldPlanner, finalize_labels, fit_hgb_anytime, FitTimer, align_proba, cap_cat_codes, choose_k_silhouette, ensure_k_labels, get_n_jobs,
                   hgb_wide_params, macro_f1, make_folds, r2, rmse, subsample_idx, tune_class_weights)

warnings.filterwarnings("ignore")
RESERVE = 4.0       # seconds kept back for writing outputs
QUICK_ROWS = 5000   # rows for the quick first model / cost probe
QUICK_ITERS = 100
QUICK_SHARE = 0.125  # the quick (anytime) model must finish within 1/8 of the budget (stops between chunks)


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
    p_raw = prep.num_tr.shape[1] + prep.cat_tr.shape[1]
    cap = max(1, max_encoded_features(p_raw) - A.shape[1])          # cost guard on the one-hot width
    C, D, _ = onehot_encode(prep.cat_tr, prep.cat_te, max_levels=30, max_total=cap)
    A, B = np.hstack([A, C]), np.hstack([B, D])
    if keep is not None:
        A = A[keep]
    return A, B


def _forest_matrix(prep, keep):
    An, Bn = numeric_dense(prep.num_tr, prep.num_te, indicators=True, scale=False)
    Ac, Bc = ordinal_encode(prep.cat_tr, prep.cat_te, max_levels=250)
    return np.hstack([An, np.nan_to_num(Ac, nan=-1)])[keep], np.hstack([Bn, np.nan_to_num(Bc, nan=-1)])


MANY_CLASSES = 10   # HGB grows one tree per class per iteration: >= 10 classes -> "fast HGB" configuration


def _hgb_params(n, n_classes=2):
    if n_classes >= MANY_CLASSES:
        # fast HGB for many classes (M1d): the per-iteration cost is n_classes trees, so learn faster with fewer
        # iterations (sklearn's default HGB setting: lr 0.1), mild early stopping
        p = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
                 early_stopping=n >= 1000, validation_fraction=0.12, n_iter_no_change=20)
        p["random_state"] = SEED
        return p
    if n < 1000:
        p = dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=max(5, n // 60),
                 l2_regularization=1.0, early_stopping=False)
    else:
        p = dict(learning_rate=0.08, max_iter=800, max_leaf_nodes=31, min_samples_leaf=20,
                 l2_regularization=1.0, early_stopping=True, validation_fraction=0.12, n_iter_no_change=60)
    p["random_state"] = SEED
    return p


HEAVY_WORK = 5e6   # n x features x classes above this -> 3 folds (time headroom on wide multiclass tables)


def _n_folds(n, p=0, n_classes=1):
    # one HGB tree per class per iteration: with many classes the CV cost is folds x classes trees, so use
    # 3 folds (M1); also 3 folds for very large / wide tables
    if n_classes >= MANY_CLASSES:
        return 3
    return 5 if (n <= 30000 and n * p * max(1, n_classes) <= HEAVY_WORK) else 3


def _iteration_cap(timer, n_tr, n_folds, models_per_fold, budget):
    """Largest HistGB max_iter that fits the remaining budget (cost guard; only binds on slow machines)."""
    rate = timer.rate.get("hgb")
    if rate is None:
        return None
    per_fold = (budget.remaining() - RESERVE) / n_folds / max(1, models_per_fold)
    # the quick fit over-estimates per-iteration cost (binning overhead, no early stopping): factor 3 keeps
    # this a safety net against cost blow-ups rather than a speed limiter
    return int(3.0 * per_fold / (rate * n_tr))


# --------------------------------------------------------------------------- classification
def capped_balanced_weights(counts, cap=25.0):
    """'balanced' class weights, capped so near-singleton classes do not dominate training."""
    n, present = counts.sum(), counts > 0
    w = np.where(present, n / (present.sum() * np.maximum(counts, 1)), 1.0)
    w = np.minimum(w, cap * w[present].min())
    return {int(c): float(w[c]) for c in np.where(present)[0]}


def sqrt_balanced_weights(counts):
    """Square root of the balanced weights: softer for multiclass (full balancing over-weights rare classes
    and makes early stopping stop after ~50 iterations)."""
    n, present = counts.sum(), counts > 0
    w = np.where(present, np.sqrt(n / (present.sum() * np.maximum(counts, 1))), 1.0)
    return {int(c): float(w[c]) for c in np.where(present)[0]}


def _decision_score(P, y, C):
    """Macro-F1 of the deployed decision rule (OOF-tuned class weights)."""
    w, f0, f1 = tune_class_weights(P, y, C)
    return (f1, w) if f1 - f0 > 0.002 else (f0, np.ones(C))


def _classification(task, prep, budget, emit, log):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

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
    params = _hgb_params(n, C)
    params.update(hgb_wide_params(A.shape[1]))
    if "max_bins" in params:
        A, B = cap_cat_codes(mask, params["max_bins"], A, B)
    cw = None
    if imb > 3:
        # one rule for binary and multiclass (round 2): sqrt-balanced weights + OOF decision tuning. Full
        # balancing is redundant with the tuned decision rule and distorts the early-stopping loss.
        cw = sqrt_balanced_weights(counts)
        params["class_weight"] = cw
    params["categorical_features"] = mask if mask.any() else None
    info.update(imbalance_ratio=float(imb),
                class_weight=None if cw is None else "sqrt_balanced")

    # quick model on <= 5k rows: a good answer on disk within seconds + cost probe for the guard
    timer = FitTimer()
    qi = subsample_idx(y, QUICK_ROWS, "classification")
    t0 = time.perf_counter()
    q_iters = QUICK_ITERS if C == 2 else max(30, QUICK_ITERS * 2 // C)    # multiclass: C trees per iteration
    qp = dict(params, max_iter=q_iters, learning_rate=0.1, early_stopping=False)
    qm, _ = fit_hgb_anytime(HistGradientBoostingClassifier(**qp), A[qi], y[qi], time.perf_counter() + QUICK_SHARE * budget.total, first_chunk=10)
    timer.observe("hgb", len(qi), max(1, qm.n_iter_), time.perf_counter() - t0)
    emit(classes[np.argmax(align_proba(qm, qm.predict_proba(B), C), 1)], "quick_hgb")

    use_et = C >= 5
    use_lr = n < 2000 or n / C < 100
    folds = make_folds(y, _n_folds(n, A.shape[1], C), "classification")
    n_tr = int(len(y) * (1 - 1 / len(folds)))
    # (round 3: the old rate-based iteration cap is gone -- the quick fit is now itself truncated, so its rate
    #  over-estimates the per-iteration cost; every HGB fit is interruptible and the fold planner bounds the CV)

    members = {"hgb": (np.full((n, C), np.nan), np.zeros((len(B), C)))}
    if use_et:
        FA, FB = _forest_matrix(prep, keep)
        members["et"] = (np.full((n, C), np.nan), np.zeros((len(B), C)))
    if use_lr:
        from sklearn.linear_model import LogisticRegression
        LA, LB = _linear_matrix(prep, keep)
        members["lr"] = (np.full((n, C), np.nan), np.zeros((len(B), C)))
    done, truncated = 0, 0
    planner = FoldPlanner(len(folds), budget, RESERVE)
    for tr, va in folds:
        if not planner.more():
            log(f"budget: CV reduced to {planner.planned} folds (measured fold time x{SLOW_CPU_SAFETY} safety)")
            break
        fold_ok = True
        t_fold, dl = time.perf_counter(), planner.deadline()
        for name in list(members):
            est = timer.estimate(name, len(tr), 1) if name != "hgb" else None
            last = timer.rate.get(f"{name}_fold")
            if done and last is not None and budget.remaining() - RESERVE < 1.3 * last:
                fold_ok = False
                break
            if est is not None and budget.remaining() - RESERVE < 1.3 * est:
                fold_ok = False
                break
            t = time.perf_counter()
            if name == "hgb":
                m, cut = fit_hgb_anytime(HistGradientBoostingClassifier(**params), A[tr], y[tr], dl)
                truncated += int(cut)
                Pv, Pt = m.predict_proba(A[va]), m.predict_proba(B)
            elif name == "et":
                m = ExtraTreesClassifier(100, max_features="sqrt", min_samples_leaf=1, n_jobs=get_n_jobs(),
                                         random_state=SEED,
                                         **({"class_weight": "balanced_subsample"} if cw else {})).fit(FA[tr], y[tr])
                Pv, Pt = m.predict_proba(FA[va]), m.predict_proba(FB)
            else:
                m = LogisticRegression(C=0.5, max_iter=2000, class_weight=cw).fit(LA[tr], y[tr])
                Pv, Pt = m.predict_proba(LA[va]), m.predict_proba(LB)
            members[name][0][va] = align_proba(m, Pv, C)
            members[name][1][:] += align_proba(m, Pt, C)
            timer.rate[f"{name}_fold"] = time.perf_counter() - t
            timer.observe(name, len(tr), 1, time.perf_counter() - t)
        if not fold_ok:
            log("budget: stopping CV early")
            break
        done += 1
        planner.fold_done(time.perf_counter() - t_fold)
        if done == 1:
            emit(classes[np.argmax(members["hgb"][1], 1)], "hgb_fold1")
    info.update(cv_folds_done=done, hgb_fits_truncated=truncated, folds_reduced_from=planner.reduced_from)
    if done == 0:
        return info
    used = np.concatenate([va for _, va in folds[:done]])
    oofs = {k: v[0][used] for k, v in members.items()}
    tests = {k: v[1] / done for k, v in members.items()}
    yu = y[used]

    # blend weights by OOF macro-F1 of the deployed decision rule (HGB anchor, then ET, then LR)
    oof, test = oofs["hgb"], tests["hgb"]
    best, wcls = _decision_score(oof, yu, C)
    weights = {"hgb": 1.0}
    for name in ("et", "lr"):
        if name not in oofs:
            continue
        choice = (0.0, best, wcls)
        for a in (0.25, 0.5, 0.75):
            s, w = _decision_score((1 - a) * oof + a * oofs[name], yu, C)
            if s > choice[1] + 0.002:
                choice = (a, s, w)
        a, best, wcls = choice
        if a > 0:
            oof, test = (1 - a) * oof + a * oofs[name], (1 - a) * test + a * tests[name]
            weights = {k: v * (1 - a) for k, v in weights.items()}
            weights[name] = a
    info["blend_weights"] = {k: round(v, 3) for k, v in weights.items()}
    if len(weights) > 1:
        info["model"] = "blend(" + " + ".join(f"{v:.2f}*{k}" for k, v in weights.items()) + ")"
    info["oof_macro_f1"] = round(best, 5)
    info["oof_accuracy"] = round(float(np.mean(np.argmax(oof * wcls, 1) == yu)), 5)
    info["class_weights_decision"] = [round(float(x), 4) for x in wcls]
    emit(classes[np.argmax(test * wcls, 1)], "final")
    return info


# --------------------------------------------------------------------------- regression
LOG_TOLERANCE = 0.005   # prefer log1p when its OOF R2 is within this of the raw target (RMSLE-safe)


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
    params = _hgb_params(n)
    params.update(hgb_wide_params(A.shape[1]))
    if "max_bins" in params:
        A, B = cap_cat_codes(mask, params["max_bins"], A, B)
    params["categorical_features"] = mask if mask.any() else None
    nonneg = bool(y.min() >= 0)
    sk = float(skew(y)) if np.std(y) > 0 else 0.0
    transforms = ["raw"] + (["log1p"] if nonneg and sk > 0.75 else [])
    info.update(target_skew=round(sk, 4), transforms_tried=transforms)

    timer = FitTimer()
    qi = subsample_idx(y, QUICK_ROWS, "regression")
    t0 = time.perf_counter()
    q_log = "log1p" in transforms
    qm, _ = fit_hgb_anytime(HistGradientBoostingRegressor(**dict(params, max_iter=QUICK_ITERS, learning_rate=0.1,
                                                                 early_stopping=False)),
                            A[qi], np.log1p(y[qi]) if q_log else y[qi], time.perf_counter() + QUICK_SHARE * budget.total, first_chunk=10)
    timer.observe("hgb", len(qi), max(1, qm.n_iter_), time.perf_counter() - t0)
    qp = qm.predict(B)
    emit(_clip(np.expm1(qp) if q_log else qp, y), "quick_hgb")

    folds = make_folds(y, _n_folds(n, A.shape[1]), "regression")
    n_tr = int(n * (1 - 1 / len(folds)))
    # (round 3: the old rate-based iteration cap is gone -- the quick fit is now itself truncated, so its rate
    #  over-estimates the per-iteration cost; every HGB fit is interruptible and the fold planner bounds the CV)

    oof = {t: np.full(n, np.nan) for t in transforms}
    test = {t: np.zeros(len(B)) for t in transforms}
    done, last, truncated = 0, None, 0
    planner = FoldPlanner(len(folds), budget, RESERVE)
    for tr, va in folds:
        if not planner.more() or (done and budget.remaining() - RESERVE < 1.3 * last):
            log("budget: stopping CV early")
            break
        t0 = time.perf_counter()
        dl = planner.deadline()
        for i_t, t in enumerate(transforms):
            yt = np.log1p(y[tr]) if t == "log1p" else y[tr]
            share = t0 + (dl - t0) * (i_t + 1) / len(transforms)
            m, cut = fit_hgb_anytime(HistGradientBoostingRegressor(**params), A[tr], yt, share)
            truncated += int(cut)
            pv, pt = m.predict(A[va]), m.predict(B)
            if t == "log1p":
                pv, pt = np.expm1(np.clip(pv, None, 60)), np.expm1(np.clip(pt, None, 60))
            oof[t][va], test[t] = pv, test[t] + pt
        done += 1
        last = time.perf_counter() - t0
        planner.fold_done(last)
        if done == 1:
            emit(_clip(test[transforms[-1]], y), "hgb_fold1")
    info.update(hgb_fits_truncated=truncated, folds_reduced_from=planner.reduced_from)
    if done == 0:
        return info
    have = ~np.isnan(oof["raw"])
    scores = {t: r2(y[have], oof[t][have]) for t in transforms}
    best_t = "raw"
    if "log1p" in scores and scores["log1p"] >= scores["raw"] - LOG_TOLERANCE:
        best_t = "log1p"
    pred_oof, pred = oof[best_t], test[best_t] / done
    info.update(target_transform=best_t, oof_r2_by_transform={t: round(s, 6) for t, s in scores.items()},
                cv_folds_done=done)

    if n < 500 and budget.has(RESERVE + 2):
        from sklearn.linear_model import RidgeCV
        LA, LB = _linear_matrix(prep, keep)
        oof_l, test_l = np.full(n, np.nan), np.zeros(len(LB))
        for tr, va in folds[:done]:
            yt = np.log1p(y[tr]) if best_t == "log1p" else y[tr]
            r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(LA[tr], yt)
            lo, hi = yt.min(), yt.max()               # no extrapolation beyond the training target range
            pv, pt = np.clip(r.predict(LA[va]), lo, hi), np.clip(r.predict(LB), lo, hi)
            if best_t == "log1p":
                pv, pt = np.expm1(pv), np.expm1(pt)
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
    if best_t == "log1p":
        info["model"] += "[log1p]"
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
def _concat(a, b):
    import pandas as pd
    return a if b is None else pd.concat([a, b], axis=0, ignore_index=True)


def _clustering(task, prep, budget, emit, log):
    from sklearn.cluster import KMeans
    import feature_weighting as FW

    num, cat = _concat(prep.num_tr, prep.num_te), _concat(prep.cat_tr, prep.cat_te)
    X, groups, names, is_cat = FW.cluster_matrix(num, cat)
    n = len(X)
    n_pred = task.n_predict
    info = {"model": "KMeans", "n_features_used": int(X.shape[1])}
    k = task.n_clusters
    if not k:
        k = choose_k_silhouette(X)
        info["k_chosen_by_silhouette"] = int(k)
    k = int(min(k, n))
    km = KMeans(k, n_init=10 if n <= 20000 else 4, random_state=SEED).fit(X)
    emit(finalize_labels(km.labels_, X, k, n_pred), "kmeans_all_features")
    if k < 2 or n < 10:
        return info

    # 1) relevance filter: low sparse weight AND confirmed by a permutation test (shared SH1)
    keep = list(range(len(groups)))
    if len(groups) >= 3 and budget.has(RESERVE + 4):
        r = FW.fit_sparse(X, groups, k, budget)
        drop, diag = FW.relevance_filter(X, groups, k, r["w"])
        info["sparse_weights"] = {names[g]: round(float(r["w"][g]), 4) for g in range(len(groups))}
        info["relevance_tests"] = {names[g]: v for g, v in diag.items()}
        if drop.any():
            keep = [g for g in range(len(groups)) if not drop[g]]
            info["dropped_irrelevant"] = [names[g] for g in range(len(groups)) if drop[g]]
    cols = np.concatenate([groups[g] for g in keep])
    num_keep = [g for g in keep if not is_cat[g]]
    Xf = X[:, cols]
    # continuous columns inside Xf (for the GMM-diag candidate)
    pos = {c: i for i, c in enumerate(cols)}
    num_cols = np.array([pos[c] for g in num_keep for c in groups[g]], int) if num_keep else None
    if Xf.shape[1] > 30:                       # wide: PCA to 90 % variance first (lean)
        from sklearn.decomposition import PCA
        sub = Xf if n <= 20000 else Xf[np.random.RandomState(SEED).choice(n, 20000, replace=False)]
        Xf = PCA(n_components=0.9, svd_solver="full", random_state=SEED).fit(sub).transform(Xf)
        num_cols = None
        info["pca_components"] = int(Xf.shape[1])

    # 2) small candidate set, chosen by agreement (medoid); no BIC-gain acceptance (SH2)
    cands = FW.candidate_partitions(Xf, num_cols, k, budget, RESERVE)
    if "kmeans" in cands:
        emit(finalize_labels(cands["kmeans"], Xf, k, n_pred), "kmeans_filtered")
    pick, scores = FW.agreement_medoid(cands)
    info.update(candidates=list(cands), agreement=scores, k=k)
    info["model"] = ("relevance-filtered " if "dropped_irrelevant" in info else "") + pick
    labels = finalize_labels(cands[pick], Xf, k, n_pred)
    emit(labels, "final")
    return info
