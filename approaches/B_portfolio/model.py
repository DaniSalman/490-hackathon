"""Approach B -- AutoML-style portfolio + greedy ensemble selection, with a DETERMINISTIC candidate plan.

supervised : a quick HistGB on <= 5k rows is written first (anytime + cost probe). The portfolio (HistGB
             configs, linear, ExtraTrees, RandomForest, kNN) is decided from the data size/shape only, so the
             same model set is used on any CPU; the wall clock is used only as a safety abort. All candidates
             share the SAME K folds; candidates are added in order and the search stops when the last one
             improved the ensemble's OOF score by < 0.002. Final predictor = Caruana et al. (2004) greedy
             forward selection with replacement on OOF predictions (macro-F1 / R2); test = weighted average of
             fold-bagged test predictions. Linear members are clipped to the training target range (log space
             when log1p) and are excluded for test rows outside the training feature range.
             Classification ends with OOF-tuned class weights for macro-F1; regression tries raw and log1p.
clustering : group-aware sparse k-means first acts as a relevance filter (clearly irrelevant features dropped
             from EVERY view; a sparse-weighted view is added); then preprocessing views x algorithms (KMeans,
             GMM, Ward, spectral), all forced to k clusters; the CONSENSUS partition (medoid by mean pairwise
             ARI) is returned, ties broken by silhouette.
"""
import time
import warnings

import numpy as np

from cleaning import (encode_classes, encode_regression_target, max_encoded_features, numeric_dense, onehot_encode,
                      ordinal_encode, prepare)
from utils import (SEED, SLOW_CPU_SAFETY, FoldPlanner, finalize_labels, fit_hgb_anytime, FitTimer, align_proba, cap_cat_codes, choose_k_silhouette, ensure_k_labels, get_n_jobs,
                   hgb_wide_params, log_loss, macro_f1, make_folds, r2, relabel_contiguous, rmse, subsample_idx,
                   tune_class_weights)

warnings.filterwarnings("ignore")
RESERVE = 5.0
QUICK_SHARE = 0.125   # the quick (anytime) model must finish within 1/8 of the budget (stops between chunks)
MIN_GAIN = 0.002       # stop adding candidates when the ensemble OOF score improves less than this
CORE = 3               # candidates always evaluated before the stopping rule applies


def solve(task, budget, emit, log):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(get_n_jobs()):
        prep = prepare(task)
        log(f"prepared: {prep.num_tr.shape[1]} numeric, {prep.cat_tr.shape[1]} categorical features")
        if task.task_type == "clustering":
            info = _clustering(task, prep, budget, emit, log)
        else:
            info = _supervised(task, prep, budget, emit, log)
    info["approach"] = "B_portfolio"
    info["_prep_info"] = prep.info
    return info


# --------------------------------------------------------------------------- feature views
class Views:
    def __init__(self, prep, keep):
        self.prep, self.keep, self._c = prep, keep, {}
        self.p_raw = prep.num_tr.shape[1] + prep.cat_tr.shape[1]

    def tree(self):
        if "tree" not in self._c:
            p = self.prep
            Ac, Bc = ordinal_encode(p.cat_tr, p.cat_te, max_levels=250)
            A = np.hstack([p.num_tr.to_numpy(float), Ac])[self.keep]
            B = np.hstack([p.num_te.to_numpy(float), Bc])
            mask = np.array([False] * p.num_tr.shape[1] + [True] * Ac.shape[1])
            wide = hgb_wide_params(A.shape[1])
            if wide:
                A, B = cap_cat_codes(mask, wide["max_bins"], A, B)
            self._c["tree"] = (A, B, mask)
        return self._c["tree"]

    def forest(self):
        if "forest" not in self._c:
            p = self.prep
            An, Bn = numeric_dense(p.num_tr, p.num_te, indicators=True, log_skewed=False, scale=False)
            Ac, Bc = ordinal_encode(p.cat_tr, p.cat_te, max_levels=250)
            Ac, Bc = np.nan_to_num(Ac, nan=-1.0), np.nan_to_num(Bc, nan=-1.0)
            cap = max(0, max_encoded_features(self.p_raw) - An.shape[1] - Ac.shape[1])
            Oa, Ob, _ = onehot_encode(p.cat_tr, p.cat_te, max_levels=8, min_count=5, max_total=cap)
            self._c["forest"] = (np.hstack([An, Ac, Oa])[self.keep], np.hstack([Bn, Bc, Ob]))
        return self._c["forest"]

    def linear(self):
        if "linear" not in self._c:
            p = self.prep
            An, Bn = numeric_dense(p.num_tr, p.num_te, indicators=True, log_skewed=True, scale=True)
            cap = max(1, max_encoded_features(self.p_raw) - An.shape[1])
            Oa, Ob, _ = onehot_encode(p.cat_tr, p.cat_te, max_levels=40, min_count=3, max_total=cap)
            self._c["linear"] = (np.hstack([An, Oa])[self.keep], np.hstack([Bn, Ob]))
            # test rows outside the training range of any numeric feature (linear models extrapolate there)
            k = p.num_tr.shape[1]
            if k:
                lo, hi = An[self.keep][:, :k].min(0), An[self.keep][:, :k].max(0)
                span = hi - lo
                Z = Bn[:, :k]
                self._c["ood"] = ((Z < lo - 0.1 * span) | (Z > hi + 0.1 * span)).any(1)
            else:
                self._c["ood"] = np.zeros(len(Bn), bool)
        return self._c["linear"]

    def out_of_range(self):
        self.linear()
        return self._c["ood"]


def _capped_balanced(counts, cap=25.0):
    n, present = counts.sum(), counts > 0
    w = np.where(present, n / (present.sum() * np.maximum(counts, 1)), 1.0)
    w = np.minimum(w, cap * w[present].min())
    return {int(c): float(w[c]) for c in np.where(present)[0]}


def _portfolio(kind, n, p_lin, p_tree, cw, nj, heavy=False, n_classes=2):
    """Deterministic list of (name, view, factory, relative_cost), decided by data size/shape only.

    heavy (work n x p x classes > HEAVY_WORK): only the core candidates (hgb_a, linear, ExtraTrees, hgb_b)."""
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    clf = kind == "classification"
    H = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
    extra = {"class_weight": cw} if (clf and cw) else {}
    small, large = n < 1000, n > 10000
    many = clf and n_classes >= MANY_CLASSES
    hgb_a = (dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=max(5, n // 60),
                  l2_regularization=1.0, early_stopping=False) if small else
             dict(learning_rate=0.1, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.5,
                  early_stopping=True, validation_fraction=0.12, n_iter_no_change=25))
    hgb_b = (dict(learning_rate=0.03, max_iter=400, max_leaf_nodes=7, min_samples_leaf=max(8, n // 40),
                  l2_regularization=3.0, early_stopping=False) if small else
             dict(learning_rate=0.06, max_iter=1000, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=3.0,
                  early_stopping=True, validation_fraction=0.12, n_iter_no_change=30))
    try:  # feature subsampling per split (sklearn >= 1.4); silently skipped on older versions
        H(max_features=0.7)
        if not small:
            hgb_b["max_features"] = 0.7
    except TypeError:
        pass
    if many:   # one tree per class per iteration: learn faster, fewer iterations (sklearn default lr 0.1)
        hgb_a.update(max_iter=100, n_iter_no_change=20)
        hgb_b.update(learning_rate=0.1, max_iter=100, n_iter_no_change=20)
    wide = hgb_wide_params(p_tree)
    hgb_a.update(wide)
    hgb_b.update(wide)
    hgb_c = dict(**wide, learning_rate=0.05, max_iter=1000, max_leaf_nodes=63, min_samples_leaf=10, l2_regularization=1.0,
                 early_stopping=True, validation_fraction=0.12, n_iter_no_change=30)
    trees = 200 if not large else 100
    ms = None if not large else min(1.0, 8000 / n)          # bounded forest cost on large n
    out = [("hgb_a", "tree", lambda m: H(random_state=SEED, categorical_features=m, **hgb_a, **extra), 1.0)]
    if clf:
        out.append(("logreg", "linear", lambda m: LogisticRegression(C=1.0, max_iter=1000, **extra), 0.3))
        out.append(("extratrees", "forest", lambda m: ExtraTreesClassifier(
            trees, max_features="sqrt", min_samples_leaf=1, bootstrap=ms is not None, max_samples=ms,
            n_jobs=nj, random_state=SEED, **({"class_weight": "balanced_subsample"} if cw else {})), 1.2))
    else:
        out.append(("ridge", "linear", lambda m: RidgeCV(alphas=np.logspace(-3, 3, 13)), 0.2))
        out.append(("extratrees", "forest", lambda m: ExtraTreesRegressor(
            trees, max_features=0.7, min_samples_leaf=2, bootstrap=ms is not None, max_samples=ms,
            n_jobs=nj, random_state=SEED), 1.5))
    out.append(("hgb_b", "tree", lambda m: H(random_state=SEED, categorical_features=m, **hgb_b, **extra), 1.5))
    if heavy:
        return out
    if clf:
        out.append(("randomforest", "forest", lambda m: RandomForestClassifier(
            trees, max_features="sqrt", min_samples_leaf=1, max_samples=ms, n_jobs=nj, random_state=SEED,
            **({"class_weight": "balanced_subsample"} if cw else {})), 1.5))
    else:
        out.append(("randomforest", "forest", lambda m: RandomForestRegressor(
            trees, max_features=0.5, min_samples_leaf=2, max_samples=ms, n_jobs=nj, random_state=SEED), 2.0))
    if not large and p_lin <= 300:
        kk = int(np.clip(np.sqrt(n) / 2, 5, 30))
        K = KNeighborsClassifier if clf else KNeighborsRegressor
        out.append(("knn", "linear", lambda m: K(n_neighbors=kk, weights="distance", n_jobs=nj), 0.5))
    if 2000 <= n <= 10000:
        out.append(("hgb_c", "tree", lambda m: H(random_state=SEED, categorical_features=m, **hgb_c, **extra), 1.5))
    return out


LINEAR_MEMBERS = ("ridge", "logreg")
MANY_CLASSES = 10        # >= 10 classes: fast HGB configs and 3 folds (cost = folds x classes trees per iteration)
HEAVY_CELLS = 8e5        # size class: n x tree-features above this -> 3 folds + core portfolio (wide/large tables)
LOG_TOLERANCE = 0.005    # B2: keep only log1p members when log1p hgb_a is within this OOF R2 of raw
LOG_ENSEMBLE_TOLERANCE = 0.01   # B2: log-only ensemble accepted within 0.01 OOF R2 (RMSLE safety, Seoul -0.007)
OOD_CAP = 2.0            # B3: on out-of-range rows a linear member may deviate from the trees by <= x2


# --------------------------------------------------------------------------- supervised
def _supervised(task, prep, budget, emit, log):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    clf = task.task_type == "classification"
    if clf:
        y, classes, keep = encode_classes(task.y_train)
        C = len(classes)
        if C < 2:
            emit(np.repeat(classes[0] if C else 0, task.n_predict), "single_class")
            return {"model": "constant"}
        counts = np.bincount(y, minlength=C)
        imb = counts.max() / max(1, counts[counts > 0].min())
        cw = _capped_balanced(counts) if imb > 3 else None
    else:
        y, keep = encode_regression_target(task.y_train)
        if len(y) < 5:
            emit(np.full(task.n_predict, float(np.median(y)) if len(y) else 0.0), "median")
            return {"model": "median"}
        C, cw, imb = None, None, None
    n = len(y)
    views = Views(prep, keep)
    nj = get_n_jobs()

    transforms = ["raw"]
    if not clf:
        from scipy.stats import skew
        if y.min() >= 0 and np.std(y) > 0 and skew(y) > 0.75:
            transforms.append("log1p")

    # quick model on <= 5k rows -> good answer on disk within seconds, and the per-fit cost probe
    A, B, mask = views.tree()
    qi = subsample_idx(y, 5000, task.task_type)
    t0 = time.perf_counter()
    q_iters = 100 if (not clf or C == 2) else max(30, 200 // C)
    H = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
    qm = H(max_iter=q_iters, learning_rate=0.1, categorical_features=mask if mask.any() else None,
           random_state=SEED, **({"class_weight": cw} if cw else {}))
    q_log = "log1p" in transforms
    qm, _ = fit_hgb_anytime(qm, A[qi], np.log1p(y[qi]) if q_log else y[qi], time.perf_counter() + QUICK_SHARE * budget.total)
    timer = FitTimer()
    timer.observe("hgb_iter", len(qi), max(1, qm.n_iter_), time.perf_counter() - t0)
    if clf:
        emit(classes[np.argmax(align_proba(qm, qm.predict_proba(B), C), 1)], "quick_hgb")
    else:
        qp = qm.predict(B)
        emit(_clip(np.expm1(np.clip(qp, None, 60)) if q_log else qp, y), "quick_hgb")

    work = n * A.shape[1]
    heavy = work > HEAVY_CELLS or (clf and C >= MANY_CLASSES)   # many classes: core portfolio (per-class cost)
    folds = make_folds(y, 3 if (heavy or n > 30000 or (clf and C >= MANY_CLASSES)) else 5, task.task_type)
    p_lin = views.linear()[0].shape[1]
    portfolio = _portfolio(task.task_type, n, p_lin, A.shape[1], cw, nj, heavy, C or 0)
    info = {"candidates": [], "transforms_tried": transforms, "size_class": "heavy" if heavy else "normal",
            "work": int(work)}
    # B1 affordability (safety only): if the portfolio is projected not to fit, fall back to the FULL single
    # hgb_a with an iteration cap from the measured quick-fit rate (as A / C) -- not to the quick model
    n_tr = int(n * (1 - 1 / len(folds)))
    rate = timer.rate["hgb_iter"]
    est_anchor = rate * n_tr * 150 * len(folds) * len(transforms) / 3.0
    if budget.remaining() - RESERVE < est_anchor:
        # the anchor is interruptible (chunked fits + fold planner), so no iteration cap and never the quick model
        portfolio = [portfolio[0]]
        info["fallback"] = "single hgb_a: portfolio projected not to fit"
        log("budget: " + info["fallback"])
    info["portfolio_plan"] = [p[0] for p in portfolio]

    results, base_cost, best_tf, best_score = [], None, "raw", -np.inf
    ens_score, stopped = -np.inf, None
    for name, view, make, rel in portfolio:
        if stopped:
            break
        for tf in (transforms if name == "hgb_a" else [best_tf]):
            est = (base_cost or 0) * rel
            if clf and C > 2 and view != "tree":
                # HistGB grows one tree per class per iteration; forests / linear / kNN fit all classes at
                # once, so their cost relative to the HGB anchor shrinks by the number of classes
                est /= C
            if results and budget.remaining() - RESERVE < est * 1.2:      # safety abort only
                log(f"safety skip {name}: est {est:.1f}s > remaining {budget.remaining():.1f}s")
                info["candidates"].append({"model": name, "skipped": "budget_safety"})
                continue
            planner = FoldPlanner(len(folds), budget, RESERVE) if not results else None
            res = _run_cv(name, view, make, views, y, folds, clf, C, tf, budget, log, planner)
            if res is not None and planner is not None and res["folds_done"] < len(folds):
                info["folds_reduced"] = "%d -> %d (measured anchor fold time x%.0f safety)" % (
                    len(folds), res["folds_done"], SLOW_CPU_SAFETY)
                folds = folds[:res["folds_done"]]
            used = np.concatenate([va for _, va in folds])
            if res is None:
                info["candidates"].append({"model": name, "skipped": "aborted_budget_safety"})
                continue
            if base_cost is None:
                base_cost = res["seconds"] / len(transforms)
            results.append(res)
            info["candidates"].append({"model": res["name"], "oof_score": round(res["score"], 6),
                                       "seconds": round(res["seconds"], 2)})
            log(f"{res['name']}: oof={res['score']:.5f} ({res['seconds']:.1f}s)")
            if res["score"] > best_score:
                best_score = res["score"]
                if name == "hgb_a":
                    best_tf = tf
                emit(classes[np.argmax(res["test"], 1)] if clf else _clip(res["test"], y), f"single:{res['name']}")
        if name == "hgb_a" and len(transforms) > 1:
            # later members follow the log target when it is within LOG_TOLERANCE of raw (RMSLE safety)
            sc = {r["tf"]: r["score"] for r in results if r["name"].startswith("hgb_a")}
            if "log1p" in sc and "raw" in sc and sc["log1p"] >= sc["raw"] - LOG_TOLERANCE:
                best_tf = "log1p"
        n_models = len({r["name"].split("[")[0] for r in results})
        if n_models >= CORE and results:
            _, s_new = _caruana(_on(results, used), y[used], clf, C)
            if s_new - ens_score < MIN_GAIN and n_models > CORE:
                stopped = f"after {name}: ensemble gain {s_new - ens_score:.4f} < {MIN_GAIN}"
                log("stop: " + stopped)
            ens_score = max(ens_score, s_new)
    info["stopped"] = stopped
    info["target_transform"] = best_tf if not clf else None
    if not results:
        info.update(model="quick HistGB (no candidate finished)")
        return info

    yu = y[used]
    weights, ens_all = _caruana(_on(results, used), yu, clf, C)
    if not clf and "log1p" in transforms:
        # B2 (RMSLE safety, decision-level tie-break): if the ensemble of log-target members only is within
        # LOG_ENSEMBLE_TOLERANCE OOF R2 of the unrestricted ensemble, use it (same R2, much better relative error)
        log_res = [r for r in results if r["tf"] == "log1p"]
        if log_res and len(log_res) < len(results):
            w_log, ens_log = _caruana(_on(log_res, used), yu, clf, C)
            info["log_only_ensemble"] = {"oof_r2_all": round(ens_all, 5), "oof_r2_log_only": round(ens_log, 5)}
            if ens_log >= ens_all - LOG_ENSEMBLE_TOLERANCE:
                results, weights = log_res, w_log
                info["log_only_ensemble"]["chosen"] = True
    info["ensemble_weights"] = {results[i]["name"]: round(w, 4) for i, w in enumerate(weights) if w > 0}
    info["model"] = "CaruanaEnsemble(" + ", ".join(f"{k}:{v}" for k, v in info["ensemble_weights"].items()) + ")"
    oof = sum(w * results[i]["oof"][used] for i, w in enumerate(weights) if w > 0)
    test = sum(w * results[i]["test"] for i, w in enumerate(weights) if w > 0)
    if not clf:
        # B3: on test rows outside the training feature range a linear member is CAPPED to within a factor
        # OOD_CAP of the non-linear members (extrapolation is trusted only when corroborated in magnitude)
        lin = [i for i, w in enumerate(weights) if w > 0 and results[i]["name"].split("[")[0] in LINEAR_MEMBERS]
        w_lin = sum(weights[i] for i in lin)
        ood = views.out_of_range()
        if lin and 0 < w_lin < 1 and ood.any():
            rest = sum(w * results[i]["test"] for i, w in enumerate(weights) if w > 0 and i not in lin) / (1 - w_lin)
            capped = rest * (1 - w_lin)
            for i in lin:
                if y.min() >= 0:
                    lo, hi = rest / OOD_CAP, rest * OOD_CAP
                else:
                    span = (OOD_CAP - 1) * (np.abs(rest) + y.std())
                    lo, hi = rest - span, rest + span
                capped = capped + weights[i] * np.clip(results[i]["test"], np.minimum(lo, hi), np.maximum(lo, hi))
            info["out_of_range_rows"] = int(ood.sum())
            info["out_of_range_rows_linear_capped"] = int((ood & (np.abs(capped - test) > 1e-9)).sum())
            test = np.where(ood, capped, test)
    if clf:
        wcls = np.ones(C)
        wt, f0, f1 = tune_class_weights(oof, yu, C)
        if f1 - f0 > 0.003:
            wcls = wt
        info.update(oof_macro_f1=round(max(f0, f1) if f1 - f0 > 0.003 else f0, 5),
                    oof_accuracy=round(float(np.mean(np.argmax(oof * wcls, 1) == yu)), 5),
                    class_weights_decision=[round(float(v), 4) for v in wcls], n_classes=int(C),
                    imbalance_ratio=float(imb))
        emit(classes[np.argmax(test * wcls, 1)], "ensemble")
    else:
        info.update(oof_rmse=round(rmse(yu, oof), 6), oof_r2=round(r2(yu, oof), 6))
        emit(_clip(test, y), "ensemble")
    return info


def _on(results, used):
    return [dict(r, oof=r["oof"][used]) for r in results]


def _run_cv(name, view, make, views, y, folds, clf, C, tf, budget, log, planner=None):
    """CV of one candidate on the shared folds. HistGB fits run in warm-start chunks with a clock check between
    chunks (M1c): the anchor (planner given) keeps a truncated fit and may complete fewer folds; any other
    candidate whose fit would cross its deadline is aborted -- the watchdog is never needed."""
    if view == "tree":
        A, B, mask = views.tree()
        m_arg = mask if mask.any() else None
    else:
        A, B = views.forest() if view == "forest" else views.linear()
        m_arg = None
    n = len(y)
    linear = name in LINEAR_MEMBERS
    oof = np.zeros((n, C)) if clf else np.zeros(n)
    test = np.zeros((len(B), C)) if clf else np.zeros(len(B))
    t0 = time.perf_counter()
    done = 0
    for i, (tr, va) in enumerate(folds):
        if planner is not None and not planner.more():
            break
        if i and planner is None:
            per = (time.perf_counter() - t0) / i
            if budget.remaining() - RESERVE < per * 1.1:                 # per-model safety check
                log(f"abort {name}: budget")
                return None
        t_f = time.perf_counter()
        dl = planner.deadline() if planner is not None else time.perf_counter() + max(
            0.5, (budget.remaining() - RESERVE) / (len(folds) - i))
        m = make(m_arg)
        hgb = "HistGradientBoosting" in type(m).__name__
        if clf:
            if hgb:
                m, cut = fit_hgb_anytime(m, A[tr], y[tr], dl)
                if cut and planner is None:
                    log(f"abort {name}: fit would cross its deadline")
                    return None
            else:
                m.fit(A[tr], y[tr])
            oof[va] = align_proba(m, m.predict_proba(A[va]), C)
            test += align_proba(m, m.predict_proba(B), C)
        else:
            yt = np.log1p(y[tr]) if tf == "log1p" else y[tr]
            if hgb:
                m, cut = fit_hgb_anytime(m, A[tr], yt, dl)
                if cut and planner is None:
                    log(f"abort {name}: fit would cross its deadline")
                    return None
            else:
                m.fit(A[tr], yt)
            pv, pt = m.predict(A[va]), m.predict(B)
            if linear:   # no extrapolation beyond the training target range (in log space when log1p)
                pv, pt = np.clip(pv, yt.min(), yt.max()), np.clip(pt, yt.min(), yt.max())
            if tf == "log1p":
                pv, pt = np.expm1(np.clip(pv, None, 60)), np.expm1(np.clip(pt, None, 60))
            oof[va], test = pv, test + pt
        done += 1
        if planner is not None:
            planner.fold_done(time.perf_counter() - t_f)
    if done == 0:
        return None
    test /= done
    idx = np.concatenate([va for _, va in folds[:done]])
    sc = _score(y[idx], oof[idx], clf, C)
    return {"name": name + ("" if tf == "raw" else f"[{tf}]"), "oof": oof, "test": test, "score": sc,
            "seconds": time.perf_counter() - t0, "tf": tf, "folds_done": done}


def _score(y, oof, clf, C):
    """Higher is better. Classification: macro-F1 (+ tiny log-loss tiebreak); regression: R2."""
    if clf:
        return macro_f1(y, np.argmax(oof, 1), C) - 1e-4 * log_loss(y, oof)
    return r2(y, oof)


def _with_iter_cap(entry, it_cap):
    name, view, make, rel = entry

    def capped(m):
        est = make(m)
        est.set_params(max_iter=int(min(est.get_params()["max_iter"], it_cap)))
        return est
    return (name, view, capped, rel)


def _caruana(results, y, clf, C, n_steps=30):
    m = len(results)
    scores = [r["score"] for r in results]
    first = int(np.argmax(scores))
    counts = np.zeros(m)
    counts[first] = 1
    cur = results[first]["oof"].copy()
    best_score, best_counts = scores[first], counts.copy()
    for step in range(1, n_steps):
        cand = [(_score(y, (cur * step + results[j]["oof"]) / (step + 1), clf, C), j) for j in range(m)]
        s, j = max(cand)
        counts[j] += 1
        cur = (cur * step + results[j]["oof"]) / (step + 1)
        if s > best_score + 1e-9:
            best_score, best_counts = s, counts.copy()
    return best_counts / best_counts.sum(), best_score


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


def _cluster_views(num, cat, sparse_view=None):
    Oh, _, _ = onehot_encode(cat, None, max_levels=15, min_count=2, scale=1 / np.sqrt(2))
    S, _ = numeric_dense(num, None, indicators=False, log_skewed=False, scale=True)
    L, _ = numeric_dense(num, None, indicators=False, log_skewed=True, scale=True)
    R, _ = numeric_dense(num, None, indicators=False, log_skewed=True, scale=True, robust=True)
    views = {"std": np.hstack([S, Oh]), "logstd": np.hstack([L, Oh]), "robust": np.hstack([np.clip(R, -5, 5), Oh])}
    for k in list(views):
        if views[k].shape[1] == 0:
            views[k] = np.zeros((len(num), 1))
    X = views["logstd"]
    if X.shape[1] >= 4:
        from sklearn.decomposition import PCA
        n = len(X)
        sub = X if n <= 20000 else X[np.random.RandomState(SEED).choice(n, 20000, replace=False)]
        pca = PCA(n_components=0.95, svd_solver="full", whiten=True, random_state=SEED).fit(sub)
        views["pca_white"] = pca.transform(X)
    if sparse_view is not None:
        views["sparse_w"] = sparse_view
    return views


def _clustering(task, prep, budget, emit, log):
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    import feature_weighting as FW

    num, cat = _concat(prep.num_tr, prep.num_te), _concat(prep.cat_tr, prep.cat_te)
    n = len(num)
    n_pred = task.n_predict
    info = {"candidates": []}

    # 0) anchor KMeans on all features -> written immediately
    Xg, groups, gnames, is_cat = FW.cluster_matrix(num, cat)
    k = task.n_clusters or choose_k_silhouette(Xg)
    k = int(min(k, n))
    info["k"] = k
    km0 = KMeans(k, n_init=20 if n <= 20000 else 5, random_state=SEED).fit(Xg)
    emit(finalize_labels(km0.labels_, Xg, k, n_pred), "kmeans_anchor")

    # 1) relevance filter shared by every view (+ a sparse-weighted view)
    sparse_view = None
    if len(groups) >= 3 and k >= 2 and n >= 10 and budget.has(RESERVE + 4):
        r = FW.fit_sparse(Xg, groups, k, budget)
        w = r["w"]
        drop, tests = FW.relevance_filter(Xg, groups, k, w)       # B4: shared confirmed filter (SH1)
        info["relevance_tests"] = {gnames[g]: v for g, v in tests.items()}
        info["sparse_weights"] = {gnames[g]: round(float(w[g]), 4) for g in range(len(groups))}
        if drop.any() and not drop.all():
            gone = {gnames[g] for g in range(len(groups)) if drop[g]}
            info["dropped_irrelevant"] = sorted(gone)
            num = num[[c for c in num.columns if str(c) not in gone]]
            cat = cat[[c for c in cat.columns if str(c) not in gone]]
            sparse_view = FW.weighted(Xg, groups, w / w.max())
    views = _cluster_views(num, cat, sparse_view)
    rng = np.random.RandomState(SEED)
    sub = np.arange(n) if n <= 20000 else np.sort(rng.choice(n, 10000, replace=False))
    cands = []

    def add(name, labels):
        labels = np.asarray(labels)
        nun = len(np.unique(labels))
        ok = nun == k
        info["candidates"].append({"name": name, "n_clusters_found": int(nun), "used": bool(ok)})
        if ok:
            cands.append((name, relabel_contiguous(labels)))

    X0 = views["logstd"]
    algos = ["kmeans", "gmm", "ward", "spectral"]
    for vname, X in views.items():
        for algo in algos:
            if budget.remaining() < RESERVE + 2:
                info["stopped_early"] = True
                break
            try:
                if algo == "kmeans":
                    add(f"kmeans|{vname}", KMeans(k, n_init=10 if n <= 20000 else 3, random_state=SEED).fit_predict(X))
                elif algo == "gmm":
                    p = X.shape[1]
                    cov = "full" if p <= 12 else "diag"
                    g = GaussianMixture(k, covariance_type=cov, n_init=2, reg_covar=1e-4, random_state=SEED,
                                        max_iter=200).fit(X[sub])
                    add(f"gmm_{cov}|{vname}", g.predict(X))
                elif algo == "ward" and n <= 8000:
                    add(f"ward|{vname}", AgglomerativeClustering(k, linkage="ward").fit_predict(X))
                elif algo == "spectral" and n <= 2500 and vname in ("logstd", "pca_white"):
                    sc = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=min(15, n - 1),
                                            assign_labels="cluster_qr", random_state=SEED)
                    add(f"spectral|{vname}", sc.fit_predict(X))
            except Exception as e:  # noqa: BLE001
                info["candidates"].append({"name": f"{algo}|{vname}", "error": repr(e)[:120]})
        if vname == "logstd" and cands:
            emit(finalize_labels(cands[0][1], X0, k, n_pred), "kmeans_filtered")
    # 2) consensus: medoid partition by mean ARI to all others
    m = len(cands)
    if m == 0:
        info["model"] = "kmeans_anchor"
        return info
    if m >= 3:
        M = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                M[i, j] = M[j, i] = adjusted_rand_score(cands[i][1], cands[j][1])
        mean_ari = (M.sum(1) - 1) / (m - 1)
        top = np.where(mean_ari >= mean_ari.max() - 0.01)[0]
        if len(top) > 1:
            idx = rng.choice(n, min(n, 3000), replace=False)
            sil = [silhouette_score(X0[idx], cands[i][1][idx]) if len(np.unique(cands[i][1][idx])) > 1 else -1
                   for i in top]
            pick = int(top[int(np.argmax(sil))])
        else:
            pick = int(top[0])
        info["mean_ari_to_others"] = {cands[i][0]: round(float(mean_ari[i]), 4) for i in range(m)}
    else:
        pick = 0
    name, labels = cands[pick]
    info["model"] = f"consensus_medoid:{name}" + (" (relevance-filtered)" if "dropped_irrelevant" in info else "")
    labels = finalize_labels(labels, X0, k, n_pred)
    emit(labels, "consensus")
    return info
