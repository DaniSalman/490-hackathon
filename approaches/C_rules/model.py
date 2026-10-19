"""Approach C -- profile-driven rules ("expert system") + specialised pipelines.

The data profile (size, width, categorical cardinality, imbalance, target skew, datetime presence)
plus one cheap "linearity probe" deterministically select the pipeline -- no portfolio search.

supervised rules
  encoding   : low-cardinality categoricals native in HistGB / one-hot for linear models;
               high-cardinality (>30 levels) -> cross-fitted target encoding; datetimes get cyclic features.
  regime     : small (n<400)            -> 0.5 * regularised linear + 0.5 * ExtraTrees
               wide (p >> n)            -> 0.5 * linear + 0.5 * HistGB
               linear-ish (probe)       -> 0.5 * linear + 0.5 * HistGB
               many classes, few rows/class -> 0.6 * ExtraTrees + 0.4 * linear
               otherwise (nonlinear)    -> HistGB with size-tiered hyper-parameters
  imbalance  : ratio > 1.5 -> capped balanced weights; OOF macro-F1 class-weight tuning if minority >= 20
  target     : log1p when y >= 0 and skew > 0.75 (rule); predictions clipped
  final fit  : refit on all training rows (HistGB iterations = 1.1 x median early-stopped CV iterations);
               falls back to the fold-averaged models if the budget is short.
clustering    : sparse k-means (Witten & Tibshirani 2010) to down-weight / drop irrelevant features, sparsity
               chosen by a permutation gap statistic; then KMeans vs GMM (diag/full) chosen by BIC with a
               degenerate-cluster guard.
"""
import time
import warnings

import numpy as np

from cleaning import encode_classes, encode_regression_target, numeric_dense, onehot_encode, ordinal_encode, prepare
from utils import (SEED, align_proba, choose_k_silhouette, ensure_k_labels, get_n_jobs, hgb_wide_params, cap_cat_codes, macro_f1,
                   make_folds, r2,
                   relabel_contiguous, rmse, tune_class_weights)

warnings.filterwarnings("ignore")
RESERVE = 4.0
HIGH_CARD = 30


def solve(task, budget, emit, log):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(get_n_jobs()):
        prep = prepare(task, cyclic_datetime=True)
        log(f"prepared: {prep.num_tr.shape[1]} numeric, {prep.cat_tr.shape[1]} categorical features")
        if task.task_type == "clustering":
            info = _clustering(task, prep, budget, emit, log)
        else:
            info = _supervised(task, prep, budget, emit, log)
    info["approach"] = "C_rules"
    info["_prep_info"] = prep.info
    return info


# --------------------------------------------------------------------------- features
class Features:
    """Builds per-fold matrices; target encoding is fitted only on the rows it is given."""

    def __init__(self, prep, keep, clf):
        self.clf = clf
        cat_tr = prep.cat_tr.iloc[np.where(keep)[0]].reset_index(drop=True)
        card = cat_tr.nunique() if cat_tr.shape[1] else []
        self.hi = [c for c in cat_tr.columns if card[c] > HIGH_CARD]
        self.lo = [c for c in cat_tr.columns if card[c] <= HIGH_CARD]
        # tree view: numeric (raw, NaN) + ordinal low-card cats
        Ao, Bo = ordinal_encode(cat_tr[self.lo], prep.cat_te[self.lo], max_levels=250)
        self.tree_tr = np.hstack([prep.num_tr.to_numpy(float)[keep], Ao])
        self.tree_te = np.hstack([prep.num_te.to_numpy(float), Bo])
        self.tree_mask = np.array([False] * prep.num_tr.shape[1] + [True] * len(self.lo))
        self.wide = wide = hgb_wide_params(self.tree_tr.shape[1] + len(self.hi))
        if wide:
            self.tree_tr, self.tree_te = cap_cat_codes(self.tree_mask, wide["max_bins"], self.tree_tr, self.tree_te)
        # linear view: imputed/log/scaled numeric + one-hot low-card cats
        An, Bn = numeric_dense(prep.num_tr.iloc[np.where(keep)[0]], prep.num_te, indicators=True, log_skewed=True)
        Oa, Ob, _ = onehot_encode(cat_tr[self.lo], prep.cat_te[self.lo], max_levels=HIGH_CARD, min_count=3)
        self.lin_tr, self.lin_te = np.hstack([An, Oa]), np.hstack([Bn, Ob])
        # forest view (for ExtraTrees): imputed numeric + ordinal codes (-1 for missing)
        Fn, Fm = numeric_dense(prep.num_tr.iloc[np.where(keep)[0]], prep.num_te, indicators=True, scale=False)
        self.for_tr = np.hstack([Fn, np.nan_to_num(Ao, nan=-1)])
        self.for_te = np.hstack([Fm, np.nan_to_num(Bo, nan=-1)])
        self.hi_tr = cat_tr[self.hi].to_numpy(dtype=object) if self.hi else None
        self.hi_te = prep.cat_te[self.hi].to_numpy(dtype=object) if self.hi else None

    def _te(self, tr_idx, y_tr, other):
        """Cross-fitted target encoding on rows tr_idx; plain transform for `other` (list of arrays)."""
        if not self.hi:
            return np.zeros((len(tr_idx), 0)), [np.zeros((len(o), 0)) for o in other]
        from sklearn.preprocessing import TargetEncoder
        te = TargetEncoder(target_type="multiclass" if (self.clf and len(np.unique(y_tr)) > 2) else "auto",
                           random_state=SEED)
        a = te.fit_transform(self.hi_tr[tr_idx], y_tr)
        return a, [te.transform(o) for o in other]

    def view(self, kind, tr_idx, y_tr, va_idx=None):
        """Return (X_tr, [X_va], X_test, cat_mask_or_None)."""
        base_tr = {"tree": self.tree_tr, "linear": self.lin_tr, "forest": self.for_tr}[kind]
        base_te = {"tree": self.tree_te, "linear": self.lin_te, "forest": self.for_te}[kind]
        other = [self.hi_tr[va_idx]] if (self.hi and va_idx is not None) else []
        if self.hi:
            other.append(self.hi_te)
        te_tr, te_other = self._te(tr_idx, y_tr, other)
        if kind == "linear" and te_tr.shape[1]:
            mu, sd = te_tr.mean(0), te_tr.std(0) + 1e-9
            te_tr = (te_tr - mu) / sd
            te_other = [(o - mu) / sd for o in te_other]
        Xtr = np.hstack([base_tr[tr_idx], te_tr])
        Xva = None
        if va_idx is not None:
            Xva = np.hstack([base_tr[va_idx], te_other[0]]) if self.hi else base_tr[va_idx]
        Xte = np.hstack([base_te, te_other[-1] if self.hi else np.zeros((len(base_te), 0))])
        mask = None
        if kind == "tree":
            mask = np.concatenate([self.tree_mask, np.zeros(te_tr.shape[1], bool)])
            mask = mask if mask.any() else None
        return Xtr, Xva, Xte, mask


# --------------------------------------------------------------------------- rules
def meta_features(prep, y, clf, feats):
    from scipy.stats import skew
    n = len(y)
    mf = {"n": int(n), "p_numeric": int(prep.num_tr.shape[1]), "p_categorical": int(prep.cat_tr.shape[1]),
          "p_linear": int(feats.lin_tr.shape[1]), "n_high_card": len(feats.hi),
          "has_datetime": bool(prep.info.get("datetime_columns")),
          "missing_ratio": float(np.isnan(prep.num_tr.to_numpy(float)).mean()) if prep.num_tr.shape[1] else 0.0}
    if clf:
        counts = np.bincount(y)
        counts = counts[counts > 0]
        mf.update(n_classes=int(len(counts)), imbalance_ratio=float(counts.max() / counts.min()),
                  minority_count=int(counts.min()))
    else:
        mf.update(target_skew=float(skew(y)) if np.std(y) > 0 else 0.0, target_nonneg=bool(y.min() >= 0))
    return mf


def _hgb_params(n):
    if n < 1000:
        return dict(learning_rate=0.04, max_iter=300, max_leaf_nodes=15, min_samples_leaf=max(5, n // 50),
                    l2_regularization=1.0, early_stopping=False)
    if n < 2000:
        p = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=15)
    elif n < 20000:
        p = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20)
    else:
        p = dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=40)
    p.update(max_iter=1500, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
             n_iter_no_change=30)
    return p


def _linearity_probe(feats, y, clf, C, budget):
    """3-fold CV on <=3000 rows: linear model vs a quick HistGB. Returns (linear_score, tree_score)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    n = len(y)
    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(n, min(n, 3000), replace=False))
    folds = make_folds(y[idx], 3, "classification" if clf else "regression")
    ls, ts = [], []
    for tr, va in folds:
        tr, va = idx[tr], idx[va]
        Ltr, Lva, _, _ = feats.view("linear", tr, y[tr], va)
        Ttr, Tva, _, mask = feats.view("tree", tr, y[tr], va)
        if clf:
            lm = LogisticRegression(C=1.0, max_iter=1000).fit(Ltr, y[tr])
            tm = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, categorical_features=mask,
                                                random_state=SEED).fit(Ttr, y[tr])
            ls.append(macro_f1(y[va], lm.predict(Lva), C))
            ts.append(macro_f1(y[va], tm.predict(Tva), C))
        else:
            lm = Ridge(alpha=1.0).fit(Ltr, y[tr])
            tm = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, categorical_features=mask,
                                               random_state=SEED).fit(Ttr, y[tr])
            ls.append(r2(y[va], lm.predict(Lva)))
            ts.append(r2(y[va], tm.predict(Tva)))
        if not budget.has(RESERVE + 10):
            break
    return float(np.mean(ls)), float(np.mean(ts))


def decide(mf, probe):
    """The rule base. Returns a plan dict (fully logged in the profile)."""
    n, p = mf["n"], mf["p_linear"]
    plan = {"rules_fired": []}
    if n < 400:
        plan["regime"], plan["models"] = "small", [("linear", 0.5), ("extratrees", 0.5)]
        plan["rules_fired"].append("n<400 -> linear+ExtraTrees blend")
    elif mf.get("n_classes", 0) >= 8 and n / mf["n_classes"] < 150:
        plan["regime"], plan["models"] = "many_classes", [("extratrees", 0.6), ("linear", 0.4)]
        plan["rules_fired"].append("n_classes>=8 & n/class<150 -> ExtraTrees+linear blend")
    elif p > 150 and n < 10 * p:
        plan["regime"], plan["models"] = "wide", [("linear", 0.5), ("hgb", 0.5)]
        plan["rules_fired"].append("p>150 & n<10p -> linear+HGB blend")
    elif probe is not None and probe[0] >= probe[1] - 0.005:
        plan["regime"], plan["models"] = "linear", [("linear", 0.5), ("hgb", 0.5)]
        plan["rules_fired"].append(f"probe linear {probe[0]:.4f} >= tree {probe[1]:.4f} - 0.005 -> linear+HGB blend")
    else:
        plan["regime"], plan["models"] = "nonlinear", [("hgb", 1.0)]
        plan["rules_fired"].append("default nonlinear -> HGB")
    if "imbalance_ratio" in mf:
        plan["balanced"] = mf["imbalance_ratio"] > 1.5
        plan["tune_threshold"] = plan["balanced"] and mf["minority_count"] >= 20
        if plan["balanced"]:
            plan["rules_fired"].append("imbalance>1.5 -> capped balanced weights")
        if plan["tune_threshold"]:
            plan["rules_fired"].append("minority>=20 -> OOF macro-F1 class-weight tuning")
    else:
        plan["log_target"] = mf["target_nonneg"] and mf["target_skew"] > 0.75
        if plan["log_target"]:
            plan["rules_fired"].append("y>=0 & skew>0.75 -> log1p target")
    plan["folds"] = 5 if n <= 20000 else 3
    plan["refit_full"] = True
    if mf.get("has_datetime"):
        plan["rules_fired"].append("datetime -> calendar + cyclic features")
    if mf.get("n_high_card"):
        plan["rules_fired"].append("high-cardinality categoricals -> target encoding")
    return plan


# --------------------------------------------------------------------------- supervised
def _capped_balanced(counts, cap=25.0):
    n, present = counts.sum(), counts > 0
    w = np.where(present, n / (present.sum() * np.maximum(counts, 1)), 1.0)
    w = np.minimum(w, cap * w[present].min())
    return {int(c): float(w[c]) for c in np.where(present)[0]}


def _make(kind, clf, n, plan, counts, nj, n_iter=None, wide=None):
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, RidgeCV
    cw = _capped_balanced(counts) if (clf and plan.get("balanced")) else None
    if kind == "hgb":
        prm = _hgb_params(n)
        prm.update(wide or {})
        if n_iter:
            prm.update(max_iter=int(n_iter), early_stopping=False)
        H = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return lambda mask: H(random_state=SEED, categorical_features=mask, **prm,
                              **({"class_weight": cw} if cw else {}))
    if kind == "linear":
        if clf:
            Cr = 1.0 if n >= 2000 else 0.3
            return lambda mask: LogisticRegression(C=Cr, max_iter=2000, **({"class_weight": cw} if cw else {}))
        return lambda mask: RidgeCV(alphas=np.logspace(-3, 3, 13))
    if clf:
        return lambda mask: ExtraTreesClassifier(500, max_features="sqrt", min_samples_leaf=2, n_jobs=nj,
                                                 random_state=SEED,
                                                 **({"class_weight": "balanced_subsample"} if cw else {}))
    return lambda mask: ExtraTreesRegressor(500, max_features=0.7, min_samples_leaf=2, n_jobs=nj, random_state=SEED)


VIEW = {"hgb": "tree", "linear": "linear", "extratrees": "forest"}


def _predict(m, X, clf, C, log_t):
    if clf:
        return align_proba(m, m.predict_proba(X), C)
    p = m.predict(X)
    return np.expm1(np.clip(p, None, 60)) if log_t else p


def _supervised(task, prep, budget, emit, log):
    clf = task.task_type == "classification"
    if clf:
        y, classes, keep = encode_classes(task.y_train)
        C = len(classes)
        if C < 2:
            emit(np.repeat(classes[0] if C else 0, task.n_predict), "single_class")
            return {"model": "constant"}
        counts = np.bincount(y, minlength=C)
    else:
        y, keep = encode_regression_target(task.y_train)
        if len(y) < 5:
            emit(np.full(task.n_predict, float(np.median(y)) if len(y) else 0.0), "median")
            return {"model": "median"}
        C, counts = None, None
    n = len(y)
    nj = get_n_jobs()
    feats = Features(prep, keep, clf)
    mf = meta_features(prep, y, clf, feats)
    probe = None
    many = clf and mf["n_classes"] >= 8 and n / mf["n_classes"] < 150
    if n >= 400 and not many and not (mf["p_linear"] > 150 and n < 10 * mf["p_linear"]):
        probe = _linearity_probe(feats, y if clf else (np.log1p(y) if (mf["target_nonneg"] and mf["target_skew"] > 0.75) else y),
                                 clf, C, budget)
    plan = decide(mf, probe)
    log(f"plan: {plan['regime']} {plan['models']} | {plan['rules_fired']}")
    info = {"meta_features": mf, "plan": {k: v for k, v in plan.items() if k != "models"},
            "blend": {k: w for k, w in plan["models"]}, "linearity_probe": probe}
    log_t = bool(plan.get("log_target"))
    yt = np.log1p(y) if log_t else y

    folds = make_folds(y, plan["folds"], task.task_type)
    all_idx = np.arange(n)
    oof = {k: (np.zeros((n, C)) if clf else np.zeros(n)) for k, _ in plan["models"]}
    test_bag = {k: 0.0 for k, _ in plan["models"]}
    iters = {k: [] for k, _ in plan["models"]}
    done, t0 = 0, time.perf_counter()
    for tr, va in folds:
        if done and budget.remaining() - RESERVE < (time.perf_counter() - t0) / done * 2.2:
            log("budget: stopping CV early")
            break
        for kind, _ in plan["models"]:
            Xtr, Xva, Xte, mask = feats.view(VIEW[kind], tr, y[tr], va)
            m = _make(kind, clf, len(tr), plan, counts, nj, wide=feats.wide)(mask).fit(Xtr, yt[tr])
            oof[kind][va] = _predict(m, Xva, clf, C, log_t)
            test_bag[kind] = test_bag[kind] + _predict(m, Xte, clf, C, log_t)
            if kind == "hgb" and hasattr(m, "n_iter_"):
                iters[kind].append(int(m.n_iter_))
        done += 1
        if done == 1:
            tb = sum(w * test_bag[k] for k, w in plan["models"])
            emit(classes[np.argmax(tb, 1)] if clf else _clip(tb, y), "fold1")
    used = np.concatenate([va for _, va in folds[:done]])
    oof_bl = sum(w * oof[k] for k, w in plan["models"])
    test_p = sum(w * test_bag[k] / done for k, w in plan["models"])
    info["cv_folds_done"] = done

    # refit on all rows (rule: final models see 100% of the data)
    cv_time = time.perf_counter() - t0
    if plan["refit_full"] and budget.remaining() - RESERVE > cv_time / max(done, 1) * 1.6:
        refit = 0.0
        for kind, w in plan["models"]:
            n_it = int(np.median(iters[kind]) * 1.1) + 1 if iters[kind] and kind == "hgb" and n >= 1000 else None
            Xtr, _, Xte, mask = feats.view(VIEW[kind], all_idx, y, None)
            m = _make(kind, clf, n, plan, counts, nj, n_iter=n_it, wide=feats.wide)(mask).fit(Xtr, yt)
            refit = refit + w * _predict(m, Xte, clf, C, log_t)
        test_p = refit
        info["final_fit"] = "refit_full"
    else:
        info["final_fit"] = "fold_average"

    if clf:
        wcls = np.ones(C)
        if plan.get("tune_threshold"):
            wt, f0, f1 = tune_class_weights(oof_bl[used], y[used], C, n_rounds=1)
            if f1 - f0 > 0.002:
                wcls = wt
            info["oof_macro_f1_argmax"] = round(f0, 5)
        info["oof_macro_f1"] = round(macro_f1(y[used], np.argmax(oof_bl[used] * wcls, 1), C), 5)
        info["class_weights_decision"] = [round(float(v), 4) for v in wcls]
        info["model"] = "+".join(f"{w}*{k}" for k, w in plan["models"])
        emit(classes[np.argmax(test_p * wcls, 1)], "final")
    else:
        info["oof_rmse"] = round(rmse(y[used], oof_bl[used]), 6)
        info["oof_r2"] = round(r2(y[used], oof_bl[used]), 6)
        info["model"] = "+".join(f"{w}*{k}" for k, w in plan["models"]) + ("[log1p]" if log_t else "")
        emit(_clip(test_p, y), "final")
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


def _wkmeans(X, w, k, n_init, seed=SEED):
    from sklearn.cluster import KMeans
    sel = w > 0
    Xw = X[:, sel] * np.sqrt(w[sel])
    km = KMeans(k, n_init=n_init, random_state=seed).fit(Xw)
    return km.labels_


def _bcss(X, labels, k):
    """Per-feature between-cluster sum of squares."""
    tss = ((X - X.mean(0)) ** 2).sum(0)
    wss = np.zeros(X.shape[1])
    for c in range(k):
        Z = X[labels == c]
        if len(Z):
            wss += ((Z - Z.mean(0)) ** 2).sum(0)
    return tss - wss


def _update_w(a, s):
    a = np.maximum(a, 0)
    if a.sum() <= 0:
        return np.ones_like(a) / np.sqrt(len(a))
    w = a / np.linalg.norm(a)
    if np.abs(w).sum() <= s:
        return w
    lo, hi = 0.0, a.max()
    for _ in range(50):
        d = (lo + hi) / 2
        st = np.maximum(a - d, 0)
        w = st / (np.linalg.norm(st) + 1e-12)
        if w.sum() > s:
            lo = d
        else:
            hi = d
    st = np.maximum(a - hi, 0)
    return st / (np.linalg.norm(st) + 1e-12)


def sparse_kmeans(X, k, s, n_init=5, n_iter=8):
    p = X.shape[1]
    w = np.ones(p) / np.sqrt(p)
    labels = None
    for _ in range(n_iter):
        labels = _wkmeans(X, w, k, n_init)
        a = _bcss(X, labels, k)
        w_new = _update_w(a, s)
        if np.abs(w_new - w).sum() / np.abs(w).sum() < 1e-4:
            w = w_new
            break
        w = w_new
    a = _bcss(X, labels, k)
    return labels, w, float((w * a).sum())


def _choose_s(X, k, budget, rng, B=3):
    """Permutation gap statistic over a small sparsity grid (Witten & Tibshirani 2010)."""
    p = X.shape[1]
    grid = np.unique(np.round(np.linspace(1.2, np.sqrt(p), 5), 3))
    gaps, sds = [], []
    for s in grid:
        _, _, o = sparse_kmeans(X, k, s, n_init=3, n_iter=5)
        ob = []
        for _ in range(B):
            Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
            ob.append(sparse_kmeans(Xp, k, s, n_init=2, n_iter=5)[2])
        lb = np.log(np.maximum(ob, 1e-12))
        gaps.append(np.log(max(o, 1e-12)) - lb.mean())
        sds.append(lb.std())
        if not budget.has(RESERVE + 6):
            break
    gaps, sds = np.array(gaps), np.array(sds)
    best = int(np.argmax(gaps))
    # smallest s within one sd of the best gap (sparser is preferred)
    ok = np.where(gaps >= gaps[best] - sds[best])[0]
    return float(grid[ok.min()]), dict(zip([round(float(g), 3) for g in grid[:len(gaps)]],
                                           [round(float(v), 4) for v in gaps]))


def _clustering(task, prep, budget, emit, log):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    num, cat = _concat(prep.num_tr, prep.num_te), _concat(prep.cat_tr, prep.cat_te)
    A, _ = numeric_dense(num, None, indicators=False, log_skewed=True, scale=True)
    Oh, _, names = onehot_encode(cat, None, max_levels=15, min_count=2, scale=1 / np.sqrt(2))
    X = np.hstack([A, Oh])
    if X.shape[1] == 0:
        X = np.zeros((len(num), 1))
    n, p = X.shape
    n_pred = task.n_predict
    k = task.n_clusters or choose_k_silhouette(X)
    k = int(min(k, n))
    rng = np.random.RandomState(SEED)
    info = {"k": k, "n_features_in": int(p)}

    base = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    emit(ensure_k_labels(base, X, k)[-n_pred:], "kmeans_all_features")

    # 1) sparse k-means feature weighting (rule: only when there are >= 3 features)
    w = np.ones(p) / np.sqrt(p)
    if p >= 3 and k >= 2:
        sub = X if n <= 3000 else X[rng.choice(n, 3000, replace=False)]
        s, gaps = _choose_s(sub, k, budget, rng)
        labels_s, w, _ = sparse_kmeans(X, k, s, n_init=10 if n <= 20000 else 3)
        info.update(sparsity_s=s, gap_by_s=gaps)
        colnames = list(num.columns) + names
        info["feature_weights"] = {colnames[j] if j < len(colnames) else f"f{j}": round(float(w[j]), 4)
                                   for j in np.argsort(-w)[:30]}
        info["n_features_selected"] = int((w > 1e-6).sum())
    sel = w > 1e-6
    Xs = X[:, sel] * np.sqrt(w[sel] / w[sel].max())
    km = KMeans(k, n_init=20 if n <= 20000 else 4, random_state=SEED).fit(Xs)
    labels = km.labels_
    emit(ensure_k_labels(labels, Xs, k)[-n_pred:], "sparse_kmeans")
    info["model"] = "sparse_kmeans"

    # 2) KMeans vs GMM by BIC (rule: only if budget allows and n >= 20*k)
    if n >= 20 * k and budget.has(RESERVE + 3):
        fit_idx = np.arange(n) if n <= 20000 else rng.choice(n, 20000, replace=False)
        ps = Xs.shape[1]
        types = ["spherical", "diag"] + (["full"] if ps <= 8 else [])
        bics = {}
        best = None
        for cov in types:
            try:
                g = GaussianMixture(k, covariance_type=cov, means_init=km.cluster_centers_, reg_covar=1e-4,
                                    random_state=SEED, max_iter=200).fit(Xs[fit_idx])
                bics[cov] = float(g.bic(Xs[fit_idx]))
                if best is None or bics[cov] < bics[best[0]]:
                    best = (cov, g)
            except Exception:  # noqa: BLE001
                continue
        info["gmm_bic"] = {c: round(v, 2) for c, v in bics.items()}
        if best is not None and best[0] != "spherical":
            gl = best[1].predict(Xs)
            sizes = np.bincount(gl, minlength=k)
            if (sizes >= max(2, 0.01 * n)).all():
                labels = gl
                info["model"] = f"sparse_kmeans->GMM({best[0]})"
            else:
                info["gmm_rejected"] = "degenerate cluster sizes"
    labels = ensure_k_labels(relabel_contiguous(labels), Xs, k)
    emit(labels[-n_pred:], "final")
    return info
