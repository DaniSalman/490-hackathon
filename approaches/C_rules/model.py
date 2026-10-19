"""Approach C -- profile-driven rules ("expert system") + specialised pipelines.

The data profile (size, width, categorical cardinality, imbalance, target skew, datetime presence)
plus one cheap "linearity probe" deterministically select the pipeline -- no portfolio search.
A quick HistGB (100 iterations, <= 5k rows) is written to disk before the probe and the CV; its fit time
caps HistGB iterations, and every model fit inside the fold loop is budget-checked.

supervised rules
  encoding   : low-cardinality categoricals native in HistGB / one-hot for linear models;
               high-cardinality (>30 levels) -> cross-fitted target encoding with an explicit target type
               (continuous / binary / multiclass); ordinal fallback when the encoded width would exceed
               max(5p, p+200) or TargetEncoder is unavailable; datetimes get cyclic features.
  regime     : small (n<400)            -> 0.5 * regularised linear + 0.5 * ExtraTrees
               many classes, few rows/class -> 0.6 * ExtraTrees + 0.4 * linear
               wide (p >> n)            -> 0.5 * linear + 0.5 * HistGB (regression: only if the probe's linear
                                           model beats the tree by >= 0.01 R2)
               linear-ish (probe)       -> 0.5 * linear + 0.5 * HistGB (classification: linear >= tree - 0.005;
                                           regression: linear >= tree + 0.01)
               otherwise (nonlinear)    -> HistGB with size-tiered hyper-parameters
  imbalance  : ratio > 1.5 -> capped balanced weights; OOF macro-F1 class-weight tuning if minority >= 20
  target     : log1p when y >= 0 and skew > 0.75 (rule); linear members clipped to the training target range
               (in log space); predictions clipped
  final fit  : refit on all training rows (HistGB iterations = 1.1 x median early-stopped CV iterations);
               falls back to the fold-averaged models if the projected refit cost does not fit.
clustering    : group-aware sparse k-means (Witten & Tibshirani 2010; one-hot blocks weighted like numerics);
               the soft-weighted space is used only when some feature weight < 5 % of the largest; then a GMM
               (diag / full) replaces KMeans only if it agrees with it (ARI >= 0.6) and improves BIC by >= 15 %
               (full covariance only when well determined), with a degenerate-cluster guard.
"""
import time
import warnings

import numpy as np

from cleaning import (encode_classes, encode_regression_target, max_encoded_features, numeric_dense, onehot_encode,
                      ordinal_encode, prepare)
from utils import (SEED, FitTimer, align_proba, choose_k_silhouette, ensure_k_labels, get_n_jobs, hgb_wide_params, cap_cat_codes, macro_f1,
                   make_folds, r2, subsample_idx,
                   relabel_contiguous, rmse, tune_class_weights)

warnings.filterwarnings("ignore")
RESERVE = 4.0
REG_LINEAR_MARGIN = 0.01   # regression: linear must beat the quick tree by this R2 margin to be blended
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

    def __init__(self, prep, keep, clf, n_classes=None):
        self.clf = clf
        # explicit target type (C1): "auto" turned an integer-valued regression target into "multiclass"
        # (one encoded column per distinct price -> 3412 features on california)
        self.target_type = "continuous" if not clf else ("binary" if n_classes == 2 else "multiclass")
        cat_tr = prep.cat_tr.iloc[np.where(keep)[0]].reset_index(drop=True)
        card = cat_tr.nunique() if cat_tr.shape[1] else []
        self.hi = [c for c in cat_tr.columns if card[c] > HIGH_CARD]
        self.lo = [c for c in cat_tr.columns if card[c] <= HIGH_CARD]
        # cost guard (S2a): target-encoded width must stay within max(5p, p+200); else ordinal fallback
        p_raw = prep.num_tr.shape[1] + prep.cat_tr.shape[1]
        te_width = len(self.hi) * (n_classes if (clf and n_classes and n_classes > 2) else 1)
        self.te_mode = "target_encoding"
        try:
            from sklearn.preprocessing import TargetEncoder  # noqa: F401  (sklearn >= 1.3)
        except ImportError:
            self.te_mode = "ordinal_fallback(no TargetEncoder)"
        if self.hi and prep.num_tr.shape[1] + len(self.lo) + te_width > max_encoded_features(p_raw):
            self.te_mode = "ordinal_fallback(width cap)"
        if self.hi and self.te_mode != "target_encoding":
            self.lo = self.lo + self.hi          # high-cardinality columns become ordinal codes (top 250 levels)
            self.hi = []
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
        tt = self.target_type
        if tt == "multiclass" and len(np.unique(y_tr)) <= 2:
            tt = "binary"
        te = TargetEncoder(target_type=tt, random_state=SEED)
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
    elif p > 150 and n < 10 * p and ("n_classes" in mf or (probe is not None and probe[0] >= probe[1] + REG_LINEAR_MARGIN)):
        plan["regime"], plan["models"] = "wide", [("linear", 0.5), ("hgb", 0.5)]
        plan["rules_fired"].append("p>150 & n<10p -> linear+HGB blend" +
                                   ("" if "n_classes" in mf else " (regression: probe linear beats tree by margin)"))
    elif probe is not None and probe[0] >= probe[1] + (-0.005 if "n_classes" in mf else REG_LINEAR_MARGIN):
        # regression needs the linear model to BEAT the tree by a margin (C3: heavy-tailed house prices)
        m = -0.005 if "n_classes" in mf else REG_LINEAR_MARGIN
        plan["regime"], plan["models"] = "linear", [("linear", 0.5), ("hgb", 0.5)]
        plan["rules_fired"].append(f"probe linear {probe[0]:.4f} >= tree {probe[1]:.4f} {m:+.3f} -> linear+HGB blend")
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


def _make(kind, clf, n, plan, counts, nj, n_iter=None, wide=None, it_cap=None):
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, RidgeCV
    cw = _capped_balanced(counts) if (clf and plan.get("balanced")) else None
    if kind == "hgb":
        prm = _hgb_params(n)
        prm.update(wide or {})
        if n_iter:
            prm.update(max_iter=int(n_iter), early_stopping=False)
        if it_cap is not None:
            prm["max_iter"] = int(max(1, min(prm["max_iter"], it_cap)))
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


def _predict(m, X, clf, C, log_t, clip=None):
    """clip: (lo, hi) in the (possibly log) target space -- used for linear members (no extrapolation)."""
    if clf:
        return align_proba(m, m.predict_proba(X), C)
    p = m.predict(X)
    if clip is not None:
        p = np.clip(p, clip[0], clip[1])
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
    feats = Features(prep, keep, clf, C)
    mf = meta_features(prep, y, clf, feats)
    log_rule = (not clf) and mf["target_nonneg"] and mf["target_skew"] > 0.75

    # C2: a cheap real model on disk BEFORE the probe and CV (HistGB, 100 iterations, <= 5k rows); it is
    # also the per-fit cost probe that caps HistGB iterations later
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    timer = FitTimer()
    qi = subsample_idx(y, 5000, task.task_type)
    Xq, _, Xqt, qmask = feats.view("tree", qi, y[qi], None)
    q_iters = 100 if (not clf or C == 2) else max(30, 200 // C)
    t0 = time.perf_counter()
    H = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
    qm = H(max_iter=q_iters, learning_rate=0.1, categorical_features=qmask, random_state=SEED,
           **feats.wide).fit(Xq, np.log1p(y[qi]) if log_rule else y[qi])
    timer.observe("hgb", len(qi), q_iters, time.perf_counter() - t0)
    q_test = _predict(qm, Xqt, clf, C, log_rule)
    emit(classes[np.argmax(q_test, 1)] if clf else _clip(q_test, y), "quick_hgb")

    probe = None
    many = clf and mf["n_classes"] >= 8 and n / mf["n_classes"] < 150
    wide = mf["p_linear"] > 150 and n < 10 * mf["p_linear"]
    if n >= 400 and not many and (not wide or not clf) and budget.has(RESERVE + 8):
        probe = _linearity_probe(feats, y if clf else (np.log1p(y) if log_rule else y), clf, C, budget)
    plan = decide(mf, probe)
    log(f"plan: {plan['regime']} {plan['models']} | {plan['rules_fired']}")
    info = {"meta_features": mf, "plan": {k: v for k, v in plan.items() if k != "models"},
            "blend": {k: w for k, w in plan["models"]}, "linearity_probe": probe,
            "high_cardinality_encoding": feats.te_mode if mf["n_high_card"] or "fallback" in feats.te_mode else None}
    log_t = bool(plan.get("log_target"))
    yt = np.log1p(y) if log_t else y

    folds = make_folds(y, plan["folds"], task.task_type)
    n_tr = int(n * (1 - 1 / len(folds)))
    # S2: HistGB iteration cap from the measured quick-fit rate (factor 3: safety net, not a speed limiter)
    it_cap = None
    if any(k == "hgb" for k, _ in plan["models"]):
        per_fold = (budget.remaining() - RESERVE) / (len(folds) + 1.5)
        it_cap = int(3.0 * per_fold / (timer.rate["hgb"] * n_tr))
        if it_cap < _hgb_params(n_tr)["max_iter"]:
            info["max_iter_capped_by_budget"] = int(max(it_cap, 0))
        if it_cap < 40:
            log("budget: keeping the quick model (CV not affordable)")
            info["model"] = "quick HistGB (budget)"
            return info
    all_idx = np.arange(n)
    oof = {k: (np.zeros((n, C)) if clf else np.zeros(n)) for k, _ in plan["models"]}
    test_bag = {k: 0.0 for k, _ in plan["models"]}
    iters = {k: [] for k, _ in plan["models"]}
    last = {}
    done = 0
    for tr, va in folds:
        fold_ok = True
        for kind, _ in plan["models"]:
            # per-model budget check inside the fold loop (C2/S2c)
            if kind in last and budget.remaining() - RESERVE < 1.3 * last[kind]:
                fold_ok = False
                break
            t1 = time.perf_counter()
            Xtr, Xva, Xte, mask = feats.view(VIEW[kind], tr, y[tr], va)
            mk = _make(kind, clf, len(tr), plan, counts, nj, wide=feats.wide, it_cap=it_cap)
            m = mk(mask).fit(Xtr, yt[tr])
            clip = (yt[tr].min(), yt[tr].max()) if (kind == "linear" and not clf) else None
            oof[kind][va] = _predict(m, Xva, clf, C, log_t, clip)
            test_bag[kind] = test_bag[kind] + _predict(m, Xte, clf, C, log_t, clip)
            if kind == "hgb" and hasattr(m, "n_iter_"):
                iters[kind].append(int(m.n_iter_))
            last[kind] = time.perf_counter() - t1
        if not fold_ok:
            log("budget: stopping CV early")
            break
        done += 1
        if done == 1:
            tb = sum(w * test_bag[k] for k, w in plan["models"])
            emit(classes[np.argmax(tb, 1)] if clf else _clip(tb, y), "fold1")
    info["cv_folds_done"] = done
    if done == 0:
        info["model"] = "quick HistGB (budget)"
        return info
    used = np.concatenate([va for _, va in folds[:done]])
    oof_bl = sum(w * oof[k][used] for k, w in plan["models"])
    test_p = sum(w * test_bag[k] / done for k, w in plan["models"])

    # refit on all rows (rule: final models see 100% of the data), if the projected cost fits
    refit_est = sum(last.values()) * n / max(1, n_tr) * 1.3
    if plan["refit_full"] and budget.remaining() - RESERVE > refit_est:
        refit = 0.0
        for kind, w in plan["models"]:
            n_it = int(np.median(iters[kind]) * 1.1) + 1 if iters[kind] and kind == "hgb" and n >= 1000 else None
            Xtr, _, Xte, mask = feats.view(VIEW[kind], all_idx, y, None)
            m = _make(kind, clf, n, plan, counts, nj, n_iter=n_it, wide=feats.wide, it_cap=it_cap)(mask).fit(Xtr, yt)
            clip = (yt.min(), yt.max()) if (kind == "linear" and not clf) else None
            refit = refit + w * _predict(m, Xte, clf, C, log_t, clip)
        test_p = refit
        info["final_fit"] = "refit_full"
    else:
        info["final_fit"] = "fold_average"

    yu = y[used]
    if clf:
        wcls = np.ones(C)
        if plan.get("tune_threshold"):
            wt, f0, f1 = tune_class_weights(oof_bl, yu, C, n_rounds=1)
            if f1 - f0 > 0.002:
                wcls = wt
            info["oof_macro_f1_argmax"] = round(f0, 5)
        info["oof_macro_f1"] = round(macro_f1(yu, np.argmax(oof_bl * wcls, 1), C), 5)
        info["class_weights_decision"] = [round(float(v), 4) for v in wcls]
        info["model"] = "+".join(f"{w}*{k}" for k, w in plan["models"])
        emit(classes[np.argmax(test_p * wcls, 1)], "final")
    else:
        info["oof_rmse"] = round(rmse(yu, oof_bl), 6)
        info["oof_r2"] = round(r2(yu, oof_bl), 6)
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


SPARSE_ACTIVE = 0.05   # weighted space only if some feature weight < 5 % of the largest (irrelevant features)
GMM_AGREEMENT = 0.60   # ARI(GMM, KMeans) required to accept a GMM partition
GMM_BIC_GAIN = 15.0    # required % BIC improvement over the spherical (KMeans-like) mixture


def _clustering(task, prep, budget, emit, log):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ARI
    from sklearn.mixture import GaussianMixture
    import feature_weighting as FW

    num, cat = _concat(prep.num_tr, prep.num_te), _concat(prep.cat_tr, prep.cat_te)
    X, groups, names, is_cat = FW.cluster_matrix(num, cat)
    n, p = X.shape
    n_pred = task.n_predict
    k = task.n_clusters or choose_k_silhouette(X)
    k = int(min(k, n))
    rng = np.random.RandomState(SEED)
    info = {"k": k, "n_features_in": int(p)}

    base = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    emit(ensure_k_labels(base, X, k)[-n_pred:], "kmeans_all_features")

    # 1) group-aware sparse k-means (C4a: one-hot blocks are judged like numerics, by explained-variance
    #    ratio); rule: the soft-weighted space is used only when sparsity finds clearly irrelevant features
    Xs = X
    info["model"] = "kmeans"
    if len(groups) >= 3 and k >= 2 and n >= 10 and budget.has(RESERVE + 4):
        r = FW.fit_sparse(X, groups, k, budget)
        w = r["w"]
        info.update(sparsity_s=r["s"], gap_by_s=r["gaps"],
                    feature_weights={names[g]: round(float(w[g]), 4) for g in np.argsort(-w)[:30]})
        if w.min() < SPARSE_ACTIVE * w.max():
            Xs = FW.weighted(X, groups, w / w.max())
            info["n_features_selected"] = int((w > 0).sum())
            info["model"] = "sparse_kmeans"
        else:
            info["sparsity_rule"] = "no clearly irrelevant feature -> unweighted space"
    km = KMeans(k, n_init=20 if n <= 20000 else 4, random_state=SEED).fit(Xs)
    labels = km.labels_
    emit(ensure_k_labels(labels, Xs, k)[-n_pred:], info["model"])

    # 2) KMeans vs GMM (C4b): a GMM replaces KMeans only if it AGREES with it (ARI) and BIC improves strongly;
    #    full covariance only when its parameters are well determined (p <= 4 or n >= 10 k p(p+1)/2)
    if n >= 20 * k and budget.has(RESERVE + 3):
        fit_idx = np.arange(n) if n <= 20000 else rng.choice(n, 20000, replace=False)
        Z = Xs[fit_idx]
        ps = Xs.shape[1]
        checks, best = {}, None
        try:
            bs = GaussianMixture(k, covariance_type="spherical", means_init=km.cluster_centers_, reg_covar=1e-4,
                                 random_state=SEED, max_iter=200).fit(Z).bic(Z)
            for cov in ("diag", "full"):
                if cov == "full" and not (ps <= 4 or len(Z) >= 10 * k * ps * (ps + 1) / 2):
                    continue
                g = GaussianMixture(k, covariance_type=cov, means_init=km.cluster_centers_, reg_covar=1e-4,
                                    random_state=SEED, max_iter=200).fit(Z)
                bic = float(g.bic(Z))
                gl = g.predict(Xs)
                gain = float((bs - bic) / abs(bs) * 100)
                agree = float(ARI(labels, gl))
                ok = (agree >= GMM_AGREEMENT and gain >= GMM_BIC_GAIN
                      and np.bincount(gl, minlength=k).min() >= max(2, 0.01 * n))
                checks[cov] = {"agreement": round(agree, 3), "bic_gain_pct": round(gain, 2), "accepted": bool(ok)}
                if ok and (best is None or bic < best[0]):
                    best = (bic, cov, gl)
        except Exception as e:  # noqa: BLE001
            checks["error"] = repr(e)[:120]
        info["gmm_checks"] = checks
        if best is not None:
            labels = best[2]
            info["model"] += f"->GMM({best[1]})"
    labels = ensure_k_labels(relabel_contiguous(labels), Xs, k)
    emit(labels[-n_pred:], "final")
    return info
