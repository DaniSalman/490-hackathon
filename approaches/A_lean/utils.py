"""Small shared helpers: time budget, thread caps, metrics, CV folds, class-weight tuning.

Only numpy is imported at module level; everything else is imported lazily.
"""
import os
import time
import warnings

import numpy as np

SEED = 0


# --------------------------------------------------------------------------- budget
class Budget:
    """Wall-clock budget measured from process start (t0 is taken in main.py before imports)."""

    def __init__(self, t0, total_seconds):
        self.t0 = t0
        self.total = float(total_seconds)

    def elapsed(self):
        return time.perf_counter() - self.t0

    def remaining(self):
        return self.total - self.elapsed()

    def has(self, seconds):
        return self.remaining() > seconds


def get_n_jobs():
    """Thread/process cap. Respects HACK_N_JOBS, then OMP_NUM_THREADS, default min(4, cores)."""
    for key in ("HACK_N_JOBS", "OMP_NUM_THREADS"):
        v = os.environ.get(key)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return max(1, min(4, os.cpu_count() or 1))


# --------------------------------------------------------------------------- metrics
def macro_f1(y, pred, n_classes):
    """Macro F1 over labels present in y or pred (same convention as sklearn)."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def log_loss(y, P):
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def r2(y, p):
    y = np.asarray(y, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - np.asarray(p, float)) ** 2) / ss) if ss > 0 else 0.0


# --------------------------------------------------------------------------- folds
def make_folds(y, n_splits, task_type, seed=SEED):
    """Return list of (train_idx, valid_idx). Stratified for classification; robust to rare classes."""
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(y)
    n_splits = int(max(2, min(n_splits, n)))
    if task_type == "classification":
        counts = np.bincount(y)
        counts = counts[counts > 0]
        if counts.max() >= n_splits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                return list(skf.split(np.zeros(n), y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n)))


def align_proba(model, P, n_classes):
    """Map predict_proba columns (model.classes_ may miss classes absent from a fold) to 0..C-1."""
    out = np.zeros((P.shape[0], n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = P
    return out


# --------------------------------------------------------------------------- decision rule
def tune_class_weights(P, y, n_classes, n_rounds=2):
    """Coordinate ascent on per-class multiplicative probability weights to maximise macro-F1.

    Returns (weights, f1_before, f1_after).
    """
    w = np.ones(n_classes)
    base = macro_f1(y, np.argmax(P, 1), n_classes)
    best = base
    if n_classes < 2:
        return w, base, best
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 31))
    classes = range(1, n_classes) if n_classes == 2 else range(n_classes)
    for _ in range(n_rounds):
        improved = False
        for c in classes:
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = macro_f1(y, np.argmax(P * w2, 1), n_classes)
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, base, best


# --------------------------------------------------------------------------- misc
def relabel_contiguous(labels):
    """Map arbitrary cluster labels to 0..m-1 ordered by first appearance-size (largest = 0)."""
    labels = np.asarray(labels)
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(len(uniq), dtype=int)
    remap[order] = np.arange(len(uniq))
    return remap[inv]


def choose_k_silhouette(X, k_range=range(2, 9), seed=SEED):
    """Used only when metadata does not provide n_clusters."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, min(n, 3000), replace=False)
    best_k, best_s = 2, -2.0
    for k in k_range:
        if k >= len(idx):
            break
        lab = KMeans(k, n_init=4, random_state=seed).fit_predict(X[idx])
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X[idx], lab)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def ensure_k_labels(labels, X, k, seed=SEED):
    """Guarantee exactly min(k, n) distinct labels 0..k-1 (split largest clusters if some are empty)."""
    labels = relabel_contiguous(labels)
    n = len(labels)
    k = min(k, n)
    m = labels.max() + 1 if n else 0
    if m > k:  # too many: merge smallest into nearest by centroid
        from sklearn.cluster import KMeans
        return relabel_contiguous(KMeans(k, n_init=3, random_state=seed).fit_predict(X))
    rng = np.random.RandomState(seed)
    while m < k:
        counts = np.bincount(labels, minlength=m)
        big = int(np.argmax(counts))
        members = np.where(labels == big)[0]
        if len(members) < 2:
            break
        # split the largest cluster along its highest-variance feature
        sub = X[members]
        j = int(np.argmax(sub.var(0))) if sub.shape[1] else 0
        v = sub[:, j] if sub.shape[1] else rng.rand(len(members))
        thr = np.median(v)
        move = members[v > thr] if (v > thr).any() else members[: len(members) // 2]
        labels[move] = m
        m += 1
    return labels
