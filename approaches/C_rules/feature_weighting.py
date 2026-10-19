"""Group-aware sparse k-means (Witten & Tibshirani 2010) for clustering feature relevance.

A "group" is one original feature: a numeric column, or all one-hot columns of one categorical. Feature
relevance is the fraction of the group's variance explained by the partition (between / total sum of
squares), so a one-hot block is judged on the same 0..1 scale as a standardised numeric column (raw
one-hot sums of squares are small and would otherwise be crushed by the sparsity penalty).
"""
import numpy as np

from cleaning import numeric_dense, onehot_encode

SEED = 0


def cluster_matrix(num, cat, max_levels=15):
    """Numeric: median impute, log1p skewed, standardise. Categoricals: one-hot / sqrt(2).

    Returns X, groups (list of column-index arrays), names, is_cat (bool per group).
    """
    A, _ = numeric_dense(num, None, indicators=False, log_skewed=True, scale=True)
    groups, names, is_cat = [], [], []
    for j, c in enumerate(num.columns):
        groups.append(np.array([j]))
        names.append(str(c))
        is_cat.append(False)
    blocks = [A]
    off = A.shape[1]
    for c in cat.columns:
        O, _, _ = onehot_encode(cat[[c]], None, max_levels=max_levels, min_count=2, scale=1 / np.sqrt(2))
        if O.shape[1] == 0:
            continue
        blocks.append(O)
        groups.append(np.arange(off, off + O.shape[1]))
        names.append(str(c))
        is_cat.append(True)
        off += O.shape[1]
    X = np.hstack(blocks) if off else np.zeros((len(num), 1))
    if off == 0:
        groups, names, is_cat = [np.array([0])], ["__const__"], [False]
    return X, groups, names, np.array(is_cat, bool)


def group_ratio(X, labels, groups):
    """Fraction of each group's total sum of squares explained by the partition (0..1)."""
    mu = X.mean(0)
    tss = ((X - mu) ** 2).sum(0)
    wss = np.zeros(X.shape[1])
    for c in np.unique(labels):
        Z = X[labels == c]
        wss += ((Z - Z.mean(0)) ** 2).sum(0)
    out = np.zeros(len(groups))
    for g, cols in enumerate(groups):
        t = tss[cols].sum()
        out[g] = (t - wss[cols].sum()) / t if t > 1e-12 else 0.0
    return np.clip(out, 0, 1)


def _update_w(a, s):
    a = np.maximum(a, 0)
    if a.sum() <= 0:
        return np.ones_like(a) / np.sqrt(len(a))
    w = a / np.linalg.norm(a)
    if w.sum() <= s:
        return w
    lo, hi = 0.0, a.max()
    for _ in range(60):
        d = (lo + hi) / 2
        st = np.maximum(a - d, 0)
        if (st / (np.linalg.norm(st) + 1e-12)).sum() > s:
            lo = d
        else:
            hi = d
    st = np.maximum(a - hi, 0)
    return st / (np.linalg.norm(st) + 1e-12)


def weighted(X, groups, w):
    """Scale each group's columns by sqrt(w_g); groups with w_g == 0 are removed."""
    cols, scale = [], []
    for g, c in enumerate(groups):
        if w[g] > 0:
            cols.append(c)
            scale.append(np.full(len(c), np.sqrt(w[g])))
    if not cols:
        return X
    cols, scale = np.concatenate(cols), np.concatenate(scale)
    return X[:, cols] * scale


def sparse_kmeans(X, groups, k, s, n_init=5, n_iter=8, seed=SEED):
    from sklearn.cluster import KMeans
    G = len(groups)
    w = np.ones(G) / np.sqrt(G)
    labels = None
    for _ in range(n_iter):
        labels = KMeans(k, n_init=n_init, random_state=seed).fit_predict(weighted(X, groups, w))
        w_new = _update_w(group_ratio(X, labels, groups), s)
        done = np.abs(w_new - w).sum() / np.abs(w).sum() < 1e-4
        w = w_new
        if done:
            break
    ratio = group_ratio(X, labels, groups)
    return labels, w, float((w * ratio).sum()), ratio


def choose_s(X, groups, k, rng, budget=None, reserve=6.0, B=3, n_grid=5):
    """Permutation gap statistic; returns (s, {s: gap}). Smallest s within 1 sd of the best gap."""
    G = len(groups)
    grid = np.unique(np.round(np.linspace(1.2, np.sqrt(G), n_grid), 3))
    gaps, sds = [], []
    for s in grid:
        o = sparse_kmeans(X, groups, k, s, n_init=3, n_iter=5)[2]
        ob = []
        for _ in range(B):
            Xp = X.copy()
            for c in groups:                     # permute each group's rows jointly (keeps one-hot structure)
                Xp[:, c] = X[rng.permutation(len(X))][:, c]
            ob.append(sparse_kmeans(Xp, groups, k, s, n_init=2, n_iter=5)[2])
        lb = np.log(np.maximum(ob, 1e-12))
        gaps.append(np.log(max(o, 1e-12)) - lb.mean())
        sds.append(lb.std())
        if budget is not None and not budget.has(reserve):
            break
    gaps, sds = np.array(gaps), np.array(sds)
    best = int(np.argmax(gaps))
    ok = np.where(gaps >= gaps[best] - sds[best])[0]
    return float(grid[ok.min()]), {round(float(g), 3): round(float(v), 4) for g, v in zip(grid, gaps)}


def fit_sparse(X, groups, k, budget=None, n_sub=3000, seed=SEED):
    """Choose s on <= n_sub rows, then run sparse k-means on all rows. Returns dict."""
    rng = np.random.RandomState(seed)
    n = len(X)
    sub = X if n <= n_sub else X[rng.choice(n, n_sub, replace=False)]
    s, gaps = choose_s(sub, groups, k, rng, budget)
    labels, w, _, ratio = sparse_kmeans(X, groups, k, s, n_init=10 if n <= 20000 else 3)
    return {"labels": labels, "w": w, "s": s, "gaps": gaps, "ratio": ratio}
