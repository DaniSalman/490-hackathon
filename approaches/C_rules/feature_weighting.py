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


# --------------------------------------------------------------------------- confirmed relevance filter
LOW_WEIGHT = 0.05      # candidate: sparse weight < 5 % of the largest
N_PERM = 200           # label permutations for the null distribution
ALPHA = 0.01           # a candidate is kept if its explained share is significant at this level


def _ratio_one(Xg, labels, k):
    """Explained share of the column block Xg under `labels` (vectorised via bincount)."""
    n = len(Xg)
    cnt = np.bincount(labels, minlength=k).astype(float)
    tss = ((Xg - Xg.mean(0)) ** 2).sum()
    if tss <= 1e-12:
        return 0.0
    bss = 0.0
    mu = Xg.mean(0)
    for j in range(Xg.shape[1]):
        s = np.bincount(labels, weights=Xg[:, j], minlength=k)
        m = np.divide(s, cnt, out=np.zeros(k), where=cnt > 0)
        bss += (cnt * (m - mu[j]) ** 2).sum()
    return bss / tss


def relevance_filter(X, groups, k, w, seed=SEED, n_sub=5000):
    """Drop a feature group only when (a) its sparse weight is low AND (b) a permutation test confirms that
    the cluster structure carried by the OTHER features explains no more of its variance than random labels.

    Principle: a feature that is relevant to the clustering must co-vary with the partition defined by the
    remaining features. (b) uses KMeans on the non-candidate groups, then compares each candidate's explained
    share with N_PERM label permutations (p-value). Returns (drop mask, diagnostics dict).
    """
    from sklearn.cluster import KMeans
    G = len(groups)
    cand = w < LOW_WEIGHT * w.max()
    drop = np.zeros(G, bool)
    diag = {}
    if not cand.any() or cand.all():
        return drop, diag
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = np.arange(n) if n <= n_sub else np.sort(rng.choice(n, n_sub, replace=False))
    core = np.concatenate([groups[g] for g in range(G) if not cand[g]])
    labels = KMeans(k, n_init=5, random_state=seed).fit_predict(X[idx][:, core])
    perms = [rng.permutation(labels) for _ in range(N_PERM)]
    for g in np.where(cand)[0]:
        Xg = X[idx][:, groups[g]]
        obs = _ratio_one(Xg, labels, k)
        null = np.array([_ratio_one(Xg, p, k) for p in perms])
        pval = (1 + (null >= obs).sum()) / (N_PERM + 1)
        drop[g] = pval > ALPHA
        diag[int(g)] = {"explained": round(float(obs), 4), "null_p99": round(float(np.quantile(null, 0.99)), 4),
                        "p_value": round(float(pval), 4), "dropped": bool(drop[g])}
    return drop, diag


# --------------------------------------------------------------------------- small candidate set + agreement
WARD_MAX_N = 6000      # Ward is O(n^2): above this it is fitted on a subsample and extended by nearest centroid


def _nearest_centroid(Xfit, labels, Xall, k):
    C = np.array([Xfit[labels == j].mean(0) if (labels == j).any() else Xfit.mean(0) for j in range(k)])
    d = (Xall ** 2).sum(1)[:, None] - 2 * Xall @ C.T + (C ** 2).sum(1)[None, :]
    return d.argmin(1)


def candidate_partitions(Xf, num_cols, k, budget=None, reserve=4.0, seed=SEED, include=None):
    """Cheap, structurally different partitions of the same rows (all forced to k groups):
      kmeans    : spherical, equal-variance groups
      ward      : hierarchical, elongated / unequal-size groups
      gmm_diag  : axis-aligned Gaussians fitted on the CONTINUOUS columns only (0/1 one-hot dims make a
                  Gaussian density model meaningless and dominate its likelihood)
      gmm_full  : full-covariance Gaussians on PCA-whitened data (correlated / rotated groups)
    `include` restricts the set (rule gating). Returns {name: labels}.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    include = include or ("kmeans", "ward", "gmm_diag", "gmm_full")
    n = len(Xf)
    rng = np.random.RandomState(seed)
    sub = np.arange(n) if n <= 4000 else np.sort(rng.choice(n, 4000, replace=False))
    out = {}

    def ok():
        return budget is None or budget.has(reserve)

    if "kmeans" in include:
        out["kmeans"] = KMeans(k, n_init=10 if n <= 20000 else 4, random_state=seed).fit_predict(Xf)
    if "ward" in include and ok():
        if n <= WARD_MAX_N:
            out["ward"] = AgglomerativeClustering(k).fit_predict(Xf)
        else:
            ws = np.sort(rng.choice(n, WARD_MAX_N, replace=False))
            lab = AgglomerativeClustering(k).fit_predict(Xf[ws])
            out["ward"] = _nearest_centroid(Xf[ws], lab, Xf, k)
    if "gmm_diag" in include and num_cols is not None and len(num_cols) >= 1 and ok():
        Z = Xf[:, num_cols]
        out["gmm_diag"] = GaussianMixture(k, covariance_type="diag", n_init=2, reg_covar=1e-4,
                                          random_state=seed).fit(Z[sub]).predict(Z)
    if "gmm_full" in include and Xf.shape[1] >= 2 and ok():
        pca = PCA(n_components=0.95, svd_solver="full", whiten=True, random_state=seed).fit(Xf[sub])
        Pw = pca.transform(Xf)
        out["gmm_full"] = GaussianMixture(k, covariance_type="full", n_init=2, reg_covar=1e-3,
                                          random_state=seed).fit(Pw[sub]).predict(Pw)
    return {name: lab for name, lab in out.items() if len(np.unique(lab)) == k}


def agreement_medoid(cands, prefer="kmeans"):
    """Partition with the highest mean ARI to the other candidates (the one most methods agree with).
    Ties (within 1e-9) go to `prefer`. Returns (name, {name: mean ARI})."""
    from sklearn.metrics import adjusted_rand_score as ARI
    names = list(cands)
    if len(names) < 3:
        return (prefer if prefer in cands else names[0]), {}
    M = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            M[i, j] = M[j, i] = ARI(cands[names[i]], cands[names[j]])
    score = M.sum(1) / (len(names) - 1)
    best = max(range(len(names)), key=lambda i: (score[i], names[i] == prefer))
    return names[best], {nm: round(float(s), 4) for nm, s in zip(names, score)}
