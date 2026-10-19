"""Validation harness: run the approaches like the hackathon assessment agent and score them.

    py validation/run_validation.py run          # warm-up + 18 datasets x 3 approaches (sequential, timed)
    py validation/run_validation.py baselines    # trivial + plain sklearn baselines (in-process)
    py validation/run_validation.py determinism  # rerun 3 datasets per approach, compare outputs
    py validation/run_validation.py stress       # 9 stress probes x 3 approaches (scored separately)
    py validation/run_validation.py score        # -> validation/results/*.csv + summary.json
    py validation/run_validation.py all          # everything above, in that order

Grader simulation: each run is `py approaches/<X>/main.py --data <dir> --out <out>` as a subprocess with
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2 (weaker grader CPU), wall time measured from process
creation to exit (interpreter start-up and imports included), hard kill of the whole process tree at 60 s
(status=timeout -> 0 points). Runs are strictly sequential. Every approach is run once on a warm-up
dataset first so cold .pyc compilation / first DLL loads do not distort the timings.

Scoring model: see SCORING below and docs/04_validation_report.md.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from score import score as score_output  # noqa: E402

SUITE = os.path.join(HERE, "data", "suite")
STRESS = os.path.join(HERE, "data", "stress")
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
APPROACHES = ["A_lean", "B_portfolio", "C_rules"]
LIMIT = 60.0
THREADS = "2"
WARMUP_DS = "k_seeds"
DET_DS = ["c_creditg", "c_bank", "k_drybean"]

N_DATASETS = 18
Q_TOTAL, P_TOTAL, S_TOTAL = 85.0, 10.0, 5.0
SCORING = {
    "quality": "per dataset 85/18 pts x norm, norm = clip((m - trivial) / (1 - trivial), 0, 1); m = macro-F1 / R2 / ARI; "
               "trivial = macro-F1 of the majority class on the test set / 0 / 0; crash, timeout or invalid output = 0",
    "quality_alt": "same but norm_rel = clip((m - trivial) / (best - trivial), 0, 1), best = best of A/B/C and the two sklearn baselines",
    "profile": "per dataset 10/18 pts x mean of 14 field scores in [0,1] (see profile_fields)",
    "speed": "per dataset 5/18 pts x f(t) x useful; f(t) = 1 for t <= 5 s, 1 - ln(t/5)/ln(12) for 5 < t < 60, 0 beyond; "
             "useful = valid and (norm >= 0.5 or norm >= 0.9 x norm of the sklearn HGB/KMeans baseline)",
}


# =========================================================================== running
def _env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = THREADS
    env.pop("HACK_N_JOBS", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _kill_tree(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def run_one(approach, data_dir, out_dir, limit=LIMIT):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    cmd = ["py", os.path.join("approaches", approach, "main.py"), "--data", data_dir, "--out", out_dir]
    log_p = os.path.join(out_dir, "stdout.txt")
    with open(log_p, "w", encoding="utf-8") as logf:
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=ROOT, env=_env(), stdout=logf, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        status = "ok"
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            _kill_tree(p.pid)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            rc, status = None, "timeout"
        wall = time.perf_counter() - t0
    with open(log_p, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if status == "ok" and rc != 0:
        status = "crash"
    meta = {"approach": approach, "dataset": os.path.basename(data_dir), "status": status, "returncode": rc,
            "wall_time": round(wall, 3), "traceback": "Traceback" in txt, "watchdog": "[watchdog]" in txt,
            "cmd": " ".join(cmd)}
    meta.update(_parse_log(txt))
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


_TS = re.compile(r"^\[\s*([\d.]+)s\]\s*(.*)$")


def _parse_log(txt):
    """Internal timestamps written by main.py: first predictions on disk, profile written, done."""
    out = {"t_first_pred": None, "t_profile": None, "t_done": None, "n_emits": 0}
    for line in txt.splitlines():
        m = _TS.match(line.strip())
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        if msg.startswith("predictions written"):
            out["n_emits"] += 1
            if out["t_first_pred"] is None:
                out["t_first_pred"] = t
        elif msg.startswith("profile written"):
            out["t_profile"] = t
        elif msg.startswith("done in"):
            out["t_done"] = t
        elif msg.startswith("loaded"):
            out["t_loaded"] = t
    return out


def _datasets(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def cmd_run(args):
    print("warm-up runs (not scored) on", WARMUP_DS, flush=True)
    for ap in APPROACHES:
        m = run_one(ap, os.path.join(SUITE, WARMUP_DS), os.path.join(RUNS, "warmup", ap))
        print(f"  warm-up {ap:12s} {m['wall_time']:6.2f}s {m['status']}", flush=True)
    for ap in APPROACHES:
        for ds in _datasets(SUITE):
            if args.only and ds not in args.only:
                continue
            m = run_one(ap, os.path.join(SUITE, ds), os.path.join(RUNS, "suite", ap, ds))
            print(f"{ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']:8s} tb={m['traceback']} wd={m['watchdog']}",
                  flush=True)


def cmd_determinism(args):
    for ap in APPROACHES:
        for ds in DET_DS:
            m = run_one(ap, os.path.join(SUITE, ds), os.path.join(RUNS, "det", ap, ds))
            print(f"det {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']}", flush=True)


def cmd_stress(args):
    for ap in APPROACHES:
        for ds in _datasets(STRESS):
            m = run_one(ap, os.path.join(STRESS, ds), os.path.join(RUNS, "stress", ap, ds))
            print(f"stress {ap:12s} {ds:20s} {m['wall_time']:6.2f}s {m['status']} tb={m['traceback']}", flush=True)


# =========================================================================== baselines
def _load_meta(d):
    with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "hidden", "truth.json"), encoding="utf-8") as f:
        truth = json.load(f)
    return meta, truth


def _baseline_frames(d, meta):
    """Plain feature handling: declared numeric -> to_numeric, declared datetime -> 5 calendar parts,
    everything else categorical strings. id_column dropped. (No ID / constant / duplicate cleaning.)"""
    tr_p = os.path.join(d, "train.csv") if os.path.isfile(os.path.join(d, "train.csv")) else os.path.join(d, "data.csv")
    tr = pd.read_csv(tr_p, low_memory=False)
    te = pd.read_csv(os.path.join(d, "test.csv"), low_memory=False) if os.path.isfile(os.path.join(d, "test.csv")) else None
    tgt, idc = meta.get("target"), meta.get("id_column")
    ft = meta.get("feature_types") or {}
    y = tr[tgt] if (tgt and meta["task"] != "clustering") else None
    full = pd.concat([tr, te], ignore_index=True) if te is not None else tr.copy()
    full = full.drop(columns=[c for c in (tgt, idc) if c and c in full.columns])
    num, cat = {}, {}
    for c in full.columns:
        t = ft.get(c) or ("numeric" if pd.api.types.is_numeric_dtype(full[c]) else "categorical")
        if t == "numeric":
            num[c] = pd.to_numeric(full[c], errors="coerce")
        elif t == "datetime":
            dt = pd.to_datetime(full[c], errors="coerce")
            for part in ("year", "month", "day", "hour", "dayofweek"):
                num[f"{c}_{part}"] = getattr(dt.dt, part).astype(float)
        else:
            cat[c] = full[c].astype(str)
    X = pd.concat([pd.DataFrame(num), pd.DataFrame(cat)], axis=1)
    ntr = len(tr)
    return X.iloc[:ntr], (X.iloc[ntr:] if te is not None else None), y, list(num), list(cat)


def _pre(num, cat, scale):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    nsteps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(nsteps), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False), cat)])


def _write_pred(out_dir, pred, meta, d):
    os.makedirs(out_dir, exist_ok=True)
    gt = pd.read_csv(os.path.join(d, "hidden", "test_labels.csv"))
    pd.DataFrame({"row_index": gt["row_index"], "prediction": pred}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


def cmd_baselines(args):
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    rows = []
    with threadpool_limits(2):
        for ds in _datasets(SUITE):
            d = os.path.join(SUITE, ds)
            meta, truth = _load_meta(d)
            task = meta["task"]
            Xtr, Xte, y, num, cat = _baseline_frames(d, meta)
            preds = {}
            t0 = time.perf_counter()
            if task == "classification":
                maj = y.value_counts().index[0]
                preds["trivial"] = np.repeat(maj, len(Xte))
                preds["sk_hgb"] = Pipeline([("pre", _pre(num, cat, False)),
                                            ("m", HistGradientBoostingClassifier(random_state=0))]).fit(Xtr, y).predict(Xte)
                preds["sk_linear"] = Pipeline([("pre", _pre(num, cat, True)),
                                               ("m", LogisticRegression(max_iter=3000))]).fit(Xtr, y).predict(Xte)
            elif task == "regression":
                preds["trivial"] = np.repeat(float(y.mean()), len(Xte))
                preds["sk_hgb"] = Pipeline([("pre", _pre(num, cat, False)),
                                            ("m", HistGradientBoostingRegressor(random_state=0))]).fit(Xtr, y).predict(Xte)
                preds["sk_linear"] = Pipeline([("pre", _pre(num, cat, True)),
                                               ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))]).fit(Xtr, y).predict(Xte)
            else:
                k = int(meta["n_clusters"])
                Z = Pipeline([("pre", _pre(num, cat, True)), ("sc", StandardScaler())]).fit_transform(Xtr)
                preds["trivial"] = np.zeros(len(Xtr), int)
                preds["sk_kmeans"] = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
                preds["sk_kmeans_raw"] = KMeans(k, n_init=10, random_state=0).fit_predict(
                    Pipeline([("pre", _pre(num, cat, False))]).fit_transform(Xtr))
            for name, p in preds.items():
                out = os.path.join(RUNS, "baselines", name, ds)
                _write_pred(out, p, meta, d)
                s = score_output(d, out)
                s.update(baseline=name, dataset=ds, task=task)
                rows.append(s)
            print(ds, {r["baseline"]: round(r.get("macro_f1", r.get("r2", r.get("ari", np.nan))), 4)
                       for r in rows if r["dataset"] == ds}, f"{time.perf_counter() - t0:.1f}s", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    cols = ["dataset", "task", "baseline", "valid", "macro_f1", "balanced_accuracy", "accuracy", "r2", "rmse", "rmsle",
            "ari", "nmi"]
    df = pd.DataFrame(rows)
    df[[c for c in cols if c in df.columns]].to_csv(os.path.join(RESULTS, "baselines.csv"), index=False)


# =========================================================================== profile scoring
def _f1_sets(pred, true):
    pred, true = set(pred), set(true)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 2 * p * r / (p + r)


def _raw_train(d):
    p = os.path.join(d, "train.csv") if os.path.isfile(os.path.join(d, "train.csv")) else os.path.join(d, "data.csv")
    return pd.read_csv(p, low_memory=False)


def profile_truth(d, base_types=None):
    meta, truth = _load_meta(d)
    df = _raw_train(d)
    task = truth["task"]
    tgt = truth["target"] if task != "clustering" else None
    feats = [c for c in df.columns if c != tgt]
    const = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    nonconst = [c for c in feats if c not in const]
    dup_pairs = set()
    for i, a in enumerate(nonconst):
        for b in nonconst[i + 1:]:
            if df[a].dtype == df[b].dtype and df[a].equals(df[b]):
                dup_pairs.add(frozenset((a, b)))
    declared = meta.get("feature_types") or base_types or {}
    exp_type = {}
    for c in feats:
        if c in truth["id_columns"]:
            exp_type[c] = {"id"}
        elif c in const:
            exp_type[c] = {"constant"}
        elif c in declared:
            t = declared[c]
            exp_type[c] = {"categorical", "boolean"} if t == "categorical" else {t}
            if t == "numeric" and df[c].dropna().nunique() == 2:
                exp_type[c] |= {"boolean"}
    gt = {"task": task, "target": tgt, "n_rows": len(df), "n_columns": df.shape[1],
          "n_duplicate_rows": int(df.duplicated().sum()), "total_missing_all": int(df.isna().sum().sum()),
          "total_missing_feat": int(df[feats].isna().sum().sum()),
          "col_missing": {c: int(df[c].isna().sum()) for c in df.columns},
          "col_unique": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
          "exp_type": exp_type, "constant": const, "dup_pairs": dup_pairs,
          "id_columns": truth["id_columns"], "id_column": truth.get("id_column"),
          "datetime": truth["datetime_columns"], "k": truth.get("n_clusters")}
    if task == "classification":
        vc = df[tgt].value_counts()
        gt["n_classes"] = int(len(vc))
        gt["class_counts"] = {str(k): int(v) for k, v in vc.items()}
    elif task == "regression":
        v = pd.to_numeric(df[tgt], errors="coerce")
        gt["y_stats"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return gt


PROFILE_FIELDS = ["task", "target", "n_rows", "n_columns", "n_duplicate_rows", "total_missing_cells", "col_missing",
                  "col_unique", "col_type", "constant_columns", "duplicate_columns", "id_like_columns",
                  "target_profile", "datetime_columns"]


def score_profile(prof, gt):
    s = {}
    s["task"] = float(prof.get("task") == gt["task"])
    s["target"] = float((prof.get("target") or None) == gt["target"])
    s["n_rows"] = float(prof.get("n_rows") == gt["n_rows"])
    s["n_columns"] = float(prof.get("n_columns") == gt["n_columns"])
    s["n_duplicate_rows"] = float(prof.get("n_duplicate_rows") == gt["n_duplicate_rows"])
    s["total_missing_cells"] = float(prof.get("total_missing_cells") in (gt["total_missing_all"], gt["total_missing_feat"]))
    cols = {c.get("name"): c for c in (prof.get("columns") or []) if isinstance(c, dict)}
    s["col_missing"] = float(np.mean([cols.get(c, {}).get("n_missing") == v for c, v in gt["col_missing"].items()]))
    s["col_unique"] = float(np.mean([cols.get(c, {}).get("n_unique") == v for c, v in gt["col_unique"].items()]))
    ok = []
    errs = []
    for c, exp in gt["exp_type"].items():
        rec = cols.get(c, {})
        good = rec.get("type") in exp or (c == gt["id_column"] and rec.get("role") == "id")
        ok.append(good)
        if not good:
            errs.append(f"{c}:{rec.get('type')}!={'/'.join(sorted(exp))}")
    s["col_type"] = float(np.mean(ok)) if ok else 1.0
    s["constant_columns"] = _f1_sets(prof.get("constant_columns") or [], gt["constant"])
    cset = set(gt["constant"])
    pred_pairs = {frozenset(p) for p in (prof.get("duplicate_columns") or []) if isinstance(p, (list, tuple))
                  and len(p) == 2 and not (set(p) & cset)}
    s["duplicate_columns"] = _f1_sets(pred_pairs, gt["dup_pairs"])
    s["id_like_columns"] = _f1_sets(prof.get("id_like_columns") or [], gt["id_columns"])
    tp = prof.get("target_profile") or {}
    if gt["task"] == "classification":
        s["target_profile"] = 0.5 * float(tp.get("n_classes") == gt["n_classes"]) + \
            0.5 * float({str(k): v for k, v in (tp.get("class_counts") or {}).items()} == gt["class_counts"])
    elif gt["task"] == "regression":
        vals = []
        for k, v in gt["y_stats"].items():
            pv = tp.get(k)
            vals.append(pv is not None and abs(pv - v) <= 1e-6 * max(1.0, abs(v)))
        s["target_profile"] = float(np.mean(vals))
    else:
        s["target_profile"] = float(tp.get("n_clusters_required") == gt["k"])
    s["datetime_columns"] = _f1_sets(prof.get("datetime_columns") or [], gt["datetime"])
    return s, errs


# =========================================================================== scoring
def speed_factor(t):
    if t is None or not np.isfinite(t) or t >= LIMIT:
        return 0.0
    if t <= 5.0:
        return 1.0
    return float(np.clip(1 - math.log(t / 5.0) / math.log(LIMIT / 5.0), 0, 1))


PRIMARY = {"classification": "macro_f1", "regression": "r2", "clustering": "ari"}
METRIC_COLS = ["macro_f1", "balanced_accuracy", "accuracy", "r2", "rmse", "rmsle", "ari", "nmi"]


def _read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _eval_run(ap_dir, d):
    meta = _read_json(os.path.join(ap_dir, "run_meta.json")) or {"status": "missing"}
    s = score_output(d, ap_dir) if meta.get("status") == "ok" else {"valid": False, "error": meta.get("status")}
    if meta.get("status") == "ok" and not s.get("valid"):
        meta["status"] = "invalid"
    if s.get("task") == "clustering" and s.get("valid") and not s.get("k_ok"):
        meta["status"] = "invalid_k"
    prof = _read_json(os.path.join(ap_dir, "profile.json")) or {}
    mod = prof.get("modelling") or {}
    return meta, s, prof, mod


def cmd_score(args):
    os.makedirs(RESULTS, exist_ok=True)
    base = pd.read_csv(os.path.join(RESULTS, "baselines.csv"))
    rows, prow = [], []
    per_q = Q_TOTAL / N_DATASETS
    per_p = P_TOTAL / N_DATASETS
    per_s = S_TOTAL / N_DATASETS
    for ds in _datasets(SUITE):
        d = os.path.join(SUITE, ds)
        meta_ds, truth = _load_meta(d)
        task = truth["task"]
        pm = PRIMARY[task]
        b = base[base.dataset == ds].set_index("baseline")
        trivial = float(b.loc["trivial", pm]) if task == "classification" else 0.0
        sk_name = "sk_kmeans" if task == "clustering" else "sk_hgb"
        sk = float(b.loc[sk_name, pm])
        sk_norm = float(np.clip((sk - trivial) / (1 - trivial), 0, 1))
        gt = profile_truth(d)
        cand = {}
        for ap in APPROACHES:
            ap_dir = os.path.join(RUNS, "suite", ap, ds)
            meta, s, prof, mod = _eval_run(ap_dir, d)
            ok = meta.get("status") == "ok"
            m = s.get(pm) if ok else None
            norm = float(np.clip((m - trivial) / (1 - trivial), 0, 1)) if m is not None else 0.0
            useful = ok and (norm >= 0.5 or norm >= 0.9 * sk_norm)
            sf = speed_factor(meta.get("wall_time")) if ok else 0.0
            r = {"approach": ap, "dataset": ds, "task": task, "status": meta.get("status"),
                 "wall_time": meta.get("wall_time"), "t_first_pred": meta.get("t_first_pred"),
                 "t_profile": meta.get("t_profile"), "watchdog": meta.get("watchdog"),
                 "traceback": meta.get("traceback"), "model": str(mod.get("model"))[:90],
                 "final_stage": mod.get("final_stage"), "primary_metric": pm, "primary": m,
                 **{k: s.get(k) for k in METRIC_COLS if k in s}, "trivial": trivial, "sk_baseline": sk,
                 "sk_linear_or_raw": float(b.loc["sk_kmeans_raw" if task == "clustering" else "sk_linear", pm]),
                 "norm": norm, "quality_pts": norm * per_q, "useful": useful, "speed_factor": sf,
                 "speed_pts": per_s * sf * useful}
            if prof and "columns" in prof:
                fs, errs = score_profile(prof, gt)
                pscore = float(np.mean([fs[f] for f in PROFILE_FIELDS]))
            else:
                fs, errs, pscore = {f: 0.0 for f in PROFILE_FIELDS}, ["profile missing"], 0.0
            r["profile_score"] = pscore
            r["profile_pts"] = pscore * per_p
            r["total_pts"] = r["quality_pts"] + r["profile_pts"] + r["speed_pts"]
            rows.append(r)
            prow.append({"approach": ap, "dataset": ds, "task": task, **fs, "profile_score": pscore,
                         "profile_pts": pscore * per_p, "type_errors": "; ".join(errs)[:400]})
            cand[ap] = m
        best = max([v for v in list(cand.values()) + [sk, float(b.loc["sk_kmeans_raw" if task == "clustering" else "sk_linear", pm])]
                    if v is not None])
        for r in rows[-len(APPROACHES):]:
            m = r["primary"]
            r["best_ref"] = best
            r["norm_rel"] = float(np.clip((m - trivial) / (best - trivial), 0, 1)) if (m is not None and best > trivial) else 0.0
            r["quality_pts_rel"] = r["norm_rel"] * per_q
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "results.csv"), index=False, float_format="%.6g")
    pd.DataFrame(prow).to_csv(os.path.join(RESULTS, "profile_scores.csv"), index=False, float_format="%.6g")

    # baseline points (for context): quality only
    brow = []
    for ds in _datasets(SUITE):
        task = res[res.dataset == ds].task.iloc[0]
        pm = PRIMARY[task]
        b = base[base.dataset == ds].set_index("baseline")
        trivial = float(b.loc["trivial", pm]) if task == "classification" else 0.0
        for name in b.index:
            m = float(b.loc[name, pm])
            brow.append({"baseline": name, "dataset": ds, "task": task,
                         "quality_pts": float(np.clip((m - trivial) / (1 - trivial), 0, 1)) * per_q})
    bq = pd.DataFrame(brow)

    summ = {"scoring": SCORING, "approaches": {}}
    for ap in APPROACHES:
        a = res[res.approach == ap]
        summ["approaches"][ap] = {
            "quality": round(a.quality_pts.sum(), 3), "quality_rel": round(a.quality_pts_rel.sum(), 3),
            "profile": round(a.profile_pts.sum(), 3), "speed": round(a.speed_pts.sum(), 3),
            "total": round(a.total_pts.sum(), 3),
            "per_task_quality": {t: round(a[a.task == t].quality_pts.sum(), 3) for t in PRIMARY},
            "per_task_mean_primary": {t: round(a[a.task == t].primary.astype(float).mean(), 4) for t in PRIMARY},
            "n_ok": int((a.status == "ok").sum()), "statuses": a.status.value_counts().to_dict(),
            "wall_median": round(a.wall_time.median(), 2), "wall_max": round(a.wall_time.max(), 2),
            "wall_p90": round(a.wall_time.quantile(0.9), 2)}
    summ["baselines_quality"] = {n: {"total": round(g.quality_pts.sum(), 3),
                                     **{t: round(g[g.task == t].quality_pts.sum(), 3) for t in PRIMARY}}
                                 for n, g in bq.groupby("baseline")}
    summ["determinism"] = _determinism()
    summ["stress"] = _stress()
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in summ["approaches"].items()}, indent=1, default=str))
    print(json.dumps(summ["baselines_quality"], indent=1))


def _determinism():
    rows = []
    for ap in APPROACHES:
        for ds in DET_DS:
            a = os.path.join(RUNS, "suite", ap, ds, "predictions.csv")
            b = os.path.join(RUNS, "det", ap, ds, "predictions.csv")
            if not (os.path.isfile(a) and os.path.isfile(b)):
                continue
            pa, pb = pd.read_csv(a), pd.read_csv(b)
            same = pa.equals(pb)
            r = {"approach": ap, "dataset": ds, "identical": bool(same),
                 "frac_rows_equal": float((pa["prediction"].astype(str) == pb["prediction"].astype(str)).mean())
                 if len(pa) == len(pb) else 0.0}
            ma = _read_json(os.path.join(RUNS, "suite", ap, ds, "profile.json")) or {}
            mb = _read_json(os.path.join(RUNS, "det", ap, ds, "profile.json")) or {}
            r["model_run1"] = str((ma.get("modelling") or {}).get("model"))[:80]
            r["model_run2"] = str((mb.get("modelling") or {}).get("model"))[:80]
            if ds.startswith("k_"):
                from sklearn.metrics import adjusted_rand_score
                r["ari_between_runs"] = float(adjusted_rand_score(pa["prediction"], pb["prediction"]))
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "determinism.csv"), index=False)
    return rows


def _stress():
    rows = []
    if not os.path.isdir(os.path.join(RUNS, "stress")):
        return rows
    for ds in _datasets(STRESS):
        d = os.path.join(STRESS, ds)
        base_name = {"telco": "c_telco", "house": "r_house", "penguins": "k_penguins"}[ds.split("_")[1]]
        base_meta, _ = _load_meta(os.path.join(SUITE, base_name))
        _, truth = _load_meta(d)
        gt = profile_truth(d, base_types=base_meta.get("feature_types"))
        for ap in APPROACHES:
            ap_dir = os.path.join(RUNS, "stress", ap, ds)
            meta, s, prof, mod = _eval_run(ap_dir, d)
            pm = PRIMARY[truth["task"]]
            base_row = os.path.join(RUNS, "suite", ap, base_name)
            bs = score_output(os.path.join(SUITE, base_name), base_row)
            r = {"approach": ap, "probe": ds, "status": meta.get("status"), "wall_time": meta.get("wall_time"),
                 "traceback": meta.get("traceback"), "primary_metric": pm, "primary": s.get(pm),
                 "base_primary": bs.get(pm), "model": str(mod.get("model"))[:80]}
            if prof and "columns" in prof:
                fs, errs = score_profile(prof, gt)
                r["profile_score"] = float(np.mean([fs[f] for f in PROFILE_FIELDS]))
                r["profile_col_type"] = fs["col_type"]
                r["type_errors"] = "; ".join(errs)[:300]
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "stress_results.csv"), index=False, float_format="%.6g")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "baselines", "determinism", "stress", "score", "all"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd in ("run", "all"):
        cmd_run(a)
    if a.cmd in ("determinism", "all"):
        cmd_determinism(a)
    if a.cmd in ("stress", "all"):
        cmd_stress(a)
    if a.cmd in ("baselines", "all"):
        cmd_baselines(a)
    if a.cmd in ("score", "all"):
        cmd_score(a)


if __name__ == "__main__":
    main()
