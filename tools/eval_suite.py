"""Developer re-evaluation on the Validator's round-1 suite, WITHOUT touching validation/results.

Reuses (read-only import) the Validator's grader simulation and scoring functions from
validation/run_validation.py, but writes runs to runs/<tag>/ (gitignored).

    py tools/eval_suite.py run   --tag r1fix --approaches A_lean B_portfolio C_rules [--only k_iris ...]
    py tools/eval_suite.py score --tag r1fix [--compare validation/results/results.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "validation"))
import run_validation as RV  # noqa: E402

PER_Q, PER_P, PER_S = 85 / 18, 10 / 18, 5 / 18


def cmd_run(a):
    base = os.path.join(ROOT, "runs", a.tag)
    for ap in a.approaches:
        RV.run_one(ap, os.path.join(RV.SUITE, RV.WARMUP_DS), os.path.join(base, "warmup", ap))
        for ds in RV._datasets(RV.SUITE):
            if a.only and ds not in a.only:
                continue
            m = RV.run_one(ap, os.path.join(RV.SUITE, ds), os.path.join(base, ap, ds))
            print(f"{ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']:8s} tb={m['traceback']} "
                  f"wd={m['watchdog']} first_pred={m.get('t_first_pred')}", flush=True)


def score_tag(tag, approaches=None):
    base = pd.read_csv(os.path.join(RV.RESULTS, "baselines.csv"))
    rows = []
    root = os.path.join(ROOT, "runs", tag)
    approaches = approaches or [d for d in sorted(os.listdir(root)) if d != "warmup"]
    for ds in RV._datasets(RV.SUITE):
        d = os.path.join(RV.SUITE, ds)
        _, truth = RV._load_meta(d)
        task = truth["task"]
        pm = RV.PRIMARY[task]
        b = base[base.dataset == ds].set_index("baseline")
        trivial = float(b.loc["trivial", pm]) if task == "classification" else 0.0
        sk = float(b.loc["sk_kmeans" if task == "clustering" else "sk_hgb", pm])
        sk_norm = float(np.clip((sk - trivial) / (1 - trivial), 0, 1))
        gt = RV.profile_truth(d)
        for ap in approaches:
            ap_dir = os.path.join(root, ap, ds)
            if not os.path.isdir(ap_dir):
                continue
            meta, s, prof, mod = RV._eval_run(ap_dir, d)
            ok = meta.get("status") == "ok"
            m = s.get(pm) if ok else None
            norm = float(np.clip((m - trivial) / (1 - trivial), 0, 1)) if m is not None else 0.0
            useful = ok and (norm >= 0.5 or norm >= 0.9 * sk_norm)
            sf = RV.speed_factor(meta.get("wall_time")) if ok else 0.0
            if prof and "columns" in prof:
                fs, _ = RV.score_profile(prof, gt)
                ps = float(np.mean([fs[f] for f in RV.PROFILE_FIELDS]))
            else:
                ps = 0.0
            rows.append({"approach": ap, "dataset": ds, "task": task, "status": meta.get("status"),
                         "wall_time": meta.get("wall_time"), "t_first_pred": meta.get("t_first_pred"),
                         "watchdog": meta.get("watchdog"), "primary": m, "rmsle": s.get("rmsle"),
                         "quality_pts": norm * PER_Q, "profile_pts": ps * PER_P, "speed_pts": PER_S * sf * useful,
                         "model": str(mod.get("model"))[:80]})
    df = pd.DataFrame(rows)
    df["total_pts"] = df.quality_pts + df.profile_pts + df.speed_pts
    return df


def summarise(df):
    out = {}
    for ap, g in df.groupby("approach"):
        out[ap] = {"quality": round(g.quality_pts.sum(), 2), "profile": round(g.profile_pts.sum(), 2),
                   "speed": round(g.speed_pts.sum(), 2), "total": round(g.total_pts.sum(), 2),
                   **{f"q_{t[:3]}": round(g[g.task == t].quality_pts.sum(), 2) for t in RV.PRIMARY},
                   "wall_median": round(g.wall_time.median(), 2), "wall_max": round(g.wall_time.max(), 2),
                   "n_ok": int((g.status == "ok").sum())}
    return out


def cmd_score(a):
    df = score_tag(a.tag)
    os.makedirs(os.path.join(ROOT, "runs", a.tag), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "runs", a.tag, "scores.csv"), index=False, float_format="%.6g")
    summ = summarise(df)
    with open(os.path.join(ROOT, "runs", a.tag, "summary_points.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    if a.compare:
        old = pd.read_csv(a.compare)
        m = df.merge(old[["approach", "dataset", "primary", "wall_time"]], on=["approach", "dataset"],
                     suffixes=("", "_before"))
        for ap, g in m.groupby("approach"):
            print(f"\n== {ap}")
            for _, r in g.iterrows():
                print(f"{r.dataset:14s} {r.primary_before:8.4f} -> {r.primary:8.4f}   "
                      f"{r.wall_time_before:6.2f}s -> {r.wall_time:6.2f}s   {r.model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--approaches", nargs="*", default=["A_lean", "B_portfolio", "C_rules"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--compare")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    main()
