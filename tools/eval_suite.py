"""Developer re-evaluation on the Validator's suites, WITHOUT touching anything under validation/.

Reuses (read-only import) the Validator's grader simulation and scoring functions from
validation/run_validation.py; round-2 paths/baselines are passed explicitly. Runs go to runs/<tag>/<suite>/.

    py tools/eval_suite.py run   --tag r3fix --suite r1 r2 r3 [--approaches A_lean ...] [--only k_synth ...]
                                 [--threads 1] [--slowdown 2.5]
    py tools/eval_suite.py score --tag r2fix --suite r1 r2 [--before r1=runs/r1final/scores.csv
                                                           r2=validation/results_round2/results.csv]
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
SUITES = {
    "r1": {"data": os.path.join(ROOT, "validation", "data", "suite"), "warmup": "k_seeds",
           "baselines": os.path.join(ROOT, "validation", "results", "baselines.csv")},
    "r2": {"data": os.path.join(ROOT, "validation", "data", "suite_r2"), "warmup": "k_wholesale",
           "baselines": os.path.join(ROOT, "validation", "results_round2", "baselines.csv")},
    "r3": {"data": os.path.join(ROOT, "validation", "data", "suite_r3"), "warmup": "k_glass",
           "baselines": os.path.join(ROOT, "validation", "results_round3", "baselines.csv")},
}


def _root(tag, suite):
    # backwards compatible: round-1-only tags written before this tool knew about suites
    legacy = os.path.join(ROOT, "runs", tag)
    new = os.path.join(legacy, suite)
    if suite == "r1" and not os.path.isdir(new) and os.path.isdir(os.path.join(legacy, "A_lean")):
        return legacy
    return new


def cmd_run(a):
    RV.THREADS = str(a.threads)
    limit = RV.LIMIT
    if a.slowdown and a.slowdown != 1:
        os.environ["HACK_SLOWDOWN"] = str(a.slowdown)       # inherited by the subprocess (main.py M3 flag)
        limit = RV.LIMIT / a.slowdown                        # the grader's 60 s kill scales the same way
    for suite in a.suite:
        S = SUITES[suite]
        base = os.path.join(ROOT, "runs", a.tag, suite)
        for ap in a.approaches:
            RV.run_one(ap, os.path.join(S["data"], S["warmup"]), os.path.join(base, "warmup", ap), limit)
            for ds in RV._datasets(S["data"]):
                if a.only and ds not in a.only:
                    continue
                m = RV.run_one(ap, os.path.join(S["data"], ds), os.path.join(base, ap, ds), limit)
                if a.slowdown and a.slowdown != 1:      # record the scale: speed points use wall time x scale
                    mp = os.path.join(base, ap, ds, "run_meta.json")
                    with open(mp, encoding="utf-8") as f:
                        mm = json.load(f)
                    mm["time_scale"] = a.slowdown
                    with open(mp, "w", encoding="utf-8") as f:
                        json.dump(mm, f, indent=1)
                print(f"{suite} {ap:12s} {ds:16s} {m['wall_time']:6.2f}s {m['status']:8s} tb={m['traceback']} "
                      f"wd={m['watchdog']}", flush=True)


def score_suite(tag, suite, approaches=None):
    S = SUITES[suite]
    base = pd.read_csv(S["baselines"])
    rows = []
    root = _root(tag, suite)
    if not os.path.isdir(root):
        return pd.DataFrame()
    approaches = approaches or [d for d in sorted(os.listdir(root)) if d in RV.APPROACHES]
    for ds in RV._datasets(S["data"]):
        d = os.path.join(S["data"], ds)
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
            wt = meta.get("wall_time")
            scale = float(meta.get("time_scale") or 1.0)
            sf = RV.speed_factor(wt * scale if wt is not None else None) if ok else 0.0
            if prof and "columns" in prof:
                fs, _ = RV.score_profile(prof, gt)
                ps = float(np.mean([fs[f] for f in RV.PROFILE_FIELDS]))
            else:
                ps = 0.0
            rows.append({"suite": suite, "approach": ap, "dataset": ds, "task": task, "status": meta.get("status"),
                         "wall_time": meta.get("wall_time"), "watchdog": meta.get("watchdog"), "primary": m,
                         "rmsle": s.get("rmsle"), "balanced_accuracy": s.get("balanced_accuracy"),
                         "quality_pts": norm * PER_Q, "profile_pts": ps * PER_P, "speed_pts": PER_S * sf * useful,
                         "model": str(mod.get("model"))[:80]})
    df = pd.DataFrame(rows)
    if len(df):
        df["total_pts"] = df.quality_pts + df.profile_pts + df.speed_pts
    return df


def summarise(df):
    out = {}
    for ap, g in df.groupby("approach"):
        out[ap] = {"quality": round(g.quality_pts.sum(), 2), "profile": round(g.profile_pts.sum(), 2),
                   "speed": round(g.speed_pts.sum(), 2), "total": round(g.total_pts.sum(), 2),
                   **{f"q_{t[:3]}": round(g[g.task == t].quality_pts.sum(), 2) for t in RV.PRIMARY},
                   "wall_median": round(g.wall_time.median(), 2), "wall_max": round(g.wall_time.max(), 2),
                   "n_ok": int((g.status == "ok").sum()), "n": int(len(g))}
    return out


def cmd_score(a):
    frames = [score_suite(a.tag, s) for s in a.suite]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True)
    out_dir = os.path.join(ROOT, "runs", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "scores_all.csv"), index=False, float_format="%.6g")
    summ = {s: summarise(df[df.suite == s]) for s in df.suite.unique()}
    if len(summ) > 1:
        summ["combined"] = summarise(df)
    with open(os.path.join(out_dir, "summary_points.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    for spec in a.before or []:
        suite, path = spec.split("=", 1)
        old = pd.read_csv(path)
        m = df[df.suite == suite].merge(old[["approach", "dataset", "primary", "wall_time"]],
                                        on=["approach", "dataset"], suffixes=("", "_before"))
        for ap, g in m.groupby("approach"):
            print(f"\n== {suite} {ap}")
            for _, r in g.iterrows():
                flag = "  <<<" if (r.primary - r.primary_before) < -0.05 else ""
                print(f"{r.dataset:16s} {r.primary_before:8.4f} -> {r.primary:8.4f}  "
                      f"{r.wall_time_before:6.2f}s -> {r.wall_time:6.2f}s  {r.model}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suite", nargs="*", default=["r1", "r2", "r3"], choices=["r1", "r2", "r3"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--slowdown", type=float, default=None, help="simulate an s-times slower CPU (HACK_SLOWDOWN)")
    ap.add_argument("--approaches", nargs="*", default=["A_lean", "B_portfolio", "C_rules"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--before", nargs="*", help="suite=path/to/results.csv for a before/after listing")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    main()
