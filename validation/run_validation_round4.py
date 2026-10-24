"""Round-4 (final) validation harness. Same grader simulation and scoring model as rounds 1-3: it reuses
run_validation.py (unchanged) and run_validation_round3.py's helpers, with paths redirected.

Layout: validation/runs_r4/<tag>/suite/<approach>/<dataset>/ (gitignored); results -> validation/results_round4/

    py validation/run_validation_round4.py run --tag head                       # HEAD on suite r4, 2 threads
    py validation/run_validation_round4.py run --tag v_98f5fe1 --code-root <wt> # round-2-fix code
    py validation/run_validation_round4.py onethread                            # HEAD, 1 thread
    py validation/run_validation_round4.py slowdown --only <heavy datasets>     # HEAD, HACK_SLOWDOWN=2.5, kill 24 s
    py validation/run_validation_round4.py determinism | stress
    py validation/run_validation_round4.py recheck --suite r1 [--only ...]      # HEAD on older suites
    py validation/run_validation_round4.py baselines
    py validation/run_validation_round4.py score
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_validation as RV  # noqa: E402
import run_validation_round3 as R3  # noqa: E402

R3.SUITES["r4"] = os.path.join(HERE, "data", "suite_r4")
R3.RUNS3 = os.path.join(HERE, "runs_r4")
R3.RES3 = os.path.join(HERE, "results_round4")
R3.BASELINES["r3"] = os.path.join(HERE, "results_round3", "baselines.csv")
R3.BASELINES["r4"] = os.path.join(R3.RES3, "baselines.csv")
R3.WARMUP["r4"] = "k_userknow"
STRESS4 = os.path.join(HERE, "data", "stress_r4")
RUNS4, RES4 = R3.RUNS3, R3.RES3
APPROACHES = RV.APPROACHES
DET4 = ["c_bankchurn", "k_satimage"]
HEAVY = ["c_miniboone", "c_texture", "c_yeast", "r_medical", "r_house16h", "r_elevators", "k_satimage", "k_mfeat"]
SPEC_EXAMPLES = ["c_telco", "c_ai4i", "r_house", "r_bike", "k_mall"]


def run_loop(tag, suite_dir, warm, code_root=None, threads="2", only=None, slowdown=None):
    saved = (RV.ROOT, RV.THREADS, os.environ.get("HACK_SLOWDOWN"))
    RV.ROOT, RV.THREADS = (code_root or ROOT), threads
    limit = RV.LIMIT
    if slowdown:
        os.environ["HACK_SLOWDOWN"] = str(slowdown)
        limit = RV.LIMIT / float(slowdown)
    try:
        base = os.path.join(RUNS4, tag)
        for ap in APPROACHES:
            RV.run_one(ap, os.path.join(suite_dir, warm), os.path.join(base, "warmup", ap), limit=limit)
        for ap in APPROACHES:
            for ds in RV._datasets(suite_dir):
                if only and ds not in only:
                    continue
                m = RV.run_one(ap, os.path.join(suite_dir, ds), os.path.join(base, "suite", ap, ds), limit=limit)
                print(f"{tag:12s} {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']:8s} wd={m['watchdog']} tb={m['traceback']}",
                      flush=True)
    finally:
        RV.ROOT, RV.THREADS = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("HACK_SLOWDOWN", None)
        else:
            os.environ["HACK_SLOWDOWN"] = saved[2]


def _patch_baselines_for_variant2():
    """run_validation's baselines assume clustering variant 1. For variant 2 (train/test files) cluster the union
    of train + test rows and score the test rows only (same semantics as the solvers)."""
    orig_frames, orig_write = RV._baseline_frames, RV._write_pred

    def frames(d, meta):
        Xtr, Xte, y, num, cat = orig_frames(d, meta)
        if meta["task"] == "clustering" and Xte is not None:
            return pd.concat([Xtr, Xte], ignore_index=True), None, y, num, cat
        return Xtr, Xte, y, num, cat

    def write(out_dir, pred, meta, d):
        gt = pd.read_csv(os.path.join(d, "hidden", "test_labels.csv"))
        pred = np.asarray(pred)
        if len(pred) != len(gt):
            pred = pred[-len(gt):]
        return orig_write(out_dir, pred, meta, d)

    RV._baseline_frames, RV._write_pred = frames, write


def _brief(s):
    return {ap: {k: v[k] for k in ("quality", "quality_rel", "profile", "speed", "total", "per_task_quality",
                                   "wall_median", "wall_max", "statuses")} for ap, v in s["approaches"].items()}


def cmd_score(a):
    os.makedirs(RES4, exist_ok=True)
    out = {"scoring": RV.SCORING}
    vrows = []
    for tag, label in (("head", "HEAD 43699f2"), ("v_98f5fe1", "round-2 fix 98f5fe1")):
        if not os.path.isdir(os.path.join(RUNS4, tag)):
            continue
        res, prof, summ = R3.score_tag(tag, "r4")
        if tag == "head":
            res.to_csv(os.path.join(RES4, "results.csv"), index=False, float_format="%.6g")
            prof.to_csv(os.path.join(RES4, "profile_scores.csv"), index=False, float_format="%.6g")
            out["baselines_quality_round4"] = summ["baselines_quality"]
        r_ = res.copy()
        r_.insert(0, "version", label)
        vrows.append(r_)
        out.setdefault("versions_round4", {})[label] = _brief(summ)
    if vrows:
        pd.concat(vrows).to_csv(os.path.join(RES4, "version_comparison.csv"), index=False, float_format="%.6g")
    # 1-thread and slowdown passes
    for tag, fname in (("onethread", "onethread_timing.csv"), ("slowdown25", "slowdown_timing.csv")):
        if not os.path.isdir(os.path.join(RUNS4, tag)) or not os.path.isfile(os.path.join(RES4, "results.csv")):
            continue
        r2t = pd.read_csv(os.path.join(RES4, "results.csv"))
        rows = []
        for ap in APPROACHES:
            for ds in RV._datasets(R3.SUITES["r4"]):
                d = os.path.join(RUNS4, tag, "suite", ap, ds)
                if not os.path.isdir(d):
                    continue
                meta, s, prof, mod = RV._eval_run(d, os.path.join(R3.SUITES["r4"], ds))
                pm = RV.PRIMARY[s.get("task") or RV._load_meta(os.path.join(R3.SUITES["r4"], ds))[1]["task"]]
                base = r2t[(r2t.approach == ap) & (r2t.dataset == ds)].iloc[0]
                rows.append({"approach": ap, "dataset": ds, "status": meta.get("status"), "watchdog": meta.get("watchdog"),
                             "wall_time": meta.get("wall_time"), "wall_time_2thr": base.wall_time,
                             "primary": s.get(pm), "primary_2thr": base.primary})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(RES4, fname), index=False, float_format="%.6g")
        out[tag] = {ap: {"median": round(g.wall_time.median(), 2), "max": round(g.wall_time.max(), 2),
                         "max_dataset": g.loc[g.wall_time.idxmax(), "dataset"],
                         "n_invalid": int((g.status != "ok").sum()), "n_watchdog": int(g.watchdog.fillna(False).sum()),
                         "n_over_40s": int((g.wall_time > 40).sum())} for ap, g in t.groupby("approach")}
    # determinism
    rows = []
    for ap in APPROACHES:
        for ds in DET4:
            a_ = os.path.join(RUNS4, "head", "suite", ap, ds, "predictions.csv")
            b_ = os.path.join(RUNS4, "det", "suite", ap, ds, "predictions.csv")
            if os.path.isfile(a_) and os.path.isfile(b_):
                eq, fr = R3._same(a_, b_)
                rows.append({"check": "rerun_2threads", "approach": ap, "dataset": ds, "identical": eq, "frac_rows_equal": fr})
        for ds in RV._datasets(R3.SUITES["r4"]):
            a_ = os.path.join(RUNS4, "head", "suite", ap, ds, "predictions.csv")
            b_ = os.path.join(RUNS4, "onethread", "suite", ap, ds, "predictions.csv")
            if os.path.isfile(a_) and os.path.isfile(b_):
                eq, fr = R3._same(a_, b_)
                rows.append({"check": "1thread_vs_2thread", "approach": ap, "dataset": ds, "identical": eq, "frac_rows_equal": fr})
    if rows:
        d = pd.DataFrame(rows)
        d.to_csv(os.path.join(RES4, "determinism.csv"), index=False)
        out["determinism"] = d.groupby(["check", "approach"]).identical.agg(["sum", "count"]).reset_index().to_dict("records")
    # stress
    rows = []
    for ds in (RV._datasets(STRESS4) if os.path.isdir(STRESS4) else []):
        d = os.path.join(STRESS4, ds)
        for ap in APPROACHES:
            ad = os.path.join(RUNS4, "stress", "suite", ap, ds)
            if not os.path.isdir(ad):
                continue
            meta, s, prof, mod = RV._eval_run(ad, d)
            rows.append({"approach": ap, "probe": ds, "status": meta.get("status"), "wall_time": meta.get("wall_time"),
                         "primary": s.get("macro_f1", s.get("r2", s.get("ari")))})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RES4, "stress_results.csv"), index=False, float_format="%.6g")
    # recheck of HEAD on older suites
    rc, rcs = [], {}
    for suite in ("r1", "r2", "r3"):
        tag = f"recheck_{suite}"
        if not os.path.isdir(os.path.join(RUNS4, tag)):
            continue
        res, _, summ = R3.score_tag(tag, suite)
        res.insert(0, "suite", suite)
        rc.append(res)
        rcs[suite] = _brief(summ)
    if rc:
        pd.concat(rc).to_csv(os.path.join(RES4, "recheck_head_older_suites.csv"), index=False, float_format="%.6g")
        out["recheck_head"] = rcs
    with open(os.path.join(RES4, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "scoring"}, indent=1, default=str)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "onethread", "slowdown", "determinism", "stress", "recheck", "baselines", "score"])
    ap.add_argument("--tag", default="head")
    ap.add_argument("--code-root")
    ap.add_argument("--suite", default="r1")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s4 = R3.SUITES["r4"]
    if a.cmd == "run":
        run_loop(a.tag, s4, R3.WARMUP["r4"], code_root=a.code_root and os.path.abspath(a.code_root), only=a.only)
    elif a.cmd == "onethread":
        run_loop("onethread", s4, R3.WARMUP["r4"], threads="1", only=a.only)
    elif a.cmd == "slowdown":
        run_loop("slowdown25", s4, R3.WARMUP["r4"], only=a.only or HEAVY, slowdown=2.5)
    elif a.cmd == "determinism":
        run_loop("det", s4, R3.WARMUP["r4"], only=DET4)
    elif a.cmd == "stress":
        run_loop("stress", STRESS4, "s4_bankchurn_missing")
    elif a.cmd == "recheck":
        run_loop(f"recheck_{a.suite}", R3.SUITES[a.suite], R3.WARMUP[a.suite], only=a.only)
    elif a.cmd == "baselines":
        os.makedirs(RES4, exist_ok=True)
        saved = (RV.SUITE, RV.RUNS, RV.RESULTS)
        RV.SUITE, RV.RUNS, RV.RESULTS = s4, os.path.join(RUNS4, "baselines_runs"), RES4
        _patch_baselines_for_variant2()
        try:
            RV.cmd_baselines(a)
        finally:
            RV.SUITE, RV.RUNS, RV.RESULTS = saved
    elif a.cmd == "score":
        cmd_score(a)


if __name__ == "__main__":
    main()
