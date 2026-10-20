"""Round-3 validation harness. Same grader simulation and SCORING MODEL as rounds 1-2
(imports validation/run_validation.py unchanged and only redirects its paths).

Layout: validation/runs_r3/<tag>/suite/<approach>/<dataset>/   (gitignored)
        validation/results_round3/                             (committed)

    py validation/run_validation_round3.py run            # HEAD on the round-3 suite (warm-up, 2 threads)
    py validation/run_validation_round3.py onethread      # HEAD on the round-3 suite at 1 thread
    py validation/run_validation_round3.py determinism    # rerun 2 datasets per approach (2 threads)
    py validation/run_validation_round3.py stress         # round-3 stress probes
    py validation/run_validation_round3.py versions --root-d14439d <wt> --root-05dce9e <wt>
    py validation/run_validation_round3.py recheck        # HEAD on the round-1 and round-2 suites
    py validation/run_validation_round3.py baselines
    py validation/run_validation_round3.py score
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_validation as RV  # noqa: E402

SUITES = {"r1": os.path.join(HERE, "data", "suite"), "r2": os.path.join(HERE, "data", "suite_r2"),
          "r3": os.path.join(HERE, "data", "suite_r3")}
BASELINES = {"r1": os.path.join(HERE, "results", "baselines.csv"),
             "r2": os.path.join(HERE, "results_round2", "baselines.csv")}
STRESS3 = os.path.join(HERE, "data", "stress_r3")
RUNS3 = os.path.join(HERE, "runs_r3")
RES3 = os.path.join(HERE, "results_round3")
BASELINES["r3"] = os.path.join(RES3, "baselines.csv")
WARMUP = {"r1": "k_seeds", "r2": "k_wholesale", "r3": "k_glass"}
DET3 = ["c_amazon", "k_moons"]
APPROACHES = RV.APPROACHES
STRESS_BASE = {"amazon": "c_amazon", "taxi": "r_taxi", "credit": "c_credit", "optdigits": "k_optdigits"}
VERSIONS = {"d14439d": "original", "05dce9e": "round-1 fix", "HEAD": "round-2 fix (98f5fe1)"}


def run_suite(tag, suite="r3", code_root=None, threads="2", only=None):
    saved_root, saved_thr = RV.ROOT, RV.THREADS
    RV.ROOT, RV.THREADS = (code_root or ROOT), threads
    try:
        base = os.path.join(RUNS3, tag)
        sdir = SUITES[suite]
        for ap in APPROACHES:
            RV.run_one(ap, os.path.join(sdir, WARMUP[suite]), os.path.join(base, "warmup", ap))
        for ap in APPROACHES:
            for ds in RV._datasets(sdir):
                if only and ds not in only:
                    continue
                m = RV.run_one(ap, os.path.join(sdir, ds), os.path.join(base, "suite", ap, ds))
                print(f"{tag:14s} {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']:8s} wd={m['watchdog']} "
                      f"tb={m['traceback']}", flush=True)
    finally:
        RV.ROOT, RV.THREADS = saved_root, saved_thr


def score_tag(tag, suite):
    """Round-1 scoring model on runs_r3/<tag>/suite. Returns (results DataFrame, summary dict)."""
    tmp = os.path.join(RES3, f"_tmp_{tag}")
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(BASELINES[suite], os.path.join(tmp, "baselines.csv"))
    saved = (RV.SUITE, RV.RUNS, RV.RESULTS, RV._determinism, RV._stress)
    RV.SUITE, RV.RUNS, RV.RESULTS = SUITES[suite], os.path.join(RUNS3, tag), tmp
    RV._determinism = lambda: []
    RV._stress = lambda: []
    try:
        RV.cmd_score(None)
    finally:
        RV.SUITE, RV.RUNS, RV.RESULTS, RV._determinism, RV._stress = saved
    res = pd.read_csv(os.path.join(tmp, "results.csv"))
    prof = pd.read_csv(os.path.join(tmp, "profile_scores.csv"))
    summ = json.load(open(os.path.join(tmp, "summary.json"), encoding="utf-8"))
    shutil.rmtree(tmp)
    return res, prof, summ


def _brief(summ):
    return {ap: {k: v[k] for k in ("quality", "quality_rel", "profile", "speed", "total", "per_task_quality",
                                   "wall_median", "wall_max", "statuses")} for ap, v in summ["approaches"].items()}


# =========================================================================== commands
def cmd_baselines(a):
    os.makedirs(RES3, exist_ok=True)
    saved = (RV.SUITE, RV.RUNS, RV.RESULTS)
    RV.SUITE, RV.RUNS, RV.RESULTS = SUITES["r3"], os.path.join(RUNS3, "baselines_runs"), RES3
    try:
        RV.cmd_baselines(a)
    finally:
        RV.SUITE, RV.RUNS, RV.RESULTS = saved


def cmd_determinism(a):
    saved = RV.THREADS
    RV.THREADS = "2"
    for ap in APPROACHES:
        for ds in DET3:
            m = RV.run_one(ap, os.path.join(SUITES["r3"], ds), os.path.join(RUNS3, "det", "suite", ap, ds))
            print(f"det {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']}", flush=True)
    RV.THREADS = saved


def cmd_stress(a):
    for ap in APPROACHES:
        for ds in RV._datasets(STRESS3):
            m = RV.run_one(ap, os.path.join(STRESS3, ds), os.path.join(RUNS3, "stress", "suite", ap, ds))
            print(f"stress {ap:12s} {ds:22s} {m['wall_time']:6.2f}s {m['status']} tb={m['traceback']}", flush=True)


def _same(a, b):
    pa, pb = pd.read_csv(a), pd.read_csv(b)
    eq = pa.equals(pb)
    frac = float((pa["prediction"].astype(str) == pb["prediction"].astype(str)).mean()) if len(pa) == len(pb) else 0.0
    return eq, frac


def determinism_rows():
    rows = []
    for ap in APPROACHES:
        for ds in DET3:
            a_ = os.path.join(RUNS3, "head", "suite", ap, ds, "predictions.csv")
            b_ = os.path.join(RUNS3, "det", "suite", ap, ds, "predictions.csv")
            if os.path.isfile(a_) and os.path.isfile(b_):
                eq, fr = _same(a_, b_)
                rows.append({"check": "rerun_2threads", "approach": ap, "dataset": ds, "identical": eq,
                             "frac_rows_equal": fr})
    for ap in APPROACHES:
        for ds in RV._datasets(SUITES["r3"]):
            a_ = os.path.join(RUNS3, "head", "suite", ap, ds, "predictions.csv")
            b_ = os.path.join(RUNS3, "onethread", "suite", ap, ds, "predictions.csv")
            if os.path.isfile(a_) and os.path.isfile(b_):
                eq, fr = _same(a_, b_)
                sa = RV.score_output(os.path.join(SUITES["r3"], ds), os.path.dirname(a_))
                sb = RV.score_output(os.path.join(SUITES["r3"], ds), os.path.dirname(b_))
                pm = RV.PRIMARY[sa["task"]]
                rows.append({"check": "1thread_vs_2thread", "approach": ap, "dataset": ds, "identical": eq,
                             "frac_rows_equal": fr, "metric_2thr": sa.get(pm), "metric_1thr": sb.get(pm)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES3, "determinism.csv"), index=False, float_format="%.6g")
    return df


def stress_rows():
    rows = []
    for ds in RV._datasets(STRESS3):
        d = os.path.join(STRESS3, ds)
        base_name = STRESS_BASE[ds.split("_")[1]]
        base_meta, _ = RV._load_meta(os.path.join(SUITES["r3"], base_name))
        _, truth = RV._load_meta(d)
        gt = RV.profile_truth(d, base_types=base_meta.get("feature_types"))
        for ap in APPROACHES:
            ap_dir = os.path.join(RUNS3, "stress", "suite", ap, ds)
            meta, s, prof, mod = RV._eval_run(ap_dir, d)
            pm = RV.PRIMARY[truth["task"]]
            bs = RV.score_output(os.path.join(SUITES["r3"], base_name), os.path.join(RUNS3, "head", "suite", ap, base_name))
            r = {"approach": ap, "probe": ds, "status": meta.get("status"), "wall_time": meta.get("wall_time"),
                 "traceback": meta.get("traceback"), "primary_metric": pm, "primary": s.get(pm),
                 "base_primary": bs.get(pm), "model": str(mod.get("model"))[:80]}
            if prof and "columns" in prof:
                fs, errs = RV.score_profile(prof, gt)
                r["profile_score"] = float(np.mean([fs[f] for f in RV.PROFILE_FIELDS]))
                r["profile_col_type"] = fs["col_type"]
                r["type_errors"] = "; ".join(errs)[:300]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES3, "stress_results.csv"), index=False, float_format="%.6g")
    return df


def _parse_dev_claims():
    """docs/07 per-dataset 'after' primary metric (suite r1 / r2)."""
    txt = open(os.path.join(ROOT, "docs", "07_fix_round2_changelog.md"), encoding="utf-8").read()
    rows, ap = [], None
    for line in txt.splitlines():
        m = re.match(r"^### (A_lean|B_portfolio|C_rules)\s*$", line.strip())
        if m:
            ap = m.group(1)
            continue
        m = re.match(r"^\| (r[12]) \| ([crk]_\w+) \| (-?[\d.]+) \| (-?[\d.]+) \|", line)
        if m and ap:
            rows.append({"approach": ap, "suite": m.group(1), "dataset": m.group(2), "dev_after": float(m.group(4))})
    return pd.DataFrame(rows)


def cmd_score(a):
    os.makedirs(RES3, exist_ok=True)
    out = {"scoring": RV.SCORING}
    # HEAD on round 3 (main result)
    res, prof, summ = score_tag("head", "r3")
    res.to_csv(os.path.join(RES3, "results.csv"), index=False, float_format="%.6g")
    prof.to_csv(os.path.join(RES3, "profile_scores.csv"), index=False, float_format="%.6g")
    out["head_round3"] = summ["approaches"]
    out["baselines_quality_round3"] = summ["baselines_quality"]
    # versions on round 3
    vrows, vsumm = [], {}
    for tag, label in (("v_d14439d", "original d14439d"), ("v_05dce9e", "round-1 fix 05dce9e"), ("head", "round-2 fix 98f5fe1")):
        if not os.path.isdir(os.path.join(RUNS3, tag)):
            continue
        r_, _, s_ = (res, None, summ) if tag == "head" else score_tag(tag, "r3")
        r_ = r_.copy()
        r_.insert(0, "version", label)
        vrows.append(r_)
        vsumm[label] = _brief(s_)
    if vrows:
        cols = ["version", "approach", "dataset", "task", "status", "wall_time", "watchdog", "primary", "quality_pts",
                "profile_pts", "speed_pts", "total_pts", "model"]
        pd.concat(vrows)[cols].to_csv(os.path.join(RES3, "version_comparison.csv"), index=False, float_format="%.6g")
    out["versions_round3"] = vsumm
    # recheck HEAD on r1 / r2 vs docs/07
    rc, rcs = [], {}
    dev = _parse_dev_claims()
    for suite in ("r1", "r2"):
        tag = f"recheck_{suite}"
        if not os.path.isdir(os.path.join(RUNS3, tag)):
            continue
        r_, _, s_ = score_tag(tag, suite)
        r_.insert(0, "suite", suite)
        rc.append(r_)
        rcs[suite] = _brief(s_)
    if rc:
        rcd = pd.concat(rc).merge(dev, on=["approach", "suite", "dataset"], how="left")
        rcd["diff_vs_dev"] = rcd["primary"] - rcd["dev_after"]
        rcd[["suite", "approach", "dataset", "task", "status", "wall_time", "primary", "dev_after", "diff_vs_dev",
             "quality_pts", "profile_pts", "speed_pts", "total_pts", "model"]].to_csv(
            os.path.join(RES3, "round1_recheck.csv"), index=False, float_format="%.6g")
        out["recheck_head"] = rcs
        out["recheck_max_abs_diff"] = float(rcd["diff_vs_dev"].abs().max())
    # 1-thread timing
    if os.path.isdir(os.path.join(RUNS3, "onethread")):
        r1t, _, s1t = score_tag("onethread", "r3")
        t = res[["approach", "dataset", "task", "wall_time", "primary"]].merge(
            r1t[["approach", "dataset", "wall_time", "primary", "status", "watchdog"]], on=["approach", "dataset"],
            suffixes=("_2thr", "_1thr"))
        t["slowdown"] = t["wall_time_1thr"] / t["wall_time_2thr"]
        t.to_csv(os.path.join(RES3, "onethread_timing.csv"), index=False, float_format="%.6g")
        out["onethread"] = {ap: {"median_1thr": round(g.wall_time_1thr.median(), 2), "max_1thr": round(g.wall_time_1thr.max(), 2),
                                 "max_dataset": g.loc[g.wall_time_1thr.idxmax(), "dataset"],
                                 "n_over_40s": int((g.wall_time_1thr > 40).sum()), "n_over_30s": int((g.wall_time_1thr > 30).sum()),
                                 "watchdog": int(g.watchdog.sum()), "total_1thr": round(s1t["approaches"][ap]["total"], 3)}
                            for ap, g in t.groupby("approach")}
    if os.path.isdir(os.path.join(RUNS3, "det")):
        d = determinism_rows()
        out["determinism"] = d.groupby(["check", "approach"]).identical.agg(["sum", "count"]).reset_index().to_dict("records")
    if os.path.isdir(os.path.join(RUNS3, "stress")):
        stress_rows()
    with open(os.path.join(RES3, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"head": _brief({"approaches": out["head_round3"]}), "versions": vsumm, "recheck": out.get("recheck_head"),
                      "recheck_max_abs_diff": out.get("recheck_max_abs_diff"), "onethread": out.get("onethread")},
                     indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "onethread", "determinism", "stress", "versions", "recheck", "baselines", "score"])
    ap.add_argument("--root-d14439d")
    ap.add_argument("--root-05dce9e")
    a = ap.parse_args()
    if a.cmd == "run":
        run_suite("head", "r3")
    elif a.cmd == "onethread":
        run_suite("onethread", "r3", threads="1")
    elif a.cmd == "determinism":
        cmd_determinism(a)
    elif a.cmd == "stress":
        cmd_stress(a)
    elif a.cmd == "versions":
        run_suite("v_d14439d", "r3", code_root=os.path.abspath(a.root_d14439d))
        run_suite("v_05dce9e", "r3", code_root=os.path.abspath(a.root_05dce9e))
    elif a.cmd == "recheck":
        run_suite("recheck_r1", "r1")
        run_suite("recheck_r2", "r2")
    elif a.cmd == "baselines":
        cmd_baselines(a)
    elif a.cmd == "score":
        cmd_score(a)


if __name__ == "__main__":
    main()
