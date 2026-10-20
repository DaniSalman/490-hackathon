"""Round-2 validation harness. Re-uses the round-1 grader simulation and SCORING MODEL unchanged
(imports validation/run_validation.py and only redirects its paths), so rounds are comparable.

    py validation/run_validation_round2.py run          # warm-up + 18 round-2 datasets x 3 approaches
    py validation/run_validation_round2.py baselines    # trivial + sklearn baselines on the round-2 suite
    py validation/run_validation_round2.py determinism  # rerun 2 datasets per approach + B 1-thread vs 2-thread
    py validation/run_validation_round2.py stress       # round-2 stress probes
    py validation/run_validation_round2.py recheck      # fixed approaches on the ROUND-1 suite (developer claim check)
    py validation/run_validation_round2.py score        # -> validation/results_round2/*
    py validation/run_validation_round2.py all

Round-1 reproducibility: run_validation.py is not modified; round-1 runs/results live in validation/runs/
and validation/results/ and are not touched (round 2 writes to validation/runs_r2/ and results_round2/).
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

R1_SUITE = RV.SUITE
R1_RESULTS = RV.RESULTS
SUITE2 = os.path.join(HERE, "data", "suite_r2")
STRESS2 = os.path.join(HERE, "data", "stress_r2")
RUNS2 = os.path.join(HERE, "runs_r2")
RES2 = os.path.join(HERE, "results_round2")
WARMUP2 = "k_wholesale"
DET2 = ["c_kick", "k_synth"]
THREAD_DET = ["c_jannis", "r_superconduct", "c_pageblocks"]
APPROACHES = RV.APPROACHES
STRESS_BASE = {"churn": "c_churn", "kick": "c_kick", "kc": "r_kc_house", "synth": "k_synth"}


def _use_round2():
    RV.SUITE, RV.STRESS, RV.RUNS, RV.RESULTS = SUITE2, STRESS2, RUNS2, RES2
    RV.WARMUP_DS, RV.DET_DS = WARMUP2, DET2
    RV._determinism = _determinism2
    RV._stress = _stress2


# =========================================================================== commands
def cmd_run(a):
    _use_round2()
    RV.cmd_run(a)


def cmd_baselines(a):
    _use_round2()
    RV.cmd_baselines(a)


def cmd_determinism(a):
    _use_round2()
    for ap in APPROACHES:
        for ds in DET2:
            m = RV.run_one(ap, os.path.join(SUITE2, ds), os.path.join(RUNS2, "det", ap, ds))
            print(f"det {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']}", flush=True)
    old = RV.THREADS
    RV.THREADS = "1"
    try:
        for ds in THREAD_DET:
            m = RV.run_one("B_portfolio", os.path.join(SUITE2, ds), os.path.join(RUNS2, "thread1", "B_portfolio", ds))
            print(f"1-thread B {ds:14s} {m['wall_time']:6.2f}s {m['status']}", flush=True)
    finally:
        RV.THREADS = old


def cmd_stress(a):
    _use_round2()
    for ap in APPROACHES:
        for ds in RV._datasets(STRESS2):
            m = RV.run_one(ap, os.path.join(STRESS2, ds), os.path.join(RUNS2, "stress", ap, ds))
            print(f"stress {ap:12s} {ds:20s} {m['wall_time']:6.2f}s {m['status']} tb={m['traceback']}", flush=True)


def cmd_recheck(a):
    """Independent re-run of the fixed approaches on the ROUND-1 suite (same protocol as round 1)."""
    base = os.path.join(RUNS2, "r1recheck")
    for ap in APPROACHES:
        RV.run_one(ap, os.path.join(R1_SUITE, "k_seeds"), os.path.join(base, "warmup", ap))
    for ap in APPROACHES:
        for ds in RV._datasets(R1_SUITE):
            m = RV.run_one(ap, os.path.join(R1_SUITE, ds), os.path.join(base, "suite", ap, ds))
            print(f"r1 {ap:12s} {ds:14s} {m['wall_time']:6.2f}s {m['status']:8s} wd={m['watchdog']}", flush=True)


# =========================================================================== extra profile fields (S3)
SUSPECT = {"?", "na", "n/a", "nan", "null", "#n/a", "missing"}


def _new_field_truth(d):
    df = RV._raw_train(d)
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = (0, 0)
            continue
        v = s.dropna().astype(str)
        st = v.str.strip()
        blank = int((st == "").sum())
        out[c] = (blank, blank + int(st.str.lower().isin(SUSPECT).sum()))
    return out


def score_new_fields(prof, d, target):
    gt = _new_field_truth(d)
    cols = {c.get("name"): c for c in (prof.get("columns") or [])}
    bl = [cols.get(c, {}).get("blank_string_count") == v[0] for c, v in gt.items()]
    su = [cols.get(c, {}).get("suspected_missing") == v[1] for c, v in gt.items()]
    feats = [c for c in gt if c != target]
    tot_b = sum(gt[c][0] for c in feats)
    tot_s = sum(gt[c][1] for c in feats)
    return {"blank_string_count": float(np.mean(bl)), "suspected_missing": float(np.mean(su)),
            "total_blank_string_cells": float(prof.get("total_blank_string_cells") == tot_b),
            "total_suspected_missing_cells": float(prof.get("total_suspected_missing_cells") == tot_s),
            "gt_total_blank": tot_b, "gt_total_suspected": tot_s}


# =========================================================================== determinism / stress
def _cmp_preds(a, b):
    pa, pb = pd.read_csv(a), pd.read_csv(b)
    r = {"identical": bool(pa.equals(pb)),
         "frac_rows_equal": float((pa["prediction"].astype(str) == pb["prediction"].astype(str)).mean())
         if len(pa) == len(pb) else 0.0}
    return r, pa, pb


def _determinism2():
    rows = []
    for ap in APPROACHES:
        for ds in DET2:
            a = os.path.join(RUNS2, "suite", ap, ds, "predictions.csv")
            b = os.path.join(RUNS2, "det", ap, ds, "predictions.csv")
            if os.path.isfile(a) and os.path.isfile(b):
                r, pa, pb = _cmp_preds(a, b)
                rows.append({"check": "rerun_2threads", "approach": ap, "dataset": ds, **r})
    for ds in THREAD_DET:
        a = os.path.join(RUNS2, "suite", "B_portfolio", ds, "predictions.csv")
        b = os.path.join(RUNS2, "thread1", "B_portfolio", ds, "predictions.csv")
        if os.path.isfile(a) and os.path.isfile(b):
            r, pa, pb = _cmp_preds(a, b)
            ma = RV._read_json(os.path.join(RUNS2, "suite", "B_portfolio", ds, "profile.json")) or {}
            mb = RV._read_json(os.path.join(RUNS2, "thread1", "B_portfolio", ds, "profile.json")) or {}
            sa = RV.score_output(os.path.join(SUITE2, ds), os.path.dirname(a))
            sb = RV.score_output(os.path.join(SUITE2, ds), os.path.dirname(b))
            pm = RV.PRIMARY[sa["task"]]
            rows.append({"check": "B_1thread_vs_2thread", "approach": "B_portfolio", "dataset": ds, **r,
                         "metric_2thr": sa.get(pm), "metric_1thr": sb.get(pm),
                         "model_2thr": str((ma.get("modelling") or {}).get("model"))[:80],
                         "model_1thr": str((mb.get("modelling") or {}).get("model"))[:80]})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RES2, "determinism.csv"), index=False)
    return rows


def _stress2():
    rows = []
    if not os.path.isdir(os.path.join(RUNS2, "stress")):
        return rows
    for ds in RV._datasets(STRESS2):
        d = os.path.join(STRESS2, ds)
        base_name = STRESS_BASE[ds.split("_")[1]]
        base_meta, _ = RV._load_meta(os.path.join(SUITE2, base_name))
        _, truth = RV._load_meta(d)
        gt = RV.profile_truth(d, base_types=base_meta.get("feature_types"))
        for ap in APPROACHES:
            ap_dir = os.path.join(RUNS2, "stress", ap, ds)
            meta, s, prof, mod = RV._eval_run(ap_dir, d)
            pm = RV.PRIMARY[truth["task"]]
            bs = RV.score_output(os.path.join(SUITE2, base_name), os.path.join(RUNS2, "suite", ap, base_name))
            r = {"approach": ap, "probe": ds, "status": meta.get("status"), "wall_time": meta.get("wall_time"),
                 "traceback": meta.get("traceback"), "primary_metric": pm, "primary": s.get(pm),
                 "base_primary": bs.get(pm), "model": str(mod.get("model"))[:80]}
            if prof and "columns" in prof:
                fs, errs = RV.score_profile(prof, gt)
                r["profile_score"] = float(np.mean([fs[f] for f in RV.PROFILE_FIELDS]))
                r["profile_col_type"] = fs["col_type"]
                r["type_errors"] = "; ".join(errs)[:300]
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES2, "stress_results.csv"), index=False, float_format="%.6g")
    return rows


# =========================================================================== developer-claim check
def _parse_dev_claims():
    """Per-dataset 'after' primary metric + wall time from docs/05 (A/B/C tables)."""
    txt = open(os.path.join(ROOT, "docs", "05_fix_round1_changelog.md"), encoding="utf-8").read()
    rows, ap = [], None
    for line in txt.splitlines():
        m = re.match(r"^### (A_lean|B_portfolio|C_rules)\s*$", line.strip())
        if m:
            ap = m.group(1)
            continue
        m = re.match(r"^\| ([crk]_\w+) \| (-?[\d.]+) \| (-?[\d.]+) \| [+-][\d.]+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m and ap:
            rows.append({"approach": ap, "dataset": m.group(1), "dev_after": float(m.group(3)),
                         "dev_wall_after": float(m.group(5)), "dev_pts_after": float(m.group(7))})
    return pd.DataFrame(rows)


def score_recheck():
    """Score runs_r2/r1recheck with the round-1 scoring (round-1 suite + baselines)."""
    tmp = os.path.join(RES2, "_r1recheck_tmp")
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(os.path.join(R1_RESULTS, "baselines.csv"), os.path.join(tmp, "baselines.csv"))
    saved = (RV.SUITE, RV.RUNS, RV.RESULTS, RV._determinism, RV._stress)
    RV.SUITE, RV.RUNS, RV.RESULTS = R1_SUITE, os.path.join(RUNS2, "r1recheck"), tmp
    RV._determinism = lambda: []
    RV._stress = lambda: []
    try:
        RV.cmd_score(None)
    finally:
        RV.SUITE, RV.RUNS, RV.RESULTS, RV._determinism, RV._stress = saved
    res = pd.read_csv(os.path.join(tmp, "results.csv"))
    summ = json.load(open(os.path.join(tmp, "summary.json"), encoding="utf-8"))
    dev = _parse_dev_claims()
    out = res.merge(dev, on=["approach", "dataset"], how="left")
    out["diff_vs_dev"] = out["primary"] - out["dev_after"]
    cols = ["approach", "dataset", "task", "status", "wall_time", "dev_wall_after", "primary", "dev_after",
            "diff_vs_dev", "quality_pts", "profile_pts", "speed_pts", "total_pts", "dev_pts_after", "model"]
    out[cols].to_csv(os.path.join(RES2, "round1_recheck.csv"), index=False, float_format="%.6g")
    shutil.rmtree(tmp)
    return {ap: {k: v[k] for k in ("quality", "profile", "speed", "total", "per_task_quality", "wall_median",
                                   "wall_max", "statuses")} for ap, v in summ["approaches"].items()}


# =========================================================================== score
def cmd_score(a):
    _use_round2()
    os.makedirs(RES2, exist_ok=True)
    RV.cmd_score(a)       # results.csv, profile_scores.csv, summary.json, determinism.csv, stress_results.csv
    # extra (separate) profile fields from S3
    rows = []
    for ds in RV._datasets(SUITE2):
        d = os.path.join(SUITE2, ds)
        _, truth = RV._load_meta(d)
        for ap in APPROACHES:
            prof = RV._read_json(os.path.join(RUNS2, "suite", ap, ds, "profile.json")) or {}
            r = {"approach": ap, "dataset": ds}
            r.update(score_new_fields(prof, d, truth.get("target")) if prof.get("columns") else {})
            rows.append(r)
    nf = pd.DataFrame(rows)
    nf.to_csv(os.path.join(RES2, "profile_new_fields.csv"), index=False)
    summ = json.load(open(os.path.join(RES2, "summary.json"), encoding="utf-8"))
    summ["profile_new_fields_mean"] = nf.groupby("approach")[["blank_string_count", "suspected_missing",
                                                              "total_blank_string_cells",
                                                              "total_suspected_missing_cells"]].mean().round(4).to_dict("index")
    if os.path.isdir(os.path.join(RUNS2, "r1recheck")):
        summ["round1_recheck"] = score_recheck()
    with open(os.path.join(RES2, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=str)
    print(json.dumps(summ.get("round1_recheck"), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "baselines", "determinism", "stress", "recheck", "score", "all"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd in ("run", "all"):
        cmd_run(a)
    if a.cmd in ("determinism", "all"):
        cmd_determinism(a)
    if a.cmd in ("stress", "all"):
        cmd_stress(a)
    if a.cmd in ("recheck", "all"):
        cmd_recheck(a)
    if a.cmd in ("baselines", "all"):
        cmd_baselines(a)
    if a.cmd in ("score", "all"):
        cmd_score(a)


if __name__ == "__main__":
    main()
