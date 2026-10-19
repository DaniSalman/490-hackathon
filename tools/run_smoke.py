"""Run one approach on every dataset folder in a directory (subprocess, threads capped) and score it.

    py tools/run_smoke.py --approach approaches/A_lean --datasets dev_data [--threads 2] [--only c_tiny r_tiny]

Prints one line per dataset: wall time (incl. interpreter start + imports), validity, main metrics.
Writes runs/<approach>/<dataset>/ outputs and runs/<approach>/summary.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--time-budget", type=float, default=40.0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.approach))
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "HACK_N_JOBS"):
        env[k] = str(a.threads)
    rows = []
    for ds in sorted(os.listdir(a.datasets)):
        d = os.path.join(a.datasets, ds)
        if not os.path.isdir(d) or (a.only and ds not in a.only):
            continue
        out = os.path.join(a.runs, name, ds)
        os.makedirs(out, exist_ok=True)
        for f in ("predictions.csv", "profile.json"):
            p = os.path.join(out, f)
            if os.path.exists(p):
                os.remove(p)
        t = time.perf_counter()
        r = subprocess.run([sys.executable, os.path.join(a.approach, "main.py"), "--data", d, "--out", out,
                            "--time-budget", str(a.time_budget)], env=env, capture_output=True, text=True,
                           timeout=120)
        wall = time.perf_counter() - t
        with open(os.path.join(out, "stdout.txt"), "w", encoding="utf-8") as f:
            f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        s = score(d, out)
        s.update(wall_s=round(wall, 2), returncode=r.returncode, had_traceback="Traceback" in r.stderr)
        try:
            with open(os.path.join(out, "profile.json"), encoding="utf-8") as f:
                prof = json.load(f)
            s["model"] = prof.get("modelling", {}).get("model")
            s["stage"] = prof.get("modelling", {}).get("final_stage")
        except Exception:
            s["model"] = None
        rows.append(s)
        keys = [k for k in ("macro_f1", "balanced_accuracy", "accuracy", "r2", "rmsle", "ari", "nmi", "k_ok") if k in s]
        metr = " ".join(f"{k}={s[k]:.4f}" if isinstance(s[k], float) else f"{k}={s[k]}" for k in keys)
        print(f"{ds:20s} {s['wall_s']:6.2f}s valid={s['valid']} rc={r.returncode} tb={s['had_traceback']} "
              f"{metr} | {s.get('model')} {s.get('error', '')}", flush=True)
    with open(os.path.join(a.runs, name, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
