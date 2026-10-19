"""Entry point (identical CLI for all approaches).

    py main.py --data <dataset_dir> --out <out_dir> [--time-budget 40] [--n-jobs N]
    py main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>

Guarantees: a valid predictions.csv + profile.json is written as early as possible and then
improved; a watchdog terminates the process before the hard limit with the best answer on disk.
"""
import time

T0 = time.perf_counter()  # the budget clock starts before any heavy import

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _parse(argv):
    ap = argparse.ArgumentParser(description="Unknown-dataset solver")
    ap.add_argument("--data", help="dataset folder (train.csv/test.csv/metadata.json)")
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--meta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=40.0, help="internal soft budget (s) from start")
    ap.add_argument("--hard-limit", type=float, default=None, help="watchdog exit time (s); default budget+12")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _cap_threads(n_jobs):
    if n_jobs:
        os.environ["HACK_N_JOBS"] = str(n_jobs)
    n = os.environ.get("HACK_N_JOBS") or os.environ.get("OMP_NUM_THREADS") or str(max(1, min(4, os.cpu_count() or 1)))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, n)
    os.environ.setdefault("HACK_N_JOBS", n)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", n)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def main(argv=None):
    args = _parse(argv if argv is not None else sys.argv[1:])
    _cap_threads(args.n_jobs)
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from io_contract import load_task, write_predictions, write_profile
    from utils import Budget

    budget = Budget(T0, args.time_budget)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(f"[{budget.elapsed():6.2f}s]", *a, flush=True))
    task = load_task(args.data, args.train, args.test, args.meta)
    log(f"loaded {task.name}: task={task.task_type} train={task.train_df.shape} "
        f"predict_rows={task.n_predict} target={task.target} k={task.n_clusters}")
    state = {"written": False, "stage": None}
    lock = threading.Lock()

    def emit(preds, stage):
        preds = np.asarray(preds)
        if len(preds) != task.n_predict:
            log(f"emit({stage}) rejected: wrong length {len(preds)}")
            return
        with lock:
            write_predictions(task, preds, args.out)
            state["written"], state["stage"] = True, stage
        log(f"predictions written ({stage})")

    # 1) immediate constant fallback
    try:
        emit(_constant_fallback(task), "constant_fallback")
    except Exception:
        traceback.print_exc()

    # 2) watchdog: exit before the hard limit; the best predictions are already on disk
    hard = args.hard_limit if args.hard_limit is not None else args.time_budget + 12.0
    def _watchdog():
        with lock:
            print(f"[watchdog] hard limit {hard:.0f}s reached; exiting with stage={state['stage']}", flush=True)
            sys.stdout.flush()
            os._exit(0)
    wd = threading.Timer(max(1.0, hard - budget.elapsed()), _watchdog)
    wd.daemon = True
    wd.start()

    # 3) profile (always written, even if modelling fails)
    profile = {}
    try:
        from data_profile import build_profile
        profile = build_profile(task)
        write_profile(profile, args.out)
        log("profile written")
    except Exception as e:
        traceback.print_exc()
        profile = {"error": f"profile failed: {e!r}", "task": task.task_type, "n_rows": int(len(task.train_df)),
                   "n_columns": int(task.train_df.shape[1])}
        try:
            write_profile(profile, args.out)
        except Exception:
            traceback.print_exc()

    # 4) modelling
    info = {}
    try:
        import model
        info = model.solve(task, budget, emit, log) or {}
    except Exception as e:
        traceback.print_exc()
        info = {"error": repr(e), "fallback_used": True}
    info["final_stage"] = state["stage"]
    info["elapsed_seconds"] = round(budget.elapsed(), 3)
    try:
        try:
            from data_profile import enrich_with_preparation
            enrich_with_preparation(profile, info.pop("_prep_info", None))
        except Exception:
            info.pop("_prep_info", None)
        profile["modelling"] = info
        write_profile(profile, args.out)
    except Exception:
        traceback.print_exc()
    log(f"done in {budget.elapsed():.2f}s (stage={state['stage']})")
    wd.cancel()
    return 0


def _constant_fallback(task):
    import numpy as np
    n = task.n_predict
    if task.task_type == "classification":
        y = task.y_train.dropna()
        return np.repeat(y.value_counts().index[0], n) if len(y) else np.zeros(n, int)
    if task.task_type == "regression":
        from cleaning import coerce_numeric
        y = coerce_numeric(task.y_train).dropna()
        return np.full(n, float(y.median()) if len(y) else 0.0)
    k = max(1, int(task.n_clusters or 2))
    return np.arange(n) % k


if __name__ == "__main__":
    sys.exit(main())
