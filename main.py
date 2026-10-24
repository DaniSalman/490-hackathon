"""Standard entry point of the submission.

    python main.py --data <dataset_dir> --out <out_dir> [--time-budget 40] [--n-jobs N]
    python main.py --train train.csv --test test.csv --meta metadata.json --out <out_dir>

Runs the selected approach (see SELECTED below and README.md). Every approach in approaches/ is a
self-contained solver with the same command line and the same outputs (predictions.csv, profile.json).
"""
import os
import runpy
import sys

SELECTED = "A_lean"  # the submitted approach: A_lean | B_portfolio | C_rules

if __name__ == "__main__":
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "approaches", SELECTED, "main.py")
    sys.argv[0] = entry
    runpy.run_path(entry, run_name="__main__")
