# EECE 490 Unknown Dataset Hackathon

Work in progress. See `docs/` for research and validation reports and `approaches/` for the three candidate solutions.

## Layout (work in progress)
- `docs/01_research_report.md` - grading analysis, assumptions, approach proposals
- `docs/02_contract_and_dataset_format.md` - assumed I/O contract, dataset folder format, how to run
- `docs/03_dev_datasets_used.md` - datasets used during development (Validator should avoid them)
- `approaches/A_lean`, `approaches/B_portfolio`, `approaches/C_rules` - three standalone candidates,
  each run as `py approaches/<X>/main.py --data <dataset_dir> --out <out_dir>`
- `tools/` - `make_dataset.py`, `score.py`, `run_smoke.py`, `build_dev_datasets.py`
