# Repository Guidelines

## Project Structure & Module Organization
Source modules stay in the root: `condition_pipeline.py` orchestrates training/inference, `feature_library.py` builds STFT/NMF descriptors, `models_manager.py` and `MIMOFIR.py` manage estimation, and `recognizers.py` provides the EVM/HMM front-end driven by `config.py`. Generated artefacts are separated into `feature/` (per-condition `.npy`), `models/` (trained FIR weights), and `result/` (EXP pipelines, plots, sensor-selection exports). Keep `images/` for publication-ready figures only, and treat `.history/` as disposable backups.

## Build, Test, and Development Commands
- `python condition_pipeline.py` — main workflow; adjust `EXP` and `UPDATE_MODEL` near the footer to target EXP1/EXP3 before running.
- `python feature_library.py` — rebuilds cached features for the hard-coded `trainlist`; duplicate the pattern for new condition groups.
- `python exp_plot.py` — recreates IEEE-style comparison PDFs from `result/sensor_selection/data/*.json`.
- `python data_split.py` — segments raw `.mat` files from `../Data/matdata` into fixed windows inside `../Data/matdata_split`.

## Coding Style & Naming Conventions
Code against Python 3.10+ with 4-space indentation, type hints, and focused docstrings (see `condition_pipeline.py`). Favor vectorized NumPy/SciPy operations, reuse helpers like `compute_feature_for_name`, and mirror the plotting palette in `exp_plot.py`. Stick with `snake_case` for symbols, `SCREAMING_SNAKE_CASE` for constants (`config.py`), and uppercase dataset identifiers such as `SW20_03`.

## Testing Guidelines
No automated suite exists, so rely on reproducible experiment runs. For pipeline edits, execute `python condition_pipeline.py` with `EXP=1` and confirm `result/pipeline/combined.mat` plus `combined_ieee.*` regenerate without warnings. For feature or recognizer changes, rerun `python feature_library.py` on a small `trainlist`, inspect the printed shapes, and follow with a short inference pass to refresh `models/`. Plot/report tweaks must be verified by running `python exp_plot.py` on at least one JSON input and reviewing the PDFs. Capture RMSE/MAPE/R² movements in the change description.

## Commit & Pull Request Guidelines
Git history favors brief, present-tense statements (for example, `完成了实验一的代码`), so keep commits focused and mention the experiment or module touched. Pull requests should summarize the scenario exercised, list the commands run, call out `config.py` or dataset changes, and link to any updated figures under `result/`. Note when artefacts are intentionally omitted because `*.mat`, `*.npy`, and `*.json` are ignored.

## Data & Configuration Notes
Do not commit raw data or generated result files; they belong under `feature/`, `models/`, `result/`, or external storage. Use `config.py` to switch between EVM and HMM strategies and to share hyper-parameters rather than scattering constants. Document custom `data_split.py` paths in the PR. When adding dependencies, ensure compatibility with the existing `numpy`, `scipy`, `scikit-learn`, and `matplotlib` stack and update the environment notes accordingly.
