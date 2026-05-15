# my_tests
Evaluation suite for benchmarking ReFinED on SemTab and specialized entity linking datasets.

## Core Scripts
* **`datasets.py`**: Main evaluation entry point (inference and scoring).
* **`run_refined.py`**: CLI for manual inference runs.
* **`benchmark.py`**: Profiling of latency, memory, and hardware.
* **`accuracy.py`**: core metric logic for Precision, Recall, and F1.

## Analysis & Utilities
* **`analysis/`**: Tools for log aggregation (`log_summary.py`), PR-curves (`pr_curve.py`), and error analysis (`qualitative_analysis.py`).
* **`utility/`**: Shared logic for model loading, batching, and argument parsing.

## Data (`data/`)
* **Specialised**: Pre-configured folders for `companies`, `movies`, `SN`, and `EL_challenge`.
* **Benchmarks**: To run SemTab, manually add datasets to `data/datasets/` (e.g., `data/datasets/Round4_2020/`).
* **Note**: Large benchmark files are excluded from Git due to size.

## Logs & Visuals (`logs/`)
Contains data logs for analysis and visualisations:
* **Evaluation Metrics**: `experimental_results_*.csv` files containing per-dataset raw metrics and human-readable summaries.
* **Fine-tuning**: `fine_tuning_logs.md` (hyperparameters and model performance) and `fine_tune_summary.pdf`.
* **Error Analysis**: `error_analysis.csv` records specific prediction faults for qualitative review.
* **Visualisations**: Generated Precision-Recall and F1-curve plots (`confidence_*.png`) for all evaluation modes.

> **Workflow:** Run `datasets.py` first to generate logs. Use `save_confidence=False` during rapid testing to save disk space.