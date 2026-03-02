# my_tests

Test and evaluation suite for benchmarking ReFinED on SemTab and specialized entity linking datasets.

---

## Files

| File | Description |
|------|-------------|
| `datasets.py` | Main evaluation runner. Loads datasets, runs ReFinED inference, logs results, and optionally saves per-sample confidence scores for PR curve analysis (`save_confidence=True`). Entry point for full benchmark runs. |
| `accuracy.py` | Computes TP/FP/FN and evaluation metrics (accuracy, precision, recall, F1) from predicted spans and ground truth. |
| `benchmark.py` | Prints environment info (GPU, batch size, etc.) before evaluation runs. Also contains a full standalone benchmarking suite with manual timing, peak memory profiling, cProfile profiling, and repeated run analysis. Originally used for early performance characterisation on CPU and GPU hardware. | `pr_curve.py` | Generates Precision-Recall and F1 vs Confidence Threshold figures by reading confidence scores saved by `datasets.py`. Does not require the model to run. Produces mode-specific figures for both cell and row prediction modes. |
| `run_refined.py` | Lightweight script for running ReFinED on a single input file via CLI. |
| `sample_test.py` | Quick sanity check script for testing model output on a small set of examples. |
| `log_analysis.py` | Reads and summarises dated CSV logs from evaluation runs, including per-mode breakdowns and best F1 per dataset. |

---

## utility/

| File | Description |
|------|-------------|
| `test_utils.py` | Shared utilities: model loading, batch inference, logging, color formatting, and dataclasses (`DatasetMetadata`, `EvalMetrics`). |
| `process_files.py` | Handles loading and parsing of input files for CLI-based evaluation. |
| `testing_args.py` | CLI argument parser for `run_refined.py`. |

---

## logs/

Contains dated CSV logs from evaluation runs:
- `experimental_results_YYYY-MM-DD.csv` — per-dataset metrics logged automatically by `datasets.py`
- `confidence_scores_YYYY-MM-DD.csv` — per-sample confidence scores and correctness labels, saved when `save_confidence=True`
- `confidence_specialized_cell.png`, `confidence_specialized_row.png` — PR curve figures for specialized datasets (companies, movies, SN)
- `confidence_semtab_cell.png`, `confidence_semtab_row.png` — PR curve figures for SemTab datasets

> **Note:** Run `datasets.py` before `pr_curve.py` — the confidence score CSV must exist before plots can be generated.
> **Note:** Set `save_confidence=False` in `datasets.py` during tuning runs (e.g. batch size sweeps) to avoid polluting the confidence log with non-final results. Flip to `True` for the final benchmark run only.

---

## Workflow

1. **Tuning run** — set `save_confidence=False`, vary `BATCH_SIZES`, single dataset
2. **Final benchmark run** — set `save_confidence=True`, `BATCH_SIZES = [32]`, all datasets, both modes
3. **Analysis** — run `pr_curve.py` to generate figures, run `log_analysis.py` for summary

---

## Supported Datasets

**Specialized:** `companies`, `movies`, `SN`

**SemTab:** `Round1_T2D`, `Round3_2019`, `Round4_2020`, `2T_Round4`, `HardTablesR2`, `HardTablesR3`

**EL Challenge:** `HTR1`, `HTR2`