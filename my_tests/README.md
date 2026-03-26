# my_tests

Evaluation suite for benchmarking ReFinEd on SemTab and specialized entity linking datasets.

---

## Scripts

| File | Description |
|------|-------------|
| `datasets.py` | Main evaluation. Loads datasets, runs ReFinED, optionally saves per-sample confidence scores for PR analysis. |
| `accuracy.py` | Computes accuracy, precision, recall, F1 from predictions. |
| `benchmark.py` | Prints environment info and provides full benchmarking (timing, memory, profiling). |
| `pr_curve.py` | Generates Precision-Recall and F1 vs confidence threshold plots from saved confidence scores. |
| `log_analysis.py` | Summarizes CSV logs from evaluation runs, including per-mode metrics. |
| `error_analysis.py` | Qualitative error analysis to identify specific prediction faults, including NIL errors. |

> Note: some additional CLI scripts exists for convenience, but are not listed.

---

## Utilities (`utility/`)

| File | Description |
|------|-------------|
| `test_utils.py` | Shared utilities: model loading, batch inference, logging, dataclasses. |

---

## Logs (`logs/`)

- `experimental_results_YYYY-MM-DD.csv` — per-dataset metrics  
- `confidence_scores_YYYY-MM-DD.csv` — per-sample confidence scores for PR curves  
- PR curve figures: `confidence_specialized_cell.png`, `confidence_semtab_row.png`, etc.  

> Run `datasets.py` before `pr_curve.py`. Use `save_confidence=False` for tuning runs to avoid polluting logs.

---

## Supported Datasets

**Specialized:** `companies`, `movies`, `SN`  
**SemTab:** `Round1_T2D`, `Round3_2019`, `Round4_2020`, `2T_Round4`, `HardTablesR2`, `HardTablesR3`  
**EL Challenge:** `HTR1`, `HTR2`
