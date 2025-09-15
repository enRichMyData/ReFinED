# =========================================
# ReFinED Benchmarking Script
# =========================================

from my_tests.process_file import process_csv
from refined.inference.processor import Refined
import sys
import os
import time
import cProfile, pstats
import tracemalloc
import sys
import torch
import numpy as np



# ================== CONFIG ==================
REPEAT_RUNS = 3           # Repeat runs to account for cold start
TOP_STATS = 20            # Number of cProfile functions to show
DEFAULT_DATA_FOLDER = "my_tests/data"

# ================== HANDLE COMMAND-LINE ARGUMENT ==================
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <input_file>")
    print(f"Supported files:\n- 'imdb_top_100.csv'\n- 'companies_test.csv'")
    sys.exit(1)
 
TEXT_FILE = sys.argv[1]
 
if not os.path.exists(TEXT_FILE):
    TEXT_FILE = os.path.join(DEFAULT_DATA_FOLDER, TEXT_FILE)
    if not os.path.exists(TEXT_FILE):
        print(f"File not found: {TEXT_FILE}")
        sys.exit(1)
print(f"[INFO] Using input file: {TEXT_FILE}")


# ================== LOAD DATA ==================
texts = process_csv(TEXT_FILE)


# ================== LOAD MODEL ==================
refined = Refined.from_pretrained(
    model_name='wikipedia_model_with_numbers',
    entity_set="wikipedia"
)


# ================== HELPER FUNCTION ==================
def run_refined(texts_subset):
    """Processes a list of texts through ReFinED."""
    return [refined.process_text(t) for t in texts_subset]


# ================== ENVIRONMENT INFO ==================
print("\n[Environment Info]")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)


# ================== MANUAL TIMING ==================
print("\n[Manual Timing & Per-Text Timing]")

per_text_times = []
start_total = time.perf_counter()

for t in texts:
    start = time.perf_counter()
    refined.process_text(t)
    per_text_times.append(time.perf_counter() - start)

end_total = time.perf_counter()
total_time = end_total - start_total

print(f"Processed {len(texts)} texts in {total_time:.2f}s")
print(f"Average per-text: {sum(per_text_times)/len(per_text_times):.4f}s")
print(f"Min: {min(per_text_times):.4f}s, Max: {max(per_text_times):.4f}s")


# ================== PEAK MEMORY ==================
tracemalloc.start()
run_refined(texts)
_, peak_memory = tracemalloc.get_traced_memory()
print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")
tracemalloc.stop()


# ================== CPROFILE PROFILING ==================
print("\n[cProfile Profiling]")
profiler = cProfile.Profile()
profiler.enable()
run_refined(texts)
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(TOP_STATS)


# ================== REPEAT RUNS ==================
print("\n[Repeat Runs Timing]")
repeat_times = []
for i in range(REPEAT_RUNS):
    start = time.perf_counter()
    run_refined(texts)
    end = time.perf_counter()
    repeat_times.append(end - start)
    print(f"Run {i+1}: {end - start:.2f}s")

print(f"Average over {REPEAT_RUNS} runs: {sum(repeat_times)/len(repeat_times):.2f}s\n")


"""
Benchmark description:

This is a script for measuring **runtime performance and memory usage** of the ReFinED entity linking pipeline
on a subset of the IMDb dataset (first 100 rows). It focuses specifically on the core pipeline:

1. `Refined.from_pretrained(...)`
    - Loads the pre-trained ReFinED model for entity linking.
    - Includes transformer-based components (RoBERTa) and numerical normalization layers.
    - Only loaded once; initialization time is not included in per-text processing time.

2. `refined.process_text(text)`
    - Main entity linking function tested per text.
    - Converts raw text into tokenized tensors.
    - Passes tokens through the model to detect and classify entity mentions.
    - Produces spans with predicted entities, candidate entity rankings, coarse types (e.g., DATE, NUMERIC), and normalized values.)
    - This function is the *primary hotspot* measured in this benchmark.

3. Timing Measurements
    - **Manual timing** using `time.perf_counter()` for total and per-text runtime.
    - Measures wall-clock time for CPU execution of 100 texts.
    - Also prints min/max/average per-text timings for insight into variability.

4. Detailed Profiling with `cProfile`
    - Collects **function-level profiling data** including_
        * Total calls
        * Cumulative time
        * Per-call time.
    - Highlights major bottlenecks:
        * Transformer forward passes (RoBERTa)
            - Transformers perform self-attention across all tokens. For n tokens, attention is roughly O(n²) operations.
            - Most of time benchmark spends in this step.
            
        * Linear layers (fully connected layers dominate runtime)
            - Tasks like Mention detection, Entity classification, Scoring candidate entities.
            - Expensive as it involves matrix multiplication, computationally heavy tasks especially on larger embedding dimensions.

        * Preprocessing steps (`process_tensors`) in ReFinED
            - Converting raw token IDs and features to tensors for the model.
            - Noticeable in cProfile due to pure Python-level work, as apart from linear layers running in native code.

    - Captures Python-level computation time, not full native memory usage of PyTorch tensors.

5. Repeat Run Analysis
    - Runs the same batch multiple times to account for initialization overhead and caching effects.
    - Provides a stable average per-run runtime.

6. Notes
    - **Excluded operations**: Online wikidata queries and candidate label fetching are deliberately omitted
    to focus purely on **local model inference performance**.
    - Memory proiling via `tracemalloc` captures Python allocations only, not full PyTorch tensor memory.
    - This benchmark is suitable for **relative comparisons**, e.g. testing different model sizes, batch sizes, or hardware.

    (ifi PC)
    - Measurements are conducted on **CPU** with PyTorch 1.12.1+cpu       (smaller library than original PyTorch provided in requirements.txt)

Purpose:
- Provides precise runtime and profiling data for **entity linking performance** of ReFinED.
- Identifies computational bottlenecks.
- Forms the basis for further optimization or resource planning
"""

"""
Results (IFI PC, RHEL 9):

System Specifications:
- CPU: Intel Core i5-6500 @ 3.20GHz (4 cores / 4 threads)
- RAM: 15 GB total, ~5.8 GB available during benchmark
- Swap: 9GB, ~4.3 GB available
- OS: Red Hat Enterprise Linux 9
- Python: 3.9.21 (compiled with GCC 11.5.0)
- PyTorch: 1.12.1+cpu
- NumPy: 1.26.4
- Hardware: CPU only, CUDA not available

Manual Timing & Per-Text Timing (100 texts):
- Total runtime: 17.00s
- Average per text: 0.1700s
- Minimum per-text: 0.0474s
- Maximum per-text: 2.6689s
- Peak Python memory usage: 2.67 MB (does not include PyTorch tensor memory)

Detailed Profiling with cProfile:
- Total function calls: 1,762,167 (1,608,174 primitive)
- Total cumulative Python runtime: 16.096s
- Major bottlenecks:
    * Transformer forward passes (RoBERTa) → ~14s cumulative
    * Linear layers (fully connected) → ~9.8s cumulative
    * Preprocessing steps (`process_tensors`) → ~15.8s cumulative
- Notes: cProfile captures Python-level calls; PyTorch native operations (e.g., matrix multiplications) are included only as cumulative Python calls but not low-level GPU/CPU instructions.

Repeat Runs (to account for caching/initialization effects):
- Run 1: 8.90s
- Run 2: 6.35s
- Run 3: 5.33s
- Average over 3 runs: 6.86s

Observations:
- Variability between runs is noticeable, likely due to initial model setup, Python caching, and system scheduling.
- Most runtime is dominated by transformer forward passes and linear layers, confirming what the cProfile output highlighted.
- Memory usage is very modest at the Python allocation level; PyTorch tensors will consume more RAM during inference.
"""
