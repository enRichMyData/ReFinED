# =========================================
# ReFinED Benchmarking Script
# =========================================

from my_tests.refined_utils import parse_args, load_input_file, load_model, run_refined

import importlib
import time
import cProfile, pstats
import tracemalloc
import sys
import torch
import numpy as np



# ================== TEST FUNCTIONS ==================
def print_environment_info():
    print("\n[Environment Info]")
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)

def manual_timing(texts, model):
    """Measure per-text and total runtime."""
    print("\n[Manual Timing & Per-Text Timing]")
    per_text_times = []
    start_total = time.perf_counter()

    for t in texts:
        start = time.perf_counter()
        model.process_text(t)
        per_text_times.append(time.perf_counter() - start)

    total_time = time.perf_counter() - start_total

    print(f"Processed {len(texts)} texts in {total_time:.2f}s")
    print(f"Average per-text: {sum(per_text_times)/len(per_text_times):.4f}s")
    print(f"Min: {min(per_text_times):.4f}s, Max: {max(per_text_times):.4f}s")

def peak_memory_usage(texts, model):
    """Measure peak memory usage of Python allocations."""
    print("\n[Peak Memory Usage]")

    tracemalloc.start()
    run_refined(texts, model)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")

def cprofile_profiling(texts, model, top_stats):
    """Profile the processing using cProfile."""
    print("\n[cProfile Profiling]")
    profiler = cProfile.Profile()
    profiler.enable()
    run_refined(texts, model)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(top_stats)

def repeat_runs_timing(texts, model, repeat_runs):
    """Run multiple times to account for warmup effects."""
    print("\n[Repeat Runs Timing]")
    repeat_times = []

    for i in range(repeat_runs):
        start = time.perf_counter()
        run_refined(texts, model)
        end = time.perf_counter()
        repeat_times.append(end - start)
        print(f"Run {i+1}: {end - start:.2f}s")

    avg_time = sum(repeat_times) / repeat_runs

    print(f"Average over {repeat_runs} runs: {avg_time:.2f}s\n")

def warmup_and_repeat_runs(texts, model, num_runs=6):
    """
    Measures sequential runs including the first warmup run.
    The first run typically takes longer, whereas the rest benefit from caching and preloaded model.
    """
    print(f"\n[Warmup + Repeat Runs Timing] ({num_runs} runs)")
    run_times = []

    for i in range(num_runs):
        start = time.perf_counter()
        run_refined(texts, model)
        end = time.perf_counter()
        run_time = end - start
        run_times.append(run_time)
        if i == 0:
            print(f"Warmup Run (Run 1): {run_time:.2f}s")
        else:
            print(f"Run {i+1}: {run_time:.2f}s")

    avg_time = sum(run_times[1:]) / (num_runs - 1) if num_runs > 1 else run_times[0]

    print(f"Average (excluding warmup) over {num_runs - 1} runs: {avg_time:.2f}s\n")
    return run_times


def main():
    # ================== CONFIG ==================
    USE_CPU = False         # using cpu or gpu
    REPEAT_RUNS = 3         # Repeat runs to account for cold start
    TOP_STATS = 20          # Number of cProfile functions to show
    DEFAULT_DATA_FOLDER = "my_tests/data"   # location of data-files


    # ======= Command-line parsing =======
    input_file, verbose = parse_args()

    # ======= Load CSV and truths =======
    texts, truths = load_input_file(filename=input_file, default_data=DEFAULT_DATA_FOLDER)

    # ======= Load model =======
    refined_model = load_model(USE_CPU=USE_CPU)

    # ======= Run benchmark =======
    print_environment_info()
    manual_timing(texts=texts, model=refined_model)
    peak_memory_usage(texts=texts, model=refined_model)
    cprofile_profiling(texts=texts, model=refined_model, top_stats=TOP_STATS)
    repeat_runs_timing(texts=texts, model=refined_model, repeat_runs=REPEAT_RUNS)

    # Special benchmark for repeated runs including warm run
    run_times = warmup_and_repeat_runs(texts=texts, model=refined_model, num_runs=10)

    # plotting for image
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(run_times) + 1), run_times, marker='o')
        plt.xlabel("Run number")
        plt.ylabel("Runtime (s)")
        plt.title("ReFinED Warmup + Repeat Runs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(DEFAULT_DATA_FOLDER + f"/warmup_runs_{input_file}.png")

if __name__ == "__main__":
    main()

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
