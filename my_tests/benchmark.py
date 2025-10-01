# =========================================
# ReFinED Benchmarking Script
# =========================================

from my_tests.utility.refined_utils import \
    load_input_file, \
    load_model, \
    run_refined_single, \
    run_refined_batch, \
    bolden, \
    blue_info_wrap

from my_tests.utility.testing_args import parse_args
from my_tests.accuracy import measure_accuracy

import importlib
import time
import cProfile, pstats
import tracemalloc
import sys
import torch
import numpy as np
import platform
import os



# ================== TEST FUNCTIONS ==================
def print_environment_info(device, batch, batch_size):
    print(bolden("\n[Environment Info]"))
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    if torch.cuda.is_available() and device != "cpu":
        print(blue_info_wrap("Running on GPU: "+torch.cuda.get_device_name(0)))
    else:
        print(blue_info_wrap("Running on CPU"))
        print(blue_info_wrap(f"System: {platform.system()} {platform.release()}"))
        print(blue_info_wrap(f"Machine: {platform.machine()}"))
        print(blue_info_wrap(f"Processor: {platform.processor()}"))
        print(blue_info_wrap(f"Architecture: {platform.architecture()[0]}"))
        print(blue_info_wrap(f"Cores: {os.cpu_count()}"))

    if batch: 
        print(blue_info_wrap("Using Batched mode"))
        print(blue_info_wrap(f"Batch size: {batch_size}"))
    print("\n")

def manual_timing(texts, model, run_fn, batch_size):
    """Measure runtime, works for both single and batch."""
    print(bolden("\n[Manual Timing & Per-Text Timing]"))

    start_total = time.perf_counter()
    run_fn(texts, model, batch_size)  # either single or batch
    total_time = time.perf_counter() - start_total

    # Per-text timings only make sense in single mode
    if run_fn == run_refined_single:
        per_text_times = []
        for t in texts:
            start = time.perf_counter()
            model.process_text(t)
            per_text_times.append(time.perf_counter() - start)

        print(f"Processed {len(texts)} texts in {total_time:.2f}s")
        print(f"Average per-text: {sum(per_text_times)/len(per_text_times):.4f}s")
        print(f"Min: {min(per_text_times):.4f}s, Max: {max(per_text_times):.4f}s")

    else:
        print(f"Processed {len(texts)} texts in {total_time:.2f}s (batch mode)")
        print("Per-text timing skipped (batch processes them together)")


def peak_memory_usage(texts, model, run_fn, batch_size):
    """Measure peak memory usage of Python allocations."""
    print(bolden("\n[Peak Memory Usage]"))

    tracemalloc.start()
    run_fn(texts, model, batch_size)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")

def cprofile_profiling(texts, model, top_stats, run_fn, batch_size):
    """Profile the processing using cProfile."""
    print(bolden("\n[cProfile Profiling]"))
    profiler = cProfile.Profile()
    profiler.enable()
    run_fn(texts, model, batch_size)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(top_stats)


def timed_runs(texts, model, num_runs, run_fn, batch_size, include_warmup=False):
    """Run model multiple times, measuring execution time."""
    times, results = [], None
    print(bolden("\n[Repeated Runs Timing]"))

    # Optional warmup run
    if include_warmup:
        start = time.perf_counter()
        out = run_fn(texts, model, batch_size)
        t = time.perf_counter() - start
        times.append(t)
        results = out  # save predictions from the warmup run
        print(f"Warmup Run: {t:.2f}s")

    # decide which runs to print
    if num_runs <= 10:
        print_indices = set(range(num_runs))
    else:
        step = max(1, num_runs // int(num_runs**0.5))
        print_indices = set(range(0, num_runs, step)) | {num_runs - 1}

    # measured runs
    for i in range(num_runs):
        start = time.perf_counter()
        out = run_fn(texts, model, batch_size)
        if results is None:
            results = out  # save predictions from first run if warmup not included
        t = time.perf_counter() - start
        times.append(t)

        if i in print_indices:
            print(f"Run {i+1}: {t:.2f}s")

    avg = sum(times[-num_runs:]) / num_runs  # average only over measured runs
    print(f"\nAverage over {num_runs} runs: {avg:.2f}s\n")

    return times, results






def main():
    # ================== CONFIG ==================
    REPEAT_RUNS = 50         # Repeat runs to account for cold start
    TOP_STATS = 20          # Number of cProfile functions to show
    DEFAULT_DATA_FOLDER = "my_tests/data"   # location of data-files
   

    # ======= Command-line parsing =======
    args = parse_args()
    input_file = args.input_file
    verbose = args.verbose
    batch_size = args.batch_size
    device = args.device
    gt_format = args.format
    batch = args.batch

    # ======= Load CSV and truths =======
    texts, truths = load_input_file(filename=input_file, default_data=DEFAULT_DATA_FOLDER, format=gt_format)

    # ======= Load model =======
    refined_model = load_model(device=device)

    # ======= Select run function based on BATCH flag =======
    run_fn = run_refined_batch if batch else run_refined_single

    # ======= Run benchmark =======
    print("\n\n======= START BENCHMARK  =======\n")

    print_environment_info(device=device, batch=batch, batch_size=batch_size)
    manual_timing(texts=texts, model=refined_model, run_fn=run_fn, batch_size=batch_size)
    peak_memory_usage(texts=texts, model=refined_model, run_fn=run_fn, batch_size=batch_size)
    cprofile_profiling(texts=texts, model=refined_model, top_stats=TOP_STATS, run_fn=run_fn, batch_size=batch_size)
    run_times, spans = timed_runs(texts=texts, model=refined_model, num_runs=REPEAT_RUNS, run_fn=run_fn, batch_size=batch_size)

    # ======= Accuracy  =======
    print(bolden("\n[Accuracy]"))
    measure_accuracy(spans, truths, LINE_LIMIT=None, verbose=verbose)
    
    print("\n\n======= END BENCHMARK  =======\n")

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
