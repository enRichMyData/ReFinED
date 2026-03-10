from datetime import datetime
from pathlib import Path
import pandas as pd
import torch.cuda
import random
import glob
import time
import csv
import os

# internal imports
from my_tests.accuracy import measure_accuracy
from my_tests.benchmark import print_environment_info
from my_tests.utility.test_utils import (
    load_model, run_refined_batch,       # model
    bolden, red_info, green_info,        # style
    log_results_to_csv, add_log_divider, # logging 1
    DatasetMetadata, get_dated_filename  # logging 2
)


DATA_FOLDER = "my_tests/data"

DATASET_FILES = {
    "2T_Round4": ("CEA_2T_WD_Targets.csv", "cea.csv"),
    "HardTablesR2": ("HardTable_CEA_WD_Round2_Targets.csv", "cea.csv"),
    "HardTablesR3": ("HardTablesR3_CEA_WD_Round3_Targets.csv", "cea.csv"),
    "Round1_T2D": ("CEA_Round1_Targets.csv", "CEA_Round1_gt_WD.csv"),
    "Round3_2019": ("CEA_Round3_Targets.csv", "CEA_Round3_gt_WD.csv"),
    "Round4_2020": ("CEA_Round4_targets.csv", "cea.csv"),
}


def save_confidence_scores(spans, truths, dataset_name, prediction_mode, model_name="wikipedia_model_with_numbers"):
    """Saves per-sample confidence scores and correctness labels for PR curve analysis."""
    conf_log_file = get_dated_filename(model_name).replace("experimental_results", "confidence_scores")

    rows = []
    for pred_spans, truth_qids in zip(spans, truths):
        if not truth_qids:
            continue
        if pred_spans:
            span = pred_spans[0]
            conf = float(getattr(span, "entity_linking_model_confidence_score", 0.0) or 0.0)
            pred_qid = getattr(span.predicted_entity, "wikidata_entity_id", "NIL") or "NIL"
        else:
            conf, pred_qid = 0.0, "NIL"
        is_correct = int(pred_qid in truth_qids or (pred_qid == "NIL" and "NIL" in truth_qids))
        rows.append([dataset_name, prediction_mode, conf, is_correct])

    write_header = not os.path.exists(conf_log_file) or os.stat(conf_log_file).st_size == 0
    with open(conf_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "mode", "confidence", "is_correct"])
        writer.writerows(rows)


# Data loading
# ===========================================================
def clean_qids(val):
    """
    Converts a raw QID value to a parsed list, returning ["NIL"] for missing or null entries
    """
    if pd.isna(val) or str(val).strip().lower() in ["nan", "nil", "none", ""]:
        return ["NIL"]
    return [x.split("/")[-1] for x in str(val).split()]



def get_dataset_metadata(df, samples_len):
    """
    Counts total targets,  NIL entries, and sample size for logging.
    """
    nil_mask = df["qid"].apply(lambda v: clean_qids(v) == ["NIL"])
    return DatasetMetadata(
        total_targets=len(df),
        nil_count=nil_mask.sum(),
        sample_size=samples_len
    )


def load_dataset_config(dataset_name):
    """
    Returns the tables folder path and ground truth DataFrame for a given dataset.
    """

    # 1. EL Challenge (HTR) logic
    if "HTR" in dataset_name:
        base = f"{DATA_FOLDER}/EL_challenge/{dataset_name}"
        tables_folder = f"{base}/tables"
        gt = pd.read_csv(f"{base}/gt/cea_gt.csv", header=None, names=["table", "row", "col", "qid"])
        return tables_folder, gt

    # 2. Specialized (SN, Movies, Companies) logic
    if dataset_name in ["SN", "movies", "companies"]:
        base = f"{DATA_FOLDER}/{dataset_name}"
        table_map = {"SN": "SN_test", "movies": "movies_test", "companies": "companies_test"}
        gt_map = {"SN": "SN_gt.csv", "movies": "el_movies_gt_wikidata.csv", "companies": "el_companies_gt_wikidata.csv"}

        gt = pd.read_csv(f"{base}/{gt_map[dataset_name]}", header=0)
        gt = gt.rename(columns={"tableName": "table", "idRow": "row", "idCol": "col", "entity": "qid"})
        gt = gt[gt["table"] == table_map[dataset_name]].copy()
        gt["row"] = gt["row"].astype(int) + 1  # Header offset
        return base, gt

    # 3. Standard SemTab logic
    base = f"{DATA_FOLDER}/datasets/{dataset_name}"
    targets_file = f"{base}/targets/{DATASET_FILES[dataset_name][0]}"
    gt_file = f"{base}/gt/{DATASET_FILES[dataset_name][1]}"

    # Handle weird T2D column ordering
    t_cols = ["table", "col", "row"] if dataset_name in ["Round1_T2D", "Round3_2019"] else ["table", "row", "col"]
    targets = pd.read_csv(targets_file, header=None, names=t_cols)
    gt = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])

    merged = targets.merge(gt, on=["table", "row", "col"], how="left")
    return f"{base}/tables", merged



def build_eval_samples(table_to_truths, tables_folder, prediction_mode):
    """
    Builds (text, ground truth) pairs from tables, using cell or row context mode.
    """
    s_build = time.perf_counter()
    texts, truths = [], []

    # fast lookup preparation
    gt_lookup = {str(k).replace(".csv", ""): df for k, df in table_to_truths.items()}
    gt_ids = sorted(gt_lookup.keys(), key=len, reverse=True)  # Longest first to avoid partial prefix matches

    # stats
    total_files = 0
    files_processed = 0

    # iterate through tables
    for table_path in glob.glob(f"{tables_folder}/*.csv"):
        full_stem = Path(table_path).stem
        total_files += 1

        # matching logic
        matched_id = None
        if full_stem in gt_lookup:
            matched_id = full_stem
        else:
            # fallback for T2D/Round 1 where filenames have extra timestamps/hashes
            for gt_id in gt_ids:
                if full_stem.startswith(gt_id):
                    matched_id = gt_id
                    break

        if not matched_id:
            continue
        # --------------------------------

        files_processed += 1

        # using header none because semtab treats headers as data rows sometimes
        df_table = pd.read_csv(table_path, header=None).astype(str)
        rows = df_table.values.tolist()

        # get relevant Ground Truth rows
        golds = gt_lookup[matched_id].copy()
        golds["qid_list"] = golds["qid"].apply(clean_qids)

        for gold_row in golds.itertuples(index=False):
            try:
                r = int(gold_row.row)
                c = int(gold_row.col)

                # bound check
                if 0 <=  r < len(rows) and 0 <= c < len(rows[r]):
                    cell = rows[r][c]

                    # cell-level mode
                    if prediction_mode == "cell":
                        text = cell

                    # row-level mode (context)
                    else:
                        context = rows[r][:c] + rows[r][c + 1:]
                        text = f"{cell} | {' | '.join(context)}"


                    # after processing:  add to lists
                    texts.append(text)
                    truths.append(gold_row.qid_list)
            except (ValueError, IndexError):
                continue


    # stat information
    build_duration = time.perf_counter() - s_build
    print(f"[{prediction_mode.upper()}] "
          f"Matched {files_processed:,}/{total_files:,} tables | "
          f"Samples: {len(texts):,} | "
          f"Time: {build_duration:.2f}s"
          f"{' (non-table files skipped)' if files_processed < total_files else ''}")

    return texts, truths

# ===========================================================


# Evaluation & Logging
# ===========================================================
def run_refined_eval(
    model, texts, truths, batch_size, dataset_name,
    prediction_mode, model_name, meta,
    save_confidence=False, log=False, verbose=False,
    sample_size=None, seed=42
):
    # 1. Optional sampling
    if sample_size and sample_size < len(texts):
        random.seed(seed)
        combined = list(zip(texts, truths))
        random.shuffle(combined)
        texts, truths = zip(*combined[:sample_size])
        texts, truths = list(texts), list(truths)
        print(f"Sampled {sample_size} entries")

    # LOGGING
    meta.sample_size = len(texts)

    # 2. Inference & Timing
    s_infer = time.perf_counter()
    spans = run_refined_batch(texts, model, batch_size)
    infer_duration = time.perf_counter() - s_infer

    # save confidence scores for PR curve analysis
    if save_confidence:
        save_confidence_scores(spans, truths, dataset_name, prediction_mode, model_name)

    # 3. Resource Metrics (VRAM & Speed)
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

    perf_data = {
        "vram": peak_vram,
        "throughput": len(texts) / infer_duration,
        "time": infer_duration
    }

    # speed, throughput, vram, batch info
    throughput = perf_data["throughput"]
    print(f"Inference time for {len(texts)} rows: {infer_duration:.2f}s")
    print(f"Throughput: {throughput:.2f} texts/sec")
    print(f"Peak VRAM usage: {peak_vram:.2f} GB")
    print(f"Batch size: {batch_size}")
    print(f"[{datetime.now():%H:%M:%S}] Finished running: '{dataset_name}'")

    # 4. Accuracy & Logging
    metrics = measure_accuracy(spans, truths, verbose)

    # logging
    if log:
        log_results_to_csv(model_name,  dataset_name, prediction_mode, batch_size, metrics, perf_data, meta)


def run_eval(
        model: object,
        model_name: str,
        dataset_name: str,
        batch_size: int,
        prediction_mode: str ="cell",
        sample_size: int = None,
        seed: int =42,
        log: bool = False,
        save_confidence: bool = False
):
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{dataset_name}' ...")

    # 1. Load configuration and data
    folder, gt_df = load_dataset_config(dataset_name)
    table_to_truths = {t: df for t, df in gt_df.groupby("table")}

    # 2. Build the text samples (Context-aware)
    texts, truths = build_eval_samples(table_to_truths, folder, prediction_mode)

    # 3. Prepare the metadata suitcase for logging
    meta = get_dataset_metadata(gt_df, len(texts))

    # 4. Run the model and log results
    run_refined_eval(
        model, texts, truths, batch_size, dataset_name,
        prediction_mode, model_name, meta,
        save_confidence, log, False, # <- verbose
        sample_size, seed
    )


if __name__ == "__main__":
    # Settings
    BATCH_SIZES = [8, 16, 32, 64]
    MODES = ["cell"] + ["row"]
    DATASETS = [
        # Specialized datasets
        ("companies", "special"),
        ("movies", "special"),
        ("SN", "special"),

        # SemTab datasets
        ("Round1_T2D", "generic"),
        ("Round3_2019", "generic"),
        ("Round4_2020", "generic"),
        ("2T_Round4", "generic"),
        ("HTR1", "generic"),
        ("HTR2", "generic"),
        ("HardTablesR2", "generic"),
        ("HardTablesR3", "generic")
    ]


    sample_size = 1000
    seed = 42
    save_conf = False
    logging = False
    device = "gpu"
    model_name = "wikipedia_model_with_numbers"
    refined_model = load_model(device=device, entity_set="wikidata", model=model_name, use_precomputed=False)

    # loop + environment
    print_environment_info(device=device, batch=True, batch_size=None)
    print(bolden(f"\nLogging {green_info('ENABLED') if logging else red_info('DISABLED')} ")+
          bolden(f"| Confidence Saving {green_info('ENABLED') if save_conf else red_info('DISABLED')}"))

    # === PREDICTION MODE SELECTION ===
    for mode in MODES:
        if logging: add_log_divider(f"STARTING MODE: {mode.upper()}", model_name)


        # === BATCH SIZE SELECTION ===
        for batch in BATCH_SIZES:
            if logging: add_log_divider(f"Mode: {mode} | Batch Size: {batch}", model_name)
            print(bolden(f"\n\n\n{'=' * 20} BATCH SIZE: {batch} {'=' * 20}\n"))


            # === DATASET SELECTION ===
            for i, (name, _) in enumerate(DATASETS, 1):
                print(bolden(f"\n\n\n{'#' * 15} [ {i}/{len(DATASETS)}: {name} | {mode.upper()} ] {'#' * 15}"))

                # unified call for both specialized and SemTab
                run_eval(
                    model=refined_model,
                    model_name=model_name,
                    dataset_name=name,
                    batch_size=batch,
                    prediction_mode=mode,
                    sample_size=sample_size,
                    seed = seed,
                    log=logging,
                    save_confidence=save_conf       #TODO turn on after tuning runs
                )