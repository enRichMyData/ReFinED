from my_tests.accuracy import measure_accuracy
from my_tests.benchmark import print_environment_info
from my_tests.utility.test_utils import (
    load_model, run_refined_batch,       # model
    bolden,                              # style
    log_results_to_csv, add_log_divider, # logging
)
# from my_tests.utility.data_loader import (
#     load_dataset_config, build_eval_samples, get_dataset_metadata
# )


from my_tests.utility.test_utils import DatasetMetadata
from pathlib import Path
import pandas as pd
import torch.cuda
import random
import glob
import time

DATA_FOLDER = "my_tests/data"

DATASET_FILES = {
    "2T_Round4": ("CEA_2T_WD_Targets.csv", "cea.csv"),
    "HardTablesR2": ("HardTable_CEA_WD_Round2_Targets.csv", "cea.csv"),
    "HardTablesR3": ("HardTablesR3_CEA_WD_Round3_Targets.csv", "cea.csv"),
    "Round1_T2D": ("CEA_Round1_Targets.csv", "CEA_Round1_gt_WD.csv"),
    "Round3_2019": ("CEA_Round3_Targets.csv", "CEA_Round3_gt_WD.csv"),
    "Round4_2020": ("CEA_Round4_targets.csv", "cea.csv"),
}


# Dat loading
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
          f"Time: {build_duration:.2f}s")

    return texts, truths

# ===========================================================


# Evaluation & Logging
# ===========================================================
def run_refined_eval(
    model, texts, truths, batch_size, dataset_name,
    prediction_mode, model_name, meta,
    verbose=False, sample_size=None, seed=42
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

    # speed, throughput
    throughput = perf_data["throughput"]
    print(f"Inference time for {len(texts)} rows: {infer_duration:.2f}s")
    print(f"Throughput: {throughput:.2f} texts/sec")
    print(f"Peak VRAM usage: {peak_vram:.2f} GB")

    # 4. Accuracy & Logging
    metrics = measure_accuracy(spans, truths, verbose)

    # logging
    log_results_to_csv(
        model_name=model_name,
        dataset_name=dataset_name,
        mode=prediction_mode,
        batch_size=batch_size,
        metrics=metrics,
        performance=perf_data,
        meta=meta
    )


def run_eval(
        model,
        dataset_name,
        batch_size,
        prediction_mode,
        sample_size
):

    # 1. Load configuration and data
    folder, gt_df = load_dataset_config(dataset_name)
    table_to_truths = {t: df for t, df in gt_df.groupby("table")}

    # 2. Build the text samples (Context-aware)
    texts, truths = build_eval_samples(table_to_truths, folder, prediction_mode)

    # 3. Prepare the metadata suitcase for logging
    meta = get_dataset_metadata(gt_df, len(texts))

    # 4. Run the model and log results
    run_refined_eval(
        model, texts, truths, batch_size,
        dataset_name=dataset_name,
        prediction_mode=prediction_mode,
        model_name="wikipedia_model_with_numbers",
        meta=meta,
        sample_size=sample_size,
        verbose=False
    )


if __name__ == "__main__":
    # Settings
    BATCH_SIZES = [8]
    MODES = ["cell", "row"]
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
    device = "gpu"
    model_path = "wikipedia_model_with_numbers"
    refined_model = load_model(device=device, entity_set="wikidata", model=model_path, use_precomputed=False)

    for mode in MODES:
        add_log_divider(f"STARTING MODE: {mode.upper()}")

        for batch in BATCH_SIZES:
            add_log_divider(f"Mode: {mode} | Batch Size: {batch}")
            print_environment_info(device=device, batch=True, batch_size=batch)

            for i, (name, _) in enumerate(DATASETS, 1):
                print(bolden(f"\n\n{'#' * 15} [ {i}/{len(DATASETS)}: {name} | {mode.upper()} ] {'#' * 15}"))

                # unified call for both specialized and SemTab
                run_eval(
                    model=refined_model,
                    dataset_name=name,
                    batch_size=batch,
                    prediction_mode=mode,
                    sample_size=sample_size
                )

#                 # OLD EVAL CALLS
#                 # ==== Specialized Datasets ====
#
#                 # Companies
#                 print(bolden(f"\n\n{'#' * 15} [ Companies ] {'#' * 15}"))
#                 run_eval_specialized(
#                     model=refined_model,
#                     dataset_name="companies",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # NOTE !
#                 # 'movies' dataset is HIGHLY affected by 'cell' vs 'row' prediction, with 'row' giving MUCH better result
#
#                 # Movies
#                 print(bolden(f"\n\n{'#' * 15} [ Movies ] {'#' * 15}"))
#                 run_eval_specialized(
#                     model=refined_model,
#                     dataset_name="movies",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # Spend Network (SN)
#                 print(bolden(f"\n\n{'#'*15} [ Spend Network (SN) ] {'#'*15}"))
#                 run_eval_specialized(
#                     model=refined_model,
#                     dataset_name="SN",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # === SemTab Datasets ===
#
#                 # Round1_T2D
#                 print(bolden(f"\n\n{'#'*15} [ Round1_T2D ] {'#'*15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="Round1_T2D",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # Round3_2019
#                 print(bolden(f"\n\n{'#'*15} [ Round3_2019 ] {'#'*15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="Round3_2019",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # Round4_2020
#                 print(bolden(f"\n\n{'#'*15} [ Round4_2020 ] {'#'*15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="Round4_2020",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # 2T_Round4
#                 print(bolden(f"\n\n{'#' * 15} [ 2T_Round4 ] {'#' * 15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="2T_Round4",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # HTR1
#                 print(bolden(f"\n\n{'#' * 15} [ HTR1 ] {'#' * 15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="HTR1",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # HTR2
#                 print(bolden(f"\n\n{'#' * 15} [ HTR2 ] {'#' * 15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="HTR2",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # HardTablesR2
#                 print(bolden(f"\n\n{'#' * 15} [ HardTablesR2 ] {'#' * 15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="HardTablesR2",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )
#
#                 # HardTablesR3
#                 print(bolden(f"\n\n{'#' * 15} [ HardTablesR3 ] {'#' * 15}"))
#                 run_eval_generic(
#                     model=refined_model,
#                     dataset_name="HardTablesR3",
#                     batch_size=batch,
#                     prediction_mode=prediction_mode,
#                     verbose=False,
#                     sample_size=sample_size
#                 )