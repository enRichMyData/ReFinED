import pandas as pd
import glob
import time
from pathlib import Path
from datetime import datetime

import torch.cuda

from my_tests.utility.test_utils import load_model, run_refined_batch, bolden
from my_tests.accuracy import measure_accuracy
from my_tests.benchmark import print_environment_info


# Lookup for targets, ground truth for different datasets
DATA_FOLDER = "my_tests/data"

DATASET_FILES = {
    "2T_Round4": ("CEA_2T_WD_Targets.csv", "cea.csv"),
    "HardTablesR2": ("HardTable_CEA_WD_Round2_Targets.csv", "cea.csv"),
    "HardTablesR3": ("HardTablesR3_CEA_WD_Round3_Targets.csv", "cea.csv"),
    "Round1_T2D": ("CEA_Round1_Targets.csv", "CEA_Round1_gt_WD.csv"),
    "Round3_2019": ("CEA_Round3_Targets.csv", "CEA_Round3_gt_WD.csv"),
    "Round4_2020": ("CEA_Round4_targets.csv", "cea.csv"),
}


def build_eval_samples(table_to_truths: dict, tables_folder: str, prediction_mode: str = "cell"):
    """
        Builds evaluation samples for both cell-level and row-level prediction modes,
        by iterating through tables, matching them to Ground Truth, and extracting text and QIDs.
    """
    s_build = time.perf_counter()
    texts, truths = [], []

    # pre-process Ground Truth keys for lookup
    gt_lookup = {str(k).replace(".csv", ""): df for k, df in table_to_truths.items()}
    gt_ids = sorted(gt_lookup.keys(), key=len, reverse=True)  # Longest first to avoid partial prefix matches

    # stats
    total_files = 0
    files_processed = 0

    # helper to handle Wikidata QIDs and NILs
    def clean_qids(val):
        if pd.isna(val) or str(val).strip().lower() in ["nan", "nil", "none", ""]:
            return ["NIL"]
        return [x.split("/")[-1] for x in str(val).split()]
    # ------------------------------------------

    # iterate through tables
    for table_path in glob.glob(f"{tables_folder}/*.csv"):
        total_files += 1
        full_stem = Path(table_path).stem

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
        df = pd.read_csv(table_path, header=None).astype(str)
        table_rows = df.values.tolist()
        num_rows = len(table_rows)

        # get relevant Ground Truth rows
        golds = gt_lookup[matched_id].copy()
        golds["qid_list"] = golds["qid"].apply(clean_qids)

        for _, gold_row in golds.iterrows():
            try:
                r = int(gold_row["row"])
                c = int(gold_row["col"])

                # bound check
                if 0 <= r < num_rows:
                    row_vals = table_rows[r]
                    if 0 <= c < len(row_vals):
                        cell_text = row_vals[c]

                        # single / cell prediction
                        if prediction_mode == "cell":
                            text = cell_text

                        # context / row prediction
                        else:
                            context = row_vals[:c] + row_vals[c + 1:]
                            text = f"{cell_text} | {' | '.join(context)}"

                        # after processing:  add to lists
                        texts.append(text)
                        truths.append(gold_row["qid_list"])
            except (ValueError, IndexError):
                continue


    # --- DEBUG PRINT STATEMENT ---
    print(f"Found {total_files:,} files in folder.")
    print(f"Matched {files_processed:,} tables against Ground Truth/Targets.")
    # --------------------------------

    build_duration = time.perf_counter() - s_build
    print(f"{prediction_mode}-level prediction mode")
    print(f"Prepared {len(texts):,} rows in time: {build_duration:.2f}s")
    return texts, truths


def run_refined_eval(
    model,
    texts: list,
    truths: list,
    batch_size: int,
    verbose: bool = True,
    sample_size: int = None,
    seed: int = 42
):
    # optional sampling
    if sample_size is not None and sample_size < len(texts):
        import random
        combined = list(zip(texts, truths))
        random.seed(seed)
        random.shuffle(combined)
        combined = combined[:sample_size]
        texts, truths = map(list, zip(*combined))
        texts, truths = list(texts), list(truths)
        print(f"Sampled {sample_size} entries")

    # inference
    s_infer = time.perf_counter()
    spans = run_refined_batch(texts, model, batch_size)
    infer_duration = time.perf_counter() - s_infer

    # speed, throughput
    throughput = len(texts) / infer_duration
    print(f"Inference time for {len(texts)} rows: {infer_duration:.2f}s")
    print(f"Throughput: {throughput:.2f} texts/sec")

    # VRAM measurement
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM usage: {peak_vram:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    measure_accuracy(spans, truths, verbose)


def run_eval_generic(
    model,
    dataset_name: str,
    batch_size: int = 512,
    prediction_mode: str = "cell",
    verbose: bool = True,
    sample_size: int = None
):
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{dataset_name}' ...")

    # HTR located in another folder
    if "HTR" in dataset_name:
        dataset = f"{DATA_FOLDER}/EL_challenge/{dataset_name}"
        tables_folder = f"{dataset}/tables"
        gt_file = f"{dataset}/gt/cea_gt.csv"
        gt = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])
        merged = gt.copy()
        total_targets = len(gt)
        nil_targets = 0

    # SemTab datasets
    else:
        dataset = f"{DATA_FOLDER}/datasets/{dataset_name}"

        # files and folders
        targets_file = f"{dataset}/targets/{DATASET_FILES[dataset_name][0]}"
        gt_file = f"{dataset}/gt/{DATASET_FILES[dataset_name][1]}"
        tables_folder = f"{dataset}/tables"

        # targets
        if dataset_name in ["Round1_T2D", "Round3_2019"]:
            targets = pd.read_csv(targets_file, header=None, names=["table", "col", "row"])
        else:
            targets = pd.read_csv(targets_file, header=None, names=["table", "row", "col"])

        # ground truth
        gt = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])

        # normalization of types and strip whitespace
        for df in [targets, gt]:
            df["table"] = df["table"].astype(str).str.strip()
            df["row"] = df["row"].astype(int)
            df["col"] = df["col"].astype(int)

        total_targets = len(targets)
        merged = targets.merge(gt, on=["table", "row", "col"], how="left")
        nil_targets = merged["qid"].apply(
            lambda v: pd.isna(v) or str(v).strip().lower() in ["nan", "nil", "none", ""]
        ).sum()

    # --- stats ---
    print(f"Total Targets in CSV: {total_targets:,}")
    print(f"Linkable Targets: {(total_targets - nil_targets):,}")
    print(f"NIL Targets (Expected): {nil_targets:,} ({(nil_targets / total_targets) * 100:.1f}%)")

    table_to_truths = {t: df for t, df in merged.groupby("table")}

    # Build samples
    texts, truths = build_eval_samples(table_to_truths, tables_folder, prediction_mode)

    # --- Evaluation ---
    print(f"[{datetime.now():%H:%M:%S}] Finished processing: '{dataset_name}'")
    run_refined_eval(model, texts, truths, batch_size, verbose, sample_size)


def run_eval_specialized(
        model,
        dataset_name: str,
        batch_size: int = 8,
        prediction_mode: str = "cell",
        verbose: bool = True,
        sample_size: int = None
):
    print(f"[{datetime.now():%H:%M:%S}] Starting Specialized Processing: '{dataset_name}' ...")

    # path setup
    base_path = f"{DATA_FOLDER}/{dataset_name}"

    # map filename based on directory
    if dataset_name == "SN":
        gt_file = f"{base_path}/SN_gt.csv"
        table_file = f"{base_path}/SN_test.csv"
        table_name = "SN_test"

    elif dataset_name == "movies":
        gt_file = f"{base_path}/el_movies_gt_wikidata.csv"
        table_file = f"{base_path}/movies_test.csv"
        table_name = "movies_test"

    elif dataset_name == "companies":
        gt_file = f"{base_path}/el_companies_gt_wikidata.csv"
        table_file = f"{base_path}/companies_test.csv"
        table_name = "companies_test"

    else:
        raise ValueError(f"Unknown specialized dataset: '{dataset_name}'")

    # load ground truth with header, rename to standard column names
    gt = pd.read_csv(gt_file, header=0)
    gt = gt.rename(columns={
        "tableName": "table",
        "idRow": "row",
        "idCol": "col",
        "entity": "qid"
    })

    # filter test split only (movies/companies share GT with train data)
    gt = gt[gt["table"] == table_name].copy()
    gt["row"] = gt["row"].astype(int)
    gt["col"] = gt["col"].astype(int)

    # offset row by 1 due to table CSV having header
    gt["row"] = gt["row"] + 1

    # stats
    total_targets = len(gt)
    nil_targets = gt["qid"].apply(
        lambda v: pd.isna(v) or str(v).strip().lower() in ["nan", "nil", "none", ""]
    ).sum()

    print(f"Total Targets: {total_targets:,}")
    print(f"NIL Targets: {nil_targets:,} ({(nil_targets / total_targets) * 100:.1f}%)")
    print(f"Linkable Targets: {(total_targets - nil_targets):,}")

    # build table_to_truths - only one table in these cases (test csv)
    table_to_truths = {table_name: gt}

    # build samples
    texts, truths = build_eval_samples(table_to_truths, base_path, prediction_mode)

    print(f"[{datetime.now():%H:%M:%S}] Finished processing: '{dataset_name}'")
    run_refined_eval(model, texts, truths, batch_size, verbose, sample_size)


if __name__ == "__main__":

    device = "gpu"
    sample_size = 1000
    modes = ["cell", "row"] # + "row"

    model = "wikipedia_model_with_numbers"
    refined_model = load_model(device=device, entity_set="wikidata", model=model, use_precomputed=False)

    for prediction_mode in modes:
        print(f"\n{'='*30} Running evaluation for prediction mode: '{prediction_mode}' {'='*30}\n")

        for batch in [8]:
        # for batch in [64, 128, 256, 512]:

            # prints environment info
            print_environment_info(device=device, batch=True, batch_size=batch)

            # ==== Specialized Datasets ====

            # Companies
            print(bolden(f"\n\n{'#' * 15} [ Companies ] {'#' * 15}"))
            run_eval_specialized(
                model=refined_model,
                dataset_name="companies",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # NOTE !
            # 'movies' dataset is HIGHLY affected by 'cell' vs 'row' prediction, with 'row' giving MUCH better result

            # Movies
            print(bolden(f"\n\n{'#' * 15} [ Movies ] {'#' * 15}"))
            run_eval_specialized(
                model=refined_model,
                dataset_name="movies",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # Spend Network (SN)
            print(bolden(f"\n\n{'#'*15} [ Spend Network (SN) ] {'#'*15}"))
            run_eval_specialized(
                model=refined_model,
                dataset_name="SN",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # === SemTab Datasets ===

            # Round1_T2D
            print(bolden(f"\n\n{'#'*15} [ Round1_T2D ] {'#'*15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="Round1_T2D",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # Round3_2019
            print(bolden(f"\n\n{'#'*15} [ Round3_2019 ] {'#'*15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="Round3_2019",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # Round4_2020
            print(bolden(f"\n\n{'#'*15} [ Round4_2020 ] {'#'*15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="Round4_2020",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # 2T_Round4
            print(bolden(f"\n\n{'#' * 15} [ 2T_Round4 ] {'#' * 15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="2T_Round4",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # HTR1
            print(bolden(f"\n\n{'#' * 15} [ HTR1 ] {'#' * 15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="HTR1",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # HTR2
            print(bolden(f"\n\n{'#' * 15} [ HTR2 ] {'#' * 15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="HTR2",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # HardTablesR2
            print(bolden(f"\n\n{'#' * 15} [ HardTablesR2 ] {'#' * 15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="HardTablesR2",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )

            # HardTablesR3
            print(bolden(f"\n\n{'#' * 15} [ HardTablesR3 ] {'#' * 15}"))
            run_eval_generic(
                model=refined_model,
                dataset_name="HardTablesR3",
                batch_size=batch,
                prediction_mode=prediction_mode,
                verbose=False,
                sample_size=sample_size
            )