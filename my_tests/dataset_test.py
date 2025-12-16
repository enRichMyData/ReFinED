import pandas as pd
import glob
import time
from pathlib import Path
from datetime import datetime

from my_tests.utility.test_utils import load_model, run_refined_batch, bolden
from my_tests.accuracy import measure_accuracy


data_folder = "my_tests/data"


def build_eval_samples(table_to_truths: dict, tables_folder: str, prediction_mode: str = "cell"):
    import time
    s_build = time.perf_counter()
    texts, truths = [], []

    for table_file in glob.glob(f"{tables_folder}/*.csv"):
        table_name = Path(table_file).stem

        if table_name not in table_to_truths:
            continue

        df = pd.read_csv(table_file).astype(str)
        table_rows = df.values.tolist()
        golds = table_to_truths[table_name].copy()
        golds["qid_list"] = golds["qid"].apply(lambda s: [x.split("/")[-1] for x in s.split()])

        for _, gold_row in golds.iterrows():
            r = int(gold_row["row"]) - 1
            c = int(gold_row["col"])

            row_vals = table_rows[r]
            cell_text = row_vals[c]

            if prediction_mode == "cell":
                text = cell_text
            else:
                text = f"{cell_text} | {' | '.join(row_vals[:c] + row_vals[c+1:])}"

            texts.append(text)
            truths.append(gold_row["qid_list"])

    build_duration = time.perf_counter() - s_build
    print(f"{prediction_mode}-level prediction mode")
    print(f"Prepared {len(texts)} rows in time: {build_duration:.2f}s")
    return texts, truths


def run_refined_eval(
        texts: list,
        truths: list,
        model: str,
        batch_size: int,
        verbose: bool =True,
        sample_size: int = None
):
    # option sampling for debug
    if sample_size is not None and sample_size < len(texts):
        import random
        s_sample = time.perf_counter()
        random.seed(42)
        combined = list(zip(texts, truths))
        random.shuffle(combined)
        combined = combined[:sample_size]
        texts, truths = map(list, zip(*combined))
        texts, truths = list(texts), list(truths)
        print(f"Sampled {sample_size} entries in time {time.perf_counter() - s_sample:.4f}s")

    # runs and times inference
    s_infer = time.perf_counter()
    spans = run_refined_batch(texts, model, batch_size)
    infer_duration = time.perf_counter() - s_infer

    # calls accuracy measurement
    print(f"Inference time for {len(texts)} rows (batch {batch_size}) in time {infer_duration:.2f}s")
    measure_accuracy(spans, truths, verbose)



def eval_htr(
        model: str,
        eval_set: str = "HTR1", # or "HTR2"
        batch_size: int = 512,
        prediction_mode: str = "cell",
        verbose: bool = True
):
    """HTR Evaluation (1 or 2)"""
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{eval_set}' ...")

    # dataset locations
    dataset = f"{data_folder}/EL_challenge/{eval_set}"
    tables_folder = f"{dataset}/tables"
    gt_file = f"{dataset}/gt/cea_gt.csv"

    # load gold data, group by table
    gt = pd.read_csv(gt_file, header=None, names=["table","row","col","qid"])
    table_to_truths = {t: df for t, df in gt.groupby("table")}

    # build eval samples
    texts, truths = build_eval_samples(table_to_truths, tables_folder, prediction_mode)

    # run evaluation
    run_refined_eval(texts, truths, model, batch_size, verbose)

def eval_2t(
        model: str,
        eval_set: str = "2T_Round4",
        batch_size: int = 512,
        prediction_mode: str = "cell",
        verbose: bool = True,
        sample_size: int = None
):
    """2T Round 4 Evaluation"""
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{eval_set}' ...")

    # dataset locations
    dataset = f"{data_folder}/datasets/{eval_set}"
    targets_file = f"{dataset}/targets/CEA_2T_WD_Targets.csv"
    gt_file = f"{dataset}/gt/cea.csv"
    tables_folder = f"{dataset}/tables"

    # load gold data, merge with targets
    targets = pd.read_csv(targets_file, header=None, names=["table","row","col"])
    gt = pd.read_csv(gt_file, header=None, names=["table","row","col","qid"])
    merged = targets.merge(gt)

    # group by table for lookup
    table_to_truths = {t: df for t, df in merged.groupby("table")}

    # build eval samples
    texts, truths = build_eval_samples(table_to_truths, tables_folder, prediction_mode)

    # run evaluation
    run_refined_eval(texts, truths, model, batch_size, verbose, sample_size)

def eval_hardtables(
        model: str,
        eval_set: str = "HardTablesR2", # or "HardTablesR3"
        batch_size: int = 512,
        prediction_mode: str = "cell",
        verbose: bool = True,
        sample_size: int = None
):
    """HardTablesR2 Evaluation"""
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{eval_set}' ...")

    HARDTABLE_FILES = {
        "HardTablesR2": ("HardTable_CEA_WD_Round2_Targets.csv", "cea.csv"),
        "HardTablesR3": ("HardTablesR3_CEA_WD_Round3_Targets.csv", "cea.csv"),
        "Round1_T2D": ("CEA_Round1_Targets.csv", "CEA_Round1_gt_WD.csv"),
        "Round3_2019": ("CEA_Round3_Targets.csv", "CEA_Round3_gt_WD.csv"),
        "Round4_2020": ("CEA_Round4_Targets.csv", "cea.csv"),
    }
    dataset = f"{data_folder}/datasets/{eval_set}"

    # dataset locations
    targets_file = f"{dataset}/targets/{HARDTABLE_FILES[eval_set][0]}"
    gt_file = f"{dataset}/gt/{HARDTABLE_FILES[eval_set][1]}"
    tables_folder = f"{dataset}/tables"

    # load gold data, merge with targets
    targets = pd.read_csv(targets_file, header=None, names=["table","row","col"])
    gt = pd.read_csv(gt_file, header=None, names=["table","row","col","qid"])
    merged = targets.merge(gt)

    # group by table for lookup
    table_to_truths = {t: df for t, df in merged.groupby("table")}

    # build eval samples
    texts, truths = build_eval_samples(table_to_truths, tables_folder, prediction_mode)

    # run evaluation
    run_refined_eval(texts, truths, model, batch_size, verbose, sample_size)


def run_eval_generic(
    model,
    dataset_name: str,
    dataset_type: str = "HTR",  # "HTR", "2T", "other"
    batch_size: int = 512,
    prediction_mode: str = "cell",
    verbose: bool = True,
    sample_size: int = None
):
    print(f"[{datetime.now():%H:%M:%S}] Starting processing: '{dataset_name}' ...")

    if dataset_type == "HTR":
        dataset = f"{data_folder}/EL_challenge/{dataset_name}"
        tables_folder = f"{dataset}/tables"
        gt_file = f"{dataset}/gt/cea_gt.csv"
        gt = pd.read_csv(gt_file, header=None, names=["table","row","col","qid"])
        table_to_truths = {t: df for t, df in gt.groupby("table")}

    else:
        # 2T or HardTables
        dataset = f"{data_folder}/datasets/{dataset_name}"
        if dataset_type == "2T":
            targets_file = f"{dataset}/targets/CEA_2T_WD_Targets.csv"
            gt_file = f"{dataset}/gt/cea.csv"

        elif dataset_type == "other":
            HARDTABLE_FILES = {
                "HardTablesR2": ("HardTable_CEA_WD_Round2_Targets.csv", "cea.csv"),
                "HardTablesR3": ("HardTablesR3_CEA_WD_Round3_Targets.csv", "cea.csv"),
                "Round1_T2D": ("CEA_Round1_Targets.csv", "CEA_Round1_gt_WD.csv"),
                "Round3_2019": ("CEA_Round3_Targets.csv", "CEA_Round3_gt_WD.csv"),
                "Round4_2020": ("CEA_Round4_Targets.csv", "cea.csv"),
            }
            targets_file = f"{dataset}/targets/{HARDTABLE_FILES[dataset_name][0]}"
            gt_file = f"{dataset}/gt/{HARDTABLE_FILES[dataset_name][1]}"

        tables_folder = f"{dataset}/tables"
        targets = pd.read_csv(targets_file, header=None, names=["table","row","col"])
        gt = pd.read_csv(gt_file, header=None, names=["table","row","col","qid"])
        merged = targets.merge(gt)
        table_to_truths = {t: df for t, df in merged.groupby("table")}

    # Build samples and run evaluation
    texts, truths = build_eval_samples(table_to_truths, tables_folder, prediction_mode)

    # Optionally sample
    if sample_size is not None and sample_size < len(texts):
        import random
        combined = list(zip(texts, truths))
        random.seed(42)
        random.shuffle(combined)
        combined = combined[:sample_size]
        texts, truths = map(list, zip(*combined))
        texts, truths = list(texts), list(truths)
        print(f"Sampled {sample_size} entries")

    s_infer = time.perf_counter()
    spans = run_refined_batch(texts, model, batch_size)
    print(f"Inference time for {len(texts)} rows: {time.perf_counter() - s_infer:.2f}s")
    measure_accuracy(spans, truths, verbose)


if __name__ == "__main__":

    sample_size = 1000

    model = "wikipedia_model_with_numbers"
    refined_model = load_model(device="gpu", entity_set="wikidata", model=model, use_precomputed=False)

    for batch in [8]:
    # for batch in [64, 128, 256, 512]:

        # HTR1
        print(bolden(f"\n\n{'#'*15} [ HTR1 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="HTR1",
            dataset_type="HTR",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # HTR2
        print(bolden(f"\n\n{'#'*15} [ HTR2 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="HTR2",
            dataset_type="HTR",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # 2T_Round4
        print(bolden(f"\n\n{'#'*15} [ 2T_Round4 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="2T_Round4",
            dataset_type="2T",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # HardTablesR2
        print(bolden(f"\n\n{'#'*15} [ HardTablesR2 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="HardTablesR2",
            dataset_type="other",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # HardTablesR3
        print(bolden(f"\n\n{'#'*15} [ HardTablesR3 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="HardTablesR3",
            dataset_type="other",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        #TODO FEIL
        # Round1_T2D
        print(bolden(f"\n\n{'#'*15} [ Round1_T2D ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="Round1_T2D",
            dataset_type="other",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # Round3_2019
        print(bolden(f"\n\n{'#'*15} [ Round3_2019 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="Round3_2019",
            dataset_type="other",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )

        # Round4_2020
        print(bolden(f"\n\n{'#'*15} [ Round4_2020 ] {'#'*15}"))
        run_eval_generic(
            model=refined_model,
            dataset_name="Round4_2020",
            dataset_type="other",
            batch_size=batch,
            prediction_mode="cell",
            verbose=False,
            sample_size=sample_size
        )



        ##################################### old

        # # ---- HTR Evaluation (1) ----
        # print(bolden(f"\n\n{'#'*15} [ HTR1 ] {'#'*15}"))
        # eval_htr(
        #     model=refined_model,
        #     eval_set="HTR1",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        # )
        #
        # # ---- HTR Evaluation (2) ----
        # print(bolden(f"\n\n{'#'*15} [ HTR2 ] {'#'*15}"))
        # eval_htr(
        #     model=refined_model,
        #     eval_set="HTR2",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        # )
        #
        # # ---- 2T Evaluation ----
        # print(bolden(f"\n\n{'#'*15} [ 2T_Round4 ] {'#'*15}"))
        # eval_2t(
        #     model=refined_model,
        #     eval_set="2T_Round4",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )
        #
        # # ---- HardTables Round 2 Evaluation ----
        # print(bolden(f"\n\n{'#'*15} [ HardTablesR2 ] {'#'*15}"))
        # eval_hardtables(
        #     model=refined_model,
        #     eval_set="HardTablesR2",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )
        #
        # # ---- HardTables Round 3 Evaluation ----
        # print(bolden(f"\n\n{'#'*15} [ HardTablesR3 ] {'#'*15}"))
        # eval_hardtables(
        #     model=refined_model,
        #     eval_set="HardTablesR3",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )
        #
        # #TODO: somethings must be wrong with this one
        # # ---- Round1 T2D Evaluation ----
        # print(bolden(f"\n\n{'#' * 15} [ Round1_T2D ] {'#' * 15}"))
        # eval_hardtables(
        #     model=refined_model,
        #     eval_set="Round1_T2D",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )
        #
        # # ---- Round3 2019 Evaluation ----
        # print(bolden(f"\n\n{'#' * 15} [ Round3_2019 ] {'#' * 15}"))
        # eval_hardtables(
        #     model=refined_model,
        #     eval_set="Round3_2019",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )
        #
        # # ---- Round4 2020 Evaluation ----
        # print(bolden(f"\n\n{'#' * 15} [ Round4_2020 ] {'#' * 15}"))
        # eval_hardtables(
        #     model=refined_model,
        #     eval_set="Round4_2020",
        #     batch_size=batch,
        #     prediction_mode="cell",
        #     verbose=False,
        #     sample_size=None
        # )

