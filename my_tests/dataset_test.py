import pandas as pd
import glob
import time
from pathlib import Path


from my_tests.utility.test_utils import load_model, run_refined_batch, bolden
from my_tests.accuracy import measure_accuracy

data_folder = "my_tests/data"


def build_eval_samples(
        table_to_truths: dict,
        tables_folder: str,
        prediction_mode: str ="cell"
):
    s_build = time.perf_counter()
    texts, truths = [], []

    # collect texts and truths
    for table_file in glob.glob(f"{tables_folder}/*.csv"):
        table_name = Path(table_file).stem

        if table_name not in table_to_truths:
            continue

        # gets (current) data-table
        df = pd.read_csv(table_file)
        current = table_to_truths[table_name]

        # select only relevant cells
        for _, row in current.iterrows():
            row_idx = int(row["row"]) - 1
            col_idx = int(row["col"])

            row_vals = df.iloc[row_idx].astype(str).tolist()
            cell_text = row_vals[col_idx]

            # cell-level prediction
            if prediction_mode == "cell":
                text = cell_text

            # row-level prediction
            elif prediction_mode == "row":
                text = f"{cell_text} | {' | '.join(row_vals[:col_idx] + row_vals[col_idx+1:])}"

            # handle multiple QIDs if needed
            qids = [q.split("/")[-1] for q in row["qid"].split()]

            texts.append(text)
            truths.append(qids)

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
    HARDTABLE_FILES = {
        "HardTablesR2": "HardTable_CEA_WD_Round2_Targets.csv",
        "HardTablesR3": "HardTablesR3_CEA_WD_Round3_Targets.csv",
    }
    dataset = f"{data_folder}/datasets/{eval_set}"

    # dataset locations
    targets_file = f"{dataset}/targets/{HARDTABLE_FILES[eval_set]}"
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


if __name__ == "__main__":

    model = "wikipedia_model_with_numbers"
    refined_model = load_model(device="gpu", entity_set="wikidata", model=model)

    # ---- HTR Evaluation (1) ----
    print(bolden(f"\n\n{'#'*15} [ HTR1 ] {'#'*15}"))
    eval_htr(
        model=refined_model,
        eval_set="HTR1",
        batch_size=64,
        prediction_mode="cell",
        verbose=False,
    )

    # ---- HTR Evaluation (2) ----
    print(bolden(f"\n\n{'#'*15} [ HTR2 ] {'#'*15}"))
    eval_htr(
        model=refined_model,
        eval_set="HTR2",
        batch_size=64,
        prediction_mode="cell",
        verbose=False
    )

    # ---- 2T Evaluation ----
    print(bolden(f"\n\n{'#'*15} [ 2T_Round4 ] {'#'*15}"))
    eval_2t(
        model=refined_model,
        eval_set="2T_Round4",
        batch_size=64,
        prediction_mode="cell",
        verbose=False,
        sample_size=100000
    )

    # ---- HardTables Round 2 Evaluation ----
    print(bolden(f"\n\n{'#'*15} [ HardTablesR2 ] {'#'*15}"))
    eval_hardtables(
        model=refined_model,
        eval_set="HardTablesR2",
        batch_size=64,
        prediction_mode="cell",
        verbose=False,
        sample_size=100000
    )

    # ---- HardTables Round 3 Evaluation ----
    print(bolden(f"\n\n{'#'*15} [ HardTablesR3 ] {'#'*15}"))
    eval_hardtables(
        model=refined_model,
        eval_set="HardTablesR3",
        batch_size=64,
        prediction_mode="cell",
        verbose=False,
        sample_size=100000
    )