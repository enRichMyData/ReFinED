import pandas as pd
import glob
import time
from pathlib import Path

from my_tests.utility.test_utils import load_model, run_refined_batch
from my_tests.accuracy import measure_accuracy


# ---- CONFIG ----
model = "wikipedia_model_with_numbers"
# model = "fine_tuned_models/merged_10k/f1_0.9229"
# model = "fine_tuned_models/merged_60k/f1_0.9254"
# model = "fine_tuned_models/companies_full/f1_0.8711"
# model = "fine_tuned_models/movies_full/f1_0.9237"
# model = "fine_tuned_models/merged_full/f1_0.8972"

# Mode:
# - "cell" = single cell prediction (only one word)
# - "row" = row-level prediction (uses row as context)


def eval_2t(
        model: str,
        eval_set: str = "2T_Round4",
        batch_size: int = 512,
        prediction_mode: str = "cell",
        verbose: bool = True
):
    # path to data and truth labels
    targets_file = f"my_tests/data/datasets/{eval_set}/targets/CEA_2T_WD_Targets.csv"
    gt_file = f"my_tests/data/datasets/{eval_set}/gt/cea.csv"
    tables_folder = f"my_tests/data/datasets/{eval_set}/tables"

    # load targets, truth labels, merge
    targets_df = pd.read_csv(targets_file, header=None, names=["table", "row", "col"])
    gt_df = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])
    eval_df = targets_df.merge(gt_df, on=["table", "row", "col"], how="left")

    # pre-built mapping: table_name -> rows DataFrame
    table_to_truths = {table: df for table, df in eval_df.groupby("table")}

    s1 = time.perf_counter()

    # collect texts and truths
    texts, truths = [], []
    for table_file in glob.glob(f"{tables_folder}/*.csv"):
        table_name = Path(table_file).stem

        if table_name not in table_to_truths:
            continue

        # gets data-table
        df = pd.read_csv(table_file)

        # select only relevant cells
        current_table_truths = table_to_truths[table_name]

        for _, row in current_table_truths.iterrows():
            row_idx = int(row["row"]) - 1
            col_idx = int(row["col"])

            row_vals = df.iloc[row_idx].astype(str).tolist()

            # cell-level prediction
            if prediction_mode == "cell":
                text = row_vals[col_idx]

            # row-level prediction
            elif prediction_mode == "row":
                context_cells = row_vals[:col_idx] + row_vals[col_idx+1:]
                text = row_vals[col_idx] + " " + " ".join(context_cells)

            # handle multiple QIDs if needed
            qids = [q.split("/")[-1] for q in row["qid"].split()]

            texts.append(text)
            truths.append(qids)

    print(f"DEBUG: Prepared {len(texts)} texts and truths in {time.perf_counter() - s1:.4f}s")

    #TODO: DELETE
    # START DEBUG    (sample of 1000 random)
    import random
    s2 = time.perf_counter()
    random.seed(42)
    combined = list(zip(texts, truths))
    random.shuffle(combined)
    combined = combined[:10000]
    texts, truths = map(list, zip(*combined))
    print(f"DEBUG: sampled 500 entries in {time.perf_counter() - s2:.4f}s")
    # END DEBUG

    # ---- RUN MODEL INFERENCE (batch for speed) ----
    print(f"Running inference on {len(texts)} cells...")
    start = time.perf_counter()
    all_spans = run_refined_batch(texts, model, batch_size)
    duration = time.perf_counter() - start

    measure_accuracy(all_spans, truths, verbose=verbose)
    print(f"Inference time for {len(texts)} texts: {duration:.2f} seconds (batch {batch_size})")
    # ============================================


if __name__ == "__main__":

    model = "wikipedia_model_with_numbers"

    refined_model = load_model(device="gpu", entity_set="wikidata", model=model)

    eval_2t(
        model=refined_model,
        eval_set="2T_Round4",
        batch_size=64,
        prediction_mode="cell"
    )

    # TLDR:
    # - big batch = SLOWER
    # - 100k lines on 64 ~ 220s (3.3 min)