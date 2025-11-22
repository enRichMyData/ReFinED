import pandas as pd
import glob
import time
from pathlib import Path


from my_tests.utility.test_utils import load_model, run_refined_batch
from my_tests.benchmark import print_environment_info
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

def eval_htr(
        model: str,
        eval_set: str = "HTR1",
        batch_size: int = 512,
        prediction_mode: str = "cell",
        all_metrics: bool = True,
        verbose: bool = True
):
    tables_folder = f"my_tests/data/EL_challenge/{eval_set}/tables"
    gt_file = f"my_tests/data/EL_challenge/{eval_set}/gt/cea_gt.csv"

    # load gold data
    gt_df = pd.read_csv(gt_file, header=None, names=["table", "row", "col", "qid"])

    # group by table for lookup
    eval_df = gt_df.groupby("table")

    texts, truths=[], []
    for table_file in glob.glob(f"{tables_folder}/*.csv"):

        # gets data-table
        table_name = Path(table_file).stem
        df = pd.read_csv(table_file)

        if table_name not in set(gt_df["table"].unique()):
            continue

        # get all cells in this table with ground truth
        current_table_truths = gt_df[gt_df["table"] == table_name]

        for _, row in current_table_truths.iterrows():
            row_idx = int(row["row"]) - 1
            col_idx = int(row["col"])

            cell_text = str(df.iat[row_idx, col_idx])
            context_cells = [str(df.iat[row_idx, c]) for c in range(df.shape[1]) if c != col_idx]

            # cell-level prediction
            if prediction_mode == "cell":
                text = cell_text

            # row-level prediction
            elif prediction_mode == "row":
                text = f"{cell_text} {' '.join(context_cells)}"

            # handle multiple QIDs if needed
            qids = [q.split("/")[-1] for q in row["qid"].split()]

            texts.append(text)
            truths.append(qids)

    # ---- RUN MODEL INFERENCE (batch for speed) ----
    print(f"Running inference on {len(texts)} cells...")
    start = time.time()
    all_spans = run_refined_batch(texts, model, batch_size)
    duration = time.time() - start

    measure_accuracy(all_spans, truths, all_metrics=all_metrics, verbose=verbose)
    print(f"Inference time for {len(texts)} texts: {duration:.2f} seconds")

    return all_spans, truths, duration



if __name__ == "__main__":

    model = "wikipedia_model_with_numbers"

    refined_model = load_model(device="gpu", entity_set="wikidata", model=model)

    all_spans, truths, duration = eval_htr(
        model=refined_model,
        eval_set="HTR1",
        batch_size=512,
        prediction_mode="cell"
    )



# OLD FUNCTION USING JSON
# import json
# def eval_el_challenge(
#         model: str,
#         eval_set: str = "HTR1",
#         batch_size: int = 512,
#         prediction_mode: str = "cell",
#         all_metrics: bool = True,
#         verbose: bool = True
# ):
#     # path to data and truth labels
#     tables_folder = f"my_tests/data/EL_challenge/{eval_set}/tables"
#     cell_to_qid_file = f"my_tests/data/EL_challenge/{eval_set}/cell_to_qid.json"
#
#
#     # ---- LOAD GOLD DATA ----
#     with open(cell_to_qid_file, "r") as f:
#         cell_to_qid = json.load(f)
#
#
#     # ---- COLLECT TEXTS AND TRUTHS ----
#     texts = []
#     truths = []
#
#     # Iterate all .csv files in /table
#     for table_file in glob.glob(f"{tables_folder}/*.csv"):
#         table_name = table_file.split("/")[-1].split(".")[0]  # /table/00X7C4X7.csv -> 00X7C4X7
#         df = pd.read_csv(table_file)
#
#         # Iterate over cells present in gold
#         if table_name not in cell_to_qid:
#             continue
#
#         # gets QIDs for every table
#         for cell_id, cell_qids in cell_to_qid[table_name].items():
#             row_idx, col_idx = map(int, cell_id.split("-"))  # "0-0","1-0", ..
#
#             cell_text = str(df.iat[row_idx, col_idx])
#             context_cells = [str(df.iat[row_idx, c]) for c in range(df.shape[1]) if c != col_idx]
#
#             # single-cell prediction
#             if prediction_mode == "cell":
#                 text = cell_text
#
#             # row-level prediction (context)
#             elif prediction_mode == "row":
#                 text = f"{cell_text},{','.join(context_cells)}"
#
#             # handles multiple qids
#             qids = cell_qids.split() if isinstance(cell_qids, str) else list(cell_qids)
#
#             texts.append(text)                          # LEDA 1245565, 0.08102   <-- (col0, col1) 00X7C4X7.csv
#             truths.append(qids)
#
#
#     # ---- RUN MODEL INFERENCE (batch for speed) ----
#     print(f"Running inference on {len(texts)} cells...")
#     start = time.time()
#     all_spans = run_refined_batch(texts, model, batch_size)
#     duration = time.time() - start
#
#     measure_accuracy(all_spans, truths, all_metrics=all_metrics, verbose=verbose)
#     print(f"Inference time for {len(texts)} texts: {duration:.2f} seconds")
#
#     return all_spans, truths, duration