import json
import time

import pandas as pd
import glob
from my_tests.utility.test_utils import load_model, run_refined_batch
from my_tests.accuracy import measure_accuracy


# ---- CONFIG ----
DATA_FOLDER = "my_tests/data/EL_challenge"
EVAL_SET = "HTR1"
# EVAL_SET = "HTR2"
TABLES_FOLDER = f"{DATA_FOLDER}/{EVAL_SET}/tables"
CELL_TO_QID_FILE = f"{DATA_FOLDER}/{EVAL_SET}/cell_to_qid.json"
# MODEL = "fine_tuned_models/merged_full/f1_0.8972" # fine-tuned on movies+companies
MODEL = "wikipedia_model_with_numbers"            # default model
ENTITIY_SET = "wikidata"
DEVICE = "gpu"  # or "cpu"

# Mode:
# - "cell" = single cell prediction (only one word)
# - "row" = row-level prediction (uses row as context)
PREDICTION_MODE = "cell"  # or "row"

# ---- LOAD MODEL ----
refined_model = load_model(device=DEVICE, model=MODEL, entity_set=ENTITIY_SET)


# ---- LOAD GOLD DATA ----
with open(CELL_TO_QID_FILE, "r") as f:
    cell_to_qid = json.load(f)


# ---- COLLECT TEXTS AND TRUTHS ----
texts = []
truths = []


# Iterate all .csv files in /table
for table_file in glob.glob(f"{TABLES_FOLDER}/*.csv"):
    table_name = table_file.split("/")[-1].split(".")[0]  # /table/00X7C4X7.csv -> 00X7C4X7
    df = pd.read_csv(table_file)

    # Iterate over cells present in gold
    if table_name not in cell_to_qid:
        continue

    # gets QIDs for every table
    for cell_id, qid in cell_to_qid[table_name].items():
        row_idx, col_idx = map(int, cell_id.split("-"))  # "0-0","1-0", ..
        try:
            # single-cell prediction
            if PREDICTION_MODE == "cell":
                text = str(df.iat[row_idx, col_idx])

            # row-level prediction (context)
            elif PREDICTION_MODE == "row":
                cell_text = str(df.iat[row_idx, col_idx])
                # include context
                context_cells = [
                    str(df.iat[row_idx, c]) for c in range(df.shape[1]) if c != col_idx
                ]
                context_text = ",".join(context_cells)
                text = f"{cell_text},{context_text}"

        except IndexError:
            continue

        texts.append(text)                          # LEDA 1245565, 0.08102   <-- (col0, col1) 00X7C4X7.csv
        truths.append((row_idx, col_idx, [qid]))    #


# ---- RUN MODEL INFERENCE (batch for speed) ----
print(f"Running inference on {len(texts)} cells...")
start = time.time()
all_spans = run_refined_batch(texts, refined_model, batch_size=64) # 64 (44.59) 256 (44.74) 512 (44.81) 1024 (44.81)
duration = time.time() - start

# ---- COMPARE ----
measure_accuracy(all_spans, truths, verbose=True)

print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")