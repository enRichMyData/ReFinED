import json
import pandas as pd
import glob
from my_tests.utility.test_utils import load_model, run_refined_batch
from my_tests.accuracy import measure_accuracy


# ---- CONFIG ----
DATA_FOLDER = "my_tests/data/EL_challenge"
EVAL_SET = "HTR1"
TABLES_FOLDER = f"{DATA_FOLDER}/{EVAL_SET}/tables"
CELL_TO_QID_FILE = f"{DATA_FOLDER}/{EVAL_SET}/cell_to_qid.json"
MODEL = "fine_tuned_models/merged_full/f1_0.8972"
DEVICE = "gpu"  # or "cpu"


# ---- LOAD MODEL ----
refined_model = load_model(device=DEVICE, model=MODEL)


# ---- LOAD GOLD DATA ----
with open(CELL_TO_QID_FILE, "r") as f:
    cell_to_qid = json.load(f)


# ---- COLLECT TEXTS AND TRUTHS ----
texts = []
truths = []

for table_file in glob.glob(f"{TABLES_FOLDER}/*.csv"):
    table_name = table_file.split("/")[-1].split(".")[0]
    df = pd.read_csv(table_file)

    # Iterate over cells present in gold
    if table_name not in cell_to_qid:
        continue

    for cell_id, qid in cell_to_qid[table_name].items():
        row_idx, col_idx = map(int, cell_id.split("-"))
        try:
            text = str(df.iat[row_idx, col_idx])
        except IndexError:
            continue
        texts.append(text)
        truths.append((row_idx, col_idx, [qid]))


# ---- RUN MODEL INFERENCE (batch for speed) ----
print(f"Running inference on {len(texts)} cells...")
all_spans = run_refined_batch(texts, refined_model, batch_size=32)


# ---- COMPARE ----
measure_accuracy(all_spans, truths, verbose=True)
