from collections import OrderedDict
from my_tests.utility.test_utils import bcolors, green_info, cyan_info
import pandas as pd
import json
import os


def process_csv(file_path):
    df = pd.read_csv(file_path)

    print(bcolors.OKGREEN + bcolors.BOLD + f'[INFO] Loaded {len(df)} rows with columns: \n{list(df.columns)}' + bcolors.ENDC)

    # concatenates entire row into a string, including context
    processed_texts = [
        ", ".join(str(row[col]) for col in df.columns if pd.notna(row[col]))
        for _, row in df.iterrows()
    ]

    return processed_texts

# ---------------- Command-line & file loader ----------------

# [5]
def get_truth_path(folder: str, default_data: str, filename: str):
    """Return path to truth file if it exists, else None (with logging)."""
    path = os.path.join(default_data, folder, filename)
    filetype = filename.split(".")[-1]
    if not os.path.exists(path):
        print(green_info(f"Ground truth file not found: '{path}'"))
        return None
    else:
        print(f"{green_info(f'[INFO] Using truth file: {filename}')} {cyan_info(f'({filetype} truth)')}")
        return path


# [3]
def extract_truths_csv(folder: str, default_data: str):
    """Load corresponding ground truth using [CSV]"""
    # special SN case
    if folder.upper() == "SN":
        filename = f"{folder}_gt.csv"
    else:
        filename = f"el_{folder}_gt_wikidata.csv"

    # resolves path
    path = get_truth_path(folder, default_data, filename)
    if not path: return None

    # reads -> filters to include only '_test' lines
    df = pd.read_csv(path)
    df = df[df["tableName"] == f"{folder}_test"]

    # extracts QID from entity links (https...entity/QID) -> (id, col, [qid])
    truths = [
        (row.idRow, row.idCol, [row.entity.split("/")[-1]] if pd.notna(row.entity) else [])
        for _, row in df.iterrows()
    ]

    print(green_info(f"[INFO] Loaded {len(truths)} entries from {path}"))
    return truths


# [3]
def extract_truths_json(folder: str, default_data: str):
    """Load corresponding ground truth using JSON"""
    path = get_truth_path(folder, default_data, f"{folder}_mention_to_qids.json")
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    print(green_info(f"[INFO] Loaded {len(data)} entries from {path}"))

    # Build a list of (row, col, qids) for each qid in json -> (rowId, rowCol[qid1, qid2, ...])
    return [(i, 0, qids if qids else []) for i, (title, qids) in enumerate(data.items())]


# [2]
def resolve_input_path(filename: str, default_data: str):
    """Return (folder, path) where file is located, raising if not found."""
    folder = filename.split("_")[0]
    filetype = filename.split(".")[-1]
    path = os.path.join(folder, filename)

    if not os.path.exists(filename):
        path = os.path.join(default_data, folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(bcolors.FAIL + f"File not found: '{path}'" + bcolors.ENDC)

    print(f"{green_info(f'[INFO] Using data file: {path}')} {cyan_info(f'({filetype} data)')}")
    return folder, path


# [1]
def load_input_file(filename: str, default_data: str, format: str):
    """Loads CSV file and corresponding ground truth JSON file from command line or default data folder."""
    folder, file_path = resolve_input_path(filename, default_data)

    # chooses truth-file format
    if format.upper() == "JSON":
        truths = extract_truths_json(folder, default_data)
    elif format.upper() == "CSV":
        truths = extract_truths_csv(folder, default_data)
    else:
        raise ValueError(f"Unknown format: {format}")

    texts = process_csv(file_path)
    return texts, truths

if __name__ == "__main__":
    texts = process_csv("data/imdb_top_100.csv")
    print("\n--- First 5 Combined Texts ---")
    for t in texts[:5]:
        print(t)
