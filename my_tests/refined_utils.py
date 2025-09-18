from my_tests.process_file import process_csv

from refined.inference.processor import Refined

import os
import torch
import argparse
import pandas as pd

class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ---------------- Command-line & file loader ----------------
def parse_args(supported_files=None):
    """
    Parses command line arguments for input CSV and optional --verbose.
    supported_files: list of filenames to display in help.
    """
    if supported_files is None:
        supported_files = ['imdb_top_100.csv', 'companies_test.csv', 'movies_test.csv', 'SN_test.csv']

    parser = argparse.ArgumentParser(
        description="Run ReFinED entity linking and measure accuracy.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed prediction info"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input CSV file. Supported:\n" +
             "\n".join([f"\t-'{f}'" for f in supported_files])
    )

    args = parser.parse_args()
    return args.input_file, args.verbose

def get_truth_path(folder: str, default_data: str, filename: str):
    """Return path to truth file if it exists, else None (with logging)."""
    path = os.path.join(default_data, folder, filename)
    if not os.path.exists(path):
        print(bcolors.FAIL + f"Ground truth file not found: '{path}'" + bcolors.ENDC)
        return None
    else:
        print(bcolors.OKGREEN + bcolors.BOLD + f"[INFO] Using truth-file: '{filename}'" + bcolors.ENDC)
        return path


def extract_truths_csv(folder: str, default_data: str):
    """Load corresponding ground truth using [CSV]"""
    path = get_truth_path(folder, default_data, f"el_{folder}_gt_wikidata.csv")
    if not path:
        return None
    df = pd.read_csv(path)
    df = df[df["tableName"] == f"{folder}_test"]
    return [(row.idRow, None, [row.entity.split("/")[-1]]) for _, row in df.iterrows()]  # (id, None, [qid])


def extract_truths_json(folder: str, default_data: str):
    """Load corresponding ground truth using JSON"""
    path = get_truth_path(folder, default_data, f"{folder}_mention_to_qids.json")
    if not path:
        return None

    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build a list of (id, title, first_qid) for each mention
    return  [(i, title, qids if qids else None) for i, (title, qids) in enumerate(data.items())]
    # (id, title, [qid1, qid2, ...])


def resolve_input_path(filename: str, default_data: str):
    """Return (folder, path) where file is located, raising if not found."""
    folder = filename.split("_")[0]
    path = os.path.join(folder, filename)
    if not os.path.exists(filename):
        path = os.path.join(default_data, folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(bcolors.FAIL + f"File not found: '{path}'" + bcolors.ENDC)
    print(bcolors.OKGREEN + bcolors.BOLD + f"[INFO] Using input file: '{path}'" + bcolors.ENDC)
    return folder, path


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


def load_model(USE_CPU=False):
    """
    Loads ReFinED pre-trained model.
    Now includes use of CPU / GPU
    """
    print(bcolors.OKGREEN + bcolors.BOLD + "[INFO] Loading ReFinED model..." + bcolors.ENDC)

    device = "cpu" if USE_CPU else "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        original_autocast = torch.amp.autocast
        torch.amp.autocast = lambda *args, **kwargs: original_autocast(device_type="cpu", dtype=torch.float32)

    return Refined.from_pretrained(
        model_name='wikipedia_model_with_numbers',
        entity_set="wikipedia",
        device=device  #    <--------- Decides to use 'cpu'  or 'gpu'
    )

def run_refined(texts, model):
    """Process a list of texts through ReFinED."""
    return [model.process_text(t) for t in texts]