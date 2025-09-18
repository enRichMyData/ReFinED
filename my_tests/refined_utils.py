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

def load_input_file(filename: str, default_data: str):
    """Loads CSV file and corresponding ground truth JSON file from command line or default data folder."""

    # Determine file paths
    folder = filename.split("_")[0]
    file_path = os.path.join(folder, filename)

    if not os.path.exists(filename):
        file_path = os.path.join(default_data, folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(bcolors.FAIL + f"File not found: '{file_path}'" + bcolors.ENDC)

    print(bcolors.OKGREEN + bcolors.BOLD + f"[INFO] Using input file: '{file_path}'" + bcolors.ENDC)

    # Load corresponding ground truth JSON
    json_filename = f"{folder}_mention_to_qids.json"
    json_path = os.path.join(default_data, folder, json_filename)
    if not os.path.exists(json_path):
        print(bcolors.FAIL + f"Ground truth JSON file not found: '{json_path}'" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + bcolors.BOLD + f"[INFO] Using truth-file: '{json_filename}'" + bcolors.ENDC)

    # Load CSV texts and truths
    texts = process_csv(file_path)

    # Extract ground truth tuples: (title, qid)
    if os.path.exists(json_path):
        df = pd.read_json(json_path)
        truths = [(title, df[title][0]) for title in df.columns]
    else: truths = None

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