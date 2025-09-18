from my_tests.process_file import process_csv

from refined.inference.processor import Refined

import os
import sys
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ================== UTILITY FUNCTIONS ==================
def load_input_file(filename: str, default_data: str):
    """Loads CSV file from command line or default data folder."""
    # Gets folder location
    folder = filename.split("_")[0]
    file_path = f"{folder}/{filename}"
    if not os.path.exists(filename):
        filename = os.path.join(default_data, file_path)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
    print(f"[INFO] Using input file: {filename}")
    return process_csv(filename)

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