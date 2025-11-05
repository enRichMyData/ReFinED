from refined.inference.processor import Refined
import torch


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ------------------ coloring ------------------
def bolden(text: str):
    """Wraps text in bold style."""
    return bcolors.BOLD + text + bcolors.ENDC

def green_info(text: str):
    """Wraps text in a green bold info style."""
    return bcolors.OKGREEN + bcolors.BOLD + text + bcolors.ENDC

def cyan_info(text: str):
    """Wraps text in a cyan bold info style."""
    return bcolors.OKCYAN + bcolors.BOLD + text + bcolors.ENDC

def blue_info(text: str):
    """Wraps text in a blue bold info style."""
    return bcolors.OKBLUE + bcolors.BOLD + text + bcolors.ENDC
#-----------------------------------------------



def load_model(device=False, model="wikipedia_model_with_numbers", entity_set="wikidata"):
    """
    Loads ReFinED pre-trained model.
    Now includes use of CPU / GPU
    """
    # ==== model options so far ==== (must have downloaded!)
    # model = "wikipedia_model_with_numbers"
    # model = "fine_tuned_models/merged_10k/f1_0.9229" # fine tuned med ~8% av treningsdata (5k hver)
    # model = "fine_tuned_models/merged_60k/f1_0.9254" # fine tuned med ~44% av treningsdata (30k hver)
    # model = "fine_tuned_models/companies_full/f1_0.8711" # fine tuned med 100% av treningsdata fra companies.csv
    # model = "fine_tuned_models/movies_full/f1_0.9237" # fine tuned med 100% av treningsdata fra companies.csv
    # model = "fine_tuned_models/merged_full/f1_0.8972" # fine tuned med 100% av treningsdata (fra begge)

    print(green_info(f"[INFO] Loading ReFinED model: ") 
          + cyan_info(f"'{model}'") 
          + green_info(", entity set: ") 
          + cyan_info(f"'{entity_set}'"))


    # GPU/CPU selection
    device = "cuda" if device == "gpu" else "cpu"
    if device == "cpu":
        import warnings
        warnings.filterwarnings("ignore", message="In CPU autocast, but the target dtype is not supported.*")
        original_autocast = torch.amp.autocast
        torch.amp.autocast = lambda *args, **kwargs: original_autocast(device_type="cpu", dtype=torch.float32)

    # ReFinED: Loading model
    return Refined.from_pretrained(
        model_name=model,
        entity_set=entity_set,
        use_precomputed_descriptions=False,
        download_files=True, # optional, downloads from S3 to local
        device=device  #    <--------- Decides to use 'cpu'  or 'gpu'
    )

def run_refined_single(texts, model, unused_arg=None):
    """Process a list of texts through ReFinED."""
    return [model.process_text(t) for t in texts]

def run_refined_batch(texts, model, batch_size: int = 16):
    """
    Process a list of texts through ReFinED using batch processing.
    """
    docs = model.process_text_batch(
        texts,
        max_batch_size=batch_size, # <--- TWEAK FOR BETTER PERFORMANCE {16, 32, 64, 128}
        prune_ner_types=True,
        return_special_spans=True
    )

    # list of all spans from batched processing
    all_spans = [doc.spans for doc in docs]
    return all_spans