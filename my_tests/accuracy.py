from my_tests.process_file import process_csv
from refined.inference.processor import Refined
import torch
import time

def load_model(USE_CPU=False):
    """Loads ReFinED pre-trained model."""
    print("[INFO] Loading ReFinED model...")

    device = "cpu" if USE_CPU else "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        original_autocast = torch.amp.autocast
        torch.amp.autocast = lambda *args, **kwargs: original_autocast(device_type="cpu", dtype=torch.float32)

    return Refined.from_pretrained(
        model_name='wikipedia_model_with_numbers',
        entity_set="wikipedia",
        device=device  #    <--------- Decides to use CPU or GPU
    )

def run_refined(texts, model):
    """Process a list of texts through ReFinED."""
    start_time = time.time()
    results = [model.process_text(t) for t in texts]
    duration = time.time() - start_time
    return results, duration


def main():

    # ====== PARAMETERS ======== #
    USE_CPU = False
    NO_LINES = 5
    # =========================== #

    # loads up input text from file
    texts = process_csv("my_tests/data/companies_test.csv")

    # Load model
    refined_model = load_model(USE_CPU=USE_CPU)

    # Retrieve texts from running ReFinED entity3. linker
    texts = texts[:NO_LINES]

    # runs entity linking, times it
    all_spans, duration = run_refined(texts=texts, model=refined_model)

    # go through each of the lines from csv
    for raw_line, doc_spans in zip(texts, all_spans):

        print(raw_line)  # line from CSV
        for span in doc_spans:
            pred_ent = span.predicted_entity
            pred_qid = getattr(pred_ent, "wikidata_entity_id", None)        # gets QID
            pred_title = getattr(pred_ent, "wikipedia_entity_title", None)  # gets title, aka text

            print(f"{span.text} → {pred_title} (QID: {pred_qid})")          # text -> prediction
        print("\n" + "-"*60 + "\n")

    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")


    # ============ CPU SWITCH =======================
    print("CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")


if __name__ == "__main__":
    main()