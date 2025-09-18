from my_tests.refined_utils import parse_args, load_input_file, load_model, run_refined, bcolors

import time
import sys
import torch


def measure_accuracy(all_spans, truths, NO_LINES, verbose=False):
    """Measure accuracy between predicted entities and actual truth values"""

    # contains predicted qids
    predicted_qids = [getattr(span.predicted_entity, "wikidata_entity_id", None)
                      if span.predicted_entity else None
                      for doc_spans in all_spans
                      for span in [doc_spans[0]]]

    # contains truth qids
    gt_qids = [truth[1] for truth in truths[:NO_LINES]]

    total = min(len(predicted_qids), len(gt_qids))
    correct_count = 0

    for predicted_qid, truth_qid in zip(predicted_qids, gt_qids):

        if verbose:
            print(f"'{predicted_qid}'=='{truth_qid}'")
            print(f"Match: {predicted_qid == truth_qid}\n")

        if predicted_qid == truth_qid:
            correct_count += 1

    # color-coded message
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    if accuracy >= 50.00: color = bcolors.OKGREEN
    else: color = bcolors.FAIL
    print(color + bcolors.BOLD + f"Accuracy: {accuracy:.2f}% ({correct_count}/{total} correct)"+bcolors.ENDC)


def main():
    # ======== CONFIG === ========
    USE_CPU = False         # using cpu or gpu
    NO_LINES = 100         # number of lines to process, None for no limit
    DEFAULT_DATA_FOLDER = "my_tests/data"   # location of data-files


    # ======= Command-line parsing =======
    input_file, verbose = parse_args()

    # ======= Load CSV and truths =======
    try: texts, truths = load_input_file(input_file, DEFAULT_DATA_FOLDER)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # ======= Load model =======
    refined_model = load_model(USE_CPU=USE_CPU)

    # Retrieve texts from running ReFinED entity3. linker
    texts = texts[:NO_LINES]

    # ======= Run inference =======
    start_time = time.time()
    all_spans = run_refined(texts=texts, model=refined_model)
    duration = time.time() - start_time
    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")

    # ======= Run measurements =======
    measure_accuracy(all_spans, truths, NO_LINES, verbose=verbose)

    # ============ CPU SWITCH ======================= #
    print("\nCUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")
    # =============================================== #


if __name__ == "__main__":
    main()