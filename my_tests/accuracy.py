from my_tests.refined_utils import parse_args, load_input_file, load_model, run_refined, bcolors

import time
import sys
import torch


def measure_accuracy(all_spans, truths, LINE_LIMIT, verbose=False):
    """Measure accuracy between predicted entities and actual truth values"""

    # contains predicted qids
    predicted_qids = [
        (i, getattr(doc_spans[0].predicted_entity, "wikidata_entity_id", None)
        if doc_spans and doc_spans[0].predicted_entity else None)
        for i, doc_spans in enumerate(all_spans[:LINE_LIMIT])
    ]

    # ground truth qids with rowid
    gt_qids = [(rowid, qid) for (rowid, title, qid) in truths[:LINE_LIMIT]]

    total = min(len(predicted_qids), len(gt_qids))
    correct_count = 0

    # compares QIDs
    for (idp, predicted_qid), (idt, truth_qids) in zip(predicted_qids, gt_qids):

        # check if predicted_qid is in the list of truth qids
        match = (predicted_qid in truth_qids) and (idp == idt)
        if match:
            correct_count += 1

        if verbose:
            print(f"[{idp}/{idt}] "
                  f"Predicted: {predicted_qid}, "
                  f"Truth: {truth_qids}, "
                  f"Match: {bcolors.OKCYAN if match else bcolors.FAIL}{match}{bcolors.ENDC}")

    # color-coded message
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    if accuracy >= 50.00: color = bcolors.OKGREEN
    else: color = bcolors.FAIL
    print(color + bcolors.BOLD + f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total} correct)"+bcolors.ENDC)


def main():
    # ======== CONFIG === ========
    USE_CPU = False         # using cpu or gpu
    LINE_LIMIT = None          # number of lines to process, None for no limit
    FORMAT = "CSV"          # what type of file for GT
    DEFAULT_DATA_FOLDER = "my_tests/data"   # location of data-files


    # ======= Command-line parsing =======
    input_file, verbose = parse_args()

    # ======= Load CSV and truths =======
    try: texts, truths = load_input_file(input_file, DEFAULT_DATA_FOLDER, FORMAT)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # ======= Load model =======
    refined_model = load_model(USE_CPU=USE_CPU)

    # Retrieve texts from running ReFinED entity3. linker
    texts = texts[:LINE_LIMIT]
    truths = truths[:LINE_LIMIT]

    # ======= Run inference =======
    start_time = time.time()
    all_spans = run_refined(texts=texts, model=refined_model)
    duration = time.time() - start_time
    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")

    # ======= Run measurements =======
    measure_accuracy(all_spans, truths, LINE_LIMIT, verbose=verbose)

    # ============ CPU SWITCH ======================= #
    print("\nCUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")
    # =============================================== #
    print(f"Truth-values retrieved using {FORMAT}")


if __name__ == "__main__":
    main()