from my_tests.refined_utils import \
    parse_args, \
    load_input_file, \
    load_model, \
    run_refined_single, \
    run_refined_batch, \
    bcolors

import time
import sys
import torch


def measure_accuracy(all_spans, truths, LINE_LIMIT, verbose=False):
    """Measure accuracy between predicted entities and actual truth values"""

    # extracts rowid, title, and qid from predictions
    predicted_qids = [
        (
            i,  # idRow / document index
            getattr(span[0].predicted_entity, "wikipedia_entity_title", None) if span else None,
            getattr(span[0].predicted_entity, "wikidata_entity_id", None) if span else None
        )
        for i, span in enumerate(all_spans)
    ]

    # ground truth qids with rowid
    gt_qids = truths[:LINE_LIMIT]

    total = min(len(predicted_qids), len(gt_qids))
    correct_count = 0

    # compares QIDs
    for (idp, title, predicted_qid), (idt, truth_qids) in zip(predicted_qids, gt_qids):

        # check if predicted_qid is in the list of truth qids
        match = (predicted_qid in truth_qids) #and (idp == idt)
        if match:
            correct_count += 1

        if verbose:
            print(
                f"[{idp:>2}|{idt:<2}] "
                f"Predicted: '{str(predicted_qid):<12}' "
                f"Truth: {str(truth_qids):<15} "
                f"Match: {bcolors.OKCYAN if match else bcolors.FAIL}{str(match):<6}{bcolors.ENDC} "
                f"({title})")

    # color-coded message
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    if accuracy >= 50.00: color = bcolors.OKGREEN
    else: color = bcolors.FAIL
    print(color + bcolors.BOLD + f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total} correct)"+bcolors.ENDC)


def main():
    # ======== CONFIG === ========
    USE_CPU = False         # using cpu or gpu
    BATCH = False           # using batched or not
    LINE_LIMIT = None          # number of lines to process, None for no limit
    FORMAT = "CSV"          # what type of file for GT
    BATCH_SIZE = 16        # batch size if using batched
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
    if BATCH: all_spans = run_refined_single(texts=texts, model=refined_model)
    else: all_spans = run_refined_batch(texts=texts, model=refined_model, batch_size=BATCH_SIZE)
    duration = time.time() - start_time

    # ======= Run measurements =======
    measure_accuracy(all_spans, truths, LINE_LIMIT, verbose=verbose)

    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")

    # ============ CPU SWITCH ======================= #
    print("\nCUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")
    # =============================================== #
    print(f"Results from file: {input_file}")
    print(f"Truth-values retrieved using {FORMAT}")


if __name__ == "__main__":
    main()