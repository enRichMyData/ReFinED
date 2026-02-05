from my_tests.utility.test_utils import (
    load_model,
    run_refined_single,
    run_refined_batch,
    bolden,
    bcolors,
)
from refined.evaluation.evaluation import get_datasets_obj, evaluate    # built-in eval
from my_tests.utility.testing_args import parse_args                    # CLI
from utility.process_files import load_input_file                       # input-handling
from my_tests.utility.test_utils import bcolors
import time
import torch
import datetime
import os

def measure_accuracy(pred_spans, truths, verbose=False):
    total = 0
    tp = 0
    fn = 0
    fp = 0
    # tn is meaningless in EL

    # compare predicted and ground truth in pairs (pred, gold)
    for i, (pred_span, truth_qids) in enumerate(zip(pred_spans, truths)):
        if not truth_qids: continue
        total += 1

        #TODO This assumes only one (main) entity per text!
        # only consider the _first_ entity, i.e. company/movie
        pred_qid = getattr(pred_span[0].predicted_entity, "wikidata_entity_id", None) if pred_span else None

        # true positive and false negative
        if pred_qid in truth_qids:
            tp += 1
        else:
            fn += 1

        # false positives
        if pred_qid is not None and pred_qid not in truth_qids:
            fp += 1

        if verbose:
            truth_display = str(truth_qids)
            if len(truth_display) > 30:
                truth_display = truth_display[:27] + "...]"
            match_color = bcolors.OKCYAN if pred_qid in truth_qids else bcolors.FAIL
            print(
                f"[{i+1:>3}] "
                f"Pred: {pred_qid or 'None':<12} "
                f"Truth: {truth_display:<35} "
                f"{match_color}"
                f"Match: {str(pred_qid in truth_qids):<5}"
                f"{bcolors.ENDC} "
                f"({pred_qid})"
            )


    # calculate metrics
    accuracy = tp / (total + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    rescolor = bcolors.OKGREEN if accuracy > 0.5 else bcolors.FAIL
    print("\n========-EVAL METRICS-=========")
    print(f"{rescolor}", end="")
    print(f"Accuracy:  {accuracy:.4f}   {accuracy*100:.2f}   ({tp}/{total}) {bcolors.ENDC}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    # print(f"No. Gold:  {len(truths)}")
    print("==============================\n")

    return accuracy, precision, recall, f1


def log_evaluation(data_folder, accuracy, metrics, input_file, batch_size, gpu):
    """saves result to file"""

    # Define default log directory
    log_dir = os.path.join(data_folder, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"evaluation_log_{input_file}_{batch_size}_{gpu}.txt")

    print(f"\nSaving log-file: '{log_path}'")

    with open(log_path, "a") as f:
        f.write(f"\n=== Run {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"File: {input_file}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        for dataset_name, metrics in metrics.items():
            f.write(f"{dataset_name}: P={metrics.get_precision():.3f}, R={metrics.get_recall():.3f}, F1={metrics.get_f1():.3f}\n")
        f.write("\n")


def evaluate_refined(refined, input_file):
    print(bolden("\n=== Running ReFinED Evaluation ==="))
    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    if "companies" in input_file.lower():
        eval_docs = list(datasets.get_companies_docs(split="test", include_gold_label=True))

    elif "movies" in input_file.lower():
        eval_docs = list(datasets.get_movie_docs(split="test", include_gold_label=True))

    elif "HTR" in input_file.upper():
        eval_docs = list(datasets.get_hardtable_docs(split=input_file, include_gold_label=True))

    else:
        raise NotImplementedError(f"docs not created for {input_file}")


    # built-in revaluation by refined
    final_metrics = evaluate(
        refined=refined,
        evaluation_dataset_name_to_docs={"EVAL": eval_docs},
        el=True,     # entity linking eval
        ed=False,    # entity disambiguation (optional)
        print_errors=False
    )

    return final_metrics


def main():
    # ------- CONFIG -------
    LINE_LIMIT = 5          # number of lines to process, None for no limit
    TEST_DIR = "my_tests"
    DEFAULT_DATA_FOLDER = f"{TEST_DIR}/data"   # location of data-files


    # ------- Command-line parsing -------
    args = parse_args()


    # ------- Load CSV and truths -------
    texts, truths = load_input_file(args.input_file, DEFAULT_DATA_FOLDER, args.format)

    # shorten input - for testing
    texts = texts[:LINE_LIMIT]
    truths = truths[:LINE_LIMIT]


    # ------- Load model -------
    refined_model = load_model(device=args.device, entity_set=args.entity_set, model=args.model)


    # ------- Run inference -------
    start_time = time.time()
    if args.batch:
        all_spans = run_refined_batch(texts=texts, model=refined_model, batch_size=args.batch_size)
    else:
        all_spans = run_refined_single(texts=texts, model=refined_model)
    duration = time.time() - start_time

    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")


    # ------- Run measurements -------
    measure_accuracy(pred_spans=all_spans, truths=truths, verbose=args.verbose)

    # ------- Run ReFinED evaluation -------
    evaluate_refined(refined_model, args.input_file)



    # ------- CPU SWITCH ------- #
    print("\nCUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and args.device == "gpu":
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")

    print(f"Results from file: {args.input_file}")
    print(f"Truth-values retrieved using {args.format}")

    # logging results to file
    # log_evaluation(TEST_DIR, accuracy, metrics, input_file, batch_size, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
