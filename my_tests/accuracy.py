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
import time
import torch
import datetime
import os


# color codes
colorOK = bcolors.OKCYAN
colorFAIL = bcolors.FAIL


def measure_accuracy(pred_spans, truths, verbose=False):
    total = min(len(pred_spans), len(truths))
    correct = 0

    # compare predicted and ground truth in pairs (pred, gt)
    for i, (pred_span, (row, col, truth_qids)) in enumerate(zip(pred_spans, truths)):
        if not pred_span or not truth_qids: continue

        # only consider first entity, aka company/movie
        primary_span = pred_span[0] if pred_span else None
        pred_qid = getattr(primary_span.predicted_entity, "wikidata_entity_id", None) if pred_span else None

        if pred_qid in truth_qids:
            correct += 1

        if verbose:
            print(
                f"[{i}] Pred: {pred_qid or 'None':<10} "
                f"Truth: {truth_qids} "
                f"Match: {colorOK if pred_qid in truth_qids else colorFAIL}"
            )

    # measure accuracy
    accuracy = (correct / total * 100) if total else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def log_evaluation(DATA_FOLDER, accuracy, metrics, input_file, batch_size, gpu):
    """saves result to file"""

    # Define default log directory
    log_dir = os.path.join(DATA_FOLDER, "logs")
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


def evaluate_refined(refined, input_file, LINE_LIMIT):
    print(bolden("\n=== Running ReFinED Evaluation ==="))
    datasets = get_datasets_obj(preprocessor=refined.preprocessor)

    if "companies" in input_file.lower():
        eval_docs = list(datasets.get_companies_docs(split="test", include_gold_label=True))[:LINE_LIMIT]

    elif "movies" in input_file.lower():
        eval_docs = list(datasets.get_movie_docs(split="test", include_gold_label=True))[:LINE_LIMIT]

    else: raise ValueError(f"Unknown dataset type in input file name: {input_file}")

    # built-in revaluation by refined
    final_metrics = evaluate(
        refined=refined,
        evaluation_dataset_name_to_docs={"EVAL": eval_docs},
        el=False,
        ed=True
    )

    # print results
    print(bolden("\n=== Final Evaluation Results ==="))
    for dataset_name, metrics in final_metrics.items():
        print(f"\nDataset: {dataset_name}")
        print(f"  Precision: {metrics.get_precision():.3f}")
        print(f"  Recall:    {metrics.get_recall():.3f}")
        print(f"  F1 Score:  {metrics.get_f1():.3f}")
        print(f"  Accuracy:  {metrics.get_accuracy():.3f}")

    return final_metrics


def main():
    # ------- CONFIG -------
    LINE_LIMIT = None          # number of lines to process, None for no limit
    TEST_DIR = "my_tests"
    DEFAULT_DATA_FOLDER = f"{TEST_DIR}/data"   # location of data-files


    # ------- Command-line parsing -------
    args = parse_args()
    input_file = args.input_file
    verbose = args.verbose
    batch_size = args.batch_size
    device = args.device
    gt_format = args.format
    batch = args.batch

    # ------- Load CSV and truths -------
    texts, truths = load_input_file(input_file, DEFAULT_DATA_FOLDER, gt_format)


    # ------- Load model -------
    model = "fine_tuned_models/merged_full/f1_0.8972" # fine tuned med 100% av treningsdata (fra begge)
    refined_model = load_model(device=device, model=model)

    # Retrieve texts from running ReFinED entity3. linker
    texts = texts[:LINE_LIMIT]
    truths = truths[:LINE_LIMIT]


    # ------- Run inference -------
    start_time = time.time()
    if batch:
        all_spans = run_refined_batch(texts=texts, model=refined_model, batch_size=batch_size)
    else:
        all_spans = run_refined_single(texts=texts, model=refined_model)
    duration = time.time() - start_time

    print(f"\nInference time for {len(texts)} texts: {duration:.2f} seconds")



    # ------- Run measurements -------
    accuracy = measure_accuracy(pred_spans=all_spans, truths=truths, verbose=verbose)
    # accuracy = measure_accuracy(all_spans, truths, LINE_LIMIT, verbose=verbose)

    # ------- Run official evaluation -------
    metrics = evaluate_refined(refined_model, input_file, LINE_LIMIT)



    # ------- CPU SWITCH ------- #
    print("\nCUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and device == "gpu":
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")

    print(f"Results from file: {input_file}")
    print(f"Truth-values retrieved using {gt_format}")

    # logging results to file
    # log_evaluation(TEST_DIR, accuracy, metrics, input_file, batch_size, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
