from my_tests.datasets import load_dataset_config, clean_qids
from my_tests.utility.test_utils import load_model, run_refined_batch
from pathlib import Path
import pandas as pd
import random
import csv


# CONFIG
# ================================================================
TARGET_DATASETS = ["HTR1", "companies", "movies", "Round1_T2D"]
SAMPLE_SIZE = 200
SEED = 42
BATCH_SIZE = 32
OUTPUT_FILE = "my_tests/logs/error_analysis.csv"
USE_EXISTING_CSV = True
# ================================================================


def run_error_analysis(model, dataset_name, mode="cell", sample_size=SAMPLE_SIZE):
    """
    Runs inference on a small sample and logs detailed per-prediction info
    including input text, predicted QID, ground truth QID, and confidence.
    """
    print(f"\nAnalysing: {dataset_name} [{mode}]")

    # gets info for specific dataset
    folder, gt_df = load_dataset_config(dataset_name)
    table_to_truths = {t: df for t, df in gt_df.groupby("table")}

    # build samples with table/row/col
    texts, truths, metadata = build_eval_samples_with_meta(
        table_to_truths, folder, mode
    )

    # sample
    if sample_size and sample_size < len(texts):
        random.seed(SEED)
        combined = list(zip(texts, truths, metadata))
        random.shuffle(combined)
        texts, truths, metadata = zip(*combined[:sample_size])
        texts, truths, metadata = list(texts), list(truths), list(metadata)

    # run inference
    spans = run_refined_batch(texts, model, BATCH_SIZE)

    # extract detailed results
    results = []
    for i, (span_list, truth_qids, meta) in enumerate(zip(spans, truths, metadata)):
        if not truth_qids:
            continue

        pred_qid = "NIL"
        confidence = 0.0
        if span_list:
            span = span_list[0]
            pred_qid = getattr(
                span.predicted_entity, "wikidata_entity_id", "NIL"
            ) or "NIL"
            confidence = float(
                getattr(span, "entity_linking_model_confidence_score", 0.0) or 0.0
            )

        is_correct = pred_qid in truth_qids or (
            pred_qid == "NIL" and "NIL" in truth_qids
        )

        results.append({
            "dataset": dataset_name,
            "mode": mode,
            "input_text": texts[i],
            "cell_value": meta["cell_value"],
            "pred_qid": pred_qid,
            "truth_qids": "|".join(truth_qids),
            "confidence": round(confidence, 4),
            "is_correct": int(is_correct),
            "table": meta["table"],
            "row": meta["row"],
            "col": meta["col"],
        })

    return results


def build_eval_samples_with_meta(table_to_truths, tables_folder, prediction_mode):
    """
    Extended version of build_eval_samples that also returns metadata
    (table, row, col, cell_value) for each sample.
    """
    import glob
    texts, truths, metadata = [], [], []

    gt_lookup = {
        str(k).replace(".csv", ""): df
        for k, df in table_to_truths.items()
    }
    gt_ids = sorted(gt_lookup.keys(), key=len, reverse=True)

    for table_path in glob.glob(f"{tables_folder}/*.csv"):
        full_stem = Path(table_path).stem
        matched_id = None

        if full_stem in gt_lookup:
            matched_id = full_stem
        else:
            for gt_id in gt_ids:
                if full_stem.startswith(gt_id):
                    matched_id = gt_id
                    break

        if not matched_id:
            continue

        import pandas as pd
        df_table = pd.read_csv(table_path, header=None).astype(str)
        rows = df_table.values.tolist()
        golds = gt_lookup[matched_id].copy()
        golds["qid_list"] = golds["qid"].apply(clean_qids)

        for gold_row in golds.itertuples(index=False):
            try:
                r = int(gold_row.row)
                c = int(gold_row.col)
                if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
                    cell = rows[r][c]
                    if prediction_mode == "cell":
                        text = cell
                    else:
                        context = rows[r][:c] + rows[r][c + 1:]
                        text = f"{cell} | {' | '.join(context)}"

                    texts.append(text)
                    truths.append(gold_row.qid_list)
                    metadata.append({
                        "cell_value": cell,
                        "table": matched_id,
                        "row": r,
                        "col": c
                    })
            except (ValueError, IndexError):
                continue

    return texts, truths, metadata


def extract_interesting_cases(results):
    """
    Filters results into categories for qualitative analysis.
    Returns dict with categorised cases.
    """
    categories = {
        # Category A: High confidence but wrong (model committed to wrong entity)
        "high_conf_wrong": [
            r for r in results
            if r["confidence"] >= 0.9 and r["is_correct"] == 0
            and r["pred_qid"] != "NIL"
        ],
        # Category B candidate: correct in cell mode, wrong in row mode
        # (requires pairing cell/row results - handled separately)

        # Category C: Model predicted entity but truth is NIL or vice versa
        "nil_mismatch": [
            r for r in results
            if (r["pred_qid"] == "NIL" and "NIL" not in r["truth_qids"])
            or (r["pred_qid"] != "NIL" and r["truth_qids"] == "NIL")
        ],
        # Category D: Model predicted NIL on non-NIL ground truth
        # (potential knowledge cutoff failures)
        "nil_on_known": [
            r for r in results
            if r["pred_qid"] == "NIL"
            and "NIL" not in r["truth_qids"]
        ],
        # Low confidence correct (model was uncertain but right)
        "low_conf_correct": [
            r for r in results
            if r["confidence"] <= 0.3 and r["is_correct"] == 1
        ],
    }
    return categories


def find_semantic_drift(cell_results, row_results):
    """
    Category B: Find cases where cell mode was correct but row mode was wrong.
    Requires running both modes on the same samples.
    """
    drift_cases = []
    for cell, row in zip(cell_results, row_results):
        if cell["cell_value"] == row["cell_value"] and cell["is_correct"] == 1 and row["is_correct"] == 0:
            drift_cases.append({
                "cell_value": cell["cell_value"],
                "input_cell": cell["input_text"],
                "input_row": row["input_text"],
                "pred_cell": cell["pred_qid"],
                "pred_row": row["pred_qid"],
                "truth": cell["truth_qids"],
                "conf_cell": cell["confidence"],
                "conf_row": row["confidence"],
                "dataset": cell["dataset"],
            })
    return drift_cases


def save_results(all_results, output_file=OUTPUT_FILE):
    """Saves all detailed results to CSV."""
    if not all_results:
        print("No results to save.")
        return

    keys = all_results[0].keys()
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} rows to {output_file}")


def print_category_summary(categories, drift_cases):
    """Prints a summary of interesting cases found."""
    print("\n" + "="*50)
    print("ERROR ANALYSIS SUMMARY")
    print("="*50)
    print(f"High confidence wrong:  {len(categories['high_conf_wrong'])}")
    print(f"NIL on known entity:    {len(categories['nil_on_known'])}")
    print(f"NIL mismatch:           {len(categories['nil_mismatch'])}")
    print(f"Low confidence correct: {len(categories['low_conf_correct'])}")
    print(f"Semantic drift cases:   {len(drift_cases)}")
    print("="*50)

    # print top 5 examples per category
    for cat_name, cases in categories.items():
        if cases:
            print(f"\n--- Top 5: {cat_name} ---")
            for c in cases[:5]:
                print(f"  Input:    {c['input_text'][:60]}")
                print(f"  Pred:     {c['pred_qid']}")
                print(f"  Truth:    {c['truth_qids']}")
                print(f"  Conf:     {c['confidence']}")
                print()

    if drift_cases:
        print("\n--- Top 5: Semantic Drift ---")
        for c in drift_cases[:5]:
            print(f"  Cell:     {c['input_cell'][:60]}")
            print(f"  Row:      {c['input_row'][:60]}")
            print(f"  Pred cell:{c['pred_cell']}")
            print(f"  Pred row: {c['pred_row']}")
            print(f"  Truth:    {c['truth']}")
            print()


if __name__ == "__main__":

    # ── Read existing results ──
    if USE_EXISTING_CSV:
        print(f"[INFO] Reading existing results from {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded {len(df)} rows across {df['dataset'].nunique()} datasets")
        all_results = df.to_dict("records")
        all_results_cell = [r for r in all_results if r["mode"] == "cell"]
        all_results_row  = [r for r in all_results if r["mode"] == "row"]

    # ── Run new inference ──
    else:
        model = load_model(
            device="gpu",
            entity_set="wikidata",
            model="wikipedia_model_with_numbers",
            use_precomputed=False
        )
        all_results_cell = []
        all_results_row  = []
        for dataset_name in TARGET_DATASETS:
            cell_results = run_error_analysis(model, dataset_name, mode="cell")
            row_results  = run_error_analysis(model, dataset_name, mode="row")
            all_results_cell.extend(cell_results)
            all_results_row.extend(row_results)
        all_results = all_results_cell + all_results_row
        save_results(all_results)

    # ── Analysis on above choice ──
    categories = extract_interesting_cases(all_results)
    drift_cases = find_semantic_drift(all_results_cell, all_results_row)
    print_category_summary(categories, drift_cases)
    print("\nDone.")