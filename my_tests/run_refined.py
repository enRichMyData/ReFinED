from my_tests.utility.test_utils import (
    load_model,
    run_refined_single,
    run_refined_batch,
)
from my_tests.utility.testing_args import parse_args                    # CLI
from utility.process_files import load_input_file                       # input-handling
import sys
import torch


# --------------------------------------------------------------------------
# Fetches candidate descriptions (via Wikidata API)
# --------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "ReFinED-EntityLinker/1.0 (https://github.com/#USERNAME#; #MAIL)" # <-- put in to use
}

def fetch_wikidata_labels(qids):
    import requests
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "format": "json",
        "props": "labels",
        "languages": "en",
    }
    response = requests.get(url, params=params, headers=HEADERS)
    data = response.json()
    labels = {}
    for qid in qids:
        entity = data.get("entities", {}).get(qid, {})
        label = entity.get("labels", {}).get("en", {}).get("value", qid)
        labels[qid] = label
    return labels


# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------

def main():
    LINE_LIMIT = 50
    DEFAULT_DATA_FOLDER = "my_tests/data"

    # ======= Command-line parsing =======
    args = parse_args()


    # ======= Load input file =======
    texts, truths = load_input_file(args.input_file, DEFAULT_DATA_FOLDER, args.format)

    # shorten input - for testing
    texts = texts[:LINE_LIMIT]


    # ======= Load model =======
    refined_model = load_model(device=args.device)


    # ======= Run inference =======
    if args.batch:
        all_spans = run_refined_batch(texts=texts, model=refined_model, batch_size=args.batch_size)
    else:
        all_spans = run_refined_single(texts=texts, model=refined_model)


    # ======= Process each input line =======
    for raw_line, doc_spans in zip(texts, all_spans):
        print("\n" + "=" * 100)
        print(f"{raw_line}\n")

        for span in doc_spans[:1]:
            pred_ent = getattr(span, "predicted_entity", None)
            if pred_ent is None:
                print(f"  Mention: [{span.text}] — No predicted entity.")
                continue

            pred_qid = getattr(pred_ent, "wikidata_entity_id", None)
            pred_title = getattr(pred_ent, "wikipedia_entity_title", None)

            print(f"  Mention: [{span.text}]")
            print(f"  → Predicted entity: [{pred_title} (QID: {pred_qid})]")
            print(f"    Type: {span.coarse_type}")

            # Candidate score
            confidence = None
            if span.candidate_entities:
                for qid, score in span.candidate_entities:
                    if qid == pred_qid:
                        confidence = score
                        break
                if confidence is not None:
                    print(f"    Candidate retrieval score: {confidence * 100:.2f}%")
                else:
                    print("    Candidate retrieval score: N/A")
            else:
                print("    No candidate entities.")

            if span.coarse_type == "DATE":
                print(f"    Normalized date: {span.date}")

            # Top-k candidates
            if span.candidate_entities:
                print("    Top-k candidate entities:")
                qids = [qid for qid, _ in span.candidate_entities[:5]]
                labels = fetch_wikidata_labels(qids)

                for candidate_qid, score in span.candidate_entities[:5]:
                    candidate_name = labels.get(candidate_qid, candidate_qid)
                    print(f"      - {candidate_name} (QID: {candidate_qid}), score: {score * 100:.2f}%")

            print("-" * 60)

    # ======= Hardware info =======
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nCUDA available? {torch.cuda.is_available()}")
    print(f"Running on: {gpu_name}\n")

    # NOTE!
    # For Top-K candidates:
    # - span.predicted_entity is the _final_ entity the model chooses after considering full context. This is the most correct link.
    # - span.candidate_entities are raw retrieval scores based on lexical or embedding similarity, not the final conidence scores.


if __name__ == "__main__":
    main()
