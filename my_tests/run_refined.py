from my_tests.refined_utils import parse_args, load_input_file, load_model, run_refined_single, run_refined_batch

import sys
import torch


#############################################################################################
# Fetches candidate descriptions (ONLINE)
import requests

HEADERS = {
    "User-Agent": "ReFinED-EntityLinker/1.0 (https://github.com/borgebj; borgebj@ifi.uio.no)"
}

def fetch_wikidata_labels(qids):
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


# NOTE Bytt ut dette kallet med fungerende LamAPI kall !
#############################################################################################


def main():
    # ======== CONFIG === ========
    USE_CPU = False         # using cpu or gpu
    BATCH = False           # using batched or not
    LINE_LIMIT = 50            # lines to process, None for no limit
    FORMAT = "JSON"          # what type of file for GT
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

    # Restricts number of texts to process
    texts = texts[:LINE_LIMIT]

    # ======= Run inference =======
    if BATCH: all_spans = run_refined_single(texts=texts, model=refined_model)
    else: all_spans = run_refined_batch(texts=texts, model=refined_model)

    # Process each input line
    for raw_line, doc_spans in zip(texts, all_spans):
        print("\n" + "=" * 100)
        print(f"{raw_line}\n")

        for span in doc_spans[:1]:
            pred_ent = span.predicted_entity
            pred_qid = getattr(pred_ent, "wikidata_entity_id", None)
            pred_title = getattr(pred_ent, "wikipedia_entity_title", None)

            print(f"  Mention: [{span.text}]")
            print(f"  → Predicted entity: [{pred_title} (QID: {pred_qid})]")
            print(f"    Type: {span.coarse_type}")

            # Confidence for predicted entity
            if span.candidate_entities:
                confidence = None
                for qid, score in span.candidate_entities:
                    if qid == pred_qid:
                        confidence = score
                        break
                if confidence is not None:
                    print(f"    Model confidence: {confidence * 100:.2f}%")
                else:
                    print("    Model confidence: N/A")
            else:
                print("    No candidate entities.")

            # Normalized DATE
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


    # ============ CPU/GPU INFO ===================== #
    print("CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available() and not USE_CPU:
        print("Running on GPU:", torch.cuda.get_device_name(0) + "\n")
    else:
        print("Running on CPU\n")
    # =============================================== #

    # NOTE!
    # For Top-K candidates:
    # - span.predicted_entity is the _final_ entity the model chooses after considering full context. This is the most correct link.
    # - span.candidate_entities are raw retrieval scores based on lexical or embedding similarity, not the final conidence scores.


if __name__ == "__main__":
    main()