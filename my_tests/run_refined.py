
# testing and entity linking
from my_tests.process_file import process_csv
from refined.inference.processor import Refined


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

# texts = [
#     "England won the FIFA World Cup in 1966.",
#     "Barack Obama was the 44th President of the United States.",
#     "Amazon was founded by Jeff Bezos.",
#     "Oslo is the capital of Norway, neighbor to Sweden.",
#     "Joe Biden was the previous president in the United States, followed by Donald Trump",
#     "Jordan met with Apple in Washington last week."
# ]

texts = process_csv("my_tests/data/imdb_top_100.csv")

refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")


for text in texts[:5]:  # Test first 5 rows
    print("\n" + "=" * 100)
    print(f"{text}\n")

    spans = refined.process_text(text)
    for span in spans:
        pred_ent = span.predicted_entity
        pred_qid = getattr(pred_ent, "wikidata_entity_id", None)
        pred_title = getattr(pred_ent, "wikipedia_entity_title", None)

        print(f"  Mention: [{span.text}]", end="")
        print(f"  →  Predicted entity: [{pred_title} (QID: {pred_qid})]")
        print(f"    Type: {span.coarse_type}")

        # Confidence from candidates
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
            print("    No candidate entities (likely numerical).")

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

        # NOTE!
        # For Top-K candidates: 
        # - span.predicted_entity is the _final_ entity the model chooses after considering full context. This is the most correct link.
        # - span.candidate_entities are raw retrieval scores based on lexical or embedding similarity, not the final conidence scores.
        