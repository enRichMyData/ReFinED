
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
    print(f'\n{text}')

    spans = refined.process_text(text)
    for span in spans:
        print(f"  {span.text} → {getattr(span.predicted_entity, 'wikipedia_entity_title', None)}")

        # Numerical Entity Types & Normalization
        if span.coarse_type != "MENTION":
            print(f"    Type: {span.coarse_type}")
        if span.coarse_type == "DATE":
            print(f"    Normalized date: {span.date}")

        # Knowledge Graphs
        qid = getattr(span.predicted_entity, 'wikidata_entity_id', None)
        print(f"    ({span.text}) linked to Wikidata QID: {qid}")

        # Top-k candidates
        if span.candidate_entities:
            print(" Candidate entities:")
            qids = [qid for qid, _ in span.candidate_entities[:5]]
            labels = fetch_wikidata_labels(qids)

            for candidate_qid, score in span.candidate_entities[:5]:
                candidate_name = labels.get(candidate_qid, candidate_qid)
                print(f"  - {candidate_name} (QID: {candidate_qid}), score: {score * 100:.2f}%")

        print("\n")
print("\n\n")