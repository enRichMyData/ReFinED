import pandas as pd
import json

from my_tests.utility.test_utils import (
    load_model,
    run_refined_single,
    run_refined_batch,
)

# DATA PREP
# ===========================================

# Initialize ReFinED model
model = load_model(device="gpu")

# Loads csv
df = pd.read_csv("my_tests/API/test.csv")

# target column to link
target_column = "company"
top_k = 3

# Koala-style JSON
table_json = {
    "header": df.columns.tolist(),
    "rows": [],
    "classified_columns": {},
    "column_types": {},
    "status": "DONE"
}



# DATA PROCESS
# ===========================================

# iterate CSV rows
for idx, row in df.iterrows():

    mention = row[target_column]
    context = " ".join([str(row[col]) for col in df.columns if col != target_column])



    # doc_spans is List[List[span]]
    doc_spans_per_doc = run_refined_single([mention], model)[0]  # first (and only) doc

    print(f"\n=====================")
    print(f"Row {idx}")
    print(f"Mention {mention}")
    print(f"context {context}")
    print(f"spans {doc_spans_per_doc}")
    print(f"candidates {len(doc_spans_per_doc[0].candidate_entities) if doc_spans_per_doc and doc_spans_per_doc[0].candidate_entities else 0}")

    candidates_json = []

    for span in doc_spans_per_doc:
        span_text = span.text
        predicted_entity = span.predicted_entity
        predicted_qid = getattr(predicted_entity, "wikidata_entity_id", None) if predicted_entity else None

        # Determine span type from predicted entity if available, else coarse type
        span_type = span.coarse_mention_type or span.coarse_type or "UNKNOWN"

        # Use candidates if available
        if span.candidate_entities:
            for candidate, score in span.candidate_entities[:top_k]:
                if isinstance(candidate, str):
                    candidate_qid = candidate
                    candidate_name = span_text
                else:
                    candidate_qid = getattr(candidate, "wikidata_entity_id", None) or "null"
                    candidate_name = getattr(candidate, "wikipedia_entity_title", span_text)

                # Only assign type if candidate matches predicted entity
                candidate_type = []
                if predicted_entity and candidate_qid == predicted_qid:
                    candidate_type = [{"id": span_type, "name": span_type}]

                candidates_json.append({
                    "id": candidate_qid,
                    "name": candidate_name,
                    "description": getattr(candidate, "description", ""),
                    "score": float(score),
                    "types": candidate_type,
                    "match": candidate_qid == predicted_qid
                })

        # No candidates → no link
        else:
            candidates_json.append({
                "id": "null",
                "name": span_text,
                "description": "",
                "score": 0.0,
                "types": [{"id": "UNKNOWN", "name": "UNKNOWN"}],
                "match": False
            })

    row_json = {
        "idRow": str(idx),
        "data": row.tolist(),
        "linked_entities": [
            {
                "idColumn": df.columns.get_loc(target_column),
                "candidates": candidates_json
            }
        ]
    }

    table_json["rows"].append(row_json)

print(json.dumps(table_json, indent=2))