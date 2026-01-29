import pandas as pd
import json

from my_tests.utility.test_utils import (
    load_model,
    run_refined_single,
    run_refined_batch,
)


# ===========================================
# dummy model for testing
class DummySpan:
    def __init__(self, text):
        self.text = text
        self.label = f"DummyEntity({text})"
        self.description = f"Description for {text}"
        self.candidate_entities = [(f"Q{1000+i}", 0.9 - i*0.1) for i in range(3)]
        self.types = [type("Type", (), {"id": "Q5", "name": "Thing"})()]
# ===========================================



# DATA PREP
# ===========================================

# Initialize ReFinED model
# model = load_model(device="cpu")

# Loads csv
df = pd.read_csv("my_tests/API/test.csv")

# target column to link
target_column = "company_name"
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

    # Perform EL on mention + context
    # uses single as a test only (change to batch!)
    # spans = run_refined_single([mention + " " + context], model)[0]

    # for testing
    spans = [DummySpan(mention + " " + context)]

    candidates_json = []

    for span in spans[:1]:
        
        # top-k candidates entities
        for qid, score in getattr(span, "candidate_entities", [])[:top_k]:
            candidates_json.append({
                "id": qid,
                "name": getattr(span, "label", qid),   # label with fallback to QID
                "description": getattr(span, "description", ""),
                "score": score,
                "types": [{"id": t.id, "name": t.name} for t in getattr(span, "types", [])],
                "match": True
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