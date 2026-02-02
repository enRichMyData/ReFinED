from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from my_tests.utility.test_utils import load_model, run_refined_single

# Model + API initialization
# ======================================
app = FastAPI()
model = load_model(device="gpu")


# Built-in dataset storage
# ======================================
DATASETS = {}

class TableRequest(BaseModel):
    data: List[dict]
    target_column: str
    top_k: int
    table_name:str = "table1" # optional

@app.post("/datasets/{dataset_name}/tables")
def process_table(dataset_name: str, request: TableRequest):
    df = pd.DataFrame(request.data)
    target_column = request.target_column
    top_k = request.top_k
    table_name = request.table_name

    table_json = {
        "header": df.columns.tolist(),
        "rows": [],
        "classified_columns": {},
        "column_types": {},
        "status": "DONE"
    }

    for idx, row in df.iterrows():

        # extract mention, and (optional) context)
        mention = row[target_column]
        context = " ".join([str(row[col]) for col in df.columns if col != target_column])

        # runs the entity linking (using ReFinED)
        doc_spans_per_doc = run_refined_single([mention], model)[0]

        candidates_json = []
        for span in doc_spans_per_doc:
            span_text = span.text
            predicted_entity = span.predicted_entity
            predicted_qid = getattr(predicted_entity, "wikidata_entity_id", None) if predicted_entity else None
            span_type = span.coarse_mention_type or span.coarse_type or "UNKNOWN"

            # checks if it has candidates
            if span.candidate_entities:
                for candidate, score in span.candidate_entities[:top_k]:
                    if isinstance(candidate, str):
                        candidate_qid = candidate
                        candidate_name = span_text
                    else:
                        candidate_qid = getattr(candidate, "wikidata_entity_id", None) or "null"
                        candidate_name = getattr(candidate, "wikipedia_entity_title", span_text)
                    candidate_type = []

                    if predicted_entity and candidate_qid == predicted_qid:
                        candidate_type = [{"id": span_type, "name": span_type}]

                    # Koala JSON format for candidates
                    candidates_json.append({
                        "id": candidate_qid,
                        "name": candidate_name,
                        "description": getattr(candidate, "description", ""),
                        "score": float(score),
                        "types": candidate_type,
                        "match": candidate_qid == predicted_qid
                    })
            else:
                candidates_json.append({
                    "id": "null",
                    "name": span_text,
                    "description": "",
                    "score": 0.0,
                    "types": [{"id": "UNKNOWN", "name": "UNKNOWN"}],
                    "match": False
                })

        # Koala JSON format for each row
        row_json = {
            "idRow": str(idx),
            "data": row.tolist(),
            "linked_entities": [
                {"idColumn": df.columns.get_loc(target_column), "candidates": candidates_json}
            ]
        }

        table_json["rows"].append(row_json)

    # saves table in built-in memory storage
    if dataset_name not in DATASETS:
        DATASETS[dataset_name] = {}
    DATASETS[dataset_name][table_name] = table_json

    return table_json

# Additional API endpoints
# ======================================
@app.get("/datasets")
def list_datasets():
    return {"datasets": list(DATASETS.keys())}

@app.get("/datasets/{dataset_name}/tables")
def list_tables(dataset_name: str):
    tables = DATASETS.get(dataset_name, {})
    return {"tables": list(tables.keys())}

@app.get("/datasets/{dataset_name}/tables/{table_name}")
def get_table(dataset_name: str, table_name: str):
    dataset = DATASETS.get(dataset_name, {})
    table = dataset.get(table_name)
    if table:
        return table
    return {"error": "Table not found"}, 404

@app.get("/datasets/{dataset_name}/tables/{table_name}/status")
def table_status(dataset_name: str, table_name:str):
    dataset = DATASETS.get(dataset_name, {})
    table = dataset.get(table_name)
    if table:
        return {"status": table.get("status", "UNKNOWN")}
    return {"error": "Table not found"}, 404



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
