import pandas as pd
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
DATASET_NAME = "test_dataset"
TABLE_NAME = "test_table"
CSV_PATH = "test.csv"
TARGET_COLUMN = "company"
TOP_K = 3


def pretty(obj):
    print(json.dumps(obj, indent=2))


# --------------------------------------------------
# 1. Load CSV and prepare payload
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
table_data = df.to_dict(orient="records")

payload = {
    "data": table_data,
    "target_column": TARGET_COLUMN,
    "top_k": TOP_K,
    "table_name": TABLE_NAME
}

# --------------------------------------------------
# 2. POST table (entity linking)
# --------------------------------------------------
print("\n[POST] Uploading table...")
post_url = f"{BASE_URL}/datasets/{DATASET_NAME}/tables"
response = requests.post(post_url, json=payload)

print("Status code:", response.status_code)
pretty(response.json())

# --------------------------------------------------
# 3. List datasets
# --------------------------------------------------
print("\n[GET] Listing datasets...")
r = requests.get(f"{BASE_URL}/datasets")
pretty(r.json())

# --------------------------------------------------
# 4. List tables in dataset
# --------------------------------------------------
print("\n[GET] Listing tables in dataset...")
r = requests.get(f"{BASE_URL}/datasets/{DATASET_NAME}/tables")
pretty(r.json())

# --------------------------------------------------
# 5. Fetch stored table
# --------------------------------------------------
print("\n[GET] Fetching table...")
r = requests.get(
    f"{BASE_URL}/datasets/{DATASET_NAME}/tables/{TABLE_NAME}"
)
pretty(r.json())

# --------------------------------------------------
# 6. Check table status
# --------------------------------------------------
print("\n[GET] Checking table status...")
r = requests.get(
    f"{BASE_URL}/datasets/{DATASET_NAME}/tables/{TABLE_NAME}/status"
)
pretty(r.json())