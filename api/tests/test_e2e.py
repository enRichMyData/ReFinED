import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:8002"
TIMEOUT_LIMIT = 60  # ReFinED can take a few seconds to initialize


# QUICK TEST FROM GEMINI !!

def run_test():
    print("🚀 Starting ReFinED (Crocodile-Compatible) Test...")
    print("-" * 50)

    # 1. CREATE JOB (Updated to match JobCreateRequest)
    print("📦 Step 1: Submitting Table...")
    payload = {
        "header": ["name", "city"],
        "rows": [
            {"name": "Marie Curie", "city": "Paris"},
            {"name": "Albert Einstein", "city": "Berlin"},
            {"name": "Steve Jobs", "city": "Cupertino"}
        ],
        "link_columns": ["name"],  # Must be a list of strings
        "table_name": "Scientist Table Test",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/jobs", json=payload)
    if response.status_code != 202:
        print(f"❌ Error creating job: {response.status_code} - {response.text}")
        return

    job_id = response.json()["job_id"]
    print(f"✅ Job Created! ID: {job_id}")

    # 2. POLL STATUS
    print("\n⏳ Step 2: Polling status (Progress Mapping)...")
    start_time = time.time()

    while True:
        status_resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
        data = status_resp.json()

        status = data["status"]
        # Accessing nested progress info block
        progress = data.get("progress", {})
        current = progress.get("row_index", 0)

        # Accessing nested ingest info block
        ingest = data.get("ingest", {})
        total = ingest.get("expected_rows", 0)

        print(f"   [STATUS]: {status:<10} | Progress: {current}/{total} rows")

        if status == "done":
            print("✨ Job completed successfully!")
            break
        elif status == "failed":
            print(f"❌ Job failed! Error: {data.get('error')}")
            return

        if time.time() - start_time > TIMEOUT_LIMIT:
            print("⏰ Test timed out!")
            return

        time.sleep(1.5)

    # 3. FETCH RESULTS
    print("\n📜 Step 3: Fetching results (Pagination Test)...")
    # Requesting only 2 results to test the cursor logic
    results_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/results?limit=2")

    if results_resp.status_code != 200:
        print(f"❌ Error fetching results: {results_resp.text}")
        return

    results_data = results_resp.json()

    print(f"✅ Received {len(results_data['results'])} results.")
    print(f"🔗 Next Cursor: {results_data.get('next_cursor')}")

    # Print a sample to verify schema
    if results_data["results"]:
        first_res = results_data["results"][0]
        print(
            f"\n🔍 Sample Match: '{first_res['mention']}' -> Linked to: {first_res['candidate_ranking'][0]['name']} ({first_res['candidate_ranking'][0]['id']})")

    print("-" * 50)
    print("🏁 Test Finished!")


if __name__ == "__main__":
    run_test()