import requests
import time
import json

BASE_URL = "http://localhost:8002"

# TEST FROM GEMINI!

def test_multipart_flow():
    print("1. Creating Multipart Job...")
    create_payload = {
        "header": ["name", "country"],
        "link_columns": ["name"],
        "mode": "multipart",
        "total_parts": 2,
        "total_rows": 4,
        "top_k": 3
    }

    # Step 1: Create Job
    resp = requests.post(f"{BASE_URL}/jobs", json=create_payload)
    job_data = resp.json()
    job_id = job_data["job_id"]
    print(f"   Job Created: {job_id}")

    # Step 2: Upload Part 1 (2 rows)
    print("2. Uploading Part 1...")
    part1 = {
        "part_number": 1,
        "rows": [
            {"cells": ["Steve Jobs", "USA"]},
            {"cells": ["Elon Musk", "South Africa"]}
        ]
    }
    requests.post(f"{BASE_URL}/jobs/{job_id}/parts", json=part1)

    # Step 3: Upload Part 2 (2 rows)
    print("3. Uploading Part 2...")
    part2 = {
        "part_number": 2,
        "rows": [
            {"cells": ["Angela Merkel", "Germany"]},
            {"cells": ["Ada Lovelace", "UK"]},
            {"cells": ["Marie Curie", "Poland"]}
        ]
    }
    requests.post(f"{BASE_URL}/jobs/{job_id}/parts", json=part2)

    # Step 4: Finalize
    print("4. Finalizing Job...")
    requests.post(f"{BASE_URL}/jobs/{job_id}/finalize")

    # Step 5: Poll Status until Done
    print("5. Polling for Completion...")
    while True:
        status_resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
        data = status_resp.json()
        status = data["status"]
        received = data["ingest"].get("received_rows", 0)

        print(f"   Status: {status} | Rows Received: {received}")

        if status == "done":
            break
        elif status == "failed":
            print("Job Failed!")
            return
        time.sleep(1)

    # Step 6: Get Results
    print("6. Fetching Results...")
    results_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/results")
    results = results_resp.json()

    print("\n--- SAMPLE RESULTS ---")
    for res in results["results"][:3]:
        print(f"Mention: {res['mention']} -> QID: {res['candidate_ranking'][0]['id']}")


if __name__ == "__main__":
    test_multipart_flow()