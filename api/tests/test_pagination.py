import requests


def seed_data():
    url = "http://localhost:8002/jobs"
    print("🚀 Seeding 15 movies for live browser testing...")

    # Create 12 quick jobs to trigger pagination
    for i in range(15):
        requests.post("http://localhost:8002/jobs", json={
            "table_name": f"Test_Dataset_{i}",
            "target_column": "Title",
            "data": [{"Title": f"Movie {i}"}],
            "top_k": 5
        })

    print("Done. Check browser Api /datasets now.")


if __name__ == "__main__":
    seed_data()