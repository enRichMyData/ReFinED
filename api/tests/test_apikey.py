import requests

# Configuration
BASE_URL = "http://localhost:8002"
VALID_KEY = "CORRECT_API_KEY"  # Replace with the actual valid API key from your .env
INVALID_KEY = "wrong-password"

def test_endpoint(name, headers):
    print(f"--- Testing: {name} ---")
    try:
        # Testing the /datasets endpoint as a simple connectivity check
        response = requests.get(f"{BASE_URL}/datasets", headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    # 1. Test with No API Key
    test_endpoint("No API Key", {})

    # 2. Test with Invalid API Key
    test_endpoint("Invalid API Key", {"X-API-Key": INVALID_KEY})

    # 3. Test with Valid API Key
    test_endpoint("Valid API Key", {"X-API-Key": VALID_KEY})