#####################################
# LamAPI test
import requests
import json

LAMAPI_BASE = "https://lamapi.hel.sintef.cloud"
LAMAPI_TOKEN = "lamapi_demo_2023"


# Example: POST /entity/bow
def post_lamapi_bow(qids):
    url = f"{LAMAPI_BASE}/entity/bow"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {LAMAPI_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = json.dumps(qids)
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"LamAPI request failed with status code {response.status_code}: {response.text}")


# Example: GET /info
def get_lamapi_info():
    url = f"{LAMAPI_BASE}/info"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {LAMAPI_TOKEN}" 
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"LamAPI request failed with status code {response.status_code}: {response.text}")


# ex 1
info = get_lamapi_info()
print("LamAPI Info:", info)

"""
# ex 2
data = post_lamapi_bow(["Q42", "Q217533", "Q60"])
print("Bag of Words output:")
for qid, bow in data.items():
    print(f"  {qid}: {bow}")
"""
#####################################