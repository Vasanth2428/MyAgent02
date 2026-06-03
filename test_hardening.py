import requests
import json
import time

url = "http://127.0.0.1:8000/query"
payload = {
    "question": "Use the web search specialist to find who won the 2024 Super Bowl, then critique the findings, and finally generate a report.",
    "session_id": "test_hardening_01",
    "mode": "agentic"
}
headers = {"Content-Type": "application/json"}

print("Sending query to API...", flush=True)
start_time = time.time()
response = requests.post(url, json=payload, headers=headers)
print(f"Request took {time.time() - start_time:.2f} seconds", flush=True)

try:
    data = response.json()
    print("\nFINAL RESPONSE:")
    print(data.get("response", "No response found"))
    print("\nSTATS:")
    print(json.dumps(data.get("stats", {}), indent=2))
except Exception as e:
    print(f"Error parsing response: {e}")
    print(response.text)
