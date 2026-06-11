import requests
import json
import time
import sys

if __name__ != "__main__":
    import unittest
    raise unittest.SkipTest("Skipping test_hardening module during collection (run directly via python test_hardening.py)")

url = "http://127.0.0.1:8000/query"
headers = {"Content-Type": "application/json"}

# Turn 1: Kerala CM query
payload1 = {
    "question": "who is the chief minister of kerala",
    "session_id": "test_hardening_session",
    "mode": "agentic"
}

# Turn 2: Ireland PM query
payload2 = {
    "question": "who is the prime minister of ireland",
    "session_id": "test_hardening_session",
    "mode": "agentic"
}

print("--- TURN 1: Sending query 'who is the chief minister of kerala' ---", flush=True)
start_time = time.time()
response1 = requests.post(url, json=payload1, headers=headers)
print(f"Turn 1 took {time.time() - start_time:.2f} seconds", flush=True)
try:
    data1 = response1.json()
    print("Turn 1 Final Answer preview:", data1.get("response", "No response")[:200])
except Exception as e:
    print(f"Error parsing response: {e}\n{response1.text}")

print("\nWaiting 5 seconds for rate limit window to clear slightly...", flush=True)
time.sleep(5)

print("\n--- TURN 2: Sending query 'who is the prime minister of ireland' ---", flush=True)
start_time = time.time()
response2 = requests.post(url, json=payload2, headers=headers)
print(f"Turn 2 took {time.time() - start_time:.2f} seconds", flush=True)
try:
    data2 = response2.json()
    print("Turn 2 Final Answer preview:", data2.get("response", "No response")[:200])
except Exception as e:
    print(f"Error parsing response: {e}\n{response2.text}")
