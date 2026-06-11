import urllib.request
import json

endpoints = ["/", "/stats", "/sessions"]
base_url = "http://127.0.0.1:8000"

for ep in endpoints:
    url = f"{base_url}{ep}"
    try:
        resp = urllib.request.urlopen(url)
        code = resp.getcode()
        print(f"Endpoint: {ep} -> Status: {code} (OK)")
        if ep == "/stats":
            data = json.loads(resp.read().decode())
            print(f"  Stats Output: {data}")
    except Exception as e:
        print(f"Endpoint: {ep} -> Error: {e}")
