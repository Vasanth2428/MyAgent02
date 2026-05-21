import requests
import json

url = "http://localhost:8000/query"
data = {"question": "What is the operating system course code?"}
response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
