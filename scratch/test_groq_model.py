import os
import httpx
from dotenv import load_dotenv

load_dotenv()

keys = {
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "GROQ_CORE_KEY": os.getenv("GROQ_CORE_KEY"),
    "GROQ_VALIDATION_KEY": os.getenv("GROQ_VALIDATION_KEY"),
    "AGENT_API_KEY": os.getenv("AGENT_API_KEY")
}

url = "https://api.groq.com/openai/v1/chat/completions"
payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0
}

for name, key in keys.items():
    if not key:
        print(f"{name}: not set")
        continue
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    try:
        response = httpx.post(url, headers=headers, json=payload)
        print(f"\n{name} limits:")
        print("  Status Code:", response.status_code)
        if response.status_code == 200:
            for k, v in response.headers.items():
                if "ratelimit" in k.lower() or "retry-after" in k.lower():
                    print(f"    {k}: {v}")
        else:
            print("  Response:", response.text)
    except Exception as e:
        print(f"Failed for {name}:", e)
