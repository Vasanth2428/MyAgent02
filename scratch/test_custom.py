import os
import sys
from dotenv import load_dotenv
sys.path.insert(0, r"c:\Users\vasan\Documents\Apphelix Intern\RAG")

load_dotenv(r"c:\Users\vasan\Documents\Apphelix Intern\RAG\config\.env")

import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth

raw_url = os.getenv("WEAVIATE_URL", "")
api_key = os.getenv("WEAVIATE_API_KEY", "")

# Normalize URL
clean_url = raw_url
if "://" in clean_url:
    clean_url = clean_url.split("://", 1)[1]

if clean_url.startswith("grpc-"):
    grpc_host = clean_url
    http_host = clean_url[5:]
else:
    http_host = clean_url
    grpc_host = "grpc-" + clean_url

if ":" in http_host:
    http_host = http_host.split(":", 1)[0]
if ":" in grpc_host:
    grpc_host = grpc_host.split(":", 1)[0]

print(f"Connecting with:")
print(f"  http_host: {http_host}")
print(f"  grpc_host: {grpc_host}")

config = wvc.init.AdditionalConfig(
    timeout=wvc.init.Timeout(init=60, query=120, insert=120)
)

try:
    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=443,
        http_secure=True,
        grpc_host=grpc_host,
        grpc_port=443,
        grpc_secure=True,
        auth_credentials=Auth.api_key(api_key),
        headers={},
        additional_config=config
    )
    print("Successfully connected using connect_to_custom with headers!")
    print(f"Meta: {client.get_meta()}")
    client.close()
except Exception as e:
    import traceback
    traceback.print_exc()
