import urllib.request
import urllib.error
import ssl

def check(url):
    print(f"Checking URL: {url}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=5) as response:
            print(f"  Status: {response.status}")
            print(f"  Headers: {dict(response.headers)}")
            print(f"  Body: {response.read().decode('utf-8')[:200]}")
    except urllib.error.HTTPError as e:
        print(f"  HTTPError: {e.code} - {e.reason}")
        try:
            print(f"  Body: {e.read().decode('utf-8')[:200]}")
        except Exception:
            pass
    except Exception as e:
        print(f"  Error: {e}")

urls = [
    "https://grpc-u45cyqi3tfsgmnwqirzjqq.c0.eu-central-1.aws.weaviate.cloud/v1/meta",
    "https://u45cyqi3tfsgmnwqirzjqq.c0.eu-central-1.aws.weaviate.cloud/v1/meta",
]

for url in urls:
    check(url)
    print()
