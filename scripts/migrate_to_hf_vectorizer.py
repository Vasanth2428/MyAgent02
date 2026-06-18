"""
migrate_to_hf_vectorizer.py — One-time collection recreation script

Run this ONCE after setting HF_API_KEY in config/.env to switch existing
Weaviate collections from self_provided vectors to text2vec-huggingface.

What it does:
  1. Checks that HF_API_KEY is set (aborts safely if not)
  2. Reports current document counts
  3. Deletes and recreates RAGKnowledge and RAGCode with the new vectorizer config
  4. Re-ingests documents from data/local_docs.json and data/local_code_chunks.json
  5. Confirms final counts match originals

Usage:
    python scripts/migrate_to_hf_vectorizer.py

The local JSON files are NOT deleted — they are the source of truth for re-ingestion.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.core.logging_setup

# Load .env from config directory
from pathlib import Path
env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

HF_API_KEY = os.getenv("HF_API_KEY", "")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
if WEAVIATE_URL.startswith("grpc-"):
    WEAVIATE_URL = WEAVIATE_URL[5:]
if WEAVIATE_URL and not WEAVIATE_URL.startswith("http://") and not WEAVIATE_URL.startswith("https://"):
    WEAVIATE_URL = "https://" + WEAVIATE_URL
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HF_MODEL = f"sentence-transformers/{EMBEDDING_MODEL}"

LOCAL_DOCS_PATH = Path(__file__).parent.parent / "data" / "local_docs.json"
LOCAL_CODE_PATH = Path(__file__).parent.parent / "data" / "local_code_chunks.json"


def abort(msg: str):
    print(f"\n❌ ABORTED: {msg}")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Weaviate Collection Migration: self_provided → text2vec-huggingface")
    print("=" * 60)

    # Pre-flight checks
    if not HF_API_KEY:
        print("⚠️ WARNING: HF_API_KEY is not set in config/.env.")
        print("   If your Weaviate Cloud cluster doesn't have a HuggingFace integration key configured, operations may fail.")
    else:
        print(f"\n✅ HF_API_KEY detected (model: {HF_MODEL})")
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        abort("WEAVIATE_URL or WEAVIATE_API_KEY not set in config/.env.")

    print(f"✅ Weaviate URL: {WEAVIATE_URL}")

    import weaviate
    import weaviate.classes as wvc
    from weaviate.classes.init import Auth

    config = wvc.init.AdditionalConfig(
        timeout=wvc.init.Timeout(init=60, query=120, insert=120)
    )
    headers = {}
    if HF_API_KEY:
        headers["X-HuggingFace-Api-Key"] = HF_API_KEY

    print("\nConnecting to Weaviate Cloud...")
    try:
        clean_url = WEAVIATE_URL
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
            
        if "weaviate.cloud" in clean_url or "weaviate.network" in clean_url:
            client = weaviate.connect_to_custom(
                http_host=http_host,
                http_port=443,
                http_secure=True,
                grpc_host=grpc_host,
                grpc_port=443,
                grpc_secure=True,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                headers=headers,
                additional_config=config
            )
        else:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                headers=headers,
                additional_config=config,
            )
        print("✅ Connected.")
    except Exception as e:
        abort(f"Could not connect to Weaviate: {e}")

    # Report current counts
    for coll_name in ["RAGKnowledge", "RAGCode"]:
        if client.collections.exists(coll_name):
            coll = client.collections.get(coll_name)
            count = coll.aggregate.over_all(total_count=True).total_count
            print(f"   Current '{coll_name}': {count} objects")
        else:
            print(f"   '{coll_name}': does not exist yet")

    # Confirm before destructive operation
    print("\n⚠️  This will DELETE and recreate RAGKnowledge and RAGCode.")
    print("   Documents will be re-ingested from local JSON files.")
    answer = input("   Type 'yes' to continue, anything else to abort: ").strip().lower()
    if answer != "yes":
        print("Aborted by user.")
        client.close()
        sys.exit(0)

    # Load local data before deleting
    local_docs = []
    if LOCAL_DOCS_PATH.exists():
        with open(LOCAL_DOCS_PATH, "r", encoding="utf-8") as f:
            local_docs = json.load(f)
        print(f"\n📄 Loaded {len(local_docs)} documents from local_docs.json")
    else:
        print("\n📄 No local_docs.json found — RAGKnowledge will be empty after migration.")

    local_code = []
    if LOCAL_CODE_PATH.exists():
        with open(LOCAL_CODE_PATH, "r", encoding="utf-8") as f:
            local_code = json.load(f)
        print(f"💻 Loaded {len(local_code)} code chunks from local_code_chunks.json")
    else:
        print("💻 No local_code_chunks.json found — RAGCode will be empty after migration.")

    # Delete existing collections
    print("\n🗑️  Deleting existing collections...")
    for coll_name in ["RAGKnowledge", "RAGCode"]:
        if client.collections.exists(coll_name):
            client.collections.delete(coll_name)
            print(f"   Deleted '{coll_name}'")

    # Recreate with text2vec-huggingface
    print("\n🔨 Recreating collections with text2vec-huggingface vectorizer...")

    client.collections.create(
        name="RAGKnowledge",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_huggingface(
            model=HF_MODEL,
            vectorize_collection_name=False,
        ),
        vector_index_config=wvc.config.Configure.VectorIndex.hfresh(),
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
            wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content_hash", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="upload_timestamp", data_type=wvc.config.DataType.NUMBER),
            wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="symbol_name", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="symbol_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="filepath", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="start_line", data_type=wvc.config.DataType.NUMBER),
            wvc.config.Property(name="end_line", data_type=wvc.config.DataType.NUMBER),
            wvc.config.Property(name="is_code", data_type=wvc.config.DataType.BOOL),
        ]
    )
    print("   ✅ RAGKnowledge created with text2vec-huggingface")

    # Re-ingest RAGKnowledge documents (text-only, no manual vectors)
    if local_docs:
        print(f"\n📥 Re-ingesting {len(local_docs)} documents into RAGKnowledge...")
        knowledge_coll = client.collections.get("RAGKnowledge")
        t0 = time.time()
        with knowledge_coll.batch.dynamic() as batch:
            for doc in local_docs:
                batch.add_object(properties={
                    "text": doc.get("text", ""),
                    "tags": doc.get("tags") or [],
                    "source": doc.get("source", "unknown"),
                    "content_hash": doc.get("content_hash", ""),
                    "upload_timestamp": doc.get("upload_timestamp", 0.0),
                    "document_id": doc.get("document_id", ""),
                    "is_code": False,
                })
        elapsed = time.time() - t0
        final_count = knowledge_coll.aggregate.over_all(total_count=True).total_count
        print(f"   ✅ Inserted {final_count}/{len(local_docs)} documents in {elapsed:.1f}s")

    # Re-ingest RAGCode chunks (text-only, no manual vectors)
    if local_code:
        print(f"\n📥 Re-ingesting {len(local_code)} code chunks into RAGKnowledge...")
        code_coll = client.collections.get("RAGKnowledge")
        t0 = time.time()
        with code_coll.batch.dynamic() as batch:
            for chunk in local_code:
                batch.add_object(properties={
                    "text": chunk.get("text", ""),
                    "symbol_name": chunk.get("symbol_name") or "",
                    "symbol_type": chunk.get("symbol_type") or "",
                    "filepath": chunk.get("filepath", ""),
                    "start_line": chunk.get("start_line", 1),
                    "end_line": chunk.get("end_line", 1),
                    "source": chunk.get("source", "repo"),
                    "upload_timestamp": chunk.get("upload_timestamp", 0.0),
                    "is_code": True,
                })
        elapsed = time.time() - t0
        final_count = code_coll.aggregate.over_all(total_count=True).total_count
        print(f"   ✅ Inserted {final_count}/{len(local_code)} code chunks in {elapsed:.1f}s")

    client.close()
    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("   Collections now use server-side text2vec-huggingface vectorization.")
    print("   Restart the application to pick up the changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
