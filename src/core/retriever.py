"""
RAG Retriever - Your Document Search Engine

This module handles searching through your uploaded documents in the Weaviate vector database.
It can find and store information using two methods:
- Vector search: Understands the meaning of your question
- Keyword search: Finds exact words you're looking for

When you upload a document, this module breaks it into pieces and stores each piece
with a mathematical "fingerprint" that helps find it later when you ask questions.
"""

import os
import re
import uuid
import logging
import time
import random
import threading
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from typing import List, Dict, Optional, Tuple, Any
import weaviate.exceptions

from src.core.config import HYBRID_ALPHA_DEFAULT, HYBRID_ALPHA_KEYWORD

# HF API key enables server-side vectorization via text2vec-huggingface.
_HF_API_KEY = os.getenv("HF_API_KEY", "")

logger = logging.getLogger("RAG.Retriever")

_TECHNICAL_KEYWORDS = [
    "error", "exception", "status", "syntax", "null", "none",
    "def ", "class ", "import ", "void", "public", "private",
    "int ", "str ", "code"
]


class WeaviateRetriever:
    """
    Connects to the Weaviate vector database to store and search documents.
    
    This class handles all the interaction with our document storage system. When you
    upload a file, it converts the text into mathematical vectors (like a fingerprint)
    so we can find similar content later. It supports both meaning-based search (vector)
    and exact keyword search.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WeaviateRetriever, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        url = os.getenv("WEAVIATE_URL", "")
        if url.startswith("grpc-"):
            url = url[5:]
        if url and not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        self.url = url
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.client = None
        self.collection = None
        self._connected = False

        config = wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=60, query=120, insert=120)
        )

        # Pass HF API key as a header so Weaviate Cloud can call the HF Inference API
        hf_headers = {}
        if _HF_API_KEY:
            hf_headers["X-HuggingFace-Api-Key"] = _HF_API_KEY
        logger.info("Using text2vec-huggingface server-side vectorizer.")

        from src.core.retry import retry

        @retry(
            retries=3,
            backoff=1.0,
            jitter=0.5,
            logger_name="RAG.Retriever"
        )
        def _connect():
            clean_url = self.url
            if "://" in clean_url:
                clean_url = clean_url.split("://", 1)[1]
            
            # Extract http and grpc hosts
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
                return weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                    headers=hf_headers,
                    additional_config=config
                )
            else:
                return weaviate.connect_to_custom(
                    http_host=http_host,
                    http_port=443,
                    http_secure=True,
                    grpc_host=grpc_host,
                    grpc_port=443,
                    grpc_secure=True,
                    auth_credentials=Auth.api_key(self.api_key),
                    headers=hf_headers,
                    additional_config=config
                )

        try:
            self.client = _connect()
            self._connected = True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate Cloud after 3 attempts: {e}")
            self._connected = False
            logger.warning("Starting in degraded mode - database operations will fail gracefully.")

        self.code_collection = None

        if self.client and self._connected:
            if not self.client.collections.exists("RAGKnowledge"):
                logger.info("Initializing 'RAGKnowledge' collection with text2vec-huggingface...")
                self.client.collections.create(
                    name="RAGKnowledge",
                    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_huggingface(
                        model="sentence-transformers/all-MiniLM-L6-v2",
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

            self.collection = self.client.collections.get("RAGKnowledge")
            if self.client.collections.exists("RAGCode"):
                self.code_collection = self.client.collections.get("RAGCode")
            else:
                self.code_collection = self.collection  # Fallback to RAGKnowledge if RAGCode doesn't exist

        self.local_docs = []
        self.local_code_chunks = []
        self.alpha = HYBRID_ALPHA_DEFAULT

        # Load persistent local fallback database if Weaviate is not connected
        import json
        try:
            if os.path.exists("data/local_docs.json"):
                with open("data/local_docs.json", "r", encoding="utf-8") as f:
                    self.local_docs = json.load(f)
                logger.info(f"Loaded {len(self.local_docs)} documents from local persistent database.")
        except Exception as e:
            logger.warning(f"Failed to load local docs from persistent storage: {e}")

        try:
            if os.path.exists("data/local_code_chunks.json"):
                with open("data/local_code_chunks.json", "r", encoding="utf-8") as f:
                    self.local_code_chunks = json.load(f)
                logger.info(f"Loaded {len(self.local_code_chunks)} code chunks from local persistent database.")
        except Exception as e:
            logger.warning(f"Failed to load local code chunks from persistent storage: {e}")



    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a Weaviate client operation with automatic retries on transient errors.
        """
        def is_weaviate_transient(e):
            err_msg = str(e).lower()
            is_t = any(
                x in err_msg 
                for x in ["timeout", "connection", "rate limit", "429", "502", "503", "504", "unavailable", "network"]
            )
            if hasattr(e, "status_code"):
                if e.status_code == 429 or (e.status_code and e.status_code >= 500):
                    is_t = True
            return is_t

        from src.core.retry import retry
        wrapped = retry(
            retries=3,
            backoff=0.5,
            jitter=0.1,
            is_transient_fn=is_weaviate_transient,
            logger_name="RAG.Retriever"
        )(func)
        return wrapped(*args, **kwargs)

    def add_documents(self, docs: List[str], tags: List[str] = None, source: str = "unknown", document_id: str = None):
        """
        Processes and indexes document chunks into Weaviate.
        Uses deterministic UUIDs to prevent duplicate entries of the same text.
        
        Args:
            docs: List of document text chunks to index.
            tags: Optional list of tags for categorization.
            source: Source document name for tracking.
            document_id: Optional unique document ID for lifecycle tracking.
        
        Returns:
            List of indexed document UUIDs for tracking.
        """
        t_start = time.time()
        
        # Compute content hash for integrity tracking
        import hashlib
        content_hash = hashlib.sha256(",".join(docs).encode()).hexdigest()[:16]

        t_embed_start = time.time()
        from src.core.services.grounding_service import _get_shared_embedding_model
        embedding_model = _get_shared_embedding_model()
        doc_vectors = embedding_model.encode(docs).tolist()
        t_embed = time.time()

        # Enhanced properties with lifecycle metadata
        doc_upload_time = time.time()
        indexed_uuids = []

        for i, doc in enumerate(docs):
            if document_id:
                chunk_id = f"{document_id}_{i}"
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            else:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc))

            indexed_uuids.append(doc_id)

            self.local_docs.append({
                "text": doc,
                "source": source,
                "tags": tags or [],
                "document_id": document_id or doc_id,
                "content_hash": content_hash,
                "upload_timestamp": doc_upload_time,
                "is_code": False,
                "vector": None
            })

        # Save to local persistent storage
        try:
            os.makedirs("data", exist_ok=True)
            import json
            with open("data/local_docs.json", "w", encoding="utf-8") as f:
                json.dump(self.local_docs, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(docs)} documents to local persistent database.")
        except Exception as e:
            logger.warning(f"Failed to persist local documents: {e}")

        if not self._connected:
            logger.warning(f"Weaviate not connected - saved document to local memory & file for: {source}")
            return indexed_uuids

        def _batch_insert():
            with self.collection.batch.dynamic() as batch:
                for i, doc in enumerate(docs):
                    doc_uuid = uuid.UUID(indexed_uuids[i])
                    properties = {
                        "text": doc,
                        "tags": tags or [],
                        "source": source,
                        "content_hash": content_hash,
                        "upload_timestamp": doc_upload_time,
                        "document_id": document_id or indexed_uuids[i],
                        "is_code": False,
                    }
                    batch.add_object(properties=properties, uuid=doc_uuid, vector=doc_vectors[i])
            failed = self.collection.batch.failed_objects
            if failed:
                raise weaviate.exceptions.WeaviateQueryError(
                    "insert",
                    f"Weaviate batch insert failed for {len(failed)} objects. First error: {failed[0].message}"
                )

        self.execute_with_retry(_batch_insert)

        t_batch = time.time()
        logger.info(
            f"Indexed {len(docs)} chunks from '{source}' (doc_id={document_id}) in {(t_batch - t_start)*1000:.1f}ms "
            f"(Insert: {(t_batch - t_embed)*1000:.1f}ms)"
        )
        return indexed_uuids

    def _detect_alpha(self, query: str) -> float:
        """
        Dynamically adjusts hybrid alpha based on query content.
        Technical/code queries shift toward BM25 keyword matching.
        """
        query_lower = query.lower()
        if any(kw in query_lower for kw in _TECHNICAL_KEYWORDS) or re.search(r'[\{\}\[\]\(\)\.\\_\|]', query):
            return HYBRID_ALPHA_KEYWORD
        return HYBRID_ALPHA_DEFAULT

    def retrieve(self, query: str, top_k: int = 5, source_filter: str = None) -> Tuple[List[Dict], float, float]:
        """
        Performs a Hybrid Search (Semantic + Keyword) with optional hard filtering.
        Returns:
            Tuple of (results, embed_latency_ms, db_search_latency_ms)
        """
        if not self._connected:
            logger.warning("Weaviate not connected - using local keyword-only fallback search (semantic scoring unavailable in offline mode)")
            t_start = time.time()
            results = []
            
            # Keyword BM25-like matching
            keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3]
            for doc in self.local_docs:
                if doc.get("is_code", False):
                    continue
                if source_filter and doc["source"] != source_filter:
                    continue
                
                keyword_score = 0.0
                text_lower = doc["text"].lower()
                for kw in keywords:
                    if kw in text_lower:
                        keyword_score += 1.0
                keyword_score = keyword_score / (len(keywords) if keywords else 1)

                results.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "tags": doc["tags"],
                    "score": keyword_score,
                    "content_hash": doc.get("content_hash", ""),
                    "document_id": doc["document_id"]
                })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            t_search = time.time()
            embed_latency_ms = 0.0
            search_latency_ms = (t_search - t_start) * 1000
            return results[:top_k], embed_latency_ms, search_latency_ms

        t_start = time.time()
        t_embed = time.time()

        filters = wvc.query.Filter.by_property("is_code").equal(False)
        if source_filter:
            filters = filters & wvc.query.Filter.by_property("source").equal(source_filter)

        alpha = self._detect_alpha(query)
        self.alpha = alpha

        # Use local embedding model for query vector (avoids server-side vectorization permissions)
        t_embed_start = time.time()
        from src.core.services.grounding_service import _get_shared_embedding_model
        embedding_model = _get_shared_embedding_model()
        query_vector = embedding_model.encode(query).tolist()
        embed_latency_ms = (time.time() - t_embed_start) * 1000

        def _query_db():
            return self.collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=filters,
                return_properties=["text", "tags", "source"],
                return_metadata=wvc.query.MetadataQuery(score=True)
            )

        response = self.execute_with_retry(_query_db)

        t_search = time.time()
        search_latency_ms = (t_search - t_start) * 1000

        res = [{
            "text": obj.properties["text"],
            "tags": obj.properties.get("tags") or [],
            "source": obj.properties.get("source"),
            "score": obj.metadata.score,
            "content_hash": obj.properties.get("content_hash"),
            "document_id": obj.properties.get("document_id"),
        } for obj in response.objects]

        logger.info(
            f"Hybrid search (alpha={alpha}) found {len(res)} results "
            f"in {(t_search - t_start)*1000:.1f}ms"
        )
        return res, embed_latency_ms, search_latency_ms

    def get_count(self) -> int:
        """Returns the total number of objects in the RAGKnowledge collection."""
        if not self._connected:
            return len(self.local_docs)
        try:
            def _aggregate():
                res = self.collection.aggregate.over_all(total_count=True)
                return res.total_count
            return self.execute_with_retry(_aggregate)
        except Exception:
            return 0

    def add_code_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Indexes code chunks into Weaviate.
        """
        doc_upload_time = time.time()
        indexed_uuids = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['filepath']}_{chunk.get('symbol_name', '')}_{i}"
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            indexed_uuids.append(doc_id)

            local_chunk = {
                "text": chunk["text"],
                "symbol_name": chunk.get("symbol_name") or "",
                "symbol_type": chunk.get("symbol_type") or "",
                "filepath": chunk["filepath"],
                "start_line": int(chunk.get("start_line", 1)),
                "end_line": int(chunk.get("end_line", 1)),
                "source": chunk.get("source") or "repo",
                "upload_timestamp": doc_upload_time,
                "is_code": True,
                "vector": None
            }
            self.local_code_chunks.append(local_chunk)

        # Compute vectors for code chunks to avoid server-side vectorization
        from src.core.services.grounding_service import _get_shared_embedding_model
        embedding_model = _get_shared_embedding_model()
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_vectors = embedding_model.encode(chunk_texts).tolist()

        # Save to local persistent storage
        try:
            os.makedirs("data", exist_ok=True)
            import json
            with open("data/local_code_chunks.json", "w", encoding="utf-8") as f:
                json.dump(self.local_code_chunks, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(chunks)} code chunks to local persistent database.")
        except Exception as e:
            logger.warning(f"Failed to persist local code chunks: {e}")

        if not self._connected or not self.client or not self.code_collection:
            logger.warning("Weaviate not connected - saved code chunk locally.")
            return indexed_uuids

        def _batch_insert():
            with self.code_collection.batch.dynamic() as batch:
                for i, chunk in enumerate(chunks):
                    doc_uuid = uuid.UUID(indexed_uuids[i])
                    properties = {
                        "text": chunk["text"],
                        "symbol_name": chunk.get("symbol_name") or "",
                        "symbol_type": chunk.get("symbol_type") or "",
                        "filepath": chunk["filepath"],
                        "start_line": int(chunk.get("start_line", 1)),
                        "end_line": int(chunk.get("end_line", 1)),
                        "source": chunk.get("source") or "repo",
                        "upload_timestamp": doc_upload_time,
                        "is_code": True,
                    }
                    batch.add_object(properties=properties, uuid=doc_uuid, vector=chunk_vectors[i])
            failed = self.code_collection.batch.failed_objects
            if failed:
                raise weaviate.exceptions.WeaviateQueryError(
                    "insert",
                    f"Weaviate batch insert failed for {len(failed)} objects. First error: {failed[0].message}"
                )
        self.execute_with_retry(_batch_insert)
        return indexed_uuids

    def search_code_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the RAGCode collection using hybrid search.
        """
        if not self._connected or not self.client or not self.code_collection:
            logger.warning("Weaviate not connected - using local keyword-only fallback search for code chunks (semantic scoring unavailable in offline mode)")

            results = []
            keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3]
            for chunk in self.local_code_chunks:
                # Keyword matching
                keyword_score = 0.0
                text_lower = chunk["text"].lower()
                for kw in keywords:
                    if kw in text_lower:
                        keyword_score += 1.0
                keyword_score = keyword_score / (len(keywords) if keywords else 1)

                results.append({
                    "chunk": chunk,
                    "score": keyword_score
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return [item["chunk"] for item in results[:limit]]

        # Use local embedding model for query vector (avoids server-side vectorization permissions)
        from src.core.services.grounding_service import _get_shared_embedding_model
        embedding_model = _get_shared_embedding_model()
        query_vector = embedding_model.encode(query).tolist()

        def _query_db():
            return self.code_collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=wvc.query.Filter.by_property("is_code").equal(True),
                return_properties=["text", "symbol_name", "symbol_type", "filepath", "start_line", "end_line", "source"]
            )
        response = self.execute_with_retry(_query_db)
        return [{
            "text": obj.properties["text"],
            "symbol_name": obj.properties.get("symbol_name"),
            "symbol_type": obj.properties.get("symbol_type"),
            "filepath": obj.properties.get("filepath"),
            "start_line": obj.properties.get("start_line"),
            "end_line": obj.properties.get("end_line"),
            "source": obj.properties.get("source"),
        } for obj in response.objects]

    def close(self):
        """Safely terminates the connection to Weaviate Cloud."""
        if self.client and self._connected:
            self.client.close()