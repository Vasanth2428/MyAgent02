"""
================================================================================
RAG CONTEXT ENGINE - RETRIEVER MODULE
================================================================================
This module manages the connection to Weaviate Cloud and implements:
- Vector Indexing (with deterministic UUIDs)
- Hybrid Search (Vector + BM25 with dynamic alpha)
- Metadata Filtering
- Local Embedding Generation
"""

import os
import re
import uuid
import logging
import time
import random
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import weaviate.exceptions

from core.config import EMBEDDING_MODEL, HYBRID_ALPHA_DEFAULT, HYBRID_ALPHA_KEYWORD

logger = logging.getLogger("RAG.Retriever")

# Technical keywords that signal a query should favor BM25 keyword matching
_TECHNICAL_KEYWORDS = [
    "error", "exception", "status", "syntax", "null", "none",
    "def ", "class ", "import ", "void", "public", "private",
    "int ", "str ", "code"
]


class WeaviateRetriever:
    """
    Service for interacting with the Weaviate vector database.
    """

    def __init__(self):
        self.url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.client = None
        self.collection = None
        self._connected = False

        # Connect to Weaviate Cloud with extended timeouts for stability
        config = wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=60, query=120, insert=120)
        )

        max_conn_retries = 3
        conn_base_delay = 1.0
        for attempt in range(max_conn_retries):
            try:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=Auth.api_key(self.api_key),
                    additional_config=config
                )
                self._connected = True
                break
            except Exception as e:
                if attempt == max_conn_retries - 1:
                    logger.error(f"Failed to connect to Weaviate Cloud after {max_conn_retries} attempts: {e}")
                    self._connected = False
                    logger.warning("Starting in degraded mode - database operations will fail gracefully.")
                else:
                    delay = conn_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"Connection attempt {attempt+1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

        if self.client and self._connected:
            # Ensure the collection schema is initialized
            if not self.client.collections.exists("RAGKnowledge"):
                logger.info("Initializing 'RAGKnowledge' collection...")
                self.client.collections.create(
                    name="RAGKnowledge",
                    vector_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                        wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                    ]
                )

            self.collection = self.client.collections.get("RAGKnowledge")
        
        self.alpha = HYBRID_ALPHA_DEFAULT

        # Load local embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def execute_with_retry(self, func, *args, **kwargs):
        """
        Executes a Weaviate client operation with automatic retries on transient errors.
        """
        max_retries = 3
        base_delay = 0.5
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (
                weaviate.exceptions.WeaviateConnectionError,
                weaviate.exceptions.UnexpectedStatusCodeError,
                weaviate.exceptions.WeaviateQueryError,
                Exception
            ) as e:
                err_name = type(e).__name__
                err_msg = str(e).lower()
                is_transient = any(
                    x in err_msg 
                    for x in ["timeout", "connection", "rate limit", "429", "502", "503", "504", "unavailable", "network"]
                )
                if hasattr(e, "status_code"):
                    if e.status_code == 429 or (e.status_code and e.status_code >= 500):
                        is_transient = True
                
                if not is_transient:
                    raise

                
                if attempt == max_retries - 1:
                    logger.error(f"Weaviate operation failed after {max_retries} attempts: {e}")
                    raise
                
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                logger.warning(
                    f"Weaviate operation failed with {err_name}: {e}. "
                    f"Retrying in {delay:.2f}s (Attempt {attempt+1}/{max_retries})...."
                )
                time.sleep(delay)

    def add_documents(self, docs: List[str], tags: List[str] = None, source: str = "unknown"):
        """
        Processes and indexes document chunks into Weaviate.
        Uses deterministic UUIDs to prevent duplicate entries of the same text.
        """
        if not self._connected:
            logger.warning(f"Weaviate not connected - skipping document indexing for: {source}")
            return
        
        t_start = time.time()

        embeddings = self.embedding_model.encode(docs)
        t_embed = time.time()
        logger.debug(f"Generated {len(docs)} embeddings in {(t_embed - t_start)*1000:.1f}ms")

        def _batch_insert():
            with self.collection.batch.dynamic() as batch:
                for i, doc in enumerate(docs):
                    doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc)
                    batch.add_object(
                        properties={"text": doc, "tags": tags or [], "source": source},
                        vector=embeddings[i].tolist() if hasattr(embeddings[i], "tolist") else embeddings[i],
                        uuid=doc_id
                    )
            failed = self.collection.batch.failed_objects
            if failed:
                raise weaviate.exceptions.WeaviateQueryError(
                    f"Weaviate batch insert failed for {len(failed)} objects. First error: {failed[0].message}"
                )

        self.execute_with_retry(_batch_insert)

        t_batch = time.time()
        logger.info(
            f"Indexed {len(docs)} chunks from '{source}' in {(t_batch - t_start)*1000:.1f}ms "
            f"(Embed: {(t_embed - t_start)*1000:.1f}ms, Insert: {(t_batch - t_embed)*1000:.1f}ms)"
        )

    def _detect_alpha(self, query: str) -> float:
        """
        Dynamically adjusts hybrid alpha based on query content.
        Technical/code queries shift toward BM25 keyword matching.
        """
        query_lower = query.lower()
        if any(kw in query_lower for kw in _TECHNICAL_KEYWORDS) or re.search(r'[\{\}\[\]\(\)\.\\_\|]', query):
            return HYBRID_ALPHA_KEYWORD
        return HYBRID_ALPHA_DEFAULT

    def retrieve(self, query: str, top_k: int = 5, source_filter: str = None) -> List[Dict]:
        """
        Performs a Hybrid Search (Semantic + Keyword) with optional hard filtering.
        """
        if not self._connected:
            logger.warning(f"Weaviate not connected - returning empty results for query: {query[:30]}...")
            self.last_embed_latency_ms = 0.0
            self.last_search_latency_ms = 0.0
            return []
        
        t_start = time.time()
        query_vector = self.embedding_model.encode(query).tolist()
        t_embed = time.time()

        # Construct property filter
        filters = wvc.query.Filter.by_property("source").equal(source_filter) if source_filter else None

        # Dynamically classify query type
        alpha = self._detect_alpha(query)
        self.alpha = alpha

        def _query_db():
            return self.collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                filters=filters,
                return_properties=["text", "tags", "source"],
                return_metadata=wvc.query.MetadataQuery(score=True)
            )

        response = self.execute_with_retry(_query_db)

        t_search = time.time()
        self.last_embed_latency_ms = (t_embed - t_start) * 1000
        self.last_search_latency_ms = (t_search - t_embed) * 1000

        res = [{
            "text": obj.properties["text"],
            "tags": obj.properties.get("tags") or [],
            "source": obj.properties.get("source"),
            "score": obj.metadata.score
        } for obj in response.objects]

        logger.info(
            f"Hybrid search (alpha={alpha}) found {len(res)} results "
            f"in {(t_search - t_start)*1000:.1f}ms"
        )
        return res

    def get_count(self) -> int:
        """Returns the total number of objects in the RAGKnowledge collection."""
        if not self._connected:
            return 0
        try:
            def _aggregate():
                res = self.collection.aggregate.over_all(total_count=True)
                return res.total_count
            return self.execute_with_retry(_aggregate)
        except Exception:
            return 0

    def close(self):
        """Safely terminates the connection to Weaviate Cloud."""
        if self.client and self._connected:
            self.client.close()
