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
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from typing import List, Dict, Optional, Tuple
import weaviate.exceptions

from src.core.config import EMBEDDING_MODEL, HYBRID_ALPHA_DEFAULT, HYBRID_ALPHA_KEYWORD

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

    def __init__(self):
        self.url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.client = None
        self.collection = None
        self._connected = False

        config = wvc.init.AdditionalConfig(
            timeout=wvc.init.Timeout(init=60, query=120, insert=120)
        )

        from src.core.retry import retry

        @retry(
            retries=3,
            backoff=1.0,
            jitter=0.5,
            logger_name="RAG.Retriever"
        )
        def _connect():
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
                additional_config=config
            )

        try:
            self.client = _connect()
            self._connected = True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate Cloud after 3 attempts: {e}")
            self._connected = False
            logger.warning("Starting in degraded mode - database operations will fail gracefully.")

        if self.client and self._connected:
            if not self.client.collections.exists("RAGKnowledge"):
                logger.info("Initializing 'RAGKnowledge' collection...")
                self.client.collections.create(
                    name="RAGKnowledge",
                    vector_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                        wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="content_hash", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="upload_timestamp", data_type=wvc.config.DataType.NUMBER),
                        wvc.config.Property(name="document_id", data_type=wvc.config.DataType.TEXT),
                    ]
                )

            self.collection = self.client.collections.get("RAGKnowledge")

        self.alpha = HYBRID_ALPHA_DEFAULT

    def _get_embedding_model(self):
        from src.core.services.grounding_service import _get_shared_embedding_model
        return _get_shared_embedding_model()

    @property
    def embedding_model(self):
        return self._get_embedding_model()

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
        if not self._connected:
            logger.warning(f"Weaviate not connected - skipping document indexing for: {source}")
            return []

        t_start = time.time()
        
        # Compute content hash for integrity tracking
        import hashlib
        content_hash = hashlib.sha256(",".join(docs).encode()).hexdigest()[:16]

        embeddings = self.embedding_model.encode(docs)
        t_embed = time.time()
        logger.debug(f"Generated {len(docs)} embeddings in {(t_embed - t_start)*1000:.1f}ms")

        indexed_uuids = []
        doc_upload_time = time.time()

        def _batch_insert():
            with self.collection.batch.dynamic() as batch:
                for i, doc in enumerate(docs):
                    # Use document_id in UUID if provided for version tracking
                    if document_id:
                        chunk_id = f"{document_id}_{i}"
                        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
                    else:
                        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc)
                    
                    indexed_uuids.append(str(doc_id))
                    
                    # Enhanced properties with lifecycle metadata
                    properties = {
                        "text": doc, 
                        "tags": tags or [], 
                        "source": source,
                        "content_hash": content_hash,
                        "upload_timestamp": doc_upload_time,
                        "document_id": document_id or str(doc_id),
                    }
                    
                    batch.add_object(
                        properties=properties,
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
            f"Indexed {len(docs)} chunks from '{source}' (doc_id={document_id}) in {(t_batch - t_start)*1000:.1f}ms "
            f"(Embed: {(t_embed - t_start)*1000:.1f}ms, Insert: {(t_batch - t_embed)*1000:.1f}ms)"
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
            logger.warning(f"Weaviate not connected - returning empty results for query: {query[:30]}...")
            return [], 0.0, 0.0

        t_start = time.time()
        query_vector = self.embedding_model.encode(query).tolist()
        t_embed = time.time()

        filters = wvc.query.Filter.by_property("source").equal(source_filter) if source_filter else None

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
        embed_latency_ms = (t_embed - t_start) * 1000
        search_latency_ms = (t_search - t_embed) * 1000

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