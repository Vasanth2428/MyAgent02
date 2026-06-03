"""
Knowledge Registry - What Documents Do We Have?

This module keeps track of all the documents you've uploaded. It lets the AI
know what's available to search through and can list sources, topics, and
document domains when asked.
"""

import os
import logging
from typing import List, Dict, Set

logger = logging.getLogger("RAG.Registry")


class KnowledgeRegistry:
    """
    Keeps track of all uploaded documents and their metadata.
    
    When someone asks "What documents do you have?" or "What can you search?",
    this class provides the list. It tracks:
    - Document sources (filenames)
    - Document domains (what kind of content: docs, sales data, etc.)
    - Available schemas (database structures)
    - Topics/tags found in documents
    """

    def __init__(self, engine):
        self.engine = engine  # Reference to the main engine for database access

    def get_sources(self) -> Set[str]:
        """Returns the set of unique source names in the knowledge base."""
        try:
            retriever = self.engine.retriever
            if not retriever or not hasattr(retriever, "collection") or retriever.collection is None:
                return set()
            
            # Fetch up to 1000 objects to find unique sources
            response = retriever.execute_with_retry(
                retriever.collection.query.fetch_objects,
                limit=1000,
                return_properties=["source"]
            )
            sources = set()
            for obj in response.objects:
                src = obj.properties.get("source")
                if src:
                    sources.add(src)
            return sources
        except Exception as e:
            logger.error(f"Error fetching sources from registry: {e}")
            return set()

    def get_indexed_datasets(self) -> List[str]:
        """Returns list of indexed dataset names."""
        return sorted(list(self.get_sources()))

    def get_document_domains(self) -> List[str]:
        """Returns document domains inferred from sources."""
        sources = self.get_sources()
        domains = set()
        for src in sources:
            src_lower = src.lower()
            if "policy" in src_lower or "guideline" in src_lower or "manual" in src_lower:
                domains.add("documentation")
            elif "sales" in src_lower or "order" in src_lower or "customer" in src_lower or "revenue" in src_lower or "analytics" in src_lower:
                domains.add("sales_analytics")
            elif "database" in src_lower or "schema" in src_lower:
                domains.add("database_schema")
            elif src_lower.endswith(".pdf"):
                domains.add("pdf_documents")
            elif src_lower.endswith(".txt"):
                domains.add("text_documents")
            else:
                domains.add("unclassified")
        return sorted(list(domains))

    def get_available_schemas(self) -> Dict[str, Dict]:
        """Returns available schemas for known datasets."""
        # Only return real schemas that the system has access to
        return {
            "sales_database": {
                "description": "Enterprise sales database (SQLite)",
                "tables": {
                    "customers": ["customer_id (PK)", "name", "email", "country", "signup_date"],
                    "orders": ["order_id (PK)", "customer_id (FK)", "order_date", "total_amount", "status"],
                    "inventory": ["product_id (PK)", "product_name", "category", "unit_price", "stock_quantity"]
                }
            },
            "system_metrics": {
                "description": "System resource monitoring stats",
                "metrics": ["cpu_usage_percent", "memory_usage_percent", "total_indexed_documents"]
            }
        }

    def get_topics(self) -> List[str]:
        """Returns unique tags or topics found in metadata."""
        try:
            retriever = self.engine.retriever
            if not retriever or not hasattr(retriever, "collection") or retriever.collection is None:
                return []
            
            response = retriever.execute_with_retry(
                retriever.collection.query.fetch_objects,
                limit=1000,
                return_properties=["tags"]
            )
            topics = set()
            for obj in response.objects:
                tags = obj.properties.get("tags")
                if tags:
                    if isinstance(tags, list):
                        topics.update(tags)
                    elif isinstance(tags, str):
                        topics.add(tags)
            return sorted(list(topics))
        except Exception as e:
            logger.error(f"Error fetching topics from registry: {e}")
            return []

    def get_registry_summary(self) -> Dict:
        """Returns a consolidated summary of the knowledge registry."""
        sources = list(self.get_sources())
        return {
            "sources": sources,
            "datasets": self.get_indexed_datasets(),
            "domains": self.get_document_domains(),
            "schemas": self.get_available_schemas(),
            "topics": self.get_topics(),
            "total_documents_count": self.engine.retriever.get_count()
        }
