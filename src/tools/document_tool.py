# Document retrieval tool for RAG worker.
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("MultiAgent.DocumentTool")


class DocumentRetrieverTool:
    """Tool that queries the existing Weaviate retriever for document search."""
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def search(self, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        """Search documents and return results."""
        try:
            results, _, _ = self.retriever.retrieve(query, top_k=top_k, source_filter=source_filter)
            return results
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return []
    
    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved documents as context string."""
        if not results:
            return "No relevant documents found."
        context = "\n\n".join([f"Document {i+1}:\n{r['text']}" for i, r in enumerate(results)])
        return context
