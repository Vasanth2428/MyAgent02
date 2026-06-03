import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.engine import RAGContextEngine
from src.core.config import PipelineConfig


@pytest.mark.asyncio
async def test_multi_hop_reasoning_evaluation():
    """Verify that multi-hop queries retrieve and reason across multiple documents."""
    # 1. Setup mock retriever returning disjoint documents
    mock_retriever = MagicMock()
    mock_retriever._connected = True
    
    # We will simulate returning the two required facts
    doc1 = {
        "text": "Alice works for Acme Corp.",
        "source": "org_chart.txt",
        "score": 0.9,
        "tags": ["employees"],
        "content_hash": "h1",
        "document_id": "d1"
    }
    doc2 = {
        "text": "Acme Corp headquarters are in Berlin.",
        "source": "office_locations.txt",
        "score": 0.8,
        "tags": ["offices"],
        "content_hash": "h2",
        "document_id": "d2"
    }
    
    # Retriever function mock (called inside thread in retrieve_async)
    def mock_retrieve(*args, **kwargs):
        # Return [results], embed_latency, db_latency
        return [doc1, doc2], 1.5, 2.5

    mock_retriever.retrieve = mock_retrieve
    mock_retriever.get_count.return_value = 2

    # 2. Setup mock LLM for query expansion, HyDE, and final generation
    mock_client = MagicMock()
    
    # Mock completions returning the final multi-hop response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Berlin"
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    # Mock async client as well
    mock_async_client = AsyncMock()
    mock_async_response = MagicMock()
    mock_async_response.choices = [MagicMock()]
    mock_async_response.choices[0].message = MagicMock()
    mock_async_response.choices[0].message.content = "Berlin"
    mock_async_response.usage = None
    mock_async_client.chat.completions.create.return_value = mock_async_response

    # 3. Create RAG Engine with mocked components
    # We use a developmental pipeline config to skip HyDE and expansion to keep test direct
    dev_config = PipelineConfig(
        enable_hyde=False,
        enable_expansion=False,
        enable_reranking=False,
        enable_compression=False
    )
    
    # Centralized mocks using patches
    with patch("src.core.engine.LLMService") as mock_llm_class:
        # Mock instance of LLMService
        mock_llm_instance = MagicMock()
        mock_llm_instance.raw_client = mock_client
        mock_llm_instance.async_client = mock_async_client
        mock_llm_class.return_value = mock_llm_instance
        
        engine = RAGContextEngine(retriever=mock_retriever, pipeline_config=dev_config)
        
        # Mock generation service's LLM client
        engine.generation_service.client = mock_client
        engine.generation_service.async_client = mock_async_client
        
        # 4. Execute multi-hop query
        query = "Where does Alice's employer have its headquarters?"
        result = await engine.ask_async(query, session_id="test_multihop_session")
        
        # 5. Assertions
        # Check that we got the expected answer
        assert "Berlin" in result["response"]
        
        # Check that both documents were retrieved and stored in final context
        context_used = result["compressed_context"]
        assert "Alice works for Acme Corp." in context_used
        assert "Acme Corp headquarters are in Berlin." in context_used
        
        # Check that metadata links back to correct sources
        assert any(r["source"] == "org_chart.txt" for r in result.get("retrieved_context", []))
        assert any(r["source"] == "office_locations.txt" for r in result.get("retrieved_context", []))
