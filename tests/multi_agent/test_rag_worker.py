# Tests for RAG worker.
import pytest
from unittest.mock import Mock, MagicMock


def test_rag_worker_no_documents():
    """Test RAG worker responds correctly when no documents found."""
    from langchain_core.messages import AIMessage
    from src.agents.rag_worker import rag_worker_node
    
    state = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "next_agent": "rag_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    
    def mock_search(query):
        return []
    
    result = rag_worker_node(state, document_tool=mock_search)
    
    assert "I don't know" in result["messages"][0].content


def test_rag_worker_with_documents():
    """Test RAG worker formats documents correctly."""
    from langchain_core.messages import AIMessage
    from src.agents.rag_worker import rag_worker_node
    
    state = {
        "messages": [{"role": "user", "content": "What is Python?"}],
        "next_agent": "rag_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    
    mock_docs = [{"text": "Python is a programming language.", "source": "doc1"}]
    
    def mock_search(query):
        return mock_docs
    
    result = rag_worker_node(state, document_tool=mock_search)
    
    assert "messages" in result
    assert result["next_agent"] == "FINISH"
