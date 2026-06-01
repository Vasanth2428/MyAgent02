# Tests for web worker.
import pytest
from unittest.mock import Mock


def test_web_worker_routing():
    """Test web worker processes search correctly."""
    from langchain_core.messages import AIMessage
    from src.agents.web_worker import web_worker_node
    
    state = {
        "messages": [{"role": "user", "content": "What is the weather today?"}],
        "next_agent": "web_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    
    def mock_search(query):
        return [{"title": "Weather", "url": "https://example.com", "content": "Sunny, 72F"}]
    
    result = web_worker_node(state, web_search_tool=mock_search)
    
    assert "messages" in result
    assert result["next_agent"] == "FINISH"


def test_web_worker_no_results():
    """Test web worker handles no results gracefully."""
    from src.agents.web_worker import web_worker_node
    
    state = {
        "messages": [{"role": "user", "content": "What is XYZ123?"}],
        "next_agent": "web_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    
    def mock_search(query):
        return []
    
    result = web_worker_node(state, web_search_tool=mock_search)
    
    assert "couldn't find" in result["messages"][0].content
