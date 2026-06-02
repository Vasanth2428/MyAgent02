# Tests for supervisor routing.
import pytest
from langgraph.graph import END


def test_supervisor_routing():
    """Test supervisor routes correctly based on query type."""
    from src.graph.workflow import route_based_on_next_agent
    
    assert route_based_on_next_agent({"next_agent": "rag_worker", "steps_remaining": 5}) == "rag_worker_node"
    assert route_based_on_next_agent({"next_agent": "web_worker", "steps_remaining": 5}) == "web_worker_node"
    assert route_based_on_next_agent({"next_agent": "utility_worker", "steps_remaining": 5}) == "utility_worker_node"
    assert route_based_on_next_agent({"next_agent": "FINISH", "steps_remaining": 5}) == END


def test_supervisor_default():
    """Test supervisor handles unknown routing."""
    from src.graph.workflow import route_based_on_next_agent
    
    result = route_based_on_next_agent({"next_agent": "unknown", "steps_remaining": 5})
    assert result == END


def test_supervisor_steps_exhausted():
    """Test supervisor stops when steps_remaining is 0."""
    from src.graph.workflow import route_based_on_next_agent
    
    result = route_based_on_next_agent({"next_agent": "rag_worker", "steps_remaining": 0})
    assert result == END
