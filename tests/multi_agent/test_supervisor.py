# Tests for supervisor routing.
import pytest


def test_supervisor_routing():
    """Test supervisor routes correctly based on query type."""
    from src.graph.supervisor import route_based_on_next_agent
    
    assert route_based_on_next_agent({"next_agent": "rag_worker"}) == "rag_worker_node"
    assert route_based_on_next_agent({"next_agent": "web_worker"}) == "web_worker_node"
    assert route_based_on_next_agent({"next_agent": "utility_worker"}) == "utility_worker_node"
    assert route_based_on_next_agent({"next_agent": "FINISH"}) == "END" or route_based_on_next_agent({"next_agent": "FINISH"}) is None


def test_supervisor_default():
    """Test supervisor handles unknown routing."""
    from src.graph.supervisor import route_based_on_next_agent
    
    result = route_based_on_next_agent({"next_agent": "unknown"})
    assert result is None or result == "END"
