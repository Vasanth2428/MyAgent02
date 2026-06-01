# Tests for full workflow.
import pytest


def test_workflow_builds():
    """Test that the workflow graph builds correctly."""
    from src.graph.workflow import build_multi_agent_graph
    
    graph = build_multi_agent_graph()
    assert graph is not None
    assert hasattr(graph, 'invoke')


def test_workflow_routing():
    """Test end-to-end routing through workflow."""
    from src.graph.workflow import build_multi_agent_graph, route_based_on_next_agent
    
    graph = build_multi_agent_graph()
    
    test_states = [
        {"next_agent": "rag_worker"},
        {"next_agent": "web_worker"},
        {"next_agent": "utility_worker"},
        {"next_agent": "FINISH"},
    ]
    
    for state in test_states:
        result = route_based_on_next_agent(state)
        assert result is not None
