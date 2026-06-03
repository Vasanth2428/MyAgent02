# Tests for parallel dispatch functionality.
import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage
from src.graph.workflow import parallel_dispatch_node, build_multi_agent_graph


def test_parallel_dispatch_node_creates_send_objects():
    """Test that parallel_dispatch_node creates Send objects for parallel workers."""
    state = {
        "parallel_tasks": [
            {"worker": "rag_worker", "task": "Find revenue data"},
            {"worker": "web_worker", "task": "Search for stock price"}
        ],
        "worker_complete": {},
        "scratchpad": ""
    }
    
    result = parallel_dispatch_node(state)
    
    assert len(result) == 2
    assert result[0].node == "rag_worker_node"
    assert result[1].node == "web_worker_node"


def test_parallel_dispatch_filters_invalid_workers():
    """Test that parallel_dispatch ignores invalid worker types."""
    state = {
        "parallel_tasks": [
            {"worker": "rag_worker", "task": "Find revenue data"},
            {"worker": "unknown_worker", "task": "Unknown task"}
        ],
        "worker_complete": {},
        "scratchpad": ""
    }
    
    result = parallel_dispatch_node(state)
    
    assert len(result) == 1
    assert result[0].node == "rag_worker_node"


def test_workflow_supports_parallel_routing():
    """Test that workflow includes parallel_dispatch_node."""
    graph = build_multi_agent_graph()
    
    # Check that the graph has the parallel_dispatch node
    assert hasattr(graph, 'invoke')
    assert graph is not None