import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.rag_worker import rag_worker_node
from src.agents.web_worker import web_worker_node
from src.agents.utility_worker import utility_worker_node
from src.graph.synthesizer import synthesizer_node
from src.graph.supervisor import supervisor_node


def test_cooperative_rag_worker_updates_scratchpad():
    """Test that rag_worker executes current_task and updates scratchpad."""
    state = {
        "messages": [HumanMessage(content="Find X and add Y")],
        "current_task": "Find the revenue of company X in documents",
        "scratchpad": "",
        "steps_remaining": 10
    }
    
    def mock_search(query):
        assert query == "Find the revenue of company X in documents"
        return [{"text": "Company X revenue is $10M", "source": "docs.pdf"}]
        
    # We patch the model invoke to avoid actual network call if wanted, or we mock it
    mock_llm_response = Mock()
    mock_llm_response.content = "Company X revenue is $10M."
    
    with patch("src.agents.rag_worker.get_reasoning_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = rag_worker_node(state, document_tool=mock_search)
        
        assert "scratchpad" in result
        assert "- [RAG Worker]: Company X revenue is $10M." in result["scratchpad"]
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["worker_type"] == "rag_worker"
        assert result["next_agent"] == "supervisor"


def test_cooperative_web_worker_updates_scratchpad():
    """Test that web_worker executes current_task and updates scratchpad."""
    state = {
        "messages": [HumanMessage(content="Find X and check weather")],
        "current_task": "Search the web for Paris weather",
        "scratchpad": "- [RAG Worker]: Company X revenue is $10M.",
        "steps_remaining": 10
    }
    
    def mock_search(query):
        assert query == "Search the web for Paris weather"
        return [{"title": "Weather", "content": "Paris is sunny and 22C", "url": "http://weather.com"}]
        
    mock_llm_response = Mock()
    mock_llm_response.content = "Paris weather is sunny, 22C."
    
    with patch("src.agents.web_worker.get_reasoning_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = web_worker_node(state, web_search_tool=mock_search)
        
        assert "scratchpad" in result
        assert "- [RAG Worker]: Company X revenue is $10M." in result["scratchpad"]
        assert "- [Web Worker]: Paris weather is sunny, 22C." in result["scratchpad"]
        assert result["worker_type"] == "web_worker"
        assert result["next_agent"] == "supervisor"


def test_cooperative_utility_worker_updates_scratchpad():
    """Test that utility_worker executes current_task and updates scratchpad."""
    state = {
        "messages": [HumanMessage(content="Calculate 5 * 25")],
        "current_task": "Calculate 5 * 25",
        "scratchpad": "- [Web Worker]: Found number 25.",
        "steps_remaining": 10
    }
    
    result = utility_worker_node(state)
    
    assert "scratchpad" in result
    assert "Result: 125" in result["messages"][0].content
    assert "- [Utility Worker]: Result: 125" in result["scratchpad"]
    assert result["worker_type"] == "utility_worker"
    assert result["next_agent"] == "supervisor"


def test_synthesizer_compiles_final_answer():
    """Test that synthesizer merges scratchpad findings into a final_answer."""
    state = {
        "messages": [HumanMessage(content="What is Company X's revenue and the weather in Paris?")],
        "scratchpad": "- [RAG Worker]: Company X revenue is $10M.\n- [Web Worker]: Paris weather is sunny, 22C.",
        "steps_remaining": 10,
        "plan": []
    }
    
    mock_llm_response = Mock()
    mock_llm_response.content = "Company X's revenue is $10M and the weather in Paris is sunny, 22C."
    
    with patch("src.graph.synthesizer.get_reasoning_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = synthesizer_node(state)
        
        assert result["final_answer"] == "Company X's revenue is $10M and the weather in Paris is sunny, 22C."
        assert result["next_agent"] == "FINISH"


def test_supervisor_planning_output():
    """Test supervisor planning output parsing and state update."""
    state = {
        "messages": [HumanMessage(content="Get revenue and compute 15% tax")],
        "plan": [],
        "scratchpad": "",
        "steps_remaining": 10
    }
    
    from src.graph.supervisor import SupervisorDecision
    mock_llm_response = SupervisorDecision(
        plan=["Find revenue", "Compute tax"],
        next_agent="rag_worker",
        current_task="Retrieve company revenue"
    )
    
    with patch("src.graph.supervisor.get_routing_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = supervisor_node(state)
        
        assert result["plan"] == ["Find revenue", "Compute tax"]
        assert result["next_agent"] == "rag_worker"
        assert result["current_task"] == "Retrieve company revenue"
        assert result["steps_remaining"] == 9


def test_supervisor_detects_human_approval():
    """Test that supervisor detects human approval and executes the pending tool."""
    state = {
        "messages": [
            HumanMessage(content="Write a new helper function"),
            AIMessage(content="I want to modify the file but direct execution of 'create_files' for 'workspace/helper.py' is blocked pending user approval."),
            HumanMessage(content="Yes, please approve and apply it.")
        ],
        "plan": ["Write code"],
        "scratchpad": "Direct execution of 'create_files' for 'workspace/helper.py' is blocked pending user approval.",
        "steps_remaining": 10
    }
    
    from src.graph.supervisor import SupervisorDecision
    mock_llm_response = SupervisorDecision(
        plan=["Write code"],
        next_agent="coding_worker",
        current_task="Create helper.py"
    )
    
    with patch("src.graph.supervisor.get_routing_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = supervisor_node(state)
        
        assert "[SYSTEM HITL]: User approved modifications" in result["scratchpad"]


def test_supervisor_detects_human_rejection():
    """Test that supervisor detects human rejection and clears the approval state."""
    state = {
        "messages": [
            HumanMessage(content="Write a new helper function"),
            AIMessage(content="I want to modify the file but direct execution of 'create_files' for 'workspace/helper.py' is blocked pending user approval."),
            HumanMessage(content="No, cancel it.")
        ],
        "plan": ["Write code"],
        "scratchpad": "Direct execution of 'create_files' for 'workspace/helper.py' is blocked pending user approval.",
        "steps_remaining": 10,
        "waiting_for_approval": True,
        "pending_file_approvals": {"workspace/helper.py": {"approved": False}}
    }
    
    from src.graph.supervisor import SupervisorDecision
    mock_llm_response = SupervisorDecision(
        plan=["Write code"],
        next_agent="coding_worker",
        current_task="Create helper.py"
    )
    
    with patch("src.graph.supervisor.get_routing_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = supervisor_node(state)
        
        assert "[SYSTEM HITL]: User rejected the proposed file modifications" in result["scratchpad"]
        assert result.get("waiting_for_approval") == False
        assert result.get("pending_file_approvals") == {}

