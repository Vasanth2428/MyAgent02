import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.scraper_worker import scraper_worker_node
from src.agents.critic_worker import critic_worker_node


def test_scraper_worker_node_success():
    """Test that scraper_worker_node scrapes the URL and appends the LLM summary to scratchpad."""
    state = {
        "messages": [HumanMessage(content="Scrape competitor specs at https://competitor.com/specs")],
        "current_task": "Scrape competitor specs at https://competitor.com/specs",
        "scratchpad": "- [RAG Worker]: Document specs retrieved.",
        "steps_remaining": 10
    }
    
    def mock_scraper(url):
        assert url == "https://competitor.com/specs"
        return "Competitor specs: 4GB RAM, 64GB storage, $199 price."
        
    mock_llm_response = Mock()
    mock_llm_response.content = "Competitor specs: 4GB RAM, 64GB storage, $199."
    
    with patch("src.agents.scraper_worker.get_reasoning_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = scraper_worker_node(state, scraper_tool=mock_scraper)
        
        # Must execute async code wrapped in sync/mock
        import asyncio
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
            
        assert "scratchpad" in result
        assert "- [RAG Worker]: Document specs retrieved." in result["scratchpad"]
        assert "- [Scraper Worker]: Content from https://competitor.com/specs:\nCompetitor specs: 4GB RAM, 64GB storage, $199." in result["scratchpad"]
        assert result["next_agent"] == "supervisor"


def test_scraper_worker_node_no_url():
    """Test that scraper_worker_node logs error when no URL is found in task or context."""
    state = {
        "messages": [HumanMessage(content="Retrieve web specs")],
        "current_task": "Retrieve web specs without link",
        "scratchpad": "",
        "steps_remaining": 10
    }
    
    # Run the scraper node
    import asyncio
    result = scraper_worker_node(state)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
        
    assert "scratchpad" in result
    assert "could not locate a valid URL" in result["messages"][0].content
    assert "- [Scraper Worker]: Error: Scraper worker could not locate a valid URL to fetch." in result["scratchpad"]
    assert result["next_agent"] == "supervisor"


def test_critic_worker_node_success():
    """Test that critic_worker_node fact-checks current findings and appends analysis."""
    state = {
        "messages": [HumanMessage(content="Check if our specs match competitors")],
        "current_task": "Fact-check competitors specs comparison",
        "scratchpad": "- [RAG Worker]: Our specs are 8GB RAM.\n- [Scraper Worker]: Competitor specs are 4GB RAM.",
        "steps_remaining": 10
    }
    
    mock_llm_response = Mock()
    mock_llm_response.content = "Comparison: Our specs (8GB) are double the competitor specs (4GB)."
    
    with patch("src.agents.critic_worker.get_reasoning_model") as mock_get_model:
        mock_model = Mock()
        mock_model.invoke.return_value = mock_llm_response
        mock_get_model.return_value = mock_model
        
        result = critic_worker_node(state)
        
        assert "scratchpad" in result
        assert "- [RAG Worker]: Our specs are 8GB RAM." in result["scratchpad"]
        assert "- [Critic Worker]: Fact-Check Analysis:\nComparison: Our specs (8GB) are double the competitor specs (4GB)." in result["scratchpad"]
        assert result["next_agent"] == "supervisor"
