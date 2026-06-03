# Tests for the utility worker.
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.utility_worker import utility_worker_node


def test_utility_worker_math_simple():
    """Test utility worker calculates simple math expressions."""
    state = {
        "messages": [HumanMessage(content="What is 2+2?")],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert "messages" in result
    assert result["next_agent"] == "supervisor"
    assert "Result: 4" in result["messages"][0].content


def test_utility_worker_math_verbal():
    """Test utility worker handles verbal math operators."""
    state = {
        "messages": [HumanMessage(content="what is 5 plus 6 times 2?")],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert "messages" in result
    assert result["next_agent"] == "supervisor"
    # 5 + 6 * 2 = 17 (due to standard operator precedence)
    assert "Result: 17" in result["messages"][0].content


def test_utility_worker_datetime():
    """Test utility worker returns current datetime."""
    state = {
        "messages": [HumanMessage(content="What is the current time?")],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert "messages" in result
    assert result["next_agent"] == "supervisor"
    assert "Current datetime:" in result["messages"][0].content


def test_utility_worker_summarize():
    """Test utility worker handles summarization request prompts."""
    state = {
        "messages": [HumanMessage(content="Can you summarize this text?")],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert "messages" in result
    assert result["next_agent"] == "supervisor"
    assert "provide the text you'd like me to summarize" in result["messages"][0].content.lower()


def test_utility_worker_fallback():
    """Test utility worker falls back on general knowledge query."""
    state = {
        "messages": [HumanMessage(content="Who was the first man on the moon?")],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert "messages" in result
    assert result["next_agent"] == "supervisor"
    assert "I can only perform calculations" in result["messages"][0].content


def test_utility_worker_empty_query():
    """Test utility worker handles state with no user query."""
    state = {
        "messages": [],
        "next_agent": "utility_worker",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    result = utility_worker_node(state)
    assert result["next_agent"] == "supervisor"
    assert "No query provided" in result["messages"][0].content