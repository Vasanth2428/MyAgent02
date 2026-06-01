# Utility worker node - handles deterministic tasks.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("MultiAgent.UtilityWorker")

UTILITY_SYSTEM_PROMPT = """You are a utility agent. You can only:
1. Use calculator for mathematical operations
2. Get current datetime using get_current_datetime
3. Summarize text using summarize_text

For anything else, respond: "I can only perform calculations, date/time queries, or summarization."
Do not answer general knowledge questions.
"""


def get_routing_model():
    """Get the LLM model for routing (uses cheaper model)."""
    model_name = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


def utility_worker_node(state: dict) -> dict:
    """
    Utility worker for calculations, datetime, and summarization.
    """
    from src.tools.utility_tools import evaluate_math, get_current_datetime, summarize_text
    from src.tools.safety_filters import sanitize_user_input
    
    messages = state.get("messages", [])
    
    last_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_query = sanitize_user_input(msg.get("content", ""))
            break
        elif isinstance(msg, HumanMessage):
            last_user_query = sanitize_user_input(msg.content)
            break
    
    if not last_user_query:
        return {"final_answer": "No query provided.", "next_agent": "FINISH"}
    
    query_lower = last_user_query.lower()
    
    try:
        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "times", "plus", "minus", "divide", "multiply"]):
            import re
            expr_match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', last_user_query)
            if expr_match:
                result = evaluate_math(expr_match.group(0))
                return {
                    "messages": [AIMessage(content=f"Result: {result}")],
                    "next_agent": "FINISH"
                }
        
        if any(word in query_lower for word in ["time", "date", "today", "now", "current", "datetime"]):
            result = get_current_datetime()
            return {
                "messages": [AIMessage(content=f"Current datetime: {result}")],
                "next_agent": "FINISH"
            }
        
        if any(word in query_lower for word in ["summarize", "summary", "condense", "shorten"]):
            return {
                "messages": [AIMessage(content="Please provide the text you'd like me to summarize.")],
                "next_agent": "FINISH"
            }
        
        return {
            "messages": [AIMessage(content="I can only perform calculations, date/time queries, or summarization.")],
            "next_agent": "FINISH"
        }
    except Exception as e:
        logger.error(f"Utility worker error: {e}")
        return {
            "messages": [AIMessage(content="Error performing operation. Please try again.")],
            "next_agent": "FINISH"
        }
