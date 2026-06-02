# Utility worker node - handles deterministic tasks.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.UtilityWorker")

UTILITY_SYSTEM_PROMPT = """You are a utility agent. You can only:
1. Use calculator for mathematical operations
2. Get current datetime using get_current_datetime
3. Summarize text using summarize_text

For anything else, respond: "I can only perform calculations, date/time queries, or summarization."
Do not answer general knowledge questions.
"""


def get_routing_model():
    """Get the LLM model for routing via Groq."""
    model_name = os.getenv("SUPERVISOR_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("AGENT_API_KEY")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


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
        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "times", "plus", "minus", "divide", "multiply", "sum", "difference"]):
            import re
            # Normalize verbal operators to symbols
            normalized_query = (
                last_user_query.lower()
                .replace("plus", "+")
                .replace("minus", "-")
                .replace("times", "*")
                .replace("divided by", "/")
                .replace("divide", "/")
            )
            
            # If the query is complex (has comparison words or is long), use LLM to solve it
            if any(word in query_lower for word in ["compare", "larger", "smaller", "difference", "which", "how much"]) or len(last_user_query.split()) > 10:
                print(f"\n[UTILITY WORKER] Math query detected (complex). Solving via LLM reasoning...")
                model = get_routing_model()
                prompt = (
                    "You are an expert mathematical assistant. Solve the following query step-by-step. "
                    "Make sure to explain your steps and calculate the final answer clearly.\n\n"
                    f"Query: {last_user_query}"
                )
                response = model.invoke([
                    SystemMessage(content=UTILITY_SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ])
                safe_response = response.content
                print(f"[UTILITY WORKER] Response:\n{safe_response}")
                return {
                    "messages": [AIMessage(content=safe_response)],
                    "next_agent": "FINISH"
                }
                
            # Find the actual mathematical expression starting with a digit, minus sign, or parenthesis
            expr_match = re.search(r'[\d\-\(][\d\s\+\-\*\/\.\(\)]*', normalized_query)
            if expr_match:
                expr_str = expr_match.group(0).strip()
                print(f"\n[UTILITY WORKER] Math query detected (simple: '{expr_str}'). Evaluating directly...")
                result = evaluate_math(expr_str)
                print(f"[UTILITY WORKER] Result: {result}")
                return {
                    "messages": [AIMessage(content=f"Result: {result}")],
                    "next_agent": "FINISH"
                }
        
        if any(word in query_lower for word in ["time", "date", "today", "now", "current", "datetime"]):
            print(f"\n[UTILITY WORKER] Datetime query detected. Retrieving current time...")
            result = get_current_datetime()
            print(f"[UTILITY WORKER] Result: {result}")
            return {
                "messages": [AIMessage(content=f"Current datetime: {result}")],
                "next_agent": "FINISH"
            }
        
        if any(word in query_lower for word in ["summarize", "summary", "condense", "shorten"]):
            print(f"\n[UTILITY WORKER] Summarization query detected...")
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
