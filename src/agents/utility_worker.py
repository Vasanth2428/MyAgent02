# Utility worker node - handles deterministic tasks.
import os
import logging
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger("MultiAgent.UtilityWorker")

UTILITY_SYSTEM_PROMPT = """You are a utility agent. You can only:
1. Use calculator for mathematical operations
2. Get current datetime using get_current_datetime
3. Summarize text using summarize_text

For anything else, respond: "I can only perform calculations, date/time queries, or summarization."
Do not answer general knowledge questions.
"""


from src.core.config import UTILITY_WORKER_MODEL_PRIMARY, UTILITY_WORKER_MODEL_FALLBACK
from src.core.model_provider import build_model_with_fallback, message_text

def get_routing_model():
    """Get the configured LLM model for utility reasoning."""
    return build_model_with_fallback(
        "utility_worker",
        UTILITY_WORKER_MODEL_PRIMARY,
        UTILITY_WORKER_MODEL_FALLBACK,
        temperature=0,
        api_key_envs=("AGENT_API_KEY",),
    )


def _build_utility_response(response: str, scratchpad: str, result_type: str = "Utility Worker") -> dict:
    """Helper to build consistent worker response with completion tracking."""
    updated = scratchpad + f"\n- [{result_type}]: {response}"
    return {
        "messages": [AIMessage(content=response, name="utility_worker")],
        "scratchpad": updated,
        "worker_complete": {result_type.lower().replace(" ", "_"): True},
        "worker_outputs": {result_type.lower().replace(" ", "_"): response},
        "worker_type": result_type.lower().replace(" ", "_"),
        "next_agent": "supervisor"
    }


def utility_worker_node(state: dict) -> dict:
    """
    Utility worker for calculations, datetime, and summarization.
    """
    from src.tools.utility_tools import evaluate_math, get_current_datetime, summarize_text
    from src.tools.safety_filters import sanitize_user_input
    
    current_task = state.get("current_task", "")
    scratchpad = state.get("scratchpad", "")
    
    target_query = current_task if current_task else ""
    if not target_query:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                target_query = sanitize_user_input(msg.get("content", ""))
                break
            elif isinstance(msg, HumanMessage):
                target_query = sanitize_user_input(msg.content)
                break
                
    if not target_query:
        return _build_utility_response("No query provided.", scratchpad, "Utility Worker")
    
    query_lower = target_query.lower()
    try:
        # Prevent collisions with code or repository analysis tasks
        coding_keywords = ["code", "file", "repository", "database", "class", "function", "compile", "bug", "syntax", "develop", "config", "logic flaws"]
        if any(kw in query_lower for kw in coding_keywords):
            print(f"\n[UTILITY WORKER] Coding-related query detected. Advising supervisor to route to coding worker.")
            fallback_msg = "This request involves code or repository analysis. Please route code/repository analysis tasks to the coding specialist."
            return _build_utility_response(fallback_msg, scratchpad, "Utility Worker")

        if any(word in query_lower for word in ["calculate", "math", "+", "-", "*", "/", "times", "plus", "minus", "divide", "multiply", "sum", "difference"]):
            import re
            normalized_query = (
                target_query.lower()
                .replace("plus", "+")
                .replace("minus", "-")
                .replace("times", "*")
                .replace("divided by", "/")
                .replace("divide", "/")
            )
            
            if any(word in query_lower for word in ["compare", "larger", "smaller", "difference", "which", "how much"]) or len(target_query.split()) > 10:
                print(f"\n[UTILITY WORKER] Math query detected (complex). Solving via LLM reasoning...")
                model = get_routing_model()
                prompt = (
                    "You are an expert mathematical assistant. Solve the following query step-by-step. "
                    "Make sure to explain your steps and calculate the final answer clearly.\n\n"
                    f"Query: {target_query}"
                )
                response = model.invoke([
                    SystemMessage(content=UTILITY_SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ])
                safe_response = message_text(response)
                print(f"[UTILITY WORKER] Response:\n{safe_response}")
                return _build_utility_response(safe_response, scratchpad, "Utility Worker")
                
            expr_match = re.search(r'[\d\-\(][\d\s\+\-\*\/\.\(\)]*', normalized_query)
            if expr_match:
                expr_str = expr_match.group(0).strip()
                print(f"\n[UTILITY WORKER] Math query detected (simple: '{expr_str}'). Evaluating directly...")
                result = evaluate_math(expr_str)
                print(f"[UTILITY WORKER] Result: {result}")
                return _build_utility_response(f"Result: {result}", scratchpad, "Utility Worker")
        
        if any(word in query_lower for word in ["time", "date", "today", "now", "current", "datetime"]):
            print(f"\n[UTILITY WORKER] Datetime query detected. Retrieving current time...")
            result = get_current_datetime()
            print(f"[UTILITY WORKER] Result: {result}")
            return _build_utility_response(f"Current datetime: {result}", scratchpad, "Utility Worker")
        
        if any(word in query_lower for word in ["summarize", "summary", "condense", "shorten"]):
            print(f"\n[UTILITY WORKER] Summarization query detected...")
            # Look for text to summarize in the scratchpad or messages first
            text_to_summarize = ""
            if scratchpad:
                text_to_summarize = scratchpad
            else:
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    if content and not any(w in content.lower() for w in ["summarize", "summary", "condense", "shorten"]):
                        text_to_summarize = content
                        break
            if text_to_summarize:
                print(f"[UTILITY WORKER] Found text to summarize ({len(text_to_summarize)} chars). Running summarizer...")
                summary = summarize_text(text_to_summarize)
                return _build_utility_response(summary, scratchpad, "Utility Worker")
            else:
                return _build_utility_response("Please provide the text you'd like me to summarize.", scratchpad, "Utility Worker")
        
        fallback_msg = "I can only perform calculations, date/time queries, or summarization."
        return _build_utility_response(fallback_msg, scratchpad, "Utility Worker")
    except Exception as e:
        logger.error(f"Utility worker error: {e}")
        return _build_utility_response("Error performing operation. Please try again.", scratchpad, "Utility Worker")
