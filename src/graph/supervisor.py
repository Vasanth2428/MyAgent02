# Supervisor node for routing queries to specialized workers.
import os
import logging
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("MultiAgent.Supervisor")

SUPERVISOR_PROMPT = """You are a supervisor routing user queries to specialized workers. Available workers:
- rag_worker: Answers from private documents only, no external info. Use when the user asks about specific documents, files, or knowledge that should be derived from uploaded content.
- web_worker: Fetches live web data. Use when the user asks for current events, real-time info, or topics not likely in documents.
- utility_worker: Handles calculations, dates, formatting, and other deterministic tasks.

Analyze the conversation history and decide which worker should handle this query. Output only a JSON object with key 'next_agent' and value being one of: 'rag_worker', 'web_worker', 'utility_worker', or 'FINISH'.
Only finish if the query has been fully answered or if no worker can help.

Examples:
- User asks "What does my document say about X?" -> rag_worker
- User asks "What is the weather today?" -> web_worker  
- User asks "Calculate 2+2" -> utility_worker
- User asks "Who am I?" after greeting -> FINISH (can't answer without context)
"""


def get_routing_model():
    """Get the LLM model for routing (uses cheaper model)."""
    model_name = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


def supervisor_node(state: dict) -> dict:
    """
    Supervisor node that routes to appropriate worker.
    
    Uses structured output to force JSON with 'next_agent' field.
    """
    model = get_routing_model()
    
    messages = state.get("messages", [])
    context_notes = state.get("context_notes", [])
    
    routing_prompt = []
    routing_prompt.append(SystemMessage(content=SUPERVISOR_PROMPT))
    
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                routing_prompt.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                routing_prompt.append(AIMessage(content=msg.get("content", "")))
        else:
            routing_prompt.append(msg)
    
    if context_notes:
        routing_prompt.append(SystemMessage(content=f"Additional context notes: {' '.join(context_notes)}"))
    
    try:
        response = model.invoke(routing_prompt)
        content = response.content.strip() if response.content else ""
        
        import json
        import re
        json_match = re.search(r'\{.*\}', content)
        if json_match:
            result = json.loads(json_match.group(0))
            next_agent = result.get("next_agent", "FINISH")
        else:
            next_agent = "FINISH"
    except Exception as e:
        logger.error(f"Supervisor routing error: {e}")
        next_agent = "FINISH"
    
    valid_workers = ["rag_worker", "web_worker", "utility_worker", "FINISH"]
    if next_agent not in valid_workers:
        next_agent = "FINISH"
    
    return {"next_agent": next_agent}
