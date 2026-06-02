# Supervisor node for routing queries to specialized workers.
import os
import logging
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.Supervisor")

SUPERVISOR_PROMPT = """You are a ROUTING-ONLY supervisor. Your ONLY job is to decide which worker handles the query.
Do NOT answer the query yourself. Do NOT solve math. Do NOT provide information.

Available workers:
- rag_worker: Use for questions about uploaded documents, files, or private knowledge.
- web_worker: Use for current events, real-time info, news, or topics requiring live web data.
- utility_worker: Use for ANY math, calculations, dates, times, formatting, or summarization.

Rules:
1. Output ONLY a JSON object: {"next_agent": "<worker_name>"}
2. next_agent MUST be exactly one of: "rag_worker", "web_worker", "utility_worker", or "FINISH"
3. Use FINISH only if the conversation already contains a complete answer.
4. When in doubt, route to a worker rather than FINISH.

Examples:
- User: "What does my document say about X?" -> {"next_agent": "rag_worker"}
- User: "What is the weather today?" -> {"next_agent": "web_worker"}
- User: "What is 2+2?" -> {"next_agent": "utility_worker"}
- User: "What is 2+2?" -> Assistant: "Result: 4" -> {"next_agent": "FINISH"}
- User: "What is the weather today?" -> Assistant: "The weather today is sunny." -> {"next_agent": "FINISH"}
"""


def get_routing_model():
    """Get the LLM model for routing (uses cheaper model via Groq)."""
    model_name = os.getenv("SUPERVISOR_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("AGENT_API_KEY")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


def supervisor_node(state: dict) -> dict:
    """
    Supervisor node that routes to appropriate worker.
    
    Uses structured output to force JSON with 'next_agent' field.
    """
    model = get_routing_model()
    
    messages = state.get("messages", [])
    context_notes = state.get("context_notes", [])
    steps = state.get("steps_remaining", 10)
    
    # Bounded execution loop: decrement steps
    new_steps = steps - 1
    
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
            next_agent = result.get("next_agent") or "FINISH"
        else:
            next_agent = "FINISH"
    except Exception as e:
        logger.error(f"Supervisor routing error: {e}")
        next_agent = "FINISH"
    
    valid_workers = ["rag_worker", "web_worker", "utility_worker", "FINISH"]
    if next_agent not in valid_workers:
        next_agent = "FINISH"
    
    state_update = {
        "next_agent": next_agent,
        "steps_remaining": new_steps
    }
    
    # If finishing, populate final_answer from the last assistant message
    if next_agent == "FINISH":
        last_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_answer = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                last_answer = msg.get("content", "")
                break
        if last_answer:
            state_update["final_answer"] = last_answer
            
    print(f"\n[SUPERVISOR] Routing decision: next_agent = '{next_agent}' | Steps remaining: {new_steps}")
    return state_update
