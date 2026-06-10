# Web search worker node - fetches live web data.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.WebWorker")

WEB_SYSTEM_PROMPT = """You are a web search agent. Use the web_search tool to find up-to-date information.
Provide source URLs when answering. Always cite your sources.
Structure responses with headers, bullet points, and clear formatting.
"""


from src.core.config import WEB_WORKER_MODEL_PRIMARY, WEB_WORKER_MODEL_FALLBACK

def get_reasoning_model():
    """Get the LLM model for complex reasoning via Groq."""
    api_key = os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=WEB_WORKER_MODEL_PRIMARY, temperature=0, api_key=api_key)
    fallback = ChatGroq(model=WEB_WORKER_MODEL_FALLBACK, temperature=0, api_key=api_key)
    return primary.with_fallbacks([fallback])


def web_worker_node(state: dict, web_search_tool: callable = None) -> dict:
    """
    Web search worker that fetches live information.
    
    Args:
        state: Current state with messages.
        web_search_tool: Function to perform web search.
    """
    from src.tools.safety_filters import sanitize_user_input, validate_tool_output, truncate_results
    
    if web_search_tool is None:
        from src.tools.web_search_tool import web_search, format_search_results
        web_search_tool = web_search
    
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
        return {
            "messages": [AIMessage(content="No query provided.", name="web_worker")],
            "scratchpad": scratchpad,
            "worker_complete": {"web_worker": True},
            "worker_outputs": {"web_worker": "No query provided."},
            "worker_type": "web_worker",
            "next_agent": "supervisor"
        }
    
    try:
        print(f"\n[WEB WORKER] Executing web search for: '{target_query}'")
        results = web_search_tool(target_query)
        results = truncate_results(results)
        
        if not results:
            print("[WEB WORKER] No relevant web search results found.")
            no_web_msg = "I couldn't find any relevant web results for your query."
            updated_scratchpad = scratchpad + f"\n- [Web Worker]: {no_web_msg}"
            return {
                "messages": [AIMessage(content=no_web_msg, name="web_worker")],
                "scratchpad": updated_scratchpad,
                "worker_complete": {"web_worker": True},
                "worker_outputs": {"web_worker": no_web_msg},
                "worker_type": "web_worker",
                "next_agent": "supervisor"
            }
        
        print(f"[WEB WORKER] Found {len(results)} search results. Synthesizing answer...")
        context = "\n\n".join([f"- {validate_tool_output(r.get('title', ''))}: {validate_tool_output(r.get('content', ''))}" for r in results])
        sources = ", ".join([r.get('url', '') for r in results if r.get('url')])
        
        model = get_reasoning_model()
        response = model.invoke([
            SystemMessage(content=WEB_SYSTEM_PROMPT),
            HumanMessage(content=f"Search results:\n{context}\n\nQuestion: {target_query}\n\nSources: {sources}")
        ])
        
        safe_response = validate_tool_output(response.content)
        print(f"[WEB WORKER] Response:\n{safe_response}")
        
        updated_scratchpad = scratchpad + f"\n- [Web Worker]: {safe_response}"
        return {
            "messages": [AIMessage(content=safe_response, name="web_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"web_worker": True},
            "worker_outputs": {"web_worker": safe_response},
            "worker_type": "web_worker",
            "next_agent": "supervisor"
        }
    except Exception as e:
        logger.error(f"Web worker error: {e}")
        err_msg = "Error searching web. Please try again."
        updated_scratchpad = scratchpad + f"\n- [Web Worker]: {err_msg}"
        return {
            "messages": [AIMessage(content=err_msg, name="web_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"web_worker": True},
            "worker_outputs": {"web_worker": err_msg},
            "worker_type": "web_worker",
            "next_agent": "supervisor"
        }