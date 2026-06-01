# Web search worker node - fetches live web data.
import os
import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("MultiAgent.WebWorker")

WEB_SYSTEM_PROMPT = """You are a web search agent. Use the web_search tool to find up-to-date information.
Provide source URLs when answering. Always cite your sources.
Structure responses with headers, bullet points, and clear formatting.
"""


def get_reasoning_model():
    """Get the LLM model for complex reasoning."""
    model_name = os.getenv("REASONING_MODEL", "gpt-4o")
    return ChatOpenAI(model=model_name, temperature=0)


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
    
    try:
        results = web_search_tool(last_user_query)
        results = truncate_results(results)
        
        if not results:
            return {
                "messages": [AIMessage(content="I couldn't find any relevant web results for your query.")],
                "next_agent": "FINISH"
            }
        
        context = "\n\n".join([f"- {validate_tool_output(r.get('title', ''))}: {validate_tool_output(r.get('content', ''))}" for r in results])
        sources = ", ".join([r.get('url', '') for r in results if r.get('url')])
        
        model = get_reasoning_model()
        response = model.invoke([
            SystemMessage(content=WEB_SYSTEM_PROMPT),
            HumanMessage(content=f"Search results:\n{context}\n\nQuestion: {last_user_query}\n\nSources: {sources}")
        ])
        
        safe_response = validate_tool_output(response.content)
        
        return {
            "messages": [AIMessage(content=safe_response)],
            "next_agent": "FINISH"
        }
    except Exception as e:
        logger.error(f"Web worker error: {e}")
        return {
            "messages": [AIMessage(content="Error searching web. Please try again.")],
            "next_agent": "FINISH"
        }
