import os
import logging
from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import SYNTHESIZER_MODEL_PRIMARY, SYNTHESIZER_MODEL_FALLBACK

logger = logging.getLogger("MultiAgent.Synthesizer")


def get_reasoning_model():
    """Get the LLM model for synthesis via Groq."""
    primary_key = os.getenv("GROQ_API_KEY")
    api_key = primary_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=SYNTHESIZER_MODEL_PRIMARY, temperature=0.3, api_key=api_key)
    fallback = ChatGroq(model=SYNTHESIZER_MODEL_FALLBACK, temperature=0.3, api_key=api_key)
    return primary.with_fallbacks([fallback])


def synthesizer_node(state: dict) -> dict:
    """
    Synthesizes the final response using the original query and accumulated scratchpad findings.
    Checks if all planned tasks are complete before finishing.
    """
    logger.info("Synthesizer node executing...")
    
    messages = state.get("messages", [])
    scratchpad = state.get("scratchpad", "")
    plan = state.get("plan", [])
    worker_complete = state.get("worker_complete", {})
    
    # Filter worker messages out of the chat history
    worker_names = {"rag_worker", "web_worker", "utility_worker", "scraper_worker", "critic_worker", "code_critic_worker"}
    clean_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("name") in worker_names:
                continue
            clean_messages.append(msg)
        else:
            if hasattr(msg, "name") and msg.name in worker_names:
                continue
            clean_messages.append(msg)
            
    # Retrieve original user query from messages (in reverse to get the latest intent)
    original_query = ""
    last_human_idx = -1
    for i, msg in enumerate(clean_messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                last_human_idx = i
        elif hasattr(msg, "content") and (msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human"):
            last_human_idx = i

    if last_human_idx != -1:
        msg = clean_messages[last_human_idx]
        original_query = msg.get("content", "") if isinstance(msg, dict) else msg.content
    elif clean_messages:
        last_msg = clean_messages[-1]
        original_query = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")

    # Format previous conversation history (excluding the current user query)
    chat_history_parts = []
    for idx, msg in enumerate(clean_messages):
        if idx == last_human_idx:
            continue
        
        role = ""
        content = ""
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        else:
            content = msg.content
            if msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human":
                role = "user"
            elif msg.__class__.__name__ == "AIMessage" or getattr(msg, "type", "") == "ai":
                role = "assistant"
        
        if role and content:
            chat_history_parts.append(f"[{role}]: {content}")
            
    chat_history_str = "\n".join(chat_history_parts) if chat_history_parts else "(No previous messages)"
        
    synthesis_prompt = f"""You are an expert assistant. Your job is to construct a direct, coherent, and comprehensive response to the user's original query by synthesizing the findings gathered by your specialized team of workers, taking into account the conversation history.

Conversation History:
{chat_history_str}

User Original Query:
"{original_query}"

Accumulated Cooperative Findings:
{scratchpad if scratchpad else "(No findings retrieved)"}

Formatting Guidelines:
1. Provide a direct and structured response answering the query.
2. Use markdown formatting (headers, lists, bold text) for clarity.
3. If findings are missing or contradictory, resolve them logically or state the missing details clearly.
4. Do NOT mention the internal team, 'scratchpad', 'workers', or 'agents' in the final response. Present the answer as a unified response from a single assistant.
5. IMPORTANT: If the task involved creating, writing, or editing files in the workspace (such as HTML, CSS, JS, Python or JSON file operations), do NOT output any file summaries, long code listings, explanations, or next steps. Instead, return a minimal, single-sentence response confirming the action, e.g., 'The file [filename] was successfully created/modified and stored in the workspace directory.'
"""

    model = get_reasoning_model()
    try:
        response = model.invoke([
            SystemMessage(content="You are a helpful AI assistant synthesizing information."),
            HumanMessage(content=synthesis_prompt)
        ])
        final_answer = response.content.strip() if response.content else ""
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error in synthesizer node: {error_str}")
        # Surface auth errors clearly to the user
        if "401" in error_str or "invalid_api_key" in error_str.lower() or "Invalid API Key" in error_str:
            final_answer = "⚠️ **System Error: LLM API Authentication Failed**\n\nThe AI service returned an authentication error. This usually means the API key has expired or was changed after the server started.\n\n**To fix this:**\n1. Verify your API keys in the `.env` file are valid\n2. Restart the server so it loads the updated keys\n\nTechnical detail: " + error_str[:200]
        elif "429" in error_str or "rate_limit" in error_str.lower():
            final_answer = "⚠️ **Rate Limit Exceeded**\n\nThe AI service is temporarily rate-limited. Please wait a moment and try again."
        elif scratchpad and "[SYSTEM ERROR]" in scratchpad:
            final_answer = f"⚠️ **System Error**\n\nThe pipeline encountered errors during processing:\n\n{scratchpad}"
        else:
            final_answer = f"⚠️ **Error generating response**\n\nHere are the raw findings gathered:\n\n{scratchpad}" if scratchpad else "⚠️ An internal error occurred. Please try again."
        
    try:
        print(f"\n[SYNTHESIZER] Compiled final answer: {final_answer[:60]}...")
    except Exception:
        try:
            safe_print = final_answer[:60].encode('ascii', errors='replace').decode('ascii')
            print(f"\n[SYNTHESIZER] Compiled final answer: {safe_print}...")
        except Exception:
            pass
    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "worker_complete": worker_complete,
        "next_agent": "FINISH"
    }
