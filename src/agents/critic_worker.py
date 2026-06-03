# Critic & analysis worker node - fact-checks scratchpad findings and verifies consistency.
import os
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.CriticWorker")

CRITIC_SYSTEM_PROMPT = """You are a Fact-Checking & Critic Specialist.
Your job is to critically evaluate the facts and findings accumulated on the blackboard scratchpad.

Your duties:
1. Cross-reference findings (e.g. check if the Web search contradicts the RAG document findings).
2. Highlight any inconsistencies, contradictions, logical errors, or calculation mistakes.
3. Suggest clear resolutions for these conflicts.
4. Keep your critique direct, objective, and factual. Focus strictly on verification.
5. CRITICAL: If you detect a severe hallucination, blatant falsehood, or critical gap that MUST be fixed, end your response with the exact token 'RETRY_REQUIRED'. If the findings are acceptable or only have minor nitpicks, do not include this token.
"""


def get_reasoning_model():
    """Get the LLM model for complex reasoning via Groq."""
    model_name = os.getenv("REASONING_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("AGENT_API_KEY")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)


def critic_worker_node(state: dict) -> dict:
    """
    Critic worker that evaluates factuality and consistency of accumulated findings.
    """
    from src.tools.safety_filters import sanitize_user_input, validate_tool_output
    
    messages = state.get("messages", [])
    scratchpad = state.get("scratchpad", "")
    current_task = state.get("current_task", "")
    
    # Retrieve original user query
    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            original_query = sanitize_user_input(msg.get("content", ""))
            break
        elif isinstance(msg, HumanMessage):
            original_query = sanitize_user_input(msg.content)
            break
            
    print(f"\n[CRITIC WORKER] Analyzing scratchpad findings...")
    
    critique_prompt = f"""Evaluate the following findings against the user inquiry:
User Query: "{original_query}"
Current Sub-Task: "{current_task}"

Findings Scratchpad:
{scratchpad if scratchpad else "(No findings logged yet)"}

Analyze and output:
- **Fact-check**: Are the findings consistent?
- **Discrepancies**: Identify any contradictions or gaps between document data and web facts.
- **Verification Status**: Confirm if the query is fully and accurately addressed.
"""
    
    model = get_reasoning_model()
    try:
        response = model.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=critique_prompt)
        ])
        safe_response = validate_tool_output(response.content)
        print(f"[CRITIC WORKER] Critique analysis complete.")
        
        updated_scratchpad = scratchpad + f"\n- [Critic Worker]: Fact-Check Analysis:\n{safe_response}"
        
        state_update = {
            "messages": [AIMessage(content=safe_response, name="critic_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"critic_worker": True},
            "worker_outputs": {"critic_worker": safe_response},
            "worker_type": "critic_worker",
            "next_agent": "supervisor"
        }
        
        if "RETRY_REQUIRED" in safe_response:
            print("[CRITIC WORKER] Hallucination detected! Forcing supervisor retry.")
            current_plan = state.get("plan", [])
            state_update["plan"] = current_plan + ["FIX ERROR: Review critic feedback and dispatch a worker to find the correct information."]
            
        return state_update
    except Exception as e:
        logger.error(f"Critic worker error: {e}")
        err_msg = "Error performing fact-check analysis."
        updated_scratchpad = scratchpad + f"\n- [Critic Worker]: {err_msg}"
        return {
            "messages": [AIMessage(content=err_msg, name="critic_worker")],
            "scratchpad": updated_scratchpad,
            "worker_complete": {"critic_worker": True},
            "worker_outputs": {"critic_worker": err_msg},
            "worker_type": "critic_worker",
            "next_agent": "supervisor"
        }
