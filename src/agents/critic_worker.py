# Critic & analysis worker node - fact-checks scratchpad findings and verifies consistency.
import os
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import CRITIC_MODEL_PRIMARY, CRITIC_MODEL_FALLBACK

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
    validation_key = os.getenv("GROQ_VALIDATION_KEY")
    api_key = validation_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=CRITIC_MODEL_PRIMARY, temperature=0, api_key=api_key)
    fallback = ChatGroq(model=CRITIC_MODEL_FALLBACK, temperature=0, api_key=api_key)
    return primary.with_fallbacks([fallback])


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
        
        retry_count = state.get("critic_retry_count", 0)
        
        if "RETRY_REQUIRED" in safe_response:
            if retry_count >= 2:
                # On second retry, assign task to a different agent approach
                # Remove the RETRY_REQUIRED token and append failure message
                safe_response = safe_response.replace("RETRY_REQUIRED", "").strip()
                safe_response += "\n[Max validation retry limit reached. Verification failed after multiple attempts. Proceeding without further retries.]"
                updated_scratchpad = scratchpad + f"\n- [Critic Worker]: Verification failed repeatedly. Aborting corrections to prevent infinite loop.\nFact-Check:\n{safe_response}"
                
                # For second retry, suggest trying a completely different approach
                # by modifying the plan to consult different types of sources
                current_plan = state.get("plan", [])
                # Add a step that suggests consulting web sources if we were using docs, or vice versa
                if any("web" in str(item).lower() or "scrape" in str(item).lower() for item in current_plan):
                    # If we were trying web approaches, suggest document approach
                    state_update_plan = current_plan + ["CONSULT DOCUMENTS: Review relevant documentation for verified information before proceeding."]
                elif any("rag" in str(item).lower() or "document" in str(item).lower() for item in current_plan):
                    # If we were trying document approaches, suggest web approach  
                    state_update_plan = current_plan + ["VERIFY ONLINE: Check current information from reliable web sources to confirm findings."]
                else:
                    # Generic different approach
                    state_update_plan = current_plan + ["ALTERNATIVE APPROACH: Use a different methodology or source type to verify the findings."]
                    
                state_update = {
                    "messages": [AIMessage(content=safe_response, name="critic_worker")],
                    "scratchpad": updated_scratchpad,
                    "worker_complete": {"critic_worker": True},
                    "worker_outputs": {"critic_worker": safe_response},
                    "worker_type": "critic_worker",
                    "next_agent": "supervisor",
                    "critic_retry_count": retry_count,  # Don't increment on final retry
                    "plan": state_update_plan
                }
            else:
                # On first retry, give hints/easier settings to the same worker
                # Extract key findings to provide specific guidance
                feedback_lines = safe_response.split('\n')
                specific_feedback = []
                for line in feedback_lines:
                    if any(keyword in line.lower() for keyword in ['contradiction', 'gap', 'error', 'mistake', 'inconsistency', 'should', 'need to']):
                        specific_feedback.append(line.strip())
                
                hints = " ".join(specific_feedback[:2]) if specific_feedback else "Review findings carefully and verify accuracy"
                
                updated_scratchpad = scratchpad + f"\n- [Critic Worker]: Fact-Check Analysis:\n{safe_response}"
                
                # For first retry, enhance current_task with specific hints for the worker
                enhanced_task = f"{current_task} [HINT: {hints}]" if hints else current_task
                
                state_update = {
                    "messages": [AIMessage(content=safe_response, name="critic_worker")],
                    "scratchpad": updated_scratchpad,
                    "worker_complete": {"critic_worker": True},
                    "worker_outputs": {"critic_worker": safe_response},
                    "worker_type": "critic_worker",
                    "next_agent": "supervisor",
                    "critic_retry_count": retry_count + 1,
                    "current_task": enhanced_task
                }
        else:
            updated_scratchpad = scratchpad + f"\n- [Critic Worker]: Fact-Check Analysis:\n{safe_response}"
            
            state_update = {
                "messages": [AIMessage(content=safe_response, name="critic_worker")],
                "scratchpad": updated_scratchpad,
                "worker_complete": {"critic_worker": True},
                "worker_outputs": {"critic_worker": safe_response},
                "worker_type": "critic_worker",
                "next_agent": "supervisor",
                "critic_retry_count": retry_count
            }
            
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
