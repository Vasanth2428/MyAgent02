# Supervisor node for routing queries to specialized workers.
import os
import logging
from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import SUPERVISOR_MODEL_PRIMARY, SUPERVISOR_MODEL_FALLBACK

logger = logging.getLogger("MultiAgent.Supervisor")

SUPERVISOR_PROMPT = """You are a central planner and supervisor for a multi-agent cooperative system.
Your job is to coordinate a team of specialized workers to solve a user's query.

Available Workers:
- rag_worker: Retrieve information from private documents only. Use for questions about uploaded files, documents, or custom knowledge base.
- web_worker: Search the web for real-time/current information, news, or general external web search.
- utility_worker: Perform calculations, math, date/time lookups, and basic formatting.
- scraper_worker: Fetch and extract text content from specific URLs or links. Use this when the query or step explicitly requires reading the content of a web page.
- critic_worker: Analyze accumulated findings, cross-reference sources, fact-check, and identify inconsistencies or gaps. Use this to critique findings before synthesis.
- report_worker: Generate comprehensive, long-form markdown reports from the accumulated findings. Use this when the user explicitly requests a report or summary document.
- coding_worker: Code generation, file creation/editing, security auditing, code review, and architecture evaluation. Use for creating files, writing code, modifying existing code, and code analysis tasks.
- code_critic_worker: Validate findings from coding_worker against repository symbols, audit patch correctness, and check for security risks.

Your duties:
1. If the 'plan' is empty, construct a step-by-step plan (up to 3 steps) to answer the user query.
2. If a plan exists, evaluate the 'scratchpad' findings against the plan to check progress.
3. Determine the next step. If all research steps are resolved, check if the user requested a report/summary document. If yes, route to the report_worker. If no report is needed, route to the synthesizer.
4. Otherwise, select the next appropriate worker. For code analysis tasks, route to coding_worker for analysis then code_critic_worker for validation.
5. Write the updated plan, next_agent, and current_task in the JSON response.
"""

class SupervisorDecision(BaseModel):
    plan: List[str] = Field(description="Step-by-step plan to answer the query")
    next_agent: Literal["rag_worker", "web_worker", "utility_worker", "scraper_worker", "critic_worker", "report_worker", "coding_worker", "code_critic_worker", "synthesizer"] = Field(description="The next agent to route to")
    current_task: str = Field(description="Specific instruction for the next worker", default="")


def get_routing_model():
    """Get the LLM model for routing with structured output."""
    primary_key = os.getenv("GROQ_API_KEY")
    api_key = primary_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=SUPERVISOR_MODEL_PRIMARY, temperature=0, api_key=api_key).with_structured_output(SupervisorDecision)
    fallback = ChatGroq(model=SUPERVISOR_MODEL_FALLBACK, temperature=0, api_key=api_key).with_structured_output(SupervisorDecision)
    return primary.with_fallbacks([fallback])


def supervisor_node(state: dict) -> dict:
    """
    Supervisor node that constructs a plan, tracks findings, and dispatches sub-tasks.
    Supports both sequential and parallel worker dispatch.
    """
    model = get_routing_model()
    
    messages = state.get("messages", [])
    context_notes = state.get("context_notes", [])
    steps = state.get("steps_remaining", 10)
    plan = state.get("plan", [])
    scratchpad = state.get("scratchpad", "")
    
    # Bounded execution loop: decrement steps
    new_steps = steps - 1

    import re
    blocked_files = re.findall(r"execution of '[^']+' for '([^']+)' is blocked", scratchpad)
    blocked_files.extend(re.findall(r"Blocked awaiting approval for \w+ on ([^\s]+)", scratchpad))
    
    # Check user's latest message for approval
    user_approved = False
    latest_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_message = msg.get("content", "").lower()
            break
        elif hasattr(msg, "content") and (msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human"):
            latest_user_message = msg.content.lower()
            break

    # Common approval words/phrases
    approval_phrases = ["approve", "yes", "ok", "go ahead", "apply", "proceed", "yep", "sure"]
    if any(phrase in latest_user_message for phrase in approval_phrases):
        user_approved = True

    if blocked_files and user_approved:
        # User has approved! Append approval token to scratchpad
        new_approvals = []
        for filepath in blocked_files:
            token = f"[APPROVED: {filepath}]"
            if token not in scratchpad:
                new_approvals.append(token)
                
        if new_approvals:
            approval_str = " ".join(new_approvals)
            scratchpad += f"\n- [SYSTEM]: User has approved changes: {approval_str}"
            logger.info(f"Human-in-the-Loop: Approved file(s): {blocked_files}")
    
    # Formulate current blackboard context
    blackboard_context = f"""
--- COOPERATIVE BLACKBOARD ---
Current Plan: {plan}
Accumulated Findings (Scratchpad):
{scratchpad if scratchpad else "(No findings yet)"}
------------------------------
"""
    
    routing_prompt = []
    routing_prompt.append(SystemMessage(content=SUPERVISOR_PROMPT))
    routing_prompt.append(SystemMessage(content=blackboard_context))
    
    # Filter worker messages only from previous turns, keeping current turn's worker messages
    worker_names = {"rag_worker", "web_worker", "utility_worker", "scraper_worker", "critic_worker", "report_worker", "coding_worker", "code_critic_worker"}
    last_human_index = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                last_human_index = i
        elif hasattr(msg, "content") and (msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human"):
            last_human_index = i

    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            if i < last_human_index and msg.get("name") in worker_names:
                continue
            if msg.get("role") == "user":
                routing_prompt.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                routing_prompt.append(AIMessage(content=msg.get("content", "")))
        else:
            if i < last_human_index and hasattr(msg, "name") and msg.name in worker_names:
                continue
            routing_prompt.append(msg)
            
    if context_notes:
        routing_prompt.append(SystemMessage(content=f"Additional context notes: {' '.join(context_notes)}"))
        
    plan_out = plan
    next_agent = "synthesizer"
    current_task = ""
    parallel_tasks = []
    
    try:
        response: SupervisorDecision = model.invoke(routing_prompt)
        plan_out = response.plan
        next_agent = response.next_agent
        current_task = response.current_task
    except Exception as e:
        error_str = str(e)
        logger.error(f"Supervisor routing/planning error: {error_str}")
        # Surface auth errors clearly instead of silently falling through
        if "401" in error_str or "invalid_api_key" in error_str.lower() or "Invalid API Key" in error_str:
            scratchpad += "\n- [SYSTEM ERROR]: LLM API authentication failed. The API key may be expired or invalid. Please restart the server after updating your .env file."
        elif "429" in error_str or "rate_limit" in error_str.lower():
            scratchpad += "\n- [SYSTEM ERROR]: LLM API rate limit exceeded. Please wait and try again."
        else:
            scratchpad += f"\n- [SYSTEM ERROR]: Supervisor LLM call failed: {error_str[:200]}"
        
    valid_agents = ["rag_worker", "web_worker", "utility_worker", "scraper_worker", "critic_worker", "report_worker", "coding_worker", "code_critic_worker", "synthesizer", "FINISH"]
    if next_agent not in valid_agents or next_agent == "FINISH":
        next_agent = "synthesizer"
        
    state_update = {
        "plan": plan_out,
        "next_agent": next_agent,
        "current_task": current_task,
        "steps_remaining": new_steps,
        "scratchpad": scratchpad
    }
    
    print(f"\n[SUPERVISOR] Next Node: '{next_agent}' | Task: '{current_task}' | Steps Left: {new_steps}")
    return state_update
