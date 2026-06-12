# Supervisor node for routing queries to specialized workers.
import os
import logging
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import SUPERVISOR_MODEL_PRIMARY, SUPERVISOR_MODEL_FALLBACK
from src.graph.worker_output_cache import (
    store_worker_output,
    get_worker_output,
    get_worker_output_summary,
)
from src.core.blackboard_reference_store import store_reference, get_reference, compact_scratchpad

logger = logging.getLogger("MultiAgent.Supervisor")

MAX_PLAN_STEPS = 8

APPROVAL_REGISTRY: Dict[str, set] = {}


def approve_file(session_id: str, filepath: str) -> None:
    APPROVAL_REGISTRY.setdefault(session_id, set()).add(os.path.realpath(filepath))


def is_file_approved(session_id: str, filepath: str) -> bool:
    return os.path.realpath(filepath) in APPROVAL_REGISTRY.get(session_id, set())


def clear_session_approvals(session_id: str) -> None:
    APPROVAL_REGISTRY.pop(session_id, None)


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
1. Construct or update a step-by-step plan (max 8 steps per turn) to answer the user query.
2. Evaluate the 'scratchpad', 'worker_summaries', and 'file_status_flags' to pick the safest next move.
3. Determine the next step. If all research steps are resolved, check if the user requested a report/summary document. If yes, route to the report_worker. If no report is needed, route to the synthesizer.
4. Otherwise, select the next appropriate worker. For code analysis tasks, route to coding_worker for analysis then code_critic_worker for validation.
5. Write the updated plan (max 8 steps), next_agent, and current_task in the JSON response.
"""


class SupervisorDecision(BaseModel):
    plan: List[str] = Field(description="Step-by-step plan to answer the query", max_length=8)
    next_agent: str = Field(description="The next agent to route to")
    current_task: str = Field(description="Specific instruction for the next worker", default="")


def get_routing_model():
    primary_key = os.getenv("GROQ_API_KEY")
    api_key = primary_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=SUPERVISOR_MODEL_PRIMARY, temperature=0, api_key=api_key).with_structured_output(SupervisorDecision)
    fallback = ChatGroq(model=SUPERVISOR_MODEL_FALLBACK, temperature=0, api_key=api_key).with_structured_output(SupervisorDecision)
    return primary.with_fallbacks([fallback])


def _get_session_id(state: dict) -> str:
    config = state.get("configurable") or {}
    return config.get("thread_id", "default")


def _display_scratchpad(scratchpad: str) -> str:
    if len(scratchpad) > 6000:
        return "[...earlier findings truncated...]\n" + scratchpad[-6000:]
    return scratchpad


def _summarize_worker_outputs(
    state: dict,
) -> Tuple[str, Dict[str, str]]:
    worker_output_ids = state.get("worker_output_ids") or {}
    worker_output_summaries = state.get("worker_output_summaries") or {}
    summaries = dict(worker_output_summaries)
    for worker_name, cache_id in worker_output_ids.items():
        summaries.setdefault(worker_name, get_worker_output_summary(cache_id))
    return "\n".join(f"- [{name}]: {summary}" for name, summary in summaries.items()) or "(No worker summaries yet)", summaries


def supervisor_node(state: dict) -> dict:
    model = get_routing_model()

    messages = state.get("messages", [])
    context_notes = state.get("context_notes") or []
    steps = state.get("steps_remaining", 10)
    plan = state.get("plan") or []
    scratchpad = state.get("scratchpad") or ""
    worker_output_ids = state.get("worker_output_ids") or {}
    worker_output_summaries = state.get("worker_output_summaries") or {}
    file_status_flags = state.get("file_status_flags") or {}
    task_hashes = state.get("task_hashes") or []
    active_document_ids = state.get("active_document_ids") or []
    retry_counter = int(state.get("retry_counter") or 0)
    import re
    blocked_files = re.findall(r"execution of '[^']+' for '([^']+)' is blocked", scratchpad)
    blocked_files.extend(re.findall(r"Blocked awaiting approval for \w+ on ([^\s]+)", scratchpad))
    
    user_approved = False
    user_rejected = False
    latest_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_message = msg.get("content", "").lower()
            break
        elif hasattr(msg, "content") and (msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human"):
            latest_user_message = msg.content.lower()
            break

    approval_phrases = ["approve", "yes", "ok", "go ahead", "apply", "proceed", "yep", "sure"]
    rejection_phrases = ["reject", "no", "deny", "stop", "cancel", "dont", "don't"]
    
    if any(phrase in latest_user_message for phrase in approval_phrases):
        user_approved = True
    elif any(phrase in latest_user_message for phrase in rejection_phrases):
        user_rejected = True

    state_update_overrides = {}

    if blocked_files:
        session_id = _get_session_id(state)
        from src.agents.coding_worker import execute_pending_approval, clear_pending_approval
        from src.tools.coding_tools import _get_absolute_path

        if user_approved:
            for filepath in blocked_files:
                try:
                    abs_path = os.path.realpath(_get_absolute_path(filepath.strip()))
                    approve_file(session_id, abs_path)
                except Exception as e:
                    logger.warning(f"Failed to register approval for {filepath}: {e}")
                    
            execution_result = execute_pending_approval(session_id)
            scratchpad += f"\n- [SYSTEM HITL]: User approved modifications. Action result: {execution_result}"
            logger.info(f"Human-in-the-Loop Approved and Executed: {execution_result}")
            
            state_update_overrides["waiting_for_approval"] = False
            state_update_overrides["pending_file_approvals"] = {}

        elif user_rejected:
            clear_pending_approval(session_id)
            scratchpad += f"\n- [SYSTEM HITL]: User rejected the proposed file modifications."
            logger.info("Human-in-the-Loop: User rejected changes.")
            
            state_update_overrides["waiting_for_approval"] = False
            state_update_overrides["pending_file_approvals"] = {}

    summaries_text, summaries = _summarize_worker_outputs(state)

    display_scratchpad = _display_scratchpad(scratchpad)

    blackboard_context = "\n".join(
        [
            "--- COOPERATIVE BLACKBOARD ---",
            f"Current Plan: {plan}",
            f"Active Document IDs: {', '.join(active_document_ids[:5]) or 'None'}",
            f"Task Hashes: {', '.join(task_hashes[:5]) or 'None'}",
            f"Retry Attempts: {retry_counter}",
            f"File Status Flags: {file_status_flags or 'None'}",
            "Accumulated Findings (Scratchpad):",
            display_scratchpad if display_scratchpad else "(No findings yet)",
            "Worker Summaries:",
            summaries_text,
            "-------------------------------",
        ]
    )

    routing_prompt = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        SystemMessage(content=blackboard_context),
    ]

    worker_names = {
        "rag_worker",
        "web_worker",
        "utility_worker",
        "scraper_worker",
        "critic_worker",
        "report_worker",
        "coding_worker",
        "code_critic_worker",
    }
    last_human_index = -1
    for i, msg in enumerate(messages):
        role = None
        content = None
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            if msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human":
                role = "user"
                content = msg.content
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "assistant")
                content = msg.content
        if role == "user":
            last_human_index = i

    for i, msg in enumerate(messages):
        role = None
        content = None
        name = None
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            name = msg.get("name")
        else:
            name = getattr(msg, "name", None)
            if msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human":
                role = "user"
                content = msg.content
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "assistant")
                content = msg.content

        if i < last_human_index and name in worker_names:
            continue
        if role == "user":
            routing_prompt.append(HumanMessage(content=content or "", name=name))
        elif role in ("assistant", "ai"):
            routing_prompt.append(AIMessage(content=content or "", name=name))
        elif content:
            routing_prompt.append(msg)

    if context_notes:
        routing_prompt.append(SystemMessage(content=f"Additional context notes: {' '.join(context_notes)}"))

    plan_out = plan if plan else []
    next_agent = "synthesizer"
    current_task = ""
    new_steps = max(0, steps - 1)
    new_retry_counter = retry_counter

    injected_traceback = ""
    if retry_counter >= 1:
        new_retry_counter = retry_counter + 1
        if retry_counter == 1:
            try:
                model = model.with_config(temperature=0.2)
            except Exception:
                pass
            injected_traceback = "\n[RETRY NOTE] Previous attempt failed. Prioritize the most recent worker summary and file_status_flags. Do not repeat prior path."
        else:
            injected_traceback = "\n[RETRY NOTE] Multiple retries occurred. Consider a different worker or strategy."
    if state.get("worker_outputs"):
        injected_traceback += "\n[RETRY NOTE] Existing worker outputs exist in memory; use cache-based summaries instead of re-running the same worker."

    try:
        if injected_traceback:
            routing_prompt.append(SystemMessage(content=injected_traceback))
        response = model.invoke(routing_prompt)
        plan_out = (response.plan or [])[:MAX_PLAN_STEPS]
        next_agent = response.next_agent
        current_task = response.current_task
    except Exception as e:
        error_str = str(e)
        logger.error(f"Supervisor routing/planning error: {error_str}")
        if "401" in error_str or "invalid_api_key" in error_str.lower():
            scratchpad += "\n- [SYSTEM ERROR]: LLM API authentication failed. The API key may be expired or invalid. Please restart the server after updating your .env file."
        elif "429" in error_str or "rate_limit" in error_str.lower():
            scratchpad += "\n- [SYSTEM ERROR]: LLM API rate limit exceeded. Please wait and try again."
        else:
            scratchpad += f"\n- [SYSTEM ERROR]: Supervisor LLM call failed: {error_str[:200]}"

    valid_agents = [
        "rag_worker",
        "web_worker",
        "utility_worker",
        "scraper_worker",
        "critic_worker",
        "report_worker",
        "coding_worker",
        "code_critic_worker",
        "synthesizer",
        "FINISH",
    ]
    if next_agent not in valid_agents or next_agent == "FINISH":
        next_agent = "synthesizer"

    # Compact scratchpad to prevent bloat (GRAPH-01)
    compact_scratchpad_text = compact_scratchpad(scratchpad)
    
    # Dynamic plan expansion support (GRAPH-03)
    # Check for "EXPAND PLAN" directives in scratchpad
    import re
    plan_expansion_match = re.search(r"EXPAND PLAN: (.+?)(?:\n|$)", scratchpad, re.IGNORECASE)
    if plan_expansion_match and plan:
        expansion_text = plan_expansion_match.group(1).strip()
        plan_out.extend([f"EXPANDED: {expansion_text}"])
    
    state_update = {
        "plan": plan_out,
        "next_agent": next_agent,
        "current_task": current_task,
        "steps_remaining": new_steps,
        "scratchpad": compact_scratchpad_text,
        "retry_counter": new_retry_counter,
        "worker_output_summaries": summaries,
    }
    
    state_update.update(state_update_overrides)

    print(f"\n[SUPERVISOR] Next Node: '{next_agent}' | Task: '{current_task}' | Steps Left: {new_steps} | Retries: {new_retry_counter}")
    return state_update
