# Supervisor node for routing queries to specialized workers.
import os
import logging
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.core.config import SUPERVISOR_MODEL_PRIMARY, SUPERVISOR_MODEL_FALLBACK
from src.core.model_provider import build_model_with_fallback
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
2. Evaluate the 'scratchpad', 'worker_summaries', 'file_status_flags', and especially 'Completed Tasks' to pick the safest next move.
3. Determine the next step. If all research steps are resolved, check if the user explicitly requested a report/summary document. If yes, route to the report_worker ONCE. If no report is needed, route to the synthesizer.
4. Otherwise, select the next appropriate worker. For code analysis tasks, route to coding_worker for analysis then code_critic_worker for validation.
5. Write the updated plan (max 8 steps), next_agent, and current_task in the JSON response.

CRITICAL DEDUPLICATION RULES (you MUST follow these before every routing decision):
- Check the 'Completed Tasks' list in the blackboard carefully. If a task fingerprint matching your intended next_agent + task already appears there, DO NOT route to that worker again.
- If the scratchpad or worker summaries already contain results from a worker for this session, do not re-run that same worker for the same sub-task.
- If the coding_worker has already scaffolded a project (keyword 'scaffold' appears in completed tasks), route to coding_worker for CONTENT only — not scaffolding again.
- If all planned steps are complete or covered by completed tasks, route DIRECTLY to 'synthesizer'. Do not route to 'report_worker' unless the user explicitly asked for a report.
- Prefer 'synthesizer' over any looping back to workers when sufficient information is available.

CODING TASK SPECIFICATION RULES:
- When routing to coding_worker for implementation or editing:
  1. Break down broad requests (e.g. "Create a fullstack crypto portfolio website") into specific, component-level tasks (e.g., "Scaffold Vite app", "Implement SQLite schema and FastAPI backend", "Develop premium dashboard interface in App.jsx"). Never dispatch a single task covering both frontend and backend.
  2. For frontend tasks, explicitly instruct coding_worker to use premium design aesthetics (HSL-tailored colors, dark mode, glassmorphism, Outfit/Inter typography, linear gradients, transitions, responsive layouts, hover animations).
  3. For backend tasks, instruct the worker to use local SQLite databases, FastAPI routes, and write validation checks.
  4. Ensure task instructions are concrete, specifying file paths and expected behaviors. Do not use vague or generic summaries.
"""



class SupervisorDecision(BaseModel):
    plan: List[str] = Field(description="Step-by-step plan to answer the query", max_length=8)
    next_agent: str = Field(description="The next agent to route to")
    current_task: str = Field(description="Specific instruction for the next worker", default="")


def get_routing_model():
    return build_model_with_fallback(
        "supervisor",
        SUPERVISOR_MODEL_PRIMARY,
        SUPERVISOR_MODEL_FALLBACK,
        temperature=0,
        api_key_envs=("GROQ_API_KEY", "AGENT_API_KEY"),
        structured_output=SupervisorDecision,
    )


def _get_session_id(state: dict) -> str:
    config = state.get("configurable") or {}
    return config.get("thread_id", "default")


def _approval_decision_from_message(message: str) -> str:
    text = (message or "").strip().lower()
    if not text:
        return ""

    approval_phrases = ["approve", "yes", "ok", "go ahead", "apply", "proceed", "yep", "sure"]
    rejection_phrases = ["reject", "no", "deny", "stop", "cancel", "dont", "don't"]

    if any(phrase in text for phrase in approval_phrases):
        return "approved"
    if any(phrase in text for phrase in rejection_phrases):
        return "rejected"
    return ""


def _approval_decision_from_state(state: dict) -> str:
    decision = str(state.get("approval_decision") or state.get("approval_requested") or "").strip().lower()
    if decision in {"approved", "approve", "yes", "true", "1"}:
        return "approved"
    if decision in {"rejected", "reject", "no", "false", "0"}:
        return "rejected"

    for approval in (state.get("pending_file_approvals") or {}).values():
        if not isinstance(approval, dict):
            continue
        item_decision = str(approval.get("decision") or approval.get("approved") or "").strip().lower()
        if item_decision in {"approved", "approve", "yes", "true", "1"}:
            return "approved"
        if item_decision in {"rejected", "reject", "no", "false", "0"}:
            return "rejected"

    return ""


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

    messages = state.get("messages", [])
    context_notes = state.get("context_notes") or []
    steps = state.get("steps_remaining", 10)
    plan = state.get("plan") or []
    # Retain the compacted scratchpad for error messages, but primary context is in scratchpad_references
    scratchpad = state.get("scratchpad") or ""
    scratchpad_references = state.get("scratchpad_references") or []
    worker_output_ids = state.get("worker_output_ids") or {}
    worker_output_summaries = state.get("worker_output_summaries") or {}
    file_status_flags = state.get("file_status_flags") or {}
    task_hashes = state.get("task_hashes") or []
    active_document_ids = state.get("active_document_ids") or []
    retry_counter = int(state.get("retry_counter") or 0)
    completed_tasks = list(state.get("completed_tasks") or [])
    
    # Structured HITL handling - avoid parsing scratchpad text
    is_waiting = state.get("waiting_for_approval", False)
    pending_approvals = state.get("pending_file_approvals") or {}
    blocked_files = []
    if is_waiting:
        if state.get("approval_filepath"):
            blocked_files.append(state.get("approval_filepath"))
        elif pending_approvals:
            blocked_files.extend(pending_approvals.keys())
    
    latest_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_message = msg.get("content", "")
            break
        elif hasattr(msg, "content") and (msg.__class__.__name__ == "HumanMessage" or getattr(msg, "type", "") == "human"):
            latest_user_message = msg.content
            break

    approval_decision = _approval_decision_from_state(state) or _approval_decision_from_message(latest_user_message)

    state_update_overrides = {}

    if blocked_files:
        session_id = _get_session_id(state)
        from src.agents.coding_worker import execute_pending_approval, clear_pending_approval, get_pending_approval
        from src.tools.coding_tools import _get_absolute_path

        pending = get_pending_approval(session_id)
        pending_tool_call_id = pending.get("tool_call_id") if pending else None

        if approval_decision == "approved":
            for filepath in blocked_files:
                try:
                    abs_path = os.path.realpath(_get_absolute_path(filepath.strip()))
                    approve_file(session_id, abs_path)
                except Exception as e:
                    logger.warning(f"Failed to register approval for {filepath}: {e}")
                    
            execution_result = execute_pending_approval(session_id)
            state_update_overrides["scratchpad_references"] = scratchpad_references + [
                f"- [SYSTEM HITL]: User approved modifications. Action result: {execution_result}"]
            logger.info(f"Human-in-the-Loop Approved and Executed: {execution_result}")
            
            state_update_overrides["waiting_for_approval"] = False
            state_update_overrides["pending_file_approvals"] = {}
            state_update_overrides["approval_decision"] = "approved"
            state_update_overrides["coding_worker_resume_tool_result"] = execution_result
            state_update_overrides["coding_worker_resume_tool_call_id"] = pending_tool_call_id

        elif approval_decision == "rejected":
            clear_pending_approval(session_id)
            state_update_overrides["scratchpad_references"] = scratchpad_references + [
                "- [SYSTEM HITL]: User rejected the proposed file modifications."]
            logger.info("Human-in-the-Loop: User rejected changes.")
            
            state_update_overrides["waiting_for_approval"] = False
            state_update_overrides["pending_file_approvals"] = {}
            state_update_overrides["approval_decision"] = "rejected"
            state_update_overrides["coding_worker_resume_tool_result"] = "Error: User rejected the proposed file modifications."
            state_update_overrides["coding_worker_resume_tool_call_id"] = pending_tool_call_id

        else:
            hitl_note = (
                f"- [SYSTEM HITL]: Awaiting user approval for "
                f"{state.get('approval_tool') or 'file operation'} on "
                f"{state.get('approval_filepath') or ', '.join(blocked_files)}."
            )
            if hitl_note not in scratchpad_references:
                scratchpad_references = scratchpad_references + [hitl_note]
                state_update_overrides["scratchpad_references"] = scratchpad_references
            logger.info("Human-in-the-Loop: Awaiting explicit approval; pausing workflow.")

        pause_update = {
            "plan": plan,
            "next_agent": "supervisor",
            "current_task": "",
            "steps_remaining": steps,
            "scratchpad": compact_scratchpad(scratchpad),
            "retry_counter": retry_counter,
            "worker_output_summaries": worker_output_summaries,
            "worker_output_ids": worker_output_ids,
            "scratchpad_references": scratchpad_references,
            "waiting_for_approval": True,
            "pending_file_approvals": pending_approvals,
            "approval_filepath": state.get("approval_filepath", ""),
            "approval_tool": state.get("approval_tool", ""),
        }
        pause_update.update(state_update_overrides)
        return pause_update

    summaries_text, summaries = _summarize_worker_outputs(state)

    # Reconstruct a readable scratchpad from references for display
    reconstructed_scratchpad = "\n".join(scratchpad_references)
    display_scratchpad = _display_scratchpad(reconstructed_scratchpad)

    blackboard_context = "\n".join(
        [
            "--- COOPERATIVE BLACKBOARD ---",
            f"Current Plan: {plan}",
            f"Active Document IDs: {', '.join(active_document_ids[:5]) or 'None'}",
            f"Task Hashes: {', '.join(task_hashes[:5]) or 'None'}",
            f"Retry Attempts: {retry_counter}",
            f"File Status Flags: {file_status_flags or 'None'}",
            f"Completed Tasks (DO NOT re-assign these): {completed_tasks if completed_tasks else 'None'}",
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
                model = get_routing_model().with_config(temperature=0.2)
            except Exception:
                pass
            injected_traceback = "\n[RETRY NOTE] Previous attempt failed. Prioritize the most recent worker summary and file_status_flags. Do not repeat prior path."
        else:
            injected_traceback = "\n[RETRY NOTE] Multiple retries occurred. Consider a different worker or strategy."
    if state.get("worker_output_ids"):
        injected_traceback += "\n[RETRY NOTE] Existing worker outputs exist in memory; use cached summaries instead of re-running the same worker."

    try:
        model = get_routing_model()
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
            err_msg = "\n- [SYSTEM ERROR]: LLM API authentication failed. The API key may be expired or invalid. Please restart the server after updating your .env file."
        elif "429" in error_str or "rate_limit" in error_str.lower():
            err_msg = "\n- [SYSTEM ERROR]: LLM API rate limit exceeded. Please wait and try again."
        else:
            err_msg = f"\n- [SYSTEM ERROR]: Supervisor LLM call failed: {error_str[:200]}"
        scratchpad += err_msg
        # Also record in persistent references for future context
        state_update_overrides["scratchpad_references"] = (state_update_overrides.get("scratchpad_references", []) + [err_msg.strip()])

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
    
    # Build task fingerprint: worker:task_summary to detect duplicates next turn
    import hashlib
    task_fingerprint = None
    if next_agent not in ("synthesizer", "FINISH") and current_task:
        raw_fp = f"{next_agent}:{current_task[:120]}"
        task_fingerprint = f"{next_agent}:{hashlib.md5(raw_fp.encode()).hexdigest()[:8]}"
        # Duplicate guard: if we are about to route to the same worker+task, override to synthesizer
        if task_fingerprint in completed_tasks:
            logger.warning(f"[SUPERVISOR] Duplicate task fingerprint detected: {task_fingerprint}. Overriding next_agent to 'synthesizer'.")
            print(f"[SUPERVISOR] Duplicate task blocked ({task_fingerprint}). Forcing synthesizer.")
            next_agent = "synthesizer"
            task_fingerprint = None  # Don't record synthesizer as a completed task fingerprint

    # Track dispatched task in completed_tasks list
    new_completed_tasks = list(completed_tasks)
    if task_fingerprint and task_fingerprint not in new_completed_tasks:
        new_completed_tasks.append(task_fingerprint)

    state_update = {
        "plan": plan_out,
        "next_agent": next_agent,
        "current_task": current_task,
        "steps_remaining": new_steps,
        "scratchpad": compact_scratchpad_text,
        "retry_counter": new_retry_counter,
        "worker_output_summaries": summaries,
        "worker_output_ids": worker_output_ids,
        "scratchpad_references": scratchpad_references,
        "completed_tasks": new_completed_tasks,
    }
    
    state_update.update(state_update_overrides)

    print(f"\n[SUPERVISOR] Next Node: '{next_agent}' | Task: '{current_task}' | Steps Left: {new_steps} | Retries: {new_retry_counter}")
    return state_update
