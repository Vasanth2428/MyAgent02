# State schema for multi-agent system.
from typing import List, Literal, Optional, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(existing: List, new: List) -> List:
        return existing + new


def merge_dict(existing: dict, new: dict) -> dict:
    if new is None:
        return existing or {}
    if not new:  # Check for empty dict {}
        return {}
    if not existing:
        return new or {}
    return {**existing, **new}


def merge_next_agent(existing: str, new: str) -> str:
    if new is None:
        return existing or ""
    return new


def merge_scratchpad(existing: str, new: str) -> str:
    if new is None:
        return existing or ""
    if new == "":
        return ""
    if not existing:
        return new or ""
    existing_lines = [line.strip() for line in existing.split("\n") if line.strip()]
    new_lines = [line.strip() for line in new.split("\n") if line.strip()]
    
    merged_lines = list(existing_lines)
    for line in new_lines:
        if line not in merged_lines:
            merged_lines.append(line)
            
    return "\n".join(merged_lines)



class ContextEngineState(TypedDict):
    """
    State for the multi-agent workflow.
    
    Attributes:
        messages: Conversation history using BaseMessage format.
        next_agent: Routing decision from supervisor.
        context_notes: Accumulated external notes (for context engine variant).
        steps_remaining: Budget for bounded loop.
        final_answer: Final response when supervisor ends.
        plan: Step-by-step plan constructed by supervisor.
        scratchpad: Accumulated findings from all workers (blackboard).
        current_task: Specific instruction for the next worker.
        worker_complete: Dict tracking completion status of each worker type.
        worker_outputs: Dict storing raw outputs from each worker.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: Annotated[str, merge_next_agent]
    context_notes: List[str]
    steps_remaining: int
    final_answer: str
    plan: List[str]
    scratchpad: Annotated[str, merge_scratchpad]
    current_task: str
    worker_complete: Annotated[Dict[str, bool], merge_dict]
    worker_outputs: Annotated[Dict[str, str], merge_dict]
    critic_retry_count: int
    pending_file_approvals: Dict[str, Dict]
    waiting_for_approval: bool
    approval_filepath: str
    approval_tool: str  
