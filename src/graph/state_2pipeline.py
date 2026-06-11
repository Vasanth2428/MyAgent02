# State schema for 2-Pipeline Engine Architecture
from typing import List, Literal, Optional, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from src.graph.state import add_messages

# Lightweight state schema - stores references instead of raw text
class AgentState(TypedDict):
    session_id: str
    # Pipeline 1: Retrieval context cache references
    context_cache_id: Optional[str]
    active_document_ids: List[str]
    task_hashes: List[str]
    file_status_flags: Dict[str, str]
    # Pipeline 2: Execution routing
    next_agent: str
    current_task: str
    steps_remaining: int
    plan: List[str]
    worker_complete: Dict[str, bool]
    retry_counter: int
    critic_retry_count: int
    # HITL security - programmatic interrupts
    waiting_for_approval: bool
    approval_requested: Optional[str]
    approval_filepath: str
    pending_file_approvals: Dict[str, Dict]
    # Worker output references (to prevent state bloat)
    worker_output_ids: Dict[str, str]  # worker_name -> cache_id
    worker_output_summaries: Dict[str, str]  # worker_name -> summary
    scratchpad_references: List[str]  # List of formatted reference strings
    # TEMPORARY FIELDS FOR WORKER NODE COMPATIBILITY (cleared after processing)
    scratchpad: str  # Temporary text scratchpad for worker nodes
    worker_outputs: Dict[str, str]  # Temporary full text outputs for worker nodes
    messages: Annotated[List[BaseMessage], add_messages]
    final_answer: str
