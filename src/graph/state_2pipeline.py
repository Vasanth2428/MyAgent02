# State schema for 2-Pipeline Engine Architecture
from typing import List, Literal, Optional, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer that appends new messages to the existing list.
    Used as the Annotated operator for the messages field in AgentState.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


def merge_dicts(left: Dict[str, any], right: Dict[str, any]) -> Dict[str, any]:
    """Reducer that merges two dictionaries.
    Used for Dict fields in AgentState to handle concurrent/parallel updates.
    """
    if left is None:
        left = {}
    if right is None:
        right = {}
    merged = left.copy()
    merged.update(right)
    return merged


def merge_scratchpad_references(left: List[str], right: List[str]) -> List[str]:
    """Reducer that merges two lists of scratchpad references,
    handling both full replacements and partial/parallel updates.
    """
    if left is None:
        left = []
    if right is None:
        right = []

    # Case 1: right is a full replacement that starts with left as a prefix
    if len(right) >= len(left) and right[:len(left)] == left:
        return right

    # Case 2: right is just new items or parallel updates
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


# Lightweight state schema - stores references instead of raw text
class AgentState(TypedDict):
    session_id: str
    # Pipeline 1: Retrieval context cache references
    context_cache_id: Optional[str]
    active_document_ids: List[str]
    task_hashes: List[str]
    file_status_flags: Annotated[Dict[str, str], merge_dicts]
    # Pipeline 2: Execution routing
    next_agent: str
    current_task: str
    steps_remaining: int
    plan: List[str]
    worker_complete: Annotated[Dict[str, bool], merge_dicts]
    retry_counter: int
    critic_retry_count: int
    # HITL security - programmatic interrupts
    waiting_for_approval: bool
    approval_requested: Optional[str]
    approval_filepath: str
    pending_file_approvals: Annotated[Dict[str, Dict], merge_dicts]
    # Worker output references (to prevent state bloat)
    worker_output_ids: Annotated[Dict[str, str], merge_dicts]  # worker_name -> cache_id
    worker_output_summaries: Annotated[Dict[str, str], merge_dicts]  # worker_name -> summary
    scratchpad_references: Annotated[List[str], merge_scratchpad_references]  # List of formatted reference strings
    # Coding worker resume state
    coding_worker_messages: Optional[List[BaseMessage]]
    coding_worker_step: Optional[int]
    coding_worker_tool_calls_count: Optional[int]
    coding_worker_resume_tool_result: Optional[str]
    coding_worker_resume_tool_call_id: Optional[str]
    bypass_hitl: Optional[bool]
    # TEMPORARY FIELDS FOR WORKER NODE COMPATIBILITY (cleared after processing)
    scratchpad: str  # Temporary text scratchpad for worker nodes
    worker_outputs: Annotated[Dict[str, str], merge_dicts]  # Temporary full text outputs for worker nodes
    messages: Annotated[List[BaseMessage], add_messages]
    final_answer: str
