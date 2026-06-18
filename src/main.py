# Main entry point for multi-agent system.
import os
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path="../config/.env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgent.Main")

from src.graph.workflow import build_multi_agent_graph, get_graph_config
from src.graph.checkpointer import setup_checkpointer
from langchain_core.messages import HumanMessage
from src.tools.safety_filters import sanitize_user_input
from src.core.logging_setup import session_id_var

multi_agent_graph = None


def initialize_graph():
    """Initialize the multi-agent graph."""
    global multi_agent_graph
    
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    
    from src.graph.checkpointer import setup_checkpointer
    checkpointer = setup_checkpointer()
    multi_agent_graph = build_multi_agent_graph(checkpointer)
    logger.info("Multi-agent graph initialized.")
    return multi_agent_graph


def run_query(query: str, thread_id: str = "default"):
    """Run a single query through the multi-agent system."""
    global multi_agent_graph
    
    # Sanitize inputs
    query = sanitize_user_input(query)
    
    if multi_agent_graph is None:
        initialize_graph()
    
    config = get_graph_config(thread_id)
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "supervisor",
        "steps_remaining": 10,
        "plan": [],
        "current_task": "",
        "worker_complete": {},
        "retry_counter": 0,
        "critic_retry_count": 0,
        "waiting_for_approval": False,
        "approval_requested": None,
        "approval_filepath": "",
        "pending_file_approvals": {},
        "patch_is_verified": False,
        "active_project": "",
        "session_id": "default_session",
        # Pipeline 1: Retrieval context cache references
        "context_cache_id": None,
        "active_document_ids": [],
        "task_hashes": [],
        "file_status_flags": {},
        # Worker output references (to prevent state bloat)
        "worker_output_ids": {},
        "worker_output_summaries": {},
        "scratchpad_references": [],
        # TEMPORARY FIELDS FOR WORKER NODE COMPATIBILITY (cleared after processing)
        "scratchpad": "",
        "worker_outputs": {},
        # Pipeline 2: Execution routing (already covered above)
        "final_answer": ""
    }
    
    token = session_id_var.set(thread_id)
    try:
        result = multi_agent_graph.invoke(initial_state, config=config)
        return result
    finally:
        session_id_var.reset(token)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2?"
    result = run_query(query)
    answer = result.get('final_answer', '') or (result.get('messages', [{}])[-1].content if result.get('messages') else 'No answer')
    print(f"Answer: {answer}")
