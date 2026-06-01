# Build and compile the multi-agent workflow.
import os
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("MultiAgent.Workflow")

from src.graph.state import ContextEngineState
from src.graph.supervisor import supervisor_node
from src.agents.rag_worker import rag_worker_node
from src.agents.web_worker import web_worker_node
from src.agents.utility_worker import utility_worker_node

MAX_RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "20"))


def route_based_on_next_agent(state: dict) -> str:
    """Conditional routing based on supervisor decision."""
    next_agent = state.get("next_agent", "FINISH")
    steps = state.get("steps_remaining", 0)
    
    # Safety guard: stop if recursion limit exceeded
    if steps <= 0:
        logger.warning("Max recursion steps reached, ending workflow.")
        return END
    
    if next_agent == "FINISH":
        return END
    elif next_agent == "rag_worker":
        return "rag_worker_node"
    elif next_agent == "web_worker":
        return "web_worker_node"
    elif next_agent == "utility_worker":
        return "utility_worker_node"
    
    return END


def decrement_steps(state: dict, worker_name: str) -> dict:
    """Decrement steps_remaining for loop control."""
    return {"steps_remaining": state.get("steps_remaining", 10) - 1}


def build_multi_agent_graph(checkpointer=None):
    """Build the StateGraph for multi-agent orchestration."""
    workflow = StateGraph(ContextEngineState)
    
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("rag_worker_node", rag_worker_node)
    workflow.add_node("web_worker_node", web_worker_node)
    workflow.add_node("utility_worker_node", utility_worker_node)
    
    workflow.set_entry_point("supervisor_node")
    
    workflow.add_conditional_edges(
        "supervisor_node",
        route_based_on_next_agent,
        {
            "rag_worker_node": "rag_worker_node",
            "web_worker_node": "web_worker_node",
            "utility_worker_node": "utility_worker_node",
            END: END
        }
    )
    
    workflow.add_edge("rag_worker_node", "supervisor_node")
    workflow.add_edge("web_worker_node", "supervisor_node")
    workflow.add_edge("utility_worker_node", "supervisor_node")
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Multi-agent workflow compiled successfully.")
    return graph


def get_graph_config(thread_id: str = "default"):
    """Get config for graph invocation with checkpointing."""
    # Sanitize thread_id
    safe_thread_id = "".join(c for c in thread_id if c.isalnum() or c in "-_")[:100]
    return {"configurable": {"thread_id": safe_thread_id}}
