# Build and compile the multi-agent workflow.
import os
import logging
from typing import List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

logger = logging.getLogger("MultiAgent.Workflow")

from src.graph.state import ContextEngineState
from src.graph.supervisor import supervisor_node
from src.agents.rag_worker import rag_worker_node
from src.agents.web_worker import web_worker_node
from src.agents.utility_worker import utility_worker_node
from src.graph.synthesizer import synthesizer_node
from src.agents.scraper_worker import scraper_worker_node
from src.agents.critic_worker import critic_worker_node
from src.agents.report_worker import report_worker_node
from src.agents.coding_worker import coding_worker_node
from src.agents.code_critic_worker import code_critic_worker_node

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
    elif next_agent == "scraper_worker":
        return "scraper_worker_node"
    elif next_agent == "critic_worker":
        return "critic_worker_node"
    elif next_agent == "report_worker":
        return "report_worker_node"
    elif next_agent == "coding_worker":
        return "coding_worker_node"
    elif next_agent == "code_critic_worker":
        return "code_critic_worker_node"
    elif next_agent == "synthesizer":
        return "synthesizer_node"
    return END


def route_after_coding_worker(state: dict) -> str:
    """Route after coding worker - check if waiting for approval to pause workflow."""
    if state.get("waiting_for_approval"):
        return END
    return "aggregate_parallel_results_node"


def aggregate_parallel_results_node(state: dict) -> dict:
    """Merge worker outputs from state into scratchpad and return update."""
    scratchpad = state.get("scratchpad", "") or ""
    worker_outputs = state.get("worker_outputs", {}) or {}
    worker_complete = state.get("worker_complete", {}) or {}
    
    new_scratchpad = scratchpad
    for worker_name, output in worker_outputs.items():
        if not output:
            continue
        specialist_labels = {
            "rag_worker": "RAG Worker",
            "web_worker": "Web Worker",
            "utility_worker": "Utility Worker",
            "scraper_worker": "Scraper Worker",
            "critic_worker": "Critic Worker",
            "report_worker": "Report Worker",
            "coding_worker": "Coding Worker",
            "code_critic_worker": "Code Critic Worker"
        }
        label = specialist_labels.get(worker_name, worker_name.replace("_", " ").title())
        pattern = f"- [{label}]:"
        
        if pattern not in new_scratchpad:
            new_scratchpad += f"\n- [{label}]: {output}"
            
    return {
        "scratchpad": new_scratchpad,
        "next_agent": "supervisor"
    }


def build_multi_agent_graph(checkpointer=None):
    """Build the StateGraph for multi-agent orchestration."""
    workflow = StateGraph(ContextEngineState)
    
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("rag_worker_node", rag_worker_node)
    workflow.add_node("web_worker_node", web_worker_node)
    workflow.add_node("utility_worker_node", utility_worker_node)
    workflow.add_node("scraper_worker_node", scraper_worker_node)
    workflow.add_node("critic_worker_node", critic_worker_node)
    workflow.add_node("report_worker_node", report_worker_node)
    workflow.add_node("coding_worker_node", coding_worker_node)
    workflow.add_node("code_critic_worker_node", code_critic_worker_node)
    workflow.add_node("synthesizer_node", synthesizer_node)
    workflow.add_node("aggregate_parallel_results_node", aggregate_parallel_results_node)
    
    workflow.set_entry_point("supervisor_node")
    
    workflow.add_conditional_edges(
        "supervisor_node",
        route_based_on_next_agent,
        {
            "rag_worker_node": "rag_worker_node",
            "web_worker_node": "web_worker_node",
            "utility_worker_node": "utility_worker_node",
            "scraper_worker_node": "scraper_worker_node",
            "critic_worker_node": "critic_worker_node",
            "report_worker_node": "report_worker_node",
            "coding_worker_node": "coding_worker_node",
            "code_critic_worker_node": "code_critic_worker_node",
            "synthesizer_node": "synthesizer_node",
            END: END
        }
    )
    
    # Workers return to aggregator to merge parallel or sequential results
    workflow.add_edge("rag_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("web_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("utility_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("scraper_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("critic_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("report_worker_node", "aggregate_parallel_results_node")
    
    # Conditional edge after coding worker to support human-in-the-loop approval
    workflow.add_conditional_edges(
        "coding_worker_node",
        route_after_coding_worker,
        {
            "aggregate_parallel_results_node": "aggregate_parallel_results_node",
            END: END
        }
    )
    
    workflow.add_edge("code_critic_worker_node", "aggregate_parallel_results_node")
    
    # Aggregator collects results back to supervisor
    workflow.add_edge("aggregate_parallel_results_node", "supervisor_node")
    
    # Synthesizer ends the workflow
    workflow.add_edge("synthesizer_node", END)
    
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
