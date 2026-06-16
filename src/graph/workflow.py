# Build and compile the multi-agent workflow.
import os
import logging
from typing import List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
from langgraph.store.memory import InMemoryStore
from src.graph.worker_output_cache import store_worker_output, get_worker_output_summary

logger = logging.getLogger("MultiAgent.Workflow")

from src.graph.state_2pipeline import AgentState
from src.graph.supervisor import supervisor_node
from src.agents.rag_worker import rag_worker_node
from src.agents.web_worker import web_worker_node
from src.agents.utility_worker import utility_worker_node
from src.graph.synthesizer import synthesizer_node
from src.agents.scraper_worker import scraper_worker_node
from src.agents.critic_worker import critic_worker_node
from src.agents.report_worker import report_worker_node
from src.agents.coding_worker import coding_worker_node, tools as coding_tools
from src.agents.code_critic_worker import code_critic_worker_node

MAX_RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "20"))


def route_based_on_next_agent(state: dict) -> str:

    if state.get("waiting_for_approval"):
        logger.info("Workflow paused: pending human approval for file operation.")
        return END

    next_agent = state.get("next_agent", "FINISH")
    steps = state.get("steps_remaining", 0)
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
    if state.get("waiting_for_approval"):
        return "supervisor_node"
    return "aggregate_parallel_results_node"


def aggregate_parallel_results_node(state: dict) -> dict:
    # Get current state values
    worker_output_ids = state.get("worker_output_ids", {})
    worker_output_summaries = state.get("worker_output_summaries", {})
    scratchpad_references = list(state.get("scratchpad_references") or [])
    active_document_ids = list(state.get("active_document_ids") or [])
    task_hashes = list(state.get("task_hashes") or [])
    file_status_flags = dict(state.get("file_status_flags") or {})

    # Preserve raw scratchpad content before clearing
    scratchpad = state.get("scratchpad", "")
    if scratchpad.strip():
        scratchpad_references.append(f"--- Step Update ---\n{scratchpad.strip()}")

    worker_outputs = state.get("worker_outputs", {})
    # Process each worker output to store in cache and create summary references
    for worker_name, output in worker_outputs.items():
        if not output:
            continue
        cache_id, summary = store_worker_output(worker_name, output)
        worker_output_ids[worker_name] = cache_id
        worker_output_summaries[worker_name] = summary
        scratchpad_references.append(f"- [{worker_name.replace('_', ' ').title()}]: {summary}")

    pending_file_approvals = dict(state.get("pending_file_approvals") or {})
    approval_filepath = state.get("approval_filepath", "")
    approval_tool = state.get("approval_tool", "")
    waiting_for_approval = bool(state.get("waiting_for_approval", False))

    return {
        "worker_output_ids": worker_output_ids,
        "worker_output_summaries": worker_output_summaries,
        "scratchpad_references": scratchpad_references,
        "worker_outputs": {},
        "scratchpad": "",
        "active_document_ids": active_document_ids,
        "task_hashes": task_hashes,
        "file_status_flags": file_status_flags,
        "pending_file_approvals": pending_file_approvals,
        "approval_filepath": approval_filepath,
        "approval_tool": approval_tool,
        "waiting_for_approval": waiting_for_approval,
        "next_agent": "supervisor",
    }


def build_multi_agent_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

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
            END: END,
        },
    )

    workflow.add_edge("rag_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("web_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("utility_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("scraper_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("critic_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("report_worker_node", "aggregate_parallel_results_node")

    # After coding worker execution, decide next step based on approval state
    workflow.add_conditional_edges(
        "coding_worker_node",
        route_after_coding_worker,
        {
            "aggregate_parallel_results_node": "aggregate_parallel_results_node",
            "supervisor_node": "supervisor_node",
            END: END,
        },
    )

    workflow.add_edge("code_critic_worker_node", "aggregate_parallel_results_node")
    workflow.add_edge("aggregate_parallel_results_node", "supervisor_node")
    workflow.add_edge("synthesizer_node", END)

    if checkpointer is None:
        from src.graph.checkpointer import setup_checkpointer
        checkpointer = setup_checkpointer()

    graph = workflow.compile(checkpointer=checkpointer, store=InMemoryStore())
    logger.info("Multi-agent workflow compiled successfully.")
    return graph


def get_graph_config(thread_id: str = "default"):
    safe_thread_id = "".join(c for c in thread_id if c.isalnum() or c in "-_")[:100]
    return {"configurable": {"thread_id": safe_thread_id}}
