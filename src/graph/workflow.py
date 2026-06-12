# Build and compile the multi-agent workflow.
import os
import logging
from typing import List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send
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
    scratchpad_references = state.get("scratchpad_references", [])
    active_document_ids = list(state.get("active_document_ids") or [])
    task_hashes = list(state.get("task_hashes") or [])
    file_status_flags = dict(state.get("file_status_flags") or {})
    
    # We need to get the worker outputs from somewhere - 
    # Actually, looking at how this is called, it seems like the worker nodes
    # would have updated the state with their outputs before this node is called.
    # But in the current design, worker nodes return state updates that include
    # worker_outputs (full text), which we don't want to store.
    
    # Let me check how this node is used in the workflow...
    # Looking at the workflow, this node is called after each worker node (except coding_worker which has special handling)
    # And the worker nodes return state dicts that get merged.
    
    # For now, I'll assume that the worker nodes have been modified to store
    # references instead of full text, and this node just needs to compile
    # those references into a view for the scratchpad.
    
    # But wait, looking at the current worker nodes (like rag_worker.py above),
    # they still return "worker_outputs" with full text and "scratchpad" with full text.
    # So I need to update the worker nodes too, or change how this works.
    
    # Let me take a step back and think about this differently.
    # Maybe the aggregate_parallel_results_node is supposed to convert
    # the full text outputs from workers into reference-based storage.
    
    # Actually, looking more carefully at the workflow:
    # 1. Worker nodes run and return state updates (including worker_outputs with full text)
    # 2. Those updates get merged into the state
    # 3. Then aggregate_parallel_results_node is called
    # 4. It processes the worker_outputs to create summaries and store them in cache
    # 5. It returns updates to replace the full text with references
    
    # So the worker nodes can still return full text temporarily, and this node
    # converts them to references.
    
    # But we don't have access to the worker_outputs in the current state parameter
    # because they would have been overwritten by the merge process.
    
    # Let me look at how the workflow is structured again...
    
    # Actually, let me check if there are any worker outputs passed in via a different mechanism.
    # Looking at the original code, I don't see where worker_outputs would be available
    # in the aggregate_parallel_results_node function.
    
    # Wait, let me re-read the original aggregate_parallel_results_node:
    # It was accessing state.get("worker_outputs") - so it EXPECTS worker_outputs to be in the state.
    
    # This means that BEFORE this node is called, the worker nodes have already
    # updated the state with their worker_outputs (full text).
    # Then this node processes those outputs to create cache entries and references.
    # Finally, it returns a state update that REPLACES the full text outputs with references.
    
    # So I need to:
    # 1. Get worker_outputs from the current state (they should be there from worker node updates)
    # 2. Process them to store in cache and get cache_ids/summaries
    # 3. Update worker_output_ids and worker_output_summaries with the references
    # 4. Create scratchpad reference strings
    # 5. Return updates to set the new reference fields and clear the old text fields
    
    # But wait, the current state parameter to this function might not have worker_outputs
    # because of how StateGraph works - each node gets the current state, makes updates,
    # and those updates are merged.
    
    # Actually, let me check the original code again to see how it worked...
    
    # Looking at the original aggregate_parallel_results_node:
    # It accessed state.get("worker_outputs") and state.get("scratchpad")
    # So those fields MUST be present in the state when this node is called.
    
    # This means the worker nodes must have put them there in their return values.
    
    # So for my updated version:
    # 1. Extract worker_outputs from state (full text from worker nodes)
    # 2. Process them to create cache entries
    # 3. Build updates for the reference fields
    # 4. Also return updates to clear/remove the old text fields (worker_outputs, scratchpad)
    #    since we don't want to store them in the new state schema
    
    worker_outputs = state.get("worker_outputs", {})
    scratchpad = state.get("scratchpad", "")

    # Process each worker output to store in cache and get reference
    for worker_name, output in worker_outputs.items():
        if not output:
            continue
        cache_id, summary = store_worker_output(worker_name, output)
        worker_output_ids[worker_name] = cache_id
        worker_output_summaries[worker_name] = summary
        # Create a reference string for the scratchpad
        scratchpad_references.append(f"- [{worker_name.replace('_', ' ').title()}]: {summary}")

    # Return state updates:
    # - Set the new reference fields
    # - Clear the old text fields (by setting them to empty/default values)
    return {
        "worker_output_ids": worker_output_ids,
        "worker_output_summaries": worker_output_summaries,
        "scratchpad_references": scratchpad_references,
        # Clear the old fields that we don't want to store in state anymore
        "worker_outputs": {},
        "scratchpad": "",
        "active_document_ids": active_document_ids,
        "task_hashes": task_hashes,
        "file_status_flags": file_status_flags,
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

    # Import ToolNode and create a tool node using coding worker tools
    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(coding_tools)
    workflow.add_node("tool_node", tool_node)

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

    # Connect coding worker to the tool node
    workflow.add_edge("coding_worker_node", "tool_node")
    # After tool execution, decide next step based on approval state
    workflow.add_conditional_edges(
        "tool_node",
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
        import os
        safe_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(safe_dir, exist_ok=True)
        db_path = os.path.join(safe_dir, os.getenv("CHECKPOINTER_DB_PATH", "checkpoints.db"))
        checkpointer = SqliteSaver.from_conn_string(db_path)

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Multi-agent workflow compiled successfully.")
    return graph


def get_graph_config(thread_id: str = "default"):
    safe_thread_id = "".join(c for c in thread_id if c.isalnum() or c in "-_")[:100]
    return {"configurable": {"thread_id": safe_thread_id}}
