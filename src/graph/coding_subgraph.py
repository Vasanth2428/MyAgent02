from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph

from src.agents.coding_worker import coding_worker_node

class CodingSubState(TypedDict):
    """Isolated state for the coding subgraph."""
    # Retain keys used by coding_worker_node
    messages: List[Any]
    scratchpad: str
    current_task: str
    # Additional optional fields can be added as needed
    # Example placeholders for future extensions
    code_history: List[Dict]
    temp_diffs: Dict[str, Any]

# Define the subgraph
coding_subgraph = StateGraph(CodingSubState)

# Add the coding worker node
coding_subgraph.add_node("coding_worker_node", coding_worker_node)

# End the subgraph after the coding worker completes
coding_subgraph.add_edge("coding_worker_node", "__end__")

# Compile the subgraph for embedding
compiled_coding_subgraph = coding_subgraph.compile()
