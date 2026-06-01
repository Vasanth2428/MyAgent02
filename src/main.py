# Main entry point for multi-agent system.
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgent.Main")

from src.graph.workflow import build_multi_agent_graph, get_graph_config
from src.graph.checkpointer import setup_checkpointer
from langchain_core.messages import HumanMessage
from src.tools.safety_filters import sanitize_user_input

multi_agent_graph = None


def initialize_graph():
    """Initialize the multi-agent graph."""
    global multi_agent_graph
    
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    
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
        "next_agent": "FINISH",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": ""
    }
    
    result = multi_agent_graph.invoke(initial_state, config=config)
    return result


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2+2?"
    result = run_query(query)
    answer = result.get('final_answer', '') or (result.get('messages', [{}])[-1].content if result.get('messages') else 'No answer')
    print(f"Answer: {answer}")
