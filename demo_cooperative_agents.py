"""
Demo script showing the cooperative multi-agent system in action.

This demonstrates:
1. Sequential worker coordination (supervisor -> worker -> supervisor -> synthesizer)
2. Parallel worker dispatch (supervisor dispatches multiple workers concurrently)
3. Shared scratchpad (blackboard) pattern for results accumulation
"""

import sys
sys.path.insert(0, '.')

from src.graph.workflow import build_multi_agent_graph, get_graph_config
from langchain_core.messages import HumanMessage


def demo_sequential_flow():
    """Demonstrate sequential worker coordination."""
    print("\n" + "="*60)
    print("DEMO: Sequential Multi-Agent Flow")
    print("="*60)
    
    graph = build_multi_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="What is 2+2 and what is the weather?")],
        "next_agent": "FINISH",
        "context_notes": [],
        "steps_remaining": 10,
        "final_answer": "",
        "plan": [],
        "scratchpad": "",
        "current_task": "",
        "worker_complete": {},
        "worker_outputs": {},
        "parallel_tasks": []
    }
    
    print("\nQuery: What is 2+2 and what is the weather?")
    print("Expected: Supervisor will route to utility_worker and web_worker sequentially")
    
    # Mock mode - actually running would hit LLM APIs
    print("\n[Note: Actual execution would call LLM APIs]")


def demo_parallel_flow_structure():
    """Show the structure for parallel dispatch."""
    print("\n" + "="*60)
    print("DEMO: Parallel Multi-Agent Flow (Structure)")
    print("="*60)
    
    from langgraph.types import Send
    
    # Example of parallel_tasks that supervisor could create
    parallel_tasks = [
        {"worker": "utility_worker", "task": "Calculate 15% tax on $100"},
        {"worker": "web_worker", "task": "Search for stock price of AAPL today"}
    ]
    
    print("\nWhen supervisor detects INDEPENDENT tasks, it can set:")
    print(f"  next_agent: 'parallel'")
    print(f"  parallel_tasks: {parallel_tasks}")
    print("\nThis triggers Send() fan-out to both workers simultaneously.")
    print("Results merge back to scratchpad, then supervisor evaluates completion.")


if __name__ == "__main__":
    demo_sequential_flow()
    demo_parallel_flow_structure()
    
    print("\n" + "="*60)
    print("KEY ARCHITECTURE CHANGES:")
    print("="*60)
    print("""
1. State now includes:
   - worker_complete: Dict[str, bool] - tracks which workers finished
   - worker_outputs: Dict[str, str] - stores worker results
   - parallel_tasks: List[Dict] - tasks for concurrent execution

2. Workers return to supervisor (not FINISH) with:
   - worker_complete flag set
   - worker_outputs updated
   - worker_type identified

3. Supervisor prompt now instructs to:
   - Set next_agent='parallel' for independent concurrent tasks
   - List parallel_tasks with worker + task for each

4. parallel_dispatch_node uses Send API for true parallel execution
""")