# State schema for multi-agent system.
from typing import List, Literal, Optional, Annotated
from langchain_core.messages import BaseMessage
try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(existing: List, new: List) -> List:
        return existing + new


class ContextEngineState(dict):
    """
    State for the multi-agent workflow.
    
    Attributes:
        messages: Conversation history using BaseMessage format.
        next_agent: Routing decision from supervisor.
        context_notes: Accumulated external notes (for context engine variant).
        steps_remaining: Budget for bounded loop.
        final_answer: Final response when supervisor ends.
    """
    pass
