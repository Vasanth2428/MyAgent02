import pytest
from src.core.memory import ConversationMemory
from src.core.engine import count_tokens


def test_conflicting_memory_resolution():
    """Verify that conflicting memories are ordered chronologically so LLM recency bias functions correctly."""
    memory = ConversationMemory()
    
    # User adds initial memory
    memory.add("The project name is Apollo.", importance=1.0, role="user")
    
    # User corrects it later
    memory.add("Actually, the project name is Orion.", importance=1.0, role="user")
    
    active_context = memory.get_active_context()
    
    # Both should be present, but the correction must appear AFTER the original
    assert "Apollo" in active_context
    assert "Orion" in active_context
    
    apollo_idx = active_context.find("Apollo")
    orion_idx = active_context.find("Orion")
    
    assert apollo_idx < orion_idx, "Correction should appear chronologically after the original fact"


def test_importance_abuse_survival():
    """Verify that recent critical facts survive even when memory is flooded with older high-importance noise."""
    memory = ConversationMemory(max_tokens=300)
    
    # Inject 15 older entries that take up space
    for i in range(15):
        memory.add(f"Irrelevant noise fact {i} that is extremely long and takes up space.", importance=10.0, role="user")
        
    # Inject a recent critical fact
    memory.add("The database port is 5432.", importance=1.0, role="user")
    
    active_context = memory.get_active_context()
    
    # Verify the recent critical fact survived and is present in active context
    assert "5432" in active_context, "Recent critical fact should survive older noise"


def test_long_horizon_memory_budget():
    """Verify that memory respects token budgets and remains stable over 100 conversation turns."""
    memory = ConversationMemory(max_tokens=300)
    
    # Simulate 100 conversation turns with distinct 3-word messages to bypass deduplication
    for i in range(100):
        memory.add(f"msg {i} usr", importance=1.0, role="user")
        memory.add(f"msg {i} ast", importance=1.0, role="assistant")
        
    active_context = memory.get_active_context()
    
    # Verify budget constraint is strictly respected
    tokens = count_tokens(active_context)
    assert tokens <= memory.max_tokens, f"Active context tokens ({tokens}) exceeded budget ({memory.max_tokens})"
    
    # Verify that the most recent turns are preserved
    assert "msg 99 ast" in active_context
    assert "msg 98 ast" in active_context
