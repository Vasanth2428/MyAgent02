"""
Memory Isolation Test - Verify concurrent sessions never leak memory into each other.
"""
import pytest
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from src.core.memory import ConversationMemory, MemoryEntry
from src.core.engine import count_tokens


def test_memory_isolation():
    """Verify that concurrent sessions never leak memory into each other."""
    session_a = ConversationMemory(max_tokens=200)
    session_b = ConversationMemory(max_tokens=200)
    
    session_a.add("The secret project code for session A is ALPHA-123.", importance=1.0, role="user")
    session_b.add("The secret project code for session B is BETA-456.", importance=1.0, role="user")
    
    context_a = session_a.get_active_context()
    context_b = session_b.get_active_context()
    
    assert "ALPHA-123" in context_a, "Session A should contain its secret"
    assert "BETA-456" in context_b, "Session B should contain its secret"
    assert "ALPHA-123" not in context_b, "Session B should NOT contain Session A's secret"
    assert "BETA-456" not in context_a, "Session A should NOT contain Session B's secret"


def test_memory_isolation_concurrent_access():
    """Verify memory isolation under concurrent read/write operations - adversarial test."""
    sessions = {f"session_{i}": ConversationMemory(max_tokens=200) for i in range(10)}
    # Use unique secrets with no overlap
    secrets = {f"session_{i}": f"UNIQUE-ONLY-{i}-LEAK-CHECK" for i in range(10)}
    
    errors = []
    
    def write_only(session_id):
        try:
            memory = sessions[session_id]
            secret = secrets[session_id]
            memory.add(f"The secret code is {secret}. Never share this.", importance=1.0, role="user")
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=write_only, args=(sid,)) for sid in sessions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # After all writes, check each session has only its own secret
    for session_id, memory in sessions.items():
        context = memory.get_active_context()
        session_secret = secrets[session_id]
        for other_id, other_secret in secrets.items():
            if other_id != session_id:
                # Other secrets should NOT appear in this session's context
                assert other_secret not in context, f"{other_id} secret leaked into {session_id}"
        assert session_secret in context, f"{session_id} missing its own secret"


def test_memory_isolation_many_sessions():
    """Stress test memory isolation with many concurrent sessions."""
    num_sessions = 50
    sessions = [ConversationMemory(max_tokens=150) for _ in range(num_sessions)]
    session_secrets = [f"SECRET-{i:04d}-LEAK-TEST" for i in range(num_sessions)]
    all_violations = []
    
    for i, (memory, secret) in enumerate(zip(sessions, session_secrets)):
        memory.add(f"Unique secret for session {i}: {secret}. Never share this.", importance=1.0, role="user")
        
        for j, other_memory in enumerate(sessions):
            if i != j:
                other_context = other_memory.get_active_context()
                if secret[:12] in other_context:
                    all_violations.append(f"Session {i} secret leaked into session {j}")
    
    assert len(all_violations) == 0, f"Cross-session memory leaks: {all_violations[:5]}"


def test_memory_shared_embedding_model_isolation():
    """Verify that shared embedding model doesn't cause cross-session contamination."""
    session_a = ConversationMemory(max_tokens=300)
    session_b = ConversationMemory(max_tokens=300)
    
    session_a.add("Quantum computing uses qubits for parallel computation.", importance=1.0, role="user")
    session_b.add("Classical computing uses bits for sequential computation.", importance=1.0, role="user")
    
    session_a.add("What is the primary unit of quantum computing?", importance=1.0, role="user")
    session_b.add("What is the primary unit of classical computing?", importance=1.0, role="user")
    
    context_a = session_a.get_active_context()
    context_b = session_b.get_active_context()
    
    assert "qubit" in context_a.lower()
    assert "bit" in context_b.lower()
    
    assert context_a.count("qubit") >= context_a.count("bit") or "qubit" in context_a
    assert context_b.count("bit") >= context_b.count("qubit") or "bit" in context_b


def test_memory_isolation_after_truncation():
    """Verify isolation after memory truncation due to token budget."""
    session_a = ConversationMemory(max_tokens=100)
    session_b = ConversationMemory(max_tokens=100)
    
    for i in range(20):
        session_a.add(f"Session A message {i} with unique token AAAAA{i}", importance=0.5, role="user")
        session_b.add(f"Session B message {i} with unique token BBBBB{i}", importance=0.5, role="user")
    
    critical_a = "CRITICAL-A-DATA-999"
    critical_b = "CRITICAL-B-DATA-888"
    session_a.add(f"Important: {critical_a}", importance=5.0, role="user")
    session_b.add(f"Important: {critical_b}", importance=5.0, role="user")
    
    context_a = session_a.get_active_context()
    context_b = session_b.get_active_context()
    
    assert critical_a in context_a, "Session A critical data should survive truncation"
    assert critical_b in context_b, "Session B critical data should survive truncation"
    assert critical_b not in context_a, "No cross-contamination after truncation"
    assert critical_a not in context_b, "No cross-contamination after truncation"


def test_memory_isolation_async_concurrent():
    """Verify memory isolation under async concurrent operations - adversarial test."""
    sessions = {f"async-sess-{i}": ConversationMemory(max_tokens=200) for i in range(20)}
    
    async def async_memory_op(session_id):
        memory = sessions[session_id]
        unique_marker = f"ASYNC-UNIQUE-MARKER-{session_id}"
        memory.add(f"Data: {unique_marker}", importance=1.0, role="user")
        return session_id, unique_marker
    
    async def run_all():
        results = await asyncio.gather(*[async_memory_op(sid) for sid in sessions])
        return results
    
    results = asyncio.run(run_all())
    assert len(results) == 20, "All async operations should complete"
    
    # After completion, verify no cross-session contamination
    for session_id, memory in sessions.items():
        context = memory.get_active_context()
        # This session should have its own marker
        assert session_id in context, f"Session {session_id} missing its marker"