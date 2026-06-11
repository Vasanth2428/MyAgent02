import os
import sqlite3
import httpx
import time

BASE_URL = "http://127.0.0.1:8000"

def wait_for_server():
    print("Waiting for server to become ready...")
    for _ in range(30):
        try:
            r = httpx.get(f"{BASE_URL}/stats")
            if r.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Server did not become ready.")
    return False

def verify_all():
    if not wait_for_server():
        return

    # 1. Create a session
    client = httpx.Client(timeout=120.0)
    res = client.post(f"{BASE_URL}/sessions", json={"title": "Test History Session"})
    session_data = res.json()
    session_id = session_data["session_id"]
    print(f"\n[1] Created session: {session_id}")

    # 2. First query in normal (Simple RAG) mode
    print("\n[2] Sending first query in Simple RAG (normal) mode...")
    q1 = "Who is the author of Atomic Habits?"
    res = client.post(f"{BASE_URL}/query", json={
        "question": q1,
        "session_id": session_id,
        "mode": "normal"
    })
    ans1 = res.json()["response"]
    print(f"Response: {ans1}")

    # Verify query 1 is in memory.db
    conn_mem = sqlite3.connect("memory.db")
    c_mem = conn_mem.cursor()
    c_mem.execute("SELECT role, text FROM memory WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    history = c_mem.fetchall()
    print("History in memory.db after query 1:")
    for role, text in history:
        print(f"  {role}: {text[:80]}...")
    assert len(history) == 2, f"Expected 2 messages, got {len(history)}"

    # 3. Follow-up query in normal (Simple RAG) mode asking about the previous turn
    print("\n[3] Sending follow-up query in Simple RAG (normal) mode...")
    q2 = "What did I just ask you?"
    res = client.post(f"{BASE_URL}/query", json={
        "question": q2,
        "session_id": session_id,
        "mode": "normal"
    })
    ans2 = res.json()["response"]
    print(f"Response: {ans2}")
    
    # Check if the memory context was generated
    raw_prompt = res.json().get("raw_prompt", "")
    print(f"Contains ### MEMORY in prompt: {'### MEMORY' in raw_prompt}")
    assert "### MEMORY" in raw_prompt, "Expected ### MEMORY header to be injected in Simple RAG prompt"

    # 4. Switch to Agentic mode and ask a follow-up referencing the history
    print("\n[4] Sending query in Agentic mode referencing history...")
    q3 = "What was my very first question in this chat?"
    res = client.post(f"{BASE_URL}/query", json={
        "question": q3,
        "session_id": session_id,
        "mode": "agentic"
    })
    ans3 = res.json()["response"]
    print(f"Agentic Response: {ans3}")

    # Check checkpoints database to ensure thread exists
    conn_check = sqlite3.connect("checkpoints/checkpoints.db")
    c_check = conn_check.cursor()
    c_check.execute("SELECT thread_id, checkpoint_id FROM checkpoints WHERE thread_id = ?", (session_id,))
    checkpoints = c_check.fetchall()
    print(f"Found {len(checkpoints)} checkpoints for thread {session_id} in checkpoints.db")
    assert len(checkpoints) > 0, "Expected checkpoints to exist in checkpoints.db during graph execution"

    # 5. Delete session and verify databases are cleaned
    print("\n[5] Deleting session...")
    res = client.delete(f"{BASE_URL}/sessions/{session_id}")
    assert res.status_code == 204, f"Delete failed with {res.status_code}"

    # Verify memory.db is cleaned
    c_mem.execute("SELECT COUNT(*) FROM memory WHERE session_id = ?", (session_id,))
    count_mem = c_mem.fetchone()[0]
    print(f"Entries remaining in memory.db: {count_mem}")
    assert count_mem == 0, f"Expected 0 memory entries, got {count_mem}"

    # Verify checkpoints.db is cleaned
    c_check.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (session_id,))
    count_check = c_check.fetchone()[0]
    print(f"Checkpoints remaining in checkpoints.db: {count_check}")
    assert count_check == 0, f"Expected 0 checkpoints, got {count_check}"

    c_check.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", (session_id,))
    count_writes = c_check.fetchone()[0]
    print(f"Writes remaining in checkpoints.db: {count_writes}")
    assert count_writes == 0, f"Expected 0 writes, got {count_writes}"

    print("\n==============================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==============================================")

if __name__ == "__main__":
    verify_all()
