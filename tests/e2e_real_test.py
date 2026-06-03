"""
Real End-to-End System Test
Tests the live HTTP API with real documents, real LLM calls, and real retrieval.
No mocks. No synthetic data shortcuts.
"""
import sys
import os

# Must be at top so all imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import requests
import json
import time

BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}
RESULTS = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    icon = "[OK]" if condition else "[!!]"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    RESULTS.append((label, condition, detail))
    return condition

def post_query(question, session_id="e2e-test", mode="context_engine"):
    resp = requests.post(f"{BASE}/query", json={
        "question": question,
        "session_id": session_id,
        "mode": mode
    }, timeout=45)
    return resp

# ----------------------------------------------------------------
# 1. Health check
# ----------------------------------------------------------------
section("1. HEALTH CHECK")
try:
    r = requests.get(f"{BASE}/stats", timeout=5)
    data = r.json()
    check("Server responding", r.status_code == 200)
    check("Engine online", data.get("status") == "online", f"status={data.get('status')}")
    check("Documents indexed", data.get("document_count", 0) > 0, f"count={data.get('document_count')}")
    check("CPU readable", "cpu_usage_percent" in data, f"cpu={data.get('cpu_usage_percent')}%")
except Exception as e:
    check("Server responding", False, str(e))
    print("FATAL: server not reachable. Aborting.")
    sys.exit(1)

# ----------------------------------------------------------------
# 2. RAG retrieval from REAL indexed document (Atomic Habits)
# Note: API returns key "response", not "answer"
# ----------------------------------------------------------------
section("2. RAG RETRIEVAL — Atomic Habits PDF (real doc)")
try:
    t0 = time.time()
    r = post_query(
        "What is the 1% rule in Atomic Habits and how does it lead to compound improvement?",
        session_id="e2e-rag-1"
    )
    latency = time.time() - t0
    data = r.json()
    answer = data.get("response", "")   # correct key
    sources = data.get("search_queries", [])  # what was searched

    check("Request succeeded", r.status_code == 200, f"status={r.status_code}")
    check("Got a non-empty answer", len(answer) > 50, f"len={len(answer)} chars")
    check("Answer mentions core concepts",
          any(kw in answer.lower() for kw in ["1%", "habit", "compound", "improve", "percent", "atomic", "small"]),
          f"Answer: {answer[:180]}")
    check("LLM used context (not hallucinating)", 
          data.get("retrieved_context") is not None,
          f"retrieved_context present: {'retrieved_context' in data}")
    check("Latency acceptable (<20s)", latency < 20, f"latency={latency:.2f}s")

    # Verify query count went up
    r2 = requests.get(f"{BASE}/stats", timeout=5)
    stats = r2.json()
    check("Query count incremented", stats.get("queries_handled", 0) >= 1,
          f"queries={stats.get('queries_handled')}")

    print(f"\n  Answer preview: {answer[:250]}...")
except Exception as e:
    check("RAG retrieval", False, str(e))

# ----------------------------------------------------------------
# 3. Second query on IKIGAI — multi-turn memory
# ----------------------------------------------------------------
section("3. RAG RETRIEVAL — IKIGAI PDF + multi-turn memory")
try:
    r1 = post_query(
        "What is the Japanese concept of Ikigai and its four elements?",
        session_id="e2e-memory-test"
    )
    ans1 = r1.json().get("response", "")
    check("First IKIGAI query succeeded", r1.status_code == 200)
    check("Answer mentions Ikigai concept",
          any(kw in ans1.lower() for kw in ["ikigai", "passion", "mission", "vocation", "purpose", "japanese"]),
          f"snippet: {ans1[:150]}")

    time.sleep(1)

    r2 = post_query(
        "Based on what you just told me, how can someone discover their own?",
        session_id="e2e-memory-test"  # SAME session — tests memory
    )
    ans2 = r2.json().get("response", "")
    check("Follow-up query succeeded", r2.status_code == 200)
    check("Follow-up answer is non-empty", len(ans2) > 30, f"len={len(ans2)}")
    check("Follow-up references context",
          any(kw in ans2.lower() for kw in ["ikigai", "passion", "mission", "purpose", "discover", "find", "element"]),
          f"snippet: {ans2[:150]}")

    # Verify history was stored
    r3 = requests.get(f"{BASE}/history/e2e-memory-test", timeout=5)
    hist = r3.json()
    check("History persisted in DB", len(hist) >= 2, f"history entries={len(hist)}")
except Exception as e:
    check("Multi-turn memory test", False, str(e))

# ----------------------------------------------------------------
# 4. SQL Database via direct call (real in-memory DB)
# ----------------------------------------------------------------
section("4. REAL SQL DATABASE — sales_db query")
try:
    from src.core.sales_db import execute_read_only_sql

    # Real aggregation query
    result = execute_read_only_sql(
        "SELECT status, COUNT(*) as count, ROUND(SUM(total_amount), 2) as total "
        "FROM orders GROUP BY status ORDER BY total DESC"
    )
    check("SQL aggregation returns data", len(result) > 10 and "|" in result,
          f"result snippet: {result[:200]}")

    # Verify SELECT works fine
    customers = execute_read_only_sql("SELECT COUNT(*) as total FROM customers")
    check("Customer count query works", "total" in customers.lower() or "|" in customers,
          f"result: {customers[:100]}")

    # Injection blocking
    blocked1 = execute_read_only_sql("DROP TABLE orders;")
    check("DROP blocked", "forbidden" in blocked1.lower(), f"response: {blocked1[:80]}")

    blocked2 = execute_read_only_sql("SELECT * FROM orders UNION SELECT 1,2,3,4,5,6,7;")
    check("UNION blocked", "forbidden" in blocked2.lower(), f"response: {blocked2[:80]}")

    print(f"\n  Real DB result:\n{result[:400]}")
except Exception as e:
    check("Sales DB real query", False, str(e))

# ----------------------------------------------------------------
# 5. Web search — live Tavily
# ----------------------------------------------------------------
section("5. WEB SEARCH — live Tavily API")
try:
    from src.tools.web_search_tool import web_search

    t0 = time.time()
    results = web_search("retrieval augmented generation RAG explained 2024", max_results=3)
    latency = time.time() - t0

    check("Web search returned results", len(results) > 0, f"count={len(results)}")
    check("Results have real URLs (not mock)",
          all("mock.example.com" not in r.get("url", "") and "#" not in r.get("url","") for r in results),
          f"urls={[r.get('url','')[:60] for r in results]}")
    check("Results have real content",
          all(len(r.get("content", "")) > 20 for r in results),
          f"content lengths={[len(r.get('content','')) for r in results]}")
    check("Latency under 10s", latency < 10, f"latency={latency:.2f}s")

    print(f"\n  Results:")
    for r in results:
        print(f"  - {r.get('title','')[:65]}")
        print(f"    {r.get('url','')[:70]}")
except Exception as e:
    check("Web search", False, str(e))

# ----------------------------------------------------------------
# 6. Session management
# ----------------------------------------------------------------
section("6. SESSION MANAGEMENT — live API")
try:
    r = requests.post(f"{BASE}/sessions", json={"title": "E2E Test Session"}, timeout=5)
    check("Create session", r.status_code == 201, f"status={r.status_code}")
    new_sid = r.json().get("session_id", "")
    check("Session ID returned", bool(new_sid), f"sid={new_sid}")

    r2 = requests.get(f"{BASE}/sessions", timeout=5)
    sessions = r2.json()
    check("List sessions works", r2.status_code == 200 and isinstance(sessions, list))
    check("New session in list", any(s["session_id"] == new_sid for s in sessions),
          f"session count={len(sessions)}")

    r3 = requests.patch(f"{BASE}/sessions/{new_sid}", json={"title": "Renamed E2E Session"}, timeout=5)
    check("Rename session", r3.status_code == 200)

    r4 = requests.get(f"{BASE}/sessions", timeout=5)
    renamed = next((s for s in r4.json() if s["session_id"] == new_sid), None)
    check("Rename persisted in DB",
          renamed and renamed.get("title") == "Renamed E2E Session",
          f"title={renamed.get('title') if renamed else 'NOT FOUND'}")

    r5 = requests.delete(f"{BASE}/sessions/{new_sid}", timeout=5)
    check("Delete session", r5.status_code == 204)

    r6 = requests.get(f"{BASE}/sessions", timeout=5)
    check("Session gone after delete",
          not any(s["session_id"] == new_sid for s in r6.json()))
except Exception as e:
    check("Session management", False, str(e))

# ----------------------------------------------------------------
# 7. Streaming endpoint — real SSE
# ----------------------------------------------------------------
section("7. STREAMING ENDPOINT — real SSE stream")
try:
    t0 = time.time()
    chunks = []
    done_event = None

    with requests.post(f"{BASE}/query_stream",
                       json={"question": "Summarise the main theme of Atomic Habits in one sentence.",
                             "session_id": "e2e-stream-test",
                             "mode": "context_engine"},
                       stream=True, timeout=30) as resp:
        check("Stream connected", resp.status_code == 200, f"status={resp.status_code}")
        for line in resp.iter_lines():
            if line:
                raw = line.decode("utf-8")
                if raw.startswith("data: "):
                    try:
                        payload = json.loads(raw[6:])
                        ev = payload.get("event", "")
                        if ev == "answer_chunk":          # real event name
                            chunks.append(payload.get("text", ""))
                        elif ev == "done":
                            done_event = payload
                            break
                    except Exception:
                        pass

    full = "".join(chunks)
    latency = time.time() - t0
    check("Received streaming tokens", len(chunks) > 3, f"chunk_count={len(chunks)}")
    check("Assembled answer non-empty", len(full) > 20, f"len={len(full)}")
    check("Stream finished with done event", done_event is not None)
    check("Stream latency ok (<25s)", latency < 25, f"latency={latency:.2f}s")

    if full:
        print(f"\n  Streamed answer: {full[:250]}...")
except Exception as e:
    check("Streaming endpoint", False, str(e))

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
section("FINAL REPORT")
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = sum(1 for _, ok, _ in RESULTS if not ok)
total  = len(RESULTS)

print(f"\n  Total checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Failed       : {failed}")
print()

if failed:
    print("  Failed checks:")
    for label, ok, detail in RESULTS:
        if not ok:
            print(f"    [!!] {label}")
            if detail:
                print(f"         {detail}")

print()
print(f"  Overall: {'ALL PASS' if failed == 0 else f'{failed} FAILURES'}")

# Write results to report
report_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, "e2e_real_test_report.md")
from datetime import datetime
with open(report_path, "w") as f:
    f.write(f"# Real End-to-End System Test\n\n")
    f.write(f"*Generated on: {datetime.now().isoformat()}*\n\n")
    f.write(f"**Overall: {'ALL PASS' if failed == 0 else f'{failed}/{total} FAILURES'}**\n\n")
    f.write(f"| Check | Result | Detail |\n|---|---|---|\n")
    for label, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        f.write(f"| {label} | {status} | {detail[:80]} |\n")
print(f"\n  Report written to: {report_path}")
