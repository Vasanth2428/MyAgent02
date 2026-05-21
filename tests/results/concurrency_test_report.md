# Concurrency & Load Stress Test Report

> **In Plain English:** We tried to overwhelm the system by having 10 fake users all ask questions at the exact same time to see if the system crashes, slows down, or drops anyone's request.

---

## 1. Why This Test Was Conducted

When a system moves from a single-user demo to a real-world product (or an autonomous agent that fires off many parallel internal queries), it needs to handle **multiple users talking to it simultaneously** without breaking. This test simulates that worst-case scenario.

**What we did:**
- Spun up **10 simulated users**, each asking **3 questions** back-to-back.
- All 10 users hit the server at the **exact same moment** (total: 30 simultaneous requests).
- Each request triggers the full 6-phase pipeline: Query Expansion → HyDE → Hybrid Retrieval → Reranking → Compression → LLM Generation.

---

## 2. What It Achieved (Raw Results)

| Metric                  | Value          |
|------------------------|----------------|
| **Total Users**         | 10             |
| **Queries Per User**    | 3              |
| **Total Requests**      | 30             |
| **Successful Requests** | 17 (56.7%)     |
| **Failed Requests**     | 13 (43.3%)     |
| **Total Elapsed Time**  | 92.10 seconds  |
| **Average Latency**     | 20.04 seconds  |
| **Min Latency**         | 10.85 seconds  |
| **Max Latency**         | 27.95 seconds  |

### Status: ⚠️ WARNING — Degraded Under Heavy Load

---

## 3. What Do These Numbers Mean?

### ✅ What Worked
- **17 out of 30 requests succeeded** even under extreme concurrent load. The system did not crash, hang, or corrupt any data.
- **SQLite (the local database) held up perfectly.** All 10 users were reading and writing session history simultaneously, and there were zero database deadlocks or corruption errors. This is a strong sign that the persistence layer is production-ready.

### ⚠️ What Struggled
- **13 requests timed out** (took longer than 30 seconds and were killed). This happened because of two bottlenecks:

  1. **Groq API Rate Limits:** Groq (the cloud LLM provider) enforces strict limits — roughly 30 requests per minute and 6,000 tokens per minute on the free tier. When 10 users hit it simultaneously, the API starts rejecting or queuing requests, causing massive delays.

  2. **CPU-Bound Reranking:** The Neural Reranker (Cross-Encoder) runs locally on your CPU. When multiple requests arrive at once, they queue up because Python can only run one reranking job at a time (due to the GIL — Global Interpreter Lock). This creates a traffic jam.

### 📊 Latency Breakdown
- The **fastest** request took ~11 seconds (acceptable for a full 6-phase RAG pipeline).
- The **slowest** request took ~28 seconds (dangerously close to the 30-second timeout).
- The **average** was ~20 seconds, meaning most users would be waiting a long time under heavy load.

---

## 4. What's To Be Done Next

Based on these findings, the following upgrades are recommended before converting to a full agent system:

| Priority | Action | Why |
|----------|--------|-----|
| **High** | Upgrade to a paid Groq tier or switch to a self-hosted LLM (e.g., Ollama) | Eliminates the 30 RPM / 6000 TPM rate limit, which caused 43% of requests to fail. |
| **High** | Add request queuing with retry logic | Instead of timing out, failed requests should be automatically retried after a short backoff period. |
| **Medium** | Move reranking to a background worker or GPU | The CPU-bound Cross-Encoder blocks the main thread. Offloading it to a separate process or GPU would allow parallel reranking. |
| **Low** | Migrate from SQLite to PostgreSQL | SQLite handled concurrency well in this test, but for 50+ simultaneous agent sessions, a proper database would be safer. |

---

## 5. Key Takeaway

> The system is **stable and data-safe** under concurrent load — no crashes, no data corruption, no deadlocks. However, it is **speed-limited** by the free-tier Groq API and single-threaded reranking. Fixing these two bottlenecks would make the system ready for real-time multi-agent workloads.
