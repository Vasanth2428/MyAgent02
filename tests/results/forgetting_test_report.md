# Forgetting & Memory Recall Test Report

> **In Plain English:** We tested whether the AI "forgets" things you told it earlier in the conversation. Can it remember a fact after 10 follow-up messages? Does it lose track of important details over time?

---

## 1. Why This Test Was Conducted

A useful AI assistant must remember what you said earlier in the conversation. If you tell the AI "my project is called Apollo" on Turn 1, and then ask "what's my project name?" on Turn 11, the AI should still know the answer. This test checks exactly that — and also tests whether the system can find a hidden "needle" buried inside a mountain of uploaded documents.

**Two sub-tests were run:**
1. **Dialogue Memory Stress Test:** Tell the AI a secret fact, send 10 unrelated messages, then ask it to recall the fact.
2. **Needle-in-a-Haystack Test:** Upload 11 documents (10 irrelevant, 1 containing a secret code), then ask the AI to find the code.

---

## 2. What It Achieved (Results)

### Test A: Dialogue Memory

| Mode | Can it recall the fact after 10 turns? |
|------|---------------------------------------|
| **Simple RAG** | ❌ No — Simple RAG has **no memory system at all**. It only sees the current question and the uploaded documents. It literally cannot remember anything from previous turns. |
| **Advanced Context Engine** | ✅ Yes — It **never forgot**, even after 10 full turns of unrelated conversation. The memory layer successfully retained the important fact throughout. |

> **Why does Simple RAG fail?** Simple RAG mode is "stateless" — it treats every question as if it's the first question ever asked. It doesn't load any conversation history into the prompt. This is fine for one-shot Q&A but terrible for multi-turn dialogue.

> **Why does Context Engine succeed?** The Context Engine has a dedicated `ConversationMemory` module that tracks every turn, assigns importance scores, and injects the most relevant history into each new prompt. Even as older turns get decayed to save space, the core facts survive because they carry higher importance weights.

### Test B: Needle-in-a-Haystack (Knowledge Retrieval)

| Mode | Found the hidden code? | Reranker Score |
|------|----------------------|----------------|
| **Simple RAG** | ✅ Yes | N/A (no reranker) |
| **Advanced Context Engine** | ✅ Yes | **10.2319** (extremely high confidence) |

> Both modes successfully extracted the hidden code from a mountain of irrelevant documents. The Advanced mode did it with much higher confidence thanks to the Cross-Encoder reranker, which scored the needle at 10.23 (well above the noise).

---

## 3. What's To Be Done Next

| Finding | Action Required |
|---------|----------------|
| Simple RAG has zero memory | This is by design — Simple RAG is meant for fast, stateless lookups. No fix needed unless we want to add a "lite memory" option. |
| Context Engine retains facts across 10 turns | ✅ Already working. For the agent upgrade, we need to split this memory into two tracks: one for user dialogue history, and one for the agent's internal "scratchpad" thoughts. |
| Both modes found the needle | ✅ The retrieval + reranking pipeline is already strong enough for agent tool usage. |

---

## 4. Key Takeaway

> The **Advanced Context Engine mode remembers everything** you tell it across multiple conversation turns and can find hidden information in large document collections with high confidence. The **Simple RAG mode is intentionally stateless** — it's a fast, no-frills search tool with no conversation tracking.
