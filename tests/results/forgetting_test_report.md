# Stress Test Report: Context Longevity & Recall Limits

This report analyzes the recall decay and forgetting trigger points under both **Advanced Context Engine** and **Simple RAG** processing modes.

---

## 1. Dialogue Memory Stress Test

*   **Objective:** Inject a key fact into conversation history, send 10 consecutive filler turns of network concepts, and observe if the model retains the fact.
*   **Token Limit:** Context Engine allocates a max budget of **300 tokens** to the active conversation history.

### Results
*   **Simple RAG Mode:** Recalled on Turn 1? **False**
    *   *Observation:* Because Simple RAG does not load any conversation history context into the prompt, it has zero recall on subsequent turns.
*   **Advanced Context Engine Mode:** Recalled on Turn 1-10? **Never (Still recalls after 10 turns)**
    *   *Observation:* The Context Engine retains the fact using its memory layer. As dialogue turns overflow the 300 token budget, older turns are decayed/dropped, leading to forgetting.

---

## 2. Knowledge Retrieval Stress Test (Needle-in-a-Haystack)

*   **Objective:** Inject 1 secret code chunk (needle) and 10 large routing protocol chunks (haystack) into Weaviate, query the system, and verify if it extracts the code.

### Results
*   **Simple RAG Mode:** Recalled? **True**
*   **Advanced Context Engine Mode:** Recalled? **True**
    *   *Reranker Score:* **10.2319**
    *   *Observation:* The Advanced mode utilizes Cross-Encoder rerankers to rank the needle higher than the noise documents, and then applies sentence compression to isolate only the target code.
