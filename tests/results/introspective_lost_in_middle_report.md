# Lost-in-the-Middle Positional Bias Report

> **In Plain English:** Research has shown that LLMs tend to 'forget' information placed in the middle of long documents. We tested whether our compressor has the same weakness by hiding a fact at the beginning, middle, and end of a document and checking if it's found in all three cases.

---

## 1. Why This Test Was Conducted

The 'Lost in the Middle' problem is a well-documented phenomenon in AI research. When models process long documents, they perform best at recalling information from the **beginning** and **end**, but struggle with information in the **middle**. If our compressor inherits this bias, the agent could miss critical facts simply because they appeared in the wrong position.

---

## 2. Results

| Position of Target Fact | Found by Compressor? | Compressed Tokens |
|------------------------|---------------------|-------------------|
| Beginning | ✅ Yes | 38 |
| Middle | ✅ Yes | 38 |
| End | ✅ Yes | 38 |

### Overall: ✅ No positional bias detected

---

## 3. What It Achieved

The compressor successfully found the target fact regardless of where it was positioned in the document. This means our system does **not** suffer from the 'Lost in the Middle' problem, because the compressor scores each segment independently based on query relevance — position doesn't matter.

---

## 4. What's To Be Done Next

| Finding | Action Required |
|---------|----------------|
| No positional bias | ✅ The compressor's segment-level scoring is position-independent. Safe for agent use. |
| Test covers 3 positions | Consider expanding to test 10+ positions across very long documents (RULER benchmark style). |

---

## 5. Key Takeaway

> The compressor scores each paragraph independently based on query relevance, making it immune to positional bias. Unlike raw LLM processing, facts placed in the middle of a document are just as likely to be retained as facts at the beginning or end.